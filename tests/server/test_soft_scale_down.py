"""Scaling a prefill down without dropping the KV it still holds.

The engine has no "stop accepting work and exit once your blocks have been
fetched" — waiting for that is upstream WIP. So the wait happens in the
orchestration layer: the member leaves the router's registry at once, keeps
running for a window, and is deleted after it. Removing an address is measured
at 18ms with requests in flight, so the cost of changing a ratio is entirely
that window, not an interruption.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum, RoleNameEnum
from gpustack.server import pd_membership
from gpustack.server.controllers import (
    _reap_drained,
    _scale_down_role,
    cancel_drain,
)

WINDOW = 60


def _instance(name, role="prefill", draining_since=None, worker_id=1):
    instance = SimpleNamespace(
        id=name,
        name=name,
        role=role,
        worker_id=worker_id,
        worker_ip="10.0.0.1",
        port=40000,
        state=ModelInstanceStateEnum.RUNNING,
        draining_since=draining_since,
    )
    instance.update = AsyncMock()
    return instance


def _model(roles=("prefill", "decode")):
    return SimpleNamespace(
        id=1,
        name="m",
        roles=[SimpleNamespace(name=name) for name in roles],
    )


def _role(name="prefill", replicas=1):
    return SimpleNamespace(name=name, replicas=replicas)


class TestRegistryRemoval:
    """Step one, and it is the only step that changes what users see."""

    def test_a_draining_member_leaves_the_router(self):
        keep = _instance("p1")
        going = _instance("p2", draining_since=datetime.now(timezone.utc))
        going.port = 40001

        members = pd_membership.desired_members(_model(), [keep, going])

        assert members == {pd_membership.member_url(keep): "prefill"}

    def test_it_is_not_a_state_change(self):
        """🔴 The container must keep serving the decodes already pulling from
        it. Only *new* work has to stop arriving, and the address is exactly
        that distinction."""
        going = _instance("p2", draining_since=datetime.now(timezone.utc))
        assert going.state == ModelInstanceStateEnum.RUNNING


class TestPicking:
    async def _scale(self, model, role, have, window=WINDOW, candidates=None):
        picked = candidates if candidates is not None else have
        with (
            patch("gpustack.envs.SCHEDULER_DRAIN_WINDOW_SECONDS", window),
            patch(
                "gpustack.server.controllers.find_scale_down_candidates",
                AsyncMock(
                    return_value=[
                        SimpleNamespace(model_instance=i, score=0) for i in picked
                    ]
                ),
            ),
            patch(
                "gpustack.server.controllers._release_and_delete", AsyncMock()
            ) as release,
        ):
            await _scale_down_role(None, model, role, have)
            return release

    @pytest.mark.asyncio
    async def test_the_surplus_is_marked_not_deleted(self):
        have = [_instance("p1"), _instance("p2")]
        release = await self._scale(_model(), _role(replicas=1), have)

        release.assert_not_called()
        assert sum(i.draining_since is not None for i in have) == 1

    @pytest.mark.asyncio
    async def test_a_second_pass_does_not_pick_another(self):
        """🔴 The debounce, and without it a burst of reconciles scales the
        role to zero one window at a time: the draining member still counts as
        surplus, so every pass picks one more."""
        draining = _instance("p1", draining_since=datetime.now(timezone.utc))
        have = [draining, _instance("p2")]

        await self._scale(_model(), _role(replicas=1), have)

        assert have[1].draining_since is None

    @pytest.mark.asyncio
    async def test_excess_is_measured_against_the_role(self):
        """Not against the model: `len(have) - model.replicas` scoped to one
        role of a 4P4D deletes eight instances in a pass."""
        have = [_instance(f"p{i}") for i in range(4)]
        await self._scale(_model(), _role(replicas=2), have)

        assert sum(i.draining_since is not None for i in have) == 2

    @pytest.mark.asyncio
    async def test_a_role_less_model_still_deletes_immediately(self):
        """No router registry to leave, so a window would only mean the member
        keeps taking new requests and then vanishes mid-request."""
        have = [_instance("i1"), _instance("i2")]
        release = await self._scale(
            SimpleNamespace(id=1, name="m", roles=None), _role(replicas=1), have
        )

        release.assert_called_once()
        assert all(i.draining_since is None for i in have)

    @pytest.mark.asyncio
    async def test_a_zero_window_deletes_immediately(self):
        have = [_instance("p1"), _instance("p2")]
        release = await self._scale(_model(), _role(replicas=1), have, window=0)

        release.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_scoring_failure_deletes_nothing(self):
        """`find_scale_down_candidates` returns [] on its internal exception,
        so empty is ambiguous and not deleting is the fail-safe reading."""
        have = [_instance("p1"), _instance("p2")]
        release = await self._scale(_model(), _role(replicas=1), have, candidates=[])

        release.assert_not_called()
        assert all(i.draining_since is None for i in have)


class TestReaping:
    async def _reap(self, instances, window=WINDOW):
        with (
            patch("gpustack.envs.SCHEDULER_DRAIN_WINDOW_SECONDS", window),
            patch(
                "gpustack.server.controllers._release_and_delete", AsyncMock()
            ) as release,
        ):
            survivors = await _reap_drained(None, instances)
            return survivors, release

    @pytest.mark.asyncio
    async def test_inside_the_window_it_keeps_running(self):
        recent = _instance(
            "p1",
            draining_since=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        survivors, release = await self._reap([recent])

        release.assert_not_called()
        assert survivors == [recent]

    @pytest.mark.asyncio
    async def test_past_the_window_it_is_deleted(self):
        old = _instance(
            "p1",
            draining_since=datetime.now(timezone.utc) - timedelta(seconds=WINDOW + 1),
        )
        keep = _instance("p2")
        survivors, release = await self._reap([old, keep])

        release.assert_called_once()
        assert survivors == [keep]

    @pytest.mark.asyncio
    async def test_a_window_that_elapsed_while_the_server_was_down(self):
        """🔴 Why the timestamp is on the row. Held in memory, a restart
        mid-window leaves a member no router knows about and nothing will ever
        delete — serving nothing, holding its cards."""
        ancient = _instance(
            "p1", draining_since=datetime.now(timezone.utc) - timedelta(days=2)
        )
        _, release = await self._reap([ancient])

        release.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_naive_timestamp_is_read_as_utc(self):
        """SQLite hands the datetime back without a zone; it was written UTC,
        and comparing it against an aware `now` raises otherwise."""
        naive = _instance(
            "p1",
            draining_since=(
                datetime.now(timezone.utc) - timedelta(seconds=WINDOW + 1)
            ).replace(tzinfo=None),
        )
        _, release = await self._reap([naive])

        release.assert_called_once()

    @pytest.mark.asyncio
    async def test_survivors_never_include_a_deleted_member(self):
        """The per-role arithmetic that follows counts what comes back. A
        reaped member left in the list makes the role look satisfied and
        suppresses the replacement a re-scale-up is waiting for."""
        old = _instance(
            "p1",
            draining_since=datetime.now(timezone.utc) - timedelta(seconds=WINDOW + 1),
        )
        survivors, _ = await self._reap([old])

        assert survivors == []


@pytest.mark.asyncio
async def test_cancelling_a_drain_puts_it_back():
    """The rollback is one field: the next membership reconcile sees an
    ordinary RUNNING member and re-registers it. Nothing restarts, because
    nothing was stopped."""
    instance = _instance("p1", draining_since=datetime.now(timezone.utc))

    await cancel_drain(None, instance)

    assert instance.draining_since is None
    assert pd_membership.desired_members(_model(), [instance]) == {
        pd_membership.member_url(instance): RoleNameEnum.PREFILL.value
    }
