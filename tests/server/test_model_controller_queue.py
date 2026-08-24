"""Per-model reconcile debouncing.

For a role-bearing model this is a correctness requirement, not a throughput
one. The trigger chain is "instance DELETED -> Model UPDATED -> reconcile", so
retiring a 4P4D generation is nine deletions and nine reconciles — and every
middle one observes a group short of members. Reconciling on each recreates
what the deletion is halfway through removing.

The coalescing policy carries the load here, and the part worth pinning is that
it is NOT plain latest-wins: `notify_model_route_target` decides whether to
publish by asking which fields moved, so keeping only the newest event's
`changed_fields` silently drops the notification an earlier event carried.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.config.config import Config
from gpustack.schemas.config import GatewayModeEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import ModelController
from gpustack.server.workqueue import WorkEvent, WorkEventType


def _controller() -> ModelController:
    config = Config(
        token="t",
        jwt_secret_key="s",
        gateway_mode=GatewayModeEnum.disabled,
        data_dir="/tmp/gpustack-test",
    )
    return ModelController(config)


def _work(model_id, type_, changed_fields):
    return WorkEvent(
        keys=(model_id,),
        type=type_,
        object=Event(
            type=EventType.UPDATED,
            data=SimpleNamespace(id=model_id, name="m"),
            changed_fields=changed_fields,
        ),
    )


# --- the coalescing policy ------------------------------------------------- #


def test_the_newest_row_wins():
    merge = ModelController._merge_events
    older = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})
    newer = _work(1, WorkEventType.MODIFIED, {"replicas": (2, 3)})

    merged = merge(older, newer)

    assert merged.object is newer.object


def test_changed_fields_are_unioned_not_replaced():
    """The reason this is not latest-wins. A `state` transition followed by an
    unrelated edit would otherwise leave the gateway holding a target it was
    never told to update."""
    merge = ModelController._merge_events
    older = _work(1, WorkEventType.MODIFIED, {"state": ("pending", "running")})
    newer = _work(1, WorkEventType.MODIFIED, {"description": ("a", "b")})

    merged = merge(older, newer)

    assert set(merged.object.changed_fields) == {"state", "description"}


def test_a_field_that_moved_twice_keeps_the_whole_span():
    """One reconcile of everything since the last one means the pair has to
    describe the span, not its final step."""
    merge = ModelController._merge_events
    older = _work(1, WorkEventType.MODIFIED, {"ready_replicas": (0, 1)})
    newer = _work(1, WorkEventType.MODIFIED, {"ready_replicas": (1, 3)})

    merged = merge(older, newer)

    assert merged.object.changed_fields["ready_replicas"] == (0, 3)


def test_a_pending_deleted_is_sticky():
    """A row that is gone must not be reconciled as if merely updated."""
    merge = ModelController._merge_events
    deleted = _work(1, WorkEventType.DELETED, {})
    updated = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})

    assert merge(deleted, updated) is deleted


def test_a_deleted_upgrades_whatever_is_pending():
    merge = ModelController._merge_events
    updated = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})
    deleted = _work(1, WorkEventType.DELETED, {})

    assert merge(updated, deleted) is deleted


# --- the burst it exists to absorb ----------------------------------------- #


@pytest.mark.asyncio
async def test_a_burst_of_deletions_reconciles_once():
    """Nine members going away is nine bus events. Nine reconciles would each
    see an incomplete group; one reconcile sees the settled state."""
    controller = _controller()
    for _ in range(9):
        controller._queue.add(
            _work(1, WorkEventType.MODIFIED, {"ready_replicas": (1, 0)})
        )

    assert len(controller._queue) == 1


@pytest.mark.asyncio
async def test_different_models_are_not_collapsed_together():
    """Keyed per model, so one busy model cannot delay another."""
    controller = _controller()
    controller._queue.add(_work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)}))
    controller._queue.add(_work(2, WorkEventType.MODIFIED, {"replicas": (1, 2)}))

    assert len(controller._queue) == 2


@pytest.mark.asyncio
async def test_one_models_events_reconcile_serially():
    """`sync_replicas` reads the instance set and then writes to it, so two
    passes overlapping on one model would both see the pre-write count and
    both create."""
    controller = _controller()
    running = []
    peak = 0

    async def _slow_reconcile(event):
        nonlocal peak
        running.append(1)
        peak = max(peak, len(running))
        await asyncio.sleep(0.01)
        running.pop()

    with patch.object(controller, "_reconcile", AsyncMock(side_effect=_slow_reconcile)):
        dispatch = asyncio.create_task(controller._dispatch())
        for i in range(4):
            controller._queue.add(
                _work(1, WorkEventType.MODIFIED, {"replicas": (i, i + 1)})
            )
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.1)
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)

    assert peak == 1, "two reconciles of one model overlapped"


@pytest.mark.asyncio
async def test_a_failed_reconcile_is_retried_with_backoff():
    controller = _controller()

    with patch.object(
        controller, "_reconcile", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        await controller._process(
            _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})
        )

    assert controller._queue.failures((1,)) == 1


@pytest.mark.asyncio
async def test_a_successful_reconcile_clears_the_backoff():
    controller = _controller()
    event = _work(1, WorkEventType.MODIFIED, {"replicas": (1, 2)})

    with patch.object(
        controller, "_reconcile", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        await controller._process(event)
    with patch.object(controller, "_reconcile", AsyncMock()):
        await controller._process(event)

    assert controller._queue.failures((1,)) == 0
