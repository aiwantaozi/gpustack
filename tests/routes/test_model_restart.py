"""The group restart endpoint.

`POST /v2/models/{id}/restart` exists because "restart" had no backend at all —
the whole API had zero restart routes, and restarting in practice meant
deleting an instance and letting replica convergence rebuild it. For a group
that sequence is a correctness bug rather than an inconvenience: between the
two deletes there is a new-generation prefill paired with an old-generation
decode, and the engines accept that pairing. A `max_model_len` mismatch
handshakes, transfers, and only surfaces on a long prompt — after prefill has
already been paid for.

So the properties pinned here are the ones that make that window impossible:
the whole generation goes down together, there is no way to ask for part of
it, and a restart already in flight is refused rather than restarted again.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.api.exceptions import ConflictException
from gpustack.routes.models import restart_model
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleSpec,
    SourceEnum,
)

TARGET = "sha1:current"


def _model(roles=None) -> Model:
    return Model(
        id=1,
        name="m",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        cluster_id=1,
        roles=roles,
    )


def _pd_roles():
    return [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1, cpu_only=True),
    ]


def _instance(id, role=None, spec_digest=TARGET) -> ModelInstance:
    return ModelInstance(
        id=id,
        name=f"m-{id}",
        model_id=1,
        model_name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        state=ModelInstanceStateEnum.RUNNING,
        role=role,
        spec_digest=spec_digest,
    )


async def _restart(model, instances, namespace=None):
    deleted = []

    async def _batch_delete(rows):
        deleted.extend(rows)
        return [r.name for r in rows]

    service = MagicMock(return_value=SimpleNamespace(batch_delete=_batch_delete))
    with (
        patch("gpustack.routes.models.Model.one_by_id", AsyncMock(return_value=model)),
        patch("gpustack.routes.models.assert_resource_visible", MagicMock()),
        patch(
            "gpustack.routes.models.model_spec_digest",
            AsyncMock(return_value=TARGET),
        ),
        patch(
            "gpustack.routes.models.ModelInstance.all_by_fields",
            AsyncMock(return_value=instances),
        ),
        # Placement drift reads the owner Principal and the Cluster; this
        # harness runs against a mock session. Covered separately in
        # tests/server/test_workload_namespace.py.
        patch(
            "gpustack.routes.models.resolve_workload_namespace",
            AsyncMock(return_value=namespace),
        ),
        patch("gpustack.routes.models.ModelInstanceService", service),
    ):
        result = await restart_model(MagicMock(), MagicMock(), 1)
    return result, deleted


@pytest.mark.asyncio
async def test_a_stale_group_goes_down_whole():
    """All of it, in one batch. Deleting members one at a time is exactly how
    the cross-generation window is produced."""
    members = [
        _instance(1, role="prefill", spec_digest="sha1:old"),
        _instance(2, role="decode", spec_digest="sha1:old"),
        _instance(3, role="router", spec_digest="sha1:old"),
    ]
    result, deleted = await _restart(_model(roles=_pd_roles()), members)

    assert result.restarted is True
    assert result.spec_digest == TARGET
    assert len(deleted) == 3
    assert {i.role for i in deleted} == {"prefill", "decode", "router"}


@pytest.mark.asyncio
async def test_rebuilding_is_left_to_replica_convergence():
    """Convergence is the one place that knows a group forms its GPU roles
    atomically and holds the router back until they run. Recreating here would
    be a second implementation of the rule that matters most."""
    members = [_instance(1, role="prefill", spec_digest="sha1:old")]
    result, _ = await _restart(_model(roles=_pd_roles()), members)

    assert result.deleted_instances == ["m-1"]
    assert "re-form" in result.message


@pytest.mark.asyncio
async def test_a_group_already_on_the_current_spec_is_still_rebuilt():
    """🔴 This asserted the opposite until the endpoint stopped short-circuiting
    on a converged group.

    Two reasons it flipped. The word on the menu is "restart", and the state an
    operator reaches for it in — a process wedged behind a socket while the
    control plane still calls it RUNNING — is exactly the one a digest
    comparison cannot see. And only a group's members carry a `spec_digest` at
    all, so the short-circuit made the same button rebuild a role-less
    deployment while doing nothing to a PD group on its current spec."""
    members = [
        _instance(1, role="prefill"),
        _instance(2, role="decode"),
        _instance(3, role="router"),
    ]
    result, deleted = await _restart(_model(roles=_pd_roles()), members)

    assert result.restarted is True
    assert sorted(result.deleted_instances) == ["m-1", "m-2", "m-3"]


@pytest.mark.asyncio
async def test_a_restart_in_flight_is_refused():
    """Mixed digests mean a previous restart is still rebuilding. Tearing down
    again would delete the replacements it just created."""
    members = [
        _instance(1, role="prefill", spec_digest=TARGET),
        _instance(2, role="decode", spec_digest="sha1:old"),
    ]
    with pytest.raises(ConflictException):
        await _restart(_model(roles=_pd_roles()), members)


@pytest.mark.asyncio
async def test_a_model_with_no_instances_reports_nothing_to_do():
    result, deleted = await _restart(_model(roles=_pd_roles()), [])

    assert result.restarted is False
    assert deleted == []
    assert result.spec_digest == TARGET


@pytest.mark.asyncio
async def test_a_role_less_model_restarts_the_same_way():
    """The mechanism is identical, so refusing here would be an arbitrary
    restriction — and the same "edited the config, nothing happened" problem
    exists for a single-instance model."""
    members = [_instance(1, spec_digest="sha1:old")]
    result, deleted = await _restart(_model(), members)

    assert result.restarted is True
    assert len(deleted) == 1


def test_the_endpoint_takes_no_role_parameter():
    """ "Restart only the decodes" is precisely the request that produces the
    cross-generation window, and the strongest rejection is to have no way to
    express it."""
    import inspect

    params = set(inspect.signature(restart_model).parameters)
    assert params == {"session", "ctx", "id"}


# --- placement is something to converge too -------------------------------- #


@pytest.mark.asyncio
async def test_members_left_in_the_old_namespace_are_restarted():
    """An upgrade leaves running members where they were, on purpose. This is
    the cure for that, and without it `placement_drifted` would be a marker
    whose only remedy is deleting the model."""
    model = _model()
    instances = [_instance(1, spec_digest=TARGET), _instance(2, spec_digest=TARGET)]
    for instance in instances:
        instance.namespace = None

    result, deleted = await _restart(model, instances, namespace="gpustack-acme")

    assert result.restarted is True
    assert len(deleted) == 2
    assert "tenant's namespace" in result.message


@pytest.mark.asyncio
async def test_members_already_where_they_belong_are_not_called_moved():
    """Drift is reported, not merely acted on. Everything is rebuilt now, so
    `restarted` no longer distinguishes anything — what still has to be right
    is the sentence, which tells the operator whether the members are coming
    back somewhere else."""
    model = _model()
    instances = [_instance(1, spec_digest=TARGET)]
    instances[0].namespace = "gpustack-acme"

    result, deleted = await _restart(model, instances, namespace="gpustack-acme")

    assert result.restarted is True
    assert len(deleted) == 1
    assert "tenant's namespace" not in result.message


@pytest.mark.asyncio
async def test_a_docker_deployment_never_looks_drifted():
    """Its instances carry no namespace and there is none to move them to, so
    it must never be described as having been moved. Asserted on the message
    rather than on `restarted`, which is now true for every teardown."""
    model = _model()
    instances = [_instance(1, spec_digest=TARGET)]

    result, deleted = await _restart(model, instances, namespace=None)

    assert result.restarted is True
    assert len(deleted) == 1
    assert "tenant's namespace" not in result.message


# --- a member that is not running anything --------------------------------- #


@pytest.mark.asyncio
async def test_a_group_with_a_failed_member_is_restarted():
    """Reporting "already run the current configuration" to someone whose
    group is half down is not merely unhelpful, it is untrue: a member in
    ERROR is not running that configuration, it is not running anything. And
    it left the one operation they reached for with nothing to do."""
    model = _model()
    instances = [
        _instance(1, spec_digest=TARGET),
        _instance(2, spec_digest=TARGET),
    ]
    instances[1].state = ModelInstanceStateEnum.ERROR

    result, deleted = await _restart(model, instances)

    assert result.restarted is True
    assert len(deleted) == 2
    # Named, because the reason it failed is usually still there and a bare
    # "restarted" invites an immediate retry of the same failure.
    assert instances[1].name in result.message
    assert "check its log" in result.message


@pytest.mark.asyncio
async def test_a_healthy_group_is_rebuilt_without_the_failure_wording():
    """A healthy group restarts too, but must not be handed the sentence
    written for a group with a dead member — that one tells the reader to go
    read a log before retrying, and there is no log to read."""
    model = _model()
    instances = [_instance(1, spec_digest=TARGET)]

    result, deleted = await _restart(model, instances)

    assert result.restarted is True
    assert len(deleted) == 1
    assert "check its log" not in result.message
