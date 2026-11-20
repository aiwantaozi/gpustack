"""When a running member is behind the config it is shown with.

The interesting cases are all about one field. `backend_version` is a spec
field the *server* also writes to: a worker that detects the engine build
records it back onto the Model, so that a later replica resolves the same build
instead of a newer one. That write is deliberate, and it is not a config
change — but it moves the digest, which is what `stale` compares.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.models import (
    BackendEnum,
    DisaggregationSpec,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceSubordinateWorker,
    ModelInstanceStateEnum,
    PDModeEnum,
    RoleSpec,
    SourceEnum,
)
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.workers import Worker
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import (
    ClusterController,
    model_members_stale,
    model_spec_digest,
)
from tests.utils.mock import mock_async_session


def _model(**kwargs) -> Model:
    return Model(
        id=1,
        name="m",
        replicas=1,
        ready_replicas=0,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        backend=BackendEnum.SGLANG,
        roles=kwargs.pop(
            "roles",
            [
                RoleSpec(name="prefill", replicas=1),
                RoleSpec(name="decode", replicas=1),
            ],
        ),
        **kwargs,
    )


def _instance(digest, version=None) -> ModelInstance:
    return ModelInstance(
        id=1,
        name="i",
        model_id=1,
        model_name="m",
        spec_digest=digest,
        backend_version=version,
    )


_no_types = patch(
    "gpustack.server.controllers.GPUInstanceType.all_by_fields",
    AsyncMock(return_value=[]),
)


def _pass():
    """A stand-in for one reconcile pass.

    `session.info` is a real dict because that is where `model_spec_digest`
    keeps its per-pass memo; a bare `MagicMock` would answer `.get` with a
    mock and hand back a digest nobody computed. A fresh one per call is a
    fresh pass.
    """
    session = MagicMock()
    session.info = {}
    return session


async def _digest(model, **kwargs):
    with _no_types:
        return await model_spec_digest(_pass(), model, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'snapshot, cluster_value, worker_value, manual_value, expected',
    [
        ({'1': 'eth0', '2': 'eth0'}, 'eth0', None, None, False),
        ({'1': 'eth0', '2': 'eth0'}, 'eth1', None, None, True),
        ({'1': 'eth0', '2': 'eth0'}, 'eth0', 'eth1', None, True),
        ({'1': 'eth0', '2': 'eth0'}, 'eth1', None, 'eth2', False),
        (None, None, None, None, False),
        (None, 'eth1', None, None, True),
    ],
)
async def test_nixl_effective_interfaces_detect_deployment_changes(
    monkeypatch, snapshot, cluster_value, worker_value, manual_value, expected
):
    model = _model(
        cluster_id=1,
        disaggregation=DisaggregationSpec(mode=PDModeEnum.SGLANG_NIXL),
        env={'UCX_NET_DEVICES': manual_value} if manual_value else None,
    )
    instance = ModelInstance(
        id=1,
        name='decode',
        model_id=1,
        model_name='m',
        spec_digest=await _digest(model),
        state=ModelInstanceStateEnum.RUNNING,
        group_id='g',
        role='decode',
        worker_id=1,
        kv_transfer_ifnames=snapshot,
        distributed_servers=DistributedServers(
            subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=2)]
        ),
    )
    monkeypatch.setattr(
        Cluster,
        'one_by_id',
        AsyncMock(
            return_value=SimpleNamespace(
                worker_config=SimpleNamespace(kv_transfer_ifname=cluster_value)
            )
        ),
    )
    monkeypatch.setattr(
        Worker,
        'all_by_ids',
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=1, status=SimpleNamespace(kv_transfer_ifname='eth0')
                ),
                SimpleNamespace(id=2, status=None, kv_transfer_ifname=worker_value),
            ]
        ),
    )
    with _no_types:
        assert await model_members_stale(_pass(), model, [instance]) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize('same_device', [False, True])
async def test_cluster_default_change_notifies_only_nixl_models(same_device):
    controller = object.__new__(ClusterController)
    controller._disable_gateway = True
    controller._sync_cluster_state = AsyncMock()
    model = _model(
        cluster_id=1, disaggregation=DisaggregationSpec(mode=PDModeEnum.SGLANG_NIXL)
    )
    unrelated = _model(cluster_id=1)
    unrelated.id = 2
    default = 'eth0' if same_device else 'eth1'
    event = Event(
        type=EventType.UPDATED,
        data=SimpleNamespace(id=1),
        changed_fields={
            'worker_config': (
                (SimpleNamespace(kv_transfer_ifname='eth0'),),
                (SimpleNamespace(kv_transfer_ifname=default),),
            )
        },
    )
    with (
        patch(
            'gpustack.server.controllers.async_session',
            return_value=mock_async_session(),
        ),
        patch.object(
            Model, 'all_by_fields', AsyncMock(return_value=[model, unrelated])
        ),
        patch('gpustack.server.controllers.event_bus.publish', AsyncMock()) as publish,
    ):
        await controller._reconcile(event)

    if same_device:
        publish.assert_not_awaited()
    else:
        publish.assert_awaited_once()
        assert publish.await_args.args[1].data.id == model.id


async def _stale(model, instances):
    with _no_types:
        return await model_members_stale(_pass(), model, instances)


@pytest.mark.asyncio
async def test_nothing_can_be_said_without_a_stamp():
    """A member created before `spec_digest` existed carries None, and reading
    that as "differs" would mark every pre-upgrade model stale on the first
    pass after an upgrade."""
    assert await _stale(_model(), [_instance(None)]) is None
    assert await _stale(_model(), []) is None


@pytest.mark.asyncio
async def test_a_member_stamped_with_the_current_spec_is_not_stale():
    model = _model()
    assert await _stale(model, [_instance(await _digest(model))]) is False


@pytest.mark.asyncio
async def test_a_real_edit_makes_a_member_stale():
    model = _model()
    stamped = _instance(await _digest(model))
    model.backend_parameters = ["--mem-fraction-static=0.9"]
    assert await _stale(model, [stamped]) is True


@pytest.mark.asyncio
async def test_recording_the_detected_version_does_not_make_a_member_stale():
    """The worker writes the engine's detected version back to an unpinned
    model, which moves the digest. Counting that as an edit would mark every
    member stale seconds after it started, with nobody having edited anything,
    and the banner would ask for a restart to adopt a value read off that very
    group."""
    model = _model()
    unpinned = await _digest(model)
    members = [
        _instance(unpinned, version="0.5.15.post1"),
        _instance(unpinned, version="0.5.15.post1"),
    ]

    model.backend_version = "0.5.15.post1"
    assert await _digest(model) != unpinned, "the digest must still move"
    assert await _stale(model, members) is False


@pytest.mark.asyncio
async def test_pinning_a_version_the_group_is_not_running_still_makes_it_stale():
    """The exemption is not "ignore backend_version". A user pinning a build
    the members are not running needs the restart prompt, or the edit silently
    never applies."""
    model = _model()
    members = [_instance(await _digest(model), version="0.5.15.post1")]
    model.backend_version = "0.5.14"
    assert await _stale(model, members) is True


@pytest.mark.asyncio
async def test_a_member_that_cannot_say_what_it_runs_stays_stale():
    """Conservative direction: without the member's own version there is no
    evidence the recorded one is what it is running."""
    model = _model()
    members = [_instance(await _digest(model), version=None)]
    model.backend_version = "0.5.15.post1"
    assert await _stale(model, members) is True


@pytest.mark.asyncio
async def test_one_member_behind_makes_the_group_stale():
    """A rolling failure leaves a group mid-generation, and the marker is about
    the group."""
    model = _model()
    unpinned = await _digest(model)
    model.backend_version = "0.5.15.post1"
    current = await _digest(model)
    members = [
        _instance(current, version="0.5.15.post1"),
        _instance(unpinned, version="0.5.14"),
    ]
    assert await _stale(model, members) is True


@pytest.mark.asyncio
async def test_the_substitute_is_distinguished_from_a_real_none():
    """`backend_version=None` must mean "unset", not "no override" — the whole
    exemption rests on being able to ask for the unset spec explicitly."""
    model = _model(backend_version="0.5.15.post1")
    assert await _digest(model) != await _digest(model, backend_version=None)
    assert await _digest(model) == await _digest(model, backend_version="0.5.15.post1")


@pytest.mark.asyncio
async def test_a_provider_shaped_row_is_untouched():
    """Sanity: nothing here depends on roles, so a role-less model behaves the
    same."""
    model = _model(roles=None)
    stamped = _instance(await _digest(model))
    assert await _stale(model, [stamped]) is False
    model.backend_version = "0.5.15.post1"
    assert await _stale(model, [stamped]) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ANALYZING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.DOWNLOADING,
        ModelInstanceStateEnum.STARTING,
        ModelInstanceStateEnum.ERROR,
    ],
)
@pytest.mark.parametrize("first_version_recorded", [False, True])
async def test_group_can_finish_starting_while_detected_version_is_recorded(
    state, first_version_recorded
):
    model = _model()
    digest = await _digest(model)
    first = _instance(digest, "0.5.15" if first_version_recorded else None)
    first.group_id = "1-group"
    first.state = (
        ModelInstanceStateEnum.RUNNING
        if first_version_recorded
        else ModelInstanceStateEnum.STARTING
    )
    waiting = _instance(digest)
    waiting.id = 2
    waiting.group_id = first.group_id
    waiting.state = state
    model.backend_version = "0.5.15"

    assert await _stale(model, [first, waiting]) is False
    assert first.spec_digest == waiting.spec_digest == digest


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", ["parameters", "role-parameters", "version"])
@pytest.mark.parametrize(
    "waiting_state",
    [
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.STARTING,
        ModelInstanceStateEnum.ERROR,
    ],
)
async def test_version_writeback_during_startup_does_not_hide_a_real_edit(
    edit, waiting_state
):
    model = _model()
    digest = await _digest(model)
    first = _instance(digest, "0.5.15")
    first.group_id = "1-group"
    first.state = ModelInstanceStateEnum.RUNNING
    waiting = _instance(digest)
    waiting.group_id = first.group_id
    waiting.state = waiting_state
    model.backend_version = "0.5.15"
    if edit == "parameters":
        model.backend_parameters = ["--mem-fraction-static=0.9"]
    elif edit == "role-parameters":
        model.roles[0].backend_parameters = ["--mem-fraction-static=0.9"]
    else:
        model.backend_version = "0.5.14"

    assert await _stale(model, [first, waiting]) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        ModelInstanceStateEnum.RUNNING,
        ModelInstanceStateEnum.UNREACHABLE,
    ],
)
async def test_unknown_version_of_running_or_unreachable_member_is_still_stale(state):
    model = _model()
    member = _instance(await _digest(model))
    member.group_id = "1-group"
    member.state = state
    model.backend_version = "0.5.15"

    assert await _stale(model, [member]) is True
