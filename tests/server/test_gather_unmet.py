"""`PreferGather` with a layer: ship it anyway, but say if you missed.

The pair used to be inexpressible — the form offered "as close as possible"
(no target, lenient) or "at least X or refuse" (target, strict), so the
combination most deployments want had nowhere to go. This is the half that
makes the lenient target mean something: without a marker afterwards, the ask
lives in the spec and the outcome lives nowhere.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.schemas.clusters import ClusterTopology
from gpustack.schemas.models import (
    DegradationReasonEnum,
    GatherSpec,
    GatherStrategyEnum,
    ModelInstanceStateEnum,
)
from gpustack.server.controllers import _gather_unmet
from tests.utils.topology_layers import layer_dict, lid

RACK = "topology.gpustack.ai/rack"
ROOM = "topology.gpustack.ai/zone"


def _worker(id_, labels):
    return SimpleNamespace(
        id=id_, name=f"w{id_}", labels=labels, status=SimpleNamespace(topology_facts={})
    )


def _instance(worker_id, state=ModelInstanceStateEnum.RUNNING, role="prefill"):
    return SimpleNamespace(worker_id=worker_id, state=state, role=role)


async def _check(gather, workers, instances, topology=None):
    # `roles` is read by `role_takes_no_accelerator`, which is how the router
    # gets excluded — a stub without it would pass by accident.
    model = SimpleNamespace(
        gather=gather,
        cluster_id=1,
        roles=[SimpleNamespace(name=name) for name in ("prefill", "decode", "router")],
    )
    cluster = SimpleNamespace(
        topology=ClusterTopology.model_validate(topology) if topology else None
    )
    with (
        patch(
            "gpustack.schemas.clusters.Cluster.one_by_id",
            return_value=cluster,
        ),
        patch(
            "gpustack.schemas.workers.Worker.all_by_field",
            return_value=workers,
        ),
    ):
        return await _gather_unmet(None, model, instances)


PREFER_RACK = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER, layer=lid("rack"))


@pytest.mark.asyncio
async def test_members_inside_the_wanted_rack_are_not_degraded():
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R1"})]
    assert await _check(PREFER_RACK, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_members_split_across_racks_are_degraded():
    """The whole point: the deployment went out, and this is the only record
    that it went out looser than asked."""
    workers = [
        _worker(1, {ROOM: "H", RACK: "R1"}),
        _worker(2, {ROOM: "H", RACK: "R2"}),
    ]
    assert await _check(PREFER_RACK, workers, [_instance(1), _instance(2)]) is True


@pytest.mark.asyncio
async def test_a_tighter_placement_than_asked_for_is_not_degraded():
    """Asked for same-room, got same-rack. Tighter is never a miss — and this
    is the direction the comparison is easiest to write backwards, since
    "looser" means *earlier* in a root-to-leaf order."""
    prefer_room = GatherSpec(
        strategy=GatherStrategyEnum.PREFER_GATHER, layer=lid("zone")
    )
    workers = [
        _worker(1, {ROOM: "H", RACK: "R1"}),
        _worker(2, {ROOM: "H", RACK: "R1"}),
    ]
    assert await _check(prefer_room, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_must_gather_never_reports_a_miss():
    """It refused at admission instead. A marker here would be a second,
    weaker answer to a question already settled."""
    must = GatherSpec(strategy=GatherStrategyEnum.MUST_GATHER, layer=lid("rack"))
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R2"})]
    assert await _check(must, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_no_layer_means_nothing_to_miss():
    """`PreferGather` on its own is "anywhere is fine", which cannot be
    disappointed."""
    bare = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER)
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R2"})]
    assert await _check(bare, workers, [_instance(1), _instance(2)]) is False


@pytest.mark.asyncio
async def test_one_placed_member_is_silence_not_a_pass():
    """There is no distance between members to be wrong about yet. Reporting
    `False` here is the same answer as "it fits", which is why the early
    return is about the question not applying rather than about it passing."""
    workers = [_worker(1, {RACK: "R1"}), _worker(2, {RACK: "R2"})]
    one = [_instance(1), _instance(2, state=ModelInstanceStateEnum.PENDING)]
    assert await _check(PREFER_RACK, workers, one) is False


@pytest.mark.asyncio
async def test_unlabelled_members_count_as_a_miss():
    """Two workers in the unclassified bucket are NOT close: the bucket means
    "we do not know where these are", and reading it as "together" would turn
    a missing label into a confident wrong answer (`common_layer`'s own
    rule). A group that cannot be shown to meet the target has not met it."""
    workers = [_worker(1, {}), _worker(2, {})]
    assert await _check(PREFER_RACK, workers, [_instance(1), _instance(2)]) is True


@pytest.mark.asyncio
async def test_a_custom_rung_is_compared_like_any_other():
    topology = {"layers": [layer_dict("Pod", ["dc/pod"], parent="zone")]}
    prefer_pod = GatherSpec(strategy=GatherStrategyEnum.PREFER_GATHER, layer=lid("Pod"))
    same = [
        _worker(1, {ROOM: "H", "dc/pod": "P1"}),
        _worker(2, {ROOM: "H", "dc/pod": "P1"}),
    ]
    assert (
        await _check(prefer_pod, same, [_instance(1), _instance(2)], topology) is False
    )

    split = [
        _worker(1, {ROOM: "H", "dc/pod": "P1"}),
        _worker(2, {ROOM: "H", "dc/pod": "P2"}),
    ]
    assert (
        await _check(prefer_pod, split, [_instance(1), _instance(2)], topology) is True
    )


@pytest.mark.asyncio
async def test_the_router_does_not_count_towards_the_target():
    """🔴 The solver excludes it for the same reason (`role_demands`): a
    router holds no weights, so it "neither competes for cards nor constrains
    which domain the group lands in". Counting it here would report a miss
    because the *proxy* landed elsewhere — a placement nothing constrained
    and which would be made again on the next reschedule."""
    workers = [
        _worker(1, {ROOM: "H", RACK: "R1"}),
        _worker(2, {ROOM: "H", RACK: "R1"}),
        _worker(3, {ROOM: "H", RACK: "R9"}),
    ]
    group = [
        _instance(1, role="prefill"),
        _instance(2, role="decode"),
        _instance(3, role="router"),
    ]
    assert await _check(PREFER_RACK, workers, group) is False


@pytest.mark.asyncio
async def test_a_router_alone_beside_one_engine_is_not_a_group():
    """Dropping the router can leave fewer than two members to compare, and
    that is silence rather than a pass."""
    workers = [_worker(1, {RACK: "R1"}), _worker(3, {RACK: "R9"})]
    assert (
        await _check(
            PREFER_RACK,
            workers,
            [_instance(1, role="prefill"), _instance(3, role="router")],
        )
        is False
    )


def test_the_reason_is_its_own_enum_value():
    assert DegradationReasonEnum.GATHER_UNMET.value == "gather_unmet"
