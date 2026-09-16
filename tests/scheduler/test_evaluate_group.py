"""What a PD deployment's evaluation says it costs.

The bug these pin down: a role-bearing model was evaluated by asking
`find_candidate` where ONE instance of the model-level spec would go. That
answer is a single replica's claim, computed without the role's overrides,
against a placement nobody ever makes — so a 2P4D group read like a 1x
deployment and a cluster that could not hold the group still evaluated as
compatible.

So the assertions below are about the three things the old path could not say:
every replica is counted, the router is counted, and an infeasible group is
refused in the solver's own words.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.schemas.model_sets import ModelSpec
from gpustack.schemas.models import ComputedResourceClaim, RoleSpec
from gpustack.scheduler import evaluator
from gpustack.scheduler.evaluator import evaluate_group
from gpustack.scheduler.group_solver import GroupInfeasible, GroupPlacement

GIB = 1024**3


def _spec(roles=None, **kwargs) -> ModelSpec:
    return ModelSpec(
        name="pd",
        source="huggingface",
        huggingface_repo_id="x/y",
        backend="vLLM",
        roles=roles,
        **kwargs,
    )


def _roles(prefill=2, decode=2, router=1):
    return [
        RoleSpec(name="prefill", replicas=prefill),
        RoleSpec(name="decode", replicas=decode),
        RoleSpec(name="router", replicas=router),
    ]


def _candidate(worker_id: int, ram: int, vram: int):
    return SimpleNamespace(
        worker=SimpleNamespace(id=worker_id, name=f"w{worker_id}"),
        gpu_indexes=[0],
        gpu_type="cuda",
        computed_resource_claim=ComputedResourceClaim(ram=ram, vram={0: vram}),
        subordinate_workers=None,
    )


async def _run(placement, commit_map=None, spec=None, workers=None):
    """Drive `evaluate_group` with the solve and the commit pass stubbed.

    Everything between them — the role projection, the router's own claim, the
    ordering and the summing — is the real code, because that is what the old
    path got wrong.
    """

    class FakeCapacity:
        def __init__(self, *a, **kw):
            pass

        async def commit(self, role, worker_ids, already):
            return (commit_map or {}).get(role, [])

    view = SimpleNamespace(root=SimpleNamespace(), scopes=lambda: [])

    with (
        patch.object(
            evaluator.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(id=1, topology=None)),
        ),
        patch.object(evaluator, "cache_instances_in", AsyncMock(return_value=[])),
        patch.object(evaluator, "build_view", lambda *a, **kw: view),
        patch.object(evaluator, "GroupCapacity", FakeCapacity),
        patch.object(
            evaluator, "solve_group_placement", AsyncMock(return_value=placement)
        ),
    ):
        return await evaluate_group(
            config=SimpleNamespace(),
            session=None,
            model=spec if spec is not None else _spec(roles=_roles()),
            workers=workers if workers is not None else [],
            model_instances=[],
            cluster_id=1,
        )


@pytest.mark.asyncio
async def test_the_total_counts_every_replica_and_the_router():
    """🔴 The headline number. Four GPU members at 40 GiB VRAM each is 160
    GiB, not the 40 the single-instance path reported — and the router's
    container memory is part of what the group asks a cluster for."""
    placement = GroupPlacement(
        layer="Rack",
        domain="rack-a",
        assignments={"prefill": [1, 1], "decode": [2, 2]},
    )
    commit = {
        "prefill": [_candidate(1, 2 * GIB, 40 * GIB) for _ in range(2)],
        "decode": [_candidate(2, 2 * GIB, 40 * GIB) for _ in range(2)],
    }
    total, claims, messages = await _run(placement, commit)

    assert messages == []
    assert total.vram == 160 * GIB
    # 4 members x 2 GiB + the router's 2 GiB floor.
    assert total.ram == 10 * GIB
    assert {c.role: c.replicas for c in claims} == {
        "prefill": 2,
        "decode": 2,
        "router": 1,
    }


@pytest.mark.asyncio
async def test_the_breakdown_is_in_the_declared_role_order():
    """The solver sorts roles by weight and a dict keeps whatever order it was
    given; the breakdown is read by a person, so it follows the deployment."""
    placement = GroupPlacement(
        layer="Rack",
        domain="rack-a",
        assignments={"decode": [2, 2], "prefill": [1, 1]},
    )
    commit = {
        "prefill": [_candidate(1, GIB, 40 * GIB) for _ in range(2)],
        "decode": [_candidate(2, GIB, 20 * GIB) for _ in range(2)],
    }
    _total, claims, _messages = await _run(placement, commit)

    assert [c.role for c in claims] == ["prefill", "decode", "router"]


@pytest.mark.asyncio
async def test_a_uniform_role_reports_what_one_replica_costs():
    placement = GroupPlacement(
        layer="Rack", domain="rack-a", assignments={"prefill": [1, 1]}
    )
    commit = {"prefill": [_candidate(1, GIB, 40 * GIB) for _ in range(2)]}
    _total, claims, _messages = await _run(
        placement, commit, spec=_spec(roles=[RoleSpec(name="prefill", replicas=2)])
    )

    prefill = next(c for c in claims if c.role == "prefill")
    assert prefill.per_replica.vram == 40 * GIB
    assert prefill.vram == 80 * GIB


@pytest.mark.asyncio
async def test_a_role_whose_members_differ_reports_no_per_replica_figure():
    """A role spread over two accelerator types sizes differently on each.
    Showing the first member's number as if it were every member's is how a
    mixed group would read as half its real size."""
    placement = GroupPlacement(
        layer="Rack", domain="rack-a", assignments={"prefill": [1, 2]}
    )
    commit = {
        "prefill": [_candidate(1, GIB, 40 * GIB), _candidate(2, GIB, 80 * GIB)],
    }
    _total, claims, _messages = await _run(
        placement, commit, spec=_spec(roles=[RoleSpec(name="prefill", replicas=2)])
    )

    prefill = next(c for c in claims if c.role == "prefill")
    assert prefill.per_replica is None
    assert prefill.vram == 120 * GIB


@pytest.mark.asyncio
async def test_an_infeasible_group_is_refused_in_the_solvers_own_words():
    """ "The group needs 8 placements in one 'Rack', and the roomiest one holds
    6" is actionable; "unable to find a schedulable worker" is not."""
    placement = GroupInfeasible(
        reason="The group needs 8 placements in one 'Rack', and the roomiest one holds 6.",
        layer="Rack",
        best_domain="rack-a",
        needed=8,
        available=6,
    )
    total, claims, messages = await _run(placement)

    assert total is None
    assert claims == []
    assert "roomiest" in messages[0]


@pytest.mark.asyncio
async def test_a_commit_that_comes_up_short_refuses_rather_than_underprices():
    """The solve said it fits; if turning that into cards disagrees, the
    cluster moved under us. Reporting the members that did resolve would be a
    total that understates the group."""
    placement = GroupPlacement(
        layer="Rack", domain="rack-a", assignments={"prefill": [1, 1]}
    )
    total, _claims, messages = await _run(
        placement, {"prefill": [_candidate(1, GIB, 40 * GIB)]}
    )

    assert total is None
    assert "GPU assignments" in messages[0]


@pytest.mark.asyncio
async def test_a_group_of_only_routers_is_priced_without_a_solve():
    """No member occupies an accelerator, so there is nothing to solve — but
    the group still has a footprint, and reporting it beats reporting
    nothing."""
    spec = _spec(roles=[RoleSpec(name="router", replicas=1)])
    total, claims, messages = await _run(
        GroupInfeasible(reason="never consulted"), spec=spec
    )

    assert messages == []
    assert total.vram == 0
    assert total.ram == 2 * GIB
    assert [c.role for c in claims] == ["router"]


@pytest.mark.asyncio
async def test_a_declared_router_memory_replaces_the_floor():
    spec = _spec(
        roles=[
            RoleSpec(name="router", replicas=1, resources={"memory": 8 * GIB}),
        ]
    )
    total, _claims, _messages = await _run(
        GroupInfeasible(reason="never consulted"), spec=spec
    )

    assert total.ram == 8 * GIB


@pytest.mark.asyncio
async def test_a_role_bearing_model_never_reaches_find_candidate():
    """The gate, stated as a test. `find_candidate` answers "where would one
    more instance go", which is not a question a group has — and its answer is
    what the old evaluation reported."""
    placement = GroupPlacement(
        layer="Rack", domain="rack-a", assignments={"prefill": [1]}
    )
    commit = {"prefill": [_candidate(1, GIB, 40 * GIB)]}

    class FakeCapacity:
        def __init__(self, *a, **kw):
            pass

        async def commit(self, role, worker_ids, already):
            return commit.get(role, [])

    view = SimpleNamespace(root=SimpleNamespace(), scopes=lambda: [])
    find_candidate = AsyncMock(return_value=(None, []))
    worker = SimpleNamespace(id=1, cluster_id=1)

    with (
        patch.object(
            evaluator.Cluster,
            "one_by_id",
            AsyncMock(return_value=SimpleNamespace(id=1, topology=None)),
        ),
        patch.object(evaluator, "cache_instances_in", AsyncMock(return_value=[])),
        patch.object(evaluator, "build_view", lambda *a, **kw: view),
        patch.object(evaluator, "GroupCapacity", FakeCapacity),
        patch.object(
            evaluator, "solve_group_placement", AsyncMock(return_value=placement)
        ),
        patch.object(evaluator.scheduler, "find_candidate", find_candidate),
        patch.object(evaluator, "set_default_spec", AsyncMock(return_value=False)),
        patch.object(evaluator, "set_gguf_model_file_path", AsyncMock()),
        patch.object(
            evaluator, "evaluate_model_input", AsyncMock(return_value=(True, []))
        ),
        patch.object(
            evaluator, "evaluate_model_metadata", AsyncMock(return_value=(True, []))
        ),
        patch.object(
            evaluator, "evaluate_environment", AsyncMock(return_value=(True, []))
        ),
    ):
        result = await evaluator.evaluate_model(
            config=SimpleNamespace(),
            session=None,
            model=_spec(roles=[RoleSpec(name="prefill", replicas=1)]),
            workers=[worker],
            model_instances=[],
            cluster_id=1,
        )

    find_candidate.assert_not_called()
    assert result.compatible is True
    assert result.resource_claim.vram == 40 * GIB
    assert result.role_resource_claims[0].role == "prefill"
    assert result.role_resource_claims_by_cluster_id[1][0].replicas == 1
