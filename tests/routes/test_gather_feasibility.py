"""The deployment form's three tiers, answered against live capacity.

What is tested here is the wiring, not the solver: the solver has its own
suite. So the cases are the ones the endpoint decides — leaf-first ordering,
the router being excluded from the group's demands, "we could not measure"
surviving all the way to the response, and a whole model spec being accepted
loosely enough that a form posting its entire state does not get a 500.
"""

from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from gpustack.api.exceptions import BadRequestException
from gpustack.routes import cluster_topology as route
from gpustack.schemas.clusters import ClusterTopology
from gpustack.scheduler.group_solver import (
    GroupInfeasible,
    GroupPlacement,
)
from gpustack.scheduler.topology import NODE_LAYER

RACK = "topology.gpustack.ai/rack"


def _worker(id: int, name: str, labels=None):
    return SimpleNamespace(id=id, name=name, labels=labels or {}, status=None)


async def _feasibility(spec, workers=None, topology=None, solve=None):
    """Call the handler with the ORM, the config and the solver stubbed.

    The solver is stubbed by default because its own behaviour is covered in
    `tests/scheduler/test_group_solver.py`; what matters here is which
    requests the endpoint makes and how it renders the answers.
    """
    workers = workers if workers is not None else [_worker(1, "w1", {RACK: "rack-a"})]
    cluster = SimpleNamespace(id=1, topology=topology)
    calls = []

    async def fake_solve(root, demands, capacity, layers, gather):
        calls.append(gather)
        if solve is None:
            return GroupPlacement(layer=gather.layer or "root", domain="rack-a")
        return solve(gather)

    with (
        patch.object(route.Cluster, "one_by_id", AsyncMock(return_value=cluster)),
        patch.object(route.Worker, "all_by_field", AsyncMock(return_value=workers)),
        patch.object(route, "assert_cluster_visible", lambda *a, **k: None),
        patch(
            "gpustack.schemas.models.ModelInstance.all", new=AsyncMock(return_value=[])
        ),
        patch("gpustack.config.config.get_global_config", lambda: SimpleNamespace()),
        patch("gpustack.scheduler.group_solver.solve_group_placement", new=fake_solve),
    ):
        result = await route.gather_feasibility(
            session=None,
            ctx=None,
            id=1,
            body=route.GatherFeasibilityRequest(model_spec=spec),
        )
    return result, calls


def _pd_spec(prefill=2, decode=2, router=True):
    """A group as the deployment form would post it.

    `source` is present because the endpoint requires it, and the endpoint
    requires it because the resource-fit selectors read it — which selector
    runs decides what "fits" means, so answering without it would answer a
    different question than the one the scheduler will.
    """
    roles = [
        {"name": "prefill", "replicas": prefill},
        {"name": "decode", "replicas": decode},
    ]
    if router:
        roles.append({"name": "router", "replicas": 1, "cpu_only": True})
    return {
        "name": "m1",
        "replicas": 1,
        "source": "huggingface",
        "huggingface_repo_id": "Qwen/Qwen3-0.6B",
        "roles": roles,
    }


# --- the tiers -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tiers_are_leaf_first():
    """The tightest choice is the one that exists whatever the cluster
    declared, and the one most deployments want, so it leads."""
    result, _ = await _feasibility(
        _pd_spec(),
        topology=ClusterTopology.model_validate(
            {"layers": [{"name": "Rack", "labelKeys": [RACK]}]}
        ),
    )
    assert [t.layer for t in result.tiers] == [NODE_LAYER, "Rack"]


@pytest.mark.asyncio
async def test_a_cluster_with_no_layers_still_offers_the_tightest_tier():
    """The leaf is built in, so "at least on the same host" is never absent —
    which is the tier the design calls the most useful one."""
    result, _ = await _feasibility(_pd_spec())
    assert [t.layer for t in result.tiers] == [NODE_LAYER]


@pytest.mark.asyncio
async def test_every_tier_is_asked_as_a_must():
    """A tier is the question "would you rather not deploy below this", so it
    has to be solved with `must` set. Solving it as a preference would make
    every tier feasible and the control meaningless."""
    _, calls = await _feasibility(
        _pd_spec(),
        topology=ClusterTopology.model_validate(
            {"layers": [{"name": "Rack", "labelKeys": [RACK]}]}
        ),
    )
    tier_calls = [c for c in calls if c.layer is not None]
    assert tier_calls and all(c.must for c in tier_calls)


@pytest.mark.asyncio
async def test_prefer_is_solved_separately_and_without_must():
    """ "As close as possible, deploy anyway" is a different question, and
    presenting it as a tier that always says yes would flatten the distinction
    the control exists to draw."""
    _, calls = await _feasibility(_pd_spec())
    prefer_calls = [c for c in calls if c.layer is None]
    assert len(prefer_calls) == 1
    assert prefer_calls[0].must is False


# --- what reaches the form -------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_feasible_tier_names_the_domain_it_would_land_in():
    result, _ = await _feasibility(_pd_spec())
    assert result.tiers[0].feasible is True
    assert result.tiers[0].domain == "rack-a"


@pytest.mark.asyncio
async def test_a_refusal_carries_the_shortfall_and_the_roomiest_domain():
    """ "rack-a is 2 cards short" is actionable; "it does not fit" is not."""

    def infeasible(gather):
        return GroupInfeasible(
            reason="The group needs 8 placements in one 'Rack', and the roomiest one holds 6.",
            layer=gather.layer,
            best_domain="rack-a",
            needed=8,
            available=6,
        )

    result, _ = await _feasibility(_pd_spec(prefill=4, decode=4), solve=infeasible)
    tier = result.tiers[0]
    assert tier.feasible is False
    assert tier.best_domain == "rack-a"
    assert (tier.needed, tier.available) == (8, 6)
    assert "roomiest" in tier.reason


@pytest.mark.asyncio
async def test_unmeasured_workers_survive_to_the_response():
    """🔴 "We could not look" and "there is no room" call for opposite
    reactions, so the count has to reach the form rather than be folded into
    `available`."""

    def unmeasurable(gather):
        return GroupInfeasible(
            reason="capacity could not be measured on 3 worker(s)",
            layer=gather.layer,
            needed=8,
            available=0,
            unmeasured=3,
        )

    result, _ = await _feasibility(_pd_spec(), solve=unmeasurable)
    assert result.tiers[0].unmeasured == 3


# --- what counts as a member ------------------------------------------------ #


@pytest.mark.asyncio
async def test_the_router_is_not_part_of_the_group_the_domain_must_hold():
    """It occupies no accelerator, so it neither competes for cards nor
    constrains the domain. Counting it would make a 4P4D look like nine
    members needing one rack."""
    from gpustack.scheduler.group_capacity import role_demands
    from gpustack.schemas.models import Model, RoleSpec

    spec = _pd_spec(prefill=4, decode=4)
    model = Model(**spec)
    # Coerced the way the route does, and for the reason the route documents:
    # SQLModel skips validation on `table=True` classes, so `roles` would stay
    # a list of raw dicts. A Model loaded from the database has real RoleSpec
    # objects; only hand construction produces the dicts.
    model.roles = [RoleSpec.model_validate(r) for r in spec["roles"]]
    demands = role_demands(model)
    assert {d["role"] for d in demands} == {"prefill", "decode"}
    assert sum(d["replicas"] for d in demands) == 8


@pytest.mark.asyncio
async def test_a_group_with_nothing_on_an_accelerator_is_trivially_feasible():
    """Saying so beats solving for it, and beats reporting a refusal nobody
    can act on."""
    result, calls = await _feasibility(
        {
            "name": "m1",
            "source": "huggingface",
            "huggingface_repo_id": "Qwen/Qwen3-0.6B",
            "roles": [{"name": "router", "replicas": 1, "cpu_only": True}],
        }
    )
    assert all(t.feasible for t in result.tiers)
    assert result.prefer.feasible is True
    # Nothing was solved: there was no group to place.
    assert calls == []


# --- the request shape ------------------------------------------------------ #


@pytest.mark.asyncio
async def test_unknown_keys_in_the_posted_spec_are_ignored_not_fatal():
    """The form posts its whole state. A stray key must not be a 500 when the
    honest answer is "ignored"."""
    spec = dict(_pd_spec())
    spec["some_ui_only_field"] = {"nested": True}
    result, _ = await _feasibility(spec)
    assert result.tiers


@pytest.mark.asyncio
async def test_a_spec_that_cannot_be_a_model_is_a_400():
    with pytest.raises(BadRequestException):
        await _feasibility(
            {
                "name": "m1",
                "source": "huggingface",
                "huggingface_repo_id": "Qwen/Qwen3-0.6B",
                "replicas": "not-a-number",
            }
        )


@pytest.mark.asyncio
async def test_a_declaration_that_cannot_become_a_tree_is_a_400():
    with pytest.raises(BadRequestException):
        await _feasibility(
            _pd_spec(),
            topology=ClusterTopology.model_validate(
                {
                    "layers": [
                        {"name": "Rack", "labelKeys": [RACK]},
                        {"name": "Zone", "labelKeys": ["z"]},
                    ]
                }
            ),
        )


# ---------------------------------------------------------------------------
# The form's live state is not a saved model, and the projection revalidates.
# ---------------------------------------------------------------------------


def test_a_single_select_bound_to_a_list_field_does_not_500():
    """🔴 The regression this exists for, found on a live server.

    The deployment form holds `categories` as a scalar while it is open and
    wraps it only on submit (`data.categories ? [data.categories] : []`). This
    endpoint is fed the *live* state, so it sees `"llm"`.

    `Model(**spec)` waves that through — SQLModel skips validation on
    `table=True` — and the failure surfaced three frames deeper, inside
    `role_effective_model`, which revalidates the *whole* model. So the
    endpoint's original assumption ("coerce only the two fields the capacity
    walk reads structurally") was wrong: the projection reads all of them.
    """
    from gpustack.routes.cluster_topology import _widen_single_selects
    from gpustack.schemas.models import Model, RoleSpec
    from gpustack.scheduler.group_capacity import role_demands

    spec = {
        "name": "pd",
        "source": "huggingface",
        "huggingface_repo_id": "a/b",
        "categories": "llm",
        "roles": [
            {"name": "prefill", "replicas": 2},
            {"name": "decode", "replicas": 2},
        ],
    }

    def build(raw):
        model = Model(**{k: v for k, v in raw.items() if k in Model.model_fields})
        model.roles = [RoleSpec.model_validate(r) for r in raw["roles"]]
        return model

    with pytest.raises(ValidationError):
        role_demands(build(spec))

    demands = role_demands(build(_widen_single_selects(spec)))
    assert [d["role"] for d in demands] == ["prefill", "decode"]


def test_widening_reads_the_annotation_rather_than_naming_the_field():
    """Named fields would have to be maintained. Reading each field's own
    declared type means the next single-select bound to a list does not
    reintroduce the same 500."""
    from gpustack.routes.cluster_topology import _declares_a_list, _widen_single_selects

    assert _declares_a_list(List[str])
    assert _declares_a_list(Optional[List[str]])
    assert not _declares_a_list(str)
    assert not _declares_a_list(Optional[int])

    # A scalar field keeps its scalar, a list keeps its list, None stays None,
    # and a key the Model does not have is left for the caller's own filter.
    assert _widen_single_selects({"replicas": 2}) == {"replicas": 2}
    assert _widen_single_selects({"categories": ["llm"]}) == {"categories": ["llm"]}
    assert _widen_single_selects({"categories": None}) == {"categories": None}
    assert _widen_single_selects({"scheduleType": "auto"}) == {"scheduleType": "auto"}
