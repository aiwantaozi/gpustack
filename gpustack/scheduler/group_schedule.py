"""Placing a whole group at once, and the gate that keeps it away from
everything else.

The group solver, the topology tree and the capacity bridge have all existed
for a while with **no caller**: nothing in the scheduler knew they were there.
This is the entry point (design §3.7.10 G-1), plus the half of step 6 the
solver does not answer — which worker is not enough, a member also needs its
cards (G-2).

**The safety property is the gate, not the algorithm.** Group scheduling runs
only for a model that has `roles`, and only the first time that generation
forms. Everything else — every single-role deployment, and every later
per-role scale-out — falls through to `_schedule_one` on a path this module
does not touch. That is what makes "the group is never spread" a decision
confined to groups: it needs no compatibility argument for existing models,
because existing models never reach here.

**Why one arriving instance places all of them.** The schedule queue delivers
instances one at a time, so a 2P2D group arrives as four separate items. Left
to itself each would be placed independently, which is exactly the greedy trap
the solver exists to remove ("place two, and the other two no longer fit").
So the first member of an unplaced group solves for the whole group and writes
every member's row; the siblings that arrive afterwards find themselves already
scheduled and are skipped.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.config.config import Config
from gpustack.schemas.clusters import Cluster, GatherStrategyEnum
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    RoleNameEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.scheduler.group_capacity import GroupCapacity, role_demands
from gpustack.scheduler.group_solver import (
    GatherRequest,
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.scheduler.topology import (
    TopologyError,
    TopologyLayerSpec,
    build_topology,
    layer_names,
)

logger = logging.getLogger(__name__)


def is_group_forming(model: Model, instances: Sequence[ModelInstance]) -> bool:
    """Whether this generation of `model` is a group that has not been placed.

    Three conditions, and each excludes a case that must keep its current
    behaviour:

    - **`roles` is non-empty.** A role-less model is not a group; it has
      replicas, and replicas are interchangeable. This is the condition that
      keeps every existing deployment out of here.
    - **At least one GPU-bearing member is unplaced.** Nothing to solve
      otherwise.
    - **No GPU-bearing member is already placed.** A group with some members
      on workers is a scale-out, not a forming: solving the whole group again
      would either move running members (it cannot) or place the new one
      against a stale picture. Scale-out stays on the per-instance path.
    """
    if not model.roles:
        return False

    gpu_members = [i for i in instances if i.role != RoleNameEnum.ROUTER.value]
    if not gpu_members:
        return False
    if all(i.worker_id is not None for i in gpu_members):
        return False
    return not any(i.worker_id is not None for i in gpu_members)


def _gather_request(model: Model, cluster: Optional[Cluster]) -> GatherRequest:
    """The group's gather requirement, model over cluster.

    The model's own `gather` wins; absent, the cluster's default applies. That
    is the inheritance §2.5.3 describes, and it is resolved here rather than in
    the solver so the solver stays a function of its arguments.
    """
    spec = getattr(model, "gather", None)
    if spec and spec.strategy:
        return GatherRequest(
            layer=spec.layer,
            must=spec.strategy == GatherStrategyEnum.MUST_GATHER,
        )
    topology = cluster.topology if cluster else None
    if topology and topology.default_gather_strategy:
        return GatherRequest(
            layer=topology.default_gather_layer,
            must=(topology.default_gather_strategy == GatherStrategyEnum.MUST_GATHER),
        )
    return GatherRequest()


async def schedule_group(
    session: AsyncSession,
    config: Config,
    model: Model,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    group_instances: List[ModelInstance],
) -> Tuple[Optional[Dict[int, object]], List[str]]:
    """Solve the whole group, and return one candidate per member row.

    Returns `(by_instance_id, messages)`. A `None` mapping means the group
    cannot be placed — all-or-nothing (D14), so the caller must not place a
    subset. `messages` carries the refusal in the solver's own words, which
    name the shortfall and the roomiest domain rather than saying "no room".
    """
    cluster = (
        await Cluster.one_by_id(session, model.cluster_id) if model.cluster_id else None
    )
    specs = [
        TopologyLayerSpec(
            layer=layer.name,
            label_keys=list(layer.label_keys or []),
            parent_layer=layer.parent_layer,
        )
        for layer in ((cluster.topology.layers if cluster and cluster.topology else []))
    ]
    try:
        root = build_topology(specs, workers)
        names = layer_names(specs)
    except TopologyError as e:
        # A declaration that cannot become a tree is an operator error, not a
        # capacity one. Refusing the group with the reason beats placing it
        # against a tree built from a guess.
        return None, [f"Cluster topology is invalid: {e}"]

    demands = [RoleDemand(**d) for d in role_demands(model)]
    if not demands:
        return None, ["The group has no member that occupies an accelerator."]

    capacity = GroupCapacity(config, model, workers, model_instances)
    placement = await solve_group_placement(
        root, demands, capacity, names, _gather_request(model, cluster)
    )
    if not isinstance(placement, GroupPlacement):
        return None, [getattr(placement, "reason", "The group does not fit.")]

    logger.info(
        "Group %s placed in %s %r",
        model.name,
        placement.layer,
        placement.domain,
    )

    # Step 6's second half. `already` accumulates across roles for the same
    # reason the count does: the second role has to see what the first took, or
    # both are offered the same cards.
    already: List[object] = []
    by_instance: Dict[int, object] = {}
    for role, worker_ids in placement.assignments.items():
        candidates = await capacity.commit(role, worker_ids, already)
        if len(candidates) != len(worker_ids):
            return None, [
                "The group's placement could not be turned into GPU "
                f"assignments for role '{role}'; the cluster changed during "
                "scheduling."
            ]
        rows = [i for i in group_instances if i.role == role and i.worker_id is None]
        if len(rows) < len(candidates):
            # Fewer rows than the solve placed: the convergence loop has not
            # created them all yet. Refuse rather than place part of a role —
            # the next cycle sees a complete picture.
            return None, [
                f"Role '{role}' has {len(rows)} unplaced members but the "
                f"group was solved for {len(candidates)}; waiting for the "
                "rest to be created."
            ]
        for row, candidate in zip(rows, candidates):
            by_instance[row.id] = candidate
            already.append(_stand_in(candidate))

    return by_instance, []


def _stand_in(candidate) -> object:
    """What the allocation accounting reads off a placed instance.

    The same four fields `offer_slot._PlacedStandIn` carries, and for the same
    reason: a real `ModelInstance` would mean either touching the session or
    filling in values that are lies.
    """
    from gpustack.scheduler.offer_slot import _PlacedStandIn

    subordinates = getattr(candidate, "subordinate_workers", None)
    return _PlacedStandIn(
        worker_id=candidate.worker.id,
        gpu_indexes=candidate.gpu_indexes,
        gpu_type=candidate.gpu_type,
        computed_resource_claim=candidate.computed_resource_claim,
        distributed_servers=(
            _Subordinates(subordinate_workers=list(subordinates))
            if subordinates
            else None
        ),
    )


class _Subordinates:
    def __init__(self, subordinate_workers):
        self.subordinate_workers = subordinate_workers


def unplaced_states() -> Tuple[ModelInstanceStateEnum, ...]:
    """The states a member is in before it has a worker.

    Named rather than inlined because the group gate reads it and so does the
    caller, and the two disagreeing would mean a group solved for members the
    caller then skips.
    """
    return (
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.ANALYZING,
    )
