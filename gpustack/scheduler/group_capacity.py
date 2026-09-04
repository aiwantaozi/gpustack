"""The bridge between "how big is this domain" and the selectors that know.

`solve_group_placement` takes a `CapacityFn` and had no implementation of one:
the only thing that can answer "how many members of this role fit on that
worker" is `count_offer_slots`, and that needs a selector factory, which needs
the role projection and the worker filters — i.e. everything `find_candidate`
does before it starts scoring. This module is that assembly, and it exists as
its own file so both consumers (the deployment form's feasibility preview and,
later, the scheduler's own group placement) go through one implementation.

**Why not reuse `find_candidate` directly.** It answers "where does one more
member go", scores the result and returns a single winner. The group solver
needs the *count* per worker, which is a different question with a different
cost profile — and going through the scoring path once per hypothetical member
per worker per domain would multiply a full selector sweep by three dimensions.
`count_offer_slots` exists precisely to collapse that.

**A worker whose capacity could not be established is left out of the mapping,
never reported as zero.** The solver reads absence as "unknown" and counts it
separately, which is what lets a refusal say "we could not measure four
workers" instead of "the cluster is full" — and those two call for opposite
reactions from an operator.
"""

from __future__ import annotations

import logging
from typing import Dict, List, NamedTuple, Optional, Sequence

from gpustack.config.config import Config
from gpustack.policies.base import WorkerFilterChain
from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.policies.worker_filters.cluster_filter import ClusterFilter
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.local_path_filter import LocalPathFilter
from gpustack.policies.worker_filters.pd_mode_filter import PDModeRuntimeFilter
from gpustack.policies.worker_filters.status_filter import StatusFilter
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    role_effective_model,
    role_container_resources,
    role_takes_no_accelerator,
)
from gpustack.schemas.workers import Worker
from gpustack.scheduler.offer_slot import _PlacedStandIn, count_offer_slots

logger = logging.getLogger(__name__)


class _RoleProjection(NamedTuple):
    """What `_eligible_for` worked out for a role, read back by two callers.

    A NamedTuple rather than a bare tuple because the role-OWN fields read
    before projection are a growing set — `cpu_only`, now `ram_claim` — and a
    positional tuple couples every read site, plus every test that builds one,
    to that count. `ram_claim` defaults so a caller that only cares about
    placement need not spell it.
    """

    model: Model
    cpu_only: bool
    ram_claim: Optional[int] = None


class GroupCapacity:
    """A `CapacityFn` for one model, with the per-role setup done once.

    Stateful on purpose. The role projection, the filter sweep and the selector
    choice do not change between domains, and the solver asks for the same role
    against different worker subsets — redoing the setup per call was measured
    upstream at a full selector sweep per domain per layer.
    """

    def __init__(
        self,
        config: Config,
        model: Model,
        workers: Sequence[Worker],
        model_instances: Sequence[ModelInstance],
    ):
        self._config = config
        self._model = model
        self._workers = {w.id: w for w in workers}
        self._model_instances = list(model_instances)
        # role -> {worker_id: worker}, after that role's filters.
        self._eligible: Dict[Optional[str], Dict[int, Worker]] = {}
        self._projected: Dict[Optional[str], "_RoleProjection"] = {}
        # (role, worker_id) -> one member's claim there, learned while counting.
        # See `_translate`.
        self._claims: Dict[tuple, tuple] = {}

    async def __call__(
        self,
        role: str,
        worker_ids: Sequence[int],
        already_placed: Sequence[object],
    ) -> Dict[int, int]:
        """How many more members of `role` each of `worker_ids` can take.

        `already_placed` is what the solve has committed so far, in the shape
        the allocation accounting reads. It is appended to the instance list
        rather than merged into it, which is what keeps the second role honest
        about what the first one took.
        """
        eligible = await self._eligible_for(role)
        if not eligible:
            return {}

        projected = self._projected[role]
        instances = self._model_instances + self._translate(already_placed)
        limit = self._limit_for(role)

        out: Dict[int, int] = {}
        for worker_id in worker_ids:
            worker = eligible.get(worker_id)
            if worker is None:
                # Filtered out for this role — a measured, definite zero, not
                # an unknown. Present-and-zero is what tells the solver the
                # domain is genuinely too small rather than unmeasurable.
                out[worker_id] = 0
                continue
            offer = await count_offer_slots(
                make_selector=lambda instances_now, p=projected: (
                    self._selector(p.model, instances_now, p.cpu_only, p.ram_claim)
                ),
                worker=worker,
                model_instances=instances,
                limit=limit,
            )
            if offer.placements:
                # Learned here so `_translate` can turn the solver's commits
                # into something the allocation accounting can read.
                first = offer.placements[0]
                self._claims[(role, worker_id)] = (
                    getattr(first, "gpu_type", None),
                    getattr(first, "computed_resource_claim", None),
                )
            if offer.unavailable and offer.slots == 0:
                # Nothing was proven about this worker. Leaving it out is the
                # difference between "no room" and "we could not look".
                continue
            out[worker_id] = offer.slots
        return out

    def _translate(self, already_placed: Sequence[object]) -> List[object]:
        """The solver's commits, in the shape the allocation accounting reads.

        🔴 Found by running this on a live server, not by a test: the solver
        commits `_Committed(worker_id, role)` — deliberately just those two,
        with its docstring saying "the capacity function re-derives the real
        resource claim itself, and inventing one here would be a second,
        quieter accounting of the same placement". That re-derivation is this
        method, and its absence surfaced as
        `'_Committed' object has no attribute 'gpu_type'` from deep inside
        `compute_worker_allocated`, which reads four fields off every entry.

        The claim comes from what the selector already produced for that
        (role, worker) while counting — the same number the placement decision
        was made against. Anything computed a second way here would be the
        second accounting the solver's docstring warns about.

        Entries that already look like instances (or stand-ins) pass through:
        the commit pass builds those itself.
        """
        out: List[object] = []
        for entry in already_placed:
            if getattr(entry, "computed_resource_claim", None) is not None or hasattr(
                entry, "distributed_servers"
            ):
                out.append(entry)
                continue
            worker_id = getattr(entry, "worker_id", None)
            role = getattr(entry, "role", "") or ""
            known = self._claims.get((role, worker_id))
            if known is None:
                # Unreachable in practice: the solver only commits to workers a
                # previous count already examined, and a count that returned
                # slots also returned placements. Skipping rather than
                # inventing a claim, and saying so, because a made-up claim
                # would make the domain look either roomier or tighter than the
                # placement it is meant to reflect.
                logger.warning(
                    "No learned claim for role %r on worker %s; the commit is "
                    "not counted against remaining capacity.",
                    role,
                    worker_id,
                )
                continue
            gpu_type, claim = known
            out.append(
                _PlacedStandIn(
                    worker_id=worker_id,
                    gpu_indexes=None,
                    gpu_type=gpu_type,
                    computed_resource_claim=claim,
                )
            )
        return out

    def _selector(self, model, instances, cpu_only, ram_claim=None):
        # Imported here rather than at module scope: `scheduler` imports a wide
        # slice of the policy stack, and this module is imported by a route.
        from gpustack.scheduler.scheduler import build_candidate_selector

        return build_candidate_selector(
            self._config,
            model,
            instances,
            cpu_only=cpu_only,
            ram_claim=ram_claim,
        )

    def _limit_for(self, role: str) -> int:
        """Never count past what the group could place.

        Every extra slot costs a full selector pass, and capacity beyond the
        group's own member count answers a question nobody asked.
        """
        for spec in self._model.roles or []:
            if spec.name == role:
                return max(int(spec.replicas or 0), 0)
        return max(int(self._model.replicas or 1), 0)

    async def _eligible_for(self, role: str) -> Dict[int, Worker]:
        if role in self._eligible:
            return self._eligible[role]

        # Read before projecting: `cpu_only` is a role-OWN field and the
        # projection flattens the role's overrides onto the model.
        cpu_only = role_takes_no_accelerator(self._model, role)
        # `resources` is role-OWN as well, and only the accelerator-free
        # branch consumes it — same read-before-projection reason.
        ram_claim = (
            role_container_resources(self._model, role).memory if cpu_only else None
        )
        model = role_effective_model(self._model, role)
        self._projected[role] = _RoleProjection(model, cpu_only, ram_claim)

        chain = WorkerFilterChain(
            [
                ClusterFilter(model),
                GPUMatchingFilter(model),
                LabelMatchingFilter(model),
                StatusFilter(model),
                BackendFrameworkFilter(model),
                LocalPathFilter(model),
                PDModeRuntimeFilter(model),
            ]
        )
        try:
            kept, _ = await chain.filter(list(self._workers.values()))
        except Exception as e:
            # A broken filter is not an empty cluster. Returning nothing here
            # would make every domain look too small and the refusal would
            # blame capacity.
            logger.warning(
                "Could not filter workers for role %r; treating every worker "
                "as eligible and letting the selectors decide: %s",
                role,
                e,
            )
            kept = list(self._workers.values())

        self._eligible[role] = {w.id: w for w in kept}
        return self._eligible[role]

    async def commit(
        self,
        role: str,
        worker_ids: Sequence[int],
        already_placed: Sequence[object],
    ) -> List[object]:
        """The concrete candidates for `worker_ids`, in the order given.

        🔴 This is the half of step 6 the solver cannot do. `GroupPlacement`
        answers *which worker* each member goes to; a member also needs which
        cards, and those only exist inside the candidates `count_offer_slots`
        already produced and threw away.

        Re-derived here rather than cached during the count, and that is
        deliberate: the count runs once per domain per layer while searching,
        so a cache would hold whichever domain was examined last — not the one
        that won. Running it again against the winning assignment is one extra
        pass and is consistent by construction.

        `worker_ids` may repeat: two members of one role on one worker means
        that worker appears twice, and the second entry must be the *second*
        candidate the selector offers, computed with the first already
        standing in. Anything else hands out the same cards twice.
        """
        eligible = await self._eligible_for(role)
        projected = self._projected[role]
        instances = self._model_instances + self._translate(already_placed)

        wanted: Dict[int, int] = {}
        for worker_id in worker_ids:
            wanted[worker_id] = wanted.get(worker_id, 0) + 1

        by_worker: Dict[int, List[object]] = {}
        for worker_id, count in wanted.items():
            worker = eligible.get(worker_id)
            if worker is None:
                return []
            offer = await count_offer_slots(
                make_selector=lambda instances_now, p=projected: (
                    self._selector(p.model, instances_now, p.cpu_only, p.ram_claim)
                ),
                worker=worker,
                model_instances=instances,
                limit=count,
            )
            if offer.slots < count:
                # The winning solve said this fits. If the commit pass
                # disagrees, the cluster moved under us — report nothing
                # rather than place some members and leave the rest, because a
                # partial group is the one state D14 exists to prevent.
                logger.warning(
                    "Commit pass came up short for role %r on worker %s: "
                    "wanted %d, got %d. Treating the group as unplaceable.",
                    role,
                    worker_id,
                    count,
                    offer.slots,
                )
                return []
            by_worker[worker_id] = list(offer.placements[:count])

        # Handed back in the order the caller asked, so a member's index in
        # the assignment list matches its candidate.
        cursor: Dict[int, int] = {}
        out: List[object] = []
        for worker_id in worker_ids:
            index = cursor.get(worker_id, 0)
            cursor[worker_id] = index + 1
            out.append(by_worker[worker_id][index])
        return out


def role_demands(model: Model) -> List[dict]:
    """The group's shape, as the solver's `RoleDemand` fields.

    Returned as plain dicts so this module does not have to import the solver's
    dataclass — the route that owns both does the construction.

    `weight` orders the roles when they compete for the same cards: the
    hungriest first. Cards-per-member is the honest proxy available here, and
    getting the order wrong is the difference between "4P4D fits" and "4P4D
    fits only if you happen to place P first".
    """
    demands: List[dict] = []
    for spec in model.roles or []:
        if role_takes_no_accelerator(model, spec.name):
            # A router occupies no accelerator, so it neither competes for
            # cards nor constrains which domain the group lands in. Counting it
            # would make a 4P4D look like nine members needing one domain.
            continue
        projected = role_effective_model(model, spec.name)
        per_member = _cards_per_member(projected)
        demands.append(
            {
                "role": spec.name,
                "replicas": max(int(spec.replicas or 0), 0),
                "weight": float(per_member),
            }
        )
    return demands


def _cards_per_member(model) -> int:
    """How many accelerators one member of this role wants.

    Only used to order the roles, so a wrong answer costs placement quality
    rather than correctness — which is why it reads the declared selector
    rather than re-deriving parallelism from backend parameters.
    """
    selector = getattr(model, "gpu_selector", None)
    per_replica = getattr(selector, "gpus_per_replica", None) if selector else None
    try:
        return max(int(per_replica or 1), 1)
    except (TypeError, ValueError):
        return 1
