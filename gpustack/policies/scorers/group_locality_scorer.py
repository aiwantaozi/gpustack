import logging
from typing import List, Optional, Sequence, Set

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
)
from gpustack.schemas.models import ModelInstance

logger = logging.getLogger(__name__)


class GroupLocalityScorer(ScheduleCandidatesScorer):
    """Pull a group's accelerator-free member onto a worker its siblings are on.

    The router is the only such member today. It is deliberately outside the
    gang — `schedule_group` excludes it and the group solver never sees it —
    which is right, because a proxy must not be able to make a group of
    weight-holding members unplaceable. But being outside the solve meant being
    outside *everything*: the router was placed by the ordinary path, scored by
    `PlacementScorer` across the whole cluster, and could land in another rack
    from the members whose every token it forwards. Nothing was wrong with the
    deployment; it just paid a network hop per request for no reason.

    🔴 **A scorer, not a filter, and that is the whole safety argument.** If no
    sibling's worker can take it the router still places, one hop further away
    — the cost is latency, never schedulability. A filter here would let a full
    worker hold the entire group hostage.

    It has data to work with because of an ordering that already exists for a
    different reason: a router's row is not created until every GPU role has a
    RUNNING member (`_role_dependencies`), since its command line is rendered
    from its peers' `ip:port`. So by the time this runs, the siblings are
    placed and their `worker_id` is known.

    Scored at the same scale as `PlacementScorer` rather than as a tiebreaker.
    Bin-packing has nothing to weigh for a 2 GiB proxy, and every candidate
    that reaches a scorer has already been found to fit — the selector only
    returns workers with enough RAM — so letting locality dominate cannot
    produce a placement that does not fit.
    """

    def __init__(
        self,
        group_id: Optional[str],
        model_instances: Sequence[ModelInstance],
        max_score: float = 100.0,
    ):
        self._group_id = group_id
        self._model_instances = model_instances
        self._max_score = max_score

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        if not candidates or self._max_score <= 0 or not self._group_id:
            return candidates

        sibling_workers = self._sibling_worker_ids()
        if not sibling_workers:
            # Nothing placed yet, so nothing to be near. Every candidate scores
            # zero and the other scorers decide, which is the old behaviour.
            return candidates

        for candidate in candidates:
            worker_id = getattr(candidate.worker, "id", None)
            candidate.score = self._max_score if worker_id in sibling_workers else 0.0

        return candidates

    def _sibling_worker_ids(self) -> Set[int]:
        """Workers already holding a member of this group.

        Counted per worker rather than weighted by how many members it holds:
        the router talks to prefill and decode over the same link, so one
        sibling on a host already buys the hop, and a second buys nothing more.
        """
        worker_ids: Set[int] = set()
        for instance in self._model_instances:
            if getattr(instance, "group_id", None) != self._group_id:
                continue
            worker_id = getattr(instance, "worker_id", None)
            if worker_id is not None:
                worker_ids.add(worker_id)
        return worker_ids
