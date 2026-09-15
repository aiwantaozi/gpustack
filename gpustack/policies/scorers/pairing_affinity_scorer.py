import logging
from typing import Dict, List, Optional, Sequence

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
)
from gpustack.schemas.models import ModelInstance, RoleNameEnum

logger = logging.getLogger(__name__)


# prefill and decode pull on each other; nothing else in a group has an
# opposite. Written as a map rather than an `if` so a third paired role would
# be one line here and nowhere else.
_OPPOSITE = {
    RoleNameEnum.PREFILL.value: RoleNameEnum.DECODE.value,
    RoleNameEnum.DECODE.value: RoleNameEnum.PREFILL.value,
}


class PairingAffinityScorer(ScheduleCandidatesScorer):
    """Pull a scaled-out prefill toward decode, and a decode toward prefill.

    🔴 **The opposite role, not "the group".** The obvious rule — prefer the
    worker already holding most of this group — is wrong, and only looks right
    because it coincides with this one when P and D are balanced. What a
    request actually pays for is a prefill and a decode being on the same host,
    so with `x` prefills, `y` decodes, and `p_j` / `d_j` of each on worker `j`,
    a router that picks the two ends independently gives::

        P(a P/D pair is local) = (1/xy) * sum_j p_j * d_j

    Adding a prefill on worker `j` raises the numerator by `d_j`, and the
    denominator `(x+1)*y` is the same wherever it lands. So the best worker is
    the one with the most *decodes*. Under 3P1D the group-count rule would pile
    the new prefill onto the prefills and move that sum by nothing at all.

    🔴 **The direction is the reverse of the group solver's tie-break**, and
    both are right. Forming a group, `_share_out` prefers the worker holding
    *fewer* of the group's members, which keeps one role from monopolising the
    roomiest workers and producing the all-P-here/all-D-there split — the one
    arrangement where no pair is local. That is a defence against a worst case
    on an empty board. This runs against a board that already has a
    distribution on it, and improves the sum it actually has.

    **A scorer, not a filter**, like `GroupLocalityScorer` beside it: a
    candidate only reaches here once the selector has found it can hold the
    member, so affinity reorders workers that all fit and can never make a
    scale-out unschedulable.

    **Never reached by a role-less model.** The scorer is added to the chain
    only for a member that has a `group_id`, which only a model with `roles`
    ever has. Existing single-role deployments score exactly as they did.
    """

    def __init__(
        self,
        group_id: Optional[str],
        role: Optional[str],
        model_instances: Sequence[ModelInstance],
        max_score: float = 200.0,
    ):
        self._group_id = group_id
        self._opposite = _OPPOSITE.get(role or "")
        self._model_instances = model_instances
        self._max_score = max_score

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        if (
            not candidates
            or self._max_score <= 0
            or not self._group_id
            or not self._opposite
        ):
            return candidates

        counts = self._opposite_counts()
        if not counts:
            # No member of the opposite role is placed yet — a group whose
            # decodes have not been scheduled, or a scale-out that happens to
            # be first. Nothing to be near, so every candidate scores zero and
            # the resource scorers decide alone.
            return candidates

        for candidate in candidates:
            worker_id = getattr(candidate.worker, "id", None)
            # 🔴 Multiplied, not normalised into the 0..max_score band. The
            # band would make the gap between two and three decodes
            # `max_score / max_count`, which shrinks as a group grows until
            # `PlacementScorer`'s 100-point spread outranks it — and then the
            # rule silently becomes "roomiest worker", which is what this
            # exists to override. Multiplying keeps every adjacent step worth
            # a full `max_score`, so the ordering is affinity first and
            # capacity only among equals. The total is unbounded; nothing
            # downstream does anything with it but take the maximum.
            candidate.score = self._max_score * counts.get(worker_id, 0)

        return candidates

    def _opposite_counts(self) -> Dict[int, int]:
        """`d_j` — this group's members of the opposite role, per worker.

        Counted from `worker_id` rather than from a RUNNING state. A sibling
        that has been placed but is still starting holds that worker's cards
        and will pair from there; excluding it would make the first two
        scale-outs of a burst both choose the same "empty" host for the same
        wrong reason.
        """
        counts: Dict[int, int] = {}
        for instance in self._model_instances:
            if getattr(instance, "group_id", None) != self._group_id:
                continue
            if getattr(instance, "role", None) != self._opposite:
                continue
            worker_id = getattr(instance, "worker_id", None)
            if worker_id is not None:
                counts[worker_id] = counts.get(worker_id, 0) + 1
        return counts
