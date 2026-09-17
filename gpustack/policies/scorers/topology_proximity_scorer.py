import logging
from typing import Dict, List, Optional, Sequence

from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
)
from gpustack.schemas.models import ModelInstance, member_worker_ids
from gpustack.scheduler.topology import (
    NODE_LAYER,
    ROOT_LAYER,
    common_layer,
    order_layers,
)
from gpustack.scheduler.topology_view import TopologyView

logger = logging.getLogger(__name__)


class TopologyProximityScorer(ScheduleCandidatesScorer):
    """Prefer a worker close to the group's placed members, by the declared tree.

    🔴 **"As near as possible" only ever meant "the same host".** The two
    scorers that pull a later member toward its group -- `PairingAffinityScorer`
    for a scaled-out prefill or decode, `GroupLocalityScorer` for the router --
    both compare `worker_id` and nothing else. Between a worker in the members'
    own rack and one three racks away they are indifferent, so the deploy
    form's top option, «as close as possible», stopped at the host and the
    tiers below it were only ever a target to be reported against, never one
    to aim at.

    This reads the cluster's own tree instead: a candidate scores by the
    tightest layer it shares with the members already placed, deeper being
    better. Sharing only the root -- or only the unclassified bucket, which
    `common_layer` refuses to treat as togetherness -- scores nothing, which is
    the honest answer for "we do not know where these are".

    **Orthogonal to `PairingAffinityScorer`, not a replacement.** That one
    answers "which host has the most of the opposite role", which is the
    probability that a request's two ends land together and is a different
    question from distance -- under 3P1D it deliberately prefers the host with
    the single decode over the one with two prefills, and no notion of
    proximity would produce that. The two are summed: pairing decides among
    hosts, proximity decides among racks.

    **A scorer, never a filter.** A candidate reaches here only after the
    selector found it can hold the member, so this reorders workers that all
    fit. Being far away is a latency cost; the refusal for a floor that must be
    honoured is `GatherFloorFilter`, and it is a separate decision on purpose.
    """

    def __init__(
        self,
        group_id: Optional[str],
        model_instances: Sequence[ModelInstance],
        view: Optional[TopologyView],
        anchors: Sequence[str] = (),
        max_score: float = 100.0,
    ):
        self._group_id = group_id
        self._model_instances = list(model_instances)
        self._view = view
        # Which roles say where the group *is*. The router holds no weights, so
        # it cannot anchor the group -- the same reason `role_demands` leaves it
        # out of the gang -- but it is scored against the anchors like anyone.
        self._anchors = set(anchors)
        self._max_score = max_score

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        if (
            not candidates
            or self._max_score <= 0
            or not self._group_id
            or self._view is None
        ):
            return candidates

        placed = {
            worker_id
            for instance in self._model_instances
            if instance.group_id == self._group_id and instance.role in self._anchors
            # A member that spans machines is near a candidate from any of
            # them; measuring only from its primary would call a candidate
            # sharing a rack with the member's other half "far".
            for worker_id in member_worker_ids(instance)
        }
        if not placed:
            # Nothing to be near yet. Every candidate scores zero and the
            # resource scorers decide alone, exactly as before this existed.
            return candidates

        depth = self._depths()
        if not depth:
            return candidates

        leaves = {
            worker_id: node
            for node in _leaves(self._view.root)
            for worker_id in node.descendant_worker_ids()
        }
        anchors = [leaves[worker_id] for worker_id in placed if worker_id in leaves]
        if not anchors:
            return candidates

        # 🔴 **Lexicographic, not a weighted sum**, and the weight below is how
        # that is spelled. A candidate that spans machines has two distances
        # and they pull against each other: how far its own machines are from
        # each other, and how far the whole of it is from the rest of the
        # group. They are not comparable quantities -- the first carries the
        # tensor-parallel all-reduce, once per layer per token, and the second
        # carries the KV transfer, once per request. Adding them would let a
        # large enough gain on the second buy a worse first, which is never
        # right at that ratio.
        #
        # So internal spread decides, and closeness to the group only breaks
        # ties among candidates that are equally tight inside. Written as a
        # penalty rather than a bonus so a single machine -- which has no pair
        # to be apart, and is every candidate on the path this code was written
        # for -- scores exactly what it scored before: zero for a worker whose
        # position is unknown, and the closeness term alone otherwise. `depth`
        # is bounded by the declared rungs, so a multiplier one past the
        # tightest makes any penalty outweigh any closeness gain, which is what
        # "lexicographic" means here. `PairingAffinityScorer` spells the same
        # trick the same way, and for the same reason.
        tightest = max(depth.values(), default=0)

        for candidate in candidates:
            span = [
                leaves[worker_id]
                for worker_id in _candidate_workers(candidate)
                if worker_id in leaves
            ]
            if not span:
                continue
            # The tightest layer this candidate shares with ANY placed member.
            # Any rather than all: the member is one process talking to one
            # peer at a time, so being in a rack with three of them is not
            # three times better than being in a rack with one -- that is
            # `PairingAffinityScorer`'s question, and it is already asked.
            near = max(
                (
                    depth.get(common_layer(anchor, node) or ROOT_LAYER, 0)
                    for anchor in anchors
                    for node in span
                ),
                default=0,
            )
            apart = tightest - _internal_depth(span, depth)
            candidate.score = (
                (candidate.score or 0)
                + self._max_score * near
                - self._max_score * (tightest + 1) * apart
            )

        return candidates

    def _depths(self) -> Dict[str, int]:
        """Layer id -> how tight it is, tightest highest.

        Read off the cluster's declaration rather than assumed, because the
        chain is the operator's: whether an accelerator domain sits above or
        below a rack is something they decided, and hardcoding an order here
        would be a second opinion about their own topology.
        """
        try:
            order = [spec.layer for spec in order_layers(self._view.specs)]
        except Exception as e:
            logger.debug("Could not order the topology layers: %s", e)
            return {}
        # Root-to-leaf, so the last is the tightest. The built-in host layer is
        # not in `specs`; it is tighter than anything declared, so it takes the
        # rank above the deepest one.
        depths = {layer: index + 1 for index, layer in enumerate(order)}
        depths[NODE_LAYER] = len(order) + 1
        depths[ROOT_LAYER] = 0
        return depths


def _candidate_workers(candidate) -> List[int]:
    """Every machine this candidate would occupy, primary first.

    A candidate that spans machines carries the rest on
    `subordinate_workers`; scoring only its primary would call it close to a
    group it is half a cluster away from.
    """
    primary = getattr(getattr(candidate, "worker", None), "id", None)
    out: List[int] = [] if primary is None else [primary]
    for subordinate in getattr(candidate, "subordinate_workers", None) or []:
        worker_id = getattr(subordinate, "worker_id", None)
        if worker_id is not None and worker_id not in out:
            out.append(worker_id)
    return out


def _internal_depth(span: List, depth: Dict[str, int]) -> int:
    """How tight this candidate is inside itself, loosest pair deciding.

    A single machine is as tight as it gets -- there is no pair to be apart --
    which is what keeps every placement made today scoring exactly as it did.
    """
    if len(span) < 2:
        return max(depth.values(), default=0)
    return min(
        depth.get(common_layer(a, b) or ROOT_LAYER, 0)
        for index, a in enumerate(span)
        for b in span[index + 1 :]
    )


def _leaves(node) -> List:
    """Every host node under `node`."""
    if not node:
        return []
    if node.layer == NODE_LAYER:
        return [node]
    out: List = []
    for child in getattr(node, "children", []) or []:
        out.extend(_leaves(child))
    return out
