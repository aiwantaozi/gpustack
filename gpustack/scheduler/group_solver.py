"""Place every member of a group at once, in the tightest domain that fits.

The question a per-instance scheduler cannot answer is "can all of these land
together". Asking it once per member and taking the first fit each time is not
the same question: a greedy walk can place members one and two in a way that
makes three and four impossible, while a different assignment would have fitted
all four. This solves for the whole group instead, and returns the placement it
proved rather than a verdict someone else has to reproduce.

That last part is the point. A feasibility *pre-check* — compute a packing, say
"yes", then let the ordinary greedy scheduler place the members — can be right
and still fail: the greedy walk is not obliged to rediscover the packing the
check found. Both are correct and the group still never starts. Here the search
and the assignment are the same pass, so the answer cannot disagree with itself.

Adapted from koordinator's network-topology solver, whose shape has been in
production; the parts it does not have are noted where they appear.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from gpustack.scheduler.topology import NODE_LAYER, TopologyNode, nodes_at_layer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleDemand:
    """How many members of one role, and how big each is.

    ``weight`` orders the roles when they compete for the same cards: the
    hungriest goes first, because a role that needs whole cards cannot use what
    a role taking slices has left behind, while the reverse usually works.
    Classic first-fit-decreasing, and it is the difference between "4P4D fits"
    and "4P4D fits only if you happen to place P first".
    """

    role: str
    replicas: int
    weight: float = 0.0


@dataclass(frozen=True)
class GatherRequest:
    """Where the group must fit, and whether "must" is meant literally."""

    layer: Optional[str] = None
    must: bool = False


@dataclass
class GroupPlacement:
    """Which worker each member goes to, and where the group ended up."""

    layer: str
    domain: str
    # role -> [worker_id, ...], one entry per replica.
    assignments: Dict[str, List[int]] = field(default_factory=dict)

    def worker_ids(self) -> List[int]:
        return [wid for ids in self.assignments.values() for wid in ids]


@dataclass
class GroupInfeasible:
    """Why it did not fit, in terms the deployment form can show.

    Carries the best domain it found rather than only the shortfall, because
    "the largest rack is 2 cards short" is actionable and "it does not fit" is
    not.
    """

    reason: str
    layer: Optional[str] = None
    best_domain: Optional[str] = None
    needed: int = 0
    available: int = 0
    # Workers whose capacity could not be measured at all. `available` counts
    # only what was actually established, so without this a cluster nobody
    # could measure and a cluster that is genuinely full produce the same
    # refusal — and "full" is the one answer that stops an operator looking
    # for a mistake. Measured on a live host: one missing global config, and
    # every worker reported zero.
    unmeasured: int = 0


# async capacity(role, worker_ids, already_placed) -> {worker_id: slots}
#
# A worker whose capacity could not be determined is left OUT of the mapping.
# Absent means unknown; present-and-zero means measured and full. Collapsing
# the two would make an unmeasurable cluster indistinguishable from a full
# one.
#
# Required to be decomposable per worker: the answer for a set of workers is
# the union of the answers for its members. The real implementation asks each
# worker separately anyway, and the property is what lets the domain-sizing
# pass below run once for the whole tree instead of once per domain per layer
# — measured at 12-18 calls for a two-worker group before, which is a full
# selector sweep each time.
#
# `already_placed` is what this solve has committed so far, in the shape the
# allocation accounting reads. Passing it back is what keeps the capacity of
# the second role honest about what the first role took.
#
# Async because the only real implementation is `count_offer_slots`, which
# drives the resource-fit selectors, which are async all the way down. The
# first draft typed this synchronous and every test passed — the mismatch
# surfaced only when wiring it to a live Ascend host, which is the argument
# for doing that early rather than at the end.
CapacityFn = Callable[[str, Sequence[int], Sequence[object]], Awaitable[Dict[int, int]]]


async def solve_group_placement(
    root: TopologyNode,
    roles: Sequence[RoleDemand],
    capacity: CapacityFn,
    layers: Sequence[str],
    gather: GatherRequest = GatherRequest(),
) -> object:
    """Place the group, or say why not.

    ``layers`` is root-to-leaf. The search runs leaf-to-root: the first layer
    with a domain that holds the whole group is the tightest one, and the
    tightest domain is the one whose members are closest together.
    """
    total = sum(r.replicas for r in roles)
    if total <= 0:
        return GroupPlacement(layer=NODE_LAYER, domain="", assignments={})

    ordered_roles = sorted(roles, key=lambda r: (-r.weight, r.role))
    # Decided once, and used for both the ceiling and the root fallback below.
    # Deriving them separately is how the first version came to log "ignoring
    # this requirement" and then refuse the deployment in its name: the ceiling
    # honoured the unknown layer by standing down, while the fallback still saw
    # `must` set and stayed switched off.
    enforced = _enforced_gather(layers, gather)
    ceiling = layers.index(enforced.layer) if enforced.layer else 0
    best: Optional[GroupInfeasible] = None

    # Every domain is sized by the hungriest role with nothing placed, which
    # makes it one question asked once for the whole tree rather than once per
    # domain per layer. The hungriest role is the honest yardstick: a domain
    # with room for eight slices and no whole card is not eight units of room
    # to a group whose first role needs whole cards.
    sizing = await capacity(ordered_roles[0].role, root.descendant_worker_ids(), [])

    # Leaf-to-root. Stopping at `ceiling` is the whole of MustGather: without
    # it the walk continues widening until the cluster root, which always fits
    # and is exactly the outcome the operator asked not to get.
    for index in range(len(layers) - 1, ceiling - 1, -1):
        layer = layers[index]
        domains = _gatherable_domains(root, layer)
        # Tightest fitting domain first, off the one sizing pass above.
        sized = []
        for d in domains:
            members = d.descendant_worker_ids()
            sized.append((sum(sizing.get(w, 0) for w in members), d.name, d))
        for _size, _name, domain in sorted(sized, key=lambda t: (t[0], t[1])):
            placement = await _fit_in_domain(domain, layer, ordered_roles, capacity)
            if isinstance(placement, GroupPlacement):
                return placement
            # `>=` so a tie is won by the later, wider layer. With `>` the
            # refusal for a rack-level requirement could name a single host,
            # since the leaf layer is examined first and its domains are just
            # as short of room — a message that contradicts itself.
            if best is None or placement.available >= best.available:
                best = placement

    # The cluster root, last. It is not in `layers` — `layer_names` returns the
    # declared layers plus the leaf — and leaving it out of the search entirely
    # would mean a group too big for any declared domain could never be placed
    # at all, `must` or not. It is a real fallback rather than a domain anyone
    # gathers into: everything is under it, so reaching here means only that
    # the members are somewhere in this cluster.
    if not enforced.must:
        placement = await _fit_in_domain(root, root.layer, ordered_roles, capacity)
        if isinstance(placement, GroupPlacement):
            return placement
        if best is None or placement.available > best.available:
            best = placement

    if best is None:
        return GroupInfeasible(
            reason="No topology domain has any capacity for this group.",
            needed=total,
        )
    if enforced.must and not best.unmeasured:
        best.reason = (
            f"The group needs {best.needed} placements in one "
            f"{enforced.layer!r}, and the roomiest one holds {best.available}."
        )
    elif enforced.must:
        # Deliberately not phrased as a capacity verdict: the number behind it
        # is a floor, not a measurement.
        best.reason = (
            f"The group needs {best.needed} placements in one "
            f"{enforced.layer!r}, but capacity could not be measured on "
            f"{best.unmeasured} worker(s), so whether it fits is unknown."
        )
    return best


def _enforced_gather(layers: Sequence[str], gather: GatherRequest) -> GatherRequest:
    """The requirement as it will actually be applied.

    A `must` naming a layer this cluster no longer declares is dropped
    *entirely* — not just from the ceiling. Half-dropping it is the bug this
    function exists to make impossible: the walk would stand down for the
    unknown layer while the root fallback stayed disabled, so a group that fits
    only at the cluster root would be refused in the name of a layer the code
    had just announced it was ignoring.

    A stale name means someone renamed or removed a layer somewhere else. That
    must not take a running deployment down.
    """
    if not gather.must or not gather.layer:
        return GatherRequest(layer=None, must=False)
    if gather.layer not in layers:
        logger.warning(
            "Ignoring a gather requirement on unknown topology layer %r; "
            "the group will be placed as if none had been asked for.",
            gather.layer,
        )
        return GatherRequest(layer=None, must=False)
    return gather


def _gatherable_domains(root: TopologyNode, layer: str) -> List[TopologyNode]:
    """Domains at ``layer`` that mean something to gather into.

    The unclassified bucket is excluded, and that is not a detail. It holds the
    workers whose position is *unknown*; gathering a group into it would be
    claiming they are together on the strength of them all being unlabelled.
    The same rule makes two unclassified workers report no common layer, and
    the two have to agree — otherwise the solver would gather onto a domain
    that the distance function says does not exist.
    """
    return [d for d in nodes_at_layer(root, layer) if not d.is_unclassified]


async def _fit_in_domain(
    domain: TopologyNode,
    layer: str,
    roles: Sequence[RoleDemand],
    capacity: CapacityFn,
) -> object:
    """Place every role inside one domain, or report how far it got."""
    worker_ids = domain.descendant_worker_ids()
    total = sum(r.replicas for r in roles)
    if not worker_ids:
        return GroupInfeasible(
            reason="empty domain", layer=layer, best_domain=domain.name, needed=total
        )

    placement = GroupPlacement(layer=layer, domain=domain.name)
    placed: List[object] = []
    placed_total = 0

    for role in roles:
        slots = await capacity(role.role, worker_ids, placed)
        share = _share_out(slots, role.replicas, placed)
        if share is None:
            unmeasured = len([w for w in worker_ids if w not in slots])
            return GroupInfeasible(
                reason=(
                    "not enough room"
                    if not unmeasured
                    else f"capacity could not be measured on {unmeasured} of "
                    f"{len(worker_ids)} workers, and what could be measured is "
                    "not enough"
                ),
                layer=layer,
                best_domain=domain.name,
                needed=total,
                available=placed_total + sum(slots.values()),
                unmeasured=unmeasured,
            )
        assigned: List[int] = []
        for worker_id, count in share:
            assigned.extend([worker_id] * count)
            placed.extend([_Committed(worker_id, role.role)] * count)
        placement.assignments[role.role] = assigned
        placed_total += role.replicas

    return placement


@dataclass
class _Committed:
    """A placement this solve has made but not written anywhere yet.

    The worker id is what the capacity function needs; the role is what the
    tie-break below needs. Nothing else — the capacity function re-derives the
    real resource claim itself, and inventing one here would be a second,
    quieter accounting of the same placement.
    """

    worker_id: int
    role: str = ""


def _share_out(
    slots: Dict[int, int],
    replicas: int,
    placed: Sequence[object] = (),
) -> Optional[List[Tuple[int, int]]]:
    """Hand ``replicas`` placements to the workers with the most room first.

    🔴 The direction here is the opposite of the one used to pick the domain,
    and both are right. Between domains the *smallest* one that fits wins, so
    the group leaves the least fragmentation behind for whoever comes next.
    Inside the chosen domain the *roomiest* worker fills first, so the group
    occupies as few workers as it can and its members stay as close as the
    domain allows. Swapping either is the single most likely way to port this
    wrongly, which is why both have a test that fails on exactly that swap.

    Among workers with equal room, the one already carrying fewer of this
    group's members wins. That tie-break is the cheap half of keeping the roles
    mixed: placing role by role, roomiest-first, otherwise packs all of one
    role onto the first workers and all of the next onto the rest — and with a
    router that pairs prefill and decode independently, an all-P/all-D split
    across two domains is the one arrangement where *no* pair is local.
    Homogeneous workers make this the common case, not a corner.
    """
    if replicas <= 0:
        return []
    if sum(slots.values()) < replicas:
        return None

    load: Dict[int, int] = {}
    for entry in placed:
        worker_id = getattr(entry, "worker_id", None)
        if worker_id is not None:
            load[worker_id] = load.get(worker_id, 0) + 1

    # Ties broken by worker id last, so a re-solve of an unchanged cluster
    # produces an unchanged plan; otherwise every reconcile would look like a
    # spec change to anything comparing placements.
    ranked = [
        worker_id
        for worker_id, _ in sorted(
            slots.items(), key=lambda kv: (-kv[1], load.get(kv[0], 0), kv[0])
        )
    ]

    # One at a time around the ranked workers, not "fill the first, then the
    # next". Greedy filling is what produces the all-P-here/all-D-there split:
    # the first role exhausts the roomiest workers, and the second has nowhere
    # left but the rest. Dealing round-robin costs nothing in locality — by the
    # time this runs, the group has already failed to fit on any single host,
    # so its members are spanning workers either way — and it is the difference
    # between every pair being remote and half of them being local.
    taken: Dict[int, int] = {}
    left = replicas
    while left > 0:
        progressed = False
        for worker_id in ranked:
            if left <= 0:
                break
            if taken.get(worker_id, 0) >= slots[worker_id]:
                continue
            taken[worker_id] = taken.get(worker_id, 0) + 1
            left -= 1
            progressed = True
        if not progressed:
            # Cannot happen: the total was checked above. Guarded anyway
            # because the alternative is an infinite loop.
            return None
    return [(worker_id, taken[worker_id]) for worker_id in ranked if worker_id in taken]


def rank_domains_for_binpack(
    domains: Sequence[TopologyNode], slots_by_worker: Dict[int, int]
) -> List[TopologyNode]:
    """Fitting domains, tightest first.

    Separate from the search so the ordering can be asserted on its own: the
    smallest domain that still fits is preferred, which leaves the larger ones
    whole for groups that will need them.
    """
    sized = [
        (sum(slots_by_worker.get(w, 0) for w in d.descendant_worker_ids()), d.name, d)
        for d in domains
    ]
    return [d for _size, _name, d in sorted(sized, key=lambda t: (t[0], t[1]))]
