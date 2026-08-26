import pytest

from gpustack.scheduler.group_solver import (
    GatherRequest,
    GroupInfeasible,
    GroupPlacement,
    RoleDemand,
    solve_group_placement,
)
from gpustack.scheduler.topology import (
    NODE_LAYER,
    ROOT_LAYER,
    TopologyLayerSpec,
    build_topology,
    layer_names,
)
from types import SimpleNamespace

RACK = "topology.gpustack.ai/rack"


def worker(id_, name, rack=None):
    labels = {RACK: rack} if rack else {}
    return SimpleNamespace(id=id_, name=name, labels=labels)


def rack_layer():
    return [TopologyLayerSpec(layer="RackLayer", label_keys=[RACK])]


def tree(workers, specs=None):
    specs = rack_layer() if specs is None else specs
    return build_topology(specs, workers), layer_names(specs)


def flat_capacity(per_worker):
    """Every worker has the same room for every role, minus what is committed."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, per_worker - used.get(w, 0)) for w in worker_ids}

    return capacity


def pd(prefill=1, decode=1):
    return [
        RoleDemand(role="prefill", replicas=prefill, weight=2.0),
        RoleDemand(role="decode", replicas=decode, weight=1.0),
    ]


# --- the tightest layer wins ----------------------------------------------- #


@pytest.mark.asyncio
async def test_a_group_that_fits_one_host_is_placed_on_one_host():
    """Leaf-to-root: the tightest domain holding the whole group wins, and no
    wider layer is even considered."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(root, pd(), flat_capacity(4), layers)

    assert isinstance(got, GroupPlacement)
    assert got.layer == NODE_LAYER
    assert len(set(got.worker_ids())) == 1


@pytest.mark.asyncio
async def test_a_group_too_big_for_one_host_widens_to_the_rack():
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    assert isinstance(got, GroupPlacement)
    assert got.layer == "RackLayer"
    assert got.domain == "rack-a"
    assert sorted(got.worker_ids()) == [1, 1, 2, 2]


@pytest.mark.asyncio
async def test_it_widens_only_as_far_as_it_has_to():
    """Two racks, and a group that fits in one. Widening to the cluster root
    would also "work" and would be the wrong answer."""
    root, layers = tree(
        [
            worker(1, "w1", "rack-a"),
            worker(2, "w2", "rack-a"),
            worker(3, "w3", "rack-b"),
        ]
    )

    got = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    assert got.layer == "RackLayer"
    assert got.domain == "rack-a"


# --- the two opposite sort directions -------------------------------------- #


@pytest.mark.asyncio
async def test_between_domains_the_tightest_that_fits_wins():
    """Binpack across domains: the group takes the smaller rack and leaves the
    bigger one whole for whoever needs it next.

    Every worker holds one member, so the group cannot fit a single host and
    the rack layer is genuinely the one deciding — otherwise this asserts
    nothing about domain ordering."""
    workers = [
        worker(1, "s1", "small"),
        worker(2, "s2", "small"),
        worker(3, "b1", "big"),
        worker(4, "b2", "big"),
        worker(5, "b3", "big"),
        worker(6, "b4", "big"),
    ]
    root, layers = tree(workers)

    got = await solve_group_placement(root, pd(1, 1), flat_capacity(1), layers)

    assert got.layer == "RackLayer"
    assert got.domain == "small"


@pytest.mark.asyncio
async def test_a_group_that_fits_the_roomiest_host_goes_there_whole():
    """The leaf layer is searched first, so a group small enough for one host
    never reaches the rack-level distribution at all — compactness comes from
    the layer walk, not from how a domain's workers are filled."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        room = {1: 1, 2: 4}
        return {w: max(0, room[w] - used.get(w, 0)) for w in worker_ids}

    root, layers = tree([worker(1, "small", "rack-a"), worker(2, "roomy", "rack-a")])

    got = await solve_group_placement(root, pd(2, 1), capacity, layers)

    assert got.layer == NODE_LAYER
    assert set(got.worker_ids()) == {2}


@pytest.mark.asyncio
async def test_inside_a_domain_the_roomiest_worker_gets_the_larger_share():
    """Round-robin, but starting from the roomiest — so when the members do not
    divide evenly, the spare goes where there is most room.

    Five members over rooms of 3 and 4 — bigger than either host, so the rack
    layer really is deciding, and with slack left over so the order of the
    round decides who carries the spare. Roomiest-first gives 3/2; starting
    from the tightest gives 2/3, which is the mutation this has to catch."""

    async def capacity(_role, worker_ids, placed):
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        room = {1: 3, 2: 4}
        return {w: max(0, room[w] - used.get(w, 0)) for w in worker_ids}

    root, layers = tree([worker(1, "small", "rack-a"), worker(2, "roomy", "rack-a")])

    got = await solve_group_placement(root, pd(3, 2), capacity, layers)

    assert got.layer == "RackLayer"
    assert sum(1 for w in got.worker_ids() if w == 2) == 3
    assert sum(1 for w in got.worker_ids() if w == 1) == 2


# --- gather ---------------------------------------------------------------- #


ZONE = "topology.kubernetes.io/zone"


def zone_rack_layers():
    return [
        TopologyLayerSpec(layer="ZoneLayer", label_keys=[ZONE]),
        TopologyLayerSpec(
            layer="RackLayer", label_keys=[RACK], parent_layer="ZoneLayer"
        ),
    ]


def zoned(id_, name, zone, rack):
    return SimpleNamespace(id=id_, name=name, labels={ZONE: zone, RACK: rack})


@pytest.mark.asyncio
async def test_must_gather_stops_the_walk_at_the_named_layer():
    """The whole of MustGather. Three layers on purpose: the group fits the
    zone but no rack, so a solver that ignored the ceiling would widen one step
    and succeed — which is exactly the outcome the operator asked not to get.
    A two-layer fixture cannot tell the two apart."""
    workers = [
        zoned(1, "w1", "z1", "rack-a"),
        zoned(2, "w2", "z1", "rack-a"),
        zoned(3, "w3", "z1", "rack-b"),
        zoned(4, "w4", "z1", "rack-b"),
    ]
    root, layers = tree(workers, zone_rack_layers())
    assert layers == ["ZoneLayer", "RackLayer", NODE_LAYER]

    loose = await solve_group_placement(root, pd(3, 3), flat_capacity(2), layers)
    assert isinstance(loose, GroupPlacement)
    assert loose.layer == "ZoneLayer"

    strict = await solve_group_placement(
        root,
        pd(3, 3),
        flat_capacity(2),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(strict, GroupInfeasible)
    assert "RackLayer" in strict.reason and "6" in strict.reason


@pytest.mark.asyncio
async def test_without_must_gather_the_same_group_is_placed_across_racks():
    root, layers = tree(
        [
            worker(1, "w1", "rack-a"),
            worker(2, "w2", "rack-a"),
            worker(3, "w3", "rack-b"),
            worker(4, "w4", "rack-b"),
        ]
    )

    got = await solve_group_placement(root, pd(3, 3), flat_capacity(2), layers)

    assert isinstance(got, GroupPlacement)
    assert len(got.worker_ids()) == 6


@pytest.mark.asyncio
async def test_the_refusal_says_how_much_room_the_roomiest_domain_had():
    """ "It does not fit" is not actionable; "the roomiest rack holds 4" is."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(4, 4),
        flat_capacity(2),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.needed == 8
    assert got.available == 4
    assert got.best_domain == "rack-a"


@pytest.mark.asyncio
async def test_a_gather_layer_that_no_longer_exists_does_not_block_scheduling():
    """A layer renamed or removed after the model was saved. Refusing here
    would take a running deployment down for an edit made somewhere else."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        flat_capacity(4),
        layers,
        GatherRequest(layer="LayerThatWasDeleted", must=True),
    )

    assert isinstance(got, GroupPlacement)


@pytest.mark.asyncio
async def test_an_unknown_gather_layer_does_not_disable_the_root_fallback():
    """Found on a live cluster, not here — the version above passes either way,
    because its group fits at the rack layer and never reaches the fallback.

    The workers sit in different zones *and* different racks, so the cluster
    root is the only domain holding both. Dropping the unknown requirement from
    the ceiling but not from the fallback leaves the group refused in the name
    of a layer the code has just logged that it is ignoring."""
    workers = [zoned(1, "w1", "z1", "rack-a"), zoned(2, "w2", "z2", "rack-b")]
    root, layers = tree(workers, zone_rack_layers())

    loose = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)
    assert isinstance(loose, GroupPlacement)
    assert loose.layer == ROOT_LAYER, "the fixture must force the root fallback"

    got = await solve_group_placement(
        root,
        pd(2, 2),
        flat_capacity(2),
        layers,
        GatherRequest(layer="SuperPodLayer", must=True),
    )

    assert isinstance(got, GroupPlacement)
    assert got.layer == ROOT_LAYER


# --- the unclassified bucket ----------------------------------------------- #


@pytest.mark.asyncio
async def test_unlabelled_workers_are_not_a_domain_to_gather_into():
    """They are the workers whose position is *unknown*. Gathering into that
    bucket would claim they are together on the strength of them all being
    unlabelled — and the distance function already says they are not."""
    root, layers = tree([worker(1, "w1"), worker(2, "w2")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        flat_capacity(1),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)


@pytest.mark.asyncio
async def test_unlabelled_workers_are_still_schedulable_without_a_requirement():
    """Not gatherable is not unusable: the leaf layer is per-worker and always
    real, so a group that fits on one host still lands."""
    root, layers = tree([worker(1, "w1"), worker(2, "w2")])

    got = await solve_group_placement(root, pd(1, 1), flat_capacity(4), layers)

    assert isinstance(got, GroupPlacement)
    assert got.layer == NODE_LAYER


# --- role order and role mixing -------------------------------------------- #


@pytest.mark.asyncio
async def test_the_hungriest_role_is_placed_first():
    """First-fit-decreasing. A role needing whole cards cannot use what a role
    taking slices left behind; the reverse usually works."""
    seen = []

    async def capacity(role, worker_ids, placed):
        seen.append(role)
        used = {}
        for entry in placed:
            used[entry.worker_id] = used.get(entry.worker_id, 0) + 1
        return {w: max(0, 4 - used.get(w, 0)) for w in worker_ids}

    root, layers = tree([worker(1, "w1", "rack-a")])
    await solve_group_placement(
        root,
        [
            RoleDemand(role="light", replicas=1, weight=1.0),
            RoleDemand(role="heavy", replicas=1, weight=9.0),
        ],
        capacity,
        layers,
    )

    assert seen[0] == "heavy"


@pytest.mark.asyncio
async def test_equal_room_prefers_the_worker_carrying_fewer_of_the_group():
    """Placing role by role, roomiest-first, otherwise packs all of one role
    onto the first workers and all of the next onto the rest. With a router
    that pairs prefill and decode independently, an all-P/all-D split is the
    one arrangement where no pair is local."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    got = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    # Each worker ends up with one prefill and one decode, not two of one kind.
    assert sorted(got.assignments["prefill"]) == [1, 2]
    assert sorted(got.assignments["decode"]) == [1, 2]


# --- degenerate inputs ----------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_empty_group_is_trivially_placed():
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(root, [], flat_capacity(4), layers)

    assert isinstance(got, GroupPlacement)
    assert got.worker_ids() == []


@pytest.mark.asyncio
async def test_a_cluster_with_no_capacity_at_all_says_so():
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(root, pd(1, 1), flat_capacity(0), layers)

    assert isinstance(got, GroupInfeasible)


@pytest.mark.asyncio
async def test_the_plan_is_stable_across_re_solves():
    """An unchanged cluster must produce an unchanged plan, or every reconcile
    looks like a spec change to anything comparing placements."""
    root, layers = tree([worker(i, f"w{i}", "rack-a") for i in range(1, 5)])

    first = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)
    second = await solve_group_placement(root, pd(2, 2), flat_capacity(2), layers)

    assert first.assignments == second.assignments


# --- what the refusal is allowed to claim ---------------------------------- #


@pytest.mark.asyncio
async def test_a_worker_that_could_not_be_measured_is_not_reported_as_full():
    """A plain misconfiguration once produced zero on every worker and a
    refusal that named capacity — the one answer that stops an operator looking
    for a mistake. Absent from the mapping means unknown; present-and-zero
    means measured and full."""
    root, layers = tree([worker(1, "w1", "rack-a"), worker(2, "w2", "rack-a")])

    async def nothing_measurable(_role, _worker_ids, _placed):
        return {}

    got = await solve_group_placement(
        root,
        pd(1, 1),
        nothing_measurable,
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.unmeasured == 2
    assert "could not be measured" in got.reason
    assert "holds 0" not in got.reason


@pytest.mark.asyncio
async def test_a_genuinely_full_cluster_still_says_so():
    """The other side of the same rule: measured and zero is a capacity
    verdict, and must keep reading like one."""
    root, layers = tree([worker(1, "w1", "rack-a")])

    got = await solve_group_placement(
        root,
        pd(1, 1),
        flat_capacity(0),
        layers,
        GatherRequest(layer="RackLayer", must=True),
    )

    assert isinstance(got, GroupInfeasible)
    assert got.unmeasured == 0
    assert "holds 0" in got.reason


@pytest.mark.asyncio
async def test_domains_are_sized_with_one_capacity_pass_not_one_each():
    """Sizing asks the same question of the same workers at every layer, and
    each ask is a full selector sweep in production. One pass for the tree."""
    # Counted in total, not by shape: sizing per domain would *add* calls
    # rather than change the one the tree-wide pass makes, so a predicate that
    # only recognises the tree-wide call cannot see the difference.
    calls = []

    async def counting(role, worker_ids, placed):
        calls.append((role, tuple(sorted(worker_ids)), len(placed)))
        return {w: 4 for w in worker_ids}

    workers = [
        zoned(1, "w1", "z1", "rack-a"),
        zoned(2, "w2", "z1", "rack-b"),
        zoned(3, "w3", "z2", "rack-c"),
    ]
    root, layers = tree(workers, zone_rack_layers())

    await solve_group_placement(root, pd(1, 1), counting, layers)

    # One tree-wide sizing pass, then one call per role placing into the
    # winning leaf. Sizing each of the three leaf domains separately would add
    # three more.
    assert calls[0] == ("prefill", (1, 2, 3), 0), "the sizing pass comes first"
    assert len(calls) == 3, f"expected 1 sizing + 2 placements, got {calls}"
