"""The vocabulary: fill a value in and a layer exists; declare nothing.

What is pinned here is the contract the table, the tree and the solver all
rely on — the owned key is always tried first, a field is active only once
someone filled it in, custom layers slot in by their parent, and the same
declarations are refused everywhere for the same reasons.

🔴 And the property this file exists for after the redesign: **there is one
chain, and the accelerator domain is an ordinary rung of it.** Nothing here
returns a second chain, a chain marker, or a scope the tree did not produce.
The keys a domain is published under survive as candidate keys, so an operator
who adds a layer pointing at one gets values immediately — which is the whole
of what the second chain used to buy.
"""

from types import SimpleNamespace

import pytest

from gpustack.scheduler.topology import (
    NODE_LAYER,
    TopologyError,
    UNCLASSIFIED,
)
from gpustack.scheduler.topology_view import build_view
from gpustack.scheduler.topology_vocabulary import (
    KNOWN_KEYS,
    RESERVED_IDS,
    VOCABULARY_IDS,
    gather_layer_names,
    primary_key_for,
    resolve,
    validate_declaration,
)

RACK = "topology.gpustack.ai/rack"
K8S_RACK = "topology.kubernetes.io/rack"
ROOM = "topology.gpustack.ai/room"
ROW = "topology.gpustack.ai/row"
CLIQUE = "nvidia.com/gpu.clique"
DOMAIN = "topology.gpustack.ai/accelerator-domain"
SWITCH = "topology.gpustack.ai/switch"

# The rung an operator declares for the accelerator domain. It is a custom
# layer like any other now — the name is not reserved, and nothing in the code
# knows it is special.
DOMAIN_LAYER = "accelerator_domain"


def worker(id_, name, labels=None, facts=None):
    return SimpleNamespace(
        id=id_,
        name=name,
        labels=labels or {},
        status=SimpleNamespace(topology_facts=facts),
    )


def layer(name, keys=(), parent=None):
    return SimpleNamespace(name=name, label_keys=list(keys), parent_layer=parent)


def topology(layers=()):
    return SimpleNamespace(layers=list(layers))


def domain_layer(parent="rack"):
    """The declaration an operator writes to place the domain on the chain."""
    return layer(DOMAIN_LAYER, [DOMAIN, CLIQUE], parent=parent)


# --- resolve ---------------------------------------------------------------- #


def test_no_declaration_is_the_vocabulary_as_is():
    resolved = resolve(None)
    assert [x.id for x in resolved.layers] == list(VOCABULARY_IDS)
    assert [x.id for x in resolved.layers] == ["room", "row", "rack"]
    assert resolved.layer("rack").label_keys == (RACK, K8S_RACK)


def test_naming_a_vocabulary_field_overrides_its_keys_but_keeps_the_owned_key_first():
    """The owned key is what the table writes; if it were not tried first a
    hand-filled value could lose to a discovered one."""
    resolved = resolve(topology([layer("rack", ["dc/rack"])]))
    assert resolved.layer("rack").label_keys == (RACK, "dc/rack")

    resolved = resolve(topology([layer("rack", ["dc/rack", RACK])]))
    assert resolved.layer("rack").label_keys == (RACK, "dc/rack")


def test_a_custom_layer_slots_in_below_its_parent():
    resolved = resolve(topology([layer("Pod", ["dc/pod"], parent="row")]))
    ids = [x.id for x in resolved.layers]
    assert ids.index("Pod") == ids.index("row") + 1
    assert resolved.layer("Pod").builtin is False


def test_the_accelerator_domain_is_declared_as_an_ordinary_custom_layer():
    """🔴 The redesign, stated once. The domain used to be the built-in rung of
    a second chain; it is now a layer the operator inserts where their hardware
    puts it, reading the keys the runtime already writes."""
    resolved = resolve(topology([domain_layer(parent="row")]))
    ids = [x.id for x in resolved.layers]
    assert ids == ["room", "row", DOMAIN_LAYER, "rack"]
    assert resolved.layer(DOMAIN_LAYER).builtin is False
    assert resolved.layer(DOMAIN_LAYER).label_keys == (DOMAIN, CLIQUE)


def test_tiers_inside_a_domain_chain_under_it_with_no_schema_change():
    """The Atlas 950 case — three bandwidth tiers inside one domain (blade 1008,
    cabinet 896, across cabinets 448 GB/s). Each is a layer; none is a field."""
    resolved = resolve(
        topology(
            [
                domain_layer(parent="row"),
                layer("cabinet", ["hw/cabinet"], parent=DOMAIN_LAYER),
                layer("blade", ["hw/blade"], parent="cabinet"),
            ]
        )
    )
    ids = [x.id for x in resolved.layers]
    assert ids == ["room", "row", DOMAIN_LAYER, "cabinet", "blade", "rack"]


def test_a_parentless_custom_layer_sits_above_the_vocabulary():
    resolved = resolve(topology([layer("Campus", ["dc/campus"])]))
    assert [x.id for x in resolved.layers][:2] == ["Campus", "room"]


def test_custom_layers_chain_under_each_other():
    resolved = resolve(
        topology([layer("B", parent="A"), layer("A", parent="rack")]),
    )
    ids = [x.id for x in resolved.layers]
    assert ids[ids.index("rack") + 1 :][:2] == ["A", "B"]


def test_a_name_is_looked_up_once_because_there_is_one_chain():
    resolved = validate_declaration(
        topology([layer("cage", ["dc/cage"], parent="rack")])
    )
    assert resolved.layer("cage") is not None
    assert resolved.layer("nonsense") is None


# --- active ----------------------------------------------------------------- #


def test_a_field_is_active_only_once_a_worker_has_a_value():
    resolved = resolve(None)
    assert resolved.active([worker(1, "w1")]) == []
    assert [x.id for x in resolved.active([worker(1, "w1", {RACK: "R1"})])] == ["rack"]
    assert [x.id for x in resolved.active([worker(1, "w1", {K8S_RACK: "R1"})])] == [
        "rack"
    ]


def test_a_discovered_fact_activates_a_declared_domain_layer_too():
    """The worker keeps writing the domain as a label (§2 of the spec keeps the
    whole of `topology_facts`), so declaring the rung is all an operator does."""
    resolved = resolve(topology([domain_layer()]))
    active = resolved.active([worker(1, "w1", facts={CLIQUE: "u.1"})])
    assert [x.id for x in active] == [DOMAIN_LAYER]


def test_a_custom_layer_stays_visible_when_nobody_matches_it():
    """An operator who wrote it down wants to see that nobody matches."""
    resolved = resolve(topology([layer("Pod", ["dc/pod"], parent="row")]))
    assert [x.id for x in resolved.active([worker(1, "w1")])] == ["Pod"]


def test_active_layers_chain_in_vocabulary_order_whatever_is_skipped():
    resolved = resolve(None)
    active = resolved.active([worker(1, "w1", {ROOM: "hall-1", RACK: "r"})])
    specs = resolved.specs(active)
    assert [(s.layer, s.parent_layer) for s in specs] == [
        ("room", None),
        ("rack", "room"),
    ]


# --- primary keys ----------------------------------------------------------- #


def test_primary_keys_are_found_by_name_alone():
    resolved = resolve(
        topology(
            [
                layer("Pod", ["dc/pod", "other/pod"], parent="row"),
                domain_layer(),
            ]
        )
    )
    assert primary_key_for(resolved, "rack") == RACK
    assert primary_key_for(resolved, "room") == ROOM
    assert primary_key_for(resolved, "row") == ROW
    assert primary_key_for(resolved, DOMAIN_LAYER) == DOMAIN
    assert primary_key_for(resolved, "Pod") == "dc/pod"
    assert primary_key_for(resolved, NODE_LAYER) is None
    assert primary_key_for(resolved, "nonsense") is None


# --- validation: the same refusals everywhere ------------------------------- #


@pytest.mark.parametrize(
    "layers, message",
    [
        ([layer("A", parent="nope")], "unknown parent"),
        ([layer("A"), layer("B")], "share a parent"),
        ([layer("A", parent="B"), layer("B", parent="A")], "not reachable"),
        ([layer(NODE_LAYER)], "reserved"),
        ([layer("ClusterTopologyLayer")], "reserved"),
        ([layer("A"), layer("A")], "Duplicate"),
    ],
)
def test_unbuildable_declarations_are_refused(layers, message):
    with pytest.raises(TopologyError, match=message):
        validate_declaration(topology(layers))


def test_only_the_root_and_the_leaf_are_reserved_names():
    """🔴 The set shrank with the second chain. `rack` was never reserved
    against its own chain — naming it is how an operator overrides its keys —
    and `accelerator_domain` was only reserved because it was the *other*
    chain's built-in rung. There is no other chain."""
    assert RESERVED_IDS == frozenset({"ClusterTopologyLayer", NODE_LAYER})
    assert DOMAIN_LAYER not in RESERVED_IDS
    assert "room" not in RESERVED_IDS
    assert "row" not in RESERVED_IDS


def test_a_custom_layer_may_be_called_accelerator_domain():
    """Which is the recommended spelling, now that it is just a name."""
    resolved = validate_declaration(topology([domain_layer()]))
    assert resolved.layer(DOMAIN_LAYER) is not None


def test_a_parent_on_a_vocabulary_entry_is_ignored():
    """The UI serialises the whole chain uniformly; a builtin's place is fixed
    regardless of what it says its parent is."""
    resolved = validate_declaration(
        topology([layer("rack", ["dc/rack"], parent="room")])
    )
    assert [x.id for x in resolved.layers] == list(VOCABULARY_IDS)
    assert resolved.layer("rack").label_keys == (RACK, "dc/rack")


def test_a_valid_declaration_is_returned_resolved():
    resolved = validate_declaration(topology([layer("Pod", ["dc/pod"], parent="row")]))
    assert resolved.layer("Pod") is not None


def test_gather_layer_names_are_the_host_then_the_chain():
    names = gather_layer_names(resolve(None))
    assert names == [NODE_LAYER, "room", "row", "rack"]
    assert len(names) == len(set(names))


def test_an_undeclared_domain_is_not_a_gather_target():
    """🔴 The contract change the deployment form has to see: `gather.layer`
    has no `accelerator_domain` special value any more. A cluster that did not
    declare the rung does not offer it."""
    assert DOMAIN_LAYER not in gather_layer_names(resolve(None))
    assert DOMAIN_LAYER in gather_layer_names(resolve(topology([domain_layer()])))


# --- the candidate keys the second chain left behind ------------------------ #


def test_the_domain_and_switch_keys_are_offered_as_candidates():
    """§2: the built-in layers shrink to three, but the keys a fleet already
    publishes stay in the suggestion list so adding the layer yields values at
    once rather than after a relabelling campaign."""
    keys = {k.key for k in KNOWN_KEYS}
    assert {
        DOMAIN,
        CLIQUE,
        "accelerator.topograph.run/domain",
        "network.topology.nvidia.com/accelerator",
        SWITCH,
        "fabric.topograph.run/tier-0",
    } <= keys


def test_every_candidate_key_fits_a_rung_that_exists():
    """`fits` is where the Advanced panel offers to insert the layer, so it can
    only name the three built-ins that survive."""
    for known in KNOWN_KEYS:
        assert known.fits, known.key
        assert set(known.fits) <= set(VOCABULARY_IDS), known.key


# --- the domain is an ordinary tree rung ------------------------------------ #


def test_a_declared_domain_is_a_tree_rung_with_an_unclassified_bucket():
    workers = [
        worker(1, "w1", facts={CLIQUE: "u.1"}),
        worker(2, "w2", facts={CLIQUE: "u.1"}),
        worker(3, "w3", facts={CLIQUE: "u.2"}),
        worker(4, "w4"),
    ]
    view = build_view(topology([domain_layer()]), workers)
    groups = view.nodes(DOMAIN_LAYER)

    by_name = {g.name: sorted(g.descendant_worker_ids()) for g in groups}
    assert by_name == {"u.1": [1, 2], "u.2": [3], UNCLASSIFIED: [4]}


def test_a_hand_filled_domain_wins_over_the_discovered_one():
    view = build_view(
        topology([domain_layer()]),
        [worker(1, "w1", {DOMAIN: "hccs-b"}, facts={CLIQUE: "u.1"})],
    )
    assert [g.name for g in view.nodes(DOMAIN_LAYER)] == ["hccs-b"]


def test_a_tier_inside_the_domain_nests_under_it_not_beside_it():
    """The pair, not the cabinet alone: cabinet R1 in super pod 3 and cabinet
    R1 in super pod 4 are not the same place — and with a chain that falls out
    of the tree rather than out of a string concatenation."""
    workers = [
        worker(1, "w1", {"hw/cabinet": "R1"}, facts={DOMAIN: "spod-3"}),
        worker(2, "w2", {"hw/cabinet": "R1"}, facts={DOMAIN: "spod-3"}),
        worker(3, "w3", {"hw/cabinet": "R1"}, facts={DOMAIN: "spod-4"}),
        worker(4, "w4", facts={DOMAIN: "spod-3"}),  # no cabinet: unclassified there
    ]
    view = build_view(
        topology(
            [
                domain_layer(),
                layer("cabinet", ["hw/cabinet"], parent=DOMAIN_LAYER),
            ]
        ),
        workers,
    )

    cabinets = view.nodes("cabinet")
    named = {
        (c.parent.name, c.name): sorted(c.descendant_worker_ids())
        for c in cabinets
        if not c.is_unclassified
    }
    assert named == {("spod-3", "R1"): [1, 2], ("spod-4", "R1"): [3]}
    assert view.unclassified_at("cabinet") == [4]


# --- the view's scopes: one list, and it comes out of the tree --------------- #


def test_the_search_runs_host_first_then_the_declared_rungs_outward():
    workers = [
        worker(1, "w1", {RACK: "R1", ROOM: "hall-1"}, facts={CLIQUE: "u.1"}),
        worker(2, "w2", {RACK: "R1", ROOM: "hall-1"}, facts={CLIQUE: "u.1"}),
    ]
    view = build_view(topology([domain_layer()]), workers)

    assert [s.name for s in view.scopes()] == [
        NODE_LAYER,
        DOMAIN_LAYER,
        "rack",
        "room",
    ]


def test_the_domain_is_a_scope_like_any_other():
    """🔴 The candidate set the redesign removes: the domain used to arrive as
    a scope list of its own, never ranked against the tree's. Here it is a rung
    between the host and the rack, in one order the solver widens along."""
    workers = [
        worker(1, "w1", {RACK: "R1"}, facts={CLIQUE: "u.1"}),
        worker(2, "w2", {RACK: "R1"}, facts={CLIQUE: "u.1"}),
    ]
    view = build_view(topology([domain_layer()]), workers)

    names = [s.name for s in view.scopes()]
    assert names.index(DOMAIN_LAYER) < names.index("rack")
    assert view.tiers() == names


def test_a_field_nobody_filled_is_not_a_scope():
    view = build_view(None, [worker(1, "w1"), worker(2, "w2")])
    assert [s.name for s in view.scopes()] == [NODE_LAYER]
    assert view.domain_count("rack") == 0


def test_a_custom_layer_nobody_matches_is_shown_but_not_offered():
    view = build_view(
        topology([layer("Pod", ["dc/pod"], parent="row")]), [worker(1, "w1")]
    )
    assert [x.id for x in view.active] == ["Pod"]
    assert "Pod" not in view.tiers()
