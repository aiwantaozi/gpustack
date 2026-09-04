"""The vocabulary: fill a value in and a layer exists; declare nothing.

What is pinned here is the contract the table, the tree and the solver all
rely on — the owned key is always tried first, a field is active only once
someone filled it in, custom layers slot in by their parent, and the same
declarations are refused everywhere for the same reasons.
"""

from types import SimpleNamespace

import pytest

from gpustack.scheduler.topology import (
    ACCELERATOR_DOMAIN_LAYER,
    NODE_LAYER,
    TopologyError,
    UNCLASSIFIED,
    group_by_domain,
)
from gpustack.scheduler.topology_view import build_view
from gpustack.scheduler.topology_vocabulary import (
    ACCELERATOR_DOMAIN,
    ACCELERATOR_SUB_DOMAIN,
    VOCABULARY_IDS,
    primary_key_for,
    resolve,
    validate_declaration,
)

RACK = "topology.gpustack.ai/rack"
K8S_RACK = "topology.kubernetes.io/rack"
CLIQUE = "nvidia.com/gpu.clique"
DOMAIN = "topology.gpustack.ai/accelerator-domain"


def worker(id_, name, labels=None, facts=None):
    return SimpleNamespace(
        id=id_,
        name=name,
        labels=labels or {},
        status=SimpleNamespace(topology_facts=facts),
    )


def layer(name, keys=(), parent=None):
    return SimpleNamespace(name=name, label_keys=list(keys), parent_layer=parent)


def topology(layers=(), domain=None):
    return SimpleNamespace(layers=list(layers), accelerator_domain=domain)


# --- resolve ---------------------------------------------------------------- #


def test_no_declaration_is_the_vocabulary_as_is():
    resolved = resolve(None)
    assert [x.id for x in resolved.chain] == list(VOCABULARY_IDS)
    assert resolved.layer("rack").label_keys == (RACK, K8S_RACK)
    assert resolved.domain.label_keys == (DOMAIN, CLIQUE)


def test_naming_a_vocabulary_field_overrides_its_keys_but_keeps_the_owned_key_first():
    """The owned key is what the table writes; if it were not tried first a
    hand-filled value could lose to a discovered one."""
    resolved = resolve(topology([layer("rack", ["dc/rack"])]))
    assert resolved.layer("rack").label_keys == (RACK, "dc/rack")

    resolved = resolve(topology([layer("rack", ["dc/rack", RACK])]))
    assert resolved.layer("rack").label_keys == (RACK, "dc/rack")


def test_a_custom_layer_slots_in_below_its_parent():
    resolved = resolve(topology([layer("Pod", ["dc/pod"], parent="zone")]))
    ids = [x.id for x in resolved.chain]
    assert ids.index("Pod") == ids.index("zone") + 1
    assert resolved.layer("Pod").builtin is False


def test_a_parentless_custom_layer_sits_above_the_vocabulary():
    resolved = resolve(topology([layer("Campus", ["dc/campus"])]))
    assert [x.id for x in resolved.chain][:2] == ["Campus", "region"]


def test_custom_layers_chain_under_each_other():
    resolved = resolve(
        topology([layer("B", parent="A"), layer("A", parent="rack")]),
    )
    ids = [x.id for x in resolved.chain]
    assert ids[ids.index("rack") + 1 :][:2] == ["A", "B"]


# --- active ----------------------------------------------------------------- #


def test_a_field_is_active_only_once_a_worker_has_a_value():
    resolved = resolve(None)
    assert resolved.active([worker(1, "w1")]) == []
    assert [x.id for x in resolved.active([worker(1, "w1", {RACK: "R1"})])] == ["rack"]
    assert [x.id for x in resolved.active([worker(1, "w1", {K8S_RACK: "R1"})])] == [
        "rack"
    ]


def test_a_discovered_fact_activates_a_field_too():
    resolved = resolve(None)
    assert [
        x.id
        for x in resolved.active(
            [worker(1, "w1", facts={"topology.gpustack.ai/switch": "aa"})]
        )
    ] == ["switch"]


def test_a_custom_layer_stays_visible_when_nobody_matches_it():
    """An operator who wrote it down wants to see that nobody matches."""
    resolved = resolve(topology([layer("Pod", ["dc/pod"], parent="zone")]))
    assert [x.id for x in resolved.active([worker(1, "w1")])] == ["Pod"]


def test_active_layers_chain_in_vocabulary_order_whatever_is_skipped():
    resolved = resolve(None)
    active = resolved.active(
        [worker(1, "w1", {"topology.gpustack.ai/zone": "z", RACK: "r"})]
    )
    specs = resolved.specs(active)
    assert [(s.layer, s.parent_layer) for s in specs] == [
        ("zone", None),
        ("rack", "zone"),
    ]


# --- primary keys ----------------------------------------------------------- #


def test_primary_keys():
    resolved = resolve(topology([layer("Pod", ["dc/pod", "other/pod"], parent="zone")]))
    assert primary_key_for(resolved, "rack") == RACK
    assert primary_key_for(resolved, ACCELERATOR_DOMAIN) == DOMAIN
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
        ([layer(ACCELERATOR_DOMAIN)], "reserved"),
        ([layer("A"), layer("A")], "Duplicate"),
    ],
)
def test_unbuildable_declarations_are_refused(layers, message):
    with pytest.raises(TopologyError, match=message):
        validate_declaration(topology(layers))


def test_a_parent_on_a_vocabulary_entry_is_ignored():
    """The UI serialises the whole chain uniformly; a builtin's place is fixed
    regardless of what it says its parent is."""
    resolved = validate_declaration(
        topology([layer("rack", ["dc/rack"], parent="row")])
    )
    assert [x.id for x in resolved.chain] == list(VOCABULARY_IDS)
    assert resolved.layer("rack").label_keys == (RACK, "dc/rack")


def test_a_valid_declaration_is_returned_resolved():
    resolved = validate_declaration(topology([layer("Pod", ["dc/pod"], parent="zone")]))
    assert resolved.layer("Pod") is not None


# --- group_by_domain -------------------------------------------------------- #


def test_domains_group_flat_with_an_unclassified_bucket():
    workers = [
        worker(1, "w1", facts={CLIQUE: "u.1"}),
        worker(2, "w2", facts={CLIQUE: "u.1"}),
        worker(3, "w3", facts={CLIQUE: "u.2"}),
        worker(4, "w4"),
    ]
    groups = group_by_domain(workers, (DOMAIN, CLIQUE))

    by_name = {g.name: sorted(g.descendant_worker_ids()) for g in groups}
    assert by_name == {"u.1": [1, 2], "u.2": [3], UNCLASSIFIED: [4]}
    assert all(g.layer == ACCELERATOR_DOMAIN_LAYER for g in groups)


def test_a_hand_filled_domain_wins_over_the_discovered_one():
    workers = [worker(1, "w1", {DOMAIN: "hccs-b"}, facts={CLIQUE: "u.1"})]
    groups = group_by_domain(workers, (DOMAIN, CLIQUE))
    assert [g.name for g in groups] == ["hccs-b"]


def test_sub_domains_pair_the_domain_with_the_rack():
    """The pair, not the rack alone: R1 in super pod 3 and R1 in super pod 4
    are not the same place."""
    workers = [
        worker(1, "w1", {RACK: "R1"}, facts={DOMAIN: "spod-3"}),
        worker(2, "w2", {RACK: "R1"}, facts={DOMAIN: "spod-3"}),
        worker(3, "w3", {RACK: "R1"}, facts={DOMAIN: "spod-4"}),
        worker(4, "w4", facts={DOMAIN: "spod-3"}),  # no rack: not closer to anyone
    ]
    groups = group_by_domain(workers, (DOMAIN, CLIQUE), sub_domain_keys=(RACK,))

    by_name = {g.name: sorted(g.descendant_worker_ids()) for g in groups}
    assert by_name == {"spod-3/R1": [1, 2], "spod-4/R1": [3]}


# --- the view's scopes ------------------------------------------------------ #


def test_scopes_run_host_domain_then_the_tree_up_from_the_switch():
    workers = [
        worker(
            1,
            "w1",
            {RACK: "R1", "topology.gpustack.ai/zone": "z"},
            facts={CLIQUE: "u.1", "topology.gpustack.ai/switch": "s1"},
        ),
        worker(
            2,
            "w2",
            {RACK: "R1", "topology.gpustack.ai/zone": "z"},
            facts={CLIQUE: "u.1", "topology.gpustack.ai/switch": "s1"},
        ),
    ]
    view = build_view(None, workers)

    assert [s.name for s in view.scopes()] == [
        NODE_LAYER,
        ACCELERATOR_DOMAIN,
        "switch",
        "rack",
        "zone",
    ]
    assert view.tier_names() == [
        NODE_LAYER,
        ACCELERATOR_DOMAIN,
        "switch",
        "rack",
        "zone",
    ]


def test_the_sub_domain_scope_precedes_the_domain_but_is_not_a_tier():
    domain = SimpleNamespace(label_keys=[], sub_domain_keys=[RACK])
    workers = [
        worker(1, "w1", {RACK: "R1"}, facts={DOMAIN: "spod-3"}),
        worker(2, "w2", {RACK: "R2"}, facts={DOMAIN: "spod-3"}),
    ]
    view = build_view(topology(domain=domain), workers)

    assert [s.name for s in view.scopes()] == [
        NODE_LAYER,
        ACCELERATOR_SUB_DOMAIN,
        ACCELERATOR_DOMAIN,
        "rack",
    ]
    assert ACCELERATOR_SUB_DOMAIN not in view.tier_names()


def test_a_field_nobody_filled_is_not_a_scope():
    view = build_view(None, [worker(1, "w1"), worker(2, "w2")])
    assert [s.name for s in view.scopes()] == [NODE_LAYER]
    assert view.has_domains is False


def test_a_custom_layer_nobody_matches_is_shown_but_not_offered():
    view = build_view(
        topology([layer("Pod", ["dc/pod"], parent="zone")]), [worker(1, "w1")]
    )
    assert [x.id for x in view.active] == ["Pod"]
    assert "Pod" not in view.tier_names()
