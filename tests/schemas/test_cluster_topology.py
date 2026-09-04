import pytest

from gpustack.schemas.clusters import (
    ClusterTopology,
    ClusterUpdate,
    GatherStrategyEnum,
)

RACK = "topology.gpustack.ai/rack"
ZONE = "topology.kubernetes.io/zone"


def cluster(topology: dict):
    return ClusterUpdate(name="c1", topology=ClusterTopology.model_validate(topology))


def test_a_cluster_may_declare_no_topology_at_all():
    """The field is optional and NULL is not a degraded state: a cluster with
    no layers still schedules and still offers the tightest gather choice,
    because the leaf layer is built in and takes the worker's name."""
    assert ClusterUpdate(name="c1").topology is None


def test_layers_round_trip_through_their_camel_case_aliases():
    """The wire form is camelCase like the neighbouring k8s_options."""
    c = cluster(
        {
            "layers": [
                {"name": "Zone", "labelKeys": [ZONE]},
                {"name": "Rack", "labelKeys": [RACK], "parentLayer": "Zone"},
            ],
            "defaultGatherStrategy": "MustGather",
            "defaultGatherLayer": "Rack",
        }
    )

    assert [layer.name for layer in c.topology.layers] == ["Zone", "Rack"]
    assert c.topology.layers[1].parent_layer == "Zone"
    assert c.topology.layers[1].label_keys == [RACK]
    assert c.topology.default_gather_strategy is GatherStrategyEnum.MUST_GATHER


def test_declaration_order_does_not_have_to_be_root_first():
    """The chain is derived, which is the whole reason it is stored as a chain
    rather than an ordered list."""
    c = cluster(
        {
            "layers": [
                {"name": "Rack", "parentLayer": "Zone"},
                {"name": "Zone"},
            ]
        }
    )

    assert {layer.name for layer in c.topology.layers} == {"Rack", "Zone"}


def test_a_layer_may_declare_no_label_keys():
    """Legal, and it means every worker is unclassified at that layer — a loss
    of resolution, which is what this whole structure is allowed to lose."""
    c = cluster({"layers": [{"name": "Rack"}]})

    assert c.topology.layers[0].label_keys == []


# --- the declaration is validated; the data never is ----------------------- #


@pytest.mark.parametrize(
    "topology, expected",
    [
        (
            {"layers": [{"name": "A", "parentLayer": "nope"}]},
            "unknown parent",
        ),
        (
            {"layers": [{"name": "A"}, {"name": "B"}]},
            "share a parent",
        ),
        (
            {
                "layers": [
                    {"name": "A", "parentLayer": "B"},
                    {"name": "B", "parentLayer": "A"},
                ]
            },
            "not reachable",
        ),
        (
            {
                "layers": [
                    {"name": "Z"},
                    {"name": "A", "parentLayer": "Z"},
                    {"name": "B", "parentLayer": "Z"},
                ]
            },
            "share a parent",
        ),
        (
            {"layers": [{"name": "accelerator_domain"}]},
            "reserved",
        ),
        (
            {"layers": [{"name": "A"}, {"name": "A", "parentLayer": "A"}]},
            "Duplicate",
        ),
    ],
)
def test_an_unbuildable_declaration_is_refused_at_save_time(topology, expected):
    """Refused because the operator's intent is unknowable, not because the
    fleet is mislabelled — see the next test for the difference."""
    with pytest.raises(ValueError, match=expected):
        cluster(topology)


def test_saving_does_not_require_any_worker_to_be_labelled_yet():
    """The layers are declared *before* the labels exist: the tree is how an
    operator sees who is still missing one. Validating the data here would make
    labelling a precondition for saving, which is backwards."""
    c = cluster({"layers": [{"name": "Rack", "labelKeys": ["nobody.has/this"]}]})

    assert c.topology.layers[0].label_keys == ["nobody.has/this"]


# --- the inherited gather default ------------------------------------------ #


def test_the_builtin_node_layer_is_selectable_without_declaring_anything():
    """The tightest choice must not depend on configuration; it is the one an
    operator with no RDMA needs, and the leaf layer always exists."""
    c = cluster(
        {
            "defaultGatherStrategy": "MustGather",
            "defaultGatherLayer": "NodeTopologyLayer",
        }
    )

    assert c.topology.default_gather_layer == "NodeTopologyLayer"


def test_a_gather_layer_that_names_nothing_is_refused():
    with pytest.raises(ValueError, match="is not a known layer"):
        cluster({"layers": [{"name": "Rack"}], "defaultGatherLayer": "Zone"})


def test_a_strategy_without_a_layer_is_refused():
    """ "Must gather" has to say must gather *where*."""
    with pytest.raises(ValueError, match="needs default_gather_layer"):
        cluster({"layers": [{"name": "Rack"}], "defaultGatherStrategy": "MustGather"})


def test_a_custom_layer_may_hang_under_a_vocabulary_field():
    """The vocabulary is the chain; a custom layer names the rung it sits
    under, which is how a fabric with a tier the vocabulary lacks is spelled."""
    c = cluster(
        {
            "layers": [{"name": "Pod", "parentLayer": "zone", "labelKeys": ["dc/pod"]}],
            "defaultGatherLayer": "Pod",
        }
    )

    assert c.topology.layers[0].parent_layer == "zone"


def test_the_accelerator_domain_is_a_selectable_gather_layer():
    """It is not a rung of the tree, but it is a scope the solver walks, so a
    cluster may make "same domain" its default."""
    c = cluster(
        {
            "defaultGatherStrategy": "MustGather",
            "defaultGatherLayer": "accelerator_domain",
        }
    )

    assert c.topology.default_gather_layer == "accelerator_domain"


def test_vocabulary_keys_can_be_overridden_by_naming_the_field():
    c = cluster(
        {
            "layers": [{"name": "rack", "labelKeys": ["dc.example.com/rack"]}],
            "acceleratorDomain": {"labelKeys": ["ds.coreweave.com/nvlink.domain"]},
        }
    )

    assert c.topology.layers[0].label_keys == ["dc.example.com/rack"]
    assert c.topology.accelerator_domain.label_keys == [
        "ds.coreweave.com/nvlink.domain"
    ]


def test_a_layer_without_a_strategy_is_allowed():
    """Naming the layer alone is inert, and inert is the right default: it
    records the operator's intended granularity without forcing a policy."""
    c = cluster({"layers": [{"name": "Rack"}], "defaultGatherLayer": "Rack"})

    assert c.topology.default_gather_strategy is None
