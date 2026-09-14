import pytest

from gpustack.schemas.clusters import ClusterTopology, ClusterUpdate

RACK = "topology.gpustack.ai/rack"
ROOM = "topology.gpustack.ai/room"
DOMAIN = "topology.gpustack.ai/accelerator-domain"


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
                {"name": "Hall", "labelKeys": [ROOM]},
                {"name": "Rack", "labelKeys": [RACK], "parentLayer": "Hall"},
            ],
        }
    )

    assert [layer.name for layer in c.topology.layers] == ["Hall", "Rack"]
    assert c.topology.layers[1].parent_layer == "Hall"
    assert c.topology.layers[1].label_keys == [RACK]


def test_declaration_order_does_not_have_to_be_root_first():
    """The chain is derived, which is the whole reason it is stored as a chain
    rather than an ordered list."""
    c = cluster(
        {
            "layers": [
                {"name": "Rack", "parentLayer": "Hall"},
                {"name": "Hall"},
            ]
        }
    )

    assert {layer.name for layer in c.topology.layers} == {"Rack", "Hall"}


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
            {"layers": [{"name": "NodeTopologyLayer"}]},
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


# --- what the declaration is, and is not ----------------------------------- #


def test_the_cluster_carries_no_gather_default_any_more():
    """🔴 `defaultGatherStrategy` / `defaultGatherLayer` are gone.

    They were the cluster-level inheritance source: an operator who knew the
    fabric set the strict choice once, and every model that said nothing
    inherited it. What killed it is that the two ways of being wrong are not
    the same size. Without a default, a group that wanted `rack` and did not
    ask is placed looser than ideal — it runs, slower. With one, it inherits
    `MustGather` and the deployment is *refused*, for a floor the deploy form
    never showed and the deployer cannot see. A cluster-level failure policy
    is one person arming a rejection on another's behalf.

    The fabric knowledge still reaches the deployer: the form derives its
    tiers from `layers`. Only the silent override is gone.

    Old rows keep the keys, and they are read straight past — the model is
    `extra="ignore"`, the same non-migration used for `acceleratorLayers`.
    """
    c = cluster(
        {
            "layers": [{"name": "Rack", "labelKeys": [RACK]}],
            "defaultGatherStrategy": "MustGather",
            "defaultGatherLayer": "Rack",
        }
    )

    assert not hasattr(c.topology, "default_gather_strategy")
    assert not hasattr(c.topology, "default_gather_layer")
    assert set(c.topology.model_dump(by_alias=True)) == {"layers"}


def test_a_stale_default_naming_a_deleted_layer_no_longer_blocks_saving():
    """The trap this removal sprang. `defaultGatherLayer` was validated against
    the declared layers, so when a release dropped a vocabulary field — `zone`
    and `region` both went — every cluster whose stored default named one
    became unsaveable, on a field the operator was not editing and the UI never
    showed. With no field there is no validation and no trap."""
    c = cluster({"layers": [{"name": "Rack"}], "defaultGatherLayer": "zone"})

    assert [layer.name for layer in c.topology.layers] == ["Rack"]


def test_a_custom_layer_may_hang_under_a_vocabulary_field():
    """The vocabulary is the chain; a custom layer names the rung it sits
    under, which is how a fabric with a tier the vocabulary lacks is spelled."""
    c = cluster(
        {"layers": [{"name": "Pod", "parentLayer": "row", "labelKeys": ["dc/pod"]}]}
    )

    assert c.topology.layers[0].parent_layer == "row"


def test_the_accelerator_domain_is_a_layer_the_operator_declares():
    """🔴 The redesign. It used to be the built-in rung of a second chain, and
    every cluster could name it whether or not it had one. Now it is an
    ordinary custom layer: it exists on the chain once declared, and nowhere
    otherwise."""
    c = cluster(
        {
            "layers": [
                {
                    "name": "accelerator_domain",
                    "labelKeys": [DOMAIN, "nvidia.com/gpu.clique"],
                    "parentLayer": "row",
                }
            ]
        }
    )

    assert c.topology.layers[0].parent_layer == "row"


def test_a_tier_inside_the_domain_is_a_layer_too():
    c = cluster(
        {
            "layers": [
                {"name": "accelerator_domain", "labelKeys": [DOMAIN]},
                {
                    "name": "cabinet",
                    "labelKeys": ["hw/cabinet"],
                    "parentLayer": "accelerator_domain",
                },
            ]
        }
    )

    assert c.topology.layers[1].parent_layer == "accelerator_domain"


def test_vocabulary_keys_can_be_overridden_by_naming_the_field():
    c = cluster({"layers": [{"name": "rack", "labelKeys": ["dc.example.com/rack"]}]})

    assert c.topology.layers[0].label_keys == ["dc.example.com/rack"]


def test_the_chain_takes_as_many_tiers_as_the_hardware_has():
    """Atlas 950 has three bandwidth tiers inside one domain (blade 1008,
    cabinet 896, across cabinets 448 GB/s). Each is a layer, not a schema
    change — which is the property the second chain was built for and the one
    chain keeps."""
    c = cluster(
        {
            "layers": [
                {"name": "accelerator_domain", "labelKeys": [DOMAIN]},
                {
                    "name": "cabinet",
                    "labelKeys": ["hw/cabinet"],
                    "parentLayer": "accelerator_domain",
                },
                {"name": "blade", "labelKeys": ["hw/blade"], "parentLayer": "cabinet"},
            ]
        }
    )

    assert [layer.name for layer in c.topology.layers] == [
        "accelerator_domain",
        "cabinet",
        "blade",
    ]


def test_an_accelerator_layers_field_is_ignored_rather_than_migrated():
    """🔴 §3: no data migration, no compatibility fallback. `ClusterTopology`
    is `extra="ignore"`, so a cluster saved under the two-chain model loads
    with its second chain silently dropped — the failure mode chosen over a
    migration that would have had to guess where on the one chain each rung
    belonged."""
    c = cluster(
        {
            "layers": [{"name": "rack", "labelKeys": [RACK]}],
            "acceleratorLayers": [{"name": "cabinet", "labelKeys": ["hw/cabinet"]}],
        }
    )

    assert not hasattr(c.topology, "accelerator_layers")
    assert [layer.name for layer in c.topology.layers] == ["rack"]
