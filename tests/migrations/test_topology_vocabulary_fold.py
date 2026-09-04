"""The preset layers the old UI inserted fold into the vocabulary."""

import importlib.util
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[2]
    / "gpustack/migrations/versions/2026_09_04_1000-e8f9a0b1c2d3_topology_vocabulary.py"
)
_spec = importlib.util.spec_from_file_location("topology_vocabulary_migration", _PATH)
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


def test_preset_layers_fold_and_the_gather_layer_is_renamed():
    topology = {
        "layers": [
            {"name": "Zone", "labelKeys": ["topology.kubernetes.io/zone"]},
            {
                "name": "Rack",
                "labelKeys": ["topology.gpustack.ai/rack"],
                "parentLayer": "Zone",
            },
        ],
        "defaultGatherStrategy": "MustGather",
        "defaultGatherLayer": "Rack",
    }
    folded, renames = migration._fold(topology)

    assert renames == {"Zone": "zone", "Rack": "rack"}
    assert folded["layers"] == []
    assert folded["defaultGatherLayer"] == "rack"


def test_a_custom_layer_under_a_preset_is_re_parented_to_the_vocabulary():
    topology = {
        "layers": [
            {"name": "Zone", "labelKeys": ["topology.kubernetes.io/zone"]},
            {"name": "Pod", "labelKeys": ["dc/pod"], "parentLayer": "Zone"},
        ]
    }
    folded, renames = migration._fold(topology)

    assert renames == {"Zone": "zone"}
    assert folded["layers"] == [
        {"name": "Pod", "labelKeys": ["dc/pod"], "parentLayer": "zone"}
    ]


def test_a_preset_name_with_customised_keys_is_left_alone():
    """Same name, different keys: the operator meant something else by it."""
    topology = {"layers": [{"name": "Rack", "labelKeys": ["dc/rack"]}]}
    folded, renames = migration._fold(topology)

    assert renames == {}
    assert folded == topology
