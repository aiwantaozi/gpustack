"""The single accelerator-domain object becomes a second layer chain.

⚠️ **The chain this migration writes no longer exists.** The 2026-09-14 review
collapsed the two chains into one, so `acceleratorLayers` is a key nothing
reads. The migration is kept — it is a pure JSON rewrite, and dropping it would
break the revision chain for a database that has already run it — so its
conversion is still pinned here, plus the one property that actually matters
now: the shape it emits is inert rather than invalid.

What the rest pins is that nothing is lost in the conversion and nothing that
did not have to move is touched — a migration that rewrote every cluster row to
drop a null key would be noise in the audit trail of every deployment that ever
ran, and it would still change nothing.
"""

import importlib.util
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[2]
    / "gpustack/migrations/versions"
    / "2026_09_11_1000-b1c2d3e4f5a6_topology_accelerator_layers.py"
)
_spec = importlib.util.spec_from_file_location("accelerator_layers_migration", _PATH)
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


def test_custom_domain_keys_become_the_chains_first_rung():
    converted = migration._convert(
        {"acceleratorDomain": {"labelKeys": ["ds.coreweave.com/nvlink.domain"]}}
    )

    assert converted["acceleratorLayers"] == [
        {
            "name": "accelerator_domain",
            "labelKeys": ["ds.coreweave.com/nvlink.domain"],
        }
    ]
    assert "acceleratorDomain" not in converted


def test_sub_domain_keys_become_a_second_rung_parented_on_the_first():
    """The tier that used to be a field beside the domain is now a rung under
    it — which is what lets a third one be added later without a schema."""
    converted = migration._convert(
        {"acceleratorDomain": {"subDomainKeys": ["topology.gpustack.ai/rack"]}}
    )

    assert converted["acceleratorLayers"] == [
        {
            "name": "accelerator_sub_domain",
            "labelKeys": ["topology.gpustack.ai/rack"],
            "parentLayer": "accelerator_domain",
        }
    ]


def test_both_key_lists_convert_in_order():
    converted = migration._convert(
        {
            "acceleratorDomain": {
                "labelKeys": ["hw/spod"],
                "subDomainKeys": ["hw/cabinet"],
            }
        }
    )

    assert [layer["name"] for layer in converted["acceleratorLayers"]] == [
        "accelerator_domain",
        "accelerator_sub_domain",
    ]
    assert converted["acceleratorLayers"][1]["parentLayer"] == "accelerator_domain"


def test_the_snake_case_spelling_is_read_too():
    """`model_dump()` writes snake, the API writes camel; both are in the wild."""
    converted = migration._convert({"accelerator_domain": {"label_keys": ["hw/spod"]}})
    assert converted["acceleratorLayers"][0]["labelKeys"] == ["hw/spod"]
    assert "accelerator_domain" not in converted


def test_an_empty_domain_object_produces_no_layers():
    """It meant "the built-in keys", and an empty chain means the same."""
    converted = migration._convert({"acceleratorDomain": {}})
    assert converted.get("acceleratorLayers") is None


def test_a_cluster_with_nothing_to_convert_is_not_rewritten():
    assert migration._convert({"layers": []}) is None
    assert migration._convert({"acceleratorDomain": None}) is None
    assert migration._convert({"accelerator_domain": None, "layers": []}) is None


# --- gather constraints need no rewriting ----------------------------------- #


def test_a_gather_default_is_left_exactly_as_it_was():
    """The migration rewrites keys, never constraints."""
    assert migration._convert({"defaultGatherLayer": "accelerator_domain"}) is None
    assert migration._convert({"defaultGatherLayer": "rack"}) is None

    converted = migration._convert(
        {
            "acceleratorDomain": {"labelKeys": ["hw/spod"]},
            "defaultGatherStrategy": "MustGather",
            "defaultGatherLayer": "accelerator_domain",
        }
    )
    assert converted["defaultGatherLayer"] == "accelerator_domain"
    assert "defaultGatherChain" not in converted


# --- what the converted shape means now ------------------------------------ #


def test_the_converted_shape_is_inert_rather_than_invalid():
    """🔴 The property that lets this migration stay.

    `acceleratorLayers` is a key the single-chain `ClusterTopology` does not
    declare, and the model is `extra="ignore"` — so nothing downstream has to
    know the second chain ever existed. That is the §3 decision ("no data
    migration, no compatibility fallback") landing on an already-written row:
    the declaration stops meaning anything, and nothing breaks.

    🔴 What this test no longer claims is that such a row *loads*. It cannot:
    a layer grew an `id` separate from its name, and a row of this vintage has
    only the name. Nothing has to load it either — a later step of the same
    unreleased bundle (`_clear_layer_identity`) empties `layers` on every
    cluster, so by the time any code reads one of these rows it carries no
    layers at all. Asserting on `_convert` alone is what is left that is true.
    """
    converted = migration._convert(
        {
            "layers": [{"name": "rack", "labelKeys": ["dc/rack"]}],
            "acceleratorDomain": {
                "labelKeys": ["hw/spod"],
                "subDomainKeys": ["hw/cabinet"],
            },
        }
    )
    assert [layer["name"] for layer in converted["acceleratorLayers"]] == [
        "accelerator_domain",
        "accelerator_sub_domain",
    ]
    # The first chain is carried through untouched; only the second is folded.
    assert [layer["name"] for layer in converted["layers"]] == ["rack"]
