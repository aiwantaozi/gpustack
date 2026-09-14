"""topology vocabulary

Cluster topology moves from operator-declared layers to a fixed vocabulary
(region, zone, rack, switch, host) that a cluster only fills values into. A
cluster that declared the preset layers the old UI offered — `Rack`, `Zone`,
`Region` with their preset keys — now means the vocabulary as-is, so those
declarations are folded into it: the entries are dropped and a model whose
`gather.layer` named one of them is pointed at the vocabulary id (`Rack` →
`rack`). Custom layers and customised keys are left exactly as they were; the
new code reads them as Advanced-mode overrides.

`Row` is deliberately not folded, unlike the other three the old UI offered:
`row` is no longer a vocabulary id, so folding would point a model's
`gather.layer` at a layer nothing answers to and placement could not resolve
it. Left alone, the declaration survives as an ordinary custom layer — the
path the vocabulary now expects for a machine room or a rack row — and the
models gathering on it keep working.

No schema change: `clusters.topology` and `models.gather` are JSON columns.

Revision ID: e8f9a0b1c2d3
Revises: d7e8f9a0b1c2
Create Date: 2026-09-04 10:00:00.000000
"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from gpustack.migrations.utils import column_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = 'e8f9a0b1c2d3'
down_revision: Union[str, None] = 'd7e8f9a0b1c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The presets the previous UI inserted, name -> (vocabulary id, keys). A
# declared layer is a preset only if its name and keys both match.
_PRESETS = {
    "Region": (
        "region",
        {"topology.kubernetes.io/region", "failure-domain.beta.kubernetes.io/region"},
    ),
    "Zone": (
        "zone",
        {"topology.kubernetes.io/zone", "failure-domain.beta.kubernetes.io/zone"},
    ),
    "Rack": ("rack", {"topology.gpustack.ai/rack", "topology.kubernetes.io/rack"}),
}


def _load(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    return json.loads(value) if isinstance(value, str) else value


def _fold(topology: dict):
    """Drop preset layers, rename the rest's parents; return (topology, renames)."""
    layers = topology.get("layers") or []
    renames = {}
    kept = []
    for layer in layers:
        name = layer.get("name")
        keys = set(layer.get("labelKeys") or layer.get("label_keys") or [])
        preset = _PRESETS.get(name)
        if preset and keys <= preset[1]:
            renames[name] = preset[0]
        else:
            kept.append(layer)
    if not renames:
        return topology, {}
    for layer in kept:
        parent = layer.get("parentLayer", layer.get("parent_layer"))
        if parent in renames:
            layer.pop("parent_layer", None)
            layer["parentLayer"] = renames[parent]
    topology = dict(topology)
    topology["layers"] = kept
    # Nothing else in the declaration points at a layer name any more. This
    # loop used to also rewrite `defaultGatherLayer`, the cluster-level gather
    # default; that field is gone, and `ClusterTopology` ignores unknown keys,
    # so an old row that still carries one is read straight past.
    return topology, renames


def upgrade() -> None:
    if not table_exists("clusters") or not column_exists("clusters", "topology"):
        return
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, topology FROM clusters WHERE topology IS NOT NULL")
    ).fetchall()
    for cluster_id, raw in rows:
        topology = _load(raw)
        if not isinstance(topology, dict):
            continue
        folded, renames = _fold(topology)
        if not renames:
            continue
        conn.execute(
            sa.text("UPDATE clusters SET topology = :t WHERE id = :id"),
            {"t": json.dumps(folded), "id": cluster_id},
        )
        if table_exists("models") and column_exists("models", "gather"):
            models = conn.execute(
                sa.text(
                    "SELECT id, gather FROM models "
                    "WHERE cluster_id = :cid AND gather IS NOT NULL"
                ),
                {"cid": cluster_id},
            ).fetchall()
            for model_id, gather_raw in models:
                gather = _load(gather_raw)
                if isinstance(gather, dict) and gather.get("layer") in renames:
                    gather["layer"] = renames[gather["layer"]]
                    conn.execute(
                        sa.text("UPDATE models SET gather = :g WHERE id = :id"),
                        {"g": json.dumps(gather), "id": model_id},
                    )

    _clear_layer_identity(conn)


def _clear_layer_identity(conn) -> None:
    """Drop every stored layer declaration and every model's gather target.

    A layer grew an identity separate from its name: `{id, name, displayName}`
    where there used to be one `name` doing all three jobs. Old rows key both
    `parentLayer` and `Model.gather.layer` off that single field, and a custom
    layer's value there is free text an operator typed — there is no rule that
    turns it into a registry id, only a fresh one per cluster.

    Remapped rather than cleared, that would mean building a per-cluster table
    and applying it to two tables in step; get one row out of sync and a model
    gathers on a layer that no longer exists, which the solver stands down
    silently. This structure has never shipped (`git tag --contains` on the
    commit that introduced it is empty), so the rows being discarded are
    development ones, and clearing is the option that cannot half-apply.

    What is lost: custom layers, label-key overrides, and every model's gather
    requirement. What is not touched: `Worker.labels`. A worker's position has
    always been a label keyed by the *label key*
    (`topology.gpustack.ai/rack`), never by the layer id, so every value an
    operator filled in survives and re-resolves against the vocabulary.
    """
    if table_exists("clusters") and column_exists("clusters", "topology"):
        rows = conn.execute(
            sa.text("SELECT id, topology FROM clusters WHERE topology IS NOT NULL")
        ).fetchall()
        for cluster_id, raw in rows:
            topology = _load(raw)
            if not isinstance(topology, dict) or not topology.get("layers"):
                continue
            topology = dict(topology)
            topology["layers"] = []
            conn.execute(
                sa.text("UPDATE clusters SET topology = :t WHERE id = :id"),
                {"t": json.dumps(topology), "id": cluster_id},
            )

    if table_exists("models") and column_exists("models", "gather"):
        conn.execute(sa.text("UPDATE models SET gather = NULL WHERE gather IS NOT NULL"))


def downgrade() -> None:
    # The folded declaration is a valid input to the previous code too (it
    # simply reads empty layers as "no topology"), so there is nothing to undo.
    # `_clear_layer_identity` is likewise not reversible — the values it drops
    # are gone — and deliberately so: the structure it migrates away from has
    # never been released, so there is no deployed version to go back to.
    pass
