"""the accelerator domain becomes a second layer chain

⚠️ **Superseded, and deliberately left in place.** The 2026-09-14 review
collapsed the two chains into one — `accelerator_layers` no longer exists on
`ClusterTopology`, which is `extra="ignore"`, so the key this migration writes
is simply not read any more. It stays because it is a pure JSON rewrite with no
DDL: removing it would break the revision chain for any database that has
already run it, while keeping it changes nothing for one that has not. No
follow-up migration converts these rungs onto the one chain either — the review
decided against a compatibility fallback, because where on the chain a domain
belongs is the operator's answer and not one a migration can guess.

`Cluster.topology.accelerator_domain` was a single object with `labelKeys` and
`subDomainKeys` — a domain, and one optional tier inside it. Two levels, fixed.
Atlas 950 has three (blade 1008 GB/s, cabinet 896, across cabinets 448), so the
shape could not hold the hardware that is shipping, and adding a third would
have been another schema change.

It becomes `accelerator_layers`: a parent chain with the same structure as
`layers`, built by the same code, with no upper bound on depth. The conversion
is mechanical:

    acceleratorDomain.labelKeys      -> accelerator_layers[0] ("accelerator_domain")
    acceleratorDomain.subDomainKeys  -> accelerator_layers[1] ("accelerator_sub_domain",
                                        parented on the first)

Only non-empty key lists produce a layer. An `acceleratorDomain` that was
present but empty meant "the built-in keys", and an empty `accelerator_layers`
means exactly the same thing — so it is simply dropped.

⚠️ One deliberate semantic change on the sub-domain, and it is a widening.
`group_by_domain` used to *drop* a worker that had a domain but no sub-domain
value from the sub-domain grouping. On a chain that worker lands in the
unclassified bucket under its own domain instead: still not gathered with
anyone (the solver excludes that bucket, as it always has), but now visible as
"3 workers have no cabinet yet", which is the number the topology page leads
with. No worker changes domain, and no placement becomes possible that was not.

Gather constraints need no rewriting at all, and that is by design: layer names
are unique across both chains, so a saved `gather.layer` of `accelerator_domain`
still names exactly one rung — now the top of the accelerator chain rather than
a scope beside the tree. Which chain it is on is derived from the name at read
time. (The cluster-level `defaultGatherLayer` this paragraph also used to cover
no longer exists at all; see `_convert`.)

⚠️ One name the old model could produce is *not* carried over, because it never
existed as a layer: the sub-domain scope was called `accelerator_sub_domain`
internally but was never selectable as a `gather.layer` (`gather_layer_names`
offered the tree, the leaf and `accelerator_domain`, and `tier_names` filtered
the sub-domain out on purpose). So no stored constraint can name it, and the
layer this migration creates under that name starts with no references.

No schema change: `clusters.topology` is a JSON column.

Revision ID: b1c2d3e4f5a6
Revises: a0b1c2d3e4f5
Create Date: 2026-09-11 10:00:00.000000
"""

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from gpustack.migrations.utils import column_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, None] = 'a0b1c2d3e4f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DOMAIN_LAYER = "accelerator_domain"
_SUB_DOMAIN_LAYER = "accelerator_sub_domain"


def _load(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    return json.loads(value) if isinstance(value, str) else value


def _get(obj: dict, *names):
    """Read a field under either serialised spelling (camel or snake)."""
    for name in names:
        if obj.get(name) is not None:
            return obj[name]
    return None


def _convert(topology: dict):
    """Return the rewritten topology, or None when nothing had to change."""
    changed = False
    out = dict(topology)

    domain = _get(out, "acceleratorDomain", "accelerator_domain")
    if domain is not None:
        # A key serialised as null is left alone rather than rewritten: the new
        # schema ignores it, and touching every row that merely carries
        # `"accelerator_domain": null` would be a migration that changes
        # nothing on almost every cluster there is.
        out.pop("acceleratorDomain", None)
        out.pop("accelerator_domain", None)
        changed = True

    layers = []
    if isinstance(domain, dict):
        keys = _get(domain, "labelKeys", "label_keys") or []
        sub_keys = _get(domain, "subDomainKeys", "sub_domain_keys") or []
        if keys:
            layers.append({"name": _DOMAIN_LAYER, "labelKeys": list(keys)})
        if sub_keys:
            # Parented on the domain rung whether or not the domain's own keys
            # were customised: the tier is inside the domain either way, and
            # the built-in rung exists with or without an override entry.
            layers.append(
                {
                    "name": _SUB_DOMAIN_LAYER,
                    "labelKeys": list(sub_keys),
                    "parentLayer": _DOMAIN_LAYER,
                }
            )
    if layers:
        out["acceleratorLayers"] = layers
        changed = True

    # `defaultGatherLayer` is deliberately left alone — now for a blunter
    # reason than when this was written. The field has since been removed from
    # `ClusterTopology` altogether (cluster-level gather inheritance is gone),
    # and the model ignores unknown keys, so whatever an old row stores there
    # is never read. Rewriting it would be work on a value with no reader.
    return out if changed else None


def upgrade() -> None:
    conn = op.get_bind()

    if table_exists("clusters") and column_exists("clusters", "topology"):
        rows = conn.execute(
            sa.text("SELECT id, topology FROM clusters WHERE topology IS NOT NULL")
        ).fetchall()
        for cluster_id, raw in rows:
            topology = _load(raw)
            if not isinstance(topology, dict):
                continue
            converted = _convert(topology)
            if converted is None:
                continue
            conn.execute(
                sa.text("UPDATE clusters SET topology = :t WHERE id = :id"),
                {"t": json.dumps(converted), "id": cluster_id},
            )


def downgrade() -> None:
    # Not reversible without loss, and not worth pretending otherwise: a chain
    # of three rungs has no representation in `acceleratorDomain`, which is the
    # whole reason for this migration. The previous code reads an unknown
    # `acceleratorLayers` key as absent (`extra="ignore"`) and falls back to the
    # built-in domain keys, so rolling the code back degrades resolution
    # rather than breaking — which is the failure mode this subsystem is
    # allowed to have.
    pass
