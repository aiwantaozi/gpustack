"""The fixed vocabulary of places a worker can be, and how a worker's place is read.

An operator describes a machine room, not a schema: "node-9 is in rack R3".
The layers a machine room has are a small, stable set — region, zone, room,
row, rack, the access switch, the host — so they are declared once here, in
order, and a cluster only ever fills in *values*. A field with a value on at
least one worker is a layer of that cluster's tree; a field nobody filled in is
not. There is no "declare a layer" step.

Two things sit beside the tree rather than in it:

- **The accelerator domain** (NVLink / HCCS / UB reach). It nests in no fixed
  place — inside a host on an 8-card server, across sixteen racks on a
  CloudMatrix384 — so it is a flat grouping with its own keys, consulted by the
  solver as its own candidate set.
- **Custom layers**, for the fleet whose fabric has a rung this list does not
  name. They live in ``Cluster.topology.layers`` with the parent chain the tree
  has always used, and slot between the vocabulary's fields.

Every field owns one key under ``topology.gpustack.ai/``, listed first among its
candidates. That is what the table writes when an operator fills in a value,
and because it is tried first, a hand-filled value always wins over whatever a
cloud, a device or a discovery tool wrote under another key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from gpustack.scheduler.topology import (
    ACCELERATOR_DOMAIN_LAYER,
    ACCELERATOR_SUB_DOMAIN_LAYER,
    NODE_LAYER,
    TopologyLayerSpec,
    effective_topology_labels,
)

GPUSTACK_PREFIX = "topology.gpustack.ai/"

ACCELERATOR_DOMAIN = ACCELERATOR_DOMAIN_LAYER
"""The id of the domain field, in the same namespace as layer ids so a saved
``gather.layer`` can name it."""

ACCELERATOR_SUB_DOMAIN = ACCELERATOR_SUB_DOMAIN_LAYER
"""The solver's scope for "same domain and same sub-domain"; never a field an
operator fills in directly (its keys point at another field's)."""


@dataclass(frozen=True)
class VocabularyField:
    id: str
    label_keys: Tuple[str, ...]
    """Any-of, first present wins. The ``topology.gpustack.ai/`` key is first."""

    @property
    def primary_key(self) -> str:
        return self.label_keys[0]


# Root-to-leaf. The order is the one thing this module refuses to let a
# cluster change: a tree whose rows sit inside racks is not a tree anyone
# recognises, and a fixed order is what lets two clusters mean the same thing
# by "rack".
VOCABULARY: Tuple[VocabularyField, ...] = (
    VocabularyField(
        "region",
        (GPUSTACK_PREFIX + "region", "topology.kubernetes.io/region"),
    ),
    VocabularyField(
        "zone",
        (GPUSTACK_PREFIX + "zone", "topology.kubernetes.io/zone"),
    ),
    VocabularyField("room", (GPUSTACK_PREFIX + "room",)),
    VocabularyField("row", (GPUSTACK_PREFIX + "row",)),
    VocabularyField(
        "rack",
        (GPUSTACK_PREFIX + "rack", "topology.kubernetes.io/rack"),
    ),
    # The leaf switch a worker's ports are cabled to, as its LLDP neighbour
    # reports itself. Discovered by the worker, not filled in, and the closest
    # thing to a rack a host can learn on its own.
    VocabularyField(
        "switch",
        (GPUSTACK_PREFIX + "switch", "fabric.topograph.run/tier-0"),
    ),
)

VOCABULARY_IDS = tuple(f.id for f in VOCABULARY)

ACCELERATOR_DOMAIN_KEYS: Tuple[str, ...] = (
    GPUSTACK_PREFIX + "accelerator-domain",
    "nvidia.com/gpu.clique",
)

SWITCH_NAME_KEY = GPUSTACK_PREFIX + "switch-name"
"""Where the worker records the switch's own name beside its chassis id, so the
UI can show ``CE8875-50`` instead of a MAC. Read for display only; membership
is decided by the chassis id."""

RESERVED_IDS = frozenset(VOCABULARY_IDS) | {ACCELERATOR_DOMAIN, ACCELERATOR_SUB_DOMAIN}
"""Ids a custom layer may not take."""


@dataclass(frozen=True)
class KnownKey:
    """A label key some vendor or tool is known to write, offered in the
    Advanced panel so nobody has to remember the spelling."""

    key: str
    vendor: str
    fits: Tuple[str, ...]
    note: str = ""


KNOWN_KEYS: Tuple[KnownKey, ...] = (
    KnownKey(
        "fabric.topograph.run/tier-0",
        "Topograph",
        ("switch", "rack"),
        "The switch closest to the node.",
    ),
    KnownKey(
        "fabric.topograph.run/tier-1",
        "Topograph",
        ("row", "room"),
        "One tier above the leaf switch.",
    ),
    KnownKey(
        "fabric.topograph.run/tier-2",
        "Topograph",
        ("room", "zone"),
        "Two tiers above the leaf switch.",
    ),
    KnownKey(
        "accelerator.topograph.run/domain",
        "Topograph",
        (ACCELERATOR_DOMAIN,),
        "NVLink domain as Topograph discovers it.",
    ),
    KnownKey(
        "network.topology.nvidia.com/accelerator",
        "NVIDIA",
        (ACCELERATOR_DOMAIN,),
        "NVLink domain.",
    ),
    KnownKey(
        "network.topology.nvidia.com/block",
        "NVIDIA",
        ("rack", "row"),
        "IB fabric block.",
    ),
    KnownKey(
        "network.topology.nvidia.com/spine",
        "NVIDIA",
        ("room", "zone"),
        "IB fabric spine.",
    ),
    KnownKey(
        "network.topology.nvidia.com/datacenter",
        "NVIDIA",
        ("zone", "region"),
        "IB fabric datacenter.",
    ),
    KnownKey(
        "cloud.google.com/gce-topology-subblock",
        "GKE",
        ("rack", ACCELERATOR_DOMAIN),
        "On A4X this is the NVL72 domain.",
    ),
    KnownKey(
        "cloud.google.com/gce-topology-block",
        "GKE",
        ("row", "room"),
        "One fast network.",
    ),
    KnownKey(
        "topology.k8s.aws/network-node-layer-3",
        "EKS",
        ("rack",),
        "Finest EKS network layer.",
    ),
    KnownKey(
        "topology.k8s.aws/ultraserver-id",
        "EKS",
        (ACCELERATOR_DOMAIN,),
        "GB200 UltraServer NVL72 domain.",
    ),
    KnownKey(
        "ds.coreweave.com/nvlink.domain",
        "CoreWeave",
        (ACCELERATOR_DOMAIN,),
        "NVL72 domain.",
    ),
    KnownKey(
        "topology.kubernetes.io/zone", "Kubernetes", ("zone",), "Well-known zone label."
    ),
    KnownKey(
        "topology.kubernetes.io/region",
        "Kubernetes",
        ("region",),
        "Well-known region label.",
    ),
)


def source_of(worker, key: str) -> str:
    """Which side of the merge a key came from, for the UI's source badge."""
    labels = getattr(worker, "labels", None) or {}
    return "user" if key in labels else "discovered"


@dataclass(frozen=True)
class ResolvedLayer:
    """One rung of a cluster's tree, vocabulary or custom, keys resolved."""

    id: str
    label_keys: Tuple[str, ...]
    builtin: bool

    @property
    def primary_key(self) -> Optional[str]:
        return self.label_keys[0] if self.label_keys else None

    def spec(self, parent: Optional[str]) -> TopologyLayerSpec:
        return TopologyLayerSpec(
            layer=self.id, label_keys=self.label_keys, parent_layer=parent
        )


@dataclass(frozen=True)
class ResolvedDomain:
    label_keys: Tuple[str, ...]
    sub_domain_keys: Tuple[str, ...]

    @property
    def primary_key(self) -> str:
        return self.label_keys[0] if self.label_keys else ACCELERATOR_DOMAIN_KEYS[0]


@dataclass
class ResolvedTopology:
    """A cluster's declaration with the vocabulary filled in.

    ``chain`` is every layer root-to-leaf (leaf excluded) whether or not any
    worker has a value for it; ``active`` is the subset the tree is built from.
    """

    chain: List[ResolvedLayer] = field(default_factory=list)
    domain: ResolvedDomain = field(
        default_factory=lambda: ResolvedDomain(ACCELERATOR_DOMAIN_KEYS, ())
    )

    def layer(self, layer_id: str) -> Optional[ResolvedLayer]:
        for layer in self.chain:
            if layer.id == layer_id:
                return layer
        return None

    def active(self, workers: Iterable) -> List[ResolvedLayer]:
        """The layers at least one worker has a value for.

        This is the rule that replaces declaring: fill a field in and the tree
        grows a layer; leave it empty and it does not exist. Custom layers are
        kept even when empty — an operator who wrote one down wants to see
        that nobody matches it, and the preview says so — but they still do not
        become a tier the deployment form can ask for.
        """
        labels = [effective_topology_labels(w) for w in workers]
        out: List[ResolvedLayer] = []
        for layer in self.chain:
            if not layer.builtin or any(
                _has_value(lb, layer.label_keys) for lb in labels
            ):
                out.append(layer)
        return out

    def specs(self, layers: Sequence[ResolvedLayer]) -> List[TopologyLayerSpec]:
        """Chain the given layers into what ``build_topology`` takes."""
        specs: List[TopologyLayerSpec] = []
        parent: Optional[str] = None
        for layer in layers:
            specs.append(layer.spec(parent))
            parent = layer.id
        return specs


def _has_value(labels: Mapping[str, str], keys: Sequence[str]) -> bool:
    return any(labels.get(k) for k in keys)


def resolve(topology) -> ResolvedTopology:
    """Fill the vocabulary into a cluster's ``ClusterTopology`` (or None).

    ``layers`` empty is the common case and means the vocabulary as-is. A
    non-empty ``layers`` is the Advanced panel's work: an entry named after a
    vocabulary field replaces that field's keys; any other entry is a custom
    layer whose place in the chain is fixed by its ``parent_layer``.

    Custom layers are spliced in by their parent: right below the parent they
    name, which may be a vocabulary field or another custom layer. A custom
    layer with no parent sits at the top, above the vocabulary. The vocabulary
    itself never moves.
    """
    declared = list(getattr(topology, "layers", None) or [])
    overrides = {}
    customs = []
    for entry in declared:
        name = getattr(entry, "name", None)
        if not name:
            continue
        if name in VOCABULARY_IDS:
            overrides[name] = tuple(getattr(entry, "label_keys", None) or ())
        else:
            customs.append(entry)

    chain: List[ResolvedLayer] = []
    for vocab in VOCABULARY:
        keys = overrides.get(vocab.id, vocab.label_keys)
        # The owned key stays first whatever the override said: it is the key
        # the table writes, and if it were not tried first a hand-filled value
        # could lose to a discovered one — the one ordering this design forbids.
        keys = (vocab.primary_key,) + tuple(k for k in keys if k != vocab.primary_key)
        chain.append(ResolvedLayer(vocab.id, keys, builtin=True))

    # Splice customs below their parent. Repeated until stable so a custom
    # layer under another custom layer lands after both are placed.
    pending = list(customs)
    while pending:
        progressed = False
        for entry in list(pending):
            parent = getattr(entry, "parent_layer", None)
            keys = tuple(getattr(entry, "label_keys", None) or ())
            layer = ResolvedLayer(entry.name, keys, builtin=False)
            if parent is None:
                chain.insert(0, layer)
            else:
                index = next((i for i, x in enumerate(chain) if x.id == parent), None)
                if index is None:
                    continue
                chain.insert(index + 1, layer)
            pending.remove(entry)
            progressed = True
        if not progressed:
            # A parent that does not exist. The schema validator refuses this
            # on save; here it is dropped so a stale row cannot take the
            # scheduler down.
            break

    domain_spec = getattr(topology, "accelerator_domain", None)
    domain_keys = (
        tuple(getattr(domain_spec, "label_keys", None) or ()) or ACCELERATOR_DOMAIN_KEYS
    )
    domain_keys = (ACCELERATOR_DOMAIN_KEYS[0],) + tuple(
        k for k in domain_keys if k != ACCELERATOR_DOMAIN_KEYS[0]
    )
    sub_keys = tuple(getattr(domain_spec, "sub_domain_keys", None) or ())
    return ResolvedTopology(chain=chain, domain=ResolvedDomain(domain_keys, sub_keys))


def validate_declaration(topology) -> ResolvedTopology:
    """Refuse a declaration that cannot become a tree; return it resolved.

    Only the declaration is judged, never the data: a custom layer naming a
    parent that does not exist, taking a vocabulary id as its name, or forking
    the chain means the operator's intent is unknowable, while a worker missing
    a value is a normal state the tree has a place for. Raised as
    ``TopologyError`` so the schema validator and the scheduler refuse the
    same declarations for the same reasons.
    """
    from gpustack.scheduler.topology import TopologyError, layer_names

    layers = list(getattr(topology, "layers", None) or [])
    seen = set()
    custom_names = {
        layer_.name
        for layer_ in layers
        if layer_.name and layer_.name not in VOCABULARY_IDS
    }
    for layer in layers:
        if not layer.name:
            raise TopologyError("A topology layer must have a name.")
        if layer.name in seen:
            raise TopologyError(f"Duplicate topology layer {layer.name!r}.")
        seen.add(layer.name)
        if layer.name in RESERVED_IDS and layer.name not in VOCABULARY_IDS:
            raise TopologyError(f"{layer.name!r} is reserved and cannot be a layer.")
        # A vocabulary entry's `parent_layer` is ignored rather than refused:
        # its place in the chain is fixed, and a client that serialises the
        # whole chain uniformly (each entry pointing at its predecessor) is
        # not wrong about anything that matters.
        if (
            layer.name not in VOCABULARY_IDS
            and layer.parent_layer
            and layer.parent_layer not in VOCABULARY_IDS
            and layer.parent_layer not in custom_names
        ):
            raise TopologyError(
                f"Topology layer {layer.name!r} names an unknown parent "
                f"{layer.parent_layer!r}."
            )

    # A fork: two custom layers under the same parent (or two at the top).
    # The tree the scheduler walks has one path from root to leaf, and a fork
    # would make "how many layers up" have no single answer.
    parents = [
        layer_.parent_layer for layer_ in layers if layer_.name not in VOCABULARY_IDS
    ]
    if len(parents) != len(set(parents)):
        raise TopologyError(
            "Two custom topology layers share a parent; the layers must form a "
            "single chain."
        )

    resolved = resolve(topology)
    placed = {layer.id for layer in resolved.chain}
    unreachable = sorted(name for name in custom_names if name not in placed)
    if unreachable:
        raise TopologyError(
            f"Topology layers {', '.join(unreachable)} are not reachable from the "
            "cluster root; the layers must form a single chain."
        )
    layer_names(resolved.specs(resolved.chain))
    return resolved


def gather_layer_names(resolved: ResolvedTopology) -> List[str]:
    """Everything a `gather.layer` may name: the chain, the leaf, the domain."""
    from gpustack.scheduler.topology import layer_names

    return layer_names(resolved.specs(resolved.chain)) + [ACCELERATOR_DOMAIN]


def primary_key_for(resolved: ResolvedTopology, field_id: str) -> Optional[str]:
    """The key the table writes for a field, or None for a field with none.

    The leaf has no key: a host is itself. A custom layer's first key is what
    it writes, which is why the Advanced panel tells an operator to put the
    key they mean to write first.
    """
    if field_id == ACCELERATOR_DOMAIN:
        return resolved.domain.primary_key
    if field_id == NODE_LAYER:
        return None
    layer = resolved.layer(field_id)
    return layer.primary_key if layer else None


def display_name(field_id: str) -> str:
    """An English fallback for the UI, which localises the builtin ids itself."""
    return {
        "region": "Region",
        "zone": "Zone",
        "room": "Room",
        "row": "Row",
        "rack": "Rack",
        "switch": "Access switch",
        ACCELERATOR_DOMAIN: "Accelerator domain",
        NODE_LAYER: "Host",
    }.get(field_id, field_id)
