"""The fixed vocabulary of places a worker can be, and how a worker's place is read.

An operator describes a machine room, not a schema: "node-9 is in rack R3".
The layers a machine room has are a small, stable set — the room, the row of
cabinets, the cabinet, the host — so they are declared once here, in order, and
a cluster only ever fills in *values*. A field with a value on at least one
worker is a layer of that cluster's tree; a field nobody filled in is not.
There is no "declare a layer" step for the built-in three.

🔴 **There used to be two vocabularies, because there were two chains** — the
network one above and an accelerator one whose single built-in rung was the
NVLink/HCCS/UB domain, held apart from the network chain because the domain's
containment direction versus a rack comes out three different ways across four
shipping hardware generations. Review overturned that. A domain whose boundary
is a run of contiguous cabinets is expressible as one rung of the one chain,
and on every generation that ships it is contiguous; where the domain sits
*inside* one machine (910B2), "same domain" and "same host" are the same
constraint and the built-in leaf already covers it. So there is one chain, the
operator decides where the domain rung goes on it, and the keys the domain is
published under survive as **candidate keys** (``KNOWN_KEYS``) rather than as a
second vocabulary.

**Custom layers** are for every rung this list does not name — the accelerator
domain among them. They live in ``Cluster.topology.layers`` with the parent
chain the tree has always used, and slot between the vocabulary's fields.

Every field owns one key under ``topology.gpustack.ai/``, listed first among its
candidates. That is what the table writes when an operator fills in a value,
and because it is tried first, a hand-filled value always wins over whatever a
cloud, a device or a discovery tool wrote under another key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from gpustack.scheduler.topology import (
    NODE_LAYER,
    ROOT_LAYER,
    TopologyLayerSpec,
    effective_topology_labels,
)

GPUSTACK_PREFIX = "topology.gpustack.ai/"


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
#
# 🔴 `region`, `zone` and `switch` were removed in review, and `room`/`row`
# restored. `region`/`zone` are cloud words for a failure domain, not for a
# distance a KV transfer notices; `switch` is a fact the worker discovers by
# itself and writes as a label, so it needs no built-in rung — an operator who
# wants to place by it adds a layer pointing at its key, which is the same
# operation as adding one for the accelerator domain.
VOCABULARY: Tuple[VocabularyField, ...] = (
    VocabularyField("room", (GPUSTACK_PREFIX + "room",)),
    VocabularyField("row", (GPUSTACK_PREFIX + "row",)),
    VocabularyField(
        "rack",
        (GPUSTACK_PREFIX + "rack", "topology.kubernetes.io/rack"),
    ),
)

VOCABULARY_IDS = tuple(f.id for f in VOCABULARY)

SWITCH_KEY = GPUSTACK_PREFIX + "switch"
"""The leaf switch a worker's ports are cabled to, as its LLDP neighbour
reports itself. Written by the worker, never built in: it is a *fact*, and a
fact becomes a place only when an operator declares a layer that reads it."""

SWITCH_NAME_KEY = GPUSTACK_PREFIX + "switch-name"
"""Where the worker records the switch's own name beside its chassis id, so the
UI can show ``CE8875-50`` instead of a MAC. Read for display only; membership
is decided by the chassis id."""

RESERVED_IDS = frozenset({ROOT_LAYER, NODE_LAYER})
"""Names no declared layer may take: the implicit root, and the leaf.

🔴 It used to also hold the *other* chain's built-in ids — with two chains,
``rack`` on the accelerator chain and ``accelerator_domain`` on the network one
had to be refused, or a layer name could not be read without first asking which
chain it came from. With one chain there is no "other chain" to protect a name
from, and a vocabulary id was never reserved against its own chain anyway:
naming ``rack`` in the declaration is how an operator overrides that field's
keys. So the set collapses to the two names that are not layers at all.

``accelerator_domain`` in particular is now free, and using it is the
recommended way to spell the domain rung as a custom layer."""


def declared_layers(topology) -> List:
    """The entries a cluster declared.

    One accessor rather than every call site reaching for the attribute, so a
    ``ClusterTopology`` and a test stub are read the same way.
    """
    return list(getattr(topology, "layers", None) or [])


@dataclass(frozen=True)
class KnownKey:
    """A label key some vendor or tool is known to write, offered in the
    Advanced panel so nobody has to remember the spelling."""

    key: str
    vendor: str
    fits: Tuple[str, ...]
    note: str = ""


# 🔴 ``fits`` names the built-in rung a key is *nearest* to, and that is all it
# is: a hint for where to insert the layer that reads it. The accelerator-domain
# and switch keys are in here rather than in ``VOCABULARY`` for the reason at
# the top of this module — they are facts the fleet publishes, and which rung
# they amount to is the operator's call, not ours.
KNOWN_KEYS: Tuple[KnownKey, ...] = (
    KnownKey(
        GPUSTACK_PREFIX + "accelerator-domain",
        "GPUStack",
        ("rack", "row"),
        "NVLink/HCCS/UB domain, as the worker's runtime reports it.",
    ),
    KnownKey(
        "nvidia.com/gpu.clique",
        "NVIDIA",
        ("rack", "row"),
        "NVLink domain, written by the driver.",
    ),
    KnownKey(
        "accelerator.topograph.run/domain",
        "Topograph",
        ("rack", "row"),
        "NVLink domain as Topograph discovers it.",
    ),
    KnownKey(
        "network.topology.nvidia.com/accelerator",
        "NVIDIA",
        ("rack", "row"),
        "NVLink domain.",
    ),
    KnownKey(
        GPUSTACK_PREFIX + "switch",
        "GPUStack",
        ("rack",),
        "The switch closest to the node, as the worker's LLDP probe heard it.",
    ),
    KnownKey(
        "fabric.topograph.run/tier-0",
        "Topograph",
        ("rack",),
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
        ("room",),
        "Two tiers above the leaf switch.",
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
        ("room",),
        "IB fabric spine.",
    ),
    KnownKey(
        "network.topology.nvidia.com/datacenter",
        "NVIDIA",
        ("room",),
        "IB fabric datacenter.",
    ),
    KnownKey(
        "cloud.google.com/gce-topology-subblock",
        "GKE",
        ("rack",),
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
        ("rack",),
        "GB200 UltraServer NVL72 domain.",
    ),
    KnownKey(
        "ds.coreweave.com/nvlink.domain",
        "CoreWeave",
        ("rack",),
        "NVL72 domain.",
    ),
    KnownKey(
        "topology.kubernetes.io/zone",
        "Kubernetes",
        ("room",),
        "Well-known zone label.",
    ),
    KnownKey(
        "topology.kubernetes.io/region",
        "Kubernetes",
        ("room",),
        "Well-known region label.",
    ),
)


def source_of(worker, key: str) -> str:
    """Which side of the merge a key came from, for the UI's source badge."""
    labels = getattr(worker, "labels", None) or {}
    return "user" if key in labels else "discovered"


@dataclass(frozen=True)
class ResolvedLayer:
    """One rung of a cluster's chain, vocabulary or custom, keys resolved."""

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


@dataclass
class ResolvedTopology:
    """A cluster's declaration with the vocabulary filled in, root-to-leaf.

    ``layers`` is every rung whether or not any worker has a value for it;
    ``active`` is the subset a tree is built from.
    """

    layers: List[ResolvedLayer] = field(default_factory=list)

    def layer(self, layer_id: str) -> Optional[ResolvedLayer]:
        for layer in self.layers:
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
        for layer in self.layers:
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

    Empty declarations are the common case and mean the vocabulary as-is:
    room/row/rack. A non-empty list is the Advanced panel's work — an entry
    named after a vocabulary field replaces that field's keys, and any other
    entry is a custom layer whose place is fixed by its ``parent_layer``.

    Custom layers are spliced in by their parent: right below the parent they
    name, which may be a vocabulary field or another custom layer. A custom
    layer with no parent sits at the top, above the vocabulary. The vocabulary
    itself never moves.
    """
    vocabulary_ids = {v.id for v in VOCABULARY}

    overrides: Dict[str, Tuple[str, ...]] = {}
    customs = []
    for entry in declared_layers(topology):
        name = getattr(entry, "name", None)
        if not name:
            continue
        if name in vocabulary_ids:
            overrides[name] = tuple(getattr(entry, "label_keys", None) or ())
        else:
            customs.append(entry)

    layers: List[ResolvedLayer] = []
    for vocab in VOCABULARY:
        keys = overrides.get(vocab.id, vocab.label_keys)
        # The owned key stays first whatever the override said: it is the key
        # the table writes, and if it were not tried first a hand-filled value
        # could lose to a discovered one — the one ordering this design forbids.
        keys = (vocab.primary_key,) + tuple(k for k in keys if k != vocab.primary_key)
        layers.append(ResolvedLayer(vocab.id, keys, builtin=True))

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
                layers.insert(0, layer)
            else:
                index = next((i for i, x in enumerate(layers) if x.id == parent), None)
                if index is None:
                    continue
                layers.insert(index + 1, layer)
            pending.remove(entry)
            progressed = True
        if not progressed:
            # A parent that does not exist. The schema validator refuses this
            # on save; here it is dropped so a stale row cannot take the
            # scheduler down.
            break

    return ResolvedTopology(layers)


def validate_declaration(topology) -> ResolvedTopology:
    """Refuse a declaration that cannot become a tree; return it resolved.

    Only the declaration is judged, never the data: a custom layer naming a
    parent that does not exist, taking a reserved id as its name or forking the
    chain means the operator's intent is unknowable, while a worker missing a
    value is a normal state the tree has a place for. Raised as
    ``TopologyError`` so the schema validator and the scheduler refuse the same
    declarations for the same reasons.
    """
    from gpustack.scheduler.topology import TopologyError, layer_names

    resolved = resolve(topology)

    vocabulary_ids = {v.id for v in VOCABULARY}
    layers = declared_layers(topology)
    seen = set()
    custom_names = {
        layer_.name
        for layer_ in layers
        if layer_.name and layer_.name not in vocabulary_ids
    }
    for layer in layers:
        if not layer.name:
            raise TopologyError("A topology layer must have a name.")
        if layer.name in seen:
            raise TopologyError(f"Duplicate topology layer {layer.name!r}.")
        seen.add(layer.name)
        if layer.name in RESERVED_IDS:
            raise TopologyError(f"{layer.name!r} is reserved and cannot be a layer.")
        # A vocabulary entry's `parent_layer` is ignored rather than refused:
        # its place in the chain is fixed, and a client that serialises the
        # whole chain uniformly (each entry pointing at its predecessor) is
        # not wrong about anything that matters.
        if (
            layer.name not in vocabulary_ids
            and layer.parent_layer
            and layer.parent_layer not in vocabulary_ids
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
        layer_.parent_layer for layer_ in layers if layer_.name not in vocabulary_ids
    ]
    if len(parents) != len(set(parents)):
        raise TopologyError(
            "Two custom topology layers share a parent; the layers must form a "
            "single chain."
        )

    placed = {layer.id for layer in resolved.layers}
    unreachable = sorted(name for name in custom_names if name not in placed)
    if unreachable:
        raise TopologyError(
            f"Topology layers {', '.join(unreachable)} are not reachable from the "
            "cluster root; the layers must form a single chain."
        )

    layer_names(resolved.specs(resolved.layers))
    return resolved


def gather_layer_names(resolved: ResolvedTopology) -> List[str]:
    """Every layer a saved ``gather.layer`` may name: the host, then the chain."""
    return [NODE_LAYER] + [layer.id for layer in resolved.layers]


def primary_key_for(resolved: ResolvedTopology, field_id: str) -> Optional[str]:
    """The key the table writes for a field, or None for a field with none.

    The leaf has no key: a host is itself. A custom layer's first key is what
    it writes, which is why the Advanced panel tells an operator to put the
    key they mean to write first.
    """
    if field_id == NODE_LAYER:
        return None
    layer = resolved.layer(field_id)
    return layer.primary_key if layer else None


def display_name(field_id: str) -> str:
    """An English fallback for the UI, which localises the builtin ids itself."""
    return {
        "room": "Room",
        "row": "Row",
        "rack": "Rack",
        NODE_LAYER: "Host",
    }.get(field_id, field_id)
