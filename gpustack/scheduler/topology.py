"""The cluster's network topology, as a tree of workers.

The tree answers one question for the group scheduler: *how far apart are two
workers*. Everything else here exists to build that tree out of the only source
of truth a worker has for where it physically sits — its labels.

**Layers are declared, not discovered.** Only the root and the leaf are built
in; every layer between them is named by the operator, because no vendor's
label scheme is universal and hard-coding one would exclude the rest.

**The leaf never collapses.** ``NodeTopologyLayer`` takes the worker's name
rather than a label, so a cluster with no topology declared at all still gets a
usable tree: one leaf per worker under the root. That is what makes every
failure here a loss of *resolution* rather than a loss of *service* — a missing
or mistyped label can only make two workers look equally distant, never make a
worker unschedulable. The scheduler reads distance to score, not to filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# The two layer names the operator cannot use for a layer of their own. The
# root is implicit (every declared layer without a parent hangs off it) and the
# leaf is the worker itself.
ROOT_LAYER = "ClusterTopologyLayer"
NODE_LAYER = "NodeTopologyLayer"

RESERVED_LAYERS = frozenset({ROOT_LAYER, NODE_LAYER})

# The domain a worker lands in when the layer's label keys all miss. Named
# rather than dropped: an unclassified worker is still schedulable, and the UI
# needs something to hang "20 workers are missing this label" off.
UNCLASSIFIED = "<unclassified>"

# The accelerator domain is not a layer of the tree; it is its own flat grouping
# (see ``group_by_domain``). Named here so a ``TopologyNode`` can say which
# grouping it belongs to.
ACCELERATOR_DOMAIN_LAYER = "accelerator_domain"
ACCELERATOR_SUB_DOMAIN_LAYER = "accelerator_sub_domain"


@dataclass(frozen=True)
class TopologyLayerSpec:
    """One declared layer.

    ``label_keys`` is any-of rather than a single key on purpose: the same
    physical layer is spelled differently by every vendor and cloud
    (``topology.kubernetes.io/zone`` on one fleet, a private key on the next),
    and a cluster that mixes them should not force the operator to relabel
    everything first. The first key present on the worker wins, and which one
    matched is reported so the UI can show it.
    """

    layer: str
    label_keys: Sequence[str] = ()
    parent_layer: Optional[str] = None


@dataclass
class TopologyNode:
    """A domain, or a worker when ``layer == NODE_LAYER``."""

    layer: str
    name: str
    parent: Optional["TopologyNode"] = None
    children: List["TopologyNode"] = field(default_factory=list)
    worker_ids: List[int] = field(default_factory=list)
    # Which of the layer's any-of keys actually matched, for the worker(s)
    # under this domain. None at the root, the leaf, and the unclassified
    # bucket, none of which are reached through a label.
    matched_label_key: Optional[str] = None

    @property
    def is_unclassified(self) -> bool:
        return self.name == UNCLASSIFIED

    def descendant_worker_ids(self) -> List[int]:
        if self.layer == NODE_LAYER:
            return list(self.worker_ids)
        out: List[int] = []
        for child in self.children:
            out.extend(child.descendant_worker_ids())
        return out


class TopologyError(ValueError):
    """A declaration that cannot be turned into a tree.

    Raised only for the declaration — never for the data. A cycle or a dangling
    parent means the operator's intent is unknowable; a worker missing a label
    means only that it is unclassified, which is a normal state.
    """


def order_layers(specs: Sequence[TopologyLayerSpec]) -> List[TopologyLayerSpec]:
    """Root-to-leaf order, derived from the parent chain.

    The chain is stored rather than an ordered list because inserting a layer
    into an ordered list renumbers every layer below it, and these names are
    referenced from saved model configurations. With a chain, a new layer names
    its parent and nothing else moves.

    A layer with no parent hangs off the implicit root. Declaring several such
    layers is a fork, which is rejected: the tree the scheduler walks has one
    path from root to leaf, and a fork would make "how many layers up" have no
    single answer.
    """
    if not specs:
        return []

    by_name: Dict[str, TopologyLayerSpec] = {}
    for spec in specs:
        if not spec.layer:
            raise TopologyError("A topology layer must have a name.")
        if spec.layer == ROOT_LAYER:
            raise TopologyError(
                f"{ROOT_LAYER!r} is the implicit root and cannot be declared. "
                "Leave `parent_layer` unset on the topmost layer instead."
            )
        if spec.layer in by_name:
            raise TopologyError(f"Duplicate topology layer {spec.layer!r}.")
        by_name[spec.layer] = spec

    # Before the root check, not after: a layer whose parent is a typo has no
    # root either, and "you declared no topmost layer" would send the operator
    # looking in the wrong place.
    children: Dict[str, List[TopologyLayerSpec]] = {}
    for spec in specs:
        if spec.parent_layer is None:
            continue
        if spec.parent_layer not in by_name:
            raise TopologyError(
                f"Topology layer {spec.layer!r} names an unknown parent "
                f"{spec.parent_layer!r}."
            )
        children.setdefault(spec.parent_layer, []).append(spec)

    roots = [s for s in specs if not s.parent_layer]
    if not roots:
        raise TopologyError(
            "Every topology layer names a parent, so none of them hangs off the "
            "cluster root. Leave `parent_layer` unset on the topmost layer."
        )
    if len(roots) > 1:
        names = ", ".join(sorted(r.layer for r in roots))
        raise TopologyError(
            f"More than one topology layer hangs off the cluster root ({names}). "
            "The layers must form a single chain from the cluster down to the node."
        )

    # No cycle guard below, deliberately. Every layer names at most one parent,
    # so the declaration is a forest of in-trees and a cycle can never be
    # reached from a root: a cycle whose members all have parents either leaves
    # no root at all (caught above) or sits unreachable beside one (caught by
    # the reachability check below). A guard here could not fire.
    ordered: List[TopologyLayerSpec] = []
    current: Optional[TopologyLayerSpec] = roots[0]
    while current is not None:
        ordered.append(current)
        kids = children.get(current.layer, [])
        if len(kids) > 1:
            names = ", ".join(sorted(k.layer for k in kids))
            raise TopologyError(
                f"Topology layer {current.layer!r} has more than one child "
                f"({names}). The layers must form a single chain."
            )
        current = kids[0] if kids else None

    if len(ordered) != len(specs):
        missing = ", ".join(sorted(set(by_name) - {o.layer for o in ordered}))
        raise TopologyError(
            f"Topology layers {missing} are not reachable from the cluster root; "
            "the layers must form a single chain."
        )

    return ordered


def effective_topology_labels(worker) -> Dict[str, str]:
    """What a worker's position is read from.

    The worker's own labels laid over what its runtime discovered
    (``status.topology_facts``). The order is the whole policy: a hand-filled
    value overrides a discovered one, and clearing the hand-filled key uncovers
    the discovered one again.
    """
    status = getattr(worker, "status", None)
    facts = getattr(status, "topology_facts", None) or {}
    labels = getattr(worker, "labels", None) or {}
    return {**facts, **labels}


def _domain_of(labels: Mapping[str, str], spec: TopologyLayerSpec):
    """The domain a worker belongs to at one layer, and the key that said so.

    Any-of: the declared keys are tried in order and the first one carrying a
    non-empty value wins. A blank value counts as absent — an empty label is a
    labelling accident, and treating it as a domain name would silently gather
    every half-labelled worker into one bogus domain.
    """
    for key in spec.label_keys:
        value = (labels or {}).get(key)
        if value:
            return value, key
    return None, None


def build_topology(
    specs: Sequence[TopologyLayerSpec],
    workers: Iterable,
) -> TopologyNode:
    """Build the tree. Declaration errors raise; data gaps do not.

    ``workers`` needs only ``id``, ``name`` and ``labels``; it is typed loosely
    so the scheduler can pass ORM rows and the tests can pass stubs.
    """
    ordered = order_layers([s for s in specs if s.layer != NODE_LAYER])
    root = TopologyNode(layer=ROOT_LAYER, name=ROOT_LAYER)

    for worker in workers:
        worker_id = getattr(worker, "id", None)
        if worker_id is None:
            continue
        labels = effective_topology_labels(worker)

        parent = root
        for spec in ordered:
            name, matched = _domain_of(labels, spec)
            if name is None:
                name = UNCLASSIFIED
            parent = _child(parent, spec.layer, name, matched)

        # The leaf is the worker itself, keyed by name rather than by any
        # label: this is the layer that must never collapse.
        leaf_name = getattr(worker, "name", None) or str(worker_id)
        leaf = _child(parent, NODE_LAYER, leaf_name, None)
        leaf.worker_ids.append(worker_id)

    return root


def _child(
    parent: TopologyNode, layer: str, name: str, matched_label_key: Optional[str]
) -> TopologyNode:
    for existing in parent.children:
        if existing.layer == layer and existing.name == name:
            return existing
    node = TopologyNode(
        layer=layer, name=name, parent=parent, matched_label_key=matched_label_key
    )
    parent.children.append(node)
    return node


def group_by_domain(
    workers: Iterable,
    label_keys: Sequence[str],
    sub_domain_keys: Sequence[str] = (),
) -> List[TopologyNode]:
    """Group workers by accelerator domain, flat, beside the tree.

    A domain is the set of workers whose accelerators can address each other's
    memory (NVLink, HCCS, UB). It nests nowhere fixed in the tree — inside a
    host on an 8-card server, across sixteen racks on a CloudMatrix384 — so it
    is not a layer of it. Each returned node is one domain holding leaf nodes
    for its workers; workers with no domain value are collected under the
    unclassified bucket, which the solver excludes like any other.

    With ``sub_domain_keys`` the grouping is by *(domain, sub-domain)* pair
    instead, and only workers carrying both values take part. The pair, not
    the sub-domain value alone: a rack named ``R1`` may exist in two super pods,
    and only the one inside the same domain is "closer".
    """
    layer = (
        ACCELERATOR_SUB_DOMAIN_LAYER if sub_domain_keys else ACCELERATOR_DOMAIN_LAYER
    )
    root = TopologyNode(layer=ROOT_LAYER, name=ROOT_LAYER)
    domain_spec = TopologyLayerSpec(layer=layer, label_keys=tuple(label_keys))
    sub_spec = TopologyLayerSpec(layer=layer, label_keys=tuple(sub_domain_keys))

    for worker in workers:
        worker_id = getattr(worker, "id", None)
        if worker_id is None:
            continue
        labels = effective_topology_labels(worker)
        name, matched = _domain_of(labels, domain_spec)
        if name is not None and sub_domain_keys:
            sub_name, _ = _domain_of(labels, sub_spec)
            if sub_name is None:
                # No sub-domain value: this worker is not "closer" to anyone
                # at this scope. It still counts at the domain scope.
                continue
            name = f"{name}/{sub_name}"
        if name is None:
            name = UNCLASSIFIED
        domain = _child(root, layer, name, matched)
        leaf_name = getattr(worker, "name", None) or str(worker_id)
        leaf = _child(domain, NODE_LAYER, leaf_name, None)
        leaf.worker_ids.append(worker_id)

    return root.children


def layer_names(specs: Sequence[TopologyLayerSpec]) -> List[str]:
    """Root-to-leaf layer names, leaf included.

    What the deployment form's "at least in the same ___" choices are built
    from. The leaf is appended unconditionally, which is why the tightest
    choice is offered even by a cluster that declared no topology at all.
    """
    return [s.layer for s in order_layers(list(specs))] + [NODE_LAYER]


def common_layer(a: TopologyNode, b: TopologyNode) -> Optional[str]:
    """The tightest layer whose domain contains both, or None.

    Two workers in the same unclassified bucket are *not* treated as close:
    the bucket means "we do not know where these are", and reading that as
    "these are together" would turn missing labels into confident wrong
    answers. This is the one place where the unclassified bucket behaves
    differently from a real domain.
    """
    # Walk b upward and stop at the first node a also sits under. By identity
    # rather than by position, so the answer does not depend on the two chains
    # having equal depth.
    ancestors_of_a = {id(node) for node in _ancestry(a)}
    for node in reversed(_ancestry(b)):
        if id(node) not in ancestors_of_a:
            continue
        if node.layer == ROOT_LAYER:
            # Sharing only the root is sharing nothing: every worker in the
            # cluster is under it.
            return None
        if node.is_unclassified:
            return None
        return node.layer
    return None


def _ancestry(node: TopologyNode) -> List[TopologyNode]:
    """Root-to-node, inclusive."""
    chain: List[TopologyNode] = []
    current: Optional[TopologyNode] = node
    while current is not None:
        chain.append(current)
        current = current.parent
    chain.reverse()
    return chain


def nodes_at_layer(root: TopologyNode, layer: str) -> List[TopologyNode]:
    """Every domain at one layer, in declaration order."""
    if root.layer == layer:
        return [root]
    out: List[TopologyNode] = []
    for child in root.children:
        out.extend(nodes_at_layer(child, layer))
    return out


def unclassified_at(root: TopologyNode, layer: str) -> List[int]:
    """Workers that fell into the unclassified bucket of one layer.

    The number the topology page leads with, because a worker landing here is
    the failure this whole module is most likely to hit and the one that
    reports nothing on its own.
    """
    out: List[int] = []
    for node in nodes_at_layer(root, layer):
        if node.is_unclassified:
            out.extend(node.descendant_worker_ids())
    return out
