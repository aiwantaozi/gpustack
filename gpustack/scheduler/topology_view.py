"""A cluster's topology, resolved against its fleet, for everyone who reads it.

The scheduler, the topology page and the deployment form's feasibility check
all need the same answer to "what does this cluster's declaration do to these
workers": which vocabulary fields are in use, the tree they produce, the
accelerator domains beside it, and the ordered scopes the solver walks. This
is that one computation, so the three can never disagree about the same
cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from gpustack.scheduler.group_solver import GatherScope, tree_scopes
from gpustack.scheduler.topology import (
    NODE_LAYER,
    TopologyLayerSpec,
    TopologyNode,
    build_topology,
    effective_topology_labels,
    group_by_domain,
    layer_names,
)
from gpustack.scheduler.topology_vocabulary import (
    ACCELERATOR_DOMAIN,
    ACCELERATOR_SUB_DOMAIN,
    SWITCH_NAME_KEY,
    ResolvedLayer,
    ResolvedTopology,
    source_of,
    validate_declaration,
)


@dataclass
class Location:
    """One worker's value for one field, and where it came from."""

    value: str
    source: str
    key: str
    discovered_value: Optional[str] = None
    display: Optional[str] = None


@dataclass
class TopologyView:
    resolved: ResolvedTopology
    workers: List[object]
    active: List[ResolvedLayer]
    specs: List[TopologyLayerSpec]
    root: TopologyNode
    layers: List[str]
    """Root-to-leaf, leaf included — the tree's layers."""
    domains: List[TopologyNode]
    sub_domains: List[TopologyNode]
    locations: Dict[int, Dict[str, Location]] = field(default_factory=dict)

    @property
    def has_domains(self) -> bool:
        return any(not d.is_unclassified for d in self.domains)

    @property
    def has_sub_domains(self) -> bool:
        return bool(self.sub_domains)

    def scopes(self) -> List[GatherScope]:
        """The solver's search order, tightest first.

        Host, then the accelerator sub-domain and domain, then the tree's
        layers up from the switch. The domain sits between host and the tree
        because inside it the transfer runs over the accelerator fabric — an
        order of magnitude faster than any switch hop, whatever the domain's
        physical extent. Scopes with nothing in them are left out so the
        deployment form does not offer a tier no group could satisfy.
        """
        tree = tree_scopes(self.root, self.layers)
        host, above = tree[0], tree[1:]
        out = [host]
        if self.has_sub_domains:
            out.append(GatherScope(ACCELERATOR_SUB_DOMAIN, self.sub_domains))
        if self.has_domains:
            out.append(GatherScope(ACCELERATOR_DOMAIN, self.domains))
        for scope in above:
            if any(not d.is_unclassified for d in scope.domains):
                out.append(scope)
        return out

    def tier_names(self) -> List[str]:
        """The scopes the deployment form may ask for, tightest first.

        The sub-domain is a search refinement, not a choice: "at least the
        same rack" already carries its physical meaning.
        """
        return [s.name for s in self.scopes() if s.name != ACCELERATOR_SUB_DOMAIN]

    def unclassified_at(self, layer_id: str) -> List[int]:
        """Workers with no value at this layer, across every bucket.

        The tree has one unclassified bucket *per parent* — the workers with a
        zone but no rack sit under their zone, the ones with neither sit under
        the zone-level bucket — so the answer is the union, not the first hit.
        """
        out: List[int] = []
        for node in self._nodes(layer_id):
            if node.is_unclassified:
                out.extend(node.descendant_worker_ids())
        return out

    def domain_count(self, layer_id: str) -> int:
        return len([n for n in self._nodes(layer_id) if not n.is_unclassified])

    def _nodes(self, layer_id: str) -> List[TopologyNode]:
        if layer_id == ACCELERATOR_DOMAIN:
            return self.domains
        from gpustack.scheduler.topology import nodes_at_layer

        return nodes_at_layer(self.root, layer_id)


def build_view(topology, workers: Iterable) -> TopologyView:
    """Resolve ``topology`` (a ``ClusterTopology`` or None) over ``workers``.

    Raises ``TopologyError`` only for a declaration that cannot become a tree;
    a worker missing a value is a normal state with a place in the result.
    """
    workers = list(workers)
    resolved = validate_declaration(topology)
    active = resolved.active(workers)
    specs = resolved.specs(active)
    root = build_topology(specs, workers)
    layers = layer_names(specs)
    domains = group_by_domain(workers, resolved.domain.label_keys)
    sub_domains = (
        group_by_domain(
            workers, resolved.domain.label_keys, resolved.domain.sub_domain_keys
        )
        if resolved.domain.sub_domain_keys
        else []
    )
    view = TopologyView(
        resolved=resolved,
        workers=workers,
        active=active,
        specs=specs,
        root=root,
        layers=layers,
        domains=domains,
        sub_domains=sub_domains,
    )
    view.locations = {
        w.id: _locations_of(w, resolved)
        for w in workers
        if getattr(w, "id", None) is not None
    }
    return view


def _locations_of(worker, resolved: ResolvedTopology) -> Dict[str, Location]:
    """Every field this worker has a value for, hand-filled or discovered."""
    merged = effective_topology_labels(worker)
    facts = getattr(getattr(worker, "status", None), "topology_facts", None) or {}
    out: Dict[str, Location] = {}

    fields: List[tuple] = [(layer.id, layer.label_keys) for layer in resolved.chain]
    fields.append((ACCELERATOR_DOMAIN, resolved.domain.label_keys))
    for field_id, keys in fields:
        for key in keys:
            value = merged.get(key)
            if not value:
                continue
            discovered = next((facts[k] for k in keys if facts.get(k)), None)
            display = (
                facts.get(SWITCH_NAME_KEY)
                if field_id == "switch" and key in facts
                else None
            )
            out[field_id] = Location(
                value=value,
                source=source_of(worker, key),
                key=key,
                discovered_value=(
                    discovered
                    if discovered != value or source_of(worker, key) == "user"
                    else None
                ),
                display=display,
            )
            break
    return out


def leaf_layer() -> str:
    return NODE_LAYER
