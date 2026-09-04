"""Where this cluster's workers are, and the two ways of changing that.

**The page opens on a filled-in table, not an empty form.** `GET /topology`
returns every field of the vocabulary with whether it is in use, every worker
with the values it has (hand-filled or discovered by its runtime) and where
each came from, the tree those values produce, and the accelerator domains
beside it — one request, so the table, the tree and the overview bar cannot
disagree.

**Filling in a value is writing a label.** `POST /topology/locations` sets one
field on a batch of workers by writing the field's own key
(`topology.gpustack.ai/rack`) into `Worker.labels`. Nothing else is stored:
the position is the label, `worker_selector` can read it, and clearing it
uncovers whatever the worker discovered on its own. The response carries the
inverse assignments so the UI's undo is the same call with the previous values.

**Changing the mapping is a different act.** Which keys a field reads from is
the Advanced panel's business and goes through `PUT /clusters/{id}` after a
`POST /topology/preview` of the unsaved mapping. Values take effect at once;
mappings are previewed and saved. The two are kept apart on purpose.

**Counts, not just names.** Every domain carries its worker, GPU and free-GPU
totals, because "which rack can hold my 2P2D" is the question, and the
unclassified bucket carries the worker ids behind it so "3 workers have no
rack yet" is one click away from being fixed.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from gpustack.api.exceptions import BadRequestException
from gpustack.api.tenant import assert_cluster_visible, assert_org_owned_writable
from gpustack.schemas.clusters import Cluster, ClusterTopology
from gpustack.schemas.workers import Worker
from gpustack.scheduler.topology import (
    NODE_LAYER,
    TopologyError,
    TopologyNode,
)
from gpustack.scheduler.topology_view import TopologyView, build_view
from gpustack.scheduler.topology_vocabulary import (
    ACCELERATOR_DOMAIN,
    KNOWN_KEYS,
    VOCABULARY_IDS,
    display_name,
    primary_key_for,
)
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# The view: what the page reads.                                              #
# --------------------------------------------------------------------------- #


class VocabularyFieldPublic(BaseModel):
    id: str
    name: str


class KnownKeyPublic(BaseModel):
    key: str
    vendor: str
    fits: List[str]
    note: str = ""


class VocabularyPublic(BaseModel):
    fields: List[VocabularyFieldPublic]
    known_keys: List[KnownKeyPublic]


class TopologyLayerPublic(BaseModel):
    """One field of the vocabulary or one custom layer, with whether the fleet
    uses it. Non-active fields are what the "add a field" menu offers."""

    id: str
    name: str
    builtin: bool
    active: bool
    label_keys: List[str] = []
    primary_key: Optional[str] = None
    domains: int = 0
    classified: int = 0
    unclassified: int = 0
    referenced_by_models: List[str] = []
    """Models whose `gather.layer` names this layer. A custom layer with
    references cannot be deleted without stranding them, and the Advanced
    panel says which."""


class AcceleratorDomainPublic(BaseModel):
    active: bool
    domains: int = 0
    classified: int = 0
    unclassified: int = 0
    sub_classified: int = 0
    """Workers with a domain *and* a sub-domain value: the "N / M" the
    Advanced panel shows beside the sub-domain picker."""
    label_keys: List[str] = []
    sub_domain_keys: List[str] = []
    sub_domain_field: Optional[str] = None
    """The vocabulary field whose keys `sub_domain_keys` are, when they are
    exactly one field's — the Advanced panel shows a field picker, not a key
    list, and this is how it knows which option is selected."""


class LocationPublic(BaseModel):
    value: str
    source: str
    """`user` (hand-filled label) or `discovered` (the worker's runtime)."""
    key: str
    discovered_value: Optional[str] = None
    """What the worker discovered, when a hand-filled value hides it; the UI
    says "clearing this restores nvl-a"."""
    display: Optional[str] = None
    """A readable name for an opaque value — the switch's system name beside
    its chassis id."""


class TopologyWorkerPublic(BaseModel):
    id: int
    name: str
    state: Optional[str] = None
    gpus: int = 0
    free_gpus: int = 0
    location: Dict[str, LocationPublic] = {}
    labels: Dict[str, str] = {}
    """The worker's own labels, so the Advanced panel can count who carries a
    key being typed without a second request."""


class TopologyDomainPublic(BaseModel):
    """One node of the tree, with the numbers the page leads with."""

    layer: str
    name: str
    unclassified: bool = False
    matched_label_key: Optional[str] = None
    workers: int = 0
    gpus: int = 0
    free_gpus: int = 0
    worker_ids: List[int] = []
    """Only populated on the unclassified bucket and the leaf: on the bucket it
    is what turns "3 workers have no rack" into one bulk action."""
    accelerator_domains: List[str] = []
    """The domains that appear under this node. Two or more in one rack is the
    thing an operator checks the tree for."""
    children: List["TopologyDomainPublic"] = []


class SuggestionPublic(BaseModel):
    key: str
    workers: int
    distinct_values: int
    looks_like: Optional[str] = None


class TopologyViewPublic(BaseModel):
    vocabulary: VocabularyPublic
    layers: List[TopologyLayerPublic]
    """Root-to-leaf, the leaf (`NodeTopologyLayer`) last and always active."""
    accelerator_domain: AcceleratorDomainPublic
    workers: List[TopologyWorkerPublic]
    tree: TopologyDomainPublic
    total_workers: int = 0
    unclassified_workers: int = 0
    """Workers with no value at any *active* layer, deduplicated: the number
    the overview bar leads with."""
    suggestions: List[SuggestionPublic] = []


@router.get("/{id}/topology", response_model=TopologyViewPublic)
async def get_cluster_topology(session: SessionDep, ctx: TenantContextDep, id: int):
    """The saved mapping over the live fleet."""
    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")
    workers = await Worker.all_by_field(session, "cluster_id", id)
    return await _view_public(
        cluster.topology, workers, await _gather_references(session, id)
    )


async def _gather_references(session, cluster_id: int) -> Dict[str, List[str]]:
    """layer id -> names of models whose `gather.layer` names it."""
    from gpustack.schemas.models import Model

    out: Dict[str, List[str]] = {}
    for model in await Model.all_by_field(session, "cluster_id", cluster_id):
        gather = getattr(model, "gather", None)
        layer = getattr(gather, "layer", None)
        if layer:
            out.setdefault(layer, []).append(model.name)
    return out


class TopologyPreviewRequest(BaseModel):
    """The mapping to preview. Absent means the saved one."""

    topology: Optional[ClusterTopology] = None


@router.post("/{id}/topology/preview", response_model=TopologyViewPublic)
async def preview_cluster_topology(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: Optional[TopologyPreviewRequest] = None,
):
    """The view `body.topology` would produce, without saving it.

    The Advanced panel's loop is "add a key, see who comes out of the unfilled
    bucket", and making that a save against a live cluster would turn a
    keystroke into a commitment. An invalid mapping is a 400, never a stored
    mistake; a worker missing a value is a normal state with a place in the
    result.
    """
    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")
    topology = body.topology if body and body.topology is not None else cluster.topology
    workers = await Worker.all_by_field(session, "cluster_id", id)
    return await _view_public(topology, workers, await _gather_references(session, id))


async def _view_public(
    topology, workers, gather_refs: Optional[Dict[str, List[str]]] = None
) -> TopologyViewPublic:
    gather_refs = gather_refs or {}
    try:
        view = build_view(topology, workers)
    except TopologyError as e:
        raise BadRequestException(message=str(e))

    capacity = await _worker_capacity(workers)
    tree = _to_public(view.root, capacity, view)
    by_id = {w.id: w for w in workers}

    layers: List[TopologyLayerPublic] = []
    active_ids = {layer.id for layer in view.active}
    for layer in view.resolved.chain:
        active = layer.id in active_ids
        classified = sum(1 for locs in view.locations.values() if layer.id in locs)
        layers.append(
            TopologyLayerPublic(
                id=layer.id,
                name=display_name(layer.id),
                builtin=layer.builtin,
                active=active,
                label_keys=list(layer.label_keys),
                primary_key=layer.primary_key,
                domains=view.domain_count(layer.id) if active else 0,
                classified=classified,
                unclassified=len(workers) - classified,
                referenced_by_models=sorted(gather_refs.get(layer.id, [])),
            )
        )
    layers.append(
        TopologyLayerPublic(
            id=NODE_LAYER,
            name=display_name(NODE_LAYER),
            builtin=True,
            active=True,
            domains=len(workers),
            classified=len(workers),
            referenced_by_models=sorted(gather_refs.get(NODE_LAYER, [])),
        )
    )

    domain_classified = sum(
        1 for locs in view.locations.values() if ACCELERATOR_DOMAIN in locs
    )
    sub_keys = list(view.resolved.domain.sub_domain_keys)
    sub_field = next(
        (
            layer.id
            for layer in view.resolved.chain
            if sub_keys and set(sub_keys) <= set(layer.label_keys)
        ),
        None,
    )
    sub_classified = sum(
        len(node.descendant_worker_ids())
        for node in view.sub_domains
        if not node.is_unclassified
    )
    domain_public = AcceleratorDomainPublic(
        active=view.has_domains,
        domains=view.domain_count(ACCELERATOR_DOMAIN),
        classified=domain_classified,
        unclassified=len(workers) - domain_classified,
        sub_classified=sub_classified,
        label_keys=list(view.resolved.domain.label_keys),
        sub_domain_keys=sub_keys,
        sub_domain_field=sub_field,
    )

    workers_public = [
        TopologyWorkerPublic(
            id=w.id,
            name=getattr(w, "name", None) or str(w.id),
            state=_state_of(w),
            gpus=capacity.get(w.id, _Capacity()).gpus,
            free_gpus=capacity.get(w.id, _Capacity()).free_gpus,
            location={
                field_id: LocationPublic(
                    value=loc.value,
                    source=loc.source,
                    key=loc.key,
                    discovered_value=loc.discovered_value,
                    display=loc.display,
                )
                for field_id, loc in view.locations.get(w.id, {}).items()
            },
            labels=dict(getattr(w, "labels", None) or {}),
        )
        for w in workers
        if getattr(w, "id", None) is not None
    ]

    # Unfilled at any active tree layer, deduplicated: one worker missing two
    # fields is one worker to go and fill in, not two problems.
    unfilled = set()
    for layer in view.active:
        unfilled |= set(view.unclassified_at(layer.id))

    return TopologyViewPublic(
        vocabulary=VocabularyPublic(
            fields=[
                VocabularyFieldPublic(id=i, name=display_name(i))
                for i in VOCABULARY_IDS
            ],
            known_keys=[
                KnownKeyPublic(
                    key=k.key, vendor=k.vendor, fits=list(k.fits), note=k.note
                )
                for k in KNOWN_KEYS
            ],
        ),
        layers=layers,
        accelerator_domain=domain_public,
        workers=workers_public,
        tree=tree,
        total_workers=len(by_id),
        unclassified_workers=len(unfilled),
    )


def _state_of(worker) -> Optional[str]:
    state = getattr(worker, "state", None)
    return getattr(state, "value", state) if state is not None else None


class _Capacity(BaseModel):
    gpus: int = 0
    free_gpus: int = 0


async def _worker_capacity(workers) -> Dict[int, _Capacity]:
    """Per worker: how many GPUs it has, and how many carry nothing.

    Allocation comes from ``get_worker_allocated``, which derives it from the
    current model-instance bindings — the same single source of truth the
    scheduler and the workers API read. Deriving it here from anything a worker
    self-reports would let the preview and the scheduler disagree about the
    same rack.
    """
    from gpustack.server.worker_allocated_cache import get_worker_allocated

    out: Dict[int, _Capacity] = {}
    for worker in workers:
        devices = (
            (getattr(worker.status, "gpu_devices", None) or []) if worker.status else []
        )
        indexes = [d.index for d in devices if d.index is not None]
        try:
            allocated = await get_worker_allocated(worker.id)
            used = {
                index
                for index, vram in (getattr(allocated, "vram", None) or {}).items()
                if vram
            }
        except Exception as e:
            # A worker whose allocation cannot be read is reported as fully
            # used rather than fully free: the preview's job is to answer "can
            # my group fit here", and an optimistic guess is the one answer
            # that sends an operator to a rack that cannot take the group.
            logger.warning(
                "Could not read allocation for worker %s; counting its GPUs as "
                "used in the topology preview: %s",
                getattr(worker, "name", worker.id),
                e,
            )
            used = set(indexes)
        out[worker.id] = _Capacity(
            gpus=len(indexes),
            free_gpus=len([i for i in indexes if i not in used]),
        )
    return out


def _to_public(
    node: TopologyNode, capacity: Dict[int, _Capacity], view: TopologyView
) -> TopologyDomainPublic:
    children = [_to_public(child, capacity, view) for child in node.children]
    worker_ids = node.descendant_worker_ids()
    is_leaf = node.layer == NODE_LAYER

    if children:
        workers = sum(child.workers for child in children)
        gpus = sum(child.gpus for child in children)
        free_gpus = sum(child.free_gpus for child in children)
    else:
        workers = len(worker_ids)
        gpus = sum(capacity.get(wid, _Capacity()).gpus for wid in worker_ids)
        free_gpus = sum(capacity.get(wid, _Capacity()).free_gpus for wid in worker_ids)

    domains = sorted(
        {
            view.locations[wid][ACCELERATOR_DOMAIN].value
            for wid in worker_ids
            if wid in view.locations and ACCELERATOR_DOMAIN in view.locations[wid]
        }
    )

    return TopologyDomainPublic(
        layer=node.layer,
        name=node.name,
        unclassified=node.is_unclassified,
        matched_label_key=node.matched_label_key,
        workers=workers,
        gpus=gpus,
        free_gpus=free_gpus,
        worker_ids=worker_ids if (node.is_unclassified or is_leaf) else [],
        accelerator_domains=domains,
        children=children,
    )


# --------------------------------------------------------------------------- #
# Locations: filling a field in.                                              #
# --------------------------------------------------------------------------- #


class LocationAssignment(BaseModel):
    worker_ids: List[int]
    layer: str
    """A vocabulary field id, `accelerator_domain`, or a custom layer's name."""
    value: Optional[str] = None
    """None clears the field's own key. Other sources' keys are never touched,
    which is what lets a cleared hand-filled value uncover a discovered one."""


class LocationsRequest(BaseModel):
    assignments: List[LocationAssignment]


class LocationsPublic(BaseModel):
    previous: List[LocationAssignment]
    """The inverse of what was applied: POST it back to undo."""
    topology: TopologyViewPublic


@router.post("/{id}/topology/locations", response_model=LocationsPublic)
async def set_cluster_topology_locations(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: LocationsRequest,
):
    """Set one field on a batch of workers, by writing the field's own key.

    Effective at once — position is a label, and a label change only affects
    scheduling from here on; nothing running is moved. Refused only for a
    field the cluster does not have or a worker outside the cluster: there is
    no such thing as an invalid rack name.
    """
    from gpustack.server.services import WorkerService

    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")
    assert_org_owned_writable(ctx, cluster, resource_label="cluster")

    workers = await Worker.all_by_field(session, "cluster_id", id)
    by_id = {w.id: w for w in workers}
    resolved_view = build_view(cluster.topology, workers)

    # Resolve every assignment before writing any, so a bad one refuses the
    # whole batch instead of leaving half a rack renamed.
    planned: List[tuple] = []
    for assignment in body.assignments:
        key = primary_key_for(resolved_view.resolved, assignment.layer)
        if key is None:
            raise BadRequestException(
                message=f"{assignment.layer!r} is not a field that can be filled in."
            )
        for worker_id in assignment.worker_ids:
            worker = by_id.get(worker_id)
            if worker is None:
                raise BadRequestException(
                    message=f"worker {worker_id} is not in cluster {id}."
                )
            planned.append(
                (
                    worker,
                    assignment.layer,
                    key,
                    (assignment.value or "").strip() or None,
                )
            )

    previous: Dict[tuple, Dict[int, Optional[str]]] = {}
    service = WorkerService(session)
    for worker, layer, key, value in planned:
        labels = dict(worker.labels or {})
        before = labels.get(key)
        if value is None:
            labels.pop(key, None)
        else:
            labels[key] = value
        if labels == (worker.labels or {}):
            continue
        await service.update(worker, {"labels": labels})
        previous.setdefault((layer, before), {})[worker.id] = before

    # One inverse assignment per (field, previous value): the undo of "these
    # three got R3" is "these two get back R1 and that one gets cleared".
    inverse = [
        LocationAssignment(worker_ids=sorted(ids), layer=layer, value=value)
        for (layer, value), ids in previous.items()
        if ids
    ]

    refreshed = await Worker.all_by_field(session, "cluster_id", id)
    return LocationsPublic(
        previous=inverse,
        topology=await _view_public(
            cluster.topology, refreshed, await _gather_references(session, id)
        ),
    )


# --------------------------------------------------------------------------- #
# Gather feasibility: what the deployment form's tiers each mean here.        #
# --------------------------------------------------------------------------- #


class GatherFeasibilityRequest(BaseModel):
    """The group as the form currently has it.

    A whole model spec rather than a role/replica summary, because capacity is
    decided by the resource-fit selectors and those read the backend, the
    engine parameters, the GPU selector and the per-role overrides. Sending a
    summary would mean re-deriving all of that server-side from fewer facts
    than the form already holds, and the estimate would then disagree with the
    real scheduling — which is the one thing a feasibility answer must not do.
    """

    model_spec: Dict = {}


class GatherTierPublic(BaseModel):
    """One "at least in the same ___" choice, and whether it would deploy."""

    layer: str
    name: str = ""
    feasible: bool
    domain: Optional[str] = None
    reason: Optional[str] = None
    best_domain: Optional[str] = None
    needed: int = 0
    available: int = 0
    unmeasured: int = 0
    """Workers whose capacity could not be established. Non-zero makes
    `available` a floor rather than a measurement, and the form must not
    present a floor as a capacity verdict."""


class GatherFeasibilityPublic(BaseModel):
    tiers: List[GatherTierPublic] = []
    """Tightest first: host, then the accelerator domain when the fleet has
    one, then the tree's active layers."""

    prefer: Optional[GatherTierPublic] = None
    """The "as close as possible, deploy anyway" option, carried separately:
    it is a different question (can this deploy at all)."""


@router.post(
    "/{id}/topology/gather-feasibility", response_model=GatherFeasibilityPublic
)
async def gather_feasibility(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: GatherFeasibilityRequest,
):
    """Whether this group would deploy at each gather tier, right now.

    One solve per tier against live capacity rather than an estimate, and every
    tier in one request: the alternative is a round trip per option while the
    dropdown renders.
    """
    from gpustack.config.config import get_global_config
    from gpustack.schemas.models import Model, ModelInstance, RoleSpec
    from gpustack.scheduler.group_capacity import GroupCapacity, role_demands
    from gpustack.scheduler.group_solver import (
        GatherRequest,
        GroupPlacement,
        RoleDemand,
        solve_group_placement,
    )

    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")

    # Only the fields a Model actually has: the form posts its whole state and
    # a stray key would be a 500 where the honest answer is "ignored".
    spec = {k: v for k, v in (body.model_spec or {}).items() if k in Model.model_fields}
    # 🔴 Coerced by hand, and only the two fields the capacity walk reads
    # structurally. `Model(**spec)` cannot be trusted — SQLModel skips
    # validation on `table=True` classes — but validating the whole spec
    # through `ModelSpecBase` is too strict: it requires `source`, and this
    # endpoint is called from a form that is still being filled.
    try:
        model = Model(**_widen_single_selects(_drop_unset_configs(spec)))
        model.replicas = int(spec.get("replicas") or 1)
        model.roles = [RoleSpec.model_validate(r) for r in (spec.get("roles") or [])]
    except Exception as e:
        raise BadRequestException(message=f"invalid model spec: {e}")
    model.cluster_id = id

    workers = await Worker.all_by_field(session, "cluster_id", id)
    try:
        view = build_view(cluster.topology, workers)
    except TopologyError as e:
        raise BadRequestException(message=str(e))
    scopes = view.scopes()
    tiers_wanted = view.tier_names()

    # 🔴 The edge, and it has to be here rather than one frame deeper:
    # `role_demands` revalidates the whole model, so a form value the
    # constructor waved through surfaces here as a 500 from inside the
    # scheduler unless it is turned into a 400 that names the field.
    try:
        demands = [RoleDemand(**d) for d in role_demands(model)]
    except Exception as e:
        raise BadRequestException(message=f"invalid model spec: {e}")
    if not demands:
        # Nothing in this group occupies an accelerator, so no domain
        # constrains it. Saying so is more honest than solving for it.
        return GatherFeasibilityPublic(
            tiers=[
                GatherTierPublic(layer=name, name=display_name(name), feasible=True)
                for name in tiers_wanted
            ],
            prefer=GatherTierPublic(
                layer=NODE_LAYER, name=display_name(NODE_LAYER), feasible=True
            ),
        )

    # Fleet-wide, like the scheduler's own read: allocation is derived from
    # every binding, and a distributed instance consumes VRAM on workers
    # besides the one it is filed under.
    instances = await ModelInstance.all(session)
    capacity = GroupCapacity(get_global_config(), model, workers, instances)

    tiers: List[GatherTierPublic] = []
    for name in tiers_wanted:
        result = await solve_group_placement(
            view.root, demands, capacity, scopes, GatherRequest(layer=name, must=True)
        )
        tiers.append(_tier(name, result, GroupPlacement))

    prefer_result = await solve_group_placement(
        view.root, demands, capacity, scopes, GatherRequest()
    )
    return GatherFeasibilityPublic(
        tiers=tiers,
        prefer=_tier(NODE_LAYER, prefer_result, GroupPlacement),
    )


def _widen_single_selects(spec: Dict) -> Dict:
    """Wrap a scalar into a list where the field is declared as one.

    🔴 Not a leniency, and not guessing: this is the exact transform the
    deployment form itself applies on submit. A single-select control bound to
    a list-valued field holds one value while the form is open
    (`categories: "llm"`) and is wrapped at the last moment
    (`data.categories = data.categories ? [data.categories] : []`). This
    endpoint is fed the form's *live* state, so it necessarily sees the
    unwrapped shape — and refusing it would mean the preview only works after
    submit, which is the one moment it is useless.

    Applied by reading each field's own annotation rather than by naming
    `categories`, so a second single-select bound to a list field does not
    reintroduce the same 500.
    """
    from gpustack.schemas.models import Model

    widened = dict(spec)
    for name, value in spec.items():
        if isinstance(value, (list, tuple, set, dict)):
            continue
        field = Model.model_fields.get(name)
        if field is None or not _declares_a_list(field.annotation):
            continue
        # A cleared multi-select posts `null`, and the role projection reads a
        # list field as a list: `categories: null` was a 400 on every keystroke
        # of a form that had not picked a category yet.
        widened[name] = [] if value is None else [value]
    return widened


def _drop_unset_configs(spec: Dict) -> Dict:
    """Leave out a nested config the form carries but has not filled in.

    The form keeps `speculative_config` as an object with an empty `algorithm`
    while speculative decoding is off; the role projection validates that
    enum strictly and the whole feasibility answer became a 400 for every PD
    form that had not touched speculative decoding.
    """
    cleaned = dict(spec)
    speculative = cleaned.get("speculative_config")
    if isinstance(speculative, dict) and not speculative.get("algorithm"):
        cleaned.pop("speculative_config")
    return cleaned


def _declares_a_list(annotation) -> bool:
    """Whether this annotation accepts a list, `Optional[List[...]]` included.

    Recurses through unions rather than testing the outer origin: the field
    that produced the crash is a bare `List[str]`, but the next one will be
    `Optional[List[str]]` and an outer-origin check reads that as a union and
    stops.
    """
    import typing

    origin = typing.get_origin(annotation)
    if origin in (list, set, tuple, frozenset):
        return True
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        return any(_declares_a_list(arg) for arg in typing.get_args(annotation))
    return False


def _tier(layer: str, result, placement_cls) -> GatherTierPublic:
    if isinstance(result, placement_cls):
        return GatherTierPublic(
            layer=layer,
            name=display_name(layer),
            feasible=True,
            domain=result.domain or None,
        )
    return GatherTierPublic(
        layer=layer,
        name=display_name(layer),
        feasible=False,
        reason=result.reason,
        best_domain=result.best_domain or None,
        needed=result.needed,
        available=result.available,
        unmeasured=result.unmeasured,
    )
