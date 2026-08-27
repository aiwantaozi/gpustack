"""What a topology declaration actually does to this cluster's fleet.

The scheduler side of topology has been complete for a while
(``gpustack.scheduler.topology``) and had no consumer: nothing in the product
could show an operator the tree their labels produce. This is that consumer.

**The preview takes a declaration in the request body, and that is the whole
point.** A GET over the saved topology can only show the tree the operator
already committed to, while the one question they are actually asking is "what
happens if this layer matches `topology.kubernetes.io/zone` instead" — asked
*before* saving. Making them save to find out turns a one-keystroke experiment
into a two-step wizard against a live cluster, which is exactly the shape the
design rules out.

**Counts, not just names.** The tree is not the answer either; "which rack can
hold my 2P2D" is. So every domain carries its worker, GPU and free-GPU totals,
and the unclassified bucket carries the worker ids behind it — that bucket is
the failure this whole feature is most likely to hit, and it reports nothing on
its own. Handing back the ids is what lets the page offer "label these 20" as
one click instead of a hunt.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from gpustack.api.exceptions import BadRequestException
from gpustack.api.tenant import assert_cluster_visible
from gpustack.schemas.clusters import Cluster, ClusterTopology
from gpustack.schemas.workers import Worker
from gpustack.scheduler.topology import (
    NODE_LAYER,
    TopologyError,
    TopologyLayerSpec,
    TopologyNode,
    build_topology,
    layer_names,
)
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()

logger = logging.getLogger(__name__)


class TopologyPreviewRequest(BaseModel):
    """The declaration to preview.

    ``topology`` absent means "the one that is saved", so the page can open on
    the committed state without special-casing its first render.
    """

    topology: Optional[ClusterTopology] = None


class TopologyDomainPublic(BaseModel):
    """One node of the tree, with the numbers the page leads with."""

    layer: str
    name: str
    unclassified: bool = False
    """Its label keys all missed. Kept as a flag rather than left to a name
    comparison because the sentinel is an implementation detail of the
    scheduler and the page renders this node differently — it is a prompt to
    act, not a domain."""

    matched_label_key: Optional[str] = None
    """Which of the layer's any-of keys actually matched here. The any-of list
    is the feature that lets a mixed fleet work at all, and without this the
    operator cannot tell which spelling won."""

    workers: int = 0
    gpus: int = 0
    free_gpus: int = 0
    """GPUs with nothing allocated on them. Derived from the model-instance
    bindings, the same source the scheduler and the workers API use, so this
    number cannot disagree with what the scheduler will find."""

    worker_ids: List[int] = []
    """Only populated on the unclassified bucket and the leaf. Everywhere else
    it would be a large list nobody reads; on the bucket it is what turns
    "20 workers are missing this label" into one bulk action."""

    children: List["TopologyDomainPublic"] = []


class TopologyPreviewPublic(BaseModel):
    layers: List[str]
    """Root-to-leaf, leaf included. This is also the list the deployment
    form's "at least in the same ___" choices are built from, which is why the
    leaf is always present: the tightest choice exists even for a cluster that
    declared nothing."""

    label_keys: Dict[str, List[str]] = {}
    """Layer name -> its declared any-of keys, so the page can say *which* key
    an unclassified worker is missing rather than only that it is missing
    something. Empty for the built-in leaf layer, which reads no label."""

    total_workers: int = 0
    unclassified_workers: int = 0
    """Across every layer, deduplicated: the single number the page leads
    with. A worker unclassified at two layers is one problem, not two."""

    root: TopologyDomainPublic


@router.post("/{id}/topology/preview", response_model=TopologyPreviewPublic)
async def preview_cluster_topology(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    body: Optional[TopologyPreviewRequest] = None,
):
    """The tree `body.topology` would produce over this cluster's workers.

    Read-only: nothing is saved, and an invalid declaration is a 400 rather
    than a stored mistake. Declaration errors are the only refusal — a worker
    missing a label is a normal state the tree has a place for, and rejecting
    that would make labelling a precondition for looking at the tree, which is
    backwards. The tree is how you find out who still needs labelling.
    """
    cluster = await Cluster.one_by_id(session, id)
    assert_cluster_visible(ctx, cluster, not_found_message=f"cluster {id} not found")

    topology = (body.topology if body else None) or cluster.topology
    specs = _layer_specs(topology)

    workers = await Worker.all_by_field(session, "cluster_id", id)

    try:
        root = build_topology(specs, workers)
        names = layer_names(specs)
    except TopologyError as e:
        raise BadRequestException(message=str(e))

    capacity = await _worker_capacity(workers)
    public = _to_public(root, capacity)

    return TopologyPreviewPublic(
        layers=names,
        label_keys={spec.layer: list(spec.label_keys) for spec in specs},
        total_workers=len(workers),
        unclassified_workers=len(_unclassified_ids(public)),
        root=public,
    )


def _layer_specs(topology: Optional[ClusterTopology]) -> List[TopologyLayerSpec]:
    """The stored declaration in the scheduler's own vocabulary.

    One conversion, used by both endpoints: the preview and the feasibility
    check must build the same tree from the same declaration or they would
    disagree about the same cluster in the same form.
    """
    return [
        TopologyLayerSpec(
            layer=layer.name,
            label_keys=list(layer.label_keys or []),
            parent_layer=layer.parent_layer,
        )
        for layer in (topology.layers if topology else [])
    ]


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
    node: TopologyNode, capacity: Dict[int, _Capacity]
) -> TopologyDomainPublic:
    children = [_to_public(child, capacity) for child in node.children]
    worker_ids = node.descendant_worker_ids()
    is_leaf = node.layer == NODE_LAYER

    if children:
        workers = sum(child.workers for child in children)
        gpus = sum(child.gpus for child in children)
        free_gpus = sum(child.free_gpus for child in children)
    else:
        # A leaf, or a domain whose workers hang directly off it.
        workers = len(worker_ids)
        gpus = sum(capacity.get(wid, _Capacity()).gpus for wid in worker_ids)
        free_gpus = sum(capacity.get(wid, _Capacity()).free_gpus for wid in worker_ids)

    return TopologyDomainPublic(
        layer=node.layer,
        name=node.name,
        unclassified=node.is_unclassified,
        matched_label_key=node.matched_label_key,
        workers=workers,
        gpus=gpus,
        free_gpus=free_gpus,
        # Carried only where it is acted on. Every other domain's membership is
        # already implied by its children.
        worker_ids=worker_ids if (node.is_unclassified or is_leaf) else [],
        children=children,
    )


def _unclassified_ids(node: TopologyDomainPublic) -> set:
    """Deduplicated across layers: one worker missing two labels is one
    worker to go and label, not two problems."""
    found = set(node.worker_ids) if node.unclassified else set()
    for child in node.children:
        found |= _unclassified_ids(child)
    return found


# --------------------------------------------------------------------------- #
# Gather feasibility: what the deployment form's three tiers each mean here.
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
    feasible: bool
    domain: Optional[str] = None
    """Where the group would land, when it fits."""

    reason: Optional[str] = None
    """Why not, in the solver's own words — "the roomiest rack holds 6" rather
    than "it does not fit". The number is the actionable part."""

    best_domain: Optional[str] = None
    """The roomiest domain it found. Naming it is what makes the refusal
    actionable: "rack-a is 2 cards short" tells an operator where to look."""

    needed: int = 0
    available: int = 0
    unmeasured: int = 0
    """Workers whose capacity could not be established. Non-zero makes
    `available` a floor rather than a measurement, and the form must not
    present a floor as a capacity verdict — "we could not look" and "there is
    no room" call for opposite reactions."""


class GatherFeasibilityPublic(BaseModel):
    tiers: List[GatherTierPublic] = []
    """Leaf-first: the tightest choice comes first because it is the one that
    exists unconditionally, whatever the cluster declared."""

    prefer: Optional[GatherTierPublic] = None
    """The "as close as possible, deploy anyway" option, carried separately
    rather than as a tier — it is a different question (can this deploy at
    all) and presenting it as a tier that always says yes would flatten the
    distinction the control exists to draw."""


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

    The three-tier control works because the layer names are an operator's own
    strings that a deployer may not understand, while "does it fit" is
    universal. That only holds if the answer is real, so this is one solve per
    tier against live capacity rather than an estimate.

    Every tier is answered in one request on purpose: the alternative is a
    round trip per option while the dropdown renders, and the design is
    explicit that this must not follow the form's keystrokes.
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
    # structurally. Two constraints meet here:
    #
    # `Model(**spec)` cannot be trusted — SQLModel skips validation on
    # `table=True` classes, so it accepts `replicas="not-a-number"` and leaves
    # `roles` as raw dicts, which then fails deep inside the capacity walk
    # with an AttributeError instead of at the edge with a 400.
    #
    # But validating the whole spec through `ModelSpecBase` is too strict: it
    # requires `source`, and this endpoint is called from a form that is still
    # being filled. Refusing to answer "would this fit" until every unrelated
    # field is complete defeats the point of a live feasibility check.
    try:
        model = Model(**_widen_single_selects(spec))
        model.replicas = int(spec.get("replicas") or 1)
        model.roles = [RoleSpec.model_validate(r) for r in (spec.get("roles") or [])]
    except Exception as e:
        raise BadRequestException(message=f"invalid model spec: {e}")
    model.cluster_id = id

    specs = _layer_specs(cluster.topology)
    workers = await Worker.all_by_field(session, "cluster_id", id)
    try:
        root = build_topology(specs, workers)
        names = layer_names(specs)
    except TopologyError as e:
        raise BadRequestException(message=str(e))

    # 🔴 The edge, and it has to be here rather than one frame deeper.
    # `role_demands` projects the model per role, and the projection
    # *revalidates the whole model* (`RoleEffectiveModel.model_validate`). So
    # every field has to be well-typed, not just the ones this endpoint reads —
    # a form value the constructor waved through surfaces here instead, as a
    # 500 from inside the scheduler. Turning it into a 400 that names the field
    # is the difference between "the preview is broken" and "this field is".
    try:
        demands = [RoleDemand(**d) for d in role_demands(model)]
    except Exception as e:
        raise BadRequestException(message=f"invalid model spec: {e}")
    if not demands:
        # Nothing in this group occupies an accelerator, so no domain
        # constrains it. Saying so is more honest than solving for it.
        return GatherFeasibilityPublic(
            tiers=[
                GatherTierPublic(layer=name, feasible=True) for name in reversed(names)
            ],
            prefer=GatherTierPublic(layer=names[0], feasible=True),
        )

    # Fleet-wide, like the scheduler's own read: allocation is derived from
    # every binding, and a distributed instance consumes VRAM on workers
    # besides the one it is filed under.
    instances = await ModelInstance.all(session)
    capacity = GroupCapacity(get_global_config(), model, workers, instances)

    tiers: List[GatherTierPublic] = []
    for name in reversed(names):
        result = await solve_group_placement(
            root, demands, capacity, names, GatherRequest(layer=name, must=True)
        )
        tiers.append(_tier(name, result, GroupPlacement))

    prefer_result = await solve_group_placement(
        root, demands, capacity, names, GatherRequest()
    )
    return GatherFeasibilityPublic(
        tiers=tiers,
        prefer=_tier(names[0], prefer_result, GroupPlacement),
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
        if value is None or isinstance(value, (list, tuple, set, dict)):
            continue
        field = Model.model_fields.get(name)
        if field is None:
            continue
        if _declares_a_list(field.annotation):
            widened[name] = [value]
    return widened


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
            layer=result.layer or layer,
            feasible=True,
            domain=result.domain or None,
        )
    return GatherTierPublic(
        layer=result.layer or layer,
        feasible=False,
        reason=result.reason,
        best_domain=result.best_domain or None,
        needed=result.needed,
        available=result.available,
        unmeasured=result.unmeasured,
    )
