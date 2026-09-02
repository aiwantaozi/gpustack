import logging
import math
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from fastapi.responses import RedirectResponse, StreamingResponse
from urllib.parse import urlencode
from gpustack_runtime.detector import ManufacturerEnum
from sqlalchemy.orm import selectinload
from sqlmodel import or_
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    InternalServerErrorException,
    BadRequestException,
    ForbiddenException,
    NotFoundException,
)
from gpustack.schemas.common import Pagination
from gpustack.schemas.inference_backend import is_custom_backend
from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstancesPublic,
    BackendEnum,
    ModelListParams,
)
from gpustack.schemas.cache_services import CacheService
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.gpu_instance_types import GPUInstanceType
from gpustack.schemas.workers import GPUDeviceStatus, Worker
from gpustack.utils.version import version_in_range
from gpustack.api.tenant import (
    TenantContext,
    bypass_tenant_filter,
    assert_cluster_visible,
    assert_resource_visible,
    cluster_scoped_system,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.server.db import async_session
from gpustack.server.workload_namespace import (
    placement_drifted,
    resolve_workload_namespace,
)
from gpustack.server.deps import (
    CurrentUserDep,
    ListParamsDep,
    SessionDep,
    TenantContextDep,
)
from gpustack.schemas.models import (
    PD_MODE_BACKENDS,
    LoraListEntry,
    PDModeEnum,
    RoleSpec,
    Model,
    ModelCreate,
    ModelSpecBase,
    ModelUpdate,
    ModelPublic,
    ModelsPublic,
    RoleNameEnum,
)
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.services import (
    ModelInstanceService,
    ModelService,
    WorkerService,
    revoke_model_access_cache,
)
from gpustack.server.controllers import model_spec_digest
from gpustack.server.scaling_scheduler import compute_desired_replicas
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.server.lora_adapters_discovery import list_adapters_for_base
from gpustack.server.lora_model_routes import (
    cleanup_orphan_lora_routes,
    create_lora_model_routes,
    is_lora_list_stale,
)
from gpustack.utils.command import find_int_parameter, find_parameter
from gpustack.utils.convert import safe_int
from gpustack.utils.gpu import parse_gpu_id
from gpustack.routes.model_common import (
    ModelStateFilterEnum,
    build_category_conditions,
    categories_filter,
    model_state_condition,
    model_state_stream_filter,
)
from gpustack.config.config import get_global_config
from gpustack.server.pd_metrics import PDMetricsPublic, collect_pd_metrics
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.server.prometheus_query import parse_window
from gpustack.utils.grafana import resolve_grafana_base_url
from gpustack.utils.lora_model_source import lora_route_name_for

router = APIRouter()

logger = logging.getLogger(__name__)


def _make_model_watch_filter(ctx, categories, state=None):
    """Watch-stream visibility: cluster-bound service accounts only see
    their own cluster's models; everyone keeps the categories and state
    filters. Predicates are pre-built so inactive filters cost nothing on
    the per-event hot path."""
    predicates = []
    if cluster_scoped_system(ctx):
        predicates.append(lambda data: scoped_cluster_row_visible(ctx, data))
    if state is not None:
        predicates.append(lambda data: model_state_stream_filter(data, state))
    if categories:
        predicates.append(lambda data: categories_filter(data, categories))

    def _visible(data) -> bool:
        for p in predicates:
            if not p(data):
                return False
        return True

    return _visible


@router.get("", response_model=ModelsPublic)
async def get_models(
    ctx: TenantContextDep,
    params: ModelListParams = Depends(),
    state: Optional[ModelStateFilterEnum] = Query(
        default=None,
        description="Filter by model state.",
    ),
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
    cluster_id: int = None,
    backend: Optional[str] = Query(None, description="Filter by backend."),
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {}
    if cluster_id:
        fields["cluster_id"] = cluster_id

    if backend:
        fields["backend"] = backend

    # Streaming uses field-equality only; scope by current org so non-admin
    # users never see cross-org rows via the live stream. Admin without an
    # explicit org context keeps the unfiltered cross-org stream. System
    # users (workers / cluster accounts) bypass owner scoping — they serve
    # every Org's models — but cluster-bound service accounts are narrowed
    # to their own cluster's rows below.
    if ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
        fields["owner_principal_id"] = ctx.current_principal_id

    if params.watch:
        return StreamingResponse(
            Model.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=_make_model_watch_filter(ctx, categories, state),
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = list(tenant_list_conditions(ctx, Model))
        if categories:
            conditions = build_category_conditions(session, Model, categories)
            extra_conditions.append(or_(*conditions))

        state_condition = model_state_condition(state)
        if state_condition is not None:
            extra_conditions.append(state_condition)

        order_by = params.order_by
        if order_by:
            # When sorting by "source", add additional sorting fields for deterministic ordering
            new_order_by = []
            for field, direction in order_by:
                new_order_by.append((field, direction))
                if field == "source":
                    new_order_by.append(("huggingface_repo_id", direction))
                    new_order_by.append(("huggingface_filename", direction))
                    new_order_by.append(("model_scope_model_id", direction))
                    new_order_by.append(("model_scope_file_path", direction))
                    new_order_by.append(("local_path", direction))
            order_by = new_order_by

        return await Model.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            fields=fields,
            order_by=order_by,
        )


@router.get("/adapters", response_model=Dict[str, Any])
async def get_model_adapters(
    session: SessionDep,
    user: CurrentUserDep,
    base: str = Query(
        ...,
        description=(
            "Base model repo id (e.g. Qwen/Qwen3-8B) for HF/ModelScope adapter discovery; "
            "also used to match local cached LoRAs."
        ),
    ),
    q: Optional[str] = Query(
        None,
        description="Optional keyword (Hugging Face search, ModelScope Search).",
    ),
    limit: int = Query(
        40,
        ge=1,
        le=200,
        description="Max adapter entries per remote source (HF and ModelScope).",
    ),
):
    _ = user
    return await list_adapters_for_base(session, base, q=q, limit=limit)


@router.get("/{id}", response_model=ModelPublic)
async def get_model(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    model = await Model.one_by_id(session, id, options=[selectinload(Model.instances)])
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    public = ModelPublic.model_validate(model)
    public.has_stale_lora_instances = is_lora_list_stale(model)
    return public


@router.get("/{id}/pd-metrics", response_model=PDMetricsPublic)
async def get_model_pd_metrics(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    window: str = "15m",
):
    """Whether this disaggregated group is actually disaggregating.

    The one signal that separates "PD is working" from "PD has silently
    collapsed to aggregated serving" — a failure that returns correct answers,
    logs nothing, and leaves every instance RUNNING.

    Read from Prometheus rather than from a column: the engines and the router
    already export the counters, the worker's aggregator already normalizes
    and labels them, and Prometheus already scrapes the worker over a path
    that handles tunnelled hosts. The label selector is injected server-side,
    so a caller only ever reads the series of a model it can already see.
    """
    model = await _get_model(session=session, ctx=ctx, id=id)
    if not model.disaggregation:
        raise BadRequestException(message="This deployment is not disaggregated")
    try:
        window_seconds = parse_window(window)
    except ValueError as e:
        raise BadRequestException(message=str(e))

    mode_name = getattr(model.disaggregation.mode, "value", None) or str(
        model.disaggregation.mode
    )
    return await collect_pd_metrics(
        model_id=model.id,
        mode=get_pd_mode(mode_name),
        window_seconds=window_seconds,
    )


@router.get("/{id}/dashboard")
async def get_model_dashboard(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    request: Request,
):
    model = await _get_model(session=session, ctx=ctx, id=id)

    cfg = get_global_config()

    # 🔴 A disaggregated group goes to the PD dashboard, not the model one.
    #
    # Decided here rather than in the caller because the caller has a model id
    # and this has the model: whether a deployment is a group, which connector
    # it runs, and therefore which role owns its transfer counters are all
    # server-side facts, and the last one is not even on the model — it comes
    # from the mode catalog.
    #
    # The model dashboard is not merely less specific for these, it is wrong in
    # one place: every request traverses both roles, so its request counters
    # double under PD. Sending a group there hands the user a number that is
    # 2x reality with nothing saying so.
    pd = bool(model.disaggregation)
    uid = cfg.grafana_pd_dashboard_uid if pd else cfg.grafana_model_dashboard_uid
    if not cfg.get_grafana_url() or not uid:
        raise InternalServerErrorException(
            message="Grafana dashboard settings are not configured"
        )

    cluster = None
    if model.cluster_id is not None:
        cluster = await Cluster.one_by_id(session, model.cluster_id)

    query_params = {}
    if cluster is not None:
        query_params["var-cluster_name"] = cluster.name
    query_params["var-model_name"] = model.name
    if pd:
        # Which role's transfer counter is authoritative for this connector --
        # decode where it pulls (NIXL), prefill where it pushes (SGLang). The
        # dashboard's default is `decode`, so leaving it unset would show an
        # SGLang group a flat zero for the panels that matter most.
        query_params["var-counted_role"] = _counted_role(model)

    grafana_base = resolve_grafana_base_url(cfg, request)
    slug = "gpustack-pd" if pd else "gpustack-model"
    dashboard_url = f"{grafana_base}/d/{uid}/{slug}"
    if query_params:
        dashboard_url = f"{dashboard_url}?{urlencode(query_params)}"

    return RedirectResponse(url=dashboard_url, status_code=302)


def _counted_role(model: Model) -> str:
    """The role whose KV transfer counters this model's connector populates.

    `decode` when the catalog cannot say, matching the dashboard's own default:
    the two-hop connectors are the common case, and a wrong guess here shows an
    empty panel rather than a wrong number.
    """
    mode_name = getattr(model.disaggregation.mode, "value", None) or str(
        model.disaggregation.mode
    )
    mode = get_pd_mode(mode_name)
    if mode and mode.transfer_metrics and mode.transfer_metrics.read_from_role:
        return mode.transfer_metrics.read_from_role
    return "decode"


async def _get_model(
    session: SessionDep,
    ctx,
    id: int,
):
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    return model


@router.get("/{id}/instances", response_model=ModelInstancesPublic)
async def get_model_instances(ctx: TenantContextDep, id: int, params: ListParamsDep):
    if params.watch:
        # Gate the stream on the same visibility check the non-watch
        # branch applies, so a model id outside the caller's scope can't
        # be tailed live.
        async with async_session() as session:
            model = await Model.one_by_id(session, id)
            assert_resource_visible(ctx, model, not_found_message="Model not found")
        fields = {"model_id": id}
        return StreamingResponse(
            ModelInstance.streaming(fields=fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        model = await Model.one_by_id(
            session, id, options=[selectinload(Model.instances)]
        )
        assert_resource_visible(ctx, model, not_found_message="Model not found")

        instances = model.instances
        count = len(instances)
        total_page = math.ceil(count / params.perPage)
        pagination = Pagination(
            page=params.page,
            perPage=params.perPage,
            total=count,
            totalPage=total_page,
        )

        return ModelInstancesPublic(items=instances, pagination=pagination)


def apply_scaling_schedule_baseline(
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
) -> None:
    """
    Drive ``replicas`` from an enabled scaling schedule.

    While a schedule is enabled the replica count is owned by the schedule:
    ``baseline_replicas`` plus the window rules are the user's input, and
    ``replicas`` becomes the scheduler-driven value. Set it to the count
    effective right now so the model doesn't run at a stale count until the
    scheduler's next tick. A submitted ``replicas`` is ignored in this mode.

    Call this as a server-side assignment *after* validation. Validating a
    rewritten ``replicas`` would make checks depend on the current wall clock:
    the value is a point-in-time output of the schedule, not caller intent.
    """
    schedule = getattr(model_in, "scaling_schedule", None)
    if not schedule or not schedule.enabled:
        return
    effective = compute_desired_replicas(schedule)
    if effective is not None:
        model_in.replicas = effective


def _max_intended_replicas(
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
) -> int:
    """Largest replica count this deployment could ever run.

    Only for deciding *whether* placement needs validating. The plain
    ``replicas > 0`` gate assumes a zero count means no instances are ever
    placed, which a schedule breaks: ``replicas`` is then just the count for
    right now, and the scheduler raises it later. Scaling to zero outside
    business hours is a headline use case, so a submitted 0 must still get its
    ``gpu_selector`` checked. Checks *inside* validation keep reading the
    submitted ``replicas`` — that is the caller's intent.
    """
    schedule = getattr(model_in, "scaling_schedule", None)
    if not schedule or not schedule.enabled:
        return model_in.replicas
    # An enabled schedule always carries a baseline and at least one rule.
    return max(
        model_in.replicas,
        schedule.baseline_replicas,
        *(rule.replicas for rule in schedule.rules),
    )


def validate_roles(  # noqa: C901
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
) -> None:
    """Structural checks on a multi-role deployment.

    Phase one deliberately rejects rather than reinterprets. Every rule here
    exists because the alternative — silently folding the request into
    something adjacent — is the failure mode that makes a deployment behave
    unlike what the user typed.

    `stored` is the row being updated, and every rule below is judged against
    the *merged* state. A sparse PUT carries only the fields it changes, so
    without this a request that adds a schedule without resending `roles` would
    be judged as a role-less model and the schedule would be accepted onto a
    group — exactly the combination the rule forbids, reached by not mentioning
    the thing that makes it illegal. Same shape for `replicas`. Read-only, so
    nothing here can widen what the request persists.
    """

    def field(name: str):
        submitted = getattr(model_in, name, None)
        if submitted is not None:
            return submitted
        if stored is not None and name not in getattr(
            model_in, "model_fields_set", set()
        ):
            return getattr(stored, name, None)
        return submitted

    roles = field("roles")
    disaggregation = field("disaggregation")

    if not roles:
        if disaggregation is not None:
            raise BadRequestException(
                message="disaggregation requires roles: declare a prefill and a decode role."
            )
        return

    names = [role.name for role in roles]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise BadRequestException(
            message=f"Duplicate role name(s): {', '.join(sorted(duplicates))}."
        )

    allowed = {item.value for item in RoleNameEnum}
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise BadRequestException(
            message=(
                f"Unsupported role name(s): {', '.join(unknown)}. "
                f"Supported roles are {', '.join(sorted(allowed))}."
            )
        )

    for role in roles:
        if role.name == RoleNameEnum.ROUTER.value and role.replicas != 1:
            raise BadRequestException(
                message="The router role runs exactly one replica."
            )

    # `dependencies` is a start order, so a cycle is a deployment that never
    # starts. Reject it here rather than letting the controller spin.
    known = set(names)
    graph = {role.name: list(role.dependencies or []) for role in roles}
    for name, deps in graph.items():
        for dep in deps:
            if dep not in known:
                raise BadRequestException(
                    message=f"Role '{name}' depends on '{dep}', which is not declared."
                )
            if dep == name:
                raise BadRequestException(
                    message=f"Role '{name}' cannot depend on itself."
                )
    visiting: set = set()
    done: set = set()

    def _walk(name: str) -> None:
        if name in done:
            return
        if name in visiting:
            raise BadRequestException(
                message=f"Role dependencies form a cycle through '{name}'."
            )
        visiting.add(name)
        for dep in graph.get(name, []):
            _walk(dep)
        visiting.discard(name)
        done.add(name)

    for name in graph:
        _walk(name)

    # `roles[].replicas` is the only scaling truth, so a model-level count
    # above one would be a second one. Refuse instead of quietly reading it as
    # a multiplier — an implicit mode switch is exactly what makes a
    # deployment stop matching its own spec.
    if field("replicas") not in (0, 1):
        raise BadRequestException(
            message=(
                "A model with roles uses replicas as an on/off switch (0 or 1). "
                "Scale a disaggregated deployment through roles[].replicas."
            )
        )

    # The scaling scheduler writes `model.replicas` directly, without passing
    # through this validation, so a window rule holding 3 would break the
    # deployment at its next tick rather than at submit time.
    schedule = field("scaling_schedule")
    if schedule and schedule.enabled:
        raise BadRequestException(
            message="Scheduled scaling is not supported for a model with roles."
        )

    if disaggregation is None:
        return

    counts = {name: names.count(name) for name in allowed}
    if counts[RoleNameEnum.PREFILL.value] != 1:
        raise BadRequestException(
            message="A disaggregated model needs exactly one prefill role."
        )
    if counts[RoleNameEnum.DECODE.value] != 1:
        raise BadRequestException(
            message="A disaggregated model needs exactly one decode role."
        )
    if counts[RoleNameEnum.ROUTER.value] > 1:
        raise BadRequestException(
            message="A disaggregated model has at most one router."
        )

    _reject_cache_under_a_hand_written_mode(field, roles, disaggregation)
    _reject_a_policy_the_mode_cannot_apply(disaggregation)

    # A recipe injects one engine's connector configuration into every role,
    # so a role on a different engine would receive settings it cannot read.
    permitted = PD_MODE_BACKENDS.get(disaggregation.mode.value, [])
    if permitted:
        for role in roles:
            role_backend = role.backend or field("backend")
            if role_backend and role_backend not in permitted:
                raise BadRequestException(
                    message=(
                        f"Role '{role.name}' runs backend '{role_backend}', which "
                        f"pd mode '{disaggregation.mode.value}' cannot configure "
                        f"(it targets {', '.join(permitted)}). Mixing engines "
                        f"across roles requires pd mode 'custom', where the "
                        f"connection parameters are yours to supply."
                    )
                )


_KV_LOAD_FAILURE_PLACEHOLDER = "{{kv_load_failure_policy}}"


def _reject_a_policy_the_mode_cannot_apply(disaggregation) -> None:
    """`kv_load_failure_policy` is a vLLM/NIXL setting, not a platform one.

    Only `vllm-nixl` renders it. The SGLang modes have no equivalent concept
    at all -- their KV lifecycle is a bootstrap timeout that aborts the
    request, not a load that can fail and be retried -- and Mooncake's
    connector does not read the key. So there is nothing to implement on the
    other three; what there is, is a value the user weighed and set that then
    quietly does nothing.

    🔑 Which is why this rejects rather than warns, and only for a non-default
    value. `fail` is what an engine that never sees the setting does anyway,
    so refusing it would break every group on those modes to no purpose;
    `recompute` is the deliberate choice -- trade a 500 for a silent
    recomputation -- and a user who made it and got neither is worse off than
    one who was told the mode cannot honour it.

    Derived from the recipe rather than a list of mode names: a mode that
    starts rendering the placeholder is accepted the moment it does, with
    nothing here to remember to update.
    """
    from gpustack.schemas.models import DisaggregationSpec

    policy = getattr(disaggregation, "kv_load_failure_policy", None)
    default = DisaggregationSpec.model_fields["kv_load_failure_policy"].default
    if policy is None or policy == default:
        return

    mode_name = getattr(disaggregation.mode, "value", None) or str(disaggregation.mode)
    mode = get_pd_mode(mode_name)
    if mode is None or _KV_LOAD_FAILURE_PLACEHOLDER in mode.model_dump_json():
        return

    if disaggregation.mode == PDModeEnum.CUSTOM:
        # The one mode where the setting may well be reachable, just not from
        # here: `custom` injects nothing, so every connector key is the user's
        # to write. Pointing them at another mode would be the wrong advice.
        raise BadRequestException(
            message=(
                f"pd mode 'custom' injects no connector configuration, so "
                f"kv_load_failure_policy='{policy}' would be stored and never "
                f"reach the engine. Set it inside your own "
                f"--kv-transfer-config instead."
            )
        )

    raise BadRequestException(
        message=(
            f"pd mode '{mode_name}' cannot apply kv_load_failure_policy="
            f"'{policy}': its KV connector has no such setting, so the value "
            f"would be stored and never reach the engine. Leave it at "
            f"'{default}', or use a mode whose connector reads it."
        )
    )


def _reject_cache_under_a_hand_written_mode(field, roles, disaggregation) -> None:
    """`custom` mode and an extended KV cache cannot be asked for together.

    Everywhere else the two compose: GPUStack folds the mode's connector and
    the cache's into one `MultiConnector`, which is what makes a disaggregated
    deployment with a shared cache a supported combination rather than a
    choice between them.

    `custom` is the one mode that injects no connection state at all — its
    whole contract is that the parameters are the user's. So there is nothing
    to compose the cache with, and quietly injecting a connector under a mode
    that promises not to would be the surprise this rejection exists to
    prevent. Written by hand, both still fit in one flag; the engine composes
    connectors and the user is the one holding the pen.
    """
    if disaggregation.mode != PDModeEnum.CUSTOM:
        return

    model_cache = field("extended_kv_cache")
    for role in roles:
        cache = (
            role.extended_kv_cache
            if role.extended_kv_cache is not None
            else model_cache
        )
        if cache is None or not getattr(cache, "enabled", False):
            continue
        raise BadRequestException(
            message=(
                f"Role '{role.name}' enables the extended KV cache under pd "
                "mode 'custom', which injects no connector configuration at "
                "all — so there is nothing for GPUStack to compose the cache "
                "into. Either choose a pd mode that configures a connector, "
                "where the two are combined for you, or keep 'custom' and "
                "write the combined configuration into backend_parameters."
            )
        )


# Engine parameters that must agree between prefill and decode, with the
# spellings each engine uses for them. The split below is not stylistic: the
# first entry is the one the engines do NOT check, and the rest are ones they
# do — checked here anyway so the report names the role rather than surfacing
# as a geometry assertion inside a container.
_PAIRING_MAX_LEN = ["max-model-len", "max_model_len", "context-length"]
_PAIRING_TP = ["tensor-parallel-size", "tp", "tp-size"]
_PAIRING_MUST_MATCH = {
    "dtype": ["dtype"],
    "KV cache dtype": ["kv-cache-dtype"],
    "block size": ["block-size", "page-size"],
    "KV cache layout": ["kv-cache-layout"],
}

# The hybrid KV cache manager is a boolean pair rather than a value, so it
# cannot go in the table above. Both spellings are argparse's, and both appear
# in practice: a cache provider's injection carries the disabling one, and an
# Ascend PD recipe carries the enabling one.
_HMA_DISABLE = "--disable-hybrid-kv-cache-manager"
_HMA_ENABLE = "--no-disable-hybrid-kv-cache-manager"


def _hybrid_cache_manager_enabled(parameters: List[str]) -> bool:
    """Whether HMA ends up on for a role that also gets a KV connector.

    Measured: setting `--kv-transfer-config` makes vLLM disable HMA on its own
    (`vllm/config/vllm.py`), and every disaggregated role gets that flag — so
    the two sides agree by default and the disabling flag is redundant rather
    than meaningful. What is not redundant is the *enabling* spelling: a
    connector that supports HMA can have it turned back on explicitly, and one
    side doing that while the other does not is a real divergence.

    Last spelling wins, matching argparse, so a role that carries both is read
    the way the engine would read it rather than the way the list is ordered.
    """
    enabled = False
    for token in parameters:
        name = token.split("=", 1)[0]
        if name == _HMA_ENABLE:
            enabled = True
        elif name == _HMA_DISABLE:
            enabled = False
    return enabled


def _role_parameters(role: RoleSpec, model_parameters) -> List[str]:
    """A role's effective engine parameters.

    `None` inherits, an empty list does not — a role that deliberately clears
    the model's parameters must not silently get them back.
    """
    if role.backend_parameters is None:
        return list(model_parameters or [])
    return list(role.backend_parameters)


def validate_role_pairing(  # noqa: C901
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    stored: Optional[Model] = None,
) -> None:
    """Reject prefill/decode pairs the engines will accept and serve wrongly.

    The division of labour with the engine is deliberate and documented in X1
    3.1: most handshake factors are hashed by the connector and rejected on
    contact, so re-checking them here buys attribution, not safety. Two are
    different.

    `max_model_len` is checked by nothing at all. Measured with prefill at 8192
    and decode at 4096: the handshake passes, KV transfers, short prompts
    answer normally, and only a prompt above decode's window fails — with a 400
    from decode, after prefill has already computed it. The user is left
    believing the deployment serves 8192.

    Tensor parallelism is asserted by the engine at run time, but a decode
    narrower than its prefill surfaces as an `IndexError` inside decode rather
    than as a configuration error, so the hard block is worth more than the
    assertion.

    This is a pre-check, not a mirror of the engine's factor set — vLLM's own
    source says that set is "likely to evolve significantly over time", so the
    engine stays the final judge.
    """

    def field(name: str):
        submitted = getattr(model_in, name, None)
        if submitted is not None:
            return submitted
        if stored is not None and name not in getattr(
            model_in, "model_fields_set", set()
        ):
            return getattr(stored, name, None)
        return submitted

    roles = field("roles")
    if not roles or not field("disaggregation"):
        return

    model_parameters = field("backend_parameters")
    prefill = next((r for r in roles if r.name == RoleNameEnum.PREFILL.value), None)
    decode = next((r for r in roles if r.name == RoleNameEnum.DECODE.value), None)
    if prefill is None or decode is None:
        return

    prefill_params = _role_parameters(prefill, model_parameters)
    decode_params = _role_parameters(decode, model_parameters)

    prefill_len = find_int_parameter(prefill_params, _PAIRING_MAX_LEN)
    decode_len = find_int_parameter(decode_params, _PAIRING_MAX_LEN)
    if prefill_len is not None and decode_len is not None:
        if prefill_len != decode_len:
            raise BadRequestException(
                message=(
                    f"prefill and decode declare different context lengths "
                    f"({prefill_len} vs {decode_len}). No engine checks this: "
                    f"the pair handshakes, transfers KV and answers short "
                    f"prompts, and a prompt above "
                    f"{min(prefill_len, decode_len)} tokens fails at decode "
                    f"after prefill has already computed it. Give both roles "
                    f"the same context length."
                )
            )

    prefill_tp = find_int_parameter(prefill_params, _PAIRING_TP)
    decode_tp = find_int_parameter(decode_params, _PAIRING_TP)
    if prefill_tp is not None and decode_tp is not None and decode_tp < prefill_tp:
        raise BadRequestException(
            message=(
                f"decode runs tensor parallelism {decode_tp}, below prefill's "
                f"{prefill_tp}. A decode narrower than its prefill cannot "
                f"receive that prefill's KV layout, and the engine reports it "
                f"as an IndexError inside decode rather than as a "
                f"configuration error. decode's tensor parallelism must be at "
                f"least prefill's."
            )
        )

    if _hybrid_cache_manager_enabled(prefill_params) != _hybrid_cache_manager_enabled(
        decode_params
    ):
        raise BadRequestException(
            message=(
                "prefill and decode disagree on the hybrid KV cache manager. "
                f"It is one of the factors the connector hashes, so the pair "
                f"is rejected on contact and the group never serves. Note that "
                f"a KV connector disables it on its own — the divergence comes "
                f"from one role carrying {_HMA_ENABLE} and the other not."
            )
        )

    for label, names in _PAIRING_MUST_MATCH.items():
        prefill_value = find_parameter(prefill_params, names)
        decode_value = find_parameter(decode_params, names)
        if (
            prefill_value is not None
            and decode_value is not None
            and prefill_value != decode_value
        ):
            raise BadRequestException(
                message=(
                    f"prefill and decode declare different {label} "
                    f"('{prefill_value}' vs '{decode_value}'). The connector "
                    f"rejects the pair on contact, so the group would never "
                    f"serve; the roles must agree."
                )
            )


async def validate_model_in(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
    stored: Optional[Model] = None,
):
    # `stored` is the row being updated, so a sparse PUT is judged against the
    # merged state rather than against the handful of fields it happened to
    # send. Absent on create, where there is nothing to merge.
    validate_roles(model_in, stored=stored)
    validate_role_pairing(model_in, stored=stored)

    if getattr(model_in, "gpu_type_selector", None) is not None:
        await validate_gpu_type_selector(session, model_in, cluster_id=cluster_id)

    if model_in.gpu_selector is not None and _max_intended_replicas(model_in) > 0:
        await validate_gpu_ids(session, model_in, cluster_id=cluster_id)

    if is_custom_backend(model_in.backend):
        logger.info("Skip model validation for custom backend")
        return

    if model_in.backend_parameters:
        param_gpu_layers = find_parameter(
            model_in.backend_parameters, ["ngl", "gpu-layers", "n-gpu-layers"]
        )

        if param_gpu_layers:
            int_param_gpu_layers = safe_int(param_gpu_layers, None)
            if (
                not param_gpu_layers.isdigit()
                or int_param_gpu_layers < 0
                or int_param_gpu_layers > 999
            ):
                raise BadRequestException(
                    message="Invalid backend parameter --gpu-layers. Please provide an integer in the range 0-999 (inclusive)."
                )

            if (
                int_param_gpu_layers == 0
                and model_in.gpu_selector is not None
                and len(model_in.gpu_selector.gpu_ids) > 0
            ):
                raise BadRequestException(
                    message="Cannot set --gpu-layers to 0 and manually select GPUs at the same time. Setting --gpu-layers to 0 means running on CPU only."
                )

        unsupported_params = [
            (
                ["port"],
                (
                    "Setting the port using --port is not supported. Ports are "
                    "automatically allocated by GPUStack."
                ),
            ),
            (
                ["api-key"],
                (
                    "Setting the API key using --api-key is not supported. API keys "
                    "are managed by GPUStack."
                ),
            ),
            (
                ["served-model-name"],
                (
                    "Setting the served model name using --served-model-name is not "
                    "supported. The model name is automatically set from your "
                    "deployment configuration."
                ),
            ),
        ]

        for param_names, error_message in unsupported_params:
            if find_parameter(model_in.backend_parameters, param_names):
                raise BadRequestException(message=error_message)

    validate_and_normalize_lora_list(model_in)


def validate_and_normalize_lora_list(
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
) -> None:
    """Normalize each lora_name to the stored "<base>:<short>" form.

    Accepts a bare short name and prepends the base prefix; a correct "<base>:"
    prefix is kept as-is. Rejects wrong prefixes, embedded colons, empty names,
    and duplicates. The API strips the prefix again on the way out (see
    ModelPublic._strip_lora_prefix).
    """
    lora_list = getattr(model_in, "lora_list", None)
    if not lora_list:
        return

    expected_prefix = f"{model_in.name}:"
    seen: set = set()
    for i, item in enumerate(lora_list):
        entry = LoraListEntry.model_validate(item) if isinstance(item, dict) else item
        short_name = (entry.lora_name or "").strip()
        if not short_name:
            raise BadRequestException(
                message="lora_name must not be empty in lora_list."
            )
        if ":" in short_name:
            if not short_name.startswith(expected_prefix):
                raise BadRequestException(
                    message=(
                        f"lora_name '{short_name}' must not contain ':'. Set "
                        f"lora_name to the bare adapter name (e.g. 'my-adapter'); "
                        f"the '{expected_prefix}' prefix is added automatically."
                    )
                )
            short_name = short_name[len(expected_prefix) :]
            if not short_name:
                raise BadRequestException(
                    message=(
                        f"lora_name is missing the suffix after the base model "
                        f"prefix '{expected_prefix}'."
                    )
                )
            if ":" in short_name:
                raise BadRequestException(
                    message=(
                        f"lora_name '{entry.lora_name}' must not contain a nested "
                        f"':' after the base model prefix."
                    )
                )
        entry.lora_name = lora_route_name_for(model_in.name, short_name)
        lora_list[i] = entry
        if short_name in seen:
            raise BadRequestException(
                message=f"Duplicate lora_name '{short_name}' in lora_list."
            )
        seen.add(short_name)


async def validate_gpu_type_selector(
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
):
    """Validate a model's ``gpu_type_selector`` against the local projection.

    Reads only the local ``GPUInstanceType`` projection table (synced from the
    operator watch stream); no Kubernetes client calls. Fails closed: anything
    that cannot be verified from the projection is rejected.
    """
    selector = model_in.gpu_type_selector

    gpu_selector = model_in.gpu_selector
    if gpu_selector is not None and gpu_selector.gpu_ids:
        raise BadRequestException(
            message="gpu_type_selector cannot be combined with gpu_selector: "
            "manual GPU selection and InstanceType-based sliced GPU selection "
            "are mutually exclusive."
        )

    # 🔴 Narrowed to the slicing and partition modes, and it needs to be.
    #
    # A slice is a fraction of one card the node's device plugin picks at
    # allocation time, so "more than one" has no meaning: the caller cannot say
    # which card the first one landed on. A *whole-card* claim has no such
    # difficulty — the operator's resource model hands out several at once, and
    # the container sees exactly the devices allocated, so an engine told tp=4
    # finds four. Refusing that was refusing something the layer below can do.
    #
    # ⚠️ Note this check has never actually fired on the common path:
    # `set_model_gpus_per_replica` returns early unless `gpu_selector.gpu_ids`
    # is set, and manual ids are mutually exclusive with `gpu_type_selector`
    # above — so `gpus_per_replica` is `None` for every InstanceType claim. The
    # real behaviour was "accepted, then scheduled onto one card while the
    # engine expected several", which is worse than a refusal. Whole-card
    # multi-card is handled by `InstanceTypeWholeCardSelector`; see design
    # §3.7.11.
    sliced_or_partitioned = (
        (selector.accelerator_sliced_memory_percentage or 0) > 0
        or (selector.accelerator_sliced_cores_percentage or 0) > 0
        or bool(selector.accelerator_partitioned_profile)
    )
    if (
        sliced_or_partitioned
        and gpu_selector is not None
        and gpu_selector.gpus_per_replica is not None
        and gpu_selector.gpus_per_replica > 1
    ):
        raise BadRequestException(
            message="gpus_per_replica must be 1 when a sliced or partitioned "
            "gpu_type_selector is set: one slice is a fraction of one card, so "
            "asking for several has no meaning. Use a whole-card claim (all "
            "slicing percentages zero) for a member that needs several cards."
        )

    memory_pct = selector.accelerator_sliced_memory_percentage
    cores_pct = selector.accelerator_sliced_cores_percentage
    memory_sliced = memory_pct is not None and memory_pct > 0
    cores_sliced = cores_pct is not None and cores_pct > 0
    if (memory_sliced or cores_sliced) and selector.accelerator_partitioned_profile:
        raise BadRequestException(
            message="accelerator_partitioned_profile cannot be combined with "
            "accelerator_sliced_memory_percentage or "
            "accelerator_sliced_cores_percentage: hardware partitioning and "
            "software slicing cannot both apply to one card."
        )

    effective_cluster_id = (
        cluster_id if cluster_id is not None else getattr(model_in, "cluster_id", None)
    )
    if effective_cluster_id is None:
        raise BadRequestException(
            message="A cluster must be specified when gpu_type_selector is set: "
            "the InstanceType projection is scoped per cluster."
        )

    matched_list = await GPUInstanceType.all_by_fields(
        session,
        fields={
            "cluster_id": effective_cluster_id,
            "deleted_at": None,
            "name": selector.type,
        },
    )
    if not matched_list:
        # Keep the two errors distinct: no synced types in the cluster at
        # all vs. this type missing.
        instance_types = await GPUInstanceType.all_by_fields(
            session,
            fields={"cluster_id": effective_cluster_id, "deleted_at": None},
        )
        if not instance_types:
            raise BadRequestException(
                message=f"Cluster {effective_cluster_id} has no synced GPU InstanceTypes: "
                "gpu_type_selector requires a Kubernetes cluster managed by "
                "gpustack-operator."
            )
        raise BadRequestException(
            message=f"GPU InstanceType '{selector.type}' not found in cluster "
            f"{effective_cluster_id}."
        )
    matched = matched_list[0]

    # A cluster also publishes non-accelerated InstanceTypes (a CPU-only pool),
    # and nothing about their name says so. The deploy form filters them out of
    # its GPU Type list, but an API caller can still name one — and it would
    # only fail later at scheduling, reported as a type that "does not report
    # its accelerator memory" rather than as the wrong kind of type.
    if not matched.spec.acceleratable:
        raise BadRequestException(
            message=f"GPU InstanceType '{selector.type}' in cluster "
            f"{effective_cluster_id} is not an accelerator type: "
            "gpu_type_selector requires one backed by GPUs."
        )

    # spec is projected as soon as the InstanceType appears, but status.detail is
    # backfilled by the operator afterwards, so a type can be nameable before its
    # hardware is known. Every mode needs the card's memory to size the claim
    # (a percentage of it, a profile out of it, or the whole card), so without it
    # the model would be accepted here and then never schedule — the fit would
    # report the type as unavailable or as not reporting its memory.
    detail = matched.status.detail if matched.status else None
    if detail is None or not detail.memory:
        raise BadRequestException(
            message=f"GPU InstanceType '{selector.type}' in cluster "
            f"{effective_cluster_id} does not report its accelerator memory yet: "
            "gpustack-operator has not finished backfilling the type. Retry once "
            "the type reports its hardware detail."
        )

    if selector.accelerator_partitioned_profile:
        sliced_detail = detail.sliced_detail
        physical = sliced_detail.physical if sliced_detail else None
        profiles = physical.profiles if physical and physical.profiles else []
        profile_names = {p.name for p in profiles if p.name}
        if selector.accelerator_partitioned_profile not in profile_names:
            raise BadRequestException(
                message=f"Profile '{selector.accelerator_partitioned_profile}' "
                f"is not offered by GPU InstanceType '{selector.type}' in "
                f"cluster {effective_cluster_id}. Available profiles: "
                f"{sorted(profile_names) or 'none'}."
            )


async def validate_gpu_ids(  # noqa: C901
    session: SessionDep,
    model_in: Union[ModelCreate, ModelUpdate, ModelSpecBase],
    *,
    cluster_id: Optional[int] = None,
):
    effective_cluster_id = (
        cluster_id if cluster_id is not None else getattr(model_in, "cluster_id", None)
    )

    # A selector can legitimately carry `gpus_per_replica` and no `gpu_ids`:
    # that's what it looks like when the card is picked by the operator's
    # device plugin rather than by index, which is also the shape of a
    # per-role selector. Everything below reads `gpu_ids` as a sequence, so
    # normalise it once here instead of guarding at each use.
    gpu_ids = model_in.gpu_selector.gpu_ids or []

    if gpu_ids and model_in.gpu_selector.gpus_per_replica:
        if len(gpu_ids) < model_in.gpu_selector.gpus_per_replica:
            raise BadRequestException(
                message="The number of selected GPUs must be greater than or equal to gpus_per_replica."
            )

    model_backend = model_in.backend

    if model_backend == BackendEnum.VOX_BOX and (
        len(gpu_ids) > 1
        or (
            model_in.gpu_selector.gpus_per_replica is not None
            and model_in.gpu_selector.gpus_per_replica > 1
        )
    ):
        raise BadRequestException(
            message="The vox-box backend is restricted to execution on a single NVIDIA GPU."
        )

    worker_name_set = set()
    for gpu_id in gpu_ids:
        is_valid, matched = parse_gpu_id(gpu_id)
        if not is_valid:
            raise BadRequestException(message=f"Invalid GPU ID: {gpu_id}")

        worker_name = matched.get("worker_name")
        gpu_index = safe_int(matched.get("gpu_index"), -1)
        worker_name_set.add(worker_name)

        if effective_cluster_id is None:
            raise BadRequestException(
                message=f"A cluster context is required for manual GPU selection, but was not provided. Cannot validate worker '{worker_name}'."
            )

        worker = await WorkerService(session).get_by_cluster_id_name(
            effective_cluster_id, worker_name
        )
        if not worker:
            raise BadRequestException(message=f"Worker {worker_name} not found")

        gpu = (
            next(
                (gpu for gpu in worker.status.gpu_devices if gpu.index == gpu_index),
                None,
            )
            if worker.status and worker.status.gpu_devices
            else None
        )
        if gpu:
            validate_gpu(gpu, model_backend=model_backend)

        if model_backend == BackendEnum.VLLM and len(worker_name_set) > 1:
            await validate_distributed_vllm_limit_per_worker(session, model_in, worker)

    if (
        is_custom_backend(model_backend)
        and len(worker_name_set) > 1
        and model_in.replicas == 1
    ):
        raise BadRequestException(
            message="Distributed inference across multiple workers is not supported for custom backends."
        )


def validate_gpu(gpu_device: GPUDeviceStatus, model_backend: str = ""):
    if (
        model_backend == BackendEnum.VOX_BOX
        and gpu_device.vendor != ManufacturerEnum.NVIDIA.value
    ):
        raise BadRequestException(
            "The vox-box backend is supported only on NVIDIA GPUs."
        )

    if (
        model_backend == BackendEnum.ASCEND_MINDIE
        and gpu_device.vendor != ManufacturerEnum.ASCEND.value
    ):
        raise BadRequestException(
            f"Ascend MindIE backend requires Ascend NPUs. Selected {gpu_device.vendor} GPU is not supported."
        )


async def validate_distributed_vllm_limit_per_worker(
    session: AsyncSession, model: Union[ModelCreate, ModelUpdate], worker: Worker
):
    """
    Validate that there is no more than one distributed vLLM instance per worker.
    """
    instances = await ModelInstance.all_by_field(session, "worker_id", worker.id)
    for instance in instances:
        if (
            instance.distributed_servers
            and instance.distributed_servers.subordinate_workers
            and instance.model_name != model.name
        ):
            raise BadRequestException(
                message=f"Each worker can run only one distributed vLLM instance. Worker '{worker.name}' already has '{instance.name}'."
            )


async def assert_cluster_belongs_to_org(
    ctx: TenantContext,
    session: AsyncSession,
    cluster_id: Optional[int],
    owner_principal_id: int,
    cluster: Optional[Cluster] = None,
):
    """Ensure a chosen cluster is visible to the caller and owned by the
    given Org.

    A model runs on infrastructure owned by its Org, so its cluster must
    belong to that Org — otherwise a tenant could target the platform's
    (or another Org's) cluster, stamping a cross-tenant model. A cluster the
    caller can't see is reported as missing (404), so cross-tenant cluster
    ids can't be probed via a 403-vs-404 difference; a visible cluster owned
    by another Org is a 403. No cluster chosen (``cluster_id is None``)
    leaves default-cluster resolution to pick the Org's own cluster.

    ``cluster`` may be passed pre-fetched to avoid a duplicate lookup.
    """
    if cluster_id is None:
        return
    if cluster is None:
        cluster = await Cluster.one_by_id(session, cluster_id)
    not_found = f"Cluster {cluster_id} not found"
    assert_cluster_visible(ctx, cluster, not_found_message=not_found)
    if cluster.deleted_at is not None:
        raise NotFoundException(message=not_found)
    if cluster.owner_principal_id != owner_principal_id:
        raise ForbiddenException(
            message="The selected cluster does not belong to the current organization."
        )


async def validate_shared_kv_cache(
    session: AsyncSession,
    model_in: Union[ModelCreate, ModelUpdate],
    owner_principal_id: int,
    effective_cluster_id: Optional[int],
) -> None:
    """Validate the extended-KV-cache configuration against its target
    cache service.

    "shared" mode attaches the model's inference engine to a CacheService
    row, so the service must exist, belong to the model's Org (a
    cross-tenant id is reported as missing so service ids can't be probed),
    run in the model's cluster (the engine connects over the cluster
    network), and have a provider that knows how to inject connector
    config for the model's backend. "local" mode uses no service, so a
    stray cache_service_id is rejected as a mis-configuration rather than
    silently ignored.
    """
    ext = model_in.extended_kv_cache
    if not ext or not ext.enabled:
        return

    if ext.is_local():
        if ext.cache_service_id:
            raise BadRequestException(
                message="cache_service_id is only valid when mode is 'shared'"
            )
        return

    if not ext.cache_service_id:
        raise BadRequestException(
            message=(
                "cache_service_id is required when extended KV cache "
                "mode is 'shared'"
            )
        )

    cache_service = await CacheService.one_by_id(session, ext.cache_service_id)
    if (
        cache_service is None
        or cache_service.deleted_at is not None
        or cache_service.owner_principal_id != owner_principal_id
    ):
        raise NotFoundException(message="Cache service not found")

    if (
        effective_cluster_id is not None
        and cache_service.cluster_id != effective_cluster_id
    ):
        raise BadRequestException(
            message="The cache service must be in the same cluster as the model."
        )

    provider = get_cache_provider(cache_service.provider_name)
    backend = model_in.backend or BackendEnum.VLLM.value
    if provider is None or provider.integration_for(backend) is None:
        raise BadRequestException(
            message=(
                f"Cache service provider '{cache_service.provider_name}' is "
                f"not compatible with backend '{backend}'."
            )
        )

    # Every built-in integration is framework-scoped, so a cluster whose
    # accelerators are all outside the provider's support matrix would
    # pass the framework-less check above and then degrade on every
    # instance. Pre-check against the cluster's actual accelerators;
    # accelerator-less clusters are left to scheduling.
    workers = await Worker.all_by_fields(
        session,
        fields={"cluster_id": cache_service.cluster_id},
        extra_conditions=[Worker.deleted_at.is_(None)],
    )
    frameworks = {
        device.type
        for worker in workers
        for device in (
            worker.status.gpu_devices
            if worker.status and worker.status.gpu_devices
            else []
        )
        if device.type
    }
    if frameworks and not any(
        provider.integration_for(backend, framework) for framework in frameworks
    ):
        raise BadRequestException(
            message=(
                f"Cache service provider '{cache_service.provider_name}' "
                f"has no '{backend}' integration for the cluster's "
                f"accelerators ({', '.join(sorted(frameworks))})."
            )
        )

    # A pinned engine version below an integration's declared floor would
    # receive injected args the engine does not accept (e.g.
    # --shutdown-timeout) and fail to start. Reject when the version
    # falls outside every candidate integration's range; unparseable
    # versions fail open, and an unpinned version is resolved at deploy
    # time (the injection resolver re-checks it there).
    engine_version = model_in.backend_version
    if engine_version:
        candidates = (
            [provider.integration_for(backend, framework) for framework in frameworks]
            if frameworks
            else [provider.integration_for(backend)]
        )
        ranged = [c for c in candidates if c is not None and c.versions]
        if ranged and all(
            version_in_range(engine_version, c.versions) is False for c in ranged
        ):
            ranges = ", ".join(sorted({c.versions for c in ranged}))
            raise BadRequestException(
                message=(
                    f"Backend version {engine_version} is outside the "
                    f"cache provider's supported '{backend}' range "
                    f"({ranges})."
                )
            )


@router.post(
    "",
    response_model=ModelPublic,
)
async def create_model(
    session: SessionDep, ctx: TenantContextDep, model_in: ModelCreate
):
    # Resolve the owning Org first — admin in "All" mode (no current
    # principal) inherits the chosen cluster's Org, or falls back to
    # the platform Org. The same value drives both the uniqueness
    # pre-check below and the row we stamp on insert; resolving it up
    # front keeps them in sync so the pre-check actually catches a
    # collision in the Org the model will land in.
    target_org_id = ctx.current_principal_id
    cluster = None
    if target_org_id is None and model_in.cluster_id is not None:
        # Admin "All" mode has no principal context; derive the owning Org
        # from the chosen cluster. Reused by the check below to avoid a
        # second lookup. Under an Org context the helper does the single
        # lookup itself.
        cluster = await Cluster.one_by_id(session, model_in.cluster_id)
        if cluster is None:
            raise NotFoundException(message=f"Cluster {model_in.cluster_id} not found")
        target_org_id = cluster.owner_principal_id
    if target_org_id is None:
        target_org_id = platform_principal_id()

    # The chosen cluster must exist, be visible to the caller, and be owned
    # by the target Org. In admin "All" mode target_org_id was derived from
    # the cluster above, so the ownership check is trivially satisfied and
    # this mainly rejects a missing/deleted or non-visible cluster_id.
    await assert_cluster_belongs_to_org(
        ctx, session, model_in.cluster_id, target_org_id, cluster=cluster
    )

    # Model & ModelRoute names are unique within their Org. Two Orgs
    # can each have a "llama3" without colliding.
    existing = await Model.one_by_fields(
        session,
        {"name": model_in.name, "owner_principal_id": target_org_id},
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Model with name '{model_in.name}' already exists."
        )
    should_create_route = (
        model_in.enable_model_route is not None and model_in.enable_model_route
    )
    if should_create_route:
        existing_route = await ModelRoute.one_by_fields(
            session,
            {"name": model_in.name, "owner_principal_id": target_org_id},
        )
        if existing_route:
            raise AlreadyExistsException(
                message=f"Model route with name '{model_in.name}' already exists."
            )
    await validate_model_in(session, model_in)
    # Server-side assignment, after validation: validation must see the replica
    # count the caller submitted, not the schedule-driven one.
    apply_scaling_schedule_baseline(model_in)
    await validate_shared_kv_cache(
        session, model_in, target_org_id, model_in.cluster_id
    )
    model_in_dict = model_in.model_dump(exclude={"enable_model_route"})

    # Stamp tenant scope. ModelBase has owner_principal_id defaulted to
    # PLATFORM_PRINCIPAL_ID, so `model_dump()` always emits the key —
    # `setdefault` would silently leave it at 1 even when the caller is
    # acting under a different Org. Override directly with the value we
    # resolved above.
    model_in_dict["owner_principal_id"] = target_org_id

    # Multi-tenant default: a non-platform Org's new model (and the
    # route(s) it spawns) is scoped to that Org via ALLOWED_PRINCIPALS
    # with the owning Org auto-granted below. The Default (platform) Org
    # keeps AUTHED. Caller's explicit ``access_policy`` always wins and
    # then manages its own grants via /principals. ``model_dump`` always
    # emits ``access_policy`` (it has a default), so override directly.
    org_scoped_default = (
        target_org_id is not None
        and target_org_id != platform_principal_id()
        and "access_policy" not in model_in.model_fields_set
    )
    if org_scoped_default:
        model_in_dict["access_policy"] = AccessPolicyEnum.ALLOWED_PRINCIPALS

    try:
        model: Model = await Model.create(
            session, source=model_in_dict, auto_commit=(not should_create_route)
        )
        if should_create_route:
            model_route = ModelRoute(
                name=model.name,
                description=model.description,
                categories=model.categories,
                generic_proxy=model.generic_proxy,
                created_model_id=model.id,
                access_policy=model.access_policy,
                owner_principal_id=model.owner_principal_id,
            )
            model_route: ModelRoute = await ModelRoute.create(
                session, source=model_route, auto_commit=False
            )
            model_route_target = ModelRouteTarget(
                name=f"{model.name}-deployment",
                route_name=model_route.name,
                generic_proxy=model.generic_proxy,
                model_route=model_route,
                model=model,
                weight=100,
                state=TargetStateEnum.UNAVAILABLE,
            )
            await ModelRouteTarget.create(
                session,
                source=model_route_target,
                auto_commit=False,
            )
            if org_scoped_default:
                # Auto-grant the owning Org on the primary route so its
                # members see it out of the box. The route is brand new,
                # so no existence check is needed; LoRA child routes get
                # their own grants inside create_lora_model_routes.
                session.add(
                    ModelRoutePrincipalLink(
                        route_id=model_route.id,
                        principal_id=model.owner_principal_id,
                    )
                )
            await create_lora_model_routes(
                session,
                model,
                access_policy=model.access_policy,
                generic_proxy=model.generic_proxy,
            )
            await session.commit()
            await revoke_model_access_cache(session=session)
    except BadRequestException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to create model: {e}")

    return model


@router.put(
    "/{id}",
    response_model=ModelPublic,
)
async def update_model(
    session: SessionDep, ctx: TenantContextDep, id: int, model_in: ModelUpdate
):
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")

    # Block re-pointing a model at another Org's (e.g. the Default org's
    # shared) or a non-visible cluster: its cluster must stay owned by the
    # model's Org.
    await assert_cluster_belongs_to_org(
        ctx, session, model_in.cluster_id, model.owner_principal_id
    )

    # Validate against the merged state: a sparse update carries only the
    # fields being changed, so validation would otherwise check
    # gpu_selector/gpu_type_selector mutual exclusion against half the
    # picture (e.g. setting gpu_selector on a model that already has
    # gpu_type_selector). object.__setattr__ bypasses pydantic's
    # fields-set tracking, keeping the backfill out of the persisted patch.
    for field in ("gpu_type_selector", "gpu_selector", "cluster_id"):
        if field not in model_in.model_fields_set:
            object.__setattr__(model_in, field, getattr(model, field))

    await validate_model_in(session, model_in, stored=model)
    # Server-side assignment, after validation: validation must see the replica
    # count the caller submitted, not the schedule-driven one.
    apply_scaling_schedule_baseline(model_in)
    await validate_shared_kv_cache(
        session,
        model_in,
        model.owner_principal_id,
        model_in.cluster_id or model.cluster_id,
    )

    if model_in.backend != BackendEnum.CUSTOM.value and (
        model.run_command or model.image_name
    ):
        patch = model_in.model_dump(exclude_unset=True)
        patch["run_command"] = None
        patch["image_name"] = None
        model_in = patch

    try:
        await ModelService(session).update(model, model_in, auto_commit=False)
        updated = await Model.one_by_id(session, id)
        if not updated:
            raise RuntimeError("Model not found after update")
        base_route = await ModelRoute.one_by_field(session, "name", updated.name)
        if base_route:
            await create_lora_model_routes(
                session,
                updated,
                access_policy=updated.access_policy,
                generic_proxy=updated.generic_proxy,
            )
            await cleanup_orphan_lora_routes(session, updated)
        await session.commit()
        await revoke_model_access_cache(session=session)
    except BadRequestException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to update model: {e}")

    return updated


class ModelRestartResult(BaseModel):
    """What a restart request did, so the caller can tell "converged" from
    "nothing to do" without a second read."""

    spec_digest: str
    """The generation the group is being brought onto."""
    restarted: bool
    deleted_instances: List[str] = []
    message: Optional[str] = None


@router.post("/{id}/restart", response_model=ModelRestartResult)
async def restart_model(session: SessionDep, ctx: TenantContextDep, id: int):
    """Retire the running generation so the current spec takes effect.

    Atomic by construction, and that is the point rather than an optimisation.
    "Restart" has until now meant deleting an instance and letting replica
    convergence rebuild it, which for a group produces a window holding a
    new-generation prefill beside an old-generation decode — and the engines do
    not reject that pairing. A `max_model_len` mismatch handshakes, transfers,
    and only fails on a long prompt, after prefill has already been paid for.
    So the whole generation stops before any of it starts again.

    There is deliberately no role parameter. "Restart only the decodes" is the
    request that produces exactly the cross-generation window above, and the
    strongest way to reject it is to have no way to express it.

    Idempotent on the target digest: a group already wholly on the current spec
    reports `restarted: false` rather than bouncing containers, because the
    operation this endpoint names is "converge to the current spec", not
    "cycle the processes". A restart still in flight is a 409 — the members are
    mid-replacement and a second teardown would delete the replacements.

    Placement counts as something to converge, even though it is not part of
    the digest. It is not part of the digest because it is not user intent —
    nobody asks for a namespace — but "where the members are" is still part of
    what the current configuration would produce, and after an upgrade that
    introduced per-tenant namespaces it is the only part that differs. Without
    this, `placement_drifted` would be a marker with no cure but deleting the
    model.
    """
    model = await Model.one_by_id(session, id)
    assert_resource_visible(ctx, model, not_found_message="Model not found")

    target = await model_spec_digest(session, model)
    instances = await ModelInstance.all_by_fields(
        session, fields={"model_id": model.id, "deleted_at": None}
    )

    if not instances:
        return ModelRestartResult(
            spec_digest=target,
            restarted=False,
            message="No instances to restart; the model has none running.",
        )

    digests = {instance.spec_digest for instance in instances}
    if len(digests) > 1:
        # Mixed digests mean a previous restart is still rebuilding. Tearing
        # down again here would delete the replacements it just created.
        raise ConflictException(
            message="A restart is already in progress for this model: its "
            "instances span more than one generation. Retry once they have "
            "converged."
        )

    drifted = placement_drifted(
        instances,
        await resolve_workload_namespace(
            session, model.owner_principal_id, model.cluster_id
        ),
    )
    # A member in ERROR is not running the current configuration; it is not
    # running anything. Reporting "already run the current configuration" to
    # someone whose group is half down is not merely unhelpful, it is untrue —
    # and it leaves the operation they reached for with nothing to do. The
    # group is torn down and rebuilt, which is what a restart of a group has
    # always meant here.
    #
    # No thrash risk in making this a reason to act: this endpoint is only
    # ever reached by an explicit request. Automatic recovery of a crashed
    # member is the worker's, and it has its own crash-loop brake.
    failed = [
        instance.name
        for instance in instances
        if instance.state == ModelInstanceStateEnum.ERROR
    ]
    if digests == {target} and not drifted and not failed:
        return ModelRestartResult(
            spec_digest=target,
            restarted=False,
            message="Instances already run the current configuration.",
        )

    try:
        deleted = await ModelInstanceService(session).batch_delete(list(instances))
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to restart model: {e}")

    # Rebuilding is left to replica convergence rather than done here: it is
    # the one place that knows a group forms its GPU roles atomically and holds
    # the router back until they run, and duplicating that here would be a
    # second implementation of the rule that matters most.
    return ModelRestartResult(
        spec_digest=target,
        restarted=True,
        deleted_instances=deleted,
        message=_restart_message(drifted and digests == {target}, failed),
    )


def _restart_message(moved: bool, failed: List[str]) -> str:
    """Say which of the three reasons to act applied, because they lead to
    different next steps: a spec change is expected to fix itself, a placement
    move needs nothing further, and a failed member usually means the reason
    it failed is still there."""
    if failed:
        return (
            f"Instances retired, including {len(failed)} in error "
            f"({', '.join(sorted(failed))}); the group will re-form on the "
            "current configuration. A member that failed for a reason still "
            "present will fail again — check its log before retrying."
        )
    if moved:
        return (
            "Instances retired; the group will re-form in its tenant's "
            "namespace on the current configuration."
        )
    return "Instances retired; the group will re-form on the current configuration."


@router.delete(
    "/{id}",
)
async def delete_model(session: SessionDep, ctx: TenantContextDep, id: int):
    model = await Model.one_by_id(
        session,
        id,
        options=[
            selectinload(Model.instances),
            selectinload(Model.model_route_targets),
        ],
    )
    assert_resource_visible(ctx, model, not_found_message="Model not found")

    try:
        await ModelService(session).delete(model)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model: {e}")
