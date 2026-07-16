from sqlmodel import col
import yaml
from typing import Any, Dict, List, Optional, Sequence
import aiohttp
from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlmodel import func
from gpustack import envs
from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    BadRequestException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.api.tenant import (
    bypass_tenant_filter,
    assert_cluster_resource_visible,
    cluster_resource_visibility_conditions,
    cluster_scoped_system,
    scoped_cluster_row_visible,
)
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    is_audio_model,
    is_embedding_model,
    is_image_model,
    is_reranker_model,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.schemas.datasets import Dataset
from gpustack.schemas.benchmark import (
    DATASET_CUSTOM,
    DATASET_RANDOM,
    DATASET_SHAREGPT,
    Benchmark,
    DatasetSnapshot,
    BenchmarkCreate,
    BenchmarkFullPublic,
    BenchmarkListParams,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkResultPublic,
    BenchmarkSnapshot,
    BenchmarkStateEnum,
    BenchmarkStateUpdate,
    BenchmarkUpdate,
    BenchmarkPublic,
    BenchmarksPublic,
)

from gpustack.server.services import (
    WorkerService,
)
from gpustack.server.worker_request import stream_to_worker, request_to_worker
from gpustack.utils.gpu import summary_gpu_snapshots
from gpustack.utils.snapshot import (
    create_model_instance_snapshot,
    create_worker_snapshot,
)
from gpustack.worker.logs import LogOptionsDep
from sqlalchemy.orm import defer

MAX_EXPORT_RECORDS = 20
BENCHMARK_EXPORT_FIELD_ORDER = [
    "name",
    "model_name",
    "model_instance_name",
    "profile",
    "dataset_name",
    "request_rate",
    "total_requests",
    "dataset_input_tokens",
    "dataset_output_tokens",
    "dataset_seed",
]

router = APIRouter()


def order_benchmark_export_fields(benchmark: dict) -> dict:
    ordered = {}
    for field in BENCHMARK_EXPORT_FIELD_ORDER:
        if field in benchmark:
            ordered[field] = benchmark[field]

    for field, value in benchmark.items():
        if field not in ordered:
            ordered[field] = value

    return ordered


@router.get("", response_model=BenchmarksPublic)
async def get_benchmarks(
    ctx: TenantContextDep,
    params: BenchmarkListParams = Depends(),
    search: str = None,
    state: Optional[BenchmarkStateEnum] = Query(
        default=None,
        description="Filter by benchmark state.",
    ),
    model_name: Optional[str] = Query(None, description="Filter by model name."),
    gpu_summary: Optional[str] = Query(None, description="Filter by GPU summary."),
    dataset_name: Optional[str] = Query(None, description="Filter by dataset name."),
    profile: Optional[str] = Query(None, description="Filter by profile."),
    load_type: Optional[str] = Query(
        None, description="Filter by load type (fixed_rate / concurrency)."
    ),
):
    return await _get_benchmarks(
        ctx=ctx,
        params=params,
        state=state,
        search=search,
        model_name=model_name,
        gpu_summary=gpu_summary,
        dataset_name=dataset_name,
        profile=profile,
        load_type=load_type,
    )


def _fuzzy_contains(value: Optional[str], target: Optional[str]) -> bool:
    """Return False only when the filter value is set but not contained in target."""
    if not value:
        return True
    if not target:
        return False
    return value.lower() in target.lower()


def gpu_summary_filter(data: Benchmark, gpu_summary: Optional[str]) -> bool:
    return _fuzzy_contains(gpu_summary, data.gpu_summary)


def _make_benchmark_visibility_filter(ctx):
    def _visible(b: Benchmark) -> bool:
        if cluster_scoped_system(ctx):
            return scoped_cluster_row_visible(ctx, b)
        if bypass_tenant_filter(ctx):
            return True
        org_id = getattr(b, "owner_principal_id", None)
        if (
            ctx.current_principal_id is not None
            and org_id is not None
            and org_id == ctx.current_principal_id
        ):
            return True
        if getattr(b, "cluster_id", None) in ctx.accessible_cluster_ids:
            return True
        return False

    return _visible


async def _get_benchmarks(
    ctx,
    params: BenchmarkListParams,
    search: str = None,
    state: Optional[BenchmarkStateEnum] = None,
    model_name: Optional[str] = None,
    gpu_summary: Optional[str] = None,
    dataset_name: Optional[str] = None,
    profile: Optional[str] = None,
    load_type: Optional[str] = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields["name"] = search

    fields = {}
    if state:
        fields["state"] = state

    if dataset_name:
        fields["dataset_name"] = dataset_name

    # `load_type` (fixed_rate / concurrency) filter (exact match; every row
    # carries a load_type).
    def _load_type_match(data) -> bool:
        return not load_type or data.load_type == load_type

    extra_conditions = list(cluster_resource_visibility_conditions(ctx, Benchmark))
    if gpu_summary:
        extra_conditions.append(
            func.lower(Benchmark.gpu_summary).like(f"%{gpu_summary.lower()}%")
        )
    if profile:
        extra_conditions.append(
            func.lower(Benchmark.profile).like(f"%{profile.lower()}%")
        )
    if model_name:
        extra_conditions.append(
            func.lower(Benchmark.model_name).like(f"%{model_name.lower()}%")
        )
    if load_type:
        extra_conditions.append(Benchmark.load_type == load_type)

    _benchmark_visible = _make_benchmark_visibility_filter(ctx)

    if params.watch:
        return StreamingResponse(
            Benchmark.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=lambda data: _benchmark_visible(data)
                and gpu_summary_filter(data, gpu_summary)
                and _fuzzy_contains(profile, data.profile)
                and _fuzzy_contains(model_name, data.model_name)
                and _load_type_match(data),
            ),
            media_type="text/event-stream",
        )

    order_by = params.order_by
    if order_by:
        new_order_by = []
        for field, direction in order_by:
            new_order_by.append((field, direction))
            if field in [
                "dataset_name",
                "cluster_id",
                "model_id",
                "model_name",
                "state",
            ]:
                # add additional sorting fields for deterministic ordering
                new_order_by.append(("created_at", direction))
        order_by = new_order_by

    async with async_session() as session:
        return await Benchmark.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            order_by=order_by,
            extra_conditions=extra_conditions,
            options=[defer(Benchmark.raw_metrics)],
        )


@router.get("/{id}", response_model=BenchmarkFullPublic)
async def get_benchmark(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message=f"Benchmark {id} not found"
    )
    return benchmark


async def validate_and_mutate_benchmark_in(  # noqa: C901
    session: SessionDep, benchmark_in: BenchmarkCreate
) -> Benchmark:

    if not benchmark_in.model_instance_name.strip():
        raise BadRequestException(message="Field model_instance_name must be specified")

    mutated = Benchmark(**benchmark_in.model_dump())
    instance = await ModelInstance.one_by_field(
        session, "name", benchmark_in.model_instance_name
    )
    if not instance:
        raise BadRequestException(
            message=f"Model instance '{benchmark_in.model_instance_name}' not found"
        )

    if instance.state != ModelInstanceStateEnum.RUNNING:
        raise BadRequestException(
            message=f"Model instance '{benchmark_in.model_instance_name}' not in RUNNING state"
        )

    if benchmark_in.model_id is None:
        mutated.model_id = instance.model_id
        mutated.model_name = instance.model_name

    if benchmark_in.dataset_name is None:
        raise BadRequestException(message="Field dataset_name must be specified")

    if benchmark_in.dataset_name not in [
        DATASET_RANDOM,
        DATASET_SHAREGPT,
        DATASET_CUSTOM,
    ]:
        raise BadRequestException(
            message=f"Dataset '{benchmark_in.dataset_name}' is not supported. Supported datasets are '{DATASET_RANDOM}', '{DATASET_SHAREGPT}' and '{DATASET_CUSTOM}'."
        )

    if benchmark_in.dataset_name == DATASET_RANDOM and (
        benchmark_in.dataset_input_tokens is None
        or benchmark_in.dataset_output_tokens is None
    ):
        raise BadRequestException(
            message="Fields dataset_input_tokens and dataset_output_tokens must be specified for 'Random' dataset"
        )

    dataset_snapshot: Optional[DatasetSnapshot] = None
    if benchmark_in.dataset_name == DATASET_CUSTOM:
        if benchmark_in.dataset_id is None:
            raise BadRequestException(
                message="Field dataset_id must be specified for a custom 'Dataset'"
            )
        dataset = await Dataset.one_by_id(session, benchmark_in.dataset_id)
        if not dataset:
            raise BadRequestException(
                message=f"Dataset {benchmark_in.dataset_id} not found"
            )
        # Co-location: the dataset must live on the same worker as the target
        # instance so it can be mounted into the benchmark container.
        if dataset.worker_id is not None and dataset.worker_id != instance.worker_id:
            raise BadRequestException(
                message=(
                    f"Dataset '{dataset.readable_source}' is on worker {dataset.worker_id} but "
                    f"the instance runs on worker {instance.worker_id}. Pick a dataset "
                    "on the instance's worker or create one there."
                )
            )
        # `dataset_name` stays the TYPE ("Dataset") — it is the form's dataset-type
        # selector value and must round-trip through clone/edit. The actual dataset
        # is referenced by `dataset_id` (run-time mount) and snapshotted below for
        # display (self-contained, survives resource deletion).
        dataset_snapshot = DatasetSnapshot(
            dataset_id=dataset.id,
            source=dataset.source.value,
            readable_source=dataset.readable_source,
            huggingface_repo_id=dataset.huggingface_repo_id,
            huggingface_filename=dataset.huggingface_filename,
            model_scope_model_id=dataset.model_scope_model_id,
            model_scope_file_path=dataset.model_scope_file_path,
            local_path=dataset.local_path,
            column_mapping=dataset.column_mapping,
        )

    model = await Model.one_by_id(session, mutated.model_id)
    if not model:
        raise BadRequestException(message=f"Model {mutated.model_id} not found")

    if (
        is_image_model(model)
        or is_audio_model(model)
        or is_embedding_model(model)
        or is_reranker_model(model)
    ):
        raise BadRequestException(
            message=f"Benchmarking is not supported for model type '{model.type.value}'"
        )

    if benchmark_in.request_rate <= 0:
        mutated.request_rate = (
            benchmark_in.total_requests
            if benchmark_in.total_requests is not None
            else 1000
        )  # treat non-positive request_rate as unlimited

    snapshot = await get_benchmark_snapshot(session, instance, model)
    if dataset_snapshot is not None:
        snapshot.dataset = dataset_snapshot
    mutated.snapshot = snapshot
    mutated.gpu_summary, mutated.gpu_vendor_summary = summary_gpu_snapshots(
        snapshot.gpus
    )
    mutated.worker_id = instance.worker_id
    # Server-derive tenant scope from the target instance so client-supplied
    # cluster_id can't smuggle a benchmark into another tenant, and so the
    # row is visible to the owning Org via cluster_resource_visibility.
    mutated.cluster_id = instance.cluster_id
    mutated.owner_principal_id = instance.owner_principal_id
    return mutated


@router.post(
    "",
    response_model=BenchmarkPublic,
)
async def create_benchmark(
    session: SessionDep, ctx: TenantContextDep, benchmark_in: BenchmarkCreate
):
    existing = await Benchmark.one_by_field(session, "name", benchmark_in.name)
    if existing:
        raise AlreadyExistsException(
            message=f"Benchmark with name '{benchmark_in.name}' already exists."
        )

    mutated = await validate_and_mutate_benchmark_in(session, benchmark_in)
    try:
        benchmark = await Benchmark.create(session, mutated)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create benchmark: {e}")

    return benchmark


@router.put(
    "/{id}",
    response_model=BenchmarkPublic,
)
async def update_benchmark(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    benchmark_in: BenchmarkUpdate,
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message="Benchmark not found"
    )
    try:
        await benchmark.update(session, benchmark_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update benchmark: {e}")

    return benchmark


@router.patch(
    "/{id}/state",
    response_model=BenchmarkPublic,
)
async def update_benchmark_state(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    state_update: BenchmarkStateUpdate,
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message="Benchmark not found"
    )

    if (
        state_update.state is not None
        and state_update.state == BenchmarkStateEnum.STOPPED
        and benchmark.state
        not in [
            BenchmarkStateEnum.QUEUED,
            BenchmarkStateEnum.PENDING,
            BenchmarkStateEnum.RUNNING,
        ]
    ):
        raise BadRequestException(
            message="Only benchmarks in QUEUED, PENDING, or RUNNING state can be stopped."
        )

    # Progress is monotonic within a run: a multi-stage run reports each stage's
    # slice of the overall bar, so the server must never let it move backward.
    # Reset to 0 only when (re)entering RUNNING (a fresh / re-run).
    entering_running = (
        state_update.state == BenchmarkStateEnum.RUNNING
        and benchmark.state != BenchmarkStateEnum.RUNNING
    )
    if entering_running:
        state_update.progress = 0.0
        state_update.__pydantic_fields_set__.add("progress")
    elif (
        state_update.progress is not None
        and benchmark.progress is not None
        and state_update.progress < benchmark.progress
    ):
        state_update.progress = benchmark.progress

    try:
        await benchmark.update(session, state_update)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update benchmark state: {e}"
        )

    return benchmark


async def get_benchmark_snapshot(
    session: SessionDep, mi: ModelInstance, model: Model
) -> BenchmarkSnapshot:
    # instance snapshot

    worker_snapshots = {}
    gpu_snapshots = {}
    instance_snapshots = {}

    instance_snapshots[mi.name] = create_model_instance_snapshot(mi, model)

    w: Worker = await WorkerService(session).get_by_id(mi.worker_id)
    w_snapshot, gpus_snapshots = create_worker_snapshot(w, mi.gpu_type, mi.gpu_indexes)
    if w_snapshot is not None:
        worker_snapshots[w.name] = w_snapshot
    if gpus_snapshots is not None:
        gpu_snapshots.update(gpus_snapshots)

    if mi.distributed_servers and mi.distributed_servers.subordinate_workers:
        for sub in mi.distributed_servers.subordinate_workers:
            sw: Worker = await WorkerService(session).get_by_id(sub.worker_id)
            w_snapshot, gpus_snapshots = create_worker_snapshot(
                sw, sub.gpu_type, sub.gpu_indexes
            )
            if w_snapshot is not None:
                worker_snapshots[sw.name] = w_snapshot
            if gpus_snapshots is not None:
                gpu_snapshots.update(gpus_snapshots)

    return BenchmarkSnapshot(
        instances=instance_snapshots,
        workers=worker_snapshots,
        gpus=gpu_snapshots,
    )


@router.post(
    "/{id}/metrics",
    response_model=BenchmarkPublic,
)
async def update_benchmark_metrics(
    session: SessionDep, ctx: TenantContextDep, id: int, metrics: BenchmarkMetrics
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message="Benchmark not found"
    )
    try:
        await benchmark.update(session, metrics)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update benchmark metrics: {e}"
        )

    return benchmark


@router.post(
    "/{id}/results",
    response_model=BenchmarkPublic,
)
async def update_benchmark_results(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    results: List[Dict[str, Any]],
):
    """
    Replace the benchmark's per-point results (one row per (input_tokens, rate)
    grid cell). Idempotent: existing rows for this benchmark are removed first so
    a re-run overwrites cleanly.
    """
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message="Benchmark not found"
    )
    try:
        existing = await BenchmarkResult.all_by_field(session, "benchmark_id", id)
        for row in existing:
            await row.delete(session, auto_commit=False)
        for data in results:
            await BenchmarkResult.create(
                session, source={**data, "benchmark_id": id}, auto_commit=False
            )
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(
            message=f"Failed to update benchmark results: {e}"
        )

    return benchmark


@router.get(
    "/{id}/results",
    response_model=List[BenchmarkResultPublic],
)
async def get_benchmark_results(session: SessionDep, ctx: TenantContextDep, id: int):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message="Benchmark not found"
    )
    results = await BenchmarkResult.all_by_field(session, "benchmark_id", id)
    return sorted(results, key=lambda r: (r.input_tokens or 0, r.sequence))


@router.delete(
    "/{id}",
)
async def delete_benchmark(session: SessionDep, ctx: TenantContextDep, id: int):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message="Benchmark not found"
    )

    try:
        await benchmark.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete benchmark: {e}")


@router.get("/{id}/logs")
async def get_benchmark_logs(  # noqa: C901
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    log_options: LogOptionsDep,
):
    benchmark = await Benchmark.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, benchmark, not_found_message="Benchmark not found"
    )

    worker = await Worker.one_by_id(session, benchmark.worker_id)
    if not worker:
        raise NotFoundException(message="Benchmark's worker not found")

    if benchmark.state in [
        BenchmarkStateEnum.ERROR,
        BenchmarkStateEnum.STOPPED,
        BenchmarkStateEnum.COMPLETED,
    ]:
        log_options.follow = False

    timeout = aiohttp.ClientTimeout(total=envs.PROXY_TIMEOUT, sock_connect=5)

    if log_options.follow:

        def on_exception(e: Exception, t: aiohttp.ClientTimeout) -> tuple[str, int]:
            msg = (
                str(e)
                if not isinstance(e, TimeoutError)
                else f"Log stream timed out ({t.total} seconds). Please reopen the log page."
            )
            return f"\x1b[999;1H{msg}\n", status.HTTP_500_INTERNAL_SERVER_ERROR

        return StreamingResponseWithStatusCode(
            stream_to_worker(
                worker=worker,
                method="GET",
                path=f"benchmark_logs/{benchmark.id}",
                proxy_client=request.app.state.http_client,
                no_proxy_client=request.app.state.http_client_no_proxy,
                params={
                    "tail": log_options.tail,
                    "follow": log_options.follow,
                    "benchmark_name": benchmark.name,
                },
                timeout=timeout,
                on_exception=on_exception,
                raw=True,
            ),
            media_type="application/octet-stream",
        )
    else:
        resp, body = await request_to_worker(
            worker=worker,
            method="GET",
            path=f"benchmark_logs/{benchmark.id}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            params={
                "tail": log_options.tail,
                "follow": log_options.follow,
                "benchmark_name": benchmark.name,
            },
            timeout=timeout,
        )
        return PlainTextResponse(
            content=body.decode() if body else "", status_code=resp.status
        )


@router.post("/export")
async def export_benchmarks(
    session: SessionDep,
    ctx: TenantContextDep,
    ids: list[int],
):
    if not ids:
        raise BadRequestException(message="No benchmark ids provided.")

    if len(ids) > MAX_EXPORT_RECORDS:
        raise BadRequestException(
            message=f"Export up to {MAX_EXPORT_RECORDS} records at most."
        )

    exclude_fields = [
        "id",
        "cluster_id",
        "owner_principal_id",
        "model_id",
        "worker_id",
        "created_at",
        "updated_at",
        "pid",
        "progress",
        "state_message",
        "state",
        "deleted_at",
    ]
    extra_conditions = [
        col(Benchmark.id).in_(ids),
        *cluster_resource_visibility_conditions(ctx, Benchmark),
    ]
    benchmarks: Sequence[Benchmark] = await Benchmark.all_by_fields(
        session, fields={}, extra_conditions=extra_conditions
    )
    exported_benchmarks = []
    for b in benchmarks:
        eb = b.model_dump(exclude=set(exclude_fields))
        exported_benchmarks.append(order_benchmark_export_fields(eb))

    export_data = {"benchmarks": exported_benchmarks}
    yaml_str = yaml.safe_dump(export_data, allow_unicode=True, sort_keys=False)
    return PlainTextResponse(content=yaml_str, media_type="application/x-yaml")
