import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import func, or_

from gpustack.api.exceptions import (
    BadRequestException,
    InternalServerErrorException,
)
from gpustack.api.tenant import (
    bypass_tenant_filter,
    assert_cluster_resource_visible,
    cluster_resource_visibility_conditions,
    cluster_scoped_system,
    scoped_cluster_row_visible,
)
from gpustack.schemas.datasets import (
    Dataset,
    DatasetCreate,
    DatasetListParams,
    DatasetPublic,
    DatasetStateEnum,
    DatasetUpdate,
    DatasetsPublic,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep

logger = logging.getLogger(__name__)

router = APIRouter()


def _make_dataset_visibility_filter(ctx):
    def _visible(d: Dataset) -> bool:
        if cluster_scoped_system(ctx):
            return scoped_cluster_row_visible(ctx, d)
        if bypass_tenant_filter(ctx):
            return True
        org_id = getattr(d, "owner_principal_id", None)
        if (
            ctx.current_principal_id is not None
            and org_id is not None
            and org_id == ctx.current_principal_id
        ):
            return True
        if getattr(d, "cluster_id", None) in ctx.accessible_cluster_ids:
            return True
        return False

    return _visible


def _dataset_search_clause(search: str):
    lower_search = search.lower()
    return or_(
        *[
            func.lower(Dataset.huggingface_repo_id).like(f"%{lower_search}%"),
            func.lower(Dataset.huggingface_filename).like(f"%{lower_search}%"),
            func.lower(Dataset.model_scope_model_id).like(f"%{lower_search}%"),
            func.lower(Dataset.model_scope_file_path).like(f"%{lower_search}%"),
            func.lower(Dataset.local_path).like(f"%{lower_search}%"),
        ]
    )


def search_dataset_filter(data: Dataset, search: str) -> bool:
    s = search.lower()
    for value in (
        data.huggingface_repo_id,
        data.huggingface_filename,
        data.model_scope_model_id,
        data.model_scope_file_path,
        data.local_path,
    ):
        if value and s in value.lower():
            return True
    return False


@router.get("", response_model=DatasetsPublic)
async def get_datasets(
    ctx: TenantContextDep,
    params: DatasetListParams = Depends(),
    search: str = None,
    worker_id: int = None,
    state: str = None,
):
    fields = {}
    if worker_id:
        fields["worker_id"] = worker_id
    if state:
        fields["state"] = state
    visible = _make_dataset_visibility_filter(ctx)

    if params.watch:
        filter_func = (
            (lambda data: visible(data) and search_dataset_filter(data, search))
            if search
            else visible
        )
        return StreamingResponse(
            Dataset.streaming(fields=fields, filter_func=filter_func),
            media_type="text/event-stream",
        )

    extra_conditions = list(cluster_resource_visibility_conditions(ctx, Dataset))
    if search:
        extra_conditions.append(_dataset_search_clause(search))

    async with async_session() as session:
        return await Dataset.paginated_by_query(
            session=session,
            fields=fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )


@router.get("/{id}", response_model=DatasetPublic)
async def get_dataset(session: SessionDep, ctx: TenantContextDep, id: int):
    dataset = await Dataset.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, dataset, not_found_message=f"Dataset {id} not found"
    )
    return dataset


@router.post("", response_model=DatasetPublic)
async def create_dataset(
    session: SessionDep, ctx: TenantContextDep, dataset_in: DatasetCreate
):
    if dataset_in.worker_id is None:
        raise BadRequestException(message="Field worker_id must be specified")

    source_index = dataset_in.dataset_source_index
    existing = await Dataset.one_by_fields(
        session,
        {"worker_id": dataset_in.worker_id, "source_index": source_index},
    )
    if existing:
        # Dedup: reuse the same source on the same worker rather than re-download.
        return existing

    # Derive tenant scope from the targeted worker → cluster.
    cluster_id: Optional[int] = None
    owner_principal_id: Optional[int] = None
    worker = await Worker.one_by_id(session, dataset_in.worker_id)
    if worker is None:
        raise BadRequestException(message=f"Worker {dataset_in.worker_id} not found")
    cluster_id = worker.cluster_id
    owner_principal_id = getattr(worker, "owner_principal_id", None)

    try:
        dataset = Dataset(
            **dataset_in.model_dump(),
            source_index=source_index,
            cluster_id=cluster_id,
            owner_principal_id=owner_principal_id,
        )
        dataset = await Dataset.create(session, dataset)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create dataset: {e}")

    return dataset


@router.put("/{id}", response_model=DatasetPublic)
async def update_dataset(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    dataset_in: DatasetUpdate,
):
    dataset = await Dataset.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, dataset, not_found_message=f"Dataset {id} not found"
    )
    try:
        await dataset.update(session, dataset_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update dataset: {e}")
    return dataset


@router.delete("/{id}")
async def delete_dataset(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    cleanup: Optional[bool] = None,
):
    dataset = await Dataset.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, dataset, not_found_message=f"Dataset {id} not found"
    )
    try:
        if cleanup is not None and dataset.cleanup_on_delete != cleanup:
            dataset.cleanup_on_delete = cleanup
            await dataset.update(session)
        await dataset.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete dataset: {e}")


@router.post("/{id}/reset", response_model=DatasetPublic)
async def reset_dataset(session: SessionDep, ctx: TenantContextDep, id: int):
    dataset = await Dataset.one_by_id(session, id)
    assert_cluster_resource_visible(
        ctx, dataset, not_found_message=f"Dataset {id} not found"
    )
    try:
        dataset.state = DatasetStateEnum.DOWNLOADING
        dataset.download_progress = 0
        dataset.state_message = ""
        await dataset.update(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to reset dataset: {e}")
    return dataset
