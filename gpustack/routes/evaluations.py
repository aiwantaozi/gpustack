from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.evaluations import (
    Evaluation,
    EvaluationCreate,
    EvaluationListParams,
    EvaluationPublic,
    EvaluationResult,
    EvaluationSnapshot,
    EvaluationStateUpdate,
    EvaluationsPublic,
    EvaluationStateEnum,
    EvaluationTask,
    EvaluationTaskListParams,
    EvaluationTaskPublic,
    EvaluationTasksPublic,
)
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep
from gpustack.utils.snapshot import get_model_runtime_snapshot


router = APIRouter()


@router.get("", response_model=EvaluationsPublic)
async def get_evaluations(
    params: EvaluationListParams = Depends(),
    state: Optional[EvaluationStateEnum] = None,
    model_name: Optional[str] = None,
    suite_name: Optional[str] = None,
    worker_id: Optional[int] = Query(default=None),
):
    fields = {}
    if state:
        fields["state"] = state
    if model_name:
        fields["model_name"] = model_name
    if suite_name:
        fields["suite_name"] = suite_name
    if worker_id is not None:
        fields["worker_id"] = worker_id

    if params.watch:
        return StreamingResponse(
            Evaluation.streaming(fields=fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await Evaluation.paginated_by_query(
            session=session,
            fields=fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )


@router.get("/{id}", response_model=EvaluationPublic)
async def get_evaluation(session: SessionDep, id: int):
    evaluation = await Evaluation.one_by_id(session, id)
    if not evaluation:
        raise NotFoundException(message=f"Evaluation {id} not found")
    return evaluation


@router.post("", response_model=EvaluationPublic)
async def create_evaluation(session: SessionDep, evaluation_in: EvaluationCreate):
    existing = await Evaluation.one_by_field(session, "name", evaluation_in.name)
    if existing:
        raise AlreadyExistsException(
            message=f"Evaluation '{evaluation_in.name}' already exists."
        )

    if not evaluation_in.model_instance_name:
        raise BadRequestException(message="model_instance_name must be specified")

    instance = await ModelInstance.one_by_field(
        session, "name", evaluation_in.model_instance_name
    )

    if not instance:
        raise BadRequestException(message="Model instance not found")

    if instance.state != ModelInstanceStateEnum.RUNNING:
        raise BadRequestException(
            message=f"Model instance '{instance.name}' not in RUNNING state"
        )

    model = await Model.one_by_id(session, instance.model_id)
    if not model:
        raise BadRequestException(message=f"Model {instance.model_id} not found")

    mutated = Evaluation(**evaluation_in.model_dump())
    mutated.model_id = instance.model_id
    mutated.model_name = instance.model_name
    mutated.model_instance_name = instance.name
    mutated.cluster_id = instance.cluster_id
    mutated.worker_id = instance.worker_id
    mutated.snapshot = EvaluationSnapshot.model_validate(
        await get_model_runtime_snapshot(session, instance, model)
    )

    try:
        evaluation = await Evaluation.create(session, mutated)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create evaluation: {e}")

    return evaluation


@router.patch("/{id}/state", response_model=EvaluationPublic)
async def update_evaluation_state(
    session: SessionDep, id: int, state_update: EvaluationStateUpdate
):
    evaluation = await Evaluation.one_by_id(session, id)
    if not evaluation:
        raise NotFoundException(message="Evaluation not found")

    if (
        state_update.state is not None
        and state_update.state == EvaluationStateEnum.STOPPED
        and evaluation.state
        not in [
            EvaluationStateEnum.QUEUED,
            EvaluationStateEnum.PENDING,
            EvaluationStateEnum.RUNNING,
        ]
    ):
        raise BadRequestException(
            message="Only evaluations in QUEUED, PENDING, or RUNNING state can be stopped."
        )

    try:
        await evaluation.update(session, state_update)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update evaluation state: {e}"
        )

    return evaluation


@router.post("/{id}/result", response_model=EvaluationPublic)
async def update_evaluation_result(
    session: SessionDep, id: int, payload: EvaluationResult
):
    evaluation = await Evaluation.one_by_id(session, id)
    if not evaluation:
        raise NotFoundException(message="Evaluation not found")

    existing_tasks = await EvaluationTask.all_by_field(session, "evaluation_id", id)
    for task in existing_tasks:
        await task.delete(session)

    for item in payload.tasks:
        task = EvaluationTask(evaluation_id=id, **item.model_dump())
        await EvaluationTask.create(session, task)

    try:
        await evaluation.update(
            session,
            {
                "task_count": payload.task_count,
                "sample_count": payload.sample_count,
            },
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update evaluation result: {e}"
        )

    return evaluation


@router.delete("/{id}")
async def delete_evaluation(session: SessionDep, id: int):
    evaluation = await Evaluation.one_by_id(session, id)
    if not evaluation:
        raise NotFoundException(message="Evaluation not found")

    try:
        await evaluation.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete evaluation: {e}")


task_router = APIRouter()


@task_router.get("", response_model=EvaluationTasksPublic)
async def get_evaluation_tasks(
    params: EvaluationTaskListParams = Depends(),
    evaluation_id: Optional[int] = None,
    task_group: Optional[str] = None,
):
    fields = {}
    if evaluation_id is not None:
        fields["evaluation_id"] = evaluation_id
    if task_group:
        fields["task_group"] = task_group

    async with async_session() as session:
        return await EvaluationTask.paginated_by_query(
            session=session,
            fields=fields,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )


@task_router.get("/{id}", response_model=EvaluationTaskPublic)
async def get_evaluation_task(session: SessionDep, id: int):
    task = await EvaluationTask.one_by_id(session, id)
    if not task:
        raise NotFoundException(message=f"Evaluation task {id} not found")
    return task
