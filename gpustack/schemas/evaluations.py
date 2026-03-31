from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.benchmark import (
    GPUSnapshot,
    ModelInstanceSnapshot,
    WorkerSnapshot,
)
from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    pydantic_column_type,
)


class EvaluationStateEnum(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


EvaluationModelInstanceSnapshots = Dict[str, ModelInstanceSnapshot]
EvaluationWorkerSnapshots = Dict[str, WorkerSnapshot]
EvaluationGPUSnapshots = Dict[str, GPUSnapshot]


class EvaluationSnapshot(SQLModel):
    instances: Optional[EvaluationModelInstanceSnapshots] = None
    workers: Optional[EvaluationWorkerSnapshots] = None
    gpus: Optional[EvaluationGPUSnapshots] = None


class EvaluationBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )
    suite_id: str
    suite_name: str
    category: Optional[str] = Field(default=None)
    model_id: Optional[int] = Field(default=None)
    model_name: Optional[str] = Field(default=None)
    model_instance_name: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None)
    worker_id: Optional[int] = Field(default=None)
    pid: Optional[int] = Field(default=None)
    state: EvaluationStateEnum = Field(
        default=EvaluationStateEnum.PENDING,
        index=True,
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: Optional[float] = Field(default=None)
    task_count: Optional[int] = Field(default=None)
    sample_count: Optional[int] = Field(default=None)

    def get_deployment_metadata(self) -> "EvaluationDeploymentMetadata":
        return EvaluationDeploymentMetadata(
            name=self.name,
            labels={
                "evaluation-name": self.name,
                "model-instance-name": self.model_instance_name or "",
                "type": "evaluation",
            },
        )


class EvaluationWithSnapshots(EvaluationBase):
    snapshot: Optional[EvaluationSnapshot] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(EvaluationSnapshot)),
    )


class Evaluation(EvaluationWithSnapshots, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    __tablename__ = 'evaluations'


class EvaluationListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "suite_name",
        "model_name",
        "state",
        "task_count",
        "created_at",
        "updated_at",
    ]


class EvaluationCreate(EvaluationBase):
    pass


class EvaluationUpdate(SQLModel):
    name: Optional[str] = Field(default=None, index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )
    pid: Optional[int] = None
    state: Optional[EvaluationStateEnum] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: Optional[float] = None
    task_count: Optional[int] = None
    sample_count: Optional[int] = None


class EvaluationPublic(EvaluationWithSnapshots):
    id: int
    created_at: datetime
    updated_at: datetime


EvaluationsPublic = PaginatedList[EvaluationPublic]


class EvaluationTaskBase(SQLModel):
    evaluation_id: int = Field(index=True)
    task_name: str = Field(index=True)
    task_alias: Optional[str] = None
    task_group: Optional[str] = Field(default=None, index=True)
    display_name: Optional[str] = None
    dataset_name: Optional[str] = None
    version: Optional[str] = None
    sample_count: Optional[int] = None
    n_shot: Optional[int] = None
    output_type: Optional[str] = None
    primary_metric_key: Optional[str] = Field(default=None, index=True)
    primary_metric_value: Optional[float] = Field(default=None, index=True)
    primary_stderr: Optional[str] = None
    primary_higher_is_better: Optional[bool] = None
    raw_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    config_snapshot: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )
    task_metadata: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )
    duration_seconds: Optional[float] = None


class EvaluationTask(EvaluationTaskBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    __tablename__ = 'evaluation_tasks'


class EvaluationTaskListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "task_name",
        "task_group",
        "primary_metric_key",
        "primary_metric_value",
        "sample_count",
        "n_shot",
        "created_at",
        "updated_at",
    ]


class EvaluationTaskPublic(EvaluationTaskBase):
    id: int
    created_at: datetime
    updated_at: datetime


EvaluationTasksPublic = PaginatedList[EvaluationTaskPublic]


class EvaluationStateUpdate(SQLModel):
    pid: Optional[int] = None
    state: Optional[EvaluationStateEnum] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: Optional[float] = None
    task_count: Optional[int] = None
    sample_count: Optional[int] = None


class EvaluationTaskUpsert(SQLModel):
    task_name: str
    task_alias: Optional[str] = None
    task_group: Optional[str] = None
    display_name: Optional[str] = None
    dataset_name: Optional[str] = None
    version: Optional[str] = None
    sample_count: Optional[int] = None
    n_shot: Optional[int] = None
    output_type: Optional[str] = None
    primary_metric_key: Optional[str] = None
    primary_metric_value: Optional[float] = None
    primary_stderr: Optional[str] = None
    primary_higher_is_better: Optional[bool] = None
    raw_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    config_snapshot: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )
    task_metadata: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON)
    )
    duration_seconds: Optional[float] = None


class EvaluationResult(SQLModel):
    task_count: Optional[int] = None
    sample_count: Optional[int] = None
    tasks: List[EvaluationTaskUpsert] = Field(default_factory=list)


@dataclass
class EvaluationDeploymentMetadata:
    name: str
    labels: dict[str, str]
