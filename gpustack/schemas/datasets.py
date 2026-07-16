import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, model_validator
from sqlmodel import (
    JSON,
    BigInteger,
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    Text,
)

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList
from gpustack.schemas.models import SourceEnum


class DatasetStateEnum(str, Enum):
    ERROR = "error"
    DOWNLOADING = "downloading"
    READY = "ready"


class DatasetSource(BaseModel):
    """Where a benchmark dataset comes from.

    Mirrors ``ModelSource`` (huggingface / model_scope / local_path) so datasets
    reuse the model-deployment "download first" mental model, exactly like
    ``ModelFile``: when a single-file field (``huggingface_filename`` /
    ``model_scope_file_path``) is set the worker downloads just that one file
    (single-file mode, e.g. ShareGPT's one JSON); otherwise it snapshots the whole
    dataset repo (whole-dataset mode, reserved mainly for future model evaluation).

    Load-time selectors (split / config / subset) are intentionally NOT modeled
    here — they are a consumer-side concern, not a download concern.
    """

    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    model_scope_model_id: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None

    @property
    def is_single_file(self) -> bool:
        if self.source == SourceEnum.HUGGING_FACE:
            return bool(self.huggingface_filename)
        if self.source == SourceEnum.MODEL_SCOPE:
            return bool(self.model_scope_file_path)
        return False

    @property
    def readable_source(self) -> str:
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend([self.model_scope_model_id, self.model_scope_file_path])
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])
        return "/".join([value for value in values if value is not None])

    @property
    def dataset_source_index(self) -> str:
        """Unique identity of the dataset source (dedup key).

        Includes the single-file path so one file != the whole repo.
        """
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend(
                [
                    self.source,
                    self.model_scope_model_id,
                    self.model_scope_file_path,
                ]
            )
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        filtered_values = [str(v) for v in values if v is not None]
        source_string = "/".join(filtered_values)
        return hashlib.sha256(source_string.encode()).hexdigest()

    @model_validator(mode="after")
    def check_source_fields(self):
        if self.source == SourceEnum.HUGGING_FACE:
            if not self.huggingface_repo_id:
                raise ValueError(
                    "huggingface_repo_id must be provided when source is 'huggingface'"
                )
        if self.source == SourceEnum.MODEL_SCOPE:
            if not self.model_scope_model_id:
                raise ValueError(
                    "model_scope_model_id must be provided when source is 'model_scope'"
                )
        if self.source == SourceEnum.LOCAL_PATH:
            if not self.local_path:
                raise ValueError(
                    "local_path must be provided when source is 'local_path'"
                )
        return self

    model_config = ConfigDict(protected_namespaces=())


class DatasetBase(SQLModel, DatasetSource):
    local_dir: Optional[str] = None
    worker_id: Optional[int] = None
    cleanup_on_delete: Optional[bool] = None

    size: Optional[int] = Field(sa_column=Column(BigInteger), default=None)
    download_progress: Optional[float] = None
    # Local path(s) resolved on the worker after download (single-file mode: the
    # file; whole-dataset: the dir). benchmark-runner's --data uses
    # resolved_paths[0] — mirrors ModelFile, no separate data_path field.
    resolved_paths: List[str] = Field(sa_column=Column(JSON), default=[])

    # Schema preview, probed by the worker on READY: detected column
    # names + a few truncated sample rows. Drive the column-mapping UI.
    columns: Optional[List[str]] = Field(sa_column=Column(JSON), default=None)
    sample_rows: Optional[List[Dict[str, Any]]] = Field(
        sa_column=Column(JSON), default=None
    )
    # Non-empty when column probing failed (dataset still usable via guidellm
    # auto-detect; UI falls back to manual column entry).
    inspect_error: Optional[str] = Field(default=None, sa_column=Column(Text))

    # Logical-column -> actual-column overrides for guidellm's --data-column-mapper.
    # Empty/None -> rely on guidellm auto-detection.
    column_mapping: Optional[Dict[str, str]] = Field(
        sa_column=Column(JSON), default=None
    )

    state: DatasetStateEnum = DatasetStateEnum.DOWNLOADING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class Dataset(DatasetBase, BaseModelMixin, table=True):
    __tablename__ = 'datasets'
    id: Optional[int] = Field(default=None, primary_key=True)

    # Unique index of the dataset source (dedup within a worker via create route).
    source_index: Optional[str] = Field(index=True, default=None)

    # Tenant scope. Server-derived from worker→cluster on creation; kept off the
    # create payload so clients can't smuggle overrides.
    cluster_id: Optional[int] = Field(default=None)
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("principals.id"), nullable=True),
    )


class DatasetListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "source",
        "worker_id",
        "state",
        "created_at",
        "updated_at",
    ]


class DatasetCreate(DatasetBase):
    pass


class DatasetUpdate(DatasetBase):
    pass


class DatasetPublic(DatasetBase):
    id: int
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


DatasetsPublic = PaginatedList[DatasetPublic]
