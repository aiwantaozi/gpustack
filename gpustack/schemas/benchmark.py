from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel
from sqlalchemy import JSON, Column
from sqlmodel import Field, ForeignKey, Integer, SQLModel, Text

from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    pydantic_column_type,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ExtendedKVCacheConfig,
    SpeculativeConfig,
)

from gpustack.schemas.workers import GPUDeviceInfo, OperatingSystemInfo


DATASET_RANDOM = "Random"
DATASET_SHAREGPT = "ShareGPT"
# Custom dataset resource (Dataset table). When dataset_name == DATASET_CUSTOM the
# benchmark references a downloaded Dataset via `dataset_id` (see schemas/datasets.py).
DATASET_CUSTOM = "Dataset"


class BenchmarkStateEnum(str, Enum):
    r"""
    Enum for Benchmark State

    Transitions:

       |- - Server - -|- - - - - - - Worker - - - - - - -|
       |              |                                  |
    PENDING ---> ---> ---> QUEUED ---> RUNNING ---> COMPLETED/STOPPED/ERROR
                              ^          ^
                              |          |
                              |----------|
                                         |
                                         |(Worker unreachable)
                                         v
                                     UNREACHABLE
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class ModelInstanceRuntimeInfo(BaseModel):
    computed_resource_claim: Optional[ComputedResourceClaim]
    ports: Optional[List[int]]

    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = None
    gpu_ids: Optional[List[str]] = None


class ModelInstanceSnapshot(ModelInstanceRuntimeInfo):
    id: int
    name: str
    resolved_path: Optional[str] = None

    # resource info
    state: Optional[str] = None
    state_message: Optional[str] = None

    # backend info
    backend: Optional[str] = None
    backend_version: Optional[str] = None
    api_detected_backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    injected_backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    image_name: Optional[str] = None
    run_command: Optional[str] = Field(sa_type=Text, default=None)
    env: Optional[Dict[str, str]] = Field(sa_type=JSON, default=None)

    # Extended KV Cache configuration. Currently maps to LMCache config in vLLM and SGLang.
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = Field(
        sa_type=pydantic_column_type(ExtendedKVCacheConfig), default=None
    )

    speculative_config: Optional[SpeculativeConfig] = Field(
        sa_type=pydantic_column_type(SpeculativeConfig), default=None
    )

    # subordinate workers info
    subordinate_workers: Optional[List[ModelInstanceRuntimeInfo]] = None


class WorkerSnapshot(BaseModel):
    id: int
    name: str
    cpu_total: Optional[int] = None
    memory_total: Optional[int] = None
    os: Optional[OperatingSystemInfo] = None


class GPUSnapshot(GPUDeviceInfo):
    id: str
    worker_id: int
    worker_name: str
    memory_total: Optional[int] = None
    core_total: Optional[int] = None


@dataclass
class BenchmarkDeploymentMetadata:
    name: str
    labels: dict[str, str]


class BenchmarkBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )

    profile: Optional[str] = Field(default="Custom")
    dataset_name: Optional[str] = Field(
        default=None
    )  # type selector / denormalized name: Random / ShareGPT / Dataset
    # Custom dataset resource reference (set when dataset_name == DATASET_CUSTOM).
    dataset_id: Optional[int] = Field(default=None)
    dataset_input_tokens: Optional[int] = Field(default=None)
    dataset_output_tokens: Optional[int] = Field(default=None)
    dataset_seed: Optional[int] = Field(default=None)
    # Multi-stage (ramp / manual) seed policy: True = each stage's seed is
    # base + stage_index (stages differ, spreading prefix/KV-cache reuse);
    # False = all stages share the base seed. Only meaningful for the Random
    # synthetic dataset — file datasets read in file order regardless of seed
    # until shuffle lands (design known-limit).
    dataset_seed_increment: Optional[bool] = Field(default=True)

    # Data distribution (P1, inspired by inference-perf): spread token lengths
    # around the mean instead of a single fixed value, for more realistic load.
    # Maps to guidellm's prompt_tokens_stdev/min/max + output_tokens_stdev/min/max.
    dataset_input_stdev: Optional[int] = Field(default=None)
    dataset_input_min: Optional[int] = Field(default=None)
    dataset_input_max: Optional[int] = Field(default=None)
    dataset_output_stdev: Optional[int] = Field(default=None)
    dataset_output_min: Optional[int] = Field(default=None)
    dataset_output_max: Optional[int] = Field(default=None)

    cluster_id: int = Field(default=None)
    model_id: Optional[int] = Field(default=None)
    model_name: Optional[str] = Field(
        default=None
    )  # denormalized field for easier query
    model_instance_name: str

    request_rate: int = Field(default=10)  # requests per second
    total_requests: Optional[int] = Field(
        default=None
    )  # total number of requests to send
    # v2.1 global duration cap (guidellm --max-seconds) for non-stage runs
    # (throughput / custom-sweep). Stage runs carry max_seconds per stage instead.
    max_seconds: Optional[float] = Field(default=None)

    # `load_type` is the load axis (knob):
    #   - fixed_rate  -> guidellm constant   (open-loop fixed req/s)
    #   - concurrency -> guidellm concurrent  (closed-loop N in-flight)
    # The latency-SLA scenario = concurrency + sla_avg_ttft_ms/sla_avg_tpot_ms set.
    load_type: Optional[str] = Field(default=None)  # fixed_rate / concurrency

    # v2.1 stages: per-stage independent constraints, so rate 1 and rate 1000 can
    # carry different limits. Each item: {rate: float, max_requests?: int,
    # max_seconds?: float}. Used only when auto_tune is off (Custom manual mode);
    # the runner does one single-rate guidellm run per stage.
    stages: Optional[List[Dict[str, Any]]] = Field(sa_type=JSON, default=None)

    # Auto-tune (adaptive ramp): when true, the runner ramps the load_type axis
    # (fixed_rate=req/s, concurrency=streams) with a geometric bracket + binary
    # search instead of running user-specified stages, and auto-detects the
    # answer. Target is derived: sla_* set -> SLA boundary (max knob meeting SLA);
    # otherwise -> throughput saturation (peak output tok/s). Replaces the old
    # guidellm `sweep` profile (removed).
    auto_tune: Optional[bool] = Field(default=None)
    # Auto-tune budget / bounds (used when auto_tune=true). None -> runner default.
    lower_bound: Optional[float] = Field(default=None)  # knob floor (default 1)
    upper_bound: Optional[float] = Field(default=None)  # knob ceiling (anti-runaway)
    # Per-point requests = max(min_requests, round(knob * multiplier)) is computed
    # by the runner's ramp engine with its own defaults (multiplier 10 conc / 30
    # rate, min_requests 30); not surfaced or stored here.
    max_points: Optional[int] = Field(default=None)  # max measured points (12)
    max_total_seconds: Optional[float] = Field(default=None)  # whole-run cap (1800)

    # Latency SLA: optional "<= threshold" targets used to pick the max load that
    # still meets the SLA. Each is independent; a point meets the SLA when every
    # SET threshold holds (AND) and success >= 95%. `sla_avg_ttft_ms` / `sla_avg_tpot_ms`
    # are the average TTFT / TPOT (kept from the original 2-field model); the p99
    # and end-to-end latency targets extend it (EvalScope-style 6 latency metrics).
    sla_avg_ttft_ms: Optional[float] = Field(default=None)  # avg TTFT (ms)
    sla_avg_tpot_ms: Optional[float] = Field(default=None)  # avg TPOT (ms)
    sla_p99_ttft_ms: Optional[float] = Field(default=None)  # p99 TTFT (ms)
    sla_p99_tpot_ms: Optional[float] = Field(default=None)  # p99 TPOT (ms)
    sla_avg_latency_ms: Optional[float] = Field(default=None)  # avg e2e latency (ms)
    sla_p99_latency_ms: Optional[float] = Field(default=None)  # p99 e2e latency (ms)

    # Shared prefix: guidellm prefix_buckets — a list of buckets, each
    # {prefix_tokens, prefix_count, bucket_weight}. A common prompt prefix shared
    # across requests (system prompt / RAG context) to exercise prefix-cache
    # reuse; can mix several prefix lengths by weight (e.g. 70% short / 30% long).
    prefix_buckets: Optional[List[Dict[str, Any]]] = Field(sa_type=JSON, default=None)

    # P1-5: max rate that still meets the SLA targets (computed in the 封装层).
    sla_met_rate: Optional[float] = Field(default=None)
    # P1-9: recommended concurrency from over-saturation detection.
    recommended_rate: Optional[float] = Field(default=None)
    # v2.1 best operating points (computed in 封装层 from the stage grid).
    peak_rate: Optional[float] = Field(default=None)  # rate at throughput peak
    knee_rate: Optional[float] = Field(
        default=None
    )  # rate at the latency-throughput knee
    # v2.1 test-coverage validity, computed in the 封装层 from the stage grid:
    # {"sufficient": bool, "warnings": [{"code": str, "params": {...}}]}. Drives
    # the detail page's coverage warning banner.
    validity: Optional[Dict[str, Any]] = Field(sa_type=JSON, default=None)
    # P1-10: multi-turn conversation length (guidellm `--data turns=N`).
    turns: Optional[int] = Field(default=None)
    # P1-12: warmup / cooldown (numeric: <1 = percent, >=1 = absolute count/seconds).
    warmup: Optional[float] = Field(default=None)
    cooldown: Optional[float] = Field(default=None)
    # P1-12: stopping constraints.
    max_errors: Optional[int] = Field(default=None)
    max_error_rate: Optional[float] = Field(default=None)
    stop_on_saturation: Optional[bool] = Field(default=None)

    # Benchmark state fields
    state: BenchmarkStateEnum = Field(
        default=BenchmarkStateEnum.PENDING,
        index=True,
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: Optional[float] = Field(default=None)
    worker_id: Optional[int] = Field(default=None)
    pid: Optional[int] = Field(default=None)

    def get_deployment_metadata(
        self,
    ) -> Optional[BenchmarkDeploymentMetadata]:
        """
        Get the deployment metadata for the benchmark.
        """

        return BenchmarkDeploymentMetadata(
            name=self.name,
            labels={
                "benchmark-name": self.name,
                "model-instance-name": self.model_instance_name or "",
                "type": "benchmark",
            },
        )


ModelInstanceSnapshots = Dict[str, ModelInstanceSnapshot]
WorkerSnapshots = Dict[str, WorkerSnapshot]
GPUSnapshots = Dict[str, GPUSnapshot]


class DatasetSnapshot(BaseModel):
    """Readable snapshot of the custom dataset used by a benchmark.

    Captured at creation so the detail/list can always show *which* dataset was
    used (and its column mapping) — self-contained, even if the Dataset resource
    is later deleted or lives on another page. `dataset_id` still references the
    live resource for the run-time mount.
    """

    dataset_id: Optional[int] = None
    source: Optional[str] = None  # huggingface / model_scope / local_path
    readable_source: Optional[str] = None  # e.g. "org/name/file.jsonl"
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    model_scope_model_id: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None


class BenchmarkSnapshot(BaseModel):
    instances: Optional[ModelInstanceSnapshots] = None
    workers: Optional[WorkerSnapshots] = None
    gpus: Optional[GPUSnapshots] = None
    # Readable snapshot of the custom dataset (dataset_name == "Dataset"); None
    # for Random / ShareGPT.
    dataset: Optional[DatasetSnapshot] = None


class BenchmarkMetricsLite(SQLModel):
    requests_per_second_mean: Optional[float] = Field(
        default=None, description="Mean requests per second (unit: req/s)"
    )
    request_latency_mean: Optional[float] = Field(
        default=None, description="Mean request latency (unit: seconds)"
    )
    time_per_output_token_mean: Optional[float] = Field(
        default=None, description="Mean time per output token (unit: ms)"
    )
    inter_token_latency_mean: Optional[float] = Field(
        default=None, description="Mean inter-token latency (unit: ms)"
    )
    time_to_first_token_mean: Optional[float] = Field(
        default=None, description="Mean time to first token (unit: ms)"
    )
    # P99 percentiles for the SLA-relevant latency metrics (populated from
    # guidellm's per-point percentiles). Used to evaluate p99 SLA thresholds.
    time_to_first_token_p99: Optional[float] = Field(
        default=None, description="P99 time to first token (unit: ms)"
    )
    time_per_output_token_p99: Optional[float] = Field(
        default=None, description="P99 time per output token (unit: ms)"
    )
    request_latency_p99: Optional[float] = Field(
        default=None, description="P99 request latency (unit: seconds)"
    )
    tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean tokens per second (unit: tok/s)"
    )
    output_tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean output tokens per second (unit: tok/s)"
    )
    input_tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean prompt tokens per second (unit: tok/s)"
    )
    request_concurrency_mean: Optional[float] = Field(
        default=None,
        description="Mean request concurrency (unit: number of concurrent requests)",
    )
    request_concurrency_max: Optional[float] = Field(
        default=None,
        description="Max request concurrency (unit: number of concurrent requests)",
    )
    request_total: Optional[int] = Field(
        default=None, description="Total number of requests made"
    )
    request_successful: Optional[int] = Field(
        default=None, description="Total number of successful requests"
    )
    request_errored: Optional[int] = Field(
        default=None, description="Total number of errored requests"
    )
    request_incomplete: Optional[int] = Field(
        default=None, description="Total number of incomplete requests"
    )


class BenchmarkMetrics(BenchmarkMetricsLite):
    raw_metrics: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON), default=None
    )  # deferred loading of potentially large field


class BenchmarkResultBase(BenchmarkMetricsLite):
    """
    One measured point of a benchmark task: a single (input_tokens, rate) cell.

    A benchmark task produces N x M of these (N input lengths x M rates). The
    parent `Benchmark` row keeps a single "representative" point (global throughput
    peak) in its flat metric columns for list/sort; the full grid lives here.
    """

    benchmark_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("benchmarks.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )
    )
    # Grid coordinates
    input_tokens: Optional[int] = Field(
        default=None
    )  # input-length axis (one guidellm run)
    rate: Optional[float] = Field(
        default=None
    )  # concurrency (concurrent) or req/s (constant/poisson)
    strategy_type: Optional[str] = Field(
        default=None
    )  # concurrent / constant / poisson / ...
    sequence: int = Field(default=0)  # run_index, for ordering / aligning benchmarks[i]


class BenchmarkResult(BenchmarkResultBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    raw_metrics: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON), default=None
    )  # this point's benchmarks[i] dump (includes percentiles)

    __tablename__ = 'benchmark_results'


class BenchmarkResultPublic(BenchmarkResultBase):
    id: int
    created_at: datetime
    updated_at: datetime
    # This point's benchmarks[i] dump (includes percentiles) for stage drill-down.
    # No request samples are stored (runner uses --sample-requests 0), so it stays
    # a moderate size per stage.
    raw_metrics: Optional[Dict[str, Any]] = None


class BenchmarkWithSnapshots(BenchmarkBase):
    snapshot: Optional[BenchmarkSnapshot] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(BenchmarkSnapshot)),
    )
    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class Benchmark(BenchmarkWithSnapshots, BenchmarkMetrics, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    # Tenant scope. Server-derived from cluster on creation.
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("principals.id"), nullable=True),
    )

    __tablename__ = 'benchmarks'


class BenchmarkListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "dataset_name",
        "model_name",
        "state",
        "created_at",
        "updated_at",
        # metrics fields
        "requests_per_second_mean",
        "request_latency_mean",
        "time_per_output_token_mean",
        "inter_token_latency_mean",
        "time_to_first_token_mean",
        "tokens_per_second_mean",
        "output_tokens_per_second_mean",
        "input_tokens_per_second_mean",
        "request_concurrency_mean",
        "request_concurrency_max",
        "request_total",
        "request_successful",
        "request_errored",
        "request_incomplete",
    ]


class BenchmarkCreate(BenchmarkBase):
    pass


class BenchmarkUpdate(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )


class BenchmarkStateUpdate(SQLModel):
    state: Optional[BenchmarkStateEnum] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    pid: Optional[int] = Field(default=None)
    progress: Optional[float] = None
    # v2.1 best operating points (computed by the worker 封装层 from the stage
    # grid; only the explicitly-set ones are persisted via model_fields_set).
    peak_rate: Optional[float] = None
    knee_rate: Optional[float] = None
    sla_met_rate: Optional[float] = None
    recommended_rate: Optional[float] = None
    validity: Optional[Dict[str, Any]] = None


class BenchmarkFullPublic(
    BenchmarkWithSnapshots,
    BenchmarkMetrics,
):
    id: int
    # The owning Org. Server-derived from the cluster on create and
    # therefore kept out of BenchmarkBase / Create — declared on the
    # Public schemas so readers can render the owning Org.
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class BenchmarkPublic(
    BenchmarkWithSnapshots,
    BenchmarkMetricsLite,
):
    id: int
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


BenchmarksPublic = PaginatedList[BenchmarkPublic]
