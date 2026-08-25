import copy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from croniter import croniter
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)
from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy import false as sa_false
from sqlalchemy.orm import selectinload
from sqlmodel import Field, Relationship, SQLModel, Text, select

from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.links import (
    ModelInstanceDraftModelFileLink,
    ModelInstanceModelFileLink,
)
from gpustack.utils.command import find_parameter, find_bool_parameter
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    AccessPolicyEnum,
)
from gpustack.schemas.principals import _platform_principal_id
from gpustack.schemas.cache_services import CacheConfigSnapshot

if TYPE_CHECKING:
    from gpustack.schemas.model_files import ModelFile
    from gpustack.schemas.clusters import Cluster

# Models


class SourceEnum(str, Enum):
    HUGGING_FACE = "huggingface"
    MODEL_SCOPE = "model_scope"
    LOCAL_PATH = "local_path"


class CategoryEnum(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    IMAGE = "image"
    RERANKER = "reranker"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    UNKNOWN = "unknown"


class PlacementStrategyEnum(str, Enum):
    SPREAD = "spread"
    BINPACK = "binpack"


class BackendEnum(str, Enum):
    VLLM = "vLLM"
    VOX_BOX = "VoxBox"
    ASCEND_MINDIE = "MindIE"
    SGLANG = "SGLang"
    CUSTOM = "Custom"


class BackendSourceEnum(str, Enum):
    CUSTOM = "custom"
    BUILT_IN = "built_in"
    COMMUNITY = "community"


class SpeculativeAlgorithmEnum(str, Enum):
    EAGLE3 = "eagle3"
    MTP = "mtp"
    NGRAM = "ngram"


class GPUSelector(BaseModel):
    # format of each element: "worker_name:device:gpu_index", example: "worker1:cuda:0"
    gpu_ids: Optional[List[str]] = None
    gpus_per_replica: Optional[int] = None


class GPUTypeSelector(BaseModel):
    """
    Selects a sliced GPU from a gpustack-operator InstanceType pool.

    Field names mirror ``GPUInstanceResources`` / the operator's
    InstanceResources conventions.

    Mutually exclusive with manual GPU selection: ``gpu_selector.gpu_ids`` must
    be empty, since the card is chosen by the operator's device plugin, not by
    index. A ``gpu_selector`` is otherwise allowed — this implies exactly one
    card per worker per replica, so ``gpus_per_replica`` is constrained to 1
    rather than rejected.
    """

    type: str
    """
    Name of the operator InstanceType (pool) to schedule onto.
    """

    accelerator_sliced_memory_percentage: Optional[int] = Field(
        default=None, ge=0, le=100
    )
    """
    Per-card VRAM budget requested on a sliced InstanceType, as a percentage.
    Required (in [1,100]) for a sliced request; 0 is valid only together with
    a 0/unset cores percentage and means a whole-card exclusive request.
    """

    accelerator_sliced_cores_percentage: Optional[int] = Field(
        default=None, ge=0, le=100
    )
    """
    Per-card compute budget requested on a sliced InstanceType, as a
    percentage in [1,100]; an independent dimension from memory. Defaults to
    100 when unset on a sliced request (operator webhook defaulting rule). 0
    is valid only together with a 0/unset memory percentage (whole-card
    exclusive).
    """

    accelerator_partitioned_profile: Optional[str] = None
    """
    Hardware partition profile requested on a partition-offering InstanceType,
    e.g. "1g.5gb". Mutually exclusive with non-zero slice percentages:
    hardware partitioning and software slicing cannot both apply to one card.
    """

    @model_validator(mode="after")
    def normalize_slice_percentages(self):
        if self.accelerator_partitioned_profile:
            # Slicing percentages don't apply to hardware partitioning; their
            # exclusivity with a profile is enforced by route validation.
            return self

        memory = self.accelerator_sliced_memory_percentage
        cores = self.accelerator_sliced_cores_percentage
        memory_sliced = memory is not None and memory > 0
        cores_sliced = cores is not None and cores > 0

        if not memory_sliced and not cores_sliced:
            # Whole-card exclusive mode: valid only as both-0 (or both-unset);
            # normalize unset to 0.
            self.accelerator_sliced_memory_percentage = 0
            self.accelerator_sliced_cores_percentage = 0
            return self

        if not memory_sliced:
            # Covers both "cores set, memory unset" and the mixed
            # "memory 0, cores non-zero" case: memory is required (and
            # non-zero) for any sliced request.
            raise ValueError(
                "accelerator_sliced_memory_percentage is required in the "
                "range 1-100 for a sliced request; 0 is only valid when both "
                "percentages are 0 (whole-card exclusive)"
            )
        if cores is not None and not cores_sliced:
            raise ValueError(
                "accelerator_sliced_cores_percentage must be in the range "
                "1-100; 0 is only valid when both percentages are 0 "
                "(whole-card exclusive)"
            )
        if cores is None:
            # Mirror the operator webhook: cores defaults to 100 when unset.
            self.accelerator_sliced_cores_percentage = 100
        return self


class LoraListEntry(BaseModel):
    """
    One LoRA adapter configured on a base Model (download + runtime + optional route).
    """

    lora_name: str = Field(..., min_length=1)
    """Fully-qualified LoRA id in the form "<base_model_name>:<suffix>". The API
    strips the prefix on the way out (see ModelPublic._strip_lora_prefix), so
    clients only ever see/enter the bare short name."""

    lora_repo_name: Optional[str] = None
    """HuggingFace repo id, ModelScope model id, or absolute filesystem path
    (used as a fallback when source=local_path and local_path is empty)."""

    source: str = SourceEnum.HUGGING_FACE.value
    huggingface_filename: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None

    # Runtime fields populated when mounted on an instance.
    path: Optional[str] = None
    """Resolved filesystem path when mounted on an instance."""
    model_file_id: Optional[int] = None
    """ID of the ModelFile record backing this adapter."""


class KVCacheModeEnum(str, Enum):
    LOCAL = "local"
    SHARED = "shared"

    def __str__(self):
        return self.value


class ExtendedKVCacheConfig(BaseModel):
    enabled: bool = False
    """ Enable extended KV cache for the model."""

    mode: Optional[KVCacheModeEnum] = KVCacheModeEnum.LOCAL
    """ "local": per-instance cache offloaded to CPU memory. "shared": attach to a shared cache service. Absent means local. """

    cache_service_id: Optional[int] = None
    """ ID of the CacheService to attach to. Required when mode is "shared". """

    ram_ratio: Optional[float] = 1.2
    """ RAM-to-VRAM ratio for KV cache. For example, 2.0 means the RAM is twice the size of the VRAM. """

    ram_size: Optional[int] = None
    """ Maximum size of the KV cache to be stored in local CPU memory (unit: GiB). Overrides ram_ratio if both are set. """

    chunk_size: Optional[int] = None
    """ Size for each KV cache chunk (unit: number of tokens). """

    def is_shared(self) -> bool:
        return bool(self.enabled and self.mode == KVCacheModeEnum.SHARED)

    def is_local(self) -> bool:
        return bool(self.enabled and not self.is_shared())


# A window may span at most a year. Longer values are meaningless for a
# recurring schedule and overflow the timedelta used to compute the window end.
MAX_SCALING_WINDOW_SECONDS = 366 * 24 * 3600


def _assert_satisfiable_cron(expr: str) -> None:
    """Reject cron expressions that parse but can never fire.

    ``croniter.is_valid`` accepts impossible dates such as ``0 0 30 2 *``
    (February 30th); the window would simply never open while the scheduler
    logged an evaluation failure on every tick. Resolve an occurrence to prove
    the expression is reachable.
    """
    try:
        croniter(expr).get_next(datetime)
    except Exception as e:
        raise ValueError(f"Invalid cron expression: {expr!r} ({e})")


class ScalingScheduleRule(BaseModel):
    """
    One scheduled-scaling window (GCP scaling-schedule / KEDA Cron scaler
    semantics). ``start_cron`` fires the window open; the window stays open for
    ``duration_seconds``. While ``now`` falls inside the window the model's
    replicas is driven to this rule's ``replicas``. Outside every rule's window
    the model falls back to the schedule's ``baseline_replicas``. Multiple rules
    cover multiple windows (e.g. day / night). A start + duration model (rather
    than start + end) expresses windows that cross midnight / span whole days
    (e.g. a weekend) without wrap-around ambiguity.
    """

    start_cron: str = ""
    """Cron marking the window start, e.g. "0 8 * * *" (every day at 08:00)."""
    duration_seconds: Optional[int] = Field(
        default=None, gt=0, le=MAX_SCALING_WINDOW_SECONDS
    )
    """How long the window stays open after ``start_cron`` fires, in seconds.
    Capped at a year: the window end is computed as a ``timedelta``, which
    overflows (and would surface as a 500) for astronomically large values."""
    replicas: int = Field(ge=0)
    """Desired replica count while ``now`` is inside this window."""
    name: Optional[str] = None
    """Optional human-readable label, e.g. "daytime"."""

    @field_validator("start_cron")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        # Empty is allowed for a not-yet-filled rule (e.g. a disabled schedule
        # or a freshly added row). Enabled schedules require a non-empty cron
        # for every rule — that check lives in ScalingSchedule below.
        if v:
            _assert_satisfiable_cron(v)
        return v


class ScalingSchedule(BaseModel):
    """Scheduled scaling configuration attached to a Model."""

    enabled: bool = False
    """Whether scheduled scaling drives this model's replicas."""
    baseline_replicas: Optional[int] = Field(default=None, ge=0)
    """Replica count when ``now`` is outside every rule window. Required while
    the schedule is enabled — together with ``rules`` it is the sole input to
    the effective replica count, and the model's ``replicas`` field becomes a
    scheduler-driven value rather than a user setting."""
    rules: List[ScalingScheduleRule] = Field(default_factory=list)
    """Window rules. Order does not matter: when windows overlap the one that
    started most recently wins, and windows sharing a start instant resolve to
    the largest replica count."""

    @model_validator(mode="after")
    def validate_schedule(self):
        # Only a live schedule is held to the "every rule must have valid
        # crons" bar. A disabled schedule may carry incomplete rows (they are
        # ignored at runtime), so don't 422 on them — this also keeps the
        # real-time preview from rejecting in-progress edits.
        if not self.enabled:
            return self

        if self.baseline_replicas is None:
            raise ValueError(
                "baseline_replicas is required when scaling schedule is enabled."
            )

        if not self.rules:
            raise ValueError(
                "At least one rule is required when scaling schedule is enabled."
            )

        # Field validators already proved every non-empty cron is satisfiable and
        # every duration positive; a live schedule additionally requires both to
        # be present on every rule.
        for rule in self.rules:
            if not rule.start_cron:
                raise ValueError(
                    "start_cron is required for every rule when scaling schedule "
                    "is enabled."
                )
            if not rule.duration_seconds:
                raise ValueError(
                    "duration_seconds is required for every rule when scaling "
                    "schedule is enabled."
                )
        return self


class ModelSource(BaseModel):
    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    model_scope_model_id: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None

    @property
    def model_source_key(self) -> str:
        """Returns a unique identifier for the model, independent of quantization."""
        if self.source == SourceEnum.HUGGING_FACE:
            return self.huggingface_repo_id or ""
        elif self.source == SourceEnum.MODEL_SCOPE:
            return self.model_scope_model_id or ""
        elif self.source == SourceEnum.LOCAL_PATH:
            return self.local_path or ""
        return ""

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
    def model_source_index(self) -> str:
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend(
                [self.source, self.model_scope_model_id, self.model_scope_file_path]
            )
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        # Filter out None values and join
        filtered_values = [v for v in values if v is not None]
        source_string = "/".join(filtered_values)
        return hashlib.sha256(source_string.encode()).hexdigest()

    @model_validator(mode="after")
    def check_huggingface_fields(self):
        if self.source == SourceEnum.HUGGING_FACE:
            if not self.huggingface_repo_id:
                raise ValueError(
                    "huggingface_repo_id must be provided "
                    "when source is 'huggingface'"
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


class SpeculativeConfig(BaseModel):
    """Configuration for speculative decoding."""

    enabled: bool = False
    """Whether speculative decoding is enabled."""
    algorithm: Optional[SpeculativeAlgorithmEnum] = None
    """The algorithm to use for speculative decoding."""
    draft_model: Optional[str] = None
    """The draft model to use for speculative decoding.

    It can be a draft model name from the model catalog, a local path or a model ID from the main model source."""
    num_draft_tokens: Optional[int] = None
    """The number of draft tokens."""
    # For ngram only
    ngram_min_match_length: Optional[int] = None
    """Minimum length of the n-gram to match."""
    ngram_max_match_length: Optional[int] = None
    """Maximum length of the n-gram to match."""


# Prefill/decode disaggregation. A Model with `roles` set is a *group*: one
# pool, one router, one generation at a time.


class RoleNameEnum(str, Enum):
    """Role names accepted in phase one.

    The data model allows any name — the API validation layer is what limits
    it to these three. Phase two opens up encoder / draft.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    ROUTER = "router"

    def __str__(self):
        return self.value


class PortBand(BaseModel):
    """A contiguous run of ports, not a single point.

    Some KV connectors derive several ports from one base — NIXL's side
    channel takes one per tensor-parallel rank — so a port declaration has to
    carry its width as well as its base.
    """

    base: int
    count: int = 1


class RoleSpec(BaseModel):
    """One role of a multi-role deployment.

    Every deployment field left as ``None`` inherits the ``Model``-level field
    of the same name; giving it a value overrides it. The override surface is
    deliberately the *whole* of ``backend_parameters`` and ``env`` rather than
    a PD-specific subset: measured on Ascend 910B2, prefill and decode differ
    in nearly every performance-related parameter, down to
    ``HCCL_CONNECT_TIMEOUT`` (120 vs 1200) and ``HCCL_BUFFSIZE`` (2560 vs
    1024). Any narrower surface runs out immediately.

    Note that the inherit-when-None rule is not applied here: the projection
    onto an effective per-role Model happens on the read path, and its result
    is deliberately never persisted so that one intent has one source of
    truth.
    """

    name: str
    replicas: int = Field(default=1, ge=1)
    """The x and y of xPyD, and the only scaling truth for a group.

    Not wanting a role means removing it, not setting this to zero — a zero
    would leave `dependencies` pointing at a role that never appears.
    """

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    image_name: Optional[str] = None
    run_command: Optional[str] = None
    backend_parameters: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    gpu_selector: Optional[GPUSelector] = None
    worker_selector: Optional[Dict[str, str]] = None
    gpu_type_selector: Optional[GPUTypeSelector] = None
    """The only entry point for a heterogeneous group, and the precondition
    for gang admission."""
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = None

    dependencies: Optional[List[str]] = None
    """Roles that must be ready before this one starts. Must not cycle."""
    cpu_only: bool = False
    """A router takes no GPU."""


class PDModeEnum(str, Enum):
    """A disaggregation recipe: engine plus KV connector.

    These values must match the entry names in ``pd-modes.yaml`` verbatim —
    the catalog is looked up by them, so a mismatch is a silent miss. The
    loader asserts the two sets are equal at start-up.
    """

    VLLM_NIXL = "vllm-nixl"
    SGLANG_MOONCAKE = "sglang-mooncake"
    SGLANG_NIXL = "sglang-nixl"
    VLLM_ASCEND_MOONCAKE = "vllm-ascend-mooncake"
    CUSTOM = "custom"
    """The user supplies every connection-state parameter themselves. Also the
    only way to mix engines across roles, since a recipe injects one engine's
    connector config into every role."""

    def __str__(self):
        return self.value


# Which engines a recipe can be injected into. A recipe expands into one
# engine's connector config and env, so a role running a different engine
# would be handed configuration it cannot read — e.g. `vllm-nixl` would inject
# `NixlConnector` and `VLLM_NIXL_*` into a TileRT decode and fail silently.
# Mixing engines across roles therefore has to go through `custom`.
#
# `pd-modes.yaml` is the authoritative source for this; the catalog loader
# asserts the two agree at start-up, the same way it asserts the mode names
# match. This table exists so that request validation doesn't have to wait on
# a catalog read.
PD_MODE_BACKENDS: Dict[str, List[str]] = {
    PDModeEnum.VLLM_NIXL.value: [BackendEnum.VLLM.value],
    PDModeEnum.VLLM_ASCEND_MOONCAKE.value: [BackendEnum.VLLM.value],
    PDModeEnum.SGLANG_MOONCAKE.value: [BackendEnum.SGLANG.value],
    PDModeEnum.SGLANG_NIXL.value: [BackendEnum.SGLANG.value],
    # `custom` means the user writes the connection state themselves, so any
    # engine mix is theirs to get right.
    PDModeEnum.CUSTOM.value: [],
}


class DisaggregationSpec(BaseModel):
    mode: PDModeEnum
    readiness: Literal["any_per_role", "all"] = "any_per_role"
    kv_load_failure_policy: Literal["fail", "recompute"] = "fail"
    router_kind: Optional[str] = None
    """None derives it from `mode`."""


class ModelStateEnum(str, Enum):
    """Model-level lifecycle. Deliberately *not* a copy of
    ``ModelInstanceStateEnum`` — this is an aggregate, not a per-process
    lifecycle, so it has no download/start phases.

    Degradation is not a value here. Cache not attached, bandwidth below the
    measured baseline, ratio unmet — all of those coexist with a servable
    group, so they live in the orthogonal ``degradations`` marker instead.
    """

    PENDING = "pending"
    """Nothing ready yet."""
    PARTIAL = "partial"
    """Members are up and the deployment still cannot serve — for a group, a
    role with zero ready members, or an upstream registration that has not
    succeeded.

    **Unreachable for a role-less model**: one ready replica serves, so there
    is no such condition. Being short of the requested count is
    `degradations: [ratio_unmet]` beside a RUNNING state, not this."""
    RUNNING = "running"
    """Servable: every role has at least one ready member *and* the upstream
    registration succeeded."""
    ERROR = "error"
    """A member has failed in a way it can't recover from."""

    def __str__(self):
        return self.value


class RoleStatus(BaseModel):
    """Per-role readiness detail.

    Carried on the Model row rather than computed per request because the
    list endpoint returns `ModelPublic` without instances, and the UI needs
    per-role detail on a row it hasn't expanded.
    """

    desired: int = 0
    ready: int = 0


class DegradationReasonEnum(str, Enum):
    """Reasons a group is servable but worse than asked for.

    Orthogonal to `state`, following the precedent set by `stale`: "config
    changed *and* still serving" has to be expressible as one fact, and so
    does "running but the cache never attached".
    """

    CACHE_NOT_INJECTED = "cache_not_injected"
    BANDWIDTH_DEGRADED = "bandwidth_degraded"
    PD_INEFFECTIVE = "pd_ineffective"
    """The group is serving without transferring any KV — disaggregation has
    silently collapsed into aggregated serving.

    Kept apart from `bandwidth_degraded` because the two are different
    failures with different fixes: slower-than-baseline transfer is a
    transport problem, no transfer at all is a pairing that never formed. A
    single marker would send an operator looking at the network for a
    connector that was never wired up."""

    RATIO_UNMET = "ratio_unmet"
    NO_ATOMIC_ADMISSION = "no_atomic_admission"

    PLACEMENT_DRIFTED = "placement_drifted"
    """Members are deployed somewhere other than where one created now would
    go — almost always an upgrade that introduced per-tenant namespaces, and
    occasionally an Org rename that moved the target while the Pods stayed.

    Deliberately a degradation and not a lifecycle value, and deliberately not
    fixed automatically: the members serve normally and every operation still
    finds them, so moving them would mean restarting healthy containers during
    an upgrade — a worse outcome than the one it fixes. But it cannot be
    silent either. Those Pods hold accelerators that the tenant's queue has no
    record of, so gang admission is optimistic by exactly those cards until
    they cycle. Restarting the model converges it, at a time of the operator's
    choosing."""

    def __str__(self):
        return self.value


class ModelSpecBase(SQLModel, ModelSource):
    name: str = Field(index=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )
    meta: Optional[Dict[str, Any]] = Field(sa_type=JSON, default={})

    replicas: int = Field(default=1, ge=0)
    ready_replicas: int = Field(default=0, ge=0)
    categories: List[str] = Field(sa_type=JSON, default=[])
    placement_strategy: PlacementStrategyEnum = PlacementStrategyEnum.SPREAD
    cpu_offloading: Optional[bool] = None
    distributed_inference_across_workers: Optional[bool] = None
    worker_selector: Optional[Dict[str, str]] = Field(sa_type=JSON, default={})
    gpu_selector: Optional[GPUSelector] = Field(
        sa_type=pydantic_column_type(GPUSelector), default=None
    )
    gpu_type_selector: Optional[GPUTypeSelector] = Field(
        sa_type=pydantic_column_type(GPUTypeSelector), default=None
    )

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    image_name: Optional[str] = None
    run_command: Optional[str] = Field(sa_type=Text, default=None)
    # Whether this deployment's inference server implements the Anthropic
    # Messages API itself, letting the gateway forward an inbound /v1/messages
    # untouched instead of translating it to /v1/chat/completions. False, the
    # pre-existing behavior, still serves /v1/messages -- by translating.
    #
    # A statement about the server, not about the gateway: what the operator
    # knows is whether their image is a recent enough vLLM, not what ai-proxy
    # does with that fact.
    #
    # Declared on the deployment rather than derived from its inference backend
    # because the answer belongs to the running image, and the image is settled
    # per instance (``ModelInstance.gpu_type`` picks it): one deployment can
    # spread over workers of different accelerators whose images need not agree
    # -- vllm-ascend against vllm-openai. A single ai-proxy provider entry
    # covers the whole deployment, so no per-image source can answer for it.
    #
    # Not nullable: with NULL and False meaning the same thing there would be
    # two spellings of "no" and nothing to tell a caller which to send.
    native_anthropic_api: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={"server_default": sa_false()},
    )

    env: Optional[Dict[str, str]] = Field(sa_type=JSON, default=None)
    restart_on_error: Optional[bool] = True
    distributable: Optional[bool] = False

    # Extended KV Cache configuration. Currently maps to LMCache config in vLLM and SGLang.
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = Field(
        sa_type=pydantic_column_type(ExtendedKVCacheConfig), default=None
    )

    speculative_config: Optional[SpeculativeConfig] = Field(
        sa_type=pydantic_column_type(SpeculativeConfig), default=None
    )

    # Scheduled scaling: drives `replicas` on a cron timetable.
    scaling_schedule: Optional[ScalingSchedule] = Field(
        sa_type=pydantic_column_type(ScalingSchedule), default=None
    )

    # Enable generic proxy for model, the control of generic proxy
    # is migrated to ModelAccess. Keeping this field for backward compatibility
    generic_proxy: Optional[bool] = Field(default=False)

    lora_list: Optional[List[LoraListEntry]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(List[LoraListEntry]), nullable=True),
    )

    # Empty `roles` is the backward-compatibility baseline: behaviour is
    # byte-for-byte unchanged. `roles` without `disaggregation` is plain
    # multi-role orchestration; both together is PD.
    roles: Optional[List[RoleSpec]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(List[RoleSpec]), nullable=True),
    )
    disaggregation: Optional[DisaggregationSpec] = Field(
        sa_type=pydantic_column_type(DisaggregationSpec), default=None
    )

    @model_validator(mode="after")
    def set_defaults(self):
        backend = get_backend(self)
        if self.distributed_inference_across_workers is None:
            self.distributed_inference_across_workers = (
                True
                if backend
                in [BackendEnum.VLLM, BackendEnum.ASCEND_MINDIE, BackendEnum.SGLANG]
                else False
            )
        return self


class ModelBase(ModelSpecBase):
    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")
    owner_principal_id: int = Field(
        default_factory=_platform_principal_id,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    # Deprecated field, kept for backward compatibility
    access_policy: AccessPolicyEnum = Field(default=AccessPolicyEnum.AUTHED)


class Model(ModelBase, BaseModelMixin, table=True):
    __tablename__ = 'models'
    __table_args__ = (
        # Model names are unique within their owning Org — two Orgs
        # can each have a "qwen3-0.6b" without colliding.
        UniqueConstraint(
            'owner_principal_id', 'name', name='uix_models_name_per_owner'
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    # Server-owned status. Declared here and on `ModelPublic`, deliberately
    # *not* on `ModelBase`: `ModelUpdate` inherits `ModelBase`, and the UI
    # issues whole-object PUTs (start/stop, inline replica edits), so
    # anything reachable from `ModelBase` gets written back by the client.
    # `ready_replicas` sits on `ModelSpecBase` for historical reasons and the
    # frontend has to strip it by hand — don't grow that list.
    #
    # One writer only: `sync_model_status` computes all five from a single
    # scan of the model's instances, in one transaction behind one change
    # gate. There is no second owner.
    # String, not sa.Enum — following CacheService.state. A bare
    # `Optional[ModelStateEnum]` maps to `sa.Enum(name="modelstateenum")`, and
    # that breaks twice over: asyncpg then renders `$1::modelstateenum` on
    # every read and write, against a column the migration created as VARCHAR;
    # and sa.Enum persists member *names*, so the row would hold "RUNNING"
    # while the API, the enum's own value and the `?state=` filter all say
    # "running".
    state: Optional[ModelStateEnum] = Field(
        default=None, sa_column=Column(String(length=64), nullable=True)
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    role_status: Optional[Dict[str, RoleStatus]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(Dict[str, RoleStatus]), nullable=True),
    )
    stale: Optional[bool] = Field(default=None)
    """A member's `spec_digest` differs from the model's current one, so the
    running group predates the config it's shown with. Orthogonal to `state`:
    a stale group is usually still serving."""
    degradations: Optional[List[str]] = Field(sa_type=JSON, default=None)
    """`DegradationReasonEnum` values. A list, because they coexist."""

    instances: list["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="model",
    )

    cluster: "Cluster" = Relationship(
        back_populates="cluster_models",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    model_route_targets: List["ModelRouteTarget"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={
            "lazy": "noload",
            "overlaps": "models",
            "cascade": "delete",
        },
    )

    model_routes: List["ModelRoute"] = Relationship(
        back_populates="models",
        link_model=ModelRouteTarget,
        sa_relationship_kwargs={
            "lazy": "noload",
            "overlaps": "model,model_route_targets,route_targets,model_route",
        },
    )


class ModelListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "source",
        "cluster_id",
        "replicas",
        "ready_replicas",
        "created_at",
        "updated_at",
    ]


class ModelCreate(ModelBase):
    enable_model_route: Optional[bool] = Field(default=None)


class ModelUpdate(ModelBase):
    pass


class ModelPublic(
    ModelBase,
):
    id: int
    created_at: datetime
    updated_at: datetime
    # Read-only status, mirrored from `Model`. Absent from `ModelBase` so
    # `ModelUpdate` can't accept it — see the note on `Model`.
    state: Optional[ModelStateEnum] = None
    state_message: Optional[str] = None
    role_status: Optional[Dict[str, RoleStatus]] = None
    stale: Optional[bool] = None
    degradations: Optional[List[str]] = None
    # Populated only by the detail endpoint; None on list responses.
    has_stale_lora_instances: Optional[bool] = None

    @field_serializer("lora_list")
    def _strip_lora_prefix(self, lora_list, _info):
        """Hide the internal "<base>:" prefix; clients only see the short name."""
        if not lora_list:
            return lora_list
        prefix = f"{self.name}:"
        out = []
        for entry in lora_list:
            data = entry.model_dump() if isinstance(entry, BaseModel) else dict(entry)
            name = data.get("lora_name") or ""
            if name.startswith(prefix):
                data["lora_name"] = name[len(prefix) :]
            out.append(data)
        return out


ModelsPublic = PaginatedList[ModelPublic]


class RoleEffectiveModel(ModelBase):
    """A `Model` as one role sees it — see `role_effective_model`.

    Non-table on purpose, and both reasons are load-bearing:

    * **A projection must never reach the database.** `Model.model_copy()`
      looks like the obvious way to build one, but the copy *shares the
      original's* `_sa_instance_state` — it is the same ORM identity, so the
      projection would sit one session flush away from writing a role's
      overrides onto the Model row. A non-table class cannot be added to a
      session at all, so the rule holds by construction rather than by
      everyone remembering it.
    * It records the direction of the data: nothing reads a projection back.

    It carries the spec, not the aggregate status: `state` / `role_status` /
    `degradations` live on `Model` and `ModelPublic` only. A worker or a
    scheduling pass acting on a model-wide aggregate would be reading the
    wrong thing anyway.
    """

    id: Optional[int] = None

    # Deliberately left unhashable, which is what `Model` is too — SQLModel
    # sets `__hash__ = None` on a table class the same way pydantic does for
    # any mutable model, and `ModelInstance` has to override it explicitly to
    # go into a queue. So a projection behaves like the thing it stands in
    # for, and a reader that starts hashing models fails for both rather than
    # only for role-bearing deployments.


# The RoleSpec fields that describe the role itself rather than override a
# Model field. Everything else is an override, derived rather than listed so
# that adding one to RoleSpec cannot silently fail to be projected.
_ROLE_OWN_FIELDS = frozenset({"name", "dependencies", "cpu_only"})

_ROLE_OVERRIDE_FIELDS = frozenset(RoleSpec.model_fields) - _ROLE_OWN_FIELDS


def find_role(model, role_name: Optional[str]) -> Optional[RoleSpec]:
    """The named role of `model`, or None if it has no roles or no match."""
    if not role_name:
        return None
    for role in model.roles or []:
        if role.name == role_name:
            return role
    return None


def servable_instances(model, instances):
    """The members that can answer a whole request for `model`.

    For a role-bearing group that is the router alone. Every member serves an
    OpenAI-shaped API on its own port, so handing a request to any of them
    succeeds — a prefill returns after a single token, a decode runs without
    the prefix its KV was meant to carry, and both answer 200 with plausible
    text. Balancing across the group therefore does not fail; it silently
    answers two thirds of requests wrongly.

    One function, because there are two places that route to an instance — the
    gateway's upstream registration and the direct proxy — and a rule this
    consequential must not be able to hold in one and not the other.

    A group with no running router yields nothing rather than falling back to
    its GPU members: there is no member of a group that can serve alone, so an
    empty result is the honest answer and the caller reports the group as
    unavailable.
    """
    if not getattr(model, "roles", None):
        return list(instances)
    return [
        instance
        for instance in instances
        if getattr(instance, "role", None) == RoleNameEnum.ROUTER.value
    ]


def role_takes_no_accelerator(model, role_name: Optional[str]) -> bool:
    """Whether this role should be placed without claiming any GPU.

    `cpu_only` on the role is the explicit answer, but it cannot be the only
    one. A router GPUStack assembles from the mode catalog is a proxy — it
    forwards requests to the members that hold the weights and loads none
    itself — so "takes no accelerator" is a property of what it *is*, not a
    preference someone remembered to tick. Measured consequence of relying on
    the flag alone: the router inherited the group's engine, a vLLM selector
    sized the model's weights for it, and it sat unschedulable on a two-card
    host whose cards its own prefill and decode had just filled. The deploy
    form only registers the flag on the hand-written branch, so every group
    the UI has produced carries `cpu_only: false` on its router.

    A router the user brings themselves is the exception, and it identifies
    itself by carrying an image *and* a command. That one may legitimately
    want a GPU, so its own `cpu_only` governs.
    """
    role = find_role(model, role_name)
    if role is None:
        return False
    if role.cpu_only:
        return True
    if role.name != RoleNameEnum.ROUTER.value:
        return False
    return not (role.image_name and role.run_command)


def role_effective_model(model, role_name: Optional[str]):
    """Return `model` with the named role's overrides applied.

    A `RoleSpec` field left as None means "inherit the Model field of the same
    name". Nothing downstream performs that merge: the worker's start path
    reads `self._model.<field>` in dozens of places and the scheduler's
    filters, selectors and scorers read a Model in dozens more, all of them
    expecting a single set of values. So the merge happens once, here, at the
    two points where a Model is handed to those readers — `get_model()` on the
    worker and `find_candidate()` on the server.

    `replicas` is projected too, and unconditionally: it is never None, and
    for a role-bearing model `Model.replicas` is a 0/1 deployment switch while
    `roles[].replicas` is the count. Inside these two read paths the role's
    count is the right answer — it is what decides how many GPUs one replica
    gets and whether the multi-replica overcommit rule applies. Outside them
    `Model.replicas` keeps its switch meaning, which is why this projection
    deliberately does not reach the evaluator's `set_model_gpus_per_replica`:
    that one writes back.

    Returns `model` itself when there is nothing to project, so a role-less
    deployment takes byte-for-byte the path it takes today.

    One known edge: `distributed_inference_across_workers` is defaulted from
    the *Model's* backend by `ModelSpecBase.set_defaults`, which runs before a
    role's `backend` override is applied. A role that switches engines
    therefore inherits the Model's value rather than one derived from its own
    backend. That only arises under `pd_mode=custom`, the one mode that
    permits a mixed-engine group, and there the user is already supplying the
    connection state by hand — so set it explicitly on the Model in that case.
    """
    role = find_role(model, role_name)
    if role is None:
        return model

    projected = RoleEffectiveModel.model_validate(model)
    for field in _ROLE_OVERRIDE_FIELDS:
        value = getattr(role, field, None)
        if value is None:
            continue
        # Copy, so that mutating a projected list in place — the worker
        # substitutes `{data_dir}` into `backend_parameters` that way — cannot
        # reach back into the role held by `model.roles`.
        setattr(projected, field, copy.deepcopy(value))
    return projected


# Model Instances


class ModelInstanceStateEnum(str, Enum):
    r"""
    Enum for Model Instance State

    Transitions:

       |- - - - - Scheduler - - - - |- - ServeManager - -|- - - - Controller - - - -|- ServeManager -|
       |                            |                    |                          |                |
    PENDING ---> ANALYZING ---> SCHEDULED ---> INITIALIZING ---> DOWNLOADING ---> STARTING ---> RUNNING
                     |            ^  |               |                |               |          ^
                     |            |  |               |                |               |          |(Worker ready)
                     |------------|--|---------------|----------------|---------------|----------|
                     \____________|_____________________________________________________________/|
                                  |                  ERROR                                       |(Worker unreachable)
                                  └--------------------┘                                         v
                                    (Restart on Error)                                       UNREACHABLE
    """

    INITIALIZING = "initializing"
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    ERROR = "error"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class ComputedResourceClaim(BaseModel):
    is_unified_memory: Optional[bool] = False
    offload_layers: Optional[int] = None
    total_layers: Optional[int] = None
    ram: Optional[int] = Field(default=None)  # in bytes
    vram: Optional[Dict[int, int]] = Field(default=None)  # in bytes
    tensor_split: Optional[List[int]] = Field(default=None)
    vram_utilization: Optional[float] = Field(default=None)


class ModelInstanceSubordinateWorker(BaseModel):
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    total_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    # - For model file preparation
    download_progress: Optional[float] = None
    # - For model instance serving preparation
    pid: Optional[int] = None
    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    arguments: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class DistributedServerCoordinateModeEnum(Enum):
    # DELEGATED means that the subordinate workers' coordinate is by-pass to other framework.
    DELEGATED = "delegated"
    # INITIALIZE_LATER means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance SHOULD start after the main worker initializes.
    # For example, Ascend MindIE/vLLM/SGLang instances need to start their subordinate workers after the main worker initializes.
    INITIALIZE_LATER = "initialize_later"
    # RUN_FIRST means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance MUST get ready before the main worker starts.
    RUN_FIRST = "run_first"


class DistributedServers(BaseModel):
    # Indicates how the distributed servers coordinate with the main worker.
    mode: DistributedServerCoordinateModeEnum = (
        DistributedServerCoordinateModeEnum.DELEGATED
    )
    # Indicates if subordinate workers should download model files.
    download_model_files: Optional[bool] = True
    subordinate_workers: Optional[List[ModelInstanceSubordinateWorker]] = Field(
        sa_column=Column(JSON), default=[]
    )
    model_config = ConfigDict(from_attributes=True)


@dataclass
class ModelInstanceDeploymentMetadata:
    """
    Metadata for model instance deployment.
    """

    name: str
    """
    Name for model instance deployment.
    """
    distributed: bool = False
    """
    Whether the model instance is deployed in distributed mode.
    """
    distributed_leader: bool = False
    """
    Whether the model instance is the leader in distributed mode.
    """
    distributed_follower: bool = False
    """
    Whether the model instance is a follower in distributed mode.
    """
    distributed_follower_index: Optional[int] = None
    """
    Index of the follower in distributed mode.
    It is None for leader or non-distributed mode.
    """
    namespace: Optional[str] = None
    """
    Namespace the workload lives in. None means "wherever the runtime puts a
    workload that declares none" — the deployer's configured default — which
    is what every row created before the per-tenant namespaces carries.
    """


class ModelInstanceBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_advertise_address: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    pid: Optional[int] = None
    # FIXME: Migrate to ports.
    port: Optional[int] = None
    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    download_progress: Optional[float] = None
    resolved_path: Optional[str] = None
    draft_model_source: Optional[ModelSource] = Field(
        sa_column=Column(pydantic_column_type(ModelSource)), default=None
    )
    draft_model_download_progress: Optional[float] = None
    draft_model_resolved_path: Optional[str] = None
    restart_count: Optional[int] = 0
    last_restart_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    cache_config: Optional[CacheConfigSnapshot] = Field(
        sa_column=Column(pydantic_column_type(CacheConfigSnapshot)), default=None
    )
    """Resolved shared-cache connection info; None for local/disabled KV cache."""
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])

    model_id: int = Field(default=None, foreign_key="models.id")
    model_name: str

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    api_detected_backend_version: Optional[str] = None
    injected_backend_parameters: Optional[List[str]] = Field(
        sa_column=Column(JSON), default=None
    )

    distributed_servers: Optional[DistributedServers] = Field(
        sa_column=Column(pydantic_column_type(DistributedServers)), default=None
    )
    # The "model_id" field conflicts with the protected namespace "model_" in Pydantic.
    # Disable it given that it's not a real issue for this particular field.
    model_config = ConfigDict(protected_namespaces=())

    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")
    owner_principal_id: int = Field(
        default_factory=_platform_principal_id,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    namespace: Optional[str] = None
    """The Kubernetes namespace this instance's workload lives in, resolved
    from `owner_principal_id` when the row is created and never recomputed:
    the namespace is where the Pod *is*, not where a Pod for this tenant
    *would* go today, so reading, deleting and log-streaming keep finding it
    even after the tenant's namespace convention changes.

    None on rows created before per-tenant namespaces, which is what keeps
    their Pods reachable: the runtime falls back to the deployer's configured
    default namespace, the only one those Pods were ever created in.
    """

    mounted_loras: Optional[List[LoraListEntry]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(List[LoraListEntry]), nullable=True),
    )

    @property
    def spans_workers(self) -> bool:
        """Whether this instance is actually placed across several
        workers (subordinate workers assigned at scheduling) — the
        placement fact, as opposed to the model's
        distributed_inference_across_workers permission flag."""
        dservers = self.distributed_servers
        return bool(dservers and dservers.subordinate_workers)
    role: Optional[str] = None
    """Which role of the parent Model this instance serves. None for a plain
    single-role deployment."""
    group_id: Optional[str] = Field(default=None, index=True)
    """Shared by every member of one group. A group is a *generation*, not a
    replica: one group_id is one `spec_digest`.

    Pairing binds to this rather than to peer addresses because serving ports
    were measured to change on every rebuild; addresses get resolved when the
    router config is rendered."""
    spec_digest: Optional[str] = None
    """The generation this instance was created from. Differing from the
    model's current digest is what makes the model `stale`."""
    named_ports: Optional[Dict[str, PortBand]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(Dict[str, PortBand]), nullable=True),
    )
    """Connector ports by declared name. Values are bands, not points."""

    def get_deployment_metadata(
        self,
        worker_id: int,
    ) -> Optional[ModelInstanceDeploymentMetadata]:
        """
        Get the deployment metadata for the model instance.

        Args:
            worker_id:
                The ID of the worker to get the deployment metadata for.

        Returns:
            The deployment metadata,
            or None if the model instance is not handling by the given `worker_id` worker.
        """

        dservers = self.distributed_servers
        subworkers = (
            dservers.subordinate_workers
            if dservers and dservers.subordinate_workers
            else []
        )

        name = self.name
        distributed = bool(subworkers)
        distributed_leader = distributed and self.worker_id == worker_id
        distributed_follower = distributed and not distributed_leader
        distributed_follower_index = None
        if distributed_follower:
            for idx, subworker in enumerate(subworkers):
                if subworker.worker_id == worker_id:
                    distributed_follower_index = idx
                    break
            if distributed_follower_index is not None:
                # Mutate the name to include the follower index,
                # so that each follower has a unique name.
                name += f"-f{distributed_follower_index}"

        if self.worker_id != worker_id and distributed_follower_index is None:
            # This model instance is not handling by the given worker.
            return None

        return ModelInstanceDeploymentMetadata(
            name=name,
            namespace=self.namespace,
            distributed=distributed,
            distributed_leader=distributed_leader,
            distributed_follower=distributed_follower,
            distributed_follower_index=distributed_follower_index,
        )


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    __tablename__ = 'model_instances'
    id: Optional[int] = Field(default=None, primary_key=True)

    model: Optional[Model] = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    model_files: List["ModelFile"] = Relationship(
        back_populates="instances",
        link_model=ModelInstanceModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    draft_model_files: List["ModelFile"] = Relationship(
        back_populates="draft_instances",
        link_model=ModelInstanceDraftModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    cluster: "Cluster" = Relationship(
        back_populates="cluster_model_instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    @classmethod
    async def one_by_id_with_model_files(
        cls,
        session,
        instance_id: int,
        populate_existing: bool = True,
    ) -> Optional["ModelInstance"]:
        """Load a model instance with primary/LoRA + draft model_files and model spec eagerly loaded."""
        stmt = (
            select(cls)
            .where(cls.id == instance_id)
            .options(
                selectinload(cls.model_files),
                selectinload(cls.draft_model_files),
                selectinload(cls.model),
            )
        )
        if populate_existing:
            stmt = stmt.execution_options(populate_existing=True)
        return (await session.exec(stmt)).first()

    # overwrite the hash to use in uniquequeue
    def __hash__(self):
        return self.id


class ModelInstanceCreate(ModelInstanceBase):
    pass


class ModelInstanceUpdate(ModelInstanceBase):
    pass


class ModelInstancePublic(
    ModelInstanceBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelInstancesPublic = PaginatedList[ModelInstancePublic]


class ModelInstanceLogWorker(BaseModel):
    id: int
    name: str


class ModelInstanceLogRestartEntry(BaseModel):
    """One main serve log session on disk, with optional UX label time."""

    previous: bool = False
    started_at: Optional[datetime] = Field(
        default=None,
        description=(
            "Approximate start time from the main log file metadata "
            "(birthtime if available, else mtime), UTC."
        ),
    )
    containers: List[str] = Field(
        default_factory=list,
        description=(
            "Available container names for this restart. "
            "'default' is the main workload container; others are sidecars "
            "(e.g., ['default', 'ray-head'])."
        ),
    )


class ModelInstanceLogWorkerOption(BaseModel):
    """Per-worker result for GET /model-instances/{id}/log-options (one node on disk)."""

    worker_id: int
    name: str = ""
    restarts: List[ModelInstanceLogRestartEntry] = Field(default_factory=list)
    error: Optional[str] = Field(
        default=None,
        description="If set, log options could not be fetched from this worker.",
    )


class ServeLogOptionsResponse(BaseModel):
    """Worker GET /serveLogOptions JSON; also validates that payload when the server proxies."""

    restarts: List[ModelInstanceLogRestartEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _legacy_restart_counts(cls, data: Any) -> Any:
        """Old workers only sent restart_counts; expand to restarts when `restarts` is absent."""
        if not isinstance(data, dict):
            return data
        if "restarts" in data:
            return data
        raw = data.get("restart_counts")
        if not isinstance(raw, list):
            return {**data, "restarts": []}
        counts: List[int] = []
        for x in raw:
            try:
                counts.append(int(x))
            except (TypeError, ValueError):
                continue
        counts.sort(reverse=True)
        # Map the highest restart_count to previous=False (current),
        # the second highest to previous=True.
        entries = []
        for i, c in enumerate(counts):
            entries.append({"previous": i > 0, "started_at": None})
        return {**data, "restarts": entries}


class ModelInstanceLogOptions(BaseModel):
    """Server GET /model-instances/{id}/log-options: per-worker serve log distribution."""

    main_worker_id: Optional[int] = Field(
        default=None,
        description="same as model instance worker_id.",
    )
    workers: List[ModelInstanceLogWorkerOption] = Field(
        default_factory=list,
        description=(
            "Ordered list: main worker first, then subordinate workers. "
            "Each entry reflects that worker's local serve logs."
        ),
    )


def is_gguf_model(model: Union[Model, ModelSource]):
    """
    Check if the model is a GGUF model.
    Args:
        model: Model to check.
    """
    return (
        (
            model.source == SourceEnum.HUGGING_FACE
            and model.huggingface_filename
            and model.huggingface_filename.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.MODEL_SCOPE
            and model.model_scope_file_path
            and model.model_scope_file_path.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.LOCAL_PATH
            and model.local_path
            and model.local_path.endswith(".gguf")
        )
    )


def is_audio_model(model: Model):
    """
    Check if the model is a STT or TTS model.
    Args:
        model: Model to check.
    """
    if model.backend == BackendEnum.VOX_BOX:
        return True

    if model.categories:
        return (
            'speech_to_text' in model.categories or 'text_to_speech' in model.categories
        )

    return False


def is_llm_model(model: Model):
    """
    Check if the model is an LLM model.
    Args:
        model: Model to check.
    """
    return not model.categories or CategoryEnum.LLM in model.categories


def is_omni_model(model: Model) -> bool:
    """
    Check if the model is an omni model (Image or Audio category).
    Args:
        model: Model to check.
    """

    if model.backend == BackendEnum.VLLM and find_bool_parameter(
        model.backend_parameters, ["omni"]
    ):
        return True

    OMNI_CATEGORIES = (
        CategoryEnum.IMAGE,
        CategoryEnum.TEXT_TO_SPEECH,
    )
    return any(cat in model.categories for cat in OMNI_CATEGORIES)


def is_image_model(model: Model):
    """
    Check if the model is an image model.
    Args:
        model: Model to check.
    """
    return "image" in model.categories


def is_embedding_model(model: Model):
    """
    Check if the model is an embedding model.
    Args:
        model: Model to check.
    """
    return "embedding" in model.categories


def is_reranker_model(model: Model):
    """
    Check if the model is a reranker model.
    Args:
        model: Model to check.
    """
    return "reranker" in model.categories


def get_backend(model: Model) -> str:
    if model.backend:
        return model.backend

    if is_gguf_model(model):
        return BackendEnum.CUSTOM

    return BackendEnum.VLLM


def get_mmproj_filename(model: Union[Model, ModelSource]) -> Optional[str]:
    """
    Get the mmproj filename for the model. If the mmproj is not provided in the model's
    backend parameters, it will try to find the default mmproj file.
    """
    if not is_gguf_model(model):
        return None

    if hasattr(model, "backend_parameters"):
        mmproj = find_parameter(model.backend_parameters, ["mmproj"])
        if mmproj and Path(mmproj).name == mmproj:
            return mmproj

    return "*mmproj*.gguf"
