"""The bandwidth a PD group would need, answerable before it is deployed.

A user choosing between aggregated and disaggregated deployment has one
question that decides it — "is my network fast enough for this model" — and no
way to ask it today. The measured half needs a cluster, a deployment and a
probe; by the time it can answer, the choice has already been made.

The requirement half needs none of that. `KV bytes / (TTFT budget - prefill)`
reads only the model's own config, so this endpoint answers on the model
selection screen, before anything is created. When baseline probing lands, its
number arrives as `measured` here and the response grows a comparison rather
than the page growing a second feature.

**Why a POST that takes a model source rather than a GET on a model id.** The
question is asked while a model is being configured, which is precisely when it
has no id. An id is still accepted, for the "should I turn PD on for this
existing deployment" case.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.api.tenant import assert_resource_visible
from gpustack.schemas.models import Model, ModelSource
from gpustack.schemas.workers import Worker
from gpustack.scheduler.kv_transfer_budget import (
    REFERENCE_LINKS,
    REMEDIES,
    BandwidthVerdict,
    KVFootprint,
    from_model_parameters,
    required_bandwidth,
    transfer_budget_seconds,
    transfer_seconds,
    verdict,
)
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()

logger = logging.getLogger(__name__)


class KVTransferBudgetRequest(BaseModel):
    model_id: Optional[int] = None
    model_source: Optional[ModelSource] = None
    """Either identifies a saved model or describes an unsaved one. Exactly one
    is required; accepting both and silently preferring one would let a page
    send a stale id alongside a fresh source and get an answer about neither."""

    backend_parameters: Optional[List[str]] = None
    """The engine arguments as currently configured. Read for
    `--kv-cache-dtype`, which halves the footprint and is the cheapest way out
    of an insufficient verdict — a user who has already taken it must not be
    told their link is half as adequate as it is."""

    seq_len: int = Field(default=4096, gt=0)
    ttft_budget_ms: float = Field(default=500.0, gt=0)
    prefill_ms: Optional[float] = Field(default=None, gt=0)
    """How long prefill itself takes. The one term that cannot be derived from
    the model config — it depends on the accelerator — so when it is absent the
    transfer gets a fixed share of the budget instead."""

    transfer_budget_ms: Optional[float] = Field(default=None, gt=0)
    """The transfer's window, stated directly instead of derived.

    Wins over `ttft_budget_ms` and `prefill_ms`, and exists because deriving
    the window costs a caller two assumptions to state one number. "Move this
    KV within 200 ms" is a single premise a reader can accept or reject; "a
    500 ms TTFT target, of which the transfer gets a third" is the same 167 ms
    arrived at through a share the caller did not choose and cannot see.

    The derived path stays for the caller that genuinely has a TTFT budget and
    a prefill measurement, where subtracting is the right arithmetic."""

    measured_bandwidth_bytes_per_second: Optional[float] = Field(default=None, gt=0)
    """The slot baseline probing will fill. Present now so the response shape
    does not change when it lands."""

    trust_remote_code: bool = False


class ReferenceLinkPublic(BaseModel):
    name: str
    bandwidth_bytes_per_second: float
    transfer_ms: float
    sufficient: bool


class MeasuredComparisonPublic(BaseModel):
    bandwidth_bytes_per_second: float
    ratio: float
    transfer_ms: float
    verdict: BandwidthVerdict
    remedies: List[str] = []


class KVTransferBudgetPublic(BaseModel):
    seq_len: int
    kv_cache_dtype: str
    bytes_per_token: int
    bytes_per_request: int

    layers: int
    kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    """Set instead of kv_heads/head_dim for MLA models, where one compressed
    latent is stored rather than per-head K and V. It is what makes those
    models an order of magnitude cheaper to disaggregate, so it is reported
    rather than folded into the byte count."""

    transfer_budget_ms: float
    required_bandwidth_bytes_per_second: float
    reference_links: List[ReferenceLinkPublic]
    measured: Optional[MeasuredComparisonPublic] = None


async def _resolve_model(
    session: AsyncSession, ctx, request: KVTransferBudgetRequest
) -> Model:
    if (request.model_id is None) == (request.model_source is None):
        raise BadRequestException(
            message="Provide exactly one of model_id or model_source."
        )

    if request.model_id is not None:
        model = await Model.one_by_id(session, request.model_id)
        if not model:
            raise NotFoundException(message="Model not found")
        assert_resource_visible(ctx, model, not_found_message="Model not found")
        return model

    # A transient Model, never added to the session. The config readers below
    # take a Model, and the alternative -- a second code path that takes loose
    # source fields -- is a second place for the source resolution rules to
    # drift from the ones deployment actually uses.
    return Model(
        name="kv-transfer-budget-probe",
        **request.model_source.model_dump(),
    )


@router.post("/kv-transfer-budget", response_model=KVTransferBudgetPublic)
async def estimate_kv_transfer_budget(
    session: SessionDep,
    ctx: TenantContextDep,
    request: KVTransferBudgetRequest,
):
    from gpustack.policies.candidate_selectors.base_candidate_selector import (
        ModelParameters,
    )
    from gpustack.scheduler.calculator import get_pretrained_config_with_workers

    model = await _resolve_model(session, ctx, request)

    workers = await Worker.all(session)
    try:
        pretrained_config = await get_pretrained_config_with_workers(
            model, workers, trust_remote_code=request.trust_remote_code
        )
    except Exception as e:
        # The model's own config is the only input; without it there is no
        # conservative answer to fall back on, only a confident wrong one.
        raise BadRequestException(
            message=f"Failed to read the model configuration: {e}"
        )

    params = ModelParameters()
    try:
        params.from_model_pretrained_config(model, pretrained_config)
    except Exception as e:
        raise BadRequestException(
            message=f"Failed to parse the model configuration: {e}"
        )

    footprint = from_model_parameters(
        params,
        seq_len=request.seq_len,
        backend_parameters=request.backend_parameters
        or (model.backend_parameters if request.model_id is not None else None),
    )
    if footprint is None:
        raise BadRequestException(
            message=(
                "The model configuration does not report enough to size its KV "
                "cache (layers, kv heads and head dim, or an MLA latent rank). "
                "Without those the bandwidth requirement cannot be computed."
            )
        )

    budget_seconds = (
        request.transfer_budget_ms / 1000.0
        if request.transfer_budget_ms is not None
        else transfer_budget_seconds(
            ttft_budget_seconds=request.ttft_budget_ms / 1000.0,
            prefill_seconds=(
                request.prefill_ms / 1000.0 if request.prefill_ms is not None else None
            ),
        )
    )
    if budget_seconds is None:
        raise BadRequestException(
            message=(
                "Prefill alone already consumes the whole TTFT budget, so there "
                "is no time left for a KV transfer. Raise the budget or lower "
                "the prefill estimate."
            )
        )

    required = required_bandwidth(footprint.bytes_per_request, budget_seconds)

    return KVTransferBudgetPublic(
        seq_len=footprint.seq_len,
        kv_cache_dtype=footprint.kv_cache_dtype.value,
        bytes_per_token=footprint.bytes_per_token,
        bytes_per_request=footprint.bytes_per_request,
        layers=footprint.layers,
        kv_heads=footprint.kv_heads,
        head_dim=footprint.head_dim,
        latent_dim=footprint.latent_dim,
        transfer_budget_ms=budget_seconds * 1000.0,
        required_bandwidth_bytes_per_second=required,
        reference_links=_reference_links(footprint, required),
        measured=_measured(
            footprint, required, request.measured_bandwidth_bytes_per_second
        ),
    )


def _reference_links(
    footprint: KVFootprint, required: float
) -> List[ReferenceLinkPublic]:
    """Common links scored against this model's requirement.

    A bare "you need 4.5 GB/s" is not actionable to someone who does not know
    what their NIC delivers. Naming the rungs turns it into a purchase or a
    placement decision.
    """
    out = []
    for link in REFERENCE_LINKS:
        seconds = transfer_seconds(
            footprint.bytes_per_request, link.bandwidth_bytes_per_second
        )
        out.append(
            ReferenceLinkPublic(
                name=link.name,
                bandwidth_bytes_per_second=link.bandwidth_bytes_per_second,
                transfer_ms=seconds * 1000.0,
                sufficient=link.bandwidth_bytes_per_second >= required,
            )
        )
    return out


def _measured(
    footprint: KVFootprint, required: float, measured: Optional[float]
) -> Optional[MeasuredComparisonPublic]:
    if not measured:
        return None
    ratio = measured / required
    result = verdict(ratio)
    return MeasuredComparisonPublic(
        bandwidth_bytes_per_second=measured,
        ratio=ratio,
        transfer_ms=transfer_seconds(footprint.bytes_per_request, measured) * 1000.0,
        verdict=result,
        # Only where they are the point. Attached to a sufficient link they
        # read as a warning about something that is fine.
        remedies=list(REMEDIES) if result is BandwidthVerdict.INSUFFICIENT else [],
    )
