"""The bandwidth-requirement endpoint.

The arithmetic is tested in `tests/scheduler/test_kv_transfer_budget.py`. What
is tested here is the part it has no opinion about: that the answer can be had
for a model that does not exist yet, that a deployment's already-configured KV
dtype is picked up, and that an unsizeable config is refused rather than
answered.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes import kv_transfer_budget as route
from gpustack.scheduler.kv_transfer_budget import GB, BandwidthVerdict

LLAMA_70B = SimpleNamespace(
    torch_dtype="bfloat16",
    num_hidden_layers=80,
    num_key_value_heads=8,
    head_dim=128,
    kv_lora_rank=None,
    qk_rope_head_dim=None,
)

UNSIZEABLE = SimpleNamespace(
    torch_dtype="bfloat16",
    num_hidden_layers=80,
    num_key_value_heads=None,
    head_dim=None,
    kv_lora_rank=None,
    qk_rope_head_dim=None,
)


async def _estimate(params=LLAMA_70B, model=None, **body):
    """Call the handler with the config read and the ORM stubbed."""
    body.setdefault(
        "model_source", {"source": "huggingface", "huggingface_repo_id": "m/n"}
    )
    request = route.KVTransferBudgetRequest(**body)

    class _FakeModelParameters:
        def __init__(self):
            for name in (
                "torch_dtype",
                "num_hidden_layers",
                "num_key_value_heads",
                "head_dim",
                "kv_lora_rank",
                "qk_rope_head_dim",
            ):
                setattr(self, name, getattr(params, name))

        def from_model_pretrained_config(self, *_):
            return None

    with (
        patch.object(route.Worker, "all", AsyncMock(return_value=[])),
        patch.object(route.Model, "one_by_id", AsyncMock(return_value=model)),
        patch.object(route, "assert_resource_visible", lambda *a, **k: None),
        patch(
            "gpustack.scheduler.calculator.get_pretrained_config_with_workers",
            AsyncMock(return_value=object()),
        ),
        patch(
            "gpustack.policies.candidate_selectors.base_candidate_selector.ModelParameters",
            _FakeModelParameters,
        ),
    ):
        return await route.estimate_kv_transfer_budget(
            session=None, ctx=None, request=request
        )


@pytest.mark.asyncio
async def test_it_answers_for_a_model_that_does_not_exist_yet():
    """The question is asked while a model is being configured, which is
    exactly when it has no id."""
    result = await _estimate(seq_len=4096, ttft_budget_ms=500, prefill_ms=200)

    assert result.bytes_per_request == 1_342_177_280
    assert result.transfer_budget_ms == pytest.approx(300)
    assert result.required_bandwidth_bytes_per_second / GB == pytest.approx(
        4.47, abs=0.01
    )


@pytest.mark.asyncio
async def test_the_reference_links_are_scored_against_this_model():
    """A bare "you need 4.5 GB/s" is not actionable to someone who does not
    know what their NIC delivers."""
    result = await _estimate(seq_len=4096, ttft_budget_ms=500, prefill_ms=200)
    by_name = {link.name: link for link in result.reference_links}

    assert by_name["25GbE"].sufficient is False
    assert by_name["100GbE"].sufficient is True
    # 1.34 GB over 0.12 GB/s. The design's copy quotes ~11 s for exactly this.
    assert by_name["1GbE"].transfer_ms == pytest.approx(11_185, rel=0.01)


@pytest.mark.asyncio
async def test_a_deployments_own_fp8_setting_halves_the_requirement():
    """A user who has already taken the cheapest way out must not be told
    their link is half as adequate as it is."""
    fp16 = await _estimate(ttft_budget_ms=500, prefill_ms=200)
    fp8 = await _estimate(
        ttft_budget_ms=500,
        prefill_ms=200,
        backend_parameters=["--kv-cache-dtype", "fp8"],
    )

    assert fp8.required_bandwidth_bytes_per_second * 2 == pytest.approx(
        fp16.required_bandwidth_bytes_per_second
    )
    assert fp8.kv_cache_dtype == "fp8"


@pytest.mark.asyncio
async def test_a_saved_models_engine_arguments_are_read_when_none_are_sent():
    """Asking about an existing deployment must not silently ignore how that
    deployment is already configured."""
    saved = SimpleNamespace(backend_parameters=["--kv-cache-dtype=fp8"])
    result = await _estimate(
        model=saved, model_id=1, model_source=None, ttft_budget_ms=500, prefill_ms=200
    )

    assert result.kv_cache_dtype == "fp8"


@pytest.mark.asyncio
async def test_a_measured_bandwidth_becomes_a_comparison_not_a_second_answer():
    """The slot baseline probing fills. 0.293 GB/s against a 4.47 GB/s
    requirement is 6.5% -- measured on a real 2.5GbE pair."""
    result = await _estimate(
        ttft_budget_ms=500,
        prefill_ms=200,
        measured_bandwidth_bytes_per_second=0.293 * GB,
    )

    assert result.measured.ratio == pytest.approx(0.065, abs=0.005)
    assert result.measured.verdict is BandwidthVerdict.INSUFFICIENT
    assert result.measured.transfer_ms == pytest.approx(4_581, rel=0.01)
    assert result.measured.remedies


@pytest.mark.asyncio
async def test_a_sufficient_link_is_not_handed_remedies():
    """Attached to a link that is fine, they read as a warning about it."""
    result = await _estimate(
        ttft_budget_ms=500,
        prefill_ms=200,
        measured_bandwidth_bytes_per_second=50 * GB,
    )

    assert result.measured.verdict is BandwidthVerdict.SUFFICIENT
    assert result.measured.remedies == []


@pytest.mark.asyncio
async def test_without_a_measurement_there_is_no_comparison():
    result = await _estimate(ttft_budget_ms=500, prefill_ms=200)
    assert result.measured is None


@pytest.mark.asyncio
async def test_an_unsizeable_config_is_refused_rather_than_answered():
    with pytest.raises(BadRequestException) as excinfo:
        await _estimate(params=UNSIZEABLE)
    assert "KV cache" in excinfo.value.message


@pytest.mark.asyncio
async def test_prefill_that_eats_the_whole_budget_is_refused():
    """Dividing by a negative would report a negative required bandwidth,
    which reads as "no requirement"."""
    with pytest.raises(BadRequestException):
        await _estimate(ttft_budget_ms=200, prefill_ms=500)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"model_id": None, "model_source": None},
        {"model_id": 1},
    ],
)
async def test_exactly_one_of_id_or_source_is_required(body):
    """Accepting both and preferring one silently lets a page send a stale id
    alongside a fresh source and get an answer about neither."""
    with pytest.raises(BadRequestException):
        await _estimate(**body)


@pytest.mark.asyncio
async def test_a_stated_transfer_window_wins_over_the_derived_one():
    """The panel states the window directly, and it has to be the window used.

    Derived, the same 200 ms would take a TTFT target and a share to express,
    and a reader can only argue with premises they can see.
    """
    stated = await _estimate(transfer_budget_ms=200)
    assert stated.transfer_budget_ms == pytest.approx(200.0)
    assert stated.required_bandwidth_bytes_per_second == pytest.approx(
        stated.bytes_per_request / 0.2
    )

    # Supplied alongside the derived inputs it still wins, rather than the two
    # silently averaging into a third number that is neither.
    both = await _estimate(transfer_budget_ms=200, ttft_budget_ms=500, prefill_ms=100)
    assert both.transfer_budget_ms == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_the_derived_window_still_works_without_a_stated_one():
    """The TTFT-minus-prefill path is the right arithmetic for a caller that
    has both, so it survives the addition."""
    derived = await _estimate(ttft_budget_ms=500, prefill_ms=200)
    assert derived.transfer_budget_ms == pytest.approx(300.0)
