"""The bandwidth threshold a PD group needs, from the model config alone.

The worked examples below are the design's own, and they are here as
calibration rather than as illustration: this module's whole claim is that it
can answer before any measurement exists, so the only thing that can catch it
being wrong is agreement with numbers derived independently.
"""

import pytest

from gpustack.scheduler.kv_transfer_budget import (
    GB,
    BandwidthVerdict,
    KVCacheDType,
    from_model_parameters,
    kv_footprint,
    parse_kv_cache_dtype,
    required_bandwidth,
    resolve_kv_cache_dtype,
    transfer_budget_seconds,
    transfer_seconds,
    verdict,
)


class _Params:
    """The subset of `ModelParameters` this module reads."""

    def __init__(self, **kwargs):
        self.torch_dtype = kwargs.pop("torch_dtype", "bfloat16")
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 0)
        self.num_key_value_heads = kwargs.pop("num_key_value_heads", None)
        self.head_dim = kwargs.pop("head_dim", None)
        self.kv_lora_rank = kwargs.pop("kv_lora_rank", None)
        self.qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)
        assert not kwargs


# --- the footprint ---------------------------------------------------------- #


def test_llama_3_70b_4k_matches_the_published_figure():
    """80 layers, GQA with 8 kv heads, head_dim 128, fp16 -> 1.34 GB at 4K."""
    footprint = kv_footprint(
        layers=80,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_heads=8,
        head_dim=128,
    )

    assert footprint.bytes_per_token == 327_680
    assert footprint.bytes_per_request == 1_342_177_280
    assert footprint.bytes_per_request / GB == pytest.approx(1.34, abs=0.01)


def test_mla_stores_one_latent_not_a_separate_k_and_v():
    """DeepSeek-V3: 61 layers, kv_lora_rank 512 + qk_rope_head_dim 64.

    The per-head formula's leading `2 *` does not apply here. Applying it would
    report ~0.5 GB at 4K instead of ~0.25 GB, and 0.25 GB is the entire reason
    an MLA model can be disaggregated over a link that a GQA model of the same
    size cannot use at all.
    """
    footprint = kv_footprint(
        layers=61,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )

    assert footprint.latent_dim == 576
    assert footprint.bytes_per_token == 70_272
    assert footprint.bytes_per_request / GB == pytest.approx(0.29, abs=0.01)
    assert footprint.kv_heads is None


def test_mla_is_an_order_of_magnitude_below_gqa_at_the_same_scale():
    """The comparison the deployment decision actually turns on."""
    gqa = kv_footprint(
        layers=80,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_heads=8,
        head_dim=128,
    )
    mla = kv_footprint(
        layers=61,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )

    assert gqa.bytes_per_request / mla.bytes_per_request > 4


def test_fp8_kv_cache_halves_the_footprint():
    """The cheapest way out of an insufficient verdict, so it has to show."""
    fp16 = kv_footprint(
        layers=80,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_heads=8,
        head_dim=128,
    )
    fp8 = kv_footprint(
        layers=80,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP8,
        kv_heads=8,
        head_dim=128,
    )

    assert fp8.bytes_per_request * 2 == fp16.bytes_per_request


def test_the_footprint_scales_with_sequence_length():
    short = kv_footprint(
        layers=80,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_heads=8,
        head_dim=128,
    )
    long = kv_footprint(
        layers=80,
        seq_len=32768,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_heads=8,
        head_dim=128,
    )

    assert long.bytes_per_request == short.bytes_per_request * 8
    assert long.bytes_per_request / GB == pytest.approx(10.7, abs=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        # No kv heads: an MHA/GQA model whose config did not report them.
        {"layers": 80, "kv_heads": None, "head_dim": 128},
        {"layers": 80, "kv_heads": 8, "head_dim": None},
        {"layers": 0, "kv_heads": 8, "head_dim": 128},
    ],
)
def test_an_unsizeable_config_yields_nothing_rather_than_a_guess(kwargs):
    """A wrong footprint produces a confident threshold, and the threshold is
    the one output here a user acts on."""
    assert (
        kv_footprint(seq_len=4096, kv_cache_dtype=KVCacheDType.FP16, **kwargs) is None
    )


def test_a_zero_length_sequence_has_no_footprint():
    assert (
        kv_footprint(
            layers=80,
            seq_len=0,
            kv_cache_dtype=KVCacheDType.FP16,
            kv_heads=8,
            head_dim=128,
        )
        is None
    )


# --- the dtype ---------------------------------------------------------------#


@pytest.mark.parametrize(
    "params, expected",
    [
        (["--kv-cache-dtype", "fp8"], KVCacheDType.FP8),
        (["--kv-cache-dtype=fp8_e4m3"], KVCacheDType.FP8),
        (["--kv_cache_dtype", "fp16"], KVCacheDType.FP16),
        (["--max-model-len=8192"], None),
        ([], None),
        (None, None),
    ],
)
def test_the_configured_kv_dtype_is_read_from_the_engine_arguments(params, expected):
    assert parse_kv_cache_dtype(params) == expected


def test_auto_defers_to_the_model_dtype_rather_than_overriding_it():
    """`auto` means "follow the model", so it must not resolve to a size of its
    own -- that would override the value it exists to defer to."""
    assert parse_kv_cache_dtype(["--kv-cache-dtype", "auto"]) is None
    assert (
        resolve_kv_cache_dtype("bfloat16", ["--kv-cache-dtype", "auto"])
        is KVCacheDType.FP16
    )


def test_an_explicit_kv_dtype_wins_over_the_model_dtype():
    """A user who has already halved their transfer must not be told their link
    is half as adequate as it is."""
    assert (
        resolve_kv_cache_dtype("bfloat16", ["--kv-cache-dtype", "fp8"])
        is KVCacheDType.FP8
    )


def test_from_model_parameters_reads_the_scheduler_s_own_parse():
    footprint = from_model_parameters(
        _Params(
            torch_dtype="bfloat16",
            num_hidden_layers=80,
            num_key_value_heads=8,
            head_dim=128,
        ),
        seq_len=4096,
    )

    assert footprint.bytes_per_request == 1_342_177_280


def test_from_model_parameters_prefers_an_mla_latent_over_head_counts():
    """MLA configs still report `num_key_value_heads`, and taking that path
    would silently double the number."""
    footprint = from_model_parameters(
        _Params(
            num_hidden_layers=61,
            num_key_value_heads=128,
            head_dim=192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        ),
        seq_len=4096,
    )

    assert footprint.latent_dim == 576
    assert footprint.kv_heads is None


# --- the reversed formula ---------------------------------------------------- #


def test_the_worked_example_from_the_design():
    """1.34 GB, 500 ms TTFT budget, 200 ms prefill -> ~4.5 GB/s."""
    budget = transfer_budget_seconds(ttft_budget_seconds=0.5, prefill_seconds=0.2)
    assert budget == pytest.approx(0.3)

    needed = required_bandwidth(1_342_177_280, budget)
    assert needed / GB == pytest.approx(4.47, abs=0.01)


def test_without_a_prefill_estimate_the_transfer_gets_a_third_of_the_budget():
    """Prefill time is the one term that cannot come from the model config, so
    its absence falls back to a share rather than to zero -- which would claim
    the whole budget and understate the requirement threefold."""
    assert transfer_budget_seconds(ttft_budget_seconds=0.6) == pytest.approx(0.2)


def test_a_prefill_estimate_overrides_the_share():
    assert transfer_budget_seconds(
        ttft_budget_seconds=0.6, prefill_seconds=0.1
    ) == pytest.approx(0.5)


def test_prefill_exceeding_the_budget_leaves_no_transfer_time():
    """Not a negative requirement, which would read as "no requirement"."""
    assert transfer_budget_seconds(ttft_budget_seconds=0.2, prefill_seconds=0.5) is None
    assert transfer_budget_seconds(ttft_budget_seconds=0.2, prefill_seconds=0.2) is None


def test_the_2_5gbe_measurement_is_an_order_of_magnitude_short():
    """Measured on a real pair: 0.293 GB/s moves a 70B 4K prompt's KV in 4.6 s,
    against a 300 ms budget."""
    seconds = transfer_seconds(1_342_177_280, 0.293 * GB)
    assert seconds == pytest.approx(4.6, abs=0.1)


# --- the verdict -------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (2.0, BandwidthVerdict.SUFFICIENT),
        (1.0, BandwidthVerdict.SUFFICIENT),
        (0.999, BandwidthVerdict.TIGHT),
        (0.3, BandwidthVerdict.TIGHT),
        (0.299, BandwidthVerdict.INSUFFICIENT),
        (0.06, BandwidthVerdict.INSUFFICIENT),
    ],
)
def test_the_verdict_bands(ratio, expected):
    assert verdict(ratio) is expected


def test_the_bands_are_ratios_so_they_apply_to_every_model():
    """An MLA model and a 70B GQA model differ by ~5x in what they need. A
    fixed GB/s cutoff would be wrong for one of them whichever value it took;
    the same link is sufficient for one and insufficient for the other."""
    link = 1.0 * GB
    gqa = kv_footprint(
        layers=80,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_heads=8,
        head_dim=128,
    )
    mla = kv_footprint(
        layers=61,
        seq_len=4096,
        kv_cache_dtype=KVCacheDType.FP16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )

    budget = transfer_budget_seconds(ttft_budget_seconds=0.5, prefill_seconds=0.2)
    gqa_ratio = link / required_bandwidth(gqa.bytes_per_request, budget)
    mla_ratio = link / required_bandwidth(mla.bytes_per_request, budget)

    assert verdict(gqa_ratio) is BandwidthVerdict.INSUFFICIENT
    assert verdict(mla_ratio) is BandwidthVerdict.SUFFICIENT
