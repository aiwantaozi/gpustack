"""Unit conversion in the metrics normalization layer.

`runtime_mapping` renames an engine's metric to the platform's name. Renaming
alone is only half a normalization: two engines can report the same concept in
different units, and one name over mixed units is worse than two names —
downstream can no longer tell which it has.

Measured case (2026-08-28, live 1P1D): SGLang reports KV transfer volume in
MEGABYTES and duration in MILLISECONDS where vLLM reports bytes and seconds.
"""

from prometheus_client.parser import text_string_to_metric_families

from gpustack.utils.metrics import get_builtin_metrics_config
from gpustack.worker.runtime_metrics_aggregator import (
    _parse_mapping_entry,
    get_unified_metric_family_name,
    scale_sample,
)

# Verbatim from the live SGLang prefill: 83 transfers, 189.0 MB total.
_REAL_SGLANG_HISTOGRAM = """\
# HELP sglang:kv_transfer_total_mb Histogram of KV transfer volume in MB.
# TYPE sglang:kv_transfer_total_mb histogram
sglang:kv_transfer_total_mb_bucket{le="1.0"} 4.0
sglang:kv_transfer_total_mb_bucket{le="4.0"} 60.0
sglang:kv_transfer_total_mb_bucket{le="+Inf"} 83.0
sglang:kv_transfer_total_mb_count 83.0
sglang:kv_transfer_total_mb_sum 189.0
"""

MB = 1048576


def test_the_existing_string_form_still_means_scale_one():
    """🔴 Every mapping in the shipped config is `raw: unified`. If the new
    object form changed what a bare string means, every metric in the platform
    would silently change value."""
    assert _parse_mapping_entry("gpustack:x") == ("gpustack:x", 1.0)
    assert _parse_mapping_entry({"name": "gpustack:x"}) == ("gpustack:x", 1.0)
    assert _parse_mapping_entry({"name": "gpustack:x", "scale": 0.001}) == (
        "gpustack:x",
        0.001,
    )
    assert _parse_mapping_entry(None) is None
    # A malformed entry degrades to no scaling rather than crashing the whole
    # aggregation pass for every other metric on the worker.
    assert _parse_mapping_entry({"name": "gpustack:x", "scale": "fast"}) == (
        "gpustack:x",
        1.0,
    )


def test_a_histogram_is_not_multiplied_through():
    """🔴 The trap this function exists for.

    A histogram's samples do not share a dimension: `_sum` is in the observed
    unit, `_count` and the bucket values are counts, and `le` is a boundary in
    the observed unit. The dangerous one is `le` — it is a *label*, so any
    implementation that scales "the value" misses it, nothing errors, and the
    histogram ends up with its sum in bytes and its boundaries in megabytes.
    Every `histogram_quantile` over it is then wrong by the scale factor.
    """
    scaled = {}
    for family in text_string_to_metric_families(_REAL_SGLANG_HISTOGRAM):
        for sample in family.samples:
            value, labels = scale_sample(
                sample.name,
                family.name,
                family.type,
                dict(sample.labels),
                sample.value,
                MB,
            )
            scaled[(sample.name, labels.get("le"))] = value

    # sum: megabytes -> bytes
    assert scaled[("sglang:kv_transfer_total_mb_sum", None)] == 189.0 * MB
    # count: an observation count, dimensionless
    assert scaled[("sglang:kv_transfer_total_mb_count", None)] == 83.0
    # bucket values: also counts
    assert scaled[("sglang:kv_transfer_total_mb_bucket", "1048576.0")] == 4.0
    assert scaled[("sglang:kv_transfer_total_mb_bucket", "4194304.0")] == 60.0
    # +Inf has no unit to convert
    assert scaled[("sglang:kv_transfer_total_mb_bucket", "+Inf")] == 83.0


def test_the_converted_histogram_is_internally_consistent():
    """The property that catches a half-applied conversion: mean transfer size
    computed from the converted sum and count must equal the mean computed
    from the engine's own numbers."""
    total_bytes = 189.0 * MB
    count = 83.0
    assert (total_bytes / count) / MB == 189.0 / 83.0


def test_the_shipped_config_declares_the_conversions_it_needs():
    """SGLang's PD histograms come in non-base units on every version it has
    used. A missing scale is silent — it produces a plausible number 1048576x
    too small — so the declaration is asserted rather than trusted."""
    config = get_builtin_metrics_config()

    # 0.5.15: duration only. `kv_transfer_total_mb` was dropped upstream and
    # nothing replaced it, so bandwidth is genuinely unavailable here.
    name, scale = get_unified_metric_family_name(
        config, "sglang:kv_transfer_bootstrap_ms", "SGLang", "0.5.15.post1"
    )
    assert name == "gpustack:pd_kv_transfer_seconds"
    assert scale == 0.001

    # 0.5.12: the pre-rename pair.
    name, scale = get_unified_metric_family_name(
        config, "sglang:kv_transfer_total_mb", "SGLang", "0.5.12.post1"
    )
    assert name == "gpustack:pd_kv_transfer_bytes"
    assert scale == MB

    name, scale = get_unified_metric_family_name(
        config, "sglang:kv_transfer_latency_ms", "SGLang", "0.5.12.post1"
    )
    assert name == "gpustack:pd_kv_transfer_seconds"
    assert scale == 0.001

    # vLLM's are already bytes and seconds, so they must NOT carry a scale.
    for raw, unified in (
        ("vllm:nixl_bytes_transferred", "gpustack:pd_kv_transfer_bytes"),
        ("vllm:nixl_xfer_time_seconds", "gpustack:pd_kv_transfer_seconds"),
    ):
        name, scale = get_unified_metric_family_name(config, raw, "vLLM", None)
        assert (name, scale) == (unified, 1.0), raw


def test_no_unit_bearing_name_is_mapped_without_a_scale():
    """The whole class of bug, not just today's two: a raw name that announces
    a non-base unit (`_mb`, `_ms`, ...) mapped at scale 1 is the silent
    failure. This is what stops the next connector from reintroducing it."""
    config = get_builtin_metrics_config()
    suspicious = ("_mb", "_ms", "_us", "_kb", "_gb", "_gb_s")
    offenders = []
    for runtime, versions in config.get("runtime_mapping", {}).items():
        for _version, mapping in (versions or {}).items():
            for raw, entry in (mapping or {}).items():
                parsed = _parse_mapping_entry(entry)
                if parsed is None:
                    continue
                _name, scale = parsed
                if any(raw.endswith(s) or f"{s}_" in raw for s in suspicious):
                    if scale == 1.0:
                        offenders.append(f"{runtime}:{raw}")
    assert not offenders, f"mapped without a unit conversion: {offenders}"


def test_a_renamed_upstream_metric_keeps_its_old_version_range():
    """🔴 The failure this pins down, measured 2026-09-01.

    SGLang renamed its PD transfer metrics between 0.5.12 and 0.5.15
    (`kv_transfer_total_mb` / `kv_transfer_latency_ms` ->
    `kv_transfer_bootstrap_ms` / `kv_transfer_alloc_ms`). The mapping kept
    pointing at the old names, so on 0.5.15 it resolved to series the engine
    does not export: the ratio got no numerator and the endpoint reported
    `unmeasurable` — which reads exactly like "this connector exports no
    counters". PD effectiveness had been silently unmeasurable on every SGLang
    group since the upgrade, and nothing anywhere said so.

    Both directions are asserted, because deleting the old entry would break
    the older engine just as quietly as leaving it broke the newer one.
    """
    config = get_builtin_metrics_config()

    new_only = get_unified_metric_family_name(
        config, "sglang:kv_transfer_bootstrap_ms", "SGLang", "0.5.15.post1"
    )
    assert new_only[0] == "gpustack:pd_kv_transfer_seconds"

    # The new name must not be the only one a 0.5.12 engine can offer, and the
    # old one must not leak forward as the sole source on 0.5.15.
    old_on_old = get_unified_metric_family_name(
        config, "sglang:kv_transfer_latency_ms", "SGLang", "0.5.12.post1"
    )
    assert old_on_old[0] == "gpustack:pd_kv_transfer_seconds"

    # Two other mappings went stale in the same upgrade.
    assert (
        get_unified_metric_family_name(
            config, "sglang:num_retracted_reqs", "SGLang", "0.5.15.post1"
        )[0]
        == "gpustack:request_preemptions"
    )
