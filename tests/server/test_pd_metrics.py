"""The PD verdict and the cache-missed prefill rate.

Every failure guarded here is silent, which is why each one gets a test rather
than a code comment: a group-wide sum of `prompt_tokens` is exactly twice the
traffic (a request increments it on prefill *and* on decode) and still reads as
a plausible token rate; a declared counter name without the `_total` the
Prometheus client appends at exposition returns an empty result, which surfaces
as "not measured" rather than as an error; and a verdict of `idle` against a
non-zero numerator asserts the one thing the data has already ruled out.
"""

from gpustack.schemas.metric_queries import (
    QueryScopeEnum,
    build_query,
    resolve_queries,
)
from gpustack.server.pd_metrics import (
    PDKVTransferMetrics,
    PDMemberMetrics,
    PDMetricsPublic,
    PDRoleMetrics,
    _MEMBER_FIELDS,
    _derive_role_rates,
    _fill_members,
    _fill_tokens,
    _fill_transfer,
    _fill_verdict,
    _preflight,
    _selectors,
    judge,
)
from gpustack.utils.metrics import get_builtin_metrics_config

WINDOW = 300


def _declared():
    return resolve_queries(get_builtin_metrics_config(), "pd")


def _with_roles(**roles) -> PDMetricsPublic:
    return PDMetricsPublic(available=True, roles=roles)


# --- the derived figure -------------------------------------------------


def test_missed_rate_is_prompt_minus_cached_over_window():
    result = _with_roles(
        prefill=PDRoleMetrics(prompt_tokens=30000.0, cached_prompt_tokens=12000.0)
    )
    _derive_role_rates(result, WINDOW)
    assert result.roles["prefill"].cache_missed_prompt_tokens_per_second == 60.0


def test_each_role_is_derived_from_its_own_operands():
    """Never from a group total: the same request is counted on both roles."""
    result = _with_roles(
        prefill=PDRoleMetrics(prompt_tokens=3000.0, cached_prompt_tokens=0.0),
        decode=PDRoleMetrics(prompt_tokens=3000.0, cached_prompt_tokens=3000.0),
    )
    _derive_role_rates(result, WINDOW)
    assert result.roles["prefill"].cache_missed_prompt_tokens_per_second == 10.0
    assert result.roles["decode"].cache_missed_prompt_tokens_per_second == 0.0


def test_missing_cached_counter_reads_as_unknown_not_as_the_raw_rate():
    """Falling back to the raw rate would silently reintroduce the very
    overstatement this figure exists to remove."""
    result = _with_roles(prefill=PDRoleMetrics(prompt_tokens=30000.0))
    _derive_role_rates(result, WINDOW)
    assert result.roles["prefill"].cache_missed_prompt_tokens_per_second is None


def test_missing_prompt_counter_reads_as_unknown():
    result = _with_roles(prefill=PDRoleMetrics(cached_prompt_tokens=1.0))
    _derive_role_rates(result, WINDOW)
    assert result.roles["prefill"].cache_missed_prompt_tokens_per_second is None


def test_counter_reset_inside_the_window_clamps_to_zero():
    """A negative token rate is not a reading anyone can act on."""
    result = _with_roles(
        prefill=PDRoleMetrics(prompt_tokens=100.0, cached_prompt_tokens=140.0)
    )
    _derive_role_rates(result, WINDOW)
    assert result.roles["prefill"].cache_missed_prompt_tokens_per_second == 0.0


def test_zero_window_derives_nothing_rather_than_dividing_by_it():
    result = _with_roles(
        prefill=PDRoleMetrics(prompt_tokens=1.0, cached_prompt_tokens=0.0)
    )
    _derive_role_rates(result, 0)
    assert result.roles["prefill"].cache_missed_prompt_tokens_per_second is None


# --- the declarations behind it -----------------------------------------


def test_token_counters_are_declared_per_role():
    """The invariant, not a formatting preference: without `by (role)` the sum
    spans both roles and doubles the traffic."""
    declared = _declared()
    for key in ("prompt_tokens", "cached_prompt_tokens"):
        assert declared[key].group_by == ["role"], key
        expression = build_query(
            declared[key], {QueryScopeEnum.GROUP: '{model_id="1"}'}, "300s"
        )
        assert "sum by (role)(" in expression, expression


def test_token_counters_use_the_exposed_total_suffixed_names():
    """The registered name has no suffix; the scraped series does. Reading the
    registered one returns nothing, which looks like "not measured"."""
    declared = _declared()
    assert (
        declared["prompt_tokens"].counter_increase.metric
        == "gpustack:prompt_tokens_total"
    )
    assert (
        declared["cached_prompt_tokens"].counter_increase.metric
        == "gpustack:prompt_tokens_cached_total"
    )


def test_every_per_role_declaration_has_a_field_to_land_in():
    """`_fill_roles` does `setattr(PDRoleMetrics(), key, value)`, so a
    declaration key with no matching field is a metric collected and dropped.
    Covers all per-role declarations, not only the two added here."""
    for key, query in _declared().items():
        if not query.group_by or "role" not in query.group_by:
            continue
        assert key in PDRoleMetrics.model_fields, key


# --- the verdict ------------------------------------------------------


def test_effective_when_kv_crosses_about_once_per_request():
    assert judge(transfers=7.0, requests=5.16) == "effective"


def test_aggregated_when_requests_are_routed_and_nothing_crosses():
    """The alarm this module exists for: correct answers, no errors, every
    instance RUNNING, and no KV moving."""
    assert judge(transfers=0.0, requests=100.0) == "aggregated"


def test_idle_needs_both_counters_to_agree():
    assert judge(transfers=0.0, requests=0.0) == "idle"


def test_zero_denominator_with_a_live_numerator_is_unmeasurable_not_idle():
    """Measured on a freshly started group: 2 transfers against a router
    counter Prometheus had not scraped yet.

    `idle` asserts "nobody called this group", which the numerator has already
    disproved — KV crossed, so something was called. Reporting `idle` would
    state the one thing the data rules out, and stop the reader looking.
    """
    assert judge(transfers=2.0, requests=0.0) == "unmeasurable"


def test_no_counter_at_all_is_unmeasurable():
    assert judge(transfers=None, requests=None) == "unmeasurable"
    assert judge(transfers=5.0, requests=None) == "unmeasurable"
    assert judge(transfers=None, requests=5.0) == "unmeasurable"


# --- which ratio answers, and why -----------------------------------------


def _values(**over):
    base = {
        "external_tokens": None,
        "recomputed_tokens": None,
        "requests_per_worker": None,
        "requests_total": None,
    }
    base.update(over)
    return base


def _verdict(**over) -> PDMetricsPublic:
    result = PDMetricsPublic(available=True)
    result.kv_transfer.count = over.pop("transfers", None)
    _fill_verdict(result, _values(**over))
    return result


def test_the_engines_own_token_split_wins_over_the_router_counter():
    """Both operands then come from one engine in one window, so there is no
    second counter to be missing, coarse, or scraped at a different moment."""
    r = _verdict(
        external_tokens=440.0,
        recomputed_tokens=0.0,
        requests_per_worker=8.0,
        transfers=8.0,
    )
    assert r.request_count_source == "engine_tokens"
    assert r.kv_transfers_per_request == 1.0
    assert r.status == "effective"


def test_a_prompt_decode_had_to_recompute_lowers_the_ratio():
    """Measured 2026-09-01: bypassing the router made decode prefill 32 tokens
    itself, and they landed in `local_compute`. That is the silent degradation
    — 200 OK, correct answer, no error anywhere — and this is the first signal
    that catches it without a router counter to compare against."""
    r = _verdict(external_tokens=0.0, recomputed_tokens=32.0)
    assert r.request_count_source == "engine_tokens"
    assert r.kv_transfers_per_request == 0.0
    assert r.status == "aggregated"


def test_one_member_of_a_pool_dropping_out_is_caught_as_degraded():
    """The gap the 0.01 threshold left open. A 3P1D group with one prefill's
    connector broken transfers two thirds of its traffic perfectly, and a
    single "is it above 0.01" test called that healthy — the exact shape the
    orchestrator comparison found in five of six competitors: a capability
    that stops working for part of the traffic while the deployment keeps
    serving and nothing says so."""
    r = _verdict(external_tokens=67.0, recomputed_tokens=33.0)
    assert r.status == "degraded"
    assert r.kv_transfers_per_request == 0.67


def test_the_band_does_not_swallow_the_silent_collapse():
    """`degraded` is a milder alarm, so it must not absorb the loud one."""
    assert _verdict(external_tokens=0.0, recomputed_tokens=100.0).status == (
        "aggregated"
    )


def test_a_fully_disaggregating_group_is_still_effective():
    """The healthy ratio is exactly 1.0 by the engine's own invariant, so the
    band must not fire on a group that is working."""
    assert _verdict(external_tokens=100.0, recomputed_tokens=0.0).status == "effective"


def test_a_warm_decode_prefix_cache_is_not_a_degradation():
    """🔴 The false positive the denominator exists to avoid. Measured
    2026-09-02 on a healthy 1P1D, three requests sharing a prefix: decode read
    local_compute=0, local_cache_hit=64.1, external=152.8. Over the prompt
    total that is 0.70 — inside the band — on a group where decode recomputed
    nothing at all. Over external+local_compute it is 1.0, which is the truth.
    """
    r = _verdict(external_tokens=152.8, recomputed_tokens=0.0)
    assert r.status == "effective"
    assert r.kv_transfers_per_request == 1.0


def test_the_transfer_ratio_is_never_banded():
    """🔴 The band needs a ratio whose healthy value is a known constant, and
    only the token ratio has one. Transfers-over-requests counts operations
    over requests, and how many operations a request costs is a connector
    property that runs above 1.0 — so the same 0.8 would call a connector
    averaging 0.9 transfers per request degraded while it is fine."""
    r = _verdict(requests_per_worker=100.0, transfers=67.0)
    assert r.request_count_source == "router_per_worker"
    assert r.status == "effective"


def test_the_band_is_opt_in_at_the_judge_level():
    assert judge(transfers=0.67, requests=1.0) == "effective"
    assert judge(transfers=0.67, requests=1.0, degraded_below=0.8) == "degraded"


def test_a_partial_transfer_is_a_fraction_not_a_whole_transfer():
    """What the per-transfer form cannot express: half a prompt arriving over
    the wire counts as one transfer there and as 0.5 here."""
    r = _verdict(external_tokens=220.0, recomputed_tokens=220.0)
    assert r.kv_transfers_per_request == 0.5


def test_an_engine_without_the_breakdown_falls_back_to_the_router_counter():
    """SGLang exports no per-source split, so the transfer/request form is the
    only one available there and must keep working untouched."""
    r = _verdict(requests_per_worker=5.0, transfers=5.0)
    assert r.request_count_source == "router_per_worker"
    assert r.kv_transfers_per_request == 1.0
    assert r.status == "effective"


def test_a_zero_prompt_total_does_not_claim_the_stronger_form():
    """An idle window has no tokens to attribute; falling through to the
    router counter is what distinguishes "nobody called it" from "it was
    called and nothing crossed"."""
    r = _verdict(external_tokens=0.0, recomputed_tokens=0.0, requests_total=3.0)
    assert r.request_count_source == "router_total"


# --- the declaration ------------------------------------------------------


def test_the_token_ratio_is_declared_over_the_receiving_role():
    declared = _declared()
    for key in ("external_tokens", "recomputed_tokens"):
        assert key in declared, key
        assert declared[key].counter_increase.scope is QueryScopeEnum.RECEIVING_ROLE


def test_the_receiving_role_stays_decode_when_the_counting_role_is_prefill():
    """The scopes agree on NIXL and part on a pushing connector. Reading this
    counter on prefill would tally the *sender's* own prompt tokens — every one
    of them local, so a healthy pair would report a ratio of zero."""
    declared = _declared()
    on_prefill = _selectors(7, "prefill")
    q = build_query(declared["external_tokens"], on_prefill, "15m")
    assert 'role="decode"' in q
    assert 'role="prefill"' not in q


def test_the_two_operands_name_the_two_sources_they_mean():
    """Both narrow, and to different sources. The third — `local_cache_hit` —
    is in neither, which is the whole design: it is not a shortfall, so it
    belongs in no part of a ratio measuring shortfall."""
    declared = _declared()
    assert declared["external_tokens"].counter_increase.labels == {
        "source": "external_kv_transfer"
    }
    assert declared["recomputed_tokens"].counter_increase.labels == {
        "source": "local_compute"
    }


def test_the_source_narrowing_reaches_the_query():
    declared = _declared()
    q = build_query(declared["external_tokens"], _selectors(7, "decode"), "15m")
    assert 'model_id="7"' in q
    assert 'role="decode"' in q
    assert 'source="external_kv_transfer"' in q


def test_the_transfer_count_comes_from_the_duration_histogram():
    """Not the bytes one: SGLang 0.5.15 dropped its bytes histogram, and the
    duration histogram is the only family both engines still export."""
    declared = _declared()
    assert (
        declared["transfers"].counter_increase.metric
        == "gpustack:pd_kv_transfer_seconds_count"
    )


# --- a connector with no counters of its own ------------------------------


def _mode(observable: bool, read_from_role: str = "decode"):
    from gpustack.schemas.pd_modes import PDMode, PDTransferMetrics

    return PDMode(
        name="vllm-ascend-mooncake",
        transfer_metrics=PDTransferMetrics(
            connector="mooncake",
            read_from_role=read_from_role,
            observable=observable,
        ),
    )


def _transfer(observable: bool) -> PDMetricsPublic:
    result = PDMetricsPublic(
        available=True, kv_transfer=PDKVTransferMetrics(counted_on_role="decode")
    )
    _fill_transfer(result, _transfer_values(), _mode(observable))
    return result


def _transfer_values() -> dict:
    return {
        "transfers": None,
        "failed": None,
        "expired": None,
        "kv_transfer_bytes_avg": None,
        "kv_transfer_seconds_p50": None,
        "kv_transfer_seconds_p95": None,
        "kv_transfer_seconds_p99": None,
        "seconds": None,
        "bytes": None,
    }


def test_a_mode_without_transfer_counters_is_no_longer_refused_outright():
    """It used to return `available=False` before running a single query, which
    was right only while effectiveness *was* the transfer ratio. The token
    split comes from the engine, so Mooncake now gets a verdict like any other
    mode — and refusing here would withhold the one signal it does have."""
    refusal, _, counted_role, declared = _preflight(_mode(observable=False))
    assert refusal is None
    assert counted_role == "decode"
    assert "external_tokens" in declared


def test_the_missing_speed_figures_say_so_instead_of_reading_as_zero():
    assert _transfer(observable=False).kv_transfer.rates_unavailable_reason


def test_an_idle_window_on_a_measurable_connector_stays_silent():
    """The distinction the flag has to be declared for: NIXL over a window with
    no traffic produces exactly the same empty values, and calling that "cannot
    be measured" would hide a connector that is working."""
    assert _transfer(observable=True).kv_transfer.rates_unavailable_reason is None


# --- per-request recomputation, and the volume figures ------------------


def test_the_recompute_percentiles_are_declared_on_the_receiving_role():
    """🔴 On PREFILL the same histogram is the work prefill exists to do, so a
    threshold read there fires on every healthy group. Measured on a working
    1P1D: decode had all 36 requests in `le=1.0` while prefill's spread from
    `le=10` to `+Inf`."""
    declared = _declared()
    for key in ("recomputed_tokens_p95", "recomputed_tokens_p99"):
        quantile = declared[key].histogram_quantile
        assert quantile is not None, key
        assert quantile.scope is QueryScopeEnum.RECEIVING_ROLE, key
        assert quantile.metric == "gpustack:request_prefill_kv_computed_tokens", key


def test_the_recompute_quantile_keeps_le_so_there_is_a_histogram_left():
    declared = _declared()
    expression = build_query(
        declared["recomputed_tokens_p99"],
        _selectors(1, "decode"),
        "300s",
    )
    assert "histogram_quantile(0.99, sum by (le)" in expression, expression
    assert "_bucket" in expression, expression
    assert 'role="decode"' in expression, expression


def test_the_percentiles_land_on_the_result():
    result = PDMetricsPublic(available=True)
    _fill_tokens(
        result,
        {
            "external_tokens": None,
            "recomputed_tokens_p95": 0.0,
            "recomputed_tokens_p99": 4096.0,
        },
        WINDOW,
    )
    assert result.recomputed_tokens_p95 == 0.0
    assert result.recomputed_tokens_p99 == 4096.0


def test_the_token_rate_is_over_wall_clock_and_left_in_tokens():
    """Bytes need the model's KV footprint, which means reading the pretrained
    config — a disk or network read that must not sit on a polled endpoint.
    The conversion belongs to the budget endpoint, which already does it."""
    result = PDMetricsPublic(available=True)
    _fill_tokens(result, {"external_tokens": 6000.0}, WINDOW)
    assert result.kv_transfer.external_tokens == 6000.0
    assert result.kv_transfer.external_tokens_per_second == 20.0


def test_a_zero_window_derives_no_token_rate_rather_than_dividing_by_it():
    result = PDMetricsPublic(available=True)
    _fill_tokens(result, {"external_tokens": 6000.0}, 0)
    assert result.kv_transfer.external_tokens == 6000.0
    assert result.kv_transfer.external_tokens_per_second is None


def test_the_volume_figures_survive_the_router_fallback_path():
    """`_fill_verdict` returns as soon as it has a verdict, so these have to
    be filled independently — a mode judged from the router counter still has
    the engine's token accounting."""
    result = PDMetricsPublic(available=True)
    result.kv_transfer.count = 40.0
    values = _values(requests_per_worker=40.0, external_tokens=6000.0)
    _fill_verdict(result, values)
    _fill_tokens(result, values, WINDOW)
    assert result.request_count_source == "router_per_worker"
    assert result.kv_transfer.external_tokens_per_second == 20.0


# --- per member, from the router ----------------------------------------


def test_member_counters_are_grouped_on_the_upstream_url_not_the_host():
    """🔴 One host runs several members on an xPyD group. Grouping on
    `worker_name` would collapse every member of a host into one series, which
    on a single-host group is all of them."""
    declared = _declared()
    for key in (
        "member_prefill_requests",
        "member_decode_requests",
        "member_decode_errors",
    ):
        assert declared[key].group_by == ["worker"], key
        expression = build_query(declared[key], _selectors(1, "decode"), "300s")
        assert "sum by (worker)(" in expression, expression


def test_member_counters_use_the_exposed_total_suffixed_names():
    declared = _declared()
    assert (
        declared["member_decode_errors"].counter_increase.metric
        == "gpustack:pd_router_decode_errors_total"
    )


def test_every_per_member_declaration_has_a_field_to_land_in():
    """`_fill_members` maps each declaration key to a field; a key with no
    mapping is a metric collected and dropped."""
    for key, query in _declared().items():
        if not query.group_by or "worker" not in query.group_by:
            continue
        assert key in _MEMBER_FIELDS, key
        assert _MEMBER_FIELDS[key] in PDMemberMetrics.model_fields, key


def test_members_are_keyed_by_worker_and_carry_their_own_figures():
    result = PDMetricsPublic(available=True)
    _fill_members(
        result,
        {
            "member_decode_requests": {
                "http://10.0.0.1:40000": 30.0,
                "http://10.0.0.2:40000": 15.0,
            },
            "member_decode_errors": {"http://10.0.0.2:40000": 4.0},
        },
    )
    assert set(result.members) == {"http://10.0.0.1:40000", "http://10.0.0.2:40000"}
    assert result.members["http://10.0.0.1:40000"].decode_requests == 30.0
    assert result.members["http://10.0.0.1:40000"].decode_errors is None
    assert result.members["http://10.0.0.2:40000"].decode_errors == 4.0


def test_an_unlabelled_member_series_is_dropped_rather_than_folded_in():
    result = PDMetricsPublic(available=True)
    _fill_members(result, {"member_decode_requests": {"": 45.0}})
    assert result.members == {}


def test_a_member_the_router_never_mentioned_is_absent_not_zero():
    """The absence is the finding: a zero row would say "measured, and it took
    nothing" where the truth is "the router has not mentioned it"."""
    result = PDMetricsPublic(available=True)
    _fill_members(result, {"member_decode_requests": {"http://10.0.0.1:40000": 45.0}})
    assert "http://10.0.0.2:40000" not in result.members


def test_the_untrustworthy_router_gauge_stays_unmapped():
    """🔴 Measured 2026-09-07 on a 1P1D that was serving traffic:
    `vllm_router_active_workers` read 0. Surfacing it would show "no members"
    on a healthy group."""
    config = get_builtin_metrics_config()
    mapping = config["runtime_mapping"]["vLLM"]["*"]
    assert "vllm_router_active_workers" not in mapping
    assert "vllm_router_pd_decode_errors" in mapping


def test_a_healthy_recompute_tail_reads_below_one_not_zero():
    """🔴 The false alarm this pins. vLLM's first bucket is `le=1.0` and
    `histogram_quantile` interpolates inside the bucket it lands in, so a
    group that recomputed nothing reports `quantile * 1.0` — measured 0.95 and
    0.99 on a working 1P1D. A reader (or a threshold) treating 0.99 as "one
    token recomputed" would alarm on a perfect deployment; the real signal is
    the jump past the next bucket edges, which are 2, 5 and 10."""
    result = PDMetricsPublic(available=True)
    _fill_tokens(
        result,
        {
            "external_tokens": 186.0,
            "recomputed_tokens_p95": 0.95,
            "recomputed_tokens_p99": 0.99,
        },
        WINDOW,
    )
    # Interpolation inside the first bucket, not a recomputation.
    assert result.recomputed_tokens_p95 < 1.0
    assert result.recomputed_tokens_p99 < 1.0
