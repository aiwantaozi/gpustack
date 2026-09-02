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
    PDMetricsPublic,
    PDRoleMetrics,
    _derive_role_rates,
    _fill_verdict,
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
        "counted_role_prompt_tokens": None,
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
        counted_role_prompt_tokens=440.0,
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
    r = _verdict(external_tokens=0.0, counted_role_prompt_tokens=32.0)
    assert r.request_count_source == "engine_tokens"
    assert r.kv_transfers_per_request == 0.0
    assert r.status == "aggregated"


def test_a_partial_transfer_is_a_fraction_not_a_whole_transfer():
    """What the per-transfer form cannot express: half a prompt arriving over
    the wire counts as one transfer there and as 0.5 here."""
    r = _verdict(external_tokens=220.0, counted_role_prompt_tokens=440.0)
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
    r = _verdict(
        external_tokens=0.0, counted_role_prompt_tokens=0.0, requests_total=3.0
    )
    assert r.request_count_source == "router_total"


# --- the declaration ------------------------------------------------------


def test_the_token_ratio_is_declared_over_the_receiving_role():
    declared = _declared()
    for key in ("external_tokens", "counted_role_prompt_tokens"):
        assert key in declared, key
        assert declared[key].counter_increase.scope is QueryScopeEnum.COUNTED_ROLE


def test_only_the_numerator_narrows_to_the_external_source():
    """The denominator must stay the whole prompt: narrowing both would make
    the ratio 1.0 by construction and it would never be able to fall."""
    declared = _declared()
    assert declared["external_tokens"].counter_increase.labels == {
        "source": "external_kv_transfer"
    }
    assert declared["counted_role_prompt_tokens"].counter_increase.labels is None


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
