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
