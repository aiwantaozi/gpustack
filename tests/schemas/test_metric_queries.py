"""Declared PromQL: the forms, the scopes, and the version fallback."""

import pytest
from pydantic import ValidationError

from gpustack.schemas.metric_queries import (
    MetricQuery,
    QueryScopeEnum,
    build_query,
    resolve_queries,
)
from gpustack.utils.metrics import get_builtin_metrics_config

SELECTORS = {
    QueryScopeEnum.GROUP: '{model_id="44"}',
    QueryScopeEnum.COUNTED_ROLE: '{model_id="44",role="decode"}',
}


def test_a_query_sets_exactly_one_form():
    """Two forms would make the builder pick one silently, and the two produce
    different numbers — the failure would be a plausible wrong value, not an
    error."""
    with pytest.raises(ValidationError):
        MetricQuery()
    with pytest.raises(ValidationError):
        MetricQuery(
            counter_increase={"metric": "a"},
            ratio_increase={
                "numerator": {"metric": "b"},
                "denominator": {"metric": "c"},
            },
        )
    with pytest.raises(ValidationError):
        MetricQuery(ratio_increase={"numerator": {"metric": "b"}})


def test_scope_is_per_operand():
    """🔑 The reason scope is not a property of the whole query.

    The effectiveness ratio divides transfers counted on ONE role by requests
    counted across the group. A single scope cannot express that, and either
    choice silently changes what the number means: role-scoping the
    denominator counts only the requests the router sent to one member, and
    group-scoping the numerator counts a healthy pair's zero on the other side
    as if it were a shortfall.
    """
    query = MetricQuery(
        ratio_increase={
            "numerator": {
                "metric": "gpustack:pd_kv_transfer_bytes_count",
                "scope": "counted_role",
            },
            "denominator": {
                "metric": "gpustack:pd_router_requests_total",
                "scope": "group",
            },
        }
    )
    built = build_query(query, SELECTORS, "15m")
    assert 'bytes_count{model_id="44",role="decode"}' in built
    assert 'requests_total{model_id="44"}' in built


def test_the_default_scope_is_the_group():
    """A term that says nothing about scope must not narrow to a role: the
    narrower answer looks the same and is a smaller number."""
    query = MetricQuery(counter_increase={"metric": "x"})
    assert build_query(query, SELECTORS, "5m") == 'sum(increase(x{model_id="44"}[5m]))'


def test_the_shipped_declaration_covers_what_the_endpoint_reads():
    """The endpoint looks these up by name, so a renamed key is a silently
    missing value rather than an error."""
    declared = resolve_queries(get_builtin_metrics_config(), "pd")
    for key in (
        "transfers",
        "bytes",
        "seconds",
        "failed",
        "expired",
        "requests_per_worker",
        "requests_total",
        "rate_series",
        "ratio_series",
    ):
        assert key in declared, key


def test_a_version_range_overrides_key_by_key():
    """Not wholesale: a version that changes one metric should not have to
    restate the other eight, and restating them is how they drift."""
    config = {
        "metric_queries": {
            "pd": {
                "*": {
                    "a": {"counter_increase": {"metric": "old_a"}},
                    "b": {"counter_increase": {"metric": "old_b"}},
                },
                ">=2.0.0": {"a": {"counter_increase": {"metric": "new_a"}}},
            }
        }
    }
    resolved = resolve_queries(config, "pd", "2.1.0")
    assert resolved["a"].counter_increase.metric == "new_a"
    assert resolved["b"].counter_increase.metric == "old_b"

    # Outside the range, and with no version at all, the default stands.
    assert (
        resolve_queries(config, "pd", "1.9.0")["a"].counter_increase.metric == "old_a"
    )
    assert resolve_queries(config, "pd")["a"].counter_increase.metric == "old_a"


def test_an_unknown_feature_yields_nothing_rather_than_raising():
    """The caller turns an empty set into a reasoned `available=false`, which
    is more useful than a traceback from a config lookup."""
    assert resolve_queries({}, "pd") == {}
