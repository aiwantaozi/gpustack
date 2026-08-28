"""Declared PromQL, so a metric change is a config change.

The queries a feature runs are as much a part of "how this engine reports
itself" as the metric names are, and the names already live in
`metrics_config.yaml`. Keeping the queries anywhere else means a renamed
counter is edited in one file and the expression that reads it in another —
which is exactly the drift the normalization layer exists to prevent.

Each form earns its place by a kind of metric the others get wrong, never by
convenience: every form is a branch the query builder has to pick between, and
cache-providers already shows what five of them cost to read.

**Scope is per operand, not per query.** The effectiveness ratio divides a
counter scoped to one role by a counter scoped to the whole group — the
numerator is decode's transfers (or prefill's, depending on the connector),
the denominator is what the router dispatched. A single scope per query cannot
say that, and guessing it wrong silently changes what the number means.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, model_validator


class QueryScopeEnum(str, Enum):
    """Which series a term selects."""

    GROUP = "group"
    """Everything under this model: `{model_id="N"}`."""

    COUNTED_ROLE = "counted_role"
    """Only the role whose transfer counter is authoritative for this
    connector — decode where the connector pulls, prefill where it pushes.
    Resolved by the caller from the catalog, never named here: it is a
    property of the connector, not of the query."""


class MetricTerm(BaseModel):
    """One counter and the series it is taken over."""

    metric: str
    scope: QueryScopeEnum = QueryScopeEnum.GROUP


class HistogramQuantile(BaseModel):
    """A quantile of a histogram's observed values."""

    quantile: float
    metric: str
    """Base name; `_bucket` is appended."""
    scope: QueryScopeEnum = QueryScopeEnum.GROUP


class MetricQuery(BaseModel):
    """How to compute one value. Exactly one form is set.

    Four forms, and each earns its place by a kind of metric that the others
    get wrong rather than by convenience:

    - a counter's total over the window
    - the ratio of two counters' totals
    - a gauge's average over the window — `increase()` on a gauge is
      meaningless, and a queue depth is a gauge
    - a histogram quantile — where a degradation shows first, and what a mean
      is guaranteed to hide

    A histogram's *average* needs no form of its own: it is `ratio_increase`
    over the same family's `_sum` and `_count`.
    """

    counter_increase: Optional[MetricTerm] = None
    """Total increase over the window: `sum(increase(m{...}[w]))`."""

    ratio_increase: Optional[Dict[str, MetricTerm]] = None
    """`{numerator, denominator}` — the ratio of two increases over the same
    window. The operands are summed before dividing, so the result is weighted
    by actual traffic rather than being an average of per-series ratios."""

    gauge_avg: Optional[MetricTerm] = None
    """A gauge's mean over the window: `avg_over_time`. For a gauge whose
    question is "how much of the window did this hold", not "what does it hold
    now"."""

    gauge_last: Optional[MetricTerm] = None
    """A gauge's most recent sample: `last_over_time`. Queue depth is the case
    this exists for — it describes *now*, and a mean over the window answers a
    different question badly in both directions: a backlog that formed a minute
    ago is divided by fifteen and reads as calm, while one that cleared ten
    minutes ago keeps being reported until the window rolls past it.

    The window is still the lookback, not the aggregation: it bounds how stale
    a sample may be before the series is treated as absent, which is what keeps
    a stopped exporter from reporting its last value forever."""

    histogram_quantile: Optional[HistogramQuantile] = None
    """`histogram_quantile(q, sum by (le) (rate(m_bucket[w])))`."""

    group_by: Optional[List[str]] = None
    """Labels to keep instead of collapsing. `["role"]` is why this exists:
    prefill and decode are not one population, so a figure averaged over both
    describes neither — which is the whole premise of running PD at all."""

    @model_validator(mode="after")
    def _exactly_one_form(self):
        forms = [
            name
            for name in (
                "counter_increase",
                "ratio_increase",
                "gauge_avg",
                "gauge_last",
                "histogram_quantile",
            )
            if getattr(self, name)
        ]
        if len(forms) != 1:
            raise ValueError(
                "a metric query sets exactly one form (counter_increase | "
                "ratio_increase | gauge_avg | gauge_last | "
                "histogram_quantile), got "
                f"{forms or 'none'}"
            )
        if self.ratio_increase:
            missing = {"numerator", "denominator"} - set(self.ratio_increase)
            if missing:
                raise ValueError(f"ratio_increase is missing {sorted(missing)}")
        return self


def build_query(
    query: MetricQuery, selectors: Dict[QueryScopeEnum, str], window: str
) -> str:
    """One declaration -> one PromQL expression.

    `selectors` is supplied by the caller, which is what keeps authorisation
    server-side: the scope names a *kind* of selector and the caller decides
    what it expands to, so a declaration can never widen its own reach.
    """

    by = ""
    if query.group_by:
        by = f" by ({', '.join(query.group_by)})"

    def term(value: MetricTerm) -> str:
        return f"sum{by}(increase({value.metric}{selectors[value.scope]}[{window}]))"

    if query.counter_increase:
        return term(query.counter_increase)
    if query.gauge_avg:
        value = query.gauge_avg
        return (
            f"avg{by}(avg_over_time({value.metric}"
            f"{selectors[value.scope]}[{window}]))"
        )
    if query.gauge_last:
        value = query.gauge_last
        # Averaged across series exactly as `gauge_avg` is, so switching a
        # metric between the two forms changes only the time aggregation. A
        # move to `sum` would silently redefine a role's queue from "how deep
        # per replica" to "how deep in total", which is a different number on
        # any group wider than 1P1D.
        return (
            f"avg{by}(last_over_time({value.metric}"
            f"{selectors[value.scope]}[{window}]))"
        )
    if query.histogram_quantile:
        value = query.histogram_quantile
        # `le` must survive the aggregation or there is no histogram left to
        # take a quantile of.
        labels = ["le"] + list(query.group_by or [])
        return (
            f"histogram_quantile({value.quantile}, sum by ({', '.join(labels)}) "
            f"(rate({value.metric}_bucket{selectors[value.scope]}[{window}])))"
        )
    numerator = term(query.ratio_increase["numerator"])
    denominator = term(query.ratio_increase["denominator"])
    return f"{numerator} / {denominator}"


def resolve_queries(
    config: dict, feature: str, version: Optional[str] = None
) -> Dict[str, MetricQuery]:
    """The query set for `feature`, honouring version ranges.

    Same resolution shape as `runtime_mapping`: a `"*"` default plus optional
    version ranges, with a matching range winning key by key. The version axis
    is here because our own unified metric names can change — during a rolling
    upgrade two workers publish two vintages of them, and an expression that
    assumes the newer one returns an empty result against the older, which
    reads exactly like "nothing happened".

    ⚠️ Only `"*"` is populated today. The axis is structure, not a claim that
    the names have already changed.
    """
    from gpustack.utils import version as version_utils

    section = (config.get("metric_queries") or {}).get(feature) or {}
    resolved: Dict[str, MetricQuery] = {
        key: MetricQuery.model_validate(value)
        for key, value in (section.get("*") or {}).items()
    }
    if not version:
        return resolved

    valid = version_utils.is_valid_version_str(version)
    for ver_range, queries in section.items():
        if ver_range == "*":
            continue
        matched = (
            version_utils.in_range(version, ver_range)
            if valid
            else version == ver_range
        )
        if not matched:
            continue
        for key, value in (queries or {}).items():
            resolved[key] = MetricQuery.model_validate(value)
    return resolved
