"""Whether a disaggregated group is actually disaggregating, from Prometheus.

The three things a PD deployment hides — is KV really crossing between the
roles, is the transport still as fast as it was, is anything being dropped
between the two hops — are all invisible in the deployment's own state: a
group that has silently collapsed to aggregated serving returns correct
answers, logs no errors, and shows every instance RUNNING.

**Read, not computed-and-stored.** The engines and the router expose the
counters; the worker's aggregator normalizes them onto `gpustack:pd_*` with
`model_id` and `role` labels; Prometheus scrapes the worker. So the numbers
already exist, correctly labelled, over a network path that handles tunnelled
workers. This module turns them into an answer at request time.

The label selector is injected here, server-side, so a caller can only ever
read the series of a model it was authorised for — the same discipline the
cache-service metrics endpoint follows.
"""

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp
from pydantic import BaseModel

from gpustack.schemas.metric_queries import (
    QueryScopeEnum,
    build_query,
    resolve_queries,
)
from gpustack.schemas.pd_modes import PDMode
from gpustack.utils.metrics import get_builtin_metrics_config
from gpustack.server.prometheus_query import (
    instant_value,
    promql_regex_literal,
    prometheus_url,
    query_instant,
    query_range,
)

logger = logging.getLogger(__name__)

AGGREGATED_RATIO = 0.01
"""Below this, no meaningful share of the routed requests moved any KV.

Not zero: a group can be mid-restart, or a stray probe can cross while real
traffic does not. What matters is the order of magnitude — a working pair
transfers about once per request, so anything near zero is a different
regime, not a worse one."""

DEGRADED_RATIO = 0.8
"""Below this, KV is crossing for some requests and not for others.

The band `AGGREGATED_RATIO` alone cannot see. Measured against a 3P1D group
with one prefill's connector broken, the ratio sits near 0.67 — two thirds of
the traffic disaggregating perfectly — and a single "is it above 0.01" test
calls that healthy. It is the shape the comparison report found across five of
the six orchestrators: a capability that stops working for part of the traffic
while the deployment keeps serving and nothing says so.

🔴 Only applied to a ratio whose healthy value is known to be 1.0, which is
why `judge` takes it as an argument instead of reading it here. See the
`degraded_below` parameter."""

_COLLECT_DEADLINE_SECONDS = 15.0


class PDRoleMetrics(BaseModel):
    """One role's own numbers.

    🔑 Per role because that is the premise of running PD at all: prefill owns
    time-to-first-token and decode owns time-per-output-token, so a figure
    averaged across both describes neither. Every value here is null when the
    engine did not report it in the window — never zero, which would read as
    "measured, and it was nothing".
    """

    pending_requests: Optional[float] = None
    """Mean queue depth over the window.

    The only objective signal for whether the prefill:decode ratio is right,
    and which way it is wrong: a queue that only ever builds on one side is
    that side asking for more replicas."""

    running_requests: Optional[float] = None
    """Mean requests in flight over the window."""

    time_to_first_token_seconds: Optional[float] = None
    """Mean TTFT. Owned by prefill — decode's TTFT is the time from its own
    first forward pass, not what a user waited."""

    time_per_output_token_seconds: Optional[float] = None
    """Mean inter-token latency. Owned by decode."""

    prompt_tokens: Optional[float] = None
    """Prefill tokens the role processed in the window.

    Per role because a request increments this on both of them, so a group-wide
    sum is exactly twice the traffic — and twice a token rate still reads as a
    plausible token rate."""

    cached_prompt_tokens: Optional[float] = None
    """Of those, the ones served from cache — local HBM prefix cache and the
    shared cache service together. Both engines report the sum: vLLM documents
    its counter as "local + external", SGLang's carries a `cache_source` label
    that the aggregator collapses."""

    cache_missed_prompt_tokens_per_second: Optional[float] = None
    """`(prompt_tokens - cached_prompt_tokens) / window` — the tokens prefill
    actually had to compute.

    🔑 On prefill this is the throughput signal to scale on, and raw
    `prompt_tokens` is not: prefill is only compute-bound for tokens it
    computed, so a high hit rate makes the raw figure overstate the work by
    whatever share came from cache. ByteDance's HeteroScale rejected raw prefill
    TPS for this reason and kept only "KV cache missed prefill TPS".

    Null — not the raw rate — when the cached counter is missing, because
    falling back to the raw rate would reintroduce exactly the overstatement
    this figure exists to remove, and would do it silently."""


class PDKVTransferMetrics(BaseModel):
    """The KV transfer itself: is it happening, how fast, and is any of it
    being lost."""

    count: Optional[float] = None
    """Transfers completed in the window."""

    counted_on_role: Optional[str] = None
    """Which role's counter this came from — a property of the connector, not
    a choice: NIXL has decode PULL so decode counts, SGLang has prefill PUSH
    so prefill counts. A healthy pair reports zero on the other side, so
    reading the wrong one turns a working group into an alarm."""

    bytes_per_second: Optional[float] = None
    """Bytes moved divided by time spent moving them — throughput *while
    transferring*, not per wall-clock second. Dividing by wall clock would
    make a mostly-idle window read as a collapse in throughput."""

    bytes_per_transfer: Optional[float] = None
    """Mean transfer size. Falling here precedes a slowdown: smaller chunks
    pay the per-transfer overhead more often for the same volume."""

    seconds_p50: Optional[float] = None
    seconds_p95: Optional[float] = None
    seconds_p99: Optional[float] = None
    """Transfer duration percentiles. The tail is where a degrading path shows
    up first and where a mean is guaranteed to hide it."""

    rates_unavailable_reason: Optional[str] = None
    """Why the speed figures below are empty, when the ratio above is not.

    🔑 The two answer different questions and one can be available without the
    other. "Is KV crossing" now comes from the engine's own token accounting,
    which every V1 connector produces; "how fast" needs the connector's own
    byte and duration counters, which some export and some do not. Before this
    field the whole endpoint refused when the second was missing, and a reader
    could not tell "we cannot measure this mode" from "nothing is crossing".
    """

    failures: Optional[float] = None
    """Transfers the engine reported as failed."""

    leases_expired: Optional[float] = None
    """Requests dropped between the two hops: prefill computed their KV and
    nobody ever read it. Null — not zero — for connectors that export no such
    counter, because "we cannot see this" is not "it did not happen"."""


class PDMetricsPublic(BaseModel):
    """One window's answer for one disaggregated group.

    `available=false` carries why nothing could be measured and is never
    conflated with a bad measurement: "we cannot tell" and "PD stopped
    working" call for opposite reactions from whoever reads it.
    """

    available: bool = False
    reason: Optional[str] = None
    window_seconds: Optional[int] = None

    status: Optional[str] = None
    """The verdict:

    - `effective` — KV is crossing in proportion to the requests routed.
    - `degraded` — 🟠 KV is crossing for some of the traffic and not for the
      rest. One member of a pool has stopped participating while the others
      still work, so the group serves correctly at a fraction of the benefit.
      Only ever reported where the healthy ratio is known to be 1.0; see
      `judge`.
    - `aggregated` — 🔴 requests are being routed and effectively no KV is
      crossing. Disaggregation has silently collapsed: the deployment still
      answers correctly, logs nothing and shows every instance running, so
      this is the only place that failure is visible.
    - `idle` — nothing was routed and nothing crossed. Not a degradation; a
      model nobody calls transfers nothing.
    - `unmeasurable` — the window cannot be judged. Either no denominator
      exists at all, or the two counters disagree about it: a zero denominator
      against a non-zero numerator means KV demonstrably crossed, so the group
      is not idle and the router counter simply has not been scraped yet.
      Deliberately not reported as `idle` or `aggregated`.
    """

    kv_transfers_per_request: Optional[float] = None
    """The ratio behind `status`. Around 1.0 on a healthy pair."""

    routed_request_count: Optional[float] = None
    """The denominator: requests the router dispatched in the window."""

    request_count_source: Optional[str] = None
    """Where `routed_request_count` came from:

    - `engine_tokens` — the engine's own prompt-token breakdown. The strongest
      form: both operands come from one engine, so there is no second counter
      to be missing, coarse, or scraped at a different moment.
    - `router_per_worker` — per-worker counters, so a low ratio points at one
      decode rather than at "the group".
    - `router_total` — the route aggregate. Still answers "did anything get
      routed at all", which a ratio of zero is meaningless without, but
      localises nothing.
    - `none` — no counter at all.

    Surfaced rather than inferred, because the fallback is much weaker than
    the real thing and a reader has to be able to tell which they have."""

    kv_transfer: PDKVTransferMetrics = PDKVTransferMetrics()
    roles: Dict[str, PDRoleMetrics] = {}

    kv_transfers_per_request_series: List[List[Optional[float]]] = []
    kv_transfer_bytes_per_second_series: List[List[Optional[float]]] = []
    """`[timestamp, value]` points. History is what a stored verdict could
    never give: a degradation is a step in the line, with no threshold to
    pick."""


def _selectors(model_id: int, counted_role: str) -> dict:
    """What each declared scope expands to.

    Built here rather than in the declaration, and that is the authorisation
    boundary: a query in the catalog names a *kind* of scope and this decides
    what it selects, so no declaration can widen its own reach past the model
    the caller was allowed to see.
    """
    model = f'model_id="{promql_regex_literal(str(model_id))}"'
    role = f'role="{promql_regex_literal(counted_role)}"'
    return {
        QueryScopeEnum.GROUP: "{" + model + "}",
        QueryScopeEnum.COUNTED_ROLE: "{" + model + "," + role + "}",
        QueryScopeEnum.RECEIVING_ROLE: "{" + model + ',role="decode"}',
    }


def judge(
    transfers: Optional[float],
    requests: Optional[float],
    degraded_below: Optional[float] = None,
) -> str:
    """The verdict, and the three ways it can decline to give one.

    🔴 `degraded_below` is passed by exactly one caller, and the asymmetry is
    the point. A partial-degradation band needs a ratio whose healthy value is
    a known constant, and only the token ratio has one: vLLM guarantees
    `local_compute + local_cache_hit + external_kv_transfer = total`, so a
    fully disaggregating group reads exactly 1.0 and any shortfall is the
    share of prompt tokens that had to be recomputed.

    The transfer ratio has no such baseline. Its numerator counts transfer
    *operations* and its denominator counts requests, and how many operations
    one request costs is a property of the connector — the ratio runs above
    1.0 in practice. So the same 0.8 would read a healthy connector that
    averages 0.9 transfers per request as degraded, and would miss a third of
    the traffic failing on one that averages 1.4. Worse, the two operands come
    from different exporters scraped at different moments, where 20% is inside
    the noise. A band there would be a number with no meaning, fired at
    deployments that are fine — which is the failure mode this whole endpoint
    exists to avoid.

    - no denominator at all -> `unmeasurable`. An idle group and one that has
      degraded to aggregated serving are indistinguishable without knowing
      whether anything was routed, so this must not be reported as either.
    - nothing routed *and* nothing transferred -> `idle`. A model nobody called
      transfers nothing; that is not a degradation.
    - routed, but effectively no transfers -> `aggregated`. The alarm.

    🔑 `idle` is a positive claim — "nobody called this group" — so it is only
    made when the numerator agrees. A zero denominator against a non-zero
    numerator is not an idle group: KV demonstrably crossed, so something was
    called and the two counters simply disagree about the window. That happens
    on a group whose first requests have landed but whose router counter has
    not been scraped yet, and reporting it as `idle` would state the one thing
    the data has already ruled out — and stop the reader from looking further.
    `unmeasurable` is the honest answer: the window cannot be judged.
    """
    if requests is None or transfers is None:
        return "unmeasurable"
    if requests <= 0:
        return "idle" if transfers <= 0 else "unmeasurable"
    ratio = transfers / requests
    if ratio < AGGREGATED_RATIO:
        return "aggregated"
    if degraded_below is not None and ratio < degraded_below:
        return "degraded"
    return "effective"


def _fill_transfer(result: PDMetricsPublic, values: dict, mode=None) -> None:
    """The transfer figures, straight across."""
    transfer = result.kv_transfer
    if mode and mode.transfer_metrics and not mode.transfer_metrics.observable:
        # Declared, not inferred from the values being empty: a healthy NIXL
        # pair over an idle window also reports nothing, and the two must not
        # be told apart by guessing.
        transfer.rates_unavailable_reason = (
            f"The '{mode.name}' mode's KV connector exports no transfer "
            "counters, so transfer speed and duration cannot be measured. "
            "Whether KV is crossing is answered from the engine's own token "
            "accounting instead."
        )
    transfer.count = values["transfers"]
    transfer.failures = values["failed"]
    transfer.leases_expired = values["expired"]
    transfer.bytes_per_transfer = values["kv_transfer_bytes_avg"]
    transfer.seconds_p50 = values["kv_transfer_seconds_p50"]
    transfer.seconds_p95 = values["kv_transfer_seconds_p95"]
    transfer.seconds_p99 = values["kv_transfer_seconds_p99"]

    seconds = values["seconds"]
    if seconds and seconds > 0 and values["bytes"] is not None:
        # Divided by time spent transferring, not by wall clock: a window that
        # was mostly idle would otherwise read as a collapse in throughput.
        transfer.bytes_per_second = values["bytes"] / seconds


def _fill_verdict(result: PDMetricsPublic, values: dict) -> None:
    """The ratio, what it was taken over, and how good that denominator is.

    🔑 Two ratios can answer the same question, and the better one wins when
    the engine offers it.

    **Per token (`engine_tokens`).** vLLM splits every prompt token by origin
    and guarantees `local_compute + local_cache_hit + external_kv_transfer =
    total`. The ratio is taken over the first and third only — of the tokens
    decode could not get from its own cache, the share that arrived over the
    wire rather than being computed a second time. Both operands come from the
    same engine in the same window, which removes the whole class of problems
    the other form has: no router counter to be missing or coarse, no
    two-counter skew, and a partially-transferred prompt shows up as a
    fraction instead of counting as one transfer.

    ⚠️ `local_cache_hit` is excluded from the denominator on purpose. Those
    tokens were never going to cross, so counting them as a shortfall blames
    the connector for decode's prefix cache doing its job — measured at 0.70
    on a healthy group whose decode recomputed nothing.

    **Per transfer (`router_*`).** Transfers over requests the router
    dispatched. The only form available where the engine does not break prompt
    tokens down, which today is SGLang.

    `request_count_source` says which was used, and that is not bookkeeping:
    the forms differ in what a low value localises. Per-worker router counters
    point at one decode; the route aggregate points at "the group"; the token
    ratio points at the counted role as a whole. A reader has to be able to
    tell which they are looking at.
    """
    external = values.get("external_tokens")
    recomputed = values.get("recomputed_tokens")
    if external is not None and recomputed is not None:
        # Deliberately not the prompt total: decode's own prefix-cache hits are
        # the third source, and those tokens were never going to cross. Falling
        # through when this is zero is the honest answer for a window whose
        # every token came from that cache -- PD was not exercised, so there is
        # nothing here to judge, and the transfer counters below may still know.
        boundary = external + recomputed
        if boundary > 0:
            result.routed_request_count = boundary
            result.request_count_source = "engine_tokens"
            result.kv_transfers_per_request = external / boundary
            result.status = judge(external, boundary, degraded_below=DEGRADED_RATIO)
            return

    if values["requests_per_worker"] is not None:
        result.routed_request_count = values["requests_per_worker"]
        result.request_count_source = "router_per_worker"
    elif values["requests_total"] is not None:
        result.routed_request_count = values["requests_total"]
        result.request_count_source = "router_total"
    else:
        result.request_count_source = "none"

    count = result.kv_transfer.count
    if result.routed_request_count:
        result.kv_transfers_per_request = (count or 0.0) / result.routed_request_count
    result.status = judge(count, result.routed_request_count)


def _fill_roles(result: PDMetricsPublic, by_role: dict) -> None:
    """Per-role figures, for whichever roles the series actually carried.

    Not a fixed list: a group with no decode replica yet should show the roles
    it has rather than inventing an empty one.
    """
    for key, per_role in by_role.items():
        for role, value in per_role.items():
            if not role:
                # An unlabelled series belongs to a non-PD instance; it has no
                # role to attribute and must not be folded into one.
                continue
            entry = result.roles.setdefault(role, PDRoleMetrics())
            setattr(entry, key, value)


def _derive_role_rates(result: PDMetricsPublic, window_seconds: int) -> None:
    """The one per-role figure that is computed rather than read.

    Both operands are counters over the window, so the subtraction has to
    happen after they are both in — and it happens here, per role, so it can
    never be taken over a group sum that counted every request twice.
    """
    if window_seconds <= 0:
        return
    for entry in result.roles.values():
        if entry.prompt_tokens is None or entry.cached_prompt_tokens is None:
            continue
        missed = entry.prompt_tokens - entry.cached_prompt_tokens
        # Clamped: a counter reset inside the window, or the two counters being
        # scraped a scrape apart, can put the subtraction slightly below zero.
        # A negative token rate is not a reading anyone can act on.
        entry.cache_missed_prompt_tokens_per_second = max(missed, 0.0) / window_seconds


def _preflight(mode: Optional[PDMode]):
    """Everything that can make the answer "cannot tell" before any query runs.

    Separated because each of these is a *reason*, not a failure, and the
    caller must render them as such: no Prometheus, a connector that exports
    nothing, an empty declaration. Reporting any of them as a zero ratio would
    fire the loudest alarm this endpoint has at a deployment that is fine.

    Returns `(refusal | None, base_url, counted_role, declared_queries)`.
    """
    base_url = prometheus_url()
    if not base_url:
        return (
            PDMetricsPublic(
                available=False,
                reason=(
                    "No Prometheus is configured or reachable. Set "
                    "`prometheus_url` to query an external one, or enable the "
                    "built-in observability stack."
                ),
            ),
            None,
            None,
            None,
        )

    counted_role = "decode"
    if mode and mode.transfer_metrics:
        counted_role = mode.transfer_metrics.read_from_role or "decode"

    declared = resolve_queries(get_builtin_metrics_config(), "pd")
    if not declared:
        return (
            PDMetricsPublic(
                available=False,
                reason="No PD metric queries are declared in metrics_config.yaml",
            ),
            None,
            None,
            None,
        )
    return None, base_url, counted_role, declared


async def collect_pd_metrics(
    model_id: int,
    mode: Optional[PDMode],
    window_seconds: int,
    client: Optional[aiohttp.ClientSession] = None,
) -> PDMetricsPublic:
    """One window's PD metrics for one model."""
    refusal, base_url, counted_role, declared = _preflight(mode)
    if refusal is not None:
        return refusal
    selectors = _selectors(model_id, counted_role)
    window = f"{window_seconds}s"
    # The instant values; the two `*_series` entries are charted below over a
    # narrower rate window, so they are not run here.
    queries = {
        key: build_query(query, selectors, window)
        for key, query in declared.items()
        if not key.endswith("_series")
    }

    result = PDMetricsPublic(
        available=True,
        window_seconds=window_seconds,
        kv_transfer=PDKVTransferMetrics(counted_on_role=counted_role),
    )
    owned = client is None
    try:
        if owned:
            client = aiohttp.ClientSession()

        async def _run():
            import time

            now = time.time()
            values = {}
            by_role = {}
            for key, expression in queries.items():
                try:
                    rows = await query_instant(client, base_url, expression, now)
                except Exception as e:
                    # One failing query must not blank the rest: a mode with no
                    # per-worker router counter should still get its numerator.
                    logger.debug("PD metric query %r failed: %s", key, e)
                    rows = []
                if declared[key].group_by:
                    by_role[key] = {
                        row.get("metric", {}).get("role", ""): instant_value(row)
                        for row in rows
                    }
                else:
                    values[key] = instant_value(rows[0]) if rows else None

            _fill_transfer(result, values, mode)
            _fill_verdict(result, values)
            _fill_roles(result, by_role)
            _derive_role_rates(result, window_seconds)

            step = max(window_seconds // 60, 15)
            rate_window = f"{max(step * 4, 300)}s"
            # Same declarations, evaluated over a narrower window per point so
            # the line has resolution: one `increase()` spanning the whole
            # range would flatten every change into a single average.
            series = {
                key: build_query(query, selectors, rate_window)
                for key, query in declared.items()
                if key.endswith("_series")
            }
            for key, expression in series.items():
                try:
                    rows = await query_range(
                        client, base_url, expression, now - window_seconds, now, step
                    )
                except Exception as e:
                    logger.debug("PD metric series %r failed: %s", key, e)
                    continue
                if rows:
                    setattr(
                        result,
                        {
                            "ratio_series": "kv_transfers_per_request_series",
                            "rate_series": "kv_transfer_bytes_per_second_series",
                        }[key],
                        _points(rows[0]),
                    )

        await asyncio.wait_for(_run(), timeout=_COLLECT_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        return PDMetricsPublic(available=False, reason="Prometheus queries timed out")
    except (aiohttp.ClientError, OSError) as e:
        return PDMetricsPublic(
            available=False,
            reason=f"Prometheus is unreachable: {str(e) or e.__class__.__name__}",
        )
    finally:
        if owned and client is not None:
            await client.close()
    return result


def _points(entry: dict) -> List[List[Optional[float]]]:
    """`[timestamp, value]` pairs, with non-finite samples as gaps rather than
    zeros — a hole in a chart reads as "not measured", a zero reads as "no KV
    moved", and only one of those is true."""
    points: List[List[Optional[float]]] = []
    for pair in entry.get("values") or []:
        try:
            timestamp = float(pair[0])
            value = float(pair[1])
        except (IndexError, TypeError, ValueError):
            continue
        if value != value or value in (float("inf"), float("-inf")):
            points.append([timestamp, None])
        else:
            points.append([timestamp, value])
    return points
