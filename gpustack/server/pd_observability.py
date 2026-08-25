"""Whether disaggregation is actually happening, and whether it still runs
as fast as it used to.

Every failure this module watches for is *silent*: the deployment stays up,
the API keeps answering, latency is merely worse. Detection is therefore not
a nice-to-have on top of the feature — it is the part of the feature that
tells you the rest of it works.

**Two facts decide every number here, and both have already been got wrong
once.**

🔴 *The numerator is on the decode side.* NIXL is pull-based: decode reads
from prefill, so a completed transfer is counted where the read happened.
Measured on a working cross-host 1P1D, prefill's
``nixl_xfer_time_seconds_count`` sat at 0.0 for the entire run while
decode's went to 1.0. Read prefill and a healthy pair reports "no KV ever
moved" — the exact alarm this module exists to raise, fired at a deployment
that is fine. The same run gives the two readings of one `/v1/completions`:
decode 1 transfer over 1 routed request, a ratio of 1.0, against 0/1 = 0.0
from the other side. Which side owns the counter is a *declaration*
(``PDTransferMetrics.read_from_role``) rather than a constant here, so a
push-based connector stays a YAML change and the reason stays written down
next to the value.

🔴 *The denominator of the degradation check is the group's own past, not
the link's nameplate speed.* "Effective bandwidth versus line rate" looks
right and survives one environment: cross-host 2.5GbE measured 2.34 Gb/s
against a 2.5 Gb/s link, 94%. The same criterion on 910B2 RoCE measured
~17.6 Gb/s against 200 Gb/s — **9%**. One threshold cannot be both, so any
global one either alarms forever on Ascend or never alarms on Ethernet. The
comparison is against a baseline taken from this group, in the same unit,
on the same hardware, which measures *deterioration* — the thing anyone
actually cares about — and is immune to the (engine x transport x
accelerator) spread by construction. Its known blind spot is a group that
was already degraded when its baseline was taken; the backstop for that is
a coarse floor, declarable per connector and deliberately unset until
somebody calibrates one on real hardware.

The ratio's *denominator* is not counted here either: vllm-router already
exports per-worker request counters, and per-worker is what localises the
failure to one decode. A router that serves no metrics (vllm-ascend's proxy
example) leaves the check without one, and that is reported as
"unmeasurable" rather than quietly folded into a zero — a missing
denominator and a dead transport must not produce the same output.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel

from gpustack.schemas.models import DegradationReasonEnum
from gpustack.schemas.pd_modes import PDMode, PDTransferMetrics

logger = logging.getLogger(__name__)


AGGREGATED_RATIO = 0.01
"""A ratio at or below this counts as "no KV moved at all".

Not zero: a group that transferred twice while serving a thousand requests
has degraded to aggregated in every way that matters, and demanding an
exact zero would miss it because of one stray retry."""

SILENT_WINDOWS_FOR_VERDICT = 3
"""How many consecutive windows must be silent before the verdict flips.

The design says "persistently ~0", and persistence is the whole
discrimination: one window with no transfers is a scrape that landed
between requests, three in a row while the router keeps dispatching is a
pair that stopped talking."""

DEGRADED_RATE_FRACTION = 0.5
"""Below this share of the group's own baseline is a degradation.

Wide on purpose. The rate is per *window* and the mix of request sizes
moves it, so a tight bound would report the workload changing. A halving
is outside that noise, and the failure this catches (RDMA silently falling
back to TCP) is an order of magnitude, not 20%."""

MIN_TRANSFERS_FOR_RATE = 5
"""Fewer transfers than this in a window is not a rate, it is a sample.

A single measured transfer on Ascend took 42ms where the same pair took 5ms
for a short prompt — the fixed handshake cost dominates a small sample, so a
baseline taken from one would be meaningless in both directions."""


class PDEffectivenessEnum(str, Enum):
    """Whether KV is actually crossing between the two roles."""

    EFFECTIVE = "effective"
    """Transfers are happening in proportion to the requests routed."""

    SUSPECT = "suspect"
    """Requests routed, no transfers, but not yet for long enough. Kept
    distinct from `aggregated` so a transient gap does not raise an alarm
    and does not get silently rounded to "fine" either."""

    AGGREGATED = "aggregated"
    """🔴 Persistently no KV transfer while the router keeps dispatching:
    the deployment has silently degraded to aggregated serving. It still
    answers, at the cost of every prefill being recomputed on decode."""

    IDLE = "idle"
    """No requests in the window. Undecidable, and explicitly not a
    degradation — a model nobody is calling transfers nothing."""

    UNMEASURABLE = "unmeasurable"
    """The inputs for the ratio do not exist. Never conflated with a bad
    ratio: "we cannot tell" and "it is broken" call for different
    reactions."""


class TransferHealthEnum(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BASELINE_PENDING = "baseline_pending"
    """The first qualifying window becomes the baseline, so there is
    nothing to compare it to yet."""
    UNMEASURABLE = "unmeasurable"


class DenominatorSourceEnum(str, Enum):
    """Where the ratio's denominator came from — surfaced rather than
    inferred, because the fallbacks are much weaker than the real thing and
    a reader has to be able to tell which one produced the number."""

    ROUTER_METRICS = "router_metrics"
    """Per-worker counters — the good case, because a low ratio points at
    one decode instead of at "the group"."""
    ROUTER_TOTAL = "router_total"
    """The router's route-aggregated total, used when no member matched its
    per-worker label. Still answers "did anything get routed", which is what
    a ratio of zero has to be read against, but localises nothing."""
    NO_ROUTER_METRICS = "no_router_metrics"
    """The mode declares `capabilities.metrics: false` (vllm-ascend's proxy
    example serves no /metrics at all) or declares no counter names."""
    NONE = "none"


class RateBasisEnum(str, Enum):
    """The unit the transfer rate is in. Carried with every rate because a
    baseline in one basis is meaningless against a rate in the other, and
    the basis depends on what the connector exports."""

    BYTES_PER_SECOND = "bytes_per_second"
    TRANSFERS_PER_SECOND = "transfers_per_second"
    NONE = "none"


class EngineReading(BaseModel):
    """One engine member's cumulative counters, as scraped."""

    instance_id: int
    instance_name: str = ""
    role: Optional[str] = None
    address: Optional[str] = None
    """host:port, used to match this member to the router's per-worker
    label."""

    transfers: Optional[float] = None
    seconds: Optional[float] = None
    transferred_bytes: Optional[float] = None
    failed_transfers: Optional[float] = None
    kv_expired: Optional[float] = None

    # None throughout means "the sample was not in the exposition", which is
    # a different fact from 0.0 and has to stay different: absent is an
    # engine that does not export it, zero is an engine that exports it and
    # counted nothing.


class RouterRequests(BaseModel):
    """What the router says it dispatched, at both granularities."""

    per_worker: Dict[str, float] = {}
    total: Optional[float] = None

    @classmethod
    def of(cls, value) -> Optional["RouterRequests"]:
        """Accepts a bare per-worker mapping as well, so a caller that only
        has that half does not have to construct the wrapper."""
        if value is None:
            return None
        if isinstance(value, RouterRequests):
            return value
        return cls(per_worker=dict(value))


class MemberRatio(BaseModel):
    """One decode member's own ratio — which decode stopped pulling."""

    instance_id: int
    instance_name: str
    transfers: float
    requests: Optional[float] = None
    ratio: Optional[float] = None


class PDWindowState(BaseModel):
    """What one group carries between observations.

    In memory only, and keyed by group id. A group is a generation, so a
    baseline cannot outlive one; persisting it would mean a baseline taken
    on the hardware a deployment used to sit on being compared against the
    hardware it sits on now, which is a false alarm with a database row
    behind it.
    """

    totals: Dict[str, float] = {}
    """Last cumulative value per (instance id, counter). Deltas, not
    totals, are what a verdict is computed on: a group that ran well for a
    day and stopped transferring an hour ago has an excellent lifetime
    ratio."""

    silent_windows: int = 0
    baseline_rate: Optional[float] = None
    baseline_basis: RateBasisEnum = RateBasisEnum.NONE


class PDGroupObservation(BaseModel):
    """One window's verdict on one group."""

    model_id: int
    group_id: Optional[str] = None
    observed_at: datetime

    effectiveness: PDEffectivenessEnum
    detail: str = ""
    ratio: Optional[float] = None
    transfers: float = 0.0
    requests: Optional[float] = None
    denominator: DenominatorSourceEnum = DenominatorSourceEnum.NONE
    per_member: List[MemberRatio] = []

    prefill_transfers: Optional[float] = None
    """Recorded, never used as the numerator. It exists so that a reader who
    wonders why the healthy-looking group reports 0.0 on prefill can see
    that 0.0 is the expected value for a pull-based connector."""

    transfer_health: TransferHealthEnum = TransferHealthEnum.UNMEASURABLE
    rate: Optional[float] = None
    baseline_rate: Optional[float] = None
    rate_basis: RateBasisEnum = RateBasisEnum.NONE

    failed_transfers: Optional[float] = None
    kv_expired: Optional[float] = None

    degradations: List[str] = []
    messages: List[str] = []

    @property
    def message(self) -> Optional[str]:
        return "; ".join(self.messages) or None


def sum_samples(
    families: Optional[Mapping[str, Any]],
    name: Optional[str],
    label_filter: Optional[Tuple[str, str]] = None,
) -> Optional[float]:
    """Total of every sample named `name` in a parsed exposition.

    Returns None when no such sample exists anywhere, and that distinction
    carries the module: absent means the engine does not export the counter,
    so nothing can be concluded, while 0.0 means it exports it and counted
    nothing, which is the failure. Collapsing the two would make every
    unsupported engine look like a broken one.

    Matched on the *sample* name rather than the family name because the
    interesting values are suffixed ones — a histogram's `_count`, a
    counter's `_total` — and the family is named without the suffix.
    """
    if not families or not name:
        return None
    total: Optional[float] = None
    for family in families.values():
        for sample in getattr(family, "samples", []):
            if sample.name != name:
                continue
            if label_filter is not None:
                key, value = label_filter
                if sample.labels.get(key) != value:
                    continue
            total = (total or 0.0) + sample.value
    return total


def read_engine(
    families: Optional[Mapping[str, Any]],
    metrics: Optional[PDTransferMetrics],
    expired_metric: Optional[str] = None,
    *,
    instance_id: int,
    instance_name: str = "",
    role: Optional[str] = None,
    address: Optional[str] = None,
) -> EngineReading:
    """One engine member's counters, per the connector's declaration."""
    reading = EngineReading(
        instance_id=instance_id,
        instance_name=instance_name,
        role=role,
        address=address,
    )
    if metrics is not None:
        reading.transfers = sum_samples(families, metrics.xfer_count)
        reading.seconds = sum_samples(families, metrics.xfer_seconds)
        reading.transferred_bytes = sum_samples(families, metrics.xfer_bytes)
        reading.failed_transfers = sum_samples(families, metrics.failed_transfers)
    # Scraped from both roles rather than from the side that "should" hold
    # the lease: whichever side does not export it simply has no sample, so
    # summing both is right either way and does not encode a guess.
    reading.kv_expired = sum_samples(families, expired_metric)
    return reading


def read_router_requests(
    families: Optional[Mapping[str, Any]],
    mode: Optional[PDMode],
) -> Optional[RouterRequests]:
    """Request counts from the router, per peer and in total.

    None means no denominator is available, which is a supported state and
    not an error: `capabilities.metrics: false` is a declaration that the
    router serves no exposition, and polling one that does not exist filled
    vllm-ascend's proxy log with 404s about once a second.
    """
    if mode is None or mode.router is None:
        return None
    if not mode.router.capabilities.metrics:
        return None
    spec = mode.router.request_metrics
    if not spec.available or not families:
        return None

    counts: Dict[str, float] = {}
    found = False
    for family in families.values():
        for sample in getattr(family, "samples", []):
            if sample.name not in (spec.prefill_requests, spec.decode_requests):
                continue
            found = True
            worker = sample.labels.get(spec.worker_label) or ""
            counts[worker] = counts.get(worker, 0.0) + sample.value

    # Summed over its route label: one series per route, and the denominator
    # is the group's traffic rather than any one route's.
    total = sum_samples(families, spec.total_requests)
    if not found and total is None:
        return None
    return RouterRequests(per_worker=counts, total=total)


def requests_for_address(
    counts: Optional[Mapping[str, float]], address: Optional[str]
) -> Optional[float]:
    """The router's count for one member.

    Matched on a host:port substring rather than on equality. Measured, the
    label holds the whole peer URL the router was launched with —
    ``worker="http://192.168.50.15:40005"`` — not a worker name, an id, or
    a bare address, and an exact comparison against the address GPUStack
    resolved would silently find nothing and leave the ratio without a
    denominator.
    """
    if counts is None or not address:
        return None
    for worker, value in counts.items():
        if address in worker:
            return value
    return None


def _delta(state: PDWindowState, key: str, current: Optional[float]) -> Optional[float]:
    """Increment since the previous window, remembering `current`.

    A counter that went *down* was reset by a restarted process, and the
    increment since the restart is the current value itself — the standard
    reading, and the conservative one here, since the alternative (dropping
    the window) would hide a group that restarts faster than it is polled.
    The first window has no predecessor and therefore no delta at all;
    treating the initial cumulative value as one window's worth of traffic
    would make a long-running group look like a burst.
    """
    if current is None:
        state.totals.pop(key, None)
        return None
    previous = state.totals.get(key)
    state.totals[key] = current
    if previous is None:
        return None
    if current < previous:
        return current
    return current - previous


def _sum_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _rate(
    count_delta: Optional[float],
    seconds_delta: Optional[float],
    bytes_delta: Optional[float],
) -> Tuple[Optional[float], RateBasisEnum]:
    """Throughput for this window, and the unit it is in.

    Bytes per second when the connector exports a byte counter; otherwise
    transfers per second of transfer time. The fallback is not a bandwidth
    and is not comparable to one — it is only ever compared to a baseline
    taken from the same group in the same basis, which is the whole reason
    the basis travels with the number.

    Divided by transfer *time* rather than by wall clock deliberately: a
    window in which the group was mostly idle would otherwise read as a
    collapse in throughput.
    """
    if not seconds_delta or seconds_delta <= 0:
        return None, RateBasisEnum.NONE
    if bytes_delta:
        return bytes_delta / seconds_delta, RateBasisEnum.BYTES_PER_SECOND
    if count_delta and count_delta >= MIN_TRANSFERS_FOR_RATE:
        return count_delta / seconds_delta, RateBasisEnum.TRANSFERS_PER_SECOND
    return None, RateBasisEnum.NONE


def _judge_effectiveness(
    observation: PDGroupObservation,
    state: PDWindowState,
    *,
    transfers: Optional[float],
    requests: Optional[float],
    prefill_transfers: Optional[float],
    counters_present: bool,
) -> None:
    """Fold the window's deltas into an effectiveness verdict."""
    if transfers is None:
        observation.effectiveness = PDEffectivenessEnum.UNMEASURABLE
        observation.detail = (
            # Two different absences, kept apart because one clears itself in
            # a minute and the other never will.
            "the first observation of a group has nothing to subtract from; "
            "a verdict needs two"
            if counters_present
            else "the decode side exports no KV transfer counter, so "
            "disaggregation cannot be confirmed from metrics for this mode"
        )
        return

    observation.transfers = transfers
    if prefill_transfers and not transfers:
        # Guard, not a measured case: every connector shipped today is
        # pull-based, so this can only mean the declared read side is wrong
        # for a connector that pushes. KV is demonstrably moving either way,
        # which is the one thing the verdict is about.
        observation.effectiveness = PDEffectivenessEnum.EFFECTIVE
        observation.detail = (
            "KV transfers are counted on prefill, not decode: this connector "
            "pushes rather than pulls and its read_from_role declaration "
            "should say so"
        )
        state.silent_windows = 0
        return

    if requests is None:
        # No denominator. Reported, never faked: an absolute count proves
        # transfers are happening, but its absence proves nothing at all,
        # because an idle group and a broken one both transfer zero.
        if transfers > 0:
            observation.effectiveness = PDEffectivenessEnum.EFFECTIVE
            observation.detail = (
                f"{transfers:.0f} KV transfers on the decode side; no router "
                "request metrics for this mode, so this is an absolute count "
                "and not a ratio"
            )
        else:
            observation.effectiveness = PDEffectivenessEnum.UNMEASURABLE
            observation.detail = (
                "no KV transfers and no router request metrics for this mode: "
                "an idle group and one degraded to aggregated serving are "
                "indistinguishable without a denominator"
            )
        return

    observation.requests = requests
    if requests <= 0:
        observation.effectiveness = PDEffectivenessEnum.IDLE
        observation.detail = "no requests were routed in this window"
        return

    ratio = transfers / requests
    observation.ratio = ratio
    if ratio > AGGREGATED_RATIO:
        state.silent_windows = 0
        observation.effectiveness = PDEffectivenessEnum.EFFECTIVE
        observation.detail = f"{transfers:.0f} KV transfers for {requests:.0f} requests"
        return

    state.silent_windows += 1
    if state.silent_windows < SILENT_WINDOWS_FOR_VERDICT:
        observation.effectiveness = PDEffectivenessEnum.SUSPECT
        observation.detail = (
            f"no KV transfer for {requests:.0f} routed requests "
            f"({state.silent_windows} of {SILENT_WINDOWS_FOR_VERDICT} windows)"
        )
        return

    observation.effectiveness = PDEffectivenessEnum.AGGREGATED
    observation.detail = (
        f"no KV transfer across {state.silent_windows} consecutive windows "
        f"while the router kept dispatching ({requests:.0f} requests in the "
        "last one): the deployment is serving aggregated, recomputing every "
        "prefill on decode"
    )
    observation.degradations.append(DegradationReasonEnum.PD_INEFFECTIVE.value)
    observation.messages.append(
        "PD has degraded to aggregated serving: no KV transfer while requests "
        "are being routed"
    )


def _judge_transfer_health(
    observation: PDGroupObservation,
    state: PDWindowState,
    metrics: Optional[PDTransferMetrics],
    *,
    rate: Optional[float],
    basis: RateBasisEnum,
) -> None:
    """Compare this window's rate against the group's own baseline."""
    observation.rate = rate
    observation.rate_basis = basis
    observation.baseline_rate = state.baseline_rate

    if rate is None:
        observation.transfer_health = TransferHealthEnum.UNMEASURABLE
        return

    floor = metrics.min_expected_rate if metrics else None
    if floor is not None and rate < floor:
        # The backstop for the case the baseline method cannot see: a group
        # that was already degraded when its own baseline was taken. Only
        # ever a coarse "this cannot possibly be right" magnitude, never a
        # fraction of nameplate speed.
        observation.transfer_health = TransferHealthEnum.DEGRADED
        observation.degradations.append(DegradationReasonEnum.BANDWIDTH_DEGRADED.value)
        observation.messages.append(
            f"KV transfer rate {rate:.3g} is below the declared floor {floor:.3g} "
            f"({basis.value})"
        )
        return

    if state.baseline_rate is None or state.baseline_basis != basis:
        # First qualifying window, or the basis changed under us (a connector
        # swap within a generation). Either way there is nothing to compare
        # against, so this window *becomes* the reference.
        state.baseline_rate = rate
        state.baseline_basis = basis
        observation.baseline_rate = rate
        observation.transfer_health = TransferHealthEnum.BASELINE_PENDING
        return

    if rate < state.baseline_rate * DEGRADED_RATE_FRACTION:
        observation.transfer_health = TransferHealthEnum.DEGRADED
        observation.degradations.append(DegradationReasonEnum.BANDWIDTH_DEGRADED.value)
        share = rate / state.baseline_rate * 100
        observation.messages.append(
            f"KV transfer rate {rate:.3g} {basis.value} is {share:.0f}% of this "
            f"group's baseline {state.baseline_rate:.3g}"
        )
        return

    # The baseline is deliberately not raised to match a better window.
    # Ratcheting it up would make the best window ever observed the standard
    # every later one is held to, and report the workload changing as the
    # transport degrading.
    observation.transfer_health = TransferHealthEnum.HEALTHY


def observe_group(
    *,
    model_id: int,
    group_id: Optional[str],
    decode_readings: Sequence[EngineReading],
    prefill_readings: Sequence[EngineReading] = (),
    router_requests=None,
    transfer_metrics: Optional[PDTransferMetrics] = None,
    state: Optional[PDWindowState] = None,
    now: Optional[datetime] = None,
) -> Tuple[PDGroupObservation, PDWindowState]:
    """One window's verdict on one group, plus the state to carry forward.

    Pure: every input is a value and the carried state comes in and out, so
    a verdict that depends on three consecutive windows is three calls in a
    test rather than three minutes of waiting.

    `decode_readings` is the numerator's only source — see this module's
    docstring for why reading the other side inverts the answer.
    """
    state = state.model_copy(deep=True) if state else PDWindowState()
    router = RouterRequests.of(router_requests)
    observation = PDGroupObservation(
        model_id=model_id,
        group_id=group_id,
        observed_at=now or datetime.now(timezone.utc),
        effectiveness=PDEffectivenessEnum.UNMEASURABLE,
    )

    transfer_deltas: List[Optional[float]] = []
    seconds_deltas: List[Optional[float]] = []
    bytes_deltas: List[Optional[float]] = []
    matched_per_worker = False
    for reading in decode_readings:
        prefix = f"{reading.instance_id}"
        transfer_delta = _delta(state, f"{prefix}:transfers", reading.transfers)
        transfer_deltas.append(transfer_delta)
        seconds_deltas.append(_delta(state, f"{prefix}:seconds", reading.seconds))
        bytes_deltas.append(_delta(state, f"{prefix}:bytes", reading.transferred_bytes))

        member_requests = requests_for_address(
            router.per_worker if router else None, reading.address
        )
        if member_requests is not None:
            matched_per_worker = True
        request_delta = _delta(state, f"{prefix}:requests", member_requests)
        observation.per_member.append(
            MemberRatio(
                instance_id=reading.instance_id,
                instance_name=reading.instance_name,
                transfers=transfer_delta or 0.0,
                requests=request_delta,
                ratio=(
                    (transfer_delta or 0.0) / request_delta if request_delta else None
                ),
            )
        )

    prefill_deltas = [
        _delta(state, f"{reading.instance_id}:transfers", reading.transfers)
        for reading in prefill_readings
    ]

    all_readings = list(decode_readings) + list(prefill_readings)
    observation.failed_transfers = _sum_optional(
        [reading.failed_transfers for reading in all_readings]
    )
    observation.kv_expired = _sum_optional(
        [reading.kv_expired for reading in all_readings]
    )
    observation.prefill_transfers = _sum_optional(prefill_deltas)

    requests = _sum_optional([member.requests for member in observation.per_member])
    if matched_per_worker:
        observation.denominator = DenominatorSourceEnum.ROUTER_METRICS
    elif router is not None and router.total is not None:
        # No member matched its per-worker label — a peer URL that does not
        # contain the address we resolved, or a router that only counts by
        # route. The group-level total still answers whether anything was
        # routed at all, which is the only question a ratio of zero needs,
        # while localising nothing.
        requests = _delta(state, "router:total", router.total)
        observation.denominator = DenominatorSourceEnum.ROUTER_TOTAL
    else:
        observation.denominator = DenominatorSourceEnum.NO_ROUTER_METRICS

    _judge_effectiveness(
        observation,
        state,
        transfers=_sum_optional(transfer_deltas),
        requests=requests,
        prefill_transfers=observation.prefill_transfers,
        counters_present=any(
            reading.transfers is not None for reading in decode_readings
        ),
    )

    rate, basis = _rate(
        _sum_optional(transfer_deltas),
        _sum_optional(seconds_deltas),
        _sum_optional(bytes_deltas),
    )
    _judge_transfer_health(observation, state, transfer_metrics, rate=rate, basis=basis)
    return observation, state


# ---------------------------------------------------------------------------
# The live loop.
# ---------------------------------------------------------------------------

_observations: Dict[int, PDGroupObservation] = {}
"""Latest verdict per model, in process.

Not a table. The verdict is derived from counters that are themselves
re-derivable on the next tick, and it belongs to a generation that does not
survive a redeploy — persisting it would buy nothing but the chance of
serving a stale one after a restart. What *is* persisted is the conclusion:
`sync_model_status` copies the degradation onto the Model row, which is the
part a user acts on.
"""


def get_pd_observation(model_id: Optional[int]) -> Optional[PDGroupObservation]:
    if model_id is None:
        return None
    return _observations.get(model_id)


def record_pd_observation(observation: PDGroupObservation) -> None:
    _observations[observation.model_id] = observation


def forget_pd_observation(model_id: int) -> None:
    _observations.pop(model_id, None)


def pd_observations() -> Dict[int, PDGroupObservation]:
    return dict(_observations)


def _router_metrics_address(router, mode: PDMode) -> Optional[str]:
    """Where the router serves its Prometheus exposition.

    Not its API port. vllm-router runs two listeners, and the metrics one is
    on a band GPUStack allocates — upstream's default (29000) is fixed, so
    two routers on a host collide. Scraping the API port returns 404, and a
    404 is indistinguishable from "this router has no metrics", so the
    ratio would lose its denominator without anything ever failing.

    Measured alongside that: the exposition also has to be *bound* to the
    worker's address. With `--prometheus-host` unset it listens on loopback
    only, and this scrape runs on the server rather than on the worker, so
    the catalog passes {{worker_ip}} for the same reason it passes it to
    VLLM_NIXL_SIDE_CHANNEL_HOST.
    """
    band_name = mode.router.request_metrics.port_band if mode.router else None
    if not band_name:
        return f"{router.worker_ip}:{router.port}"
    band = (getattr(router, "named_ports", None) or {}).get(band_name)
    base = getattr(band, "base", None)
    if not base:
        # The band was declared but never allocated on this instance. Nothing
        # to scrape, and guessing the API port would just collect 404s.
        logger.debug(
            f"Router {router.name} has no '{band_name}' port band allocated; "
            "the PD ratio has no denominator this window"
        )
        return None
    return f"{router.worker_ip}:{base}"


class PDObserver:
    """Scrapes each disaggregated group once a window and records a verdict.

    Server-side rather than worker-side because the join is the point: the
    numerator sits on decode's exposition, the denominator on the router's,
    and those are routinely on different workers. A worker can only ever see
    its own half.
    """

    def __init__(self, interval: int = 60):
        self._interval = interval
        self._state: Dict[Tuple[int, Optional[str]], PDWindowState] = {}

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.observe_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Failed to observe PD groups: {e}")

    async def observe_once(self):
        from gpustack.schemas.models import Model
        from gpustack.server.db import async_session

        async with async_session() as session:
            models = await Model.all(session)
        live = set()
        for model in models:
            if model.deleted_at is not None or model.disaggregation is None:
                continue
            live.add(model.id)
            try:
                await self._observe_model(model)
            except Exception as e:
                logger.error(f"Failed to observe PD group of {model.name}: {e}")
        # A model that stopped being disaggregated, or stopped existing,
        # must not leave its last verdict behind to be republished forever.
        for model_id in list(_observations):
            if model_id not in live:
                forget_pd_observation(model_id)

    async def _observe_model(self, model) -> Optional[PDGroupObservation]:
        from gpustack.schemas.models import (
            ModelInstance,
            ModelInstanceStateEnum,
            RoleNameEnum,
        )
        from gpustack.server.db import async_session
        from gpustack.server.pd_mode_catalog import get_pd_mode

        mode_name = getattr(model.disaggregation.mode, "value", None) or str(
            model.disaggregation.mode
        )
        mode = get_pd_mode(mode_name)

        async with async_session() as session:
            instances = await ModelInstance.all_by_field(session, "model_id", model.id)

        running = [
            instance
            for instance in instances
            if instance.state == ModelInstanceStateEnum.RUNNING
            and instance.worker_ip
            and instance.port
        ]
        if not running:
            forget_pd_observation(model.id)
            return None

        # One generation only. A member of the previous group still shutting
        # down carries counters from a pairing that no longer exists, and
        # mixing them in would attribute the old group's traffic to the new
        # one's ratio.
        group_id = next(
            (i.group_id for i in running if i.role == RoleNameEnum.DECODE.value),
            None,
        )
        members = [i for i in running if i.group_id == group_id]

        # Engines expose their metrics on their own HTTP port; the router
        # does not (see `_router_metrics_address`), so it is keyed separately
        # rather than by the address the rest of the members use.
        endpoints = {
            f"{i.worker_ip}:{i.port}": i
            for i in members
            if i.role != RoleNameEnum.ROUTER.value
        }
        router = next((i for i in members if i.role == RoleNameEnum.ROUTER.value), None)
        router_address = None
        if (
            router is not None
            and mode
            and mode.router
            and mode.router.capabilities.metrics
        ):
            router_address = _router_metrics_address(router, mode)
        # Otherwise the router is declared to serve no exposition and is not
        # polled at all: measured on vllm-ascend's proxy, polling an endpoint
        # that is not there produced a ~1/s 404 storm in its log and a
        # permanent false alarm.

        scraped = await self._scrape(
            list(endpoints) + ([router_address] if router_address else [])
        )

        expired_metric = (
            mode.kv_lease.expired_metric if mode and mode.kv_lease else None
        )
        transfer_metrics = mode.transfer_metrics if mode else None
        decode_readings: List[EngineReading] = []
        prefill_readings: List[EngineReading] = []
        for address, instance in endpoints.items():
            reading = read_engine(
                scraped.get(address),
                transfer_metrics,
                expired_metric,
                instance_id=instance.id,
                instance_name=instance.name,
                role=instance.role,
                address=address,
            )
            side = (
                transfer_metrics.read_from_role
                if transfer_metrics
                else RoleNameEnum.DECODE.value
            )
            if instance.role == side:
                decode_readings.append(reading)
            else:
                prefill_readings.append(reading)

        router_requests = (
            read_router_requests(scraped.get(router_address), mode)
            if router_address
            else None
        )

        key = (model.id, group_id)
        observation, state = observe_group(
            model_id=model.id,
            group_id=group_id,
            decode_readings=decode_readings,
            prefill_readings=prefill_readings,
            router_requests=router_requests,
            transfer_metrics=transfer_metrics,
            state=self._state.get(key),
        )
        # A redeploy makes a new group, and the new group's baseline is its
        # own. Dropping this model's other generations here is what stops a
        # baseline measured on hardware the deployment has since left from
        # being the standard the new one is judged against.
        self._state = {
            k: v for k, v in self._state.items() if k[0] != model.id or k == key
        }
        self._state[key] = state
        record_pd_observation(observation)
        logger.debug(
            f"PD group of {model.name}: {observation.effectiveness.value} "
            f"({observation.detail}), transfer {observation.transfer_health.value}"
        )
        return observation

    async def _scrape(self, endpoints: List[str]) -> Dict[str, Any]:
        """Fetch and parse every endpoint's exposition.

        Reuses the worker's metrics client rather than growing a second one:
        the retry, the backoff and the Prometheus parsing are already there,
        and one scraper means one place where a timeout is tuned. It is
        blocking, so it runs off the event loop.
        """
        from gpustack.worker.runtime_metrics_client import (
            Client as RuntimeMetricsClient,
        )
        from gpustack.worker.runtime_metrics_client import (
            Config as RuntimeMetricsClientConfig,
        )

        if not endpoints:
            return {}
        client = RuntimeMetricsClient(
            RuntimeMetricsClientConfig(timeout=5, max_retries=1)
        )
        return await asyncio.to_thread(client.fetch_metrics_from_endpoints, endpoints)
