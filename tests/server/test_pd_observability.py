"""Is disaggregation happening, and is it still as fast as it was.

Two measured facts drive nearly every test here, and both are cases where the
obvious implementation returns a confidently wrong answer.

NIXL is pull-based, so a completed KV transfer is counted on **decode**. On a
working 1P1D, prefill's ``nixl_xfer_time_seconds_count`` stayed 0.0 for the
whole run while decode's went to 1.0 — read the wrong side and a healthy pair
reports the exact failure the metric exists to detect.

And "effective bandwidth versus the link's nameplate speed" is not a criterion:
2.5GbE measured 94% of line rate, 910B2 RoCE measured 9%. No single threshold
can be right on both, so the comparison is against the group's own first
observation instead.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.principals import Principal
from gpustack.schemas.models import (
    DegradationReasonEnum,
    DisaggregationSpec,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    PDModeEnum,
    PortBand,
    RoleSpec,
    SourceEnum,
)
from gpustack.schemas.pd_modes import PDTransferMetrics
from gpustack.server.controllers import sync_model_status
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.server.pd_observability import (
    DEGRADED_RATE_FRACTION,
    SILENT_WINDOWS_FOR_VERDICT,
    DenominatorSourceEnum,
    EngineReading,
    PDEffectivenessEnum,
    PDGroupObservation,
    PDObserver,
    RateBasisEnum,
    TransferHealthEnum,
    forget_pd_observation,
    observe_group,
    read_engine,
    read_router_requests,
    record_pd_observation,
    requests_for_address,
    sum_samples,
)

NIXL = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
NIXL_METRICS = NIXL.transfer_metrics


def parse(text: str):
    """An exposition, in the shape the metrics client hands over."""
    return {family.name: family for family in text_string_to_metric_families(text)}


def engine_exposition(count: float, seconds: float = 0.0, failed: float = 0.0) -> str:
    """A vLLM NIXL engine's transfer counters. ``_count`` and ``_sum`` are
    samples of a histogram, ``_total`` of a counter, which is why the catalog
    declares the suffixed names."""
    return f"""# HELP vllm:nixl_xfer_time_seconds Transfer time
# TYPE vllm:nixl_xfer_time_seconds histogram
vllm:nixl_xfer_time_seconds_bucket{{le="+Inf"}} {count}
vllm:nixl_xfer_time_seconds_count {count}
vllm:nixl_xfer_time_seconds_sum {seconds}
# HELP vllm:nixl_num_failed_transfers Failed transfers
# TYPE vllm:nixl_num_failed_transfers counter
vllm:nixl_num_failed_transfers_total {failed}
"""


def router_exposition(*pairs) -> str:
    """vllm-router's per-worker request counters."""
    lines = [
        "# HELP vllm_router_pd_decode_requests_total Decode requests",
        "# TYPE vllm_router_pd_decode_requests_total counter",
    ]
    for worker, value in pairs:
        lines.append(
            f'vllm_router_pd_decode_requests_total{{worker="{worker}"}} {value}'
        )
    return "\n".join(lines) + "\n"


def decode(instance_id=1, transfers=0.0, seconds=0.0, address="10.0.0.2:8000", **kw):
    return EngineReading(
        instance_id=instance_id,
        instance_name=f"d{instance_id}",
        role="decode",
        address=address,
        transfers=transfers,
        seconds=seconds,
        **kw,
    )


def prefill(instance_id=9, transfers=0.0, **kw):
    return EngineReading(
        instance_id=instance_id,
        instance_name=f"p{instance_id}",
        role="prefill",
        address="10.0.0.1:8000",
        transfers=transfers,
        **kw,
    )


def run(windows, *, prefills=None, state=None, requests=None, metrics=NIXL_METRICS):
    """Feed successive windows through the fold, carrying state.

    A verdict that depends on three consecutive observations is three calls
    here rather than three minutes of waiting, which is the whole reason the
    fold takes its state as a value.
    """
    observation = None
    for index, readings in enumerate(windows):
        observation, state = observe_group(
            model_id=1,
            group_id="g1",
            decode_readings=readings,
            prefill_readings=(prefills[index] if prefills else ()),
            router_requests=(requests[index] if requests else None),
            transfer_metrics=metrics,
            state=state,
        )
    return observation, state


# --- which side the numerator comes from ----------------------------------- #


def test_the_numerator_is_read_from_decode_not_prefill():
    """The measured 1P1D: prefill 0.0, decode 1.0, and the pair is healthy."""
    families = parse(engine_exposition(count=0.0))
    reading = read_engine(families, NIXL_METRICS, instance_id=1, role="prefill")
    assert reading.transfers == 0.0

    families = parse(engine_exposition(count=1.0))
    reading = read_engine(families, NIXL_METRICS, instance_id=2, role="decode")
    assert reading.transfers == 1.0


def test_reading_the_prefill_side_would_report_a_healthy_pair_as_broken():
    """Pinned as a contrast, not as behaviour: the same run judged from the
    wrong side inverts the answer, which is why `read_from_role` is a
    declaration rather than something a reader has to remember."""
    requests = [{"10.0.0.2:8000": 10.0}, {"10.0.0.2:8000": 20.0}]

    def read(side, counts):
        """Two scrapes of one member, as the observer takes them."""
        return [
            [
                read_engine(
                    parse(engine_exposition(count=count)),
                    NIXL_METRICS,
                    instance_id=1,
                    role=side,
                    address="10.0.0.2:8000",
                )
            ]
            for count in counts
        ]

    # The same 1P1D run, scraped from both sides: decode 5 -> 15, prefill
    # 0.0 -> 0.0.
    right, _ = run(read("decode", [5.0, 15.0]), requests=requests)
    assert right.effectiveness == PDEffectivenessEnum.EFFECTIVE

    wrong, _ = run(read("prefill", [0.0, 0.0]), requests=requests)
    assert wrong.ratio == 0.0
    assert wrong.effectiveness == PDEffectivenessEnum.SUSPECT


def test_the_catalog_declares_decode_as_the_read_side():
    assert NIXL_METRICS.read_from_role == "decode"


def test_transfers_counted_on_prefill_are_still_transfers():
    """Guard for a push-based connector: KV is demonstrably moving, so the
    verdict must not be 'aggregated' just because the declared side is
    wrong."""
    observation, _ = run(
        [[decode(transfers=0.0)], [decode(transfers=0.0)]],
        prefills=[[prefill(transfers=0.0)], [prefill(transfers=7.0)]],
        requests=[{"10.0.0.2:8000": 1.0}, {"10.0.0.2:8000": 5.0}],
    )
    assert observation.effectiveness == PDEffectivenessEnum.EFFECTIVE
    assert "pushes rather than pulls" in observation.detail


# --- the ratio ------------------------------------------------------------- #


def test_a_ratio_of_zero_needs_persistence_before_it_is_a_verdict():
    """One silent window is a scrape that landed between requests."""
    windows = [[decode(transfers=0.0)] for _ in range(SILENT_WINDOWS_FOR_VERDICT + 1)]
    requests = [{"10.0.0.2:8000": float(i * 10)} for i in range(len(windows))]

    for count in range(2, len(windows) + 1):
        observation, _ = run(windows[:count], requests=requests[:count])
        silent = count - 1
        if silent < SILENT_WINDOWS_FOR_VERDICT:
            assert observation.effectiveness == PDEffectivenessEnum.SUSPECT
            assert observation.degradations == []
        else:
            assert observation.effectiveness == PDEffectivenessEnum.AGGREGATED
            assert observation.degradations == [
                DegradationReasonEnum.PD_INEFFECTIVE.value
            ]
            assert "aggregated" in observation.message


def test_one_good_window_clears_the_silence():
    windows = [
        [decode(transfers=0.0)],
        [decode(transfers=0.0)],
        [decode(transfers=0.0)],
        [decode(transfers=6.0)],
        [decode(transfers=6.0)],
    ]
    requests = [{"10.0.0.2:8000": float(i * 10)} for i in range(len(windows))]
    observation, state = run(windows, requests=requests)

    assert state.silent_windows == 1
    assert observation.effectiveness == PDEffectivenessEnum.SUSPECT


def test_the_ratio_is_computed_on_the_window_not_the_lifetime():
    """A group that ran well for a day and stopped an hour ago still has an
    excellent cumulative ratio, which is why deltas are what get judged."""
    windows = [[decode(transfers=1000.0)], [decode(transfers=1000.0)]]
    requests = [{"10.0.0.2:8000": 1000.0}, {"10.0.0.2:8000": 1100.0}]
    observation, _ = run(windows, requests=requests)

    assert observation.transfers == 0.0
    assert observation.requests == 100.0
    assert observation.ratio == 0.0


def test_no_requests_in_the_window_is_idle_not_degraded():
    windows = [[decode(transfers=3.0)], [decode(transfers=3.0)]]
    requests = [{"10.0.0.2:8000": 7.0}, {"10.0.0.2:8000": 7.0}]
    observation, _ = run(windows, requests=requests)

    assert observation.effectiveness == PDEffectivenessEnum.IDLE
    assert observation.degradations == []


def test_the_first_window_says_so_rather_than_claiming_no_transfers():
    observation, _ = run([[decode(transfers=4.0)]])
    assert observation.effectiveness == PDEffectivenessEnum.UNMEASURABLE
    assert "nothing to subtract from" in observation.detail


def test_a_restarted_counter_is_not_a_negative_delta():
    windows = [[decode(transfers=100.0)], [decode(transfers=3.0)]]
    requests = [{"10.0.0.2:8000": 10.0}, {"10.0.0.2:8000": 13.0}]
    observation, _ = run(windows, requests=requests)

    assert observation.transfers == 3.0
    assert observation.effectiveness == PDEffectivenessEnum.EFFECTIVE


def test_the_ratio_localises_to_one_member():
    """Per-worker is the point: a group-wide ratio says something is wrong,
    this says which decode stopped pulling."""
    windows = [
        [
            decode(instance_id=1, transfers=0.0, address="10.0.0.2:8000"),
            decode(instance_id=2, transfers=0.0, address="10.0.0.3:8000"),
        ],
        [
            decode(instance_id=1, transfers=10.0, address="10.0.0.2:8000"),
            decode(instance_id=2, transfers=0.0, address="10.0.0.3:8000"),
        ],
    ]
    requests = [
        {"http://10.0.0.2:8000": 0.0, "http://10.0.0.3:8000": 0.0},
        {"http://10.0.0.2:8000": 10.0, "http://10.0.0.3:8000": 10.0},
    ]
    observation, _ = run(windows, requests=requests)

    per_member = {m.instance_name: m.ratio for m in observation.per_member}
    assert per_member == {"d1": 1.0, "d2": 0.0}
    # The group as a whole still looks half-healthy, which is exactly the
    # case a single number hides.
    assert observation.ratio == 0.5


# --- the denominator, and its absence -------------------------------------- #


def test_the_denominator_comes_from_the_router_not_from_us():
    """Measured on a live 1P1D: the counters exist, they are per worker, and
    the label carries the whole peer URL."""
    counts = read_router_requests(
        parse(router_exposition(("http://192.168.50.15:40005", 1.0))), NIXL
    )
    assert counts.per_worker == {"http://192.168.50.15:40005": 1.0}
    assert counts.total is None


def test_the_route_aggregated_total_is_the_fallback_denominator():
    """`vllm_router_pd_requests_total` is aggregated by route, not by peer.
    It localises nothing, which is why it is the fallback and not the
    primary, but it still answers whether anything was routed at all."""
    exposition = (
        "# TYPE vllm_router_pd_requests_total counter\n"
        'vllm_router_pd_requests_total{route="/v1/completions"} 7.0\n'
        'vllm_router_pd_requests_total{route="/v1/chat/completions"} 3.0\n'
    )
    counts = read_router_requests(parse(exposition), NIXL)
    assert counts.per_worker == {}
    assert counts.total == 10.0


def test_a_member_whose_label_never_matched_falls_back_to_the_total():
    """The label is a URL the router was launched with; if the address
    GPUStack resolved is not in it, per-worker matching finds nothing and
    the group total is the only denominator left."""
    from gpustack.server.pd_observability import RouterRequests

    windows = [[decode(transfers=0.0)], [decode(transfers=4.0)]]
    requests = [RouterRequests(total=100.0), RouterRequests(total=110.0)]
    observation, _ = run(windows, requests=requests)

    assert observation.denominator == DenominatorSourceEnum.ROUTER_TOTAL
    assert observation.requests == 10.0
    assert observation.ratio == 0.4
    assert observation.effectiveness == PDEffectivenessEnum.EFFECTIVE


def test_a_member_matches_its_label_by_host_and_port():
    """The label holds whatever URL the router was launched with, so an exact
    comparison would silently find nothing."""
    counts = {"http://10.0.0.2:8000": 5.0}
    assert requests_for_address(counts, "10.0.0.2:8000") == 5.0
    assert requests_for_address(counts, "10.0.0.9:8000") is None


def test_a_router_declaring_no_metrics_has_no_denominator():
    """vllm-ascend's proxy example serves no /metrics at all — measured, and
    the reason the capability is declared rather than probed."""
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    assert ascend.router.capabilities.metrics is False
    assert read_router_requests(parse(router_exposition(("w", 1.0))), ascend) is None


def test_unverified_counter_names_are_left_undeclared():
    """SGLang's gateway is the same codebase as vllm-router, but its counter
    names were never measured here. Guessing them would give a denominator
    that reads zero for the wrong reason."""
    sglang = get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value)
    assert sglang.router.capabilities.metrics is True
    assert sglang.router.request_metrics.available is False
    assert read_router_requests(parse(router_exposition(("w", 1.0))), sglang) is None


def test_without_a_denominator_the_absolute_count_is_reported_as_such():
    observation, _ = run([[decode(transfers=0.0)], [decode(transfers=8.0)]])

    assert observation.effectiveness == PDEffectivenessEnum.EFFECTIVE
    assert observation.denominator == DenominatorSourceEnum.NO_ROUTER_METRICS
    assert observation.ratio is None
    assert "not a ratio" in observation.detail


def test_without_a_denominator_silence_is_unmeasurable_not_degraded():
    """An idle group and one degraded to aggregated both transfer zero. With
    no request count there is nothing to tell them apart, and pretending
    otherwise would alarm on every unused model."""
    observation, _ = run([[decode(transfers=2.0)], [decode(transfers=2.0)]])

    assert observation.effectiveness == PDEffectivenessEnum.UNMEASURABLE
    assert observation.degradations == []
    assert "indistinguishable" in observation.detail


def test_an_absent_counter_is_not_a_zero_counter():
    """The distinction the whole module rests on: an engine that exports
    nothing must not read as an engine that transferred nothing."""
    assert sum_samples(parse(engine_exposition(count=0.0)), "vllm:nope") is None
    assert (
        sum_samples(
            parse(engine_exposition(count=0.0)), "vllm:nixl_xfer_time_seconds_count"
        )
        == 0.0
    )

    mooncake = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    reading = read_engine(
        parse(engine_exposition(count=3.0)),
        mooncake.transfer_metrics,
        instance_id=1,
        role="decode",
    )
    assert reading.transfers is None

    observation, _ = run(
        [[decode(transfers=None)], [decode(transfers=None)]],
        requests=[{"10.0.0.2:8000": 1.0}, {"10.0.0.2:8000": 9.0}],
        metrics=mooncake.transfer_metrics,
    )
    assert observation.effectiveness == PDEffectivenessEnum.UNMEASURABLE
    assert "exports no KV transfer counter" in observation.detail


# --- degradation against the group's own baseline -------------------------- #


def test_the_first_qualifying_window_becomes_the_baseline():
    observation, state = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=20.0, seconds=2.0)],
        ]
    )
    assert observation.transfer_health == TransferHealthEnum.BASELINE_PENDING
    assert state.baseline_rate == 10.0
    assert observation.rate_basis == RateBasisEnum.TRANSFERS_PER_SECOND


def test_a_collapse_against_that_baseline_is_a_degradation():
    observation, _ = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=20.0, seconds=2.0)],  # baseline: 10/s
            [decode(transfers=40.0, seconds=22.0)],  # 20 transfers in 20s: 1/s
        ]
    )
    assert observation.transfer_health == TransferHealthEnum.DEGRADED
    assert observation.degradations == [DegradationReasonEnum.BANDWIDTH_DEGRADED.value]
    assert "baseline" in observation.message


def test_a_rate_just_inside_the_margin_is_not_a_degradation():
    """Wide on purpose: the rate moves with the mix of request sizes, and a
    tight bound would report the workload changing rather than the transport
    degrading."""
    inside = 20.0 * DEGRADED_RATE_FRACTION + 1
    observation, _ = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=20.0, seconds=1.0)],  # baseline: 20/s
            [decode(transfers=40.0, seconds=1.0 + 20.0 / inside)],
        ]
    )
    assert observation.transfer_health == TransferHealthEnum.HEALTHY
    assert observation.degradations == []


def test_the_baseline_does_not_ratchet_up_to_the_best_window():
    """Otherwise the luckiest window ever observed becomes the standard every
    later one is held to."""
    _, state = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=10.0, seconds=1.0)],  # baseline: 10/s
            [decode(transfers=1010.0, seconds=2.0)],  # 1000/s, a burst
        ]
    )
    assert state.baseline_rate == 10.0


def test_the_link_nameplate_is_not_a_valid_denominator():
    """The measured refutation: 2.34 Gb/s on a 2.5 Gb/s link is 94% and fine;
    17.6 Gb/s on a 200 Gb/s link is 9% and also fine. Two groups two orders of
    magnitude apart in absolute rate are both healthy against themselves, and
    any global threshold would have to call one of them broken."""
    ethernet, _ = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=100.0, seconds=1.0)],
            [decode(transfers=200.0, seconds=2.0)],
        ]
    )
    roce, _ = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=10000.0, seconds=1.0)],
            [decode(transfers=20000.0, seconds=2.0)],
        ]
    )
    assert ethernet.transfer_health == TransferHealthEnum.HEALTHY
    assert roce.transfer_health == TransferHealthEnum.HEALTHY
    assert roce.rate == 100 * ethernet.rate


def test_a_declared_floor_catches_a_group_that_was_born_degraded():
    """The baseline method's known blind spot — the baseline itself is bad —
    so the floor is the backstop. Null in the shipped catalog: a value is a
    calibration against real hardware, and inventing one reintroduces exactly
    the threshold the measurements refuted."""
    assert NIXL_METRICS.min_expected_rate is None

    calibrated = PDTransferMetrics(
        connector="nixl",
        xfer_count=NIXL_METRICS.xfer_count,
        xfer_seconds=NIXL_METRICS.xfer_seconds,
        min_expected_rate=5.0,
    )
    observation, _ = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=10.0, seconds=10.0)],  # 1/s, its first window
        ],
        metrics=calibrated,
    )
    assert observation.transfer_health == TransferHealthEnum.DEGRADED
    assert observation.degradations == [DegradationReasonEnum.BANDWIDTH_DEGRADED.value]
    assert "floor" in observation.message


def test_too_few_transfers_is_a_sample_not_a_rate():
    """One Ascend transfer took 42ms where a short prompt took 5ms on the same
    pair: the fixed handshake cost dominates a small sample."""
    observation, state = run(
        [
            [decode(transfers=0.0, seconds=0.0)],
            [decode(transfers=2.0, seconds=1.0)],
        ]
    )
    assert observation.rate is None
    assert observation.transfer_health == TransferHealthEnum.UNMEASURABLE
    assert state.baseline_rate is None


# --- the counters that are just counters ----------------------------------- #


def test_failures_and_expiries_are_summed_across_both_roles():
    """Whichever side exports the counter has the sample and the other has
    none, so summing both encodes no guess about which side that is."""
    observation, _ = run(
        [
            [decode(transfers=0.0, failed_transfers=1.0, kv_expired=2.0)],
            [decode(transfers=0.0, failed_transfers=4.0, kv_expired=2.0)],
        ],
        prefills=[
            [prefill(failed_transfers=None, kv_expired=1.0)],
            [prefill(failed_transfers=None, kv_expired=3.0)],
        ],
    )
    assert observation.failed_transfers == 4.0
    assert observation.kv_expired == 5.0


def test_the_expiry_counter_name_comes_from_the_lease_declaration():
    assert NIXL.kv_lease.expired_metric == "vllm:nixl_num_kv_expired_reqs"
    reading = read_engine(
        parse(
            "# TYPE vllm:nixl_num_kv_expired_reqs gauge\n"
            "vllm:nixl_num_kv_expired_reqs 7.0\n"
        ),
        NIXL_METRICS,
        NIXL.kv_lease.expired_metric,
        instance_id=1,
        role="decode",
    )
    assert reading.kv_expired == 7.0
    assert get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value).kv_lease.expired_metric is None


# --- the live loop --------------------------------------------------------- #


class _FakeSession:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


def _pd_model(mode: str) -> Model:
    return Model(
        id=1,
        name="m",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        roles=[
            RoleSpec(name="prefill"),
            RoleSpec(name="decode"),
            RoleSpec(name="router", cpu_only=True),
        ],
        disaggregation=DisaggregationSpec(mode=mode),
    )


def _member(id, role, port, named_ports=None) -> ModelInstance:
    return ModelInstance(
        id=id,
        name=f"m-{role}",
        model_id=1,
        model_name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        state=ModelInstanceStateEnum.RUNNING,
        role=role,
        group_id="g1",
        worker_ip="10.0.0.1",
        port=port,
        named_ports=named_ports,
    )


async def _scraped_endpoints(mode: str, router_bands=None):
    """Which endpoints one observation pass actually polls."""
    if router_bands is None:
        router_bands = {"prometheus": PortBand(base=40001)}
    members = [
        _member(1, "prefill", 8001),
        _member(2, "decode", 8002),
        _member(3, "router", 8000, named_ports=router_bands),
    ]
    observer = PDObserver()
    scrape = AsyncMock(return_value={})
    with (
        patch("gpustack.server.db.async_session", _FakeSession),
        patch(
            "gpustack.schemas.models.ModelInstance.all_by_field",
            AsyncMock(return_value=members),
        ),
        patch.object(PDObserver, "_scrape", scrape),
    ):
        await observer._observe_model(_pd_model(mode))
    return set(scrape.call_args[0][0])


@pytest.mark.asyncio
async def test_a_router_that_serves_no_metrics_is_never_polled():
    """Measured on vllm-ascend's proxy: polling an endpoint that is not there
    filled its log with 404s about once a second and reported a permanent
    false failure. Declaring the absence is what stops that."""
    assert await _scraped_endpoints(PDModeEnum.VLLM_ASCEND_MOONCAKE.value) == {
        "10.0.0.1:8001",
        "10.0.0.1:8002",
    }


@pytest.mark.asyncio
async def test_the_router_is_scraped_on_its_metrics_band_not_its_api_port():
    """vllm-router runs two listeners. Measured, the exposition answered on
    the allocated band (40001) and the API port serves no /metrics at all —
    scraping it would collect 404s, which read as "no denominator" rather
    than as a mistake."""
    assert await _scraped_endpoints(PDModeEnum.VLLM_NIXL.value) == {
        "10.0.0.1:8001",
        "10.0.0.1:8002",
        "10.0.0.1:40001",
    }


@pytest.mark.asyncio
async def test_an_unallocated_metrics_band_is_not_guessed_at():
    """Falling back to the API port here would poll a 404 forever."""
    assert await _scraped_endpoints(PDModeEnum.VLLM_NIXL.value, router_bands={}) == {
        "10.0.0.1:8001",
        "10.0.0.1:8002",
    }


# --- what reaches the Model row -------------------------------------------- #


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(Model.__table__.create)
        await conn.run_sync(ModelInstance.__table__.create)
        # The status owner resolves the model's workload namespace from
        # its owner Principal, so the round-trip needs that table too.
        await conn.run_sync(Principal.__table__.create)
    async with AsyncSession(engine) as session:
        yield session
    await engine.dispose()


def _model() -> Model:
    return Model(
        id=1,
        name="m",
        replicas=1,
        ready_replicas=0,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
    )


def _instance() -> ModelInstance:
    return ModelInstance(
        id=1,
        name="m-1",
        model_id=1,
        model_name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        state=ModelInstanceStateEnum.RUNNING,
    )


async def _sync(model, instances):
    update = AsyncMock()
    service = MagicMock(return_value=SimpleNamespace(update=update))
    with (
        patch(
            "gpustack.server.controllers.ModelInstance.all_by_field",
            AsyncMock(return_value=instances),
        ),
        # Placement drift reads the owner Principal and the Cluster; this
        # harness runs against a mock session. Covered in
        # tests/server/test_workload_namespace.py.
        patch(
            "gpustack.server.controllers.resolve_workload_namespace",
            AsyncMock(return_value=None),
        ),
        patch("gpustack.server.controllers.ModelService", service),
    ):
        return await sync_model_status(MagicMock(), model)


def _observation(**kwargs) -> PDGroupObservation:
    kwargs.setdefault("effectiveness", PDEffectivenessEnum.EFFECTIVE)
    return PDGroupObservation(
        model_id=1,
        observed_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _no_leftover_observation():
    yield
    forget_pd_observation(1)


@pytest.mark.asyncio
async def test_a_degradation_reaches_the_model_row():
    """The observation itself stays in memory — it is a time series. Only its
    conclusion is persisted, and by the one writer of the status fields."""
    record_pd_observation(
        _observation(
            effectiveness=PDEffectivenessEnum.AGGREGATED,
            degradations=[DegradationReasonEnum.PD_INEFFECTIVE.value],
            messages=["PD has degraded to aggregated serving"],
        )
    )
    model = _model()
    assert await _sync(model, [_instance()]) is True

    assert model.degradations == [DegradationReasonEnum.PD_INEFFECTIVE.value]
    assert "aggregated" in model.state_message
    # A degradation is orthogonal to servability: the group still answers.
    assert model.state == "running"


@pytest.mark.asyncio
async def test_both_degradations_coexist():
    record_pd_observation(
        _observation(
            degradations=[
                DegradationReasonEnum.PD_INEFFECTIVE.value,
                DegradationReasonEnum.BANDWIDTH_DEGRADED.value,
            ],
            messages=["no KV transfer", "rate is 20% of baseline"],
        )
    )
    model = _model()
    await _sync(model, [_instance()])

    assert model.degradations == [
        DegradationReasonEnum.PD_INEFFECTIVE.value,
        DegradationReasonEnum.BANDWIDTH_DEGRADED.value,
    ]
    assert "20% of baseline" in model.state_message


@pytest.mark.asyncio
async def test_the_marker_clears_when_the_group_recovers():
    record_pd_observation(
        _observation(
            effectiveness=PDEffectivenessEnum.AGGREGATED,
            degradations=[DegradationReasonEnum.PD_INEFFECTIVE.value],
            messages=["no KV transfer"],
        )
    )
    model = _model()
    await _sync(model, [_instance()])
    assert model.degradations == [DegradationReasonEnum.PD_INEFFECTIVE.value]

    record_pd_observation(_observation())
    await _sync(model, [_instance()])
    assert model.degradations is None
    assert model.state_message is None


@pytest.mark.asyncio
async def test_a_model_with_no_observation_is_untouched():
    model = _model()
    await _sync(model, [_instance()])
    assert model.degradations is None
    assert model.state_message is None


@pytest.mark.asyncio
async def test_the_degradation_survives_the_database(db_session):
    record_pd_observation(
        _observation(
            degradations=[DegradationReasonEnum.BANDWIDTH_DEGRADED.value],
            messages=["rate is 12% of this group's baseline"],
        )
    )
    db_session.add(_model())
    db_session.add(_instance())
    await db_session.commit()

    model = await Model.one_by_id(db_session, 1)
    assert await sync_model_status(db_session, model) is True

    reloaded = await Model.one_by_id(db_session, 1)
    assert reloaded.degradations == [DegradationReasonEnum.BANDWIDTH_DEGRADED.value]
    assert "12%" in reloaded.state_message
    # Unchanged world, nothing to write: otherwise every pass churns watchers.
    assert await sync_model_status(db_session, reloaded) is False
