import pytest

from gpustack.schemas.models import PD_MODE_BACKENDS, BackendEnum, PDModeEnum
from gpustack.schemas.pd_modes import (
    PDInjectTargetEnum,
    PDKVLeaseTargetEnum,
    PDMode,
    PDPeerStyleEnum,
    PDPortScopeEnum,
    PDRouterProtocolEnum,
)
from gpustack.server.pd_mode_catalog import (
    PDModeCatalogError,
    get_kv_lease,
    get_kv_leases,
    get_pd_mode,
    get_transfer_metrics,
    load_pd_mode_catalog,
    load_pd_modes,
    parse_pd_mode_catalog,
)


def _document(modes, kv_leases=None, kv_transfer_metrics=None):
    """A minimal catalog document whose mode names satisfy the enum
    assertion, so a test can isolate the assertion it is after.

    Every declared connector gets an all-null transfer-metrics entry unless
    the caller supplies one: the loader requires the two registries to cover
    the same connectors, and a test about something else should not have to
    restate that."""
    entries = {mode["name"]: mode for mode in modes}
    for name, backends in PD_MODE_BACKENDS.items():
        entries.setdefault(name, {"name": name, "backends": list(backends)})
    leases = kv_leases or []
    if kv_transfer_metrics is None:
        kv_transfer_metrics = [{"connector": lease["connector"]} for lease in leases]
    return {
        "kv_leases": leases,
        "kv_transfer_metrics": kv_transfer_metrics,
        "modes": list(entries.values()),
    }


def test_catalog_asset_loads():
    modes = load_pd_modes(reload=True)
    assert modes, "bundled pd-modes.yaml should yield at least one mode"
    # Every shipped entry parses into a typed model, not a raw dict.
    assert all(isinstance(mode, PDMode) for mode in modes)


def test_catalog_names_are_exactly_the_enum():
    """The load-time assertion's happy path: the catalog is looked up by
    mode name, so the two sets have to be equal, not merely overlapping."""
    modes = load_pd_modes()
    assert {mode.name for mode in modes} == {mode.value for mode in PDModeEnum}


def test_enum_mismatch_fails_the_load():
    """The mismatch that already happened once: `ascend-mooncake` where the
    enum says `vllm-ascend-mooncake`. It has to fail loudly, because a table
    miss injects nothing and the deployment still comes up."""
    document = _document([])
    for entry in document["modes"]:
        if entry["name"] == PDModeEnum.VLLM_ASCEND_MOONCAKE.value:
            entry["name"] = "ascend-mooncake"
    with pytest.raises(PDModeCatalogError) as excinfo:
        parse_pd_mode_catalog(document)
    message = str(excinfo.value)
    assert "PDModeEnum" in message
    assert "vllm-ascend-mooncake" in message
    assert "ascend-mooncake" in message


def test_missing_and_extra_names_both_fail_the_load():
    with pytest.raises(PDModeCatalogError):
        parse_pd_mode_catalog({"modes": [{"name": PDModeEnum.CUSTOM.value}]})
    with pytest.raises(PDModeCatalogError):
        parse_pd_mode_catalog(
            _document([{"name": "vllm-moriio", "backends": [BackendEnum.VLLM.value]}])
        )


def test_catalog_backends_agree_with_the_validation_table():
    """PD_MODE_BACKENDS exists only so request validation need not read the
    catalog. The catalog is authoritative; this is the check that keeps the
    copy honest."""
    for mode in load_pd_modes():
        assert sorted(mode.backends) == sorted(PD_MODE_BACKENDS[mode.name])
    # `custom` injects nothing, so it constrains nothing.
    assert get_pd_mode(PDModeEnum.CUSTOM.value).backends == []


def test_backends_mismatch_fails_the_load():
    document = _document(
        [{"name": PDModeEnum.VLLM_NIXL.value, "backends": [BackendEnum.SGLANG.value]}]
    )
    with pytest.raises(PDModeCatalogError) as excinfo:
        parse_pd_mode_catalog(document)
    message = str(excinfo.value)
    assert "PD_MODE_BACKENDS" in message
    assert "vllm-nixl" in message


def test_backends_use_the_backend_enum_spelling():
    """The table is keyed by BackendEnum values and a role's `backend` is
    compared against these, so "vllm" instead of "vLLM" would silently
    never match."""
    known = {backend.value for backend in BackendEnum}
    for mode in load_pd_modes():
        assert set(mode.backends) <= known


# ---------------------------------------------------------------------------
# Capability 1: where injected content lands differs per engine.
# ---------------------------------------------------------------------------


def test_injection_target_is_per_engine_not_uniform():
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    sglang = get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value)

    # NIXL: the side-channel port is an env var, and the host is a bind
    # address the engine defaults to localhost.
    nixl_prefill = nixl.role("prefill")
    assert nixl_prefill.ports[0].name == "kv_side_channel"
    assert nixl_prefill.ports[0].inject_to == PDInjectTargetEnum.ENV
    assert nixl_prefill.env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "{{worker_ip}}"
    assert (
        nixl_prefill.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "{{ports.kv_side_channel}}"
    )
    assert nixl_prefill.env["UCX_NET_DEVICES"] == "{{net_device}}"

    # Mooncake on Ascend: the same port is a field of the connector
    # descriptor, which renders into --kv-transfer-config. "The transfer
    # config carries no addresses" holds for NIXL only.
    ascend_prefill = ascend.role("prefill")
    assert ascend_prefill.ports[0].name == "kv_port"
    assert ascend_prefill.ports[0].inject_to == PDInjectTargetEnum.ARGS
    assert ascend_prefill.connector["kv_port"] == "{{ports.kv_port}}"
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT" not in ascend_prefill.env
    # Ascend's handshake variables have nothing in common with NIXL's.
    assert ascend_prefill.env["HCCL_IF_IP"] == "{{worker_ip}}"
    assert [key for key in ascend_prefill.env if key.endswith("_SOCKET_IFNAME")] == [
        "HCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME",
    ]

    # SGLang: a command-line flag, and no connector descriptor at all.
    sglang_prefill = sglang.role("prefill")
    assert sglang_prefill.ports[0].inject_to == PDInjectTargetEnum.ARGS
    assert "--disaggregation-bootstrap-port" in sglang_prefill.args
    assert "{{ports.bootstrap}}" in sglang_prefill.args
    assert sglang_prefill.connector == {}


def test_files_is_a_declarable_injection_target():
    """The third target, needed by connectors that read a config file and
    nothing else (Mooncake's transfer engine reads only the JSON that
    MOONCAKE_CONFIG_PATH points at). No shipped entry uses it yet, so the
    schema is what has to prove it."""
    mode = PDMode(
        name="file-fed",
        roles={
            "prefill": {
                "ports": [{"name": "kv_port", "inject_to": "files"}],
                "env": {"MOONCAKE_CONFIG_PATH": "/tmp/gpustack-pd-mooncake.json"},
                "files": {
                    "/tmp/gpustack-pd-mooncake.json": '{"port": {{ports.kv_port}}}'
                },
            }
        },
    )
    port = mode.role("prefill").ports[0]
    assert port.inject_to == PDInjectTargetEnum.FILES
    assert (
        "{{ports.kv_port}}"
        in mode.role("prefill").files["/tmp/gpustack-pd-mooncake.json"]
    )


def test_a_band_must_be_consumed_where_it_says_it_is():
    """A wrong inject_to is exactly the NIXL-vs-Mooncake mistake this
    schema exists to prevent, so it cannot sit there looking plausible."""
    with pytest.raises(ValueError, match="declares inject_to 'env'"):
        PDMode(
            name="misdeclared",
            roles={
                "prefill": {
                    "ports": [{"name": "kv_port", "inject_to": "env"}],
                    "args": ["--kv-port", "{{ports.kv_port}}"],
                }
            },
        )


# ---------------------------------------------------------------------------
# Capability 2: a band's width is the connector's rule, not a formula.
# ---------------------------------------------------------------------------


def test_port_band_count_is_declared_by_the_connector():
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)

    # Measured: kv_port is a base address and Mooncake binds one port per
    # *worker rank* -- TP8/DP1 held 41100-41107 and DP2xTP2 held 20001-20004
    # (rank 0 -> base+0, rank 1 -> base+2) -- so the width is the member's
    # card count, resolved at allocation time. Reading it as the
    # tensor-parallel size is indistinguishable on the TP8/DP1 sample and
    # under-reserves by a factor of dp on every DP member.
    for role in ("prefill", "decode"):
        band = ascend.role(role).ports[0]
        assert band.count == "{{accelerator_count}}"
        assert band.scope == PDPortScopeEnum.INSTANCE
    # NIXL's side channel is offset per DP index instead; phase one ships
    # no local DP, so the declared width is one.
    assert nixl.role("prefill").ports[0].count == 1


def test_shorthand_port_declaration_expands():
    mode = PDMode(
        name="shorthand",
        roles={
            "prefill": {
                "ports": ["kv_side_channel"],
                "env": {"P": "{{ports.kv_side_channel}}"},
            }
        },
    )
    band = mode.role("prefill").ports[0]
    assert band.name == "kv_side_channel"
    assert band.count == 1
    assert band.inject_to == PDInjectTargetEnum.ENV


def test_port_band_count_rejects_arithmetic():
    with pytest.raises(ValueError, match="neither an integer nor a single placeholder"):
        PDMode(
            name="bad-count",
            roles={
                "prefill": {
                    "ports": [{"name": "kv", "count": "{{tp}}*2"}],
                    "env": {"P": "{{ports.kv}}"},
                }
            },
        )


# ---------------------------------------------------------------------------
# Capability 3: cross-role references.
# ---------------------------------------------------------------------------


def test_cross_role_references_render_both_directions():
    """Measured on Ascend: both sides carry both sides' parallelism —
    prefill's connector config names decode's TP and vice versa. NIXL has no
    equivalent coupling, which is why this is declarable rather than
    hardcoded."""
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    for role in ("prefill", "decode"):
        extra = ascend.role(role).connector["kv_connector_extra_config"]
        assert extra["prefill"] == {
            "tp_size": "{{roles.prefill.tensor_parallel_size}}",
            "dp_size": "{{roles.prefill.data_parallel_size}}",
        }
        assert extra["decode"] == {
            "tp_size": "{{roles.decode.tensor_parallel_size}}",
            "dp_size": "{{roles.decode.data_parallel_size}}",
        }
    # The NIXL path declares no cross-role reference at all.
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    assert "roles." not in repr(nixl.roles)


def test_cross_role_reference_to_an_undeclared_role_fails():
    with pytest.raises(ValueError, match="undeclared role 'decode'"):
        PDMode(
            name="dangling",
            roles={"prefill": {"connector": {"tp": "{{roles.decode.tp_size}}"}}},
        )


def test_placeholders_reject_inner_spaces():
    """The renderer's pattern takes no spaces, so "{{ worker_ip }}" would
    reach the container verbatim — measured as a ZMQError on a bind
    address."""
    with pytest.raises(ValueError, match="malformed placeholder"):
        PDMode(name="spaced", roles={"prefill": {"env": {"H": "{{ worker_ip }}"}}})


# ---------------------------------------------------------------------------
# Capability 4: router capabilities.
# ---------------------------------------------------------------------------


def test_router_capabilities_are_declared_per_mode():
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    assert nixl.router.protocol == PDRouterProtocolEnum.TWO_HOP
    assert nixl.router.capabilities.metrics is True
    assert nixl.router.capabilities.models_endpoint is True
    assert nixl.router.capabilities.kv_expired_metric is True
    assert nixl.router.health_path == "/health"

    # Measured on the shipped runner (vllm_ascend 0.20.2rc1): GET /metrics and
    # GET /v1/models both 404 -- polling them produced a ~1/s 404 storm and a
    # permanent false failure -- while GET /healthcheck returns 200. The
    # endpoint set differs between vllm-ascend versions (a v0.23.0 example has
    # no health path at all), so what is asserted is the version we ship.
    # Ascend now runs the same router as the vLLM recipe, verified on 910B2
    # through a real prefill-decode pair. What stays false is the connector's
    # KV-expiry counter: Mooncake exports none, and a router with metrics does
    # not conjure an engine-side counter that was never written.
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    assert ascend.router.capabilities.metrics is True
    assert ascend.router.capabilities.models_endpoint is True
    assert ascend.router.capabilities.kv_expired_metric is False
    assert ascend.router.health_path == "/health"


def test_router_capabilities_default_to_absent():
    """An undeclared endpoint must read as absent, not assumed present."""
    mode = PDMode(
        name="bare",
        router={
            "protocol": "two_hop",
            "command": ["router"],
            "peers": {"style": "repeated_flag"},
        },
    )
    assert mode.router.capabilities.metrics is False
    assert mode.router.capabilities.models_endpoint is False
    assert mode.router.capabilities.kv_expired_metric is False
    assert mode.router.health_path is None


def test_router_peer_styles_and_prometheus_band():
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    ascend = get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value)
    sglang = get_pd_mode(PDModeEnum.SGLANG_MOONCAKE.value)
    custom = get_pd_mode(PDModeEnum.CUSTOM.value)

    assert nixl.router.peers.style == PDPeerStyleEnum.REPEATED_FLAG
    assert nixl.router.peers.prefill == {
        "flag": "--prefill",
        "value": "http://{{peer.ip}}:{{peer.port}}",
    }
    # The router's own Prometheus port is fixed at 29000 upstream and
    # always binds, so a second group's router on one host panics; it is a
    # managed band like any other.
    assert [band.name for band in nixl.router.ports] == ["prometheus"]
    assert "{{ports.prometheus}}" in nixl.router.command
    # And the exposition has to be bound to the worker's address, not to
    # the default loopback. Measured: worker_ip:40001/metrics was refused
    # while 127.0.0.1:40001/metrics served 31 series, which puts the ratio's
    # denominator out of reach of anything off-host. Same class of bug as
    # VLLM_NIXL_SIDE_CHANNEL_HOST, same fix.
    assert (
        nixl.router.command[nixl.router.command.index("--prometheus-host") + 1]
        == "{{worker_ip}}"
    )
    # Fast failure detection is the circuit breaker's, not the health
    # check's: the breaker runs on the request path and sees a dead worker at
    # real traffic rate, while a short health-check interval lands near the
    # engine's HTTP keep-alive and ejects healthy workers on a stale socket.
    # See tests/worker/test_pd_router.py for the full argument.
    for flag in ("--cb-failure-threshold", "--retry-max-retries"):
        assert flag in nixl.router.command
    assert not [c for c in nixl.router.command if str(c).startswith("--health")]

    # Hosts and ports as two parallel flags.
    # `repeated_flag` since the router swap. The `parallel_lists` renderer is
    # still supported and still tested, through a synthetic mode in
    # tests/worker/test_pd_router.py — no shipped mode needs it now, and the
    # renderer should not lose coverage for that.
    assert ascend.router.peers.style == PDPeerStyleEnum.REPEATED_FLAG
    assert ascend.router.peers.prefill == {
        "flag": "--prefill",
        "value": "http://{{peer.ip}}:{{peer.port}}",
    }
    # 🔴 `nixl` on a Mooncake-transport mode is deliberate: the flag names the
    # wire protocol shape, not the transport. Choosing `mooncake` makes the
    # router wait forever on a bootstrap server vllm-ascend does not run —
    # measured on 910B2, 30 retries and no request ever completed.
    assert ascend.router.command[ascend.router.command.index("--kv-connector") + 1] == (
        "nixl"
    )
    assert [band.name for band in ascend.router.ports] == ["prometheus"]

    # A prefill peer carries THAT PEER'S bootstrap band as a second positional
    # value. `peer.ports.` and not `ports.`: the latter is the deployment scope
    # and means the router's own band of that name, so it renders verbatim into
    # the address — which is what shipped, and what a two-prefill group could
    # not have expressed correctly even if it had resolved.
    assert "{{peer.ports.bootstrap}}" in sglang.router.peers.prefill["value"]
    assert "{{ports.bootstrap}}" not in sglang.router.peers.prefill["value"]

    # custom supplies nothing: image, command and ports are the user's.
    assert custom.router.protocol == PDRouterProtocolEnum.USER_PROVIDED
    assert custom.router.command == []
    assert custom.router.image is None
    assert custom.roles == {}


def test_user_provided_router_must_not_declare_a_command():
    with pytest.raises(ValueError, match="user_provided router"):
        PDMode(
            name="contradiction",
            router={"protocol": "user_provided", "command": ["whatever"]},
        )


def test_unknown_peer_style_fails_the_load():
    """A new wire style needs a renderer branch, so an unknown value must
    not be quietly accepted and rendered wrong."""
    with pytest.raises(ValueError):
        PDMode(
            name="future",
            router={
                "protocol": "two_hop",
                "command": ["router"],
                "peers": {"style": "grpc_delegate"},
            },
        )


# ---------------------------------------------------------------------------
# Capability 5: KV lease / abort window per connector (D31).
# ---------------------------------------------------------------------------


def test_kv_lease_windows_are_per_connector():
    leases = get_kv_leases()

    # NIXL: a lease with heartbeat renewal, in the connector's extra
    # config, and the one window with a Prometheus counter.
    nixl = leases["nixl"]
    assert nixl.param == "kv_lease_duration"
    assert nixl.inject_to == PDKVLeaseTargetEnum.CONNECTOR_EXTRA_CONFIG
    assert nixl.engine_default == 30
    assert nixl.gpustack_default == 60
    assert nixl.expired_metric == "vllm:nixl_num_kv_expired_reqs"
    assert nixl.settable is True

    # Mooncake: an env var, 16x the window, and no Prometheus counter at
    # all — an expiry is only visible in the engine log.
    mooncake = leases["mooncake"]
    assert mooncake.param == "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT"
    assert mooncake.inject_to == PDKVLeaseTargetEnum.ENV
    assert mooncake.engine_default == 480
    assert mooncake.gpustack_default == 60
    assert mooncake.expired_metric is None

    # MoRIIO: a hardcoded constant, 120x NIXL's window. On the record even
    # though no shipped mode uses it and nothing can be injected.
    moriio = leases["moriio"]
    assert moriio.engine_default == 3600
    assert moriio.settable is False
    assert moriio.inject_to == PDKVLeaseTargetEnum.NONE
    assert moriio.gpustack_default is None
    assert all(mode.kv_lease is not moriio for mode in load_pd_modes())

    # SGLang's nixl backend has no reclaim timeout at all: a cancelled
    # request strands its KV until the instance restarts.
    sglang_nixl = leases["sglang-nixl"]
    assert sglang_nixl.param is None
    assert sglang_nixl.settable is False
    assert sglang_nixl.expired_metric is None


def test_modes_resolve_their_connector_window():
    assert get_pd_mode(PDModeEnum.VLLM_NIXL.value).kv_lease is get_kv_lease("nixl")
    assert get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value).kv_lease is get_kv_lease(
        "mooncake"
    )
    assert get_pd_mode(PDModeEnum.SGLANG_NIXL.value).kv_lease is get_kv_lease(
        "sglang-nixl"
    )
    # custom configures nothing, including the window.
    assert get_pd_mode(PDModeEnum.CUSTOM.value).kv_lease is None
    # The declared window is what the connector's own config template
    # renders from.
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value)
    extra = nixl.role("prefill").connector["kv_connector_extra_config"]
    assert extra["kv_lease_duration"] == "{{kv_lease_duration}}"


def test_unknown_kv_lease_reference_fails_the_load():
    with pytest.raises(PDModeCatalogError, match="references kv_lease"):
        parse_pd_mode_catalog(
            _document([{"name": PDModeEnum.CUSTOM.value, "kv_lease": "no-such"}])
        )


def test_expired_metric_claim_must_match_the_window():
    """Two spellings of one fact: the router capability is what the metrics
    collector reads, the lease carries the metric's name."""
    document = _document(
        [
            {
                "name": PDModeEnum.VLLM_NIXL.value,
                "backends": PD_MODE_BACKENDS[PDModeEnum.VLLM_NIXL.value],
                "kv_lease": "mooncake",
                "router": {
                    "protocol": "two_hop",
                    "command": ["router"],
                    "peers": {"style": "repeated_flag"},
                    "capabilities": {"kv_expired_metric": True},
                },
            }
        ],
        kv_leases=[
            {
                "connector": "mooncake",
                "param": "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT",
                "inject_to": "env",
                "engine_default": 480,
            }
        ],
    )
    with pytest.raises(PDModeCatalogError, match="kv_expired_metric"):
        parse_pd_mode_catalog(document)


def test_unsettable_window_must_not_claim_an_injection_target():
    with pytest.raises(PDModeCatalogError, match="not settable"):
        parse_pd_mode_catalog(
            _document(
                [],
                kv_leases=[
                    {
                        "connector": "moriio",
                        "param": "VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT",
                        "inject_to": "env",
                        "settable": False,
                        "engine_default": 3600,
                    }
                ],
            )
        )


# ---------------------------------------------------------------------------
# Transfer counters (X2 3.1 / 3.2).
# ---------------------------------------------------------------------------


def test_the_read_side_is_declared_because_reading_the_wrong_one_inverts_it():
    """NIXL is pull-based: a completed transfer is counted on decode. On a
    working 1P1D, prefill's counter stayed 0.0 for the whole run and
    decode's went to 1.0, so reading prefill reports a healthy pair as
    having transferred nothing."""
    nixl = get_transfer_metrics("nixl")
    assert nixl.read_from_role == "decode"
    assert nixl.xfer_count == "vllm:nixl_xfer_time_seconds_count"
    assert nixl.xfer_seconds == "vllm:nixl_xfer_time_seconds_sum"
    assert nixl.failed_transfers == "vllm:nixl_num_failed_transfers_total"
    # No byte counter in the measured record, so the degradation check runs
    # on a rate in transfers rather than on true bandwidth.
    assert nixl.xfer_bytes is None
    # And no floor: a threshold is a calibration on real hardware, and the
    # measured spread (94% of line rate on 2.5GbE, 9% on 910B2 RoCE) is why
    # an invented one is worse than none.
    assert nixl.min_expected_rate is None


def test_a_connector_that_exports_nothing_says_so():
    """mooncake/stats.py exports zero Prometheus counters where NIXL exports
    fifteen. That is a measured fact, not an omission."""
    mooncake = get_transfer_metrics("mooncake")
    assert mooncake is not None
    assert mooncake.observable is False
    assert get_pd_mode(PDModeEnum.VLLM_ASCEND_MOONCAKE.value).transfer_metrics is (
        mooncake
    )


def test_modes_resolve_their_connector_counters_through_one_reference():
    """The connector id names the transport once; the lease window and the
    counters both hang off it, so the two cannot come to disagree."""
    for name in (PDModeEnum.VLLM_NIXL.value, PDModeEnum.SGLANG_NIXL.value):
        mode = get_pd_mode(name)
        assert mode.transfer_metrics is get_transfer_metrics(mode.kv_lease.connector)
    assert get_pd_mode(PDModeEnum.CUSTOM.value).transfer_metrics is None


def test_a_connector_missing_from_the_counter_registry_fails_the_load():
    """An all-null entry declares "exports nothing"; a missing entry behaves
    identically at runtime while meaning nobody looked."""
    with pytest.raises(PDModeCatalogError, match="Missing: \\['nixl'\\]"):
        parse_pd_mode_catalog(
            _document(
                [],
                kv_leases=[
                    {
                        "connector": "nixl",
                        "param": "kv_lease_duration",
                        "inject_to": "connector_extra_config",
                    }
                ],
                kv_transfer_metrics=[],
            )
        )


def test_transfer_seconds_without_a_count_is_not_a_rate():
    with pytest.raises(PDModeCatalogError, match="a rate needs both"):
        parse_pd_mode_catalog(
            _document(
                [],
                kv_leases=[
                    {
                        "connector": "nixl",
                        "param": "kv_lease_duration",
                        "inject_to": "connector_extra_config",
                    }
                ],
                kv_transfer_metrics=[{"connector": "nixl", "xfer_seconds": "s"}],
            )
        )


def test_the_ratios_denominator_is_declared_per_router():
    """Per-worker, which is what localises the failure to one decode rather
    than to "the group"."""
    nixl = get_pd_mode(PDModeEnum.VLLM_NIXL.value).router.request_metrics
    assert nixl.prefill_requests == "vllm_router_pd_prefill_requests_total"
    assert nixl.decode_requests == "vllm_router_pd_decode_requests_total"
    assert nixl.worker_label == "worker"
    # Coarser and measured too: aggregated by route, so it localises
    # nothing but still answers whether anything was routed.
    assert nixl.total_requests == "vllm_router_pd_requests_total"

    # SGLang's gateway is the same codebase, and the names are still NOT the
    # same — this one prefixes `smg_`, vllm-router `vllm_router_`. So the
    # guess the catalog used to refuse to make would have read zero, which is
    # exactly why it refused. Declared now because measured, not inferred.
    for name in (PDModeEnum.SGLANG_MOONCAKE.value, PDModeEnum.SGLANG_NIXL.value):
        sglang = get_pd_mode(name).router.request_metrics
        assert sglang.available is True
        assert sglang.total_requests == "smg_router_requests_total"
        assert not sglang.total_requests.startswith("vllm_router")
        # Its exposition is a second listener on a band GPUStack allocates:
        # measured, the router binds the hardcoded 29000 when left alone, so
        # two groups on a host would collide and the serving port 404s.
        assert sglang.port_band == "prometheus"

    # A router that declares no metrics has no denominator by capability
    # rather than by omission. Synthetic since the router swap: every shipped
    # mode serves metrics now, and the rule is about the capability, not about
    # whichever entry happened to lack it.
    from copy import deepcopy

    silent = deepcopy(get_pd_mode(PDModeEnum.VLLM_NIXL.value).router)
    silent.capabilities.metrics = False
    # `available` is derived from the counter names, so clearing them is what
    # makes the denominator absent — the object stays, which is why callers can
    # ask without a None check.
    silent.request_metrics.prefill_requests = None
    silent.request_metrics.decode_requests = None
    silent.request_metrics.total_requests = None
    assert silent.capabilities.metrics is False
    assert silent.request_metrics.available is False


def test_request_metrics_behind_a_metrics_false_capability_fail_the_load():
    with pytest.raises(PDModeCatalogError, match="nothing would ever scrape them"):
        parse_pd_mode_catalog(
            _document(
                [
                    {
                        "name": PDModeEnum.VLLM_NIXL.value,
                        "backends": PD_MODE_BACKENDS[PDModeEnum.VLLM_NIXL.value],
                        "router": {
                            "protocol": "two_hop",
                            "command": ["router"],
                            "peers": {"style": "repeated_flag"},
                            "capabilities": {"metrics": False},
                            "request_metrics": {"decode_requests": "x_total"},
                        },
                    }
                ]
            )
        )


# ---------------------------------------------------------------------------
# Loader behaviour.
# ---------------------------------------------------------------------------


def test_catalog_is_cached_and_reloadable():
    first = load_pd_mode_catalog()
    assert load_pd_mode_catalog() is first
    assert load_pd_mode_catalog(reload=True) is not first


def test_mode_lookup_is_case_insensitive():
    assert get_pd_mode("VLLM-NIXL") is not None
    assert get_pd_mode("no-such-mode") is None
    assert get_pd_mode("") is None


def test_malformed_document_fails_the_load():
    with pytest.raises(PDModeCatalogError, match="must be a mapping"):
        parse_pd_mode_catalog([{"name": "vllm-nixl"}])
    with pytest.raises(PDModeCatalogError, match="must be mappings"):
        parse_pd_mode_catalog({"modes": ["vllm-nixl"]})


def test_every_shipped_mode_declares_a_router_and_two_engine_roles():
    for mode in load_pd_modes():
        assert mode.router is not None, mode.name
        if mode.name == PDModeEnum.CUSTOM.value:
            continue
        assert set(mode.roles) == {"prefill", "decode"}, mode.name
        assert mode.display_name and mode.description, mode.name


def test_all_five_capabilities_round_trip_through_serialization():
    """What the endpoint serves is the parsed catalog re-serialized, and
    re-validating the dump re-runs every load-time check, so this covers
    both directions of the round trip."""
    for mode in load_pd_modes():
        assert PDMode(**mode.model_dump()) == mode
        assert PDMode.model_validate_json(mode.model_dump_json()) == mode

    dumped = {mode.name: mode.model_dump(mode="json") for mode in load_pd_modes()}
    nixl = dumped[PDModeEnum.VLLM_NIXL.value]
    ascend = dumped[PDModeEnum.VLLM_ASCEND_MOONCAKE.value]
    # 1: injection target, 2: band width, 3: cross-role reference,
    # 4: router capabilities, 5: the lease window.
    assert nixl["roles"]["prefill"]["ports"][0]["inject_to"] == "env"
    assert ascend["roles"]["prefill"]["ports"][0]["inject_to"] == "args"
    assert ascend["roles"]["decode"]["ports"][0]["count"] == "{{accelerator_count}}"
    assert (
        ascend["roles"]["prefill"]["connector"]["kv_connector_extra_config"]["decode"][
            "tp_size"
        ]
        == "{{roles.decode.tensor_parallel_size}}"
    )
    assert ascend["router"]["capabilities"] == {
        "metrics": True,
        "models_endpoint": True,
        # The connector's, not the router's: Mooncake exports no counter for an
        # expired lease, so there is nothing for a metrics-serving router to
        # forward.
        "kv_expired_metric": False,
    }
    assert nixl["kv_lease"]["inject_to"] == "connector_extra_config"
    assert nixl["kv_lease"]["engine_default"] == 30
    assert ascend["kv_lease"]["param"] == "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT"
    assert ascend["kv_lease"]["engine_default"] == 480
