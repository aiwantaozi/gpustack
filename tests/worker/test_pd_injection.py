"""The PD-mode catalog rendered into an engine launch.

Two properties are what these pin:

* a recipe reaches the process — `vllm-nixl`'s prefill role comes out as the
  three NIXL env variables plus one `--kv-transfer-config`, and its decode
  role differs from it in exactly one field (`kv_role`), because that is the
  whole of what the two sides disagree on;
* everything that cannot render correctly is visible. An unresolved
  placeholder survives verbatim with a warning rather than being blanked, and
  a launch where two sources would write `--kv-transfer-config` is refused
  instead of being silently concatenated into a flag vLLM reads once.
"""

import json
import logging
import types

import pytest

from gpustack.schemas.models import (
    DisaggregationSpec,
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    Model,
    ModelInstance,
    PDModeEnum,
    PortBand,
    RoleSpec,
    SourceEnum,
)
from gpustack.utils.template import deployment_variables
from gpustack.worker.pd_injection import (
    KV_TRANSFER_CONFIG_FLAG,
    PDInjectionError,
    render_pd_injection,
)


def _model(mode=PDModeEnum.VLLM_NIXL, roles=None, **kwargs) -> Model:
    if roles is None:
        roles = [
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1),
            RoleSpec(name="router", replicas=1, cpu_only=True),
        ]
    return Model(
        id=1,
        name="llm",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        roles=roles,
        disaggregation=(DisaggregationSpec(mode=mode) if mode is not None else None),
        **kwargs,
    )


def _instance(role="prefill", named_ports=None, **kwargs) -> ModelInstance:
    if named_ports is None:
        named_ports = {"kv_side_channel": PortBand(base=5600, count=1)}
    return ModelInstance(
        id=1,
        name="llm-0",
        model_id=1,
        model_name="llm",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        worker_id=1,
        port=40000,
        role=role,
        group_id="g1",
        named_ports=named_ports,
        **kwargs,
    )


def _variables(**overrides):
    """What `_template_variables()` hands the injector on the worker."""
    variables = deployment_variables(
        model_path="/models/llm",
        port=40000,
        worker_ip="192.168.50.10",
        model_name="llm",
        gpu_count=1,
        gpu_ids=[0],
        role=overrides.pop("role", "prefill"),
        group_id="g1",
    )
    variables["net_device"] = "eth0"
    variables["runner_image"] = "gpustack/runner:cuda12.8-vllm0.20.0"
    variables.update(overrides)
    return variables


def _connector(injection):
    """The rendered `--kv-transfer-config` document."""
    assert KV_TRANSFER_CONFIG_FLAG in injection.args
    return json.loads(injection.args[injection.args.index(KV_TRANSFER_CONFIG_FLAG) + 1])


def test_vllm_nixl_prefill_renders_the_whole_launch():
    injection = render_pd_injection(_model(), _instance(), _variables())

    assert injection.env == {
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "192.168.50.10",
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "5600",
        "UCX_NET_DEVICES": "eth0",
    }
    assert injection.args == [
        KV_TRANSFER_CONFIG_FLAG,
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer",'
        '"kv_load_failure_policy":"fail",'
        '"kv_connector_extra_config":{"kv_lease_duration":60}}',
    ]
    assert injection.files == {}
    # The lease window is a number in the document, not the string the
    # template was: the engine parses this as JSON.
    assert _connector(injection)["kv_connector_extra_config"] == {
        "kv_lease_duration": 60
    }


def test_vllm_nixl_decode_differs_only_in_kv_role():
    prefill = render_pd_injection(_model(), _instance("prefill"), _variables())
    decode = render_pd_injection(
        _model(), _instance("decode"), _variables(role="decode")
    )

    assert _connector(decode)["kv_role"] == "kv_consumer"
    assert _connector(prefill)["kv_role"] == "kv_producer"
    # Both sides bind and advertise their own side channel.
    assert decode.env == prefill.env


def test_kv_load_failure_policy_comes_from_the_deployment():
    model = _model()
    model.disaggregation.kv_load_failure_policy = "recompute"

    injection = render_pd_injection(model, _instance(), _variables())

    assert _connector(injection)["kv_load_failure_policy"] == "recompute"


def test_not_a_pd_instance_returns_none():
    # No `disaggregation` at all: plain multi-role orchestration.
    assert render_pd_injection(_model(mode=None), _instance(), _variables()) is None
    # No role: a single-role deployment.
    assert render_pd_injection(_model(), _instance(role=None), _variables()) is None


def test_router_role_is_not_an_engine_injection():
    """The router's command is assembled by the router path, not merged into
    an engine's launch — `mode.roles` has no entry for it."""
    assert (
        render_pd_injection(
            _model(),
            _instance("router", named_ports={"prometheus": PortBand(base=29000)}),
            _variables(role="router"),
            peers={"prefill": [("192.168.50.10", 40000)]},
        )
        is None
    )


def test_custom_mode_injects_nothing():
    assert (
        render_pd_injection(_model(mode=PDModeEnum.CUSTOM), _instance(), _variables())
        is None
    )


def test_unallocated_port_band_survives_verbatim_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        injection = render_pd_injection(
            _model(), _instance(named_ports={}), _variables()
        )

    # Not blanked: a blank port is a plausible-looking wrong value, while the
    # literal is the M0 failure that names itself
    # (ZMQError: No such device (addr='tcp://{{worker_ip}}:5600')).
    assert injection.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "{{ports.kv_side_channel}}"
    assert "ports.kv_side_channel" in caplog.text


def test_unresolved_net_device_survives_verbatim(caplog):
    variables = _variables()
    variables.pop("net_device")

    with caplog.at_level(logging.WARNING):
        injection = render_pd_injection(_model(), _instance(), variables)

    assert injection.env["UCX_NET_DEVICES"] == "{{net_device}}"
    assert "net_device" in caplog.text


def test_spaced_placeholder_is_not_one():
    """`{{ worker_ip }}` is not a placeholder — the catalog loader rejects the
    spelling, and the renderer would leave it alone anyway."""
    model = _model()
    injection = render_pd_injection(model, _instance(), _variables())
    rendered = render_pd_injection(
        model, _instance(), {**_variables(), " worker_ip ": "10.0.0.1"}
    )
    assert rendered.env == injection.env


@pytest.mark.parametrize(
    "extended",
    [
        ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.LOCAL),
        ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
        ),
    ],
)
def test_pd_renders_its_own_connector_beside_an_extended_cache(extended):
    """The injector's job is this role's connector, whole. A cache that also
    contributes one is folded in later by `kv_transfer`, once the whole argv
    exists — which is where the per-role ordering can be applied. Refusing
    here is what used to make the two mutually exclusive."""
    model = _model(extended_kv_cache=extended)

    injection = render_pd_injection(model, _instance(), _variables())

    assert KV_TRANSFER_CONFIG_FLAG in injection.args
    # Its own descriptor, not a composite: composing is not this seam's call.
    index = injection.args.index(KV_TRANSFER_CONFIG_FLAG)
    assert "NixlConnector" in injection.args[index + 1]


def test_a_per_role_cache_leaves_every_role_renderable():
    """Which sides take a cache is per role, and none of them is a reason to
    refuse the PD connector — the prefill side is exactly where a shared cache
    pays."""
    model = _model(
        roles=[
            RoleSpec(
                name="prefill",
                extended_kv_cache=ExtendedKVCacheConfig(
                    enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
                ),
            ),
            RoleSpec(name="decode"),
        ]
    )

    assert render_pd_injection(model, _instance("prefill"), _variables()).args
    assert render_pd_injection(
        model, _instance("decode"), _variables(role="decode")
    ).args


def test_user_written_kv_transfer_config_is_refused():
    model = _model(
        backend_parameters=[KV_TRANSFER_CONFIG_FLAG, '{"kv_connector":"Mine"}']
    )

    with pytest.raises(PDInjectionError):
        render_pd_injection(model, _instance(), _variables())


def test_sglang_pd_does_not_collide_with_a_shared_cache():
    """SGLang's disaggregation is configured through its own flags, so the
    exclusion is on the flag, not on "PD plus cache"."""
    model = _model(
        mode=PDModeEnum.SGLANG_MOONCAKE,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
        ),
    )

    injection = render_pd_injection(
        model,
        _instance(named_ports={"bootstrap": PortBand(base=8998, count=1)}),
        _variables(),
    )

    assert injection.args == [
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-bootstrap-port",
        "8998",
        "--disaggregation-transfer-backend",
        "mooncake",
    ]
    assert KV_TRANSFER_CONFIG_FLAG not in injection.args


def test_ascend_mooncake_carries_both_sides_parallelism_and_a_lease_env():
    model = _model(
        mode=PDModeEnum.VLLM_ASCEND_MOONCAKE,
        roles=[
            RoleSpec(
                name="prefill",
                backend_parameters=["--tensor-parallel-size", "8"],
            ),
            RoleSpec(
                name="decode",
                backend_parameters=["--tensor-parallel-size", "4", "--dp", "2"],
            ),
        ],
    )
    instance = _instance(named_ports={"kv_port": PortBand(base=41100, count=8)})

    injection = render_pd_injection(model, instance, _variables())
    descriptor = _connector(injection)

    # Cross-role: prefill's own config names decode's parallelism.
    assert descriptor["kv_connector_extra_config"]["prefill"]["tp_size"] == 8
    assert descriptor["kv_connector_extra_config"]["decode"]["tp_size"] == 4
    assert descriptor["kv_connector_extra_config"]["decode"]["dp_size"] == 2
    # The port lives inside the descriptor for this connector, as a number.
    assert descriptor["kv_port"] == 41100
    # A parallelism nobody wrote stays unresolved rather than defaulting to 1:
    # GPUStack injects a tensor-parallel size of its own further down.
    assert (
        descriptor["kv_connector_extra_config"]["prefill"]["dp_size"]
        == "{{roles.prefill.data_parallel_size}}"
    )
    # Mooncake's abort window is an env var, and its engine default is 8
    # minutes with no Prometheus counter to notice an expiry.
    assert injection.env["VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT"] == "60"
    assert injection.env["HCCL_SOCKET_IFNAME"] == "eth0"


def test_nixl_lease_is_not_injected_as_an_env_var():
    """NIXL's window is a connector-config field; injecting it as an env var
    as well would be a second spelling of one fact."""
    injection = render_pd_injection(_model(), _instance(), _variables())
    assert "kv_lease_duration" not in injection.env


def _backend(model, instance):
    """A worker-side server object with only what the injection seams read.

    `_model_spec` is the unprojected model and `_model` the projection, which
    is the split the start path itself has: the injector must be handed the
    former, or a cross-role reference resolves against the running role's
    values.
    """
    from gpustack.schemas.models import role_effective_model
    from gpustack.worker.backends.custom import CustomServer

    backend = CustomServer.__new__(CustomServer)
    backend._model_spec = model
    backend._model = role_effective_model(model, instance.role)
    backend._model_instance = instance
    backend._worker = types.SimpleNamespace(id=1, ip="192.168.50.10", ifname="eth0")
    backend._config = types.SimpleNamespace(data_dir="/var/lib/gpustack")
    backend._model_path = "/models/llm"
    backend.inference_backend = None
    backend._get_selected_gpu_devices = lambda: []
    backend._resolve_image = lambda backend_name=None: (
        "gpustack/runner:cuda12.8-vllm0.20.0",
        None,
    )
    return backend


def test_start_path_carries_env_args_and_attributes_them_to_gpustack():
    model = _model(backend_parameters=["--max-model-len", "8192"])
    backend = _backend(model, _instance())

    env = backend._get_configured_env()
    assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.168.50.10"
    assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "5600"
    # Derived from the worker's own interface until the net-device module
    # lands; either way it is never left as "all".
    assert env["UCX_NET_DEVICES"] == "eth0"

    tokens = backend._flatten_backend_param()
    assert tokens[0] == KV_TRANSFER_CONFIG_FLAG
    assert json.loads(tokens[1])["kv_role"] == "kv_producer"
    assert tokens[2:] == ["--max-model-len", "8192"]

    arguments = ["vllm", "serve", "/models/llm"] + tokens
    injected = backend._get_injected_backend_parameters(arguments, tokens)
    assert injected[0] == KV_TRANSFER_CONFIG_FLAG
    assert "--max-model-len" not in injected


def test_non_pd_deploy_takes_the_same_path_it_takes_today():
    model = _model(mode=None, backend_parameters=["--max-model-len", "8192"])
    backend = _backend(model, _instance())

    assert backend._pd_injection() is None
    assert backend._flatten_backend_param() == ["--max-model-len", "8192"]
    assert "VLLM_NIXL_SIDE_CHANNEL_HOST" not in backend._get_configured_env()
    assert backend._cache_injection_files() == {}


def test_a_refused_injection_stays_refused_at_every_seam():
    """The refusal must not be cached as "nothing to inject": the seams run in
    sequence, and a seam that swallowed the first one would then start the
    engine with neither connector.

    Uses the one clash that is still a refusal — a hand-written
    `--kv-transfer-config` under a mode that injects its own. The cache is no
    longer one of these, because it is composed rather than refused."""
    model = _model(
        backend_parameters=[KV_TRANSFER_CONFIG_FLAG, '{"kv_connector":"Mine"}']
    )
    backend = _backend(model, _instance())

    with pytest.raises(PDInjectionError):
        backend._get_configured_env()
    with pytest.raises(PDInjectionError):
        backend._flatten_backend_param()


def test_model_env_overrides_the_injection_but_says_so(caplog):
    model = _model(env={"UCX_NET_DEVICES": "mlx5_0:1"})
    backend = _backend(model, _instance())

    with caplog.at_level(logging.WARNING):
        env = backend._get_configured_env()

    assert env["UCX_NET_DEVICES"] == "mlx5_0:1"
    assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.168.50.10"
    assert "UCX_NET_DEVICES" in caplog.text


def test_injection_files_join_the_shared_cache_files(caplog):
    from gpustack.schemas.cache_services import CacheConfigSnapshot
    from gpustack.worker.pd_injection import PDInjection

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(
        cache_service_id=1,
        injected=True,
        files={"/tmp/cache.json": "cache", "/tmp/both.json": "cache"},
    )
    backend = _backend(_model(mode=None), instance)
    backend._pd_injection_resolved = True
    backend._pd_injection_cache = PDInjection(
        files={"/tmp/pd.json": "pd", "/tmp/both.json": "pd"}
    )

    with caplog.at_level(logging.WARNING):
        files = backend._cache_injection_files()

    assert files == {
        "/tmp/cache.json": "cache",
        "/tmp/pd.json": "pd",
        "/tmp/both.json": "pd",
    }
    assert "/tmp/both.json" in caplog.text
    # And the serving script writes every one of them before the engine runs.
    script = backend._get_serving_command_script({})
    assert "/tmp/pd.json" in script and "/tmp/cache.json" in script


def test_unknown_mode_returns_none_and_warns(caplog):
    model = _model()
    model.disaggregation = types.SimpleNamespace(
        mode="not-a-mode", kv_load_failure_policy="fail"
    )

    with caplog.at_level(logging.WARNING):
        assert render_pd_injection(model, _instance(), _variables()) is None

    assert "not-a-mode" in caplog.text


# --- the host IPC trade-off ------------------------------------------------ #


def test_a_disaggregated_member_with_a_cache_is_told_what_it_traded(caplog):
    """A shared cache wants the host IPC namespace, for the CUDA-IPC path that
    passes KV buffers instead of copying them. A KV connector wants a private
    /dev/shm, and joining the host namespace replaces it with the host's,
    dropping the shm_size the workload was given.

    Both configurations run, so this does not refuse. What it must not do is
    decide silently: the person who cares cannot otherwise see the question
    was asked, and both directions are one env away."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    backend = _backend(_model(), instance)

    with caplog.at_level(logging.WARNING):
        assert backend._host_ipc_enabled() is True

    assert "/dev/shm" in caplog.text
    assert "GPUSTACK_HOST_IPC" in caplog.text


def test_a_role_less_deployment_with_a_cache_is_not_warned(caplog):
    """No connector, no tension — the derivation is just the cache's own
    requirement and there is nothing being traded away."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.role = None
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    model = _model()
    model.disaggregation = None
    backend = _backend(model, instance)

    with caplog.at_level(logging.WARNING):
        assert backend._host_ipc_enabled() is True

    assert "/dev/shm" not in caplog.text


def test_the_warning_is_said_once(caplog):
    """It is derived on every workload build; repeating it per build would
    bury the things that happen once."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    backend = _backend(_model(), instance)

    with caplog.at_level(logging.WARNING):
        backend._host_ipc_enabled()
        backend._host_ipc_enabled()

    assert caplog.text.count("GPUSTACK_HOST_IPC") == 1


def test_the_escape_hatch_wins_and_says_nothing(caplog):
    """An explicit answer is not a trade-off being made for anyone."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    backend = _backend(_model(env={"GPUSTACK_HOST_IPC": "false"}), instance)

    with caplog.at_level(logging.WARNING):
        assert backend._host_ipc_enabled() is False

    assert "/dev/shm" not in caplog.text
