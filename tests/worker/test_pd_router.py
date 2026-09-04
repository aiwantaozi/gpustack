"""Rendering a group's router from the catalog.

There is no universal router — SGLang ships one, vLLM ships a fork of it,
vllm-ascend ships a Python proxy example, TileRT expects you to bring your own.
What generalises is the catalog format, so this is one renderer driven by
declarations rather than a family of per-engine adapters.

The properties pinned here are the ones whose absence produces a router that
*starts* and is wrong:

* peer addresses render in their own scope. The catalog first spelled them
  `{{ip}}` / `{{port}}`, and `{{port}}` already means "this member's own HTTP
  port" — a merged scope resolves every peer to the router's own port;
* the parallel-lists style builds both lists from one iteration, since a
  host list and a port list that disagree pair each host with the wrong port
  and still start;
* missing peers are a hard failure. An empty `--prefill` list makes the router
  accept traffic it has nowhere to send.
"""

import pytest

from gpustack.server.pd_mode_catalog import get_pd_mode, load_pd_modes
from gpustack.schemas.pd_modes import (
    PDMode,
    PDRouter,
    PDRouterPeers,
)
from gpustack.worker.pd_router import (
    RouterPeersUnavailable,
    group_peer_addresses,
    PeerAddress,
    render_router,
)

load_pd_modes()

VARIABLES = {
    "worker_ip": "192.168.50.15",
    "port": 40030,
    "ports.prometheus": 40040,
    "runner_image": "gpustack/runner:cuda12.9-vllm0.17.1",
    "model_name": "qwen3-0.6b",
    "group_id": "32-a1b2c3d4",
}

PEERS = {
    "prefill": [
        PeerAddress("10.0.0.1", 40027),
        PeerAddress("10.0.0.2", 40029),
    ],
    "decode": [PeerAddress("10.0.0.3", 40028)],
}


def _parallel_lists_mode() -> PDMode:
    """vllm-ascend's proxy example style, kept alive as the fallback path for
    when the young vllm-router turns out not to be usable."""
    return PDMode(
        name="custom",
        router=PDRouter(
            protocol="two_hop",
            image="{{runner_image}}",
            command=["proxy", "--port", "{{port}}"],
            peers=PDRouterPeers(
                style="parallel_lists",
                prefill={
                    "host_flag": "--prefiller-hosts",
                    "port_flag": "--prefiller-ports",
                },
                decode={
                    "host_flag": "--decoder-hosts",
                    "port_flag": "--decoder-ports",
                },
            ),
        ),
    )


# --- the shipped vllm-nixl router ------------------------------------------ #


def test_the_catalog_router_renders_a_complete_command():
    plan = render_router(get_pd_mode("vllm-nixl"), VARIABLES, PEERS)

    assert plan.image == "gpustack/runner:cuda12.9-vllm0.17.1"
    assert plan.command[0] == "vllm-router"
    assert plan.health_path == "/health"


def test_the_routers_own_placeholders_resolve():
    plan = render_router(get_pd_mode("vllm-nixl"), VARIABLES, PEERS)

    assert "{{" not in " ".join(plan.command), plan.command
    assert "192.168.50.15" in plan.command
    assert "40030" in plan.command
    # Its Prometheus port defaults to 29000 and always binds despite what its
    # --help claims, so a second group's router on one host dies in a Rust
    # panic on the collision. The named band is what prevents that.
    assert "40040" in plan.command


ROUTER_MODES = ["vllm-nixl", "sglang-mooncake", "sglang-nixl"]


@pytest.mark.parametrize("mode", ROUTER_MODES)
def test_the_breaker_is_what_carries_fast_failure_detection(mode):
    """🔴 This asserted "2", on the reasoning that ten failures is "ten users'
    requests spent learning what two would have taught it". A live run taught
    the other half of it (2026-08-28, SGLang 1P1D): under GPU contention a
    handful of timeouts opened the prefill circuit at threshold 2, and it never
    closed — every later request fast-failed in ~0.17s with "all circuits open
    or unhealthy", while the router's own health gauge for that worker read 1
    and the engine answered /health 200. Still 503 after 90s of silence, past
    the 60s half-open timer.

    So the breaker is still the fast path and still the right mechanism; the
    value was wrong. What two failures buy in detection latency they can cost
    as an unrecoverable outage, and with one replica per role there is nothing
    to fail over to — an open circuit is not failover, it is the outage.

    `--retry-max-retries` is unchanged: retries are bounded work on the request
    path and have no latching state to get stuck in.
    """
    command = render_router(get_pd_mode(mode), VARIABLES, PEERS).command

    assert command[command.index("--cb-failure-threshold") + 1] == "10"
    assert command[command.index("--retry-max-retries") + 1] == "3"


@pytest.mark.parametrize("mode", ROUTER_MODES)
def test_the_health_check_is_left_to_the_router(mode):
    """The regression this guards is re-adding a short health-check interval,
    which reads like an obvious improvement and is not.

    A short interval lands near the engine's HTTP keep-alive — uvicorn's
    default 5s, which neither vLLM nor SGLang overrides — and the router pools
    its connections to workers. At interval == 5 the engine closed a pooled
    connection as the checker reached for it on ~4.5% of ticks, and with one
    replica of a role an ejection is not failover but an outage: the next
    request came back 503 from an engine that was perfectly healthy.

    Fast detection is the breaker's job, above. The health check is allowed to
    be slow, and at 60s it is also nowhere near the keep-alive boundary.
    """
    command = render_router(get_pd_mode(mode), VARIABLES, PEERS).command

    overrides = [c for c in command if str(c).startswith("--health")]
    assert not overrides, (
        f"{mode} overrides {overrides}; fast failure detection belongs to the "
        "circuit breaker. Re-adding these needs a single-variable measurement "
        "that isolates them from it."
    )


def test_capabilities_come_from_the_catalog_not_from_assumption():
    """An undeclared endpoint must be treated as absent: polling a router that
    serves neither /metrics nor /v1/models produced a ~1/s 404 storm and a
    permanent false alarm."""
    plan = render_router(get_pd_mode("vllm-nixl"), VARIABLES, PEERS)

    assert plan.metrics is True
    assert plan.models_endpoint is True


# --- peer styles ----------------------------------------------------------- #


def test_repeated_flag_emits_one_flag_per_peer():
    plan = render_router(get_pd_mode("vllm-nixl"), VARIABLES, PEERS)
    command = plan.command

    assert command.count("--prefill") == 2
    assert command.count("--decode") == 1
    assert "http://10.0.0.1:40027" in command
    assert "http://10.0.0.2:40029" in command
    assert "http://10.0.0.3:40028" in command


def test_a_peers_port_is_not_the_routers_port():
    """The scope-collision this design changed the catalog to prevent. With a
    merged scope every peer address would carry 40030 — the router's own port —
    and the router would happily start pointing at itself."""
    plan = render_router(get_pd_mode("vllm-nixl"), VARIABLES, PEERS)

    assert "http://10.0.0.1:40030" not in plan.command
    assert "http://10.0.0.1:40027" in plan.command


def test_parallel_lists_keeps_hosts_and_ports_positionally_paired():
    plan = render_router(_parallel_lists_mode(), VARIABLES, PEERS)
    command = plan.command

    hosts = command.index("--prefiller-hosts")
    ports = command.index("--prefiller-ports")
    assert command[hosts + 1 : ports] == ["10.0.0.1", "10.0.0.2"]
    assert command[ports + 1 : ports + 3] == ["40027", "40029"]


# --- refusals -------------------------------------------------------------- #


def test_a_missing_peer_role_is_a_hard_failure():
    """An empty `--prefill` list makes the router accept traffic it has
    nowhere to send. PD failures are invisible from outside, so degrading here
    would be lying (D10)."""
    with pytest.raises(RouterPeersUnavailable, match="prefill"):
        render_router(get_pd_mode("vllm-nixl"), VARIABLES, {"decode": PEERS["decode"]})


def test_a_user_provided_router_renders_nothing():
    """Its image and command come from the role spec, so rendering one here
    would override what the user typed."""
    mode = PDMode(
        name="custom",
        router=PDRouter(protocol="user_provided"),
    )
    assert render_router(mode, VARIABLES, PEERS) is None


def test_a_mode_with_no_router_renders_nothing():
    assert render_router(PDMode(name="custom"), VARIABLES, PEERS) is None


# --- collecting the addresses ---------------------------------------------- #


class _Instance:
    def __init__(self, role, group_id, worker_id, port, state):
        self.role = role
        self.group_id = group_id
        self.worker_id = worker_id
        self.port = port
        self.state = state


def _instances():
    from gpustack.schemas.models import ModelInstanceStateEnum as S

    return [
        _Instance("prefill", "g1", 1, 40027, S.RUNNING),
        _Instance("prefill", "g1", 2, 40029, S.RUNNING),
        _Instance("decode", "g1", 1, 40028, S.RUNNING),
        _Instance("router", "g1", 1, 40030, S.RUNNING),
        # Another generation of the same model.
        _Instance("prefill", "g2", 1, 40031, S.RUNNING),
        # Not started, so its port is not yet its own.
        _Instance("decode", "g1", 2, 40032, S.PENDING),
        # A plain model's instance.
        _Instance(None, None, 1, 40033, S.RUNNING),
    ]


IPS = {1: "10.0.0.1", 2: "10.0.0.2"}


def test_peers_are_scoped_to_one_generation():
    """The pairing boundary doing its job: a router that could resolve another
    generation's member is exactly the cross-generation pair `group_id` exists
    to make structurally impossible."""
    peers = group_peer_addresses(_instances(), "g1", IPS)

    assert peers["prefill"] == [
        PeerAddress("10.0.0.1", 40027),
        PeerAddress("10.0.0.2", 40029),
    ]
    assert PeerAddress("10.0.0.1", 40031) not in peers["prefill"]


def test_only_running_members_contribute_an_address():
    """Ports are assigned worker-side at start, so an address read off a member
    that has not started is a placeholder the router would carry for life."""
    peers = group_peer_addresses(_instances(), "g1", IPS)

    assert peers["decode"] == [PeerAddress("10.0.0.1", 40028)]


def test_the_order_is_stable():
    """An unchanged membership must render an unchanged command, or every
    reconcile looks like a spec change to anything comparing commands."""
    once = group_peer_addresses(_instances(), "g1", IPS)
    again = group_peer_addresses(list(reversed(_instances())), "g1", IPS)

    assert once == again


def test_a_worker_with_no_known_ip_is_skipped_not_guessed():
    peers = group_peer_addresses(_instances(), "g1", {1: "10.0.0.1"})

    assert peers["prefill"] == [PeerAddress("10.0.0.1", 40027)]


# --- materialising a managed router ---------------------------------------- #


def _pd_model(router_role=None):
    from gpustack.schemas.models import (
        DisaggregationSpec,
        Model,
        PDModeEnum,
        RoleSpec,
        SourceEnum,
    )

    return Model(
        id=32,
        name="qwen3-0.6b",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen3-0.6B",
        owner_principal_id=1,
        backend="vLLM",
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1),
            router_role or RoleSpec(name="router", replicas=1, cpu_only=True),
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
    )


def test_a_managed_router_becomes_a_custom_backend_deployment():
    """Nothing new is taught to the worker: a router IS an image plus a command
    line, which is what the custom backend already deploys."""
    from gpustack.schemas.models import BackendEnum
    from gpustack.worker.pd_router import apply_managed_router

    model = apply_managed_router(_pd_model(), "router")

    assert model.backend == BackendEnum.CUSTOM


def test_a_router_does_not_inherit_the_groups_engine():
    """Inheriting vLLM would launch an inference server in the router's
    container and try to serve the model weights from it."""
    from gpustack.worker.pd_router import apply_managed_router

    assert _pd_model().backend == "vLLM"
    assert apply_managed_router(_pd_model(), "router").backend != "vLLM"


def test_the_gpu_roles_are_untouched():
    from gpustack.worker.pd_router import apply_managed_router

    for role in ("prefill", "decode"):
        model = apply_managed_router(_pd_model(), role)
        assert model.backend == "vLLM"
        assert model.run_command is None


def test_a_role_less_model_is_untouched():
    from gpustack.schemas.models import Model, SourceEnum
    from gpustack.worker.pd_router import apply_managed_router

    plain = Model(
        id=1,
        name="m",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
    )
    assert apply_managed_router(plain, None) is plain


def test_a_user_supplied_router_keeps_its_own_command():
    """ "Managed" is derived from whether the user gave it an image and a
    command, so a hand-written router cannot be overwritten by the catalog."""
    from gpustack.schemas.models import RoleSpec
    from gpustack.worker.pd_router import apply_managed_router, is_managed_router

    model = _pd_model(
        RoleSpec(
            name="router",
            replicas=1,
            cpu_only=True,
            image_name="me/my-router:1",
            run_command="my-router --serve",
        )
    )
    assert is_managed_router(model, "router") is False
    assert apply_managed_router(model, "router") is model


def test_with_peers_the_command_is_materialised():
    from gpustack.worker.pd_router import apply_managed_router

    model = apply_managed_router(
        _pd_model(), "router", peers=PEERS, variables=VARIABLES
    )

    assert model.image_name == "gpustack/runner:cuda12.9-vllm0.17.1"
    assert model.run_command.startswith("vllm-router ")
    assert "--prefill http://10.0.0.1:40027" in model.run_command


def test_without_peers_only_the_backend_moves():
    """The serve manager asks before the child process exists and needs only
    the backend; rendering a command from addresses there would render it
    twice."""
    from gpustack.worker.pd_router import apply_managed_router

    model = apply_managed_router(_pd_model(), "router")

    assert model.run_command is None
    assert model.image_name is None


def test_materialisation_does_not_reach_the_stored_spec():
    """Same rule as the projection it extends. A router's command holds its
    peers' addresses, which change on every scale, so a persisted one is wrong
    as soon as anything moves."""
    from gpustack.worker.pd_router import apply_managed_router

    original = _pd_model()
    apply_managed_router(original, "router", peers=PEERS, variables=VARIABLES)

    router_role = next(r for r in original.roles if r.name == "router")
    assert router_role.run_command is None
    assert original.run_command is None
    assert original.backend == "vLLM"


# --- resolving the router's image ------------------------------------------ #


def test_the_runner_image_is_resolved_against_the_groups_engine():
    """Measured twice on live clusters, the same symptom from two causes.

    First on Kubernetes: the image reached the apiserver as the literal
    `{{runner_image}}` and the pod was rejected with InvalidImageName, because
    the resolution ran inside `get_model()` before `inference_backend` was
    assigned. Naming the engine's backend in the call looked like enough.

    It was not. On 910B2 the same placeholder reached docker, which rejected it
    as an invalid reference — because naming the backend does not change which
    model gets read. A managed router is switched to the custom backend before
    this runs, and `custom` is neither a runner service nor a backend row, so
    both resolution paths came back empty. The router binary ships inside the
    ENGINE's runner image, so the ENGINE's spec and backend row are what have
    to reach the resolver.
    """
    from types import SimpleNamespace

    from gpustack.schemas.models import BackendEnum
    from gpustack.worker.backends.base import InferenceServer

    asked = {}
    engine_row = SimpleNamespace(backend_name=BackendEnum.VLLM.value)
    spec = SimpleNamespace(
        backend=BackendEnum.VLLM.value, backend_version="0.20.2-ascend-pd-custom"
    )

    def _resolve_image(backend=None, spec=None, inference_backend=None):
        asked.update(backend=backend, spec=spec, inference_backend=inference_backend)
        return "gpustack/runner:cuda12.9-vllm0.17.1", None

    fake = SimpleNamespace(
        _worker=SimpleNamespace(ifname="eno1", name="node-a"),
        _config=SimpleNamespace(kv_ifname=None),
        # The unprojected model still carries the group's engine; the projected
        # one has been switched to Custom.
        _model_spec=spec,
        _model=SimpleNamespace(backend=BackendEnum.CUSTOM.value),
        _resolve_image=_resolve_image,
        _engine_inference_backend=lambda _spec: engine_row,
    )

    variables = InferenceServer._pd_template_variables(fake)

    assert asked["backend"] == BackendEnum.VLLM.value
    assert asked["spec"] is spec, (
        "the resolver reads the model it is given; without the spec it reads "
        "the projected one, whose backend is custom"
    )
    assert asked["inference_backend"] is engine_row, (
        "a custom backend version's image lives only on the backend row, and "
        "the row this server was handed is the router's own — which is None"
    )
    assert variables["runner_image"] == "gpustack/runner:cuda12.9-vllm0.17.1"


def test_get_model_does_not_materialise_the_router():
    """The materialisation needs `inference_backend`, which `__init__` assigns
    after `get_model()` returns. Doing it inside `get_model()` is what made the
    resolution fail silently."""
    import inspect

    from gpustack.worker.backends import base

    source = inspect.getsource(base.InferenceServer.get_model)
    assert "_apply_managed_router" not in source

    # `__init__` is wrapped by a timing decorator, so its source has to come
    # from the file rather than from the callable.
    text = inspect.getsource(base)
    assigned = text.index("self.inference_backend = inference_backend")
    materialised = text.index("self._model = self._apply_managed_router(self._model)")
    fallback = text.index("backend_name=BackendEnum.CUSTOM.value")
    guard = text.index("not specified or not found")

    # After the backend registry exists — resolving the runner image reads it.
    assert (
        assigned < materialised
    ), "the router must be materialised after the backend registry exists"
    # And before the fallback that synthesises a custom backend out of an image
    # and a run command, because those are exactly what it produces.
    assert (
        materialised < fallback < guard
    ), "the router must be materialised before the custom-backend fallback"


def test_the_groups_engine_parameters_do_not_reach_the_router():
    """Observed on a live pod: `--max-model-len=8192` was appended to a
    vllm-router invocation, which has no such flag. Model-level parameters are
    inherited by projection like any other field, and the custom backend
    appends them to whatever command it is handed. The catalog's command is
    complete by construction, so there is nothing for them to add."""
    from gpustack.worker.pd_router import apply_managed_router

    model = _pd_model()
    model.backend_parameters = ["--max-model-len=8192"]

    projected = apply_managed_router(model, "router", peers=PEERS, variables=VARIABLES)

    assert projected.backend_parameters == []
    assert "--max-model-len" not in projected.run_command
    # The stored spec is untouched — a router materialisation must not reach
    # back into what the engine members read.
    assert model.backend_parameters == ["--max-model-len=8192"]


def test_a_routers_own_named_band_reaches_its_command():
    """The engine roles get `{{ports.<name>}}` from the injector, which builds
    its own context. The router never goes through the injector, so its
    `--prometheus-port` reached the container as the literal placeholder — the
    band was allocated and declared as a host port, and the process was told to
    bind a string."""
    from gpustack.worker.pd_router import apply_managed_router

    variables = dict(VARIABLES)
    variables["ports.prometheus"] = 40002

    projected = apply_managed_router(
        _pd_model(), "router", peers=PEERS, variables=variables
    )

    assert "--prometheus-port 40002" in projected.run_command
    assert "{{" not in projected.run_command


def test_a_role_image_survives_the_catalogs():
    """ "Bring your own router image" without also writing the invocation. The
    runner image does not ship `vllm-router` today, so this is the only way to
    run one at all until it does."""
    from gpustack.schemas.models import RoleSpec
    from gpustack.worker.pd_router import apply_managed_router, is_managed_router

    from gpustack.schemas.models import role_effective_model

    model = _pd_model(
        RoleSpec(name="router", replicas=1, cpu_only=True, image_name="me/has-router:1")
    )
    assert is_managed_router(model, "router") is True

    # The projection runs first in production, so the role's image is already
    # on the model by the time the router is materialised.
    projected = apply_managed_router(
        role_effective_model(model, "router"),
        "router",
        peers=PEERS,
        variables=VARIABLES,
    )

    assert projected.image_name == "me/has-router:1"
    # The catalog still supplies the invocation, peers included.
    assert projected.run_command.startswith("vllm-router ")
    assert "--prefill http://10.0.0.1:40027" in projected.run_command


def test_a_role_command_survives_the_catalogs():
    """The mirror: a hand-written invocation of something already in the
    engine's runner image."""
    from gpustack.schemas.models import RoleSpec
    from gpustack.worker.pd_router import apply_managed_router

    from gpustack.schemas.models import role_effective_model

    model = _pd_model(
        RoleSpec(name="router", replicas=1, cpu_only=True, run_command="my-router --go")
    )

    projected = apply_managed_router(
        role_effective_model(model, "router"),
        "router",
        peers=PEERS,
        variables=VARIABLES,
    )

    assert projected.run_command == "my-router --go"
    assert projected.image_name == "gpustack/runner:cuda12.9-vllm0.17.1"


# --- a peer's own named ports ---------------------------------------------- #


def test_each_prefill_peer_carries_its_own_bootstrap_port():
    """SGLang's `--prefill URL BOOTSTRAP_PORT`, which the deployment scope
    cannot express: the band is allocated per member because the engine's fixed
    default (8998) collides as soon as two prefills share a host, so there is no
    single value for the whole render."""
    peers = {
        "prefill": [
            PeerAddress("10.0.0.1", 40101, {"bootstrap": 40300}),
            PeerAddress("10.0.0.2", 40111, {"bootstrap": 40400}),
        ],
        "decode": [PeerAddress("10.0.0.3", 40102)],
    }

    command = render_router(get_pd_mode("sglang-mooncake"), VARIABLES, peers).command

    # Separate argv tokens, not one string: the router parses the pair
    # positionally.
    i = command.index("--prefill")
    assert command[i : i + 3] == ["--prefill", "http://10.0.0.1:40101", "40300"]
    j = command.index("--prefill", i + 1)
    assert command[j : j + 3] == ["--prefill", "http://10.0.0.2:40111", "40400"]


def test_a_peer_port_never_resolves_from_the_deployment_scope():
    """The regression that shipped: `{{ports.bootstrap}}` in a peer address
    reached the router verbatim, and `sglang_router` would have parsed the
    literal `{{ports.bootstrap}}` as a port number. Nothing in a rendered
    command may carry an unresolved placeholder."""
    peers = {
        "prefill": [PeerAddress("10.0.0.1", 40101, {"bootstrap": 40300})],
        "decode": [PeerAddress("10.0.0.3", 40102)],
    }

    for mode in ("sglang-mooncake", "sglang-nixl"):
        command = render_router(get_pd_mode(mode), VARIABLES, peers).command
        unresolved = [c for c in command if "{{" in str(c)]
        assert not unresolved, f"{mode} left {unresolved}"


def test_a_decode_peer_needs_no_band_and_renders_without_one():
    peers = {
        "prefill": [PeerAddress("10.0.0.1", 40101, {"bootstrap": 40300})],
        "decode": [PeerAddress("10.0.0.3", 40102)],
    }

    command = render_router(get_pd_mode("sglang-mooncake"), VARIABLES, peers).command

    i = command.index("--decode")
    assert command[i : i + 2] == ["--decode", "http://10.0.0.3:40102"]


def test_a_router_that_needs_env_to_boot_gets_it_from_the_catalog():
    """Some routers cannot start without an environment variable. Measured on
    910B2: vllm-ascend's proxy imported vllm.logger, which pulled in torch_npu,
    which refuses to load with no accelerator visible -- landing on the one
    role that is cpu_only by design.

    Synthetic now that the Ascend recipe runs vllm-router instead, which is a
    Rust binary and imports no torch at all. No shipped mode needs the escape
    hatch today, and the catalog should not lose the ability to express it for
    that reason."""
    from copy import deepcopy

    from gpustack.worker.pd_router import render_router

    mode = deepcopy(get_pd_mode("vllm-nixl"))
    mode.router.env = {"NEEDS_THIS_TO_BOOT": "0"}
    plan = render_router(mode, VARIABLES, PEERS)

    assert plan.env["NEEDS_THIS_TO_BOOT"] == "0"


def test_catalog_env_is_a_default_the_deployment_can_override():
    """The other way round from image and command, which yield wholesale to
    the role: these are boot requirements, so the ones a user did not speak to
    still arrive."""
    from gpustack.schemas.models import PDModeEnum, RoleSpec, role_effective_model
    from gpustack.worker.pd_router import apply_managed_router

    model = _pd_model(
        RoleSpec(
            name="router",
            replicas=1,
            cpu_only=True,
            env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "1", "MY_OWN": "x"},
        )
    )
    model.disaggregation.mode = PDModeEnum.VLLM_ASCEND_MOONCAKE

    # The worker projects the role before it materialises the router, so the
    # role's env is already on the model by the time the catalog's is merged.
    projected = apply_managed_router(
        role_effective_model(model, "router"),
        "router",
        peers=PEERS,
        variables=VARIABLES,
    )

    assert projected.env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "1"
    assert projected.env["MY_OWN"] == "x"


def test_a_router_env_placeholder_is_rendered_not_passed_through():
    """Same failure the whole template layer exists for: an unrendered
    {{worker_ip}} reached a container once and became `ZMQError: No such
    device`."""
    from gpustack.schemas.pd_modes import (
        PDMode,
        PDPeerStyleEnum,
        PDRouter,
        PDRouterPeers,
        PDRouterProtocolEnum,
    )
    from gpustack.worker.pd_router import render_router

    mode = PDMode(
        name="test-router-env",
        router=PDRouter(
            protocol=PDRouterProtocolEnum.TWO_HOP,
            image="img",
            command=["run"],
            env={"ADVERTISE": "{{worker_ip}}"},
            peers=PDRouterPeers(
                style=PDPeerStyleEnum.REPEATED_FLAG,
                prefill={
                    "flag": "--prefill",
                    "value": "http://{{peer.ip}}:{{peer.port}}",
                },
                decode={
                    "flag": "--decode",
                    "value": "http://{{peer.ip}}:{{peer.port}}",
                },
            ),
        ),
    )

    plan = render_router(mode, VARIABLES, PEERS)

    assert plan.env["ADVERTISE"] == "192.168.50.15"


def test_the_routers_own_parameters_are_appended_and_the_groups_are_not():
    """Two different things arrive in `backend_parameters` and only one is the
    router's.

    Inherited engine parameters put `--max-model-len=8192` on a `vllm-router`
    invocation that has no such flag, which is why they are dropped. Parameters
    written *on the router role* are the opposite case — someone asked for a
    routing policy — and appending them works because every tunable flag is
    last-wins in both shipped routers.
    """
    from gpustack.schemas.models import (
        DisaggregationSpec,
        Model,
        PDModeEnum,
        RoleSpec,
        SourceEnum,
    )
    from gpustack.worker.pd_router import apply_managed_router

    def _model(router_params):
        return Model(
            name="m",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/repo",
            backend_parameters=["--max-model-len=8192"],
            disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
            roles=[
                RoleSpec(name="prefill", replicas=1),
                RoleSpec(name="decode", replicas=1),
                RoleSpec(name="router", replicas=1, backend_parameters=router_params),
            ],
        )

    peers = {
        "prefill": [PeerAddress(ip="10.0.0.1", port=8000)],
        "decode": [PeerAddress(ip="10.0.0.2", port=8000)],
    }
    variables = {
        "worker_ip": "10.0.0.9",
        "port": 8080,
        "ports": {"prometheus": 40001},
        "runner_image": "img",
    }

    def _params(router_params):
        applied = apply_managed_router(
            _model(router_params), "router", peers=peers, variables=variables
        )
        return applied.backend_parameters

    # Nothing declared on the role: what is there is the group's, and it is
    # dropped rather than appended to a command line that cannot read it.
    assert _params(None) == []

    # Declared on the role: kept, so it lands after the catalog's own flags.
    assert _params(["--decode-policy", "cache_aware"]) == [
        "--decode-policy",
        "cache_aware",
    ]

    # An explicitly empty list is a statement too, and it must not resurrect
    # the group's parameters.
    assert _params([]) == []
