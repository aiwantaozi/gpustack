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
    "prefill": [("10.0.0.1", 40027), ("10.0.0.2", 40029)],
    "decode": [("10.0.0.3", 40028)],
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


def test_the_measured_failure_detection_values_are_carried():
    """Not tuning. With the router's own defaults, killing one prefill left
    >50% of requests returning 500 for over a minute — the health check
    interval alone defaults to 60s. With these, the same run stayed 8/8."""
    plan = render_router(get_pd_mode("vllm-nixl"), VARIABLES, PEERS)
    command = plan.command

    assert command[command.index("--health-check-interval-secs") + 1] == "5"
    assert command[command.index("--health-failure-threshold") + 1] == "1"
    assert command[command.index("--retry-max-retries") + 1] == "3"


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

    assert peers["prefill"] == [("10.0.0.1", 40027), ("10.0.0.2", 40029)]
    assert ("10.0.0.1", 40031) not in peers["prefill"]


def test_only_running_members_contribute_an_address():
    """Ports are assigned worker-side at start, so an address read off a member
    that has not started is a placeholder the router would carry for life."""
    peers = group_peer_addresses(_instances(), "g1", IPS)

    assert peers["decode"] == [("10.0.0.1", 40028)]


def test_the_order_is_stable():
    """An unchanged membership must render an unchanged command, or every
    reconcile looks like a spec change to anything comparing commands."""
    once = group_peer_addresses(_instances(), "g1", IPS)
    again = group_peer_addresses(list(reversed(_instances())), "g1", IPS)

    assert once == again


def test_a_worker_with_no_known_ip_is_skipped_not_guessed():
    peers = group_peer_addresses(_instances(), "g1", {1: "10.0.0.1"})

    assert peers["prefill"] == [("10.0.0.1", 40027)]


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
