"""Rendering a group's router from the pd-mode catalog.

There is no universal router, and that is a finding rather than a limitation:
SGLang ships one, vLLM ships a fork of that one, vllm-ascend ships a Python
proxy example, and TileRT expects you to bring your own. What generalises is
the *catalog format*, not the binary — so this module is one renderer driven by
declarations, not a family of per-engine adapters. Adding an engine whose
router reuses a known peer style is a YAML change; only a genuinely new wire
style needs a branch here, which is why an unknown style fails at catalog load
rather than being rendered wrong.

Deliberately separate from `pd_injection`: prefill and decode are *engines*
that receive injected configuration on top of a command GPUStack already knows
how to build, while a managed router is a command GPUStack builds from nothing.
The two produce different things and the injector says so — it returns None for
the router rather than pretending it has a role entry, since `vllm-nixl`'s
catalog entry keeps the router under `router:` and not under `roles:`.
"""

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel

from gpustack.schemas.pd_modes import (
    PDMode,
    PDPeerStyleEnum,
    PDRouterProtocolEnum,
)
from gpustack.utils.template import render

logger = logging.getLogger(__name__)

PeerAddress = Tuple[str, int]
PeerMap = Mapping[str, Sequence[PeerAddress]]


class RouterPlan(BaseModel):
    """What a managed router needs in order to start."""

    image: Optional[str] = None
    command: List[str] = []
    health_path: Optional[str] = None
    metrics: bool = False
    """Whether the router serves a Prometheus exposition. False is not a
    default to fall back on — vllm-ascend's proxy example has neither /metrics
    nor /v1/models, and polling them produced a ~1/s 404 storm in its log plus
    a permanent false alarm. So health checks degrade to process liveness and
    the PD-effectiveness ratio loses its denominator, both knowingly."""
    models_endpoint: bool = False


class RouterPeersUnavailable(Exception):
    """Raised when a managed router is asked to render without its peers.

    A hard failure on purpose (D10). Every other way of handling this produces
    a router that starts and answers requests wrongly: an empty `--prefill`
    list makes vllm-router accept traffic it has nowhere to send, and a
    partial one silently halves the group. PD failures are invisible from the
    outside, so degrading here would be lying.
    """


def _peer_scope(ip: str, port: int) -> Dict[str, object]:
    """The variables visible while rendering ONE peer's address.

    Deliberately not merged with the deployment scope. The catalog first spelled
    these `{{ip}}` / `{{port}}`, and `{{port}}` already means "this member's own
    HTTP port" out there — so a merged scope would resolve a peer's address to
    the *router's* port. That is a wrong address rather than a failure, which is
    the class of bug this whole design keeps having to defend against. The
    `peer.` prefix makes the two scopes unmergeable by construction.
    """
    return {"peer.ip": ip, "peer.port": port}


def _repeated_flag(spec: Mapping[str, str], peers: Sequence[PeerAddress]) -> List[str]:
    """`--prefill http://a:1 --prefill http://b:2` (vLLM router, SGLang router)."""
    flag = spec.get("flag")
    value = spec.get("value")
    if not flag or not value:
        return []
    args: List[str] = []
    for ip, port in peers:
        args.append(flag)
        args.append(render(value, _peer_scope(ip, port), context="router peer address"))
    return args


def _parallel_lists(spec: Mapping[str, str], peers: Sequence[PeerAddress]) -> List[str]:
    """`--prefiller-hosts a b --prefiller-ports 1 2` (vllm-ascend's proxy).

    The two lists are positional against each other, so they are built from one
    iteration rather than two — a filter applied to one and not the other would
    pair each host with the wrong port and still start.
    """
    host_flag = spec.get("host_flag")
    port_flag = spec.get("port_flag")
    if not host_flag or not port_flag:
        return []
    hosts = [ip for ip, _ in peers]
    ports = [str(port) for _, port in peers]
    return [host_flag, *hosts, port_flag, *ports]


def render_router(
    mode: PDMode,
    variables: Mapping[str, object],
    peers: PeerMap,
) -> Optional[RouterPlan]:
    """Render the router of `mode` for a group whose peers are `peers`.

    Returns None when the mode declares no router GPUStack manages — either it
    has none at all, or its protocol is `user_provided`, where the image and
    command come from the role spec and rendering one here would override what
    the user typed.

    `variables` is the deployment scope (the router's own `worker_ip`, `port`,
    `ports.<name>`, `runner_image`, `model_name`, `group_id`). Peer addresses
    are NOT in it — see `_peer_scope`.
    """
    router = mode.router
    if router is None:
        return None
    if router.protocol == PDRouterProtocolEnum.USER_PROVIDED:
        return None
    if router.peers is None or router.peers.style == PDPeerStyleEnum.USER_PROVIDED:
        return None

    command = [
        render(part, variables, context="router command") for part in router.command
    ]

    # Peer flags go after the declared command rather than interleaved: the
    # catalog's command is the invariant part of the invocation and the peer
    # list is the part that changes on every scale, and keeping them apart is
    # what lets a scale re-render the tail without re-deriving the head.
    peer_args: List[str] = []
    for role, spec in (
        ("prefill", router.peers.prefill),
        ("decode", router.peers.decode),
    ):
        if not spec:
            continue
        addresses = list(peers.get(role) or [])
        if not addresses:
            raise RouterPeersUnavailable(
                f"No {role} address is available for the router. The router is "
                f"created only after its dependencies report ready, so this "
                f"means a member was lost between that check and this render."
            )
        if router.peers.style == PDPeerStyleEnum.REPEATED_FLAG:
            peer_args.extend(_repeated_flag(spec, addresses))
        elif router.peers.style == PDPeerStyleEnum.PARALLEL_LISTS:
            peer_args.extend(_parallel_lists(spec, addresses))
        else:
            # Unreachable: the catalog loader rejects unknown styles, which is
            # what keeps "a new wire protocol needs code" from being discovered
            # at render time on a user's cluster.
            raise RouterPeersUnavailable(
                f"Peer style '{router.peers.style}' has no renderer."
            )

    return RouterPlan(
        image=render(router.image, variables, context="router image"),
        command=command + peer_args,
        health_path=router.health_path,
        metrics=router.capabilities.metrics,
        models_endpoint=router.capabilities.models_endpoint,
    )


def is_managed_router(model, role_name: Optional[str]) -> bool:
    """Whether this role is a router GPUStack assembles rather than one the
    user supplied.

    Derived, not declared. `RoleSpec` has no `managed` flag and deliberately so:
    "managed" is exactly "the user did not give it an image and a command", and
    a separate boolean could disagree with that — a role marked managed while
    carrying a hand-written command would have to resolve which one wins, and
    every answer to that is surprising to somebody.
    """
    from gpustack.schemas.models import RoleNameEnum, find_role

    if role_name != RoleNameEnum.ROUTER.value:
        return False
    disaggregation = getattr(model, "disaggregation", None)
    if disaggregation is None:
        return False

    role = find_role(model, role_name)
    if role is not None and role.image_name and role.run_command:
        return False

    from gpustack.server.pd_mode_catalog import get_pd_mode

    mode_name = getattr(disaggregation.mode, "value", disaggregation.mode)
    mode = get_pd_mode(mode_name)
    if mode is None or mode.router is None:
        return False
    return mode.router.protocol != PDRouterProtocolEnum.USER_PROVIDED


def apply_managed_router(
    model,
    role_name: Optional[str],
    *,
    peers: Optional[PeerMap] = None,
    variables: Optional[Mapping[str, object]] = None,
):
    """Turn a managed router's role-effective model into the shape the existing
    launch path already runs: a custom backend with an image and a command.

    Nothing new is taught to the worker. A managed router IS an image plus a
    command line, which is precisely what the custom backend already deploys,
    so expressing it that way means the container build, the port assignment,
    the health probe and the log capture all keep working with no router-shaped
    branch anywhere below this line.

    Called at two depths with different information, on purpose. The serve
    manager asks before the child process exists and passes no peers: all it
    needs is the backend, which decides the port band and the fallback
    registry, and rendering a command there would render it from addresses it
    would then have to re-render. The backend itself asks with peers, and it
    is the one that has to produce a command that will actually run.

    The result is never persisted — same rule as the projection it extends.
    A router's command holds its peers' addresses, and those change on every
    scale, so storing one would be storing a value that is wrong as soon as
    anything moves.
    """
    from gpustack.schemas.models import BackendEnum, RoleEffectiveModel

    if not is_managed_router(model, role_name):
        return model

    projected = (
        model
        if isinstance(model, RoleEffectiveModel)
        else RoleEffectiveModel.model_validate(model)
    )
    # A router runs a router binary, not an inference engine, so it must not
    # inherit the group's backend — a vLLM-shaped launch would try to serve the
    # model weights from the router's container.
    projected.backend = BackendEnum.CUSTOM

    if peers is None or variables is None:
        return projected

    from gpustack.server.pd_mode_catalog import get_pd_mode

    disaggregation = model.disaggregation
    mode_name = getattr(disaggregation.mode, "value", disaggregation.mode)
    plan = render_router(get_pd_mode(mode_name), variables, peers)
    if plan is None:
        return projected

    projected.image_name = plan.image
    # Joined rather than kept as a list because `run_command` is the field the
    # custom backend reads, and it is a string there. The renderer produced
    # already-separated tokens, so nothing here has to guess at quoting.
    projected.run_command = " ".join(plan.command)
    return projected


def group_peer_addresses(
    instances: Sequence[object],
    group_id: Optional[str],
    worker_ip_by_id: Mapping[int, str],
) -> Dict[str, List[PeerAddress]]:
    """Collect `{role: [(ip, port), ...]}` for one generation.

    Scoped by `group_id` rather than by model id, and that is the pairing
    boundary doing its job: a router must not be able to resolve a member of
    another generation, which is what makes a cross-generation pair
    structurally impossible rather than merely unlikely (F7 3.3).

    Sorted, so that a re-render with an unchanged membership produces an
    unchanged command — otherwise every reconcile would look like a spec change
    to anything comparing commands.
    """
    from gpustack.schemas.models import ModelInstanceStateEnum

    peers: Dict[str, List[PeerAddress]] = {}
    for instance in instances:
        role = getattr(instance, "role", None)
        if not role or getattr(instance, "group_id", None) != group_id:
            continue
        if getattr(instance, "state", None) != ModelInstanceStateEnum.RUNNING:
            # Only running members have had their ports assigned; an address
            # taken from a member that has not started yet is a placeholder,
            # and the router would carry it for the rest of its life.
            continue
        ip = worker_ip_by_id.get(getattr(instance, "worker_id", None))
        port = getattr(instance, "port", None)
        if not ip or not port:
            continue
        peers.setdefault(role, []).append((ip, int(port)))

    for role in peers:
        peers[role].sort()
    return peers
