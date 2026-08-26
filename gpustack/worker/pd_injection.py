"""The PD-mode catalog turned into what one member's engine actually starts with.

`pd-modes.yaml` declares a recipe per mode and, inside it, per role: which
named port bands the role needs, which connector descriptor it runs, and which
env / args / files carry them. Nothing consumed that declaration until this
module: a group came up with the roles scheduled, the ports allocated and not
one `--kv-transfer-config` or `VLLM_NIXL_SIDE_CHANNEL_HOST` on any command
line, which is a deployment that looks healthy while serving aggregated.

The rendering itself is `gpustack.utils.template` — the same substitution the
shared-cache provider catalog uses, with the same three rules that make a
missing value diagnosable:

1. an unknown placeholder is left verbatim and logged, never blanked (a
   blank host is a plausible-looking wrong value; `{{worker_ip}}` reaching
   the engine is a `ZMQError: No such device` that names itself);
2. `{{ x }}` with inner spaces is not a placeholder;
3. env values do not see each other.

What this module adds is the *context*: the renderer takes a flat map keyed by
the whole dotted name, and assembling it — the instance's port bands, the
worker's NIC, the other roles' fields, the connector's lease window — is the
caller's job. That is what `_pd_variables` below does.
"""

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from gpustack.schemas.models import role_effective_model
from gpustack.schemas.pd_modes import PDKVLeaseTargetEnum, PDMode, PDModeRole
from gpustack.server.pd_mode_catalog import get_pd_mode
from gpustack.utils.command import find_int_parameter, flatten_to_argv
from gpustack.utils.template import render, render_values

logger = logging.getLogger(__name__)

KV_TRANSFER_CONFIG_FLAG = "--kv-transfer-config"

_WHOLE_PLACEHOLDER = re.compile(r"^\{\{[A-Za-z_][A-Za-z0-9_.]*\}\}$")
"""A value that is nothing but one placeholder. Such a value renders to its
source's own type — a port band's base is a number in a connector descriptor,
not the string "5600" — while a value with text around it stays a string."""

# Parallelism is not a `RoleSpec` field: it is an engine parameter, and the
# Ascend recipe needs each side's connector config to carry the other side's.
# Read from the role's effective backend parameters under the spellings the
# two engines accept. What an absent one means depends on the member: for one
# on a single worker the engine derives tp from the cards it was given and runs
# dp at 1, so `_implicit_parallelism` supplies exactly that rule. For a member
# spanning workers the shape decides, that decision is made further down the
# vLLM path, and a guess here would put a number in a Mooncake descriptor the
# engine then contradicts — a pairing that fails at handshake. So there it
# stays unresolved, and the launch refuses rather than starting wrong.
_PARALLELISM_ALIASES: Dict[str, List[str]] = {
    "tensor_parallel_size": ["tensor-parallel-size", "tp", "tp-size"],
    "data_parallel_size": ["data-parallel-size", "dp", "dp-size"],
    "pipeline_parallel_size": ["pipeline-parallel-size", "pp", "pp-size"],
}

# The RoleSpec fields worth exposing as {{roles.<role>.<field>}}. The
# deployment-shaped overrides (env, backend_parameters, selectors) are
# deliberately not among them: they are lists and mappings, and a template can
# only ever want a scalar.
_ROLE_SCALAR_FIELDS = (
    "name",
    "replicas",
    "backend",
    "backend_version",
    "image_name",
    "cpu_only",
)


class PDInjectionError(Exception):
    """A PD injection that cannot be rendered into a correct launch.

    Raised, not degraded. A shared cache that fails to attach leaves a
    servable instance; a PD member that starts without its connector state
    joins a group it cannot hand KV to, returns 200s, and reports every NIXL
    metric as zero. The two failure postures have to differ.
    """


class PDInjection(BaseModel):
    """What one member's engine launch gains from its PD role."""

    env: Dict[str, str] = {}
    args: List[str] = []
    files: Dict[str, str] = {}
    """Container path -> contents; written by the serving script before the
    engine starts (Mooncake's transfer-engine config is read only from a file
    the engine is pointed at, so nothing else can carry it)."""

    def is_empty(self) -> bool:
        return not (self.env or self.args or self.files)


def render_pd_injection(
    model,
    instance,
    variables: Dict[str, object],
    peers: Optional[Dict[str, List[Tuple[str, int]]]] = None,
) -> Optional[PDInjection]:
    """Render this instance's role of `pd-modes.yaml` into env / args / files.

    `model` is the **unprojected** Model: it has to answer for every role, not
    just this one. `{{roles.decode.tensor_parallel_size}}` read off a
    projection would resolve decode's inherited parameters against *prefill's*
    effective ones, because a projection has already pushed the running role's
    overrides up to the Model level — a wrong number rather than a failure.
    The current role's own effective values are re-derived here, so nothing is
    lost by taking the unprojected spec.

    `variables` is the deployment-level context (`_template_variables()`),
    already carrying `{{worker_ip}}` / `{{port}}` / `{{role}}` / `{{group_id}}`
    and the two values only the worker can resolve, `{{net_device}}` and
    `{{runner_image}}`. Everything else is assembled here.

    Returns None when this is not a PD instance — no `disaggregation`, no
    role, an unknown mode, or a role the recipe injects nothing into
    (`custom`, and the router, whose command is assembled by the router path
    rather than merged into an engine's). The caller then takes today's path
    byte for byte.

    `peers` is only meaningful for a router. It is accepted so the signature
    is the one the router path will call, and ignored here: a router's role is
    not in `mode.roles`, so this returns None before reaching it. P and D
    never read across roles for an address (they rendezvous through the
    connector), which is why nothing else needs it.
    """
    disaggregation = getattr(model, "disaggregation", None)
    role_name = getattr(instance, "role", None)
    if disaggregation is None or not role_name:
        return None

    mode_name = _enum_value(getattr(disaggregation, "mode", None))
    mode = get_pd_mode(mode_name) if mode_name else None
    if mode is None:
        # The loader asserts the catalog and PDModeEnum agree at start-up, so
        # this is only reachable from a caller holding a mode this process
        # never loaded. Loud, because the symptom is silent aggregation.
        logger.warning(
            "PD mode '%s' is not in the catalog; instance %s (role '%s') starts "
            "with no connector configuration at all.",
            mode_name,
            getattr(instance, "name", None),
            role_name,
        )
        return None

    role = mode.role(role_name)
    if role is None:
        return None

    if peers:
        logger.debug(
            "Ignoring peer addresses for role '%s': peers are the router's, and "
            "the router's command is not assembled here.",
            role_name,
        )

    effective = role_effective_model(model, role_name)
    _reject_conflicting_kv_transfer_config(mode, role, effective, role_name)

    context = _pd_variables(model, mode, instance, effective, variables)
    where = f"PD mode '{mode.name}' role '{role_name}'"

    env = dict(render_values(role.env, context) or {})
    _apply_kv_lease_env(mode, env, where)

    args = [render(token, context, context=f"{where} args") for token in role.args]
    descriptor = _render_tree(role.connector, context, f"{where} connector")
    if descriptor:
        # Rendered whole, as this role's own connector. If an extended KV cache
        # also contributes one, `kv_transfer.compose_kv_transfer_config` folds
        # the two into a MultiConnector once the whole argv exists — the two
        # are complementary, and which of them a role should ask first is a
        # property of the role, not of this render.
        args += [
            KV_TRANSFER_CONFIG_FLAG,
            json.dumps(descriptor, separators=(",", ":")),
        ]

    files = {
        render(path, context, context=f"{where} file path"): render(
            content, context, context=f"{where} file {path}"
        )
        for path, content in (role.files or {}).items()
    }

    injection = PDInjection(env=env, args=args, files=files)
    _refuse_unrendered(injection, where)
    logger.info(
        "PD injection for role '%s' of mode '%s': %d env, %d args, %d files.",
        role_name,
        mode.name,
        len(injection.env),
        len(injection.args),
        len(injection.files),
    )
    return injection


_ANY_PLACEHOLDER = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_.]*\}\}")


def _refuse_unrendered(injection: "PDInjection", where: str) -> None:
    """Stop a launch carrying a placeholder that never got a value.

    The renderer deliberately leaves an unknown name in place rather than
    blanking it, because a blank is a plausible-looking wrong value. That is
    right for the render and wrong for the launch: the string then travels all
    the way into the engine.

    An *argument* survives that trip visibly — vLLM echoes its argv into the
    log, and the log scanner recognises the shape. An *environment variable*
    does not. `HCCL_SOCKET_IFNAME={{net_device}}` reaches HCCL as the literal
    name of an interface that does not exist, and what comes back is a
    transport that quietly never connects. Measured on 910B2, where the host
    has six candidate NICs and `derive_net_device` correctly refuses to guess
    between them — the refusal was right and its consequence was invisible.

    So the check moves to where both halves are already in hand and neither
    has left the process yet. Raising rather than warning, for the reason
    `PDInjectionError` exists: a PD member that starts without its connection
    state joins a group it cannot hand KV to and answers 200 to everything.
    """
    unrendered = []
    for name, value in sorted(injection.env.items()):
        for match in _ANY_PLACEHOLDER.finditer(str(value)):
            unrendered.append(f"{name}={match.group(0)}")
    for token in injection.args:
        for match in _ANY_PLACEHOLDER.finditer(str(token)):
            unrendered.append(match.group(0))

    if not unrendered:
        return

    # Named individually rather than counted: the operator has to know which
    # one to go and set, and `{{net_device}}` and `{{ports.kv_port}}` are
    # fixed in entirely different places.
    raise PDInjectionError(
        f"{where}: {len(unrendered)} configuration value(s) would reach the "
        f"engine unrendered — {', '.join(unrendered)}. A placeholder with no "
        "value is a port band that was not allocated, a network interface "
        "that could not be derived (set `kv_ifname` on the worker when the "
        "host has several), or a parallelism the role never declared."
    )


def _pd_variables(
    model,
    mode: PDMode,
    instance,
    effective,
    variables: Dict[str, object],
) -> Dict[str, object]:
    """The deployment context plus everything only the catalog's consumer can
    resolve. Flat and dotted, because that is the renderer's contract."""
    context: Dict[str, object] = dict(variables or {})
    context.update(_port_variables(instance))
    context.update(_disaggregation_variables(getattr(model, "disaggregation", None)))
    context.update(_kv_lease_variables(mode))
    context.update(_cross_role_variables(model, instance))
    # The running role's own fields, unprefixed: a declaration referring to
    # its own parallelism writes {{tensor_parallel_size}}, and only the
    # cross-role case needs the prefix.
    context.update(_role_fields(effective, getattr(instance, "role", None), instance))
    return context


def _port_variables(instance) -> Dict[str, object]:
    """`{{ports.<name>}}` is a band's base, `{{ports.<name>.count}}` its width.

    A band the allocator has not written yet is simply absent, so the
    placeholder survives into the launch with a warning beside it — which is
    what an unallocated connector port should look like, rather than a port 0
    the engine binds happily.
    """
    context: Dict[str, object] = {}
    for name, band in (getattr(instance, "named_ports", None) or {}).items():
        base = (
            band.get("base") if isinstance(band, dict) else getattr(band, "base", None)
        )
        count = (
            band.get("count", 1)
            if isinstance(band, dict)
            else getattr(band, "count", 1)
        )
        if base is None:
            continue
        context[f"ports.{name}"] = base
        context[f"ports.{name}.count"] = count
    return context


def _disaggregation_variables(disaggregation) -> Dict[str, object]:
    """`DisaggregationSpec`'s own fields, e.g. `{{kv_load_failure_policy}}`."""
    if disaggregation is None:
        return {}
    try:
        dumped = disaggregation.model_dump(mode="json")
    except Exception:  # pragma: no cover - a caller passing a stand-in
        return {}
    return {key: value for key, value in dumped.items() if value is not None}


def _kv_lease_variables(mode: PDMode) -> Dict[str, object]:
    """The connector's KV lease window, keyed by the name it is configured
    under (`{{kv_lease_duration}}` for NIXL).

    GPUStack's own default wins over the engine's when the catalog declares
    one: the point of the registry is one window across connectors instead of
    inheriting a 30s-to-480s spread from whichever connector a mode happens
    to use.
    """
    lease = mode.kv_lease
    if lease is None or not lease.settable or not lease.param:
        return {}
    value = lease.gpustack_default
    if value is None:
        value = lease.engine_default
    if value is None:
        return {}
    return {lease.param: value}


def _apply_kv_lease_env(mode: PDMode, env: Dict[str, str], where: str) -> None:
    """Set the lease window when the connector takes it as an env var.

    `inject_to: env` is otherwise a declaration nothing acts on: unlike
    `connector_extra_config`, no role declaration mentions the variable — an
    env-configured window would silently keep the engine's default (8 minutes
    for Mooncake, 16x NIXL's, with no Prometheus counter to notice it). A
    declaration that does mention it, or a per-model env, still wins.
    """
    lease = mode.kv_lease
    if lease is None or not lease.settable or not lease.param:
        return
    if lease.inject_to != PDKVLeaseTargetEnum.ENV:
        return
    if lease.param in env:
        return
    value = lease.gpustack_default
    if value is None:
        return
    logger.debug("%s: setting KV lease window %s=%s", where, lease.param, value)
    env[lease.param] = str(value)


def _cross_role_variables(model, instance=None) -> Dict[str, object]:
    """`{{roles.<role>.<field>}}` — the coupling Mooncake needs and NIXL does
    not: prefill's connector config carries decode's parallelism and vice
    versa. Each role is resolved through its own projection, so a role that
    overrides nothing reads the Model-level value rather than the running
    role's."""
    context: Dict[str, object] = {}
    for role in getattr(model, "roles", None) or []:
        name = getattr(role, "name", None)
        if not name:
            continue
        projected = role_effective_model(model, name)
        for field, value in _role_fields(projected, name, instance).items():
            context[f"roles.{name}.{field}"] = value
    return context


def _role_fields(
    effective, role_name: Optional[str], instance=None
) -> Dict[str, object]:
    """One role's referenceable fields, read off its effective Model."""
    fields: Dict[str, object] = {}
    role = None
    for candidate in getattr(effective, "roles", None) or []:
        if getattr(candidate, "name", None) == role_name:
            role = candidate
            break
    for field in _ROLE_SCALAR_FIELDS:
        value = getattr(role, field, None)
        if value is None:
            value = getattr(effective, field, None)
        if value is not None:
            fields[field] = _enum_value(value)

    parameters = getattr(effective, "backend_parameters", None) or []
    for field, aliases in _PARALLELISM_ALIASES.items():
        try:
            value = find_int_parameter(parameters, aliases)
        except Exception:  # pragma: no cover - malformed parameters
            value = None
        if value is not None:
            fields[field] = value

    fields.update(_implicit_parallelism(fields, instance, role_name))
    return fields


def _implicit_parallelism(
    declared: Dict[str, object], instance, role_name: Optional[str]
) -> Dict[str, object]:
    """The parallelism a single-worker member has whether or not it says so.

    A role that writes no `--tensor-parallel-size` is not a role with an
    unknown one: for a member on a single worker the engine path derives it
    from the cards the member was given, and a member with no data parallelism
    runs at one. Both are the rule the engine will apply, read at a point that
    already knows the inputs — not a default chosen here.

    Why this is needed at all: the Ascend recipe's connector config carries
    *both* sides' parallelism, so a 1P1D whose roles declare none left
    `{{roles.prefill.data_parallel_size}}` in the launch. Measured on 910B2 —
    every deployment failed until the user wrote out parameters that only
    restated what the engine was going to do anyway.

    Deliberately silent for a member spanning workers. There the shape decides
    dp and dpl, that decision happens further down the vLLM path, and guessing
    here would put a number in a Mooncake descriptor that the engine then
    contradicts — a pairing that fails at handshake, which is worse than the
    launch refusing.
    """
    if instance is None or _spans_workers(instance):
        return {}

    out: Dict[str, object] = {}
    if "tensor_parallel_size" not in declared:
        cards = len(getattr(instance, "gpu_indexes", None) or [])
        if cards:
            out["tensor_parallel_size"] = cards
    if "data_parallel_size" not in declared:
        out["data_parallel_size"] = 1
    return out


def _spans_workers(instance) -> bool:
    servers = getattr(instance, "distributed_servers", None)
    return bool(servers and getattr(servers, "subordinate_workers", None))


def _render_tree(value: Any, variables: Dict[str, object], where: str) -> Any:
    """Render a connector descriptor in place, keeping its structure.

    The descriptor is JSON the engine parses, not a string GPUStack pastes, so
    a value that is nothing but a placeholder comes back as the type its
    source had — `kv_port` and `tp_size` are numbers in the config Ascend was
    measured running, and a quoted "8" is a different document.
    """
    if isinstance(value, str):
        rendered = render(value, variables, context=where)
        return _coerce(value, rendered)
    if isinstance(value, dict):
        return {
            key: _render_tree(item, variables, f"{where}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_render_tree(item, variables, where) for item in value]
    return value


def _coerce(template: str, rendered: Optional[str]) -> Any:
    """Give a whole-placeholder value back its source's type.

    Only when something was actually substituted: a placeholder left
    unresolved stays the literal it is, which is the whole point of leaving it
    there.
    """
    if rendered is None or rendered == template:
        return rendered
    if not _WHOLE_PLACEHOLDER.match(template):
        return rendered
    try:
        return int(rendered)
    except (TypeError, ValueError):
        return rendered


def _reject_conflicting_kv_transfer_config(
    mode: PDMode,
    role: PDModeRole,
    effective,
    role_name: str,
) -> None:
    """Refuse a launch where two sources would write `--kv-transfer-config`.

    vLLM reads the flag once, and every source of it — this recipe's
    connector, an extended KV cache, a hand-written parameter — expands into
    one whole JSON document, not a fragment. Passing two lets argparse keep
    the last, so the group would come up with either PD or the cache silently
    absent while the UI reports both attached. The two user-facing paths are
    therefore mutually exclusive until one assembler owns the flag: pick a
    recipe and let GPUStack write the connector state, or pick `custom` and
    write all of it yourself.

    Only recipes that carry a connector descriptor are affected. SGLang's
    disaggregation is configured through its own flags, so a SGLang PD role
    and a shared cache do not collide.
    """
    if not role.connector:
        return

    parameters = flatten_to_argv(getattr(effective, "backend_parameters", None) or [])
    if any(
        token == KV_TRANSFER_CONFIG_FLAG
        or token.startswith(KV_TRANSFER_CONFIG_FLAG + "=")
        for token in parameters
    ):
        raise PDInjectionError(
            f"Role '{role_name}' sets {KV_TRANSFER_CONFIG_FLAG} in its backend "
            f"parameters while PD mode '{mode.name}' injects its own connector "
            f"into the same flag, which vLLM reads only once. Remove the "
            f"parameter, or switch the deployment to PD mode 'custom', which "
            f"injects no connection state and leaves the flag to you."
        )


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value
