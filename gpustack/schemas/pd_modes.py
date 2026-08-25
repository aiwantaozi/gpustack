import re
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, model_validator

_PLACEHOLDER_OCCURRENCE = re.compile(r"\{\{.*?\}\}")
"""Any {{...}} run, valid or not — the validator's discovery pass."""

_PLACEHOLDER = re.compile(
    r"^\{\{[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\}\}$"
)
"""A well-formed placeholder: dotted identifiers, no inner spaces. The
renderer's pattern takes no spaces either, so "{{ worker_ip }}" would reach
the container verbatim (M0: a literal "{{worker_ip}}" in
VLLM_NIXL_SIDE_CHANNEL_HOST became `ZMQError: No such device`). Rejecting the
spaced form at load time is cheaper than debugging it on a worker."""


def iter_placeholders(value: Any) -> Iterator[str]:
    """Yield every {{...}} occurrence in a declaration subtree.

    Templates are nested freely (a connector descriptor carries dicts of
    dicts), so the walk is structural rather than per-field.
    """
    if isinstance(value, str):
        for match in _PLACEHOLDER_OCCURRENCE.finditer(value):
            yield match.group(0)
    elif isinstance(value, BaseModel):
        yield from iter_placeholders(value.model_dump())
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_placeholders(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_placeholders(item)


class PDModeSourceEnum(str, Enum):
    BUILT_IN = "built_in"
    COMMUNITY = "community"
    PARTNER = "partner"


class PDInjectTargetEnum(str, Enum):
    """Where an injected value lands in the engine's launch.

    The three targets are the ``injection`` vocabulary of
    ``cache-providers.yaml``, word for word. They are not
    interchangeable per engine: NIXL's side-channel port is an env var,
    Mooncake's kv_port is a field of the connector descriptor (rendered
    into ``--kv-transfer-config``, hence ``args``), and Mooncake's
    transfer-engine config is read *only* from the JSON file
    MOONCAKE_CONFIG_PATH points at, so nothing but ``files`` can carry it.
    """

    ENV = "env"
    ARGS = "args"
    FILES = "files"


class PDPortScopeEnum(str, Enum):
    """Who a named port band belongs to (RBG's PodScoped / RoleScoped).

    hostNetwork is the locked-in default shape, so every band shipped today
    is per-instance: two members of one role on one host must not share a
    band or they collide on bind.
    """

    INSTANCE = "instance"
    ROLE = "role"


class PDPeerStyleEnum(str, Enum):
    """How a router's command line takes the group's peer addresses.

    A new engine that reuses one of these styles is YAML only. A genuinely
    new wire style (e.g. TRT-LLM's delegate protocol) needs a renderer
    branch, which is why an unknown value fails at load time instead of
    being accepted and silently rendered wrong.
    """

    REPEATED_FLAG = "repeated_flag"
    """--prefill http://a:1 --prefill http://b:2 (vLLM router, SGLang router)."""

    PARALLEL_LISTS = "parallel_lists"
    """--prefiller-hosts a b --prefiller-ports 1 2 (vllm-ascend's proxy example)."""

    USER_PROVIDED = "user_provided"
    """The user supplies image and command on the role itself."""


class PDRouterProtocolEnum(str, Enum):
    TWO_HOP = "two_hop"
    """Router calls prefill, then carries kv_transfer_params to decode."""

    SAME_DISPATCH = "same_dispatch"
    """Router dispatches to both sides and the engines rendezvous over
    their own bootstrap handshake."""

    USER_PROVIDED = "user_provided"


class PDKVLeaseTargetEnum(str, Enum):
    """Where a connector's KV lease / abort window is configured.

    Deliberately a different vocabulary from ``PDInjectTargetEnum``: the
    window is a connector-level knob, and per connector it is a field of
    ``kv_connector_extra_config``, an env var, or nothing at all.
    """

    CONNECTOR_EXTRA_CONFIG = "connector_extra_config"
    ENV = "env"
    ARGS = "args"
    FILES = "files"
    NONE = "none"
    """Not configurable — a hardcoded engine constant, or absent entirely.
    Declared anyway so the window is on the record."""


class PDKVLease(BaseModel):
    """One connector's KV lease / abort window (D31).

    The parameter name, the default and the injection target all differ per
    connector, and the windows differ by two orders of magnitude (NIXL 30s,
    Mooncake 480s, MoRIIO 3600s), so this cannot be a platform-wide
    constant. Whether the expiry is observable differs too: NIXL exports a
    Prometheus counter, Mooncake exports none.
    """

    connector: str
    """Connector identity, and the key modes reference. Unqualified ids are
    vLLM ``kv_connector`` implementations; ``sglang-*`` ids are SGLang
    ``--disaggregation-transfer-backend`` values, whose KV lifecycle is a
    different mechanism (bootstrap timeout, not a lease)."""

    param: Optional[str] = None
    """The knob's name: a connector-config field, or an env var. None when
    the connector has no knob at all."""

    inject_to: PDKVLeaseTargetEnum = PDKVLeaseTargetEnum.NONE

    settable: bool = True
    """False for a hardcoded constant. GPUStack injects nothing and the
    declaration is accounting only."""

    engine_default: Optional[int] = None
    """The engine's own default, in seconds. None when it is not in the
    measured record — nothing is injected on a guess."""

    gpustack_default: Optional[int] = None
    """The platform's default, in seconds: one window across connectors
    instead of inheriting a 16x spread. None leaves the engine default
    alone."""

    expired_metric: Optional[str] = None
    """Prometheus metric counting expiries, if the connector exports one.
    None means the event is invisible to Prometheus and only diagnosable
    from the engine log."""

    description: Optional[str] = None

    @model_validator(mode="after")
    def check_target_agrees_with_settable(self) -> "PDKVLease":
        if self.settable and self.inject_to == PDKVLeaseTargetEnum.NONE:
            raise ValueError(
                f"kv_lease '{self.connector}' is settable but declares no "
                "injection target"
            )
        if self.settable and not self.param:
            raise ValueError(
                f"kv_lease '{self.connector}' is settable but names no parameter"
            )
        if not self.settable and self.inject_to != PDKVLeaseTargetEnum.NONE:
            raise ValueError(
                f"kv_lease '{self.connector}' is not settable, so its "
                f"inject_to must be 'none', not '{self.inject_to.value}'"
            )
        if not self.settable and self.gpustack_default is not None:
            raise ValueError(
                f"kv_lease '{self.connector}' is not settable, so a "
                "gpustack_default would never be applied"
            )
        return self


class PDTransferMetrics(BaseModel):
    """The engine-side counters one connector exports about KV transfer.

    Declared per connector rather than hardcoded because the names, and
    whether they exist at all, are the connector's business: NIXL exports
    fifteen counters, Mooncake exports none, and SGLang's transfer backends
    were never measured. A mode whose entry is all-null is a mode where PD
    effectiveness is not observable from Prometheus, and saying that out
    loud is the point — the alternative is a ratio that silently reads
    zero because the metric was never there.
    """

    connector: str
    """The key modes resolve through, the same id ``kv_leases`` uses."""

    read_from_role: str = "decode"
    """🔴 Which side of the pair owns these counters.

    NIXL is *pull-based*: decode reads from prefill, so decode is where a
    transfer is counted. Measured on a working 1P1D — prefill's
    ``nixl_xfer_time_seconds_count`` stayed at 0.0 for the whole run while
    decode's was 1.0. Reading the prefill side therefore reports a healthy
    pair as "no KV ever moved", i.e. reports the exact failure this metric
    exists to detect, on a deployment that is fine.

    A declaration and not a constant because push-based connectors exist;
    for one of those this reads ``prefill`` and no code changes."""

    xfer_count: Optional[str] = None
    """Cumulative count of completed KV transfers — the numerator of the
    PD-effectiveness ratio. Spelled as it appears in the exposition
    (a histogram's ``_count`` sample), not as the family name."""

    xfer_seconds: Optional[str] = None
    """Cumulative seconds spent transferring. Paired with ``xfer_count`` it
    gives a per-window transfer rate, which is what the degradation check
    compares against the group's own baseline."""

    xfer_bytes: Optional[str] = None
    """Cumulative bytes transferred, if the connector exports it. When it
    does the rate is a true bandwidth; when it does not the check falls
    back to transfers per second of transfer time, which is a rate in a
    different unit and therefore only ever comparable to a baseline taken
    in the same unit."""

    failed_transfers: Optional[str] = None

    sample_labels: Optional[Dict[str, str]] = None
    """Labels a sample must carry to be counted, for engines that put every
    stage of a request in ONE family and separate them by label.

    SGLang is why this exists. It exports a single
    ``sglang:per_stage_req_latency_seconds`` histogram and distinguishes
    ``stage="decode_bootstrap"`` from ``stage="decode_transferred"`` — so
    summing the family without a selector counts every stage of every request
    and yields a number several times the transfer count, which as a numerator
    reads as an effectiveness ratio well above 1. Null for connectors like NIXL
    that give each counter its own name."""

    min_expected_rate: Optional[float] = None
    """Coarse floor for the first-deployment case the baseline method
    cannot see (a group that was already degraded when its baseline was
    taken). Null everywhere on purpose: a floor is a calibration against
    real hardware, and the measured spread — 94% of line rate on 2.5GbE
    versus 9% on 910B2 RoCE — is exactly why an invented number would
    either alarm forever on one platform or never on the other."""

    @model_validator(mode="after")
    def check_rate_inputs(self) -> "PDTransferMetrics":
        if self.xfer_seconds and not self.xfer_count:
            raise ValueError(
                f"transfer metrics for '{self.connector}' declare "
                "xfer_seconds without xfer_count; a rate needs both"
            )
        if self.min_expected_rate is not None and not self.xfer_count:
            raise ValueError(
                f"transfer metrics for '{self.connector}' declare a floor but "
                "no counter to measure against it"
            )
        return self

    @property
    def observable(self) -> bool:
        """Whether anything at all can be read from this connector."""
        return bool(self.xfer_count or self.xfer_bytes or self.failed_transfers)


class PDPortSpec(BaseModel):
    """A named port band a role needs allocated.

    ``[kv_side_channel]`` is shorthand for
    ``{name: kv_side_channel, count: 1, inject_to: env}``.
    """

    name: str

    count: Union[int, str] = 1
    """Band width: an integer, or a single placeholder resolved at
    allocation time. The width is decided by the connector, not by a
    platform formula — Mooncake's kv_port is a base address and the connector
    binds one port per *worker rank*, so the band is the member's card count
    (``{{accelerator_count}}``; measured TP8/DP1 -> 41100-41107 and DP2xTP2 ->
    20001-20004), while NIXL's side channel is offset per DP index instead."""

    inject_to: PDInjectTargetEnum = PDInjectTargetEnum.ENV
    """Which injection channel carries the band's base. Cross-checked at
    load time against where {{ports.<name>}} actually appears, so a wrong
    declaration cannot sit there looking plausible."""

    scope: PDPortScopeEnum = PDPortScopeEnum.INSTANCE

    @model_validator(mode="before")
    @classmethod
    def expand_shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value

    @model_validator(mode="after")
    def check_count(self) -> "PDPortSpec":
        if isinstance(self.count, str):
            if not _PLACEHOLDER.match(self.count):
                raise ValueError(
                    f"port '{self.name}' count '{self.count}' is neither an "
                    "integer nor a single placeholder"
                )
        elif self.count < 1:
            raise ValueError(f"port '{self.name}' count must be >= 1")
        return self


class PDModeRole(BaseModel):
    """What one role of a mode needs allocated and injected.

    ``connector`` is the only part that is not a cache-provider-shaped
    injection: it stays a structured descriptor because that is already the
    shape the extended-KV-cache assembler consumes, so the PD side needs no
    string surgery. Everything in it renders into ``--kv-transfer-config``,
    which is why a port carried in the descriptor declares
    ``inject_to: args``.
    """

    ports: List[PDPortSpec] = []
    connector: Dict[str, Any] = {}
    env: Dict[str, str] = {}
    args: List[str] = []
    files: Dict[str, str] = {}

    @model_validator(mode="after")
    def check_port_names_unique(self) -> "PDModeRole":
        names = [port.name for port in self.ports]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate named ports: {duplicates}")
        return self

    def channel(self, target: PDInjectTargetEnum) -> Any:
        """The declaration subtree an inject target renders into. ``args``
        covers the connector descriptor as well: it is rendered into one."""
        if target == PDInjectTargetEnum.ENV:
            return self.env
        if target == PDInjectTargetEnum.FILES:
            return self.files
        return [self.args, self.connector]

    @property
    def port_names(self) -> List[str]:
        return [port.name for port in self.ports]


class PDRouterPeers(BaseModel):
    """How the group's peer addresses reach the router's command line."""

    style: PDPeerStyleEnum

    prefill: Dict[str, str] = {}
    """Style-specific rendering: repeated_flag takes {flag, value} with
    {{peer.ip}} / {{peer.port}} per peer; parallel_lists takes {host_flag,
    port_flag}."""

    decode: Dict[str, str] = {}


class PDRouterCapabilities(BaseModel):
    """What the router actually serves.

    Every field defaults to False: an undeclared endpoint must be treated
    as absent, not assumed present. vllm-ascend's proxy example has neither
    /metrics nor /v1/models, and polling them produced a ~1/s 404 storm in
    the router log plus a permanent false alarm.
    """

    metrics: bool = False
    """Serves a Prometheus exposition. False means health checks degrade to
    process liveness plus a port probe, and the PD-effectiveness ratio
    loses its denominator."""

    models_endpoint: bool = False
    """Serves /v1/models."""

    kv_expired_metric: bool = False
    """The engine side of this mode exports a KV-lease-expiry counter.
    Mirrors ``kv_lease.expired_metric``; the loader asserts they agree."""


class PDRouterRequestMetrics(BaseModel):
    """The router-side counters that give the PD-effectiveness ratio its
    denominator.

    The denominator is *not* counted by GPUStack. vllm-router already
    exports per-worker request counters, and per-worker is the shape that
    matters: a group-wide ratio says "something is wrong somewhere", a
    per-worker one says which decode stopped pulling.

    Left empty means no denominator, and that is a first-class outcome
    rather than a gap to paper over — vllm-ascend's proxy serves no
    /metrics at all, and SGLang's gateway is the same codebase as
    vllm-router but its counter names were never verified here. Guessing
    them would produce a ratio that reads zero because the name was wrong,
    which is indistinguishable from the failure being measured.
    """

    prefill_requests: Optional[str] = None
    decode_requests: Optional[str] = None

    worker_label: str = "worker"
    """Label carrying the peer the request was dispatched to. Measured, its
    value is the whole peer URL the router was launched with
    (``http://192.168.50.15:40005``) rather than a worker name or id, so a
    member is matched on host:port and not on the whole string."""

    port_band: Optional[str] = None
    """Named port band the exposition is served on. None means the router's
    own HTTP port.

    Not the same port for vllm-router: its API and its Prometheus endpoint
    are separate listeners, and the second one is a band GPUStack allocates
    because upstream's default (29000) is fixed and two routers on a host
    would collide. Scraping the API port instead returns 404, which reads as
    "no denominator" — a check that silently stops working."""

    total_requests: Optional[str] = None
    """Group-level request counter, aggregated by route instead of by peer.
    The fallback denominator: coarser, but it still answers "did the router
    route anything at all", which is the question a ratio of zero is
    meaningless without."""

    @property
    def available(self) -> bool:
        return bool(
            self.prefill_requests or self.decode_requests or self.total_requests
        )


class PDRouter(BaseModel):
    """The router role of a mode. There is no universal router — the
    catalog format is what generalizes, not the binary."""

    protocol: PDRouterProtocolEnum

    image: Optional[str] = None
    ports: List[PDPortSpec] = []
    command: List[str] = []
    peers: Optional[PDRouterPeers] = None
    capabilities: PDRouterCapabilities = PDRouterCapabilities()
    request_metrics: PDRouterRequestMetrics = PDRouterRequestMetrics()

    health_path: Optional[str] = None
    """None means the router serves no health endpoint, so readiness falls
    back to process liveness."""

    @model_validator(mode="after")
    def check_user_provided(self) -> "PDRouter":
        if self.request_metrics.available and not self.capabilities.metrics:
            # Two spellings of one fact, the same trap `kv_expired_metric`
            # guards: a name declared behind `metrics: false` would be
            # scraped from an endpoint that does not exist.
            raise ValueError(
                "router declares request metric names but capabilities.metrics "
                "is false, so nothing would ever scrape them"
            )
        user_provided = self.protocol == PDRouterProtocolEnum.USER_PROVIDED
        if user_provided and (self.command or self.image or self.ports):
            raise ValueError(
                "a user_provided router takes its image, command and ports "
                "from the role spec, so the catalog must not declare them"
            )
        if not user_provided and not self.command:
            raise ValueError(f"a {self.protocol.value} router needs a command")
        if not user_provided and self.peers is None:
            raise ValueError(
                f"a {self.protocol.value} router needs a peers declaration"
            )
        names = [port.name for port in self.ports]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate named ports on the router: {duplicates}")
        return self

    def channel(self, target: PDInjectTargetEnum) -> Any:
        """A router is a command line: env and files are not injection
        channels for it, and the peer templates are part of the args."""
        if target in (PDInjectTargetEnum.ENV, PDInjectTargetEnum.FILES):
            return {}
        return [self.command, self.peers]

    @property
    def port_names(self) -> List[str]:
        return [port.name for port in self.ports]


class PDMode(BaseModel):
    """One disaggregation recipe: engine, KV connector and router.

    The user picks one of these from a single dropdown; every
    connection-state parameter (connector, ports, handshake variables)
    comes from here and none of it reaches the form.
    """

    name: str
    """Must equal a ``PDModeEnum`` value verbatim — the catalog is looked up
    by it. The loader asserts the two sets are equal at load time."""

    display_name: Optional[str] = None
    source: PDModeSourceEnum = PDModeSourceEnum.BUILT_IN
    description: Optional[str] = None

    backends: List[str] = []
    """Inference backends this recipe may be injected into, in
    ``BackendEnum`` spelling. A recipe expands into one engine's connector
    config, so a role on another engine would be handed configuration it
    cannot read. Empty means unconstrained (only ``custom``, which injects
    nothing). ``PD_MODE_BACKENDS`` must agree; the loader asserts it."""

    backend_versions: Optional[str] = None
    """Human-readable compatible version range. Informational, like a cache
    provider's ``versions``."""

    runtime: Optional[str] = None
    """Accelerator runtime this recipe requires (e.g. "ascend" for CANN
    plus the Ascend docker runtime). None means no constraint."""

    roles: Dict[str, PDModeRole] = {}
    """Engine roles by name. Empty injects nothing (``custom``). Role names
    are deliberately unconstrained here: the API validation layer decides
    which roles are admissible, so opening up encoder / draft later stays a
    validation change, not a catalog schema change."""

    router: Optional[PDRouter] = None

    kv_lease: Optional[PDKVLease] = None
    """Resolved by the loader from the catalog's per-connector registry.
    None means GPUStack configures no window (``custom``)."""

    transfer_metrics: Optional[PDTransferMetrics] = None
    """Resolved by the loader from the same connector id ``kv_lease``
    names — the connector identifies the transport, and the transport is
    what decides both the lease window and the counters. None means the
    mode has no connector GPUStack knows (``custom``), so PD effectiveness
    is not measurable for it."""

    def role(self, name: str) -> Optional[PDModeRole]:
        return self.roles.get(name)

    @model_validator(mode="after")
    def check_templates(self) -> "PDMode":
        """Reject a declaration whose placeholders cannot resolve.

        Structural only, on purpose: there is no allowlist of placeholder
        names, because a new engine needing a new variable must stay a YAML
        change. What is checked is that every placeholder is well formed,
        that {{ports.<name>}} names a band the same declaration allocates
        and appears in the channel the band declares, and that
        {{roles.<role>...}} names a role the mode declares.
        """
        declared_roles = set(self.roles)
        allocated = {
            port.name for holder in self._holders_only() for port in holder.ports
        }
        for scope, holder in self._holders():
            for occurrence in iter_placeholders(holder):
                if not _PLACEHOLDER.match(occurrence):
                    raise ValueError(
                        f"{scope}: malformed placeholder {occurrence} "
                        "(dotted identifiers only, no inner spaces)"
                    )
                parts = occurrence[2:-2].split(".")
                if parts[0] == "roles":
                    if len(parts) != 3:
                        raise ValueError(
                            f"{scope}: cross-role reference {occurrence} must be "
                            "{{roles.<role>.<field>}}"
                        )
                    if parts[1] not in declared_roles:
                        raise ValueError(
                            f"{scope}: cross-role reference {occurrence} names "
                            f"undeclared role '{parts[1]}'"
                        )
                elif parts[0] == "ports":
                    if len(parts) not in (2, 3) or (
                        len(parts) == 3 and parts[2] != "count"
                    ):
                        raise ValueError(
                            f"{scope}: port reference {occurrence} must be "
                            "{{ports.<name>}} or {{ports.<name>.count}}"
                        )
                    # A router reads another role's band (SGLang's router
                    # takes prefill's bootstrap port as a positional
                    # argument), so the band only has to exist somewhere in
                    # the mode.
                    if parts[1] not in allocated:
                        raise ValueError(
                            f"{scope}: {occurrence} references a port band "
                            "nothing in this mode allocates"
                        )
            # And every allocated band must be consumed where it says it is:
            # an unreferenced band is a port taken out of a 64-port pool for
            # nothing, and a band referenced from the wrong channel is the
            # NIXL-vs-Mooncake mistake this schema exists to prevent.
            for spec in holder.ports:
                references = {
                    "{{ports." + spec.name + "}}",
                    "{{ports." + spec.name + ".count}}",
                }
                found = set(iter_placeholders(holder.channel(spec.inject_to)))
                if not references & found:
                    raise ValueError(
                        f"{scope}: port band '{spec.name}' declares inject_to "
                        f"'{spec.inject_to.value}' but nothing there references it"
                    )
        return self

    def _holders(self):
        for name, role in self.roles.items():
            yield f"role '{name}'", role
        if self.router is not None:
            yield "router", self.router

    def _holders_only(self):
        return [holder for _, holder in self._holders()]


class PDModeCatalog(BaseModel):
    """The whole catalog document.

    ``kv_leases`` is keyed by connector and is the source each mode's
    resolved ``kv_lease`` came from. It survives resolution because an entry
    can legitimately have no mode: MoRIIO's window is on the record even
    though no shipped recipe uses it and GPUStack cannot change it.
    """

    kv_leases: Dict[str, PDKVLease] = {}
    kv_transfer_metrics: Dict[str, PDTransferMetrics] = {}
    """Keyed by the same connector id as ``kv_leases``, and required to
    cover every connector that appears there: an entry whose fields are all
    null declares "this connector exports nothing", which is a measured
    fact about Mooncake, while a *missing* entry would be an oversight that
    reads identically at runtime."""

    modes: List[PDMode] = []

    def mode(self, name: str) -> Optional[PDMode]:
        for mode in self.modes:
            if mode.name.lower() == (name or "").lower():
                return mode
        return None
