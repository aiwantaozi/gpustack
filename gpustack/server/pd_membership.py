"""Telling a PD router who its members are, over HTTP.

Two facts decide the shape of everything here, and both were measured on a
live 1P1D rather than read from docs:

**1. Without `--enable-igw` there is no membership API at all.** `POST /workers`
returns 400 `"PD router requires specific add_prefill_server or
add_decode_server methods"` — the single-router path drops `worker_type` before
the PD router sees it. Confirmed in upstream source: the non-igw branch builds
the router through `create_router()`, which reads `prefill_urls`/`decode_urls`
off the command line.

**2. With `--enable-igw` the command line stops working.** 🔴 This is the fact
that shapes the module. The igw branch builds the PD router with hard-coded
empty worker lists — vLLM's `create_vllm_pd_router(&[], &[], ...)` with the
upstream comment `// Empty worker list - workers added later`, and SGLang's
`create_pd_router(None, None, ...)`. The CLI peers are *never passed*, so a
request before registration gets 503 `"No available workers"`.

⇒ Registration is not an optimisation on top of a working group. Under igw it
is the only way members get in, and every router start passes through a window
where the process is up and serves nothing. That window is structural, not a
failure mode — which is why the group must not be reported servable until this
module says the registry matches.

⚠️ **The fallback is an operator action, not something this module can take.**
Going back to command-line peers means removing `--enable-igw` from the mode's
router command — the controller cannot un-launch a flag on a running process,
and restarting the router just repeats the same registration against the same
endpoint. So a persistent failure is *stated* with the way out named, and the
group stays PARTIAL rather than claiming to serve.
"""

import asyncio
import logging
from urllib.parse import quote
from typing import Dict, List, Optional, Sequence, Tuple

import aiohttp

from gpustack.schemas.models import Model, ModelInstance, RoleNameEnum
from gpustack.schemas.pd_modes import PDMembershipAPI, PDMode

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10.0
_RECONCILE_DEADLINE_SECONDS = 30.0


class MembershipOutcome:
    """What one reconcile attempt achieved, in the terms the caller acts on.

    `ok` gates the group's servability, so it is deliberately pessimistic: it
    is True only when the router's own read-back agrees with the members we
    believe exist. An accepted `POST` is not enough — upstream admits a peer
    only after probing it, and drops it silently on timeout with a default
    window tuned for small models. Without the read-back, a scale-out that
    quietly failed is indistinguishable from one that worked.
    """

    def __init__(
        self,
        ok: bool,
        reason: Optional[str] = None,
        registered: Optional[Sequence[str]] = None,
    ):
        self.ok = ok
        self.reason = reason
        self.registered = list(registered or [])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MembershipOutcome(ok={self.ok}, reason={self.reason!r})"


def member_url(instance: ModelInstance) -> Optional[str]:
    """The address the router knows a member by.

    `host:port` of the serving listener, which is what upstream stores and
    what `DELETE /workers/{url}` matches on — measured: the `worker` label on
    the router's own request counters carries the whole URL it was launched
    with, not a worker name or id.
    """
    if not instance.worker_ip or not instance.port:
        return None
    return f"http://{instance.worker_ip}:{instance.port}"


def desired_members(model: Model, instances: Sequence[ModelInstance]) -> Dict[str, str]:
    """`{url: role}` for every member that should be in the router's registry.

    Only RUNNING GPU roles. The router is excluded because it is not its own
    upstream, and a member that is not RUNNING has nothing listening — adding
    it would be admitted only to fail the router's own probe, and upstream
    drops such a peer silently.
    """
    from gpustack.schemas.models import ModelInstanceStateEnum

    members: Dict[str, str] = {}
    for instance in instances:
        if instance.role == RoleNameEnum.ROUTER.value or not instance.role:
            continue
        if instance.state != ModelInstanceStateEnum.RUNNING:
            continue
        url = member_url(instance)
        if url:
            members[url] = instance.role
    return members


def _endpoint(
    api: PDMembershipAPI, base: str, spec: Optional[str]
) -> Optional[Tuple[str, str]]:
    """A declared `"METHOD /path"` -> `(method, absolute url)`.

    The method is part of the declaration because the two shapes upstream
    ships differ in it: a REST resource (`POST /workers`) versus an action
    endpoint (`POST /instances/add`).
    """
    if not spec:
        return None
    parts = spec.split(None, 1)
    if len(parts) != 2:
        logger.warning("Malformed membership endpoint %r; ignoring", spec)
        return None
    method, path = parts[0].upper(), parts[1]
    return method, f"{base.rstrip('/')}{path}"


def _body(api: PDMembershipAPI, url: str, role: str, model_name: str) -> dict:
    """The request body for one member.

    `role_field` is named in the declaration rather than assumed, because it
    is the field the single-router path drops — so a consumer has to know what
    to look for in the read-back to tell "accepted" from "actually joined".
    """
    payload: Dict[str, object] = {}
    for key, template in (api.body or {}).items():
        payload[key] = (
            str(template)
            .replace("{{peer.url}}", url)
            .replace("{{model_name}}", model_name)
        )
    payload["url"] = url
    if api.role_field:
        payload[api.role_field] = (api.role_values or {}).get(role, role)
    return payload


async def _read_registry(
    client: aiohttp.ClientSession, api: PDMembershipAPI, base: str
) -> Optional[Dict[str, str]]:
    """`{url: role}` as the router itself reports it, or None if unreadable.

    None and empty are different answers and must stay so: an unreadable
    registry means "we cannot tell", which keeps the group PARTIAL; an empty
    one means the router genuinely has no members, which is a fact to act on.
    """
    resolved = _endpoint(api, base, api.probe)
    if resolved is None:
        return None
    method, url = resolved
    try:
        async with client.request(
            method, url, timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)
        ) as response:
            if response.status != 200:
                return None
            payload = await response.json()
    except Exception as e:
        logger.debug("Could not read the router's registry at %s: %s", url, e)
        return None

    rows = payload.get("workers") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return None
    registry: Dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        peer = row.get("url")
        if not peer:
            continue
        role = row.get(api.role_field) if api.role_field else None
        registry[str(peer)] = str(role or "")
    return registry


async def _add_missing(
    client: aiohttp.ClientSession,
    api: PDMembershipAPI,
    base: str,
    model_name: str,
    wanted: Dict[str, str],
    current: Dict[str, str],
) -> None:
    """Add every wanted member the router does not already have.

    Failures are logged rather than raised: the read-back at the end of
    `reconcile` is what decides the outcome, so a refusal here shows up there
    as a missing member with the router's own words attached. Raising would
    skip the members after it for no gain.
    """
    add = _endpoint(api, base, api.add)
    if add is None:
        return
    method, endpoint = add
    for url, role in wanted.items():
        if url in current:
            continue
        try:
            async with client.request(
                method,
                endpoint,
                json=_body(api, url, role, model_name),
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
            ) as response:
                if response.status >= 400:
                    body = (await response.text())[:200]
                    logger.warning("Router refused member %s (%s): %s", url, role, body)
        except Exception as e:
            logger.warning("Could not add member %s: %s", url, e)


async def _remove_stale(
    client: aiohttp.ClientSession,
    api: PDMembershipAPI,
    base: str,
    wanted: Dict[str, str],
    current: Dict[str, str],
) -> List[str]:
    """Drop members the group no longer has, and report what would not go.

    A failure here is NOT fatal: upstream's removal gate is global -- it waits
    for the in-flight count to reach zero -- so a member can legitimately
    linger while traffic continues, and it takes no new requests meanwhile.
    Parking a serving group in PARTIAL over such an entry would be worse.

    ⚠️ That leniency hid a bug once, which is why the kept entries are
    returned instead of only logged: an unencoded URL in the path got 405 and
    the stale member stayed, while the outcome still read ok.
    """
    remove = _endpoint(api, base, api.remove)
    kept: List[str] = []
    if remove is None:
        return kept
    method, template = remove
    for url in current:
        if url in wanted:
            continue
        # 🔴 Percent-encoded, and this cost a real bug: the member's id IS a
        # URL, so substituting it raw makes the path `/workers/http://host:port`,
        # which upstream routes to its transparent proxy instead -- 405 "Only
        # POST requests are supported for transparent proxy". Encoded it is
        # 200. `safe=""` because the `:` and `/` are exactly what must escape.
        endpoint = template.replace("{url}", quote(url, safe=""))
        try:
            async with client.request(
                method,
                endpoint,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
            ) as response:
                if response.status >= 400:
                    kept.append(f"{url} (status {response.status})")
                    logger.info(
                        "Router kept stale member %s (status %s); its removal "
                        "gate waits for in-flight requests",
                        url,
                        response.status,
                    )
        except Exception as e:
            kept.append(f"{url} ({e})")
            logger.info("Could not remove stale member %s: %s", url, e)
    return kept


async def reconcile(
    model: Model,
    mode: Optional[PDMode],
    instances: Sequence[ModelInstance],
    router_address: Optional[str],
    client: Optional[aiohttp.ClientSession] = None,
) -> MembershipOutcome:
    """Make the router's registry match the group's members, and say whether it does.

    Idempotent by construction: it diffs what the router reports against what
    the group has, so running it on every pass costs one read when nothing
    changed. That matters because this is the same call used at group
    formation and at scale-out — one code path, so the startup case cannot
    drift away from the steady-state case.
    """
    if mode is None or not mode.router.membership_api_usable:
        # The recipe does not launch the flag the API needs. Nothing to do,
        # and specifically NOT a failure: the group is on the command-line
        # path, where the router already knows its peers.
        return MembershipOutcome(ok=True, reason=None)

    if not router_address:
        return MembershipOutcome(
            ok=False, reason="the router's address is not known yet"
        )

    api = mode.router.membership_api
    wanted = desired_members(model, instances)
    if not wanted:
        return MembershipOutcome(
            ok=False, reason="no member is running yet, so none can be registered"
        )

    base = f"http://{router_address}"
    owned = client is None
    try:
        if owned:
            client = aiohttp.ClientSession()

        async def _run() -> MembershipOutcome:
            current = await _read_registry(client, api, base)
            if current is None:
                return MembershipOutcome(
                    ok=False,
                    reason=(
                        "the router's member list could not be read, so whether "
                        "it can serve is unknown"
                    ),
                )

            await _add_missing(client, api, base, model.name, wanted, current)

            # Stale entries are removed, and a failure to remove one is NOT
            # fatal: upstream's removal gate is global (it waits for the
            # in-flight count to reach zero), so a member can legitimately
            # linger while traffic continues. Treating that as broken would
            # park a serving group in PARTIAL over an entry that no longer
            # takes new requests.
            #
            # ⚠️ That leniency hid a bug of its own once — an unencoded URL in
            # the path got 405 and the stale member stayed, while the outcome
            # still read ok. So the failure is logged at INFO with the status,
            # not swallowed: "lenient" has to still be visible.
            stale_kept = await _remove_stale(client, api, base, wanted, current)

            # 🔑 The read-back, and the whole reason `ok` is trustworthy. An
            # accepted POST is not a joined member: upstream probes the peer
            # first and drops it silently on timeout.
            final = await _read_registry(client, api, base)
            if final is None:
                return MembershipOutcome(
                    ok=False, reason="the router's member list became unreadable"
                )
            missing = sorted(set(wanted) - set(final))
            if missing:
                return MembershipOutcome(
                    ok=False,
                    reason=(
                        "the router did not admit "
                        f"{', '.join(missing)} — it probes a peer before "
                        "admitting it and drops it on timeout"
                    ),
                    registered=sorted(set(wanted) & set(final)),
                )
            extra = sorted(set(final) - set(wanted))
            if extra:
                # Servable, so `ok` stays True — every wanted member is in.
                # But the reason carries what did not leave, because a
                # silently-lingering member is how the encoding bug above went
                # unnoticed.
                return MembershipOutcome(
                    ok=True,
                    reason=(
                        "still registered after removal: "
                        f"{', '.join(stale_kept or extra)}"
                    ),
                    registered=sorted(final),
                )
            return MembershipOutcome(ok=True, registered=sorted(final))

        return await asyncio.wait_for(_run(), timeout=_RECONCILE_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        return MembershipOutcome(
            ok=False, reason="the router did not answer within the deadline"
        )
    except (aiohttp.ClientError, OSError) as e:
        return MembershipOutcome(
            ok=False,
            reason=f"the router is unreachable: {str(e) or e.__class__.__name__}",
        )
    finally:
        if owned and client is not None:
            await client.close()


# ---------------------------------------------------------------------------
# The recorded outcome, which is what `upstream_registration_ready` reads.
# ---------------------------------------------------------------------------

_outcomes: Dict[int, MembershipOutcome] = {}
_failures: Dict[int, int] = {}

PERSISTENT_FAILURE_PASSES = 5
"""Consecutive failed reconciles before the message names the way out.

🔴 The escape hatch is an OPERATOR action, not something this code can take.
The router's command is rendered from `pd-modes.yaml`, so "drop `--enable-igw`
and go back to command-line peers" means editing the recipe — the controller
cannot un-launch a flag on a running process, and restarting the router only
repeats the same failed registration against the same unreachable endpoint.

So the fallback is stated rather than performed. Five passes because a single
failure is ordinary (a router that has just come up, a member still probing)
and looping on that wording would train people to ignore it; five in a row is
a router that is not going to admit its members on its own."""


def record(model_id: int, outcome: MembershipOutcome) -> None:
    if outcome.ok:
        _failures.pop(model_id, None)
    else:
        count = _failures.get(model_id, 0) + 1
        _failures[model_id] = count
        if count >= PERSISTENT_FAILURE_PASSES and outcome.reason:
            outcome = MembershipOutcome(
                ok=False,
                reason=(
                    f"{outcome.reason} (unchanged for {count} passes; the "
                    "router is running with --enable-igw, where members can "
                    "only join through this API — remove that flag from the "
                    "mode's router command and restart the group to fall back "
                    "to command-line peers)"
                ),
                registered=outcome.registered,
            )
    _outcomes[model_id] = outcome


def outcome_for(model_id: int) -> Optional[MembershipOutcome]:
    return _outcomes.get(model_id)


def forget(model_id: int) -> None:
    _outcomes.pop(model_id, None)
    _failures.pop(model_id, None)


def consecutive_failures(model_id: int) -> int:
    return _failures.get(model_id, 0)


def router_addresses(instances: Sequence[ModelInstance]) -> List[str]:
    """Every running router's `host:port`.

    A list because the group's router count is a declared replica count like
    any other, even though the shipped recipes run one: a second router with
    an empty registry would serve 503s while the first served fine, so each
    has to be reconciled on its own.
    """
    from gpustack.schemas.models import ModelInstanceStateEnum

    return [
        f"{i.worker_ip}:{i.port}"
        for i in instances
        if i.role == RoleNameEnum.ROUTER.value
        and i.state == ModelInstanceStateEnum.RUNNING
        and i.worker_ip
        and i.port
    ]
