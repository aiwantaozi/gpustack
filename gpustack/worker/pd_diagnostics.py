"""Turning a disaggregated member's failure into something actionable.

Two problems this exists for, both measured rather than anticipated.

**The last exception is not the cause.** vLLM's `_handle_failed_transfer` raises
`IndexError: list index out of range` while the real reason —
`NIXL_ERR_BACKEND`, a compatibility hash mismatch, an address already in use —
was logged earlier. Reporting the last exception therefore reports the symptom
of the symptom. It was first assumed this only happened on a tensor-parallelism
mismatch; it happens on any handshake failure. So the log is *scanned* for known
signatures and the earliest one wins, because the earliest is the one that
caused the rest.

**A crash loop looks like progress.** Every port-level failure in a
disaggregated deployment produces the same shape: the container binds, fails,
exits, is restarted, and the instance sits at `starting` forever. Nothing ever
marks it failed, so nothing surfaces it and nothing stops it. A member that has
restarted repeatedly without ever serving has failed, and saying so is the whole
point of `RestartTracker`.
"""

import logging
import re
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, Mapping, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Diagnosis(BaseModel):
    """A recognised failure and what to do about it."""

    signature: str
    """The pattern that matched, so a report can be traced back to the log."""
    line: str
    """The log line it matched on, trimmed. Kept because the actionable text
    below is a generalisation and an operator often needs the specific."""
    summary: str
    """What the user should check. Written as an instruction, not a
    restatement of the error."""


# Ordered by nothing but readability: the scan reports the EARLIEST match in the
# log, not the first pattern in this list, so the order here carries no
# precedence. Each pattern is a signature observed in a real failure, and each
# summary names the setting to look at rather than describing the error again.
_SIGNATURES: Tuple[Tuple[str, str, str], ...] = (
    (
        "NIXL_ERR_BACKEND",
        r"NIXL_ERR_BACKEND",
        "The KV transport could not reach its peer. This is almost always the "
        "network interface: check that kv_ifname names the interface carrying "
        "KV traffic on both workers, and that the address the engine "
        "advertised is one the peer can route to. Left to itself the transport "
        "picks up a container bridge and advertises an unroutable address.",
    ),
    (
        "compatibility hash mismatch",
        r"compatibility hash mismatch|hash mismatch.*kv|kv.*hash mismatch",
        "Prefill and decode were configured differently in a way the "
        "connector refuses. Compare the two roles' data type, KV cache data "
        "type, block size, KV cache layout and attention backend — the engine "
        "hashes all of them and rejects a pair that disagrees.",
    ),
    (
        "address already in use",
        r"[Aa]ddress already in use|EADDRINUSE",
        "A port this member needs was already taken. Two members of one role "
        "on one host collide unless their connector ports are allocated as "
        "separate bands; if the port below is outside the service port range, "
        "it is one the engine chose itself and GPUStack cannot reserve it.",
    ),
    (
        "rdma unavailable",
        r"rdma_create_event_channel failed|rdma_create_id failed|"
        r"No RDMA devices found|ibv_open_device failed",
        "RDMA is not usable in this container. Check that the host has an HCA "
        "and that the container has IPC_LOCK and can see the RDMA devices. "
        "Without it the transport falls back to TCP if the connector allows "
        "one, and fails outright if it does not.",
    ),
    (
        "ucx no device",
        r"No such device.*tcp://|UCX_NET_DEVICES.*not found|"
        r"ZMQError: No such device",
        "The interface name the engine was given does not exist on this "
        "worker. If it looks like an unresolved template placeholder, the "
        "value could not be derived — set kv_ifname on this worker.",
    ),
    (
        "unresolved placeholder",
        r"\{\{[A-Za-z_][A-Za-z0-9_.]*\}\}",
        "A configuration value reached the engine unrendered. The placeholder "
        "below had no value at launch: a named port that was not allocated, or "
        "a network interface that could not be derived.",
    ),
    (
        "kv connector conflict",
        r"kv[-_]transfer[-_]config.*(specified|duplicate|already)|"
        r"multiple.*kv_connector",
        "More than one KV connector configuration reached the engine, and it "
        "accepts exactly one. A disaggregated role cannot also enable an "
        "extended KV cache, and its engine parameters must not carry a "
        "hand-written --kv-transfer-config.",
    ),
)

_COMPILED = tuple(
    (name, re.compile(pattern), summary) for name, pattern, summary in _SIGNATURES
)

_PORT_IN_LINE = re.compile(r"(?<!\d)(\d{4,5})(?!\d)")

_MAX_LINE = 400
"""Log lines from an engine can be enormous (a full config dump). The line is
carried for context, not for archival."""


def diagnose(
    log_text: Optional[str],
    named_ports: Optional[Mapping[str, object]] = None,
) -> Optional[Diagnosis]:
    """The earliest recognised failure in `log_text`, or None.

    Earliest, not last, and not "highest priority". A handshake failure is
    followed by a cascade of derived errors — the transport reports it, the
    scheduler mishandles the empty result, and the exception that escapes is
    an `IndexError` about a list. Only the first line in that sequence names
    something a user can act on.

    `named_ports` lets an "address already in use" be attributed to the band it
    belongs to, which turns "port 40031 is taken" into "the kv_side_channel
    band is taken" — the difference between a number and a thing to fix.
    """
    if not log_text:
        return None

    for raw_line in log_text.splitlines():
        for name, pattern, summary in _COMPILED:
            match = pattern.search(raw_line)
            if match is None:
                continue
            line = raw_line.strip()[:_MAX_LINE]
            detail = summary
            if name == "address already in use":
                band = _attribute_port(line, named_ports)
                if band:
                    detail = f"{summary} The port belongs to the '{band}' band."
            elif name == "unresolved placeholder":
                detail = f"{summary} Unrendered: {match.group(0)}."
            return Diagnosis(signature=name, line=line, summary=detail)
    return None


def _attribute_port(
    line: str, named_ports: Optional[Mapping[str, object]]
) -> Optional[str]:
    """Which declared band a port mentioned in `line` falls inside.

    Bands, not points: a connector derives several ports from one base, so the
    number in the log is often base+n rather than the base itself, and matching
    only the base would attribute nothing in exactly the cases where two
    members collided on a derived port.
    """
    if not named_ports:
        return None
    numbers = {int(n) for n in _PORT_IN_LINE.findall(line)}
    if not numbers:
        return None
    for name, band in named_ports.items():
        base = getattr(band, "base", None)
        count = getattr(band, "count", 1) or 1
        if base is None:
            continue
        if any(base <= number < base + count for number in numbers):
            return name
    return None


class RestartTracker:
    """Recognises a member that keeps restarting without ever serving.

    Stateful and per worker process, deliberately: the signal is a *rate*, and
    the instance row records only a cumulative count and the time of the last
    restart, which cannot distinguish "restarted twice in a year" from
    "restarting every four seconds".

    The condition is two facts together, and both are needed. Restarts alone
    are normal — a member may legitimately be replaced. Never having served
    alone is normal too, briefly. Restarting repeatedly while never having
    served is the shape every port-level failure takes, and it is the shape
    that otherwise reports itself as `starting` indefinitely.
    """

    def __init__(self, threshold: int = 3, window: timedelta = timedelta(minutes=5)):
        self._threshold = threshold
        self._window = window
        self._restarts: Dict[int, Deque[datetime]] = {}
        self._served: Dict[int, bool] = {}
        self._last_count: Dict[int, int] = {}

    def observe_running(self, instance_id: int) -> None:
        """Record that this member served at least once.

        Latching rather than clearing the history: a member that served and
        then began crash-looping is a different failure (it ran, so its
        configuration is not the problem) and must not be reported as one
        that never started.
        """
        self._served[instance_id] = True

    def observe_restart_count(
        self, instance_id: int, restart_count: int, now: datetime
    ) -> bool:
        """Feed the container's cumulative restart count. Returns True the
        moment this member should be considered failed rather than starting.

        Takes the cumulative count rather than an event because that is what a
        polling loop can actually observe; the increments between polls are
        what get timestamped. A count that went down (the workload was
        recreated) resets rather than underflowing.
        """
        previous = self._last_count.get(instance_id)
        self._last_count[instance_id] = restart_count

        if previous is None or restart_count < previous:
            # First observation, or a fresh workload. Neither is a restart we
            # witnessed, and counting it would attribute the previous
            # workload's history to this one.
            self._restarts.pop(instance_id, None)
            return False
        if restart_count == previous:
            return self._is_looping(instance_id, now)

        stamps = self._restarts.setdefault(instance_id, deque())
        for _ in range(restart_count - previous):
            stamps.append(now)
        return self._is_looping(instance_id, now)

    def _is_looping(self, instance_id: int, now: datetime) -> bool:
        if self._served.get(instance_id):
            return False
        stamps = self._restarts.get(instance_id)
        if not stamps:
            return False
        cutoff = now - self._window
        while stamps and stamps[0] < cutoff:
            stamps.popleft()
        return len(stamps) >= self._threshold

    def forget(self, instance_id: int) -> None:
        """Drop a member's history. Called when it is deleted, so a recreated
        instance reusing the id does not inherit a verdict."""
        self._restarts.pop(instance_id, None)
        self._served.pop(instance_id, None)
        self._last_count.pop(instance_id, None)
