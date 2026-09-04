"""What a worker can say about where it is, from what its devices and ports report.

Two sources, both read on the worker and both static for as long as nobody
recables the host:

- **Per-device hints** the runtime attached to each GPU (`topology_hints`):
  the NVLink/HCCS domain the card is in, and the switch its RDMA port is
  cabled to. Eight cards agreeing on one domain is one fact about the host;
  eight cards on eight different switches (a rail-optimised fabric) is one
  fact too — the set of switches — and two hosts with the same set share a
  rail group.
- **Host port LLDP**, for hosts whose cards do not carry their own port
  (anything but Ascend). Listening takes a whole LLDP interval, so it runs in a
  thread off the status-report path and the last answer is reused until the
  next refresh.

The result is keyed by topology label key and read *under* the worker's own
labels, so a hand-filled value always wins.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

ACCELERATOR_DOMAIN_KEY = "topology.gpustack.ai/accelerator-domain"
NVIDIA_CLIQUE_KEY = "nvidia.com/gpu.clique"
SWITCH_KEY = "topology.gpustack.ai/switch"
SWITCH_NAME_KEY = "topology.gpustack.ai/switch-name"

_DOMAIN_KEYS = (ACCELERATOR_DOMAIN_KEY, NVIDIA_CLIQUE_KEY)
_JOIN = "+"


def facts_from_devices(devices: Iterable) -> Dict[str, str]:
    """Fold per-device hints into host-level facts.

    A domain is claimed only when every card that reports one reports the
    same; disagreement means the host straddles domains, which the model does
    not represent, so nothing is claimed and the disagreement is logged.
    Switches are collected as a sorted, de-duplicated set: one switch or many,
    the value identifies the group of ports the host hangs off.
    """
    hints: List[dict] = [
        getattr(d, "topology_hints", None) or {} for d in devices or []
    ]
    out: Dict[str, str] = {}

    for key in _DOMAIN_KEYS:
        values = {h[key] for h in hints if h.get(key)}
        if len(values) == 1:
            out[key] = values.pop()
        elif len(values) > 1:
            logger.warning(
                "Devices on this worker report different %s values (%s); "
                "not claiming a domain for the host.",
                key,
                ", ".join(sorted(values)),
            )

    switches = sorted({h[SWITCH_KEY] for h in hints if h.get(SWITCH_KEY)})
    if switches:
        out[SWITCH_KEY] = _JOIN.join(switches)
        names = sorted({h[SWITCH_NAME_KEY] for h in hints if h.get(SWITCH_NAME_KEY)})
        if names:
            out[SWITCH_NAME_KEY] = _JOIN.join(names)
    return out


def rdma_interfaces() -> List[str]:
    """Host interfaces backed by an RDMA device — the ports KV transfer uses."""
    root = Path("/sys/class/infiniband")
    if not root.is_dir():
        return []
    out = set()
    for dev in root.iterdir():
        net = dev / "device" / "net"
        if net.is_dir():
            out.update(p.name for p in net.iterdir())
    return sorted(out)


class HostSwitchProbe:
    """Listens for the switches behind the host's ports, in the background.

    One listen is a full LLDP interval (35s by default), which cannot sit on
    the status-report path. The probe runs on a thread, refreshes every
    ``refresh_seconds`` and hands back the last complete answer; a round that
    hears nothing keeps the previous one, because silence is far more often a
    quiet interval than a recabling.
    """

    def __init__(self, refresh_seconds: float = 600.0):
        self._refresh_seconds = refresh_seconds
        self._facts: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            from gpustack_runtime.detector import detect_lldp_neighbors  # noqa: F401
        except ImportError:
            # An older runtime: no listener, no facts, nothing else changes.
            return
        self._thread = threading.Thread(
            target=self._run, name="topology-switch-probe", daemon=True
        )
        self._thread.start()

    @property
    def facts(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._facts)

    def _run(self) -> None:
        while True:
            try:
                found = self.probe_once()
                if found:
                    with self._lock:
                        self._facts = found
            except Exception as e:
                logger.debug("Switch probe failed: %s", e)
            time.sleep(self._refresh_seconds)

    @staticmethod
    def probe_once(interfaces: Optional[Sequence[str]] = None) -> Dict[str, str]:
        from gpustack_runtime.detector import detect_lldp_neighbors

        wanted = list(interfaces) if interfaces is not None else rdma_interfaces()
        neighbors = detect_lldp_neighbors(interfaces=wanted or None)
        chassis = sorted({n.chassis_id for n in neighbors if n.chassis_id})
        if not chassis:
            return {}
        out = {SWITCH_KEY: _JOIN.join(chassis)}
        names = sorted({n.system_name for n in neighbors if n.system_name})
        if names:
            out[SWITCH_NAME_KEY] = _JOIN.join(names)
        return out


def merge_facts(
    device_facts: Dict[str, str], host_facts: Dict[str, str]
) -> Dict[str, str]:
    """Device-side facts win over host-side ones.

    KV transfer runs over the accelerator's own port where it has one (Ascend),
    so the switch behind that port is the one that matters; the host's NICs
    are the answer only for hosts whose cards report nothing.
    """
    return {**host_facts, **device_facts}
