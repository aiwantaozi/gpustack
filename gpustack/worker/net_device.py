import logging
from typing import Optional

from gpustack.config.config import Config
from gpustack.schemas.workers import Worker

logger = logging.getLogger(__name__)


def derive_net_device(worker: Worker, config: Config) -> Optional[str]:
    """The value of ``{{net_device}}``: the NIC carrying the KV plane.

    Priority: ``config.kv_ifname`` (per-worker escape hatch) -> ``Worker.ifname``
    -> ``None``.

    Do NOT invent a value when this returns ``None``. The renderer leaves an
    unresolved placeholder as-is and logs a WARNING, which is diagnosable;
    falling back to ``all`` is not. UCX with ``UCX_NET_DEVICES=all`` picks up
    ``docker0`` / ``br-*`` / ``flannel.1`` / ``cni0`` and writes the addresses
    behind them (``172.17.x``, ``10.42.x``) into the NIXL metadata. The peer
    cannot route to those, so ``loadRemoteMD()`` fails with
    ``NIXL_ERR_BACKEND`` -- a wrong value instead of a clean failure, at the
    far end of the handshake rather than at the point of the misconfiguration.

    Per-card interfaces are deliberately *not* consulted here, even though
    ``GPUDeviceStatus.network.iface`` exists:

    - On Ascend the detector reads it out of ``hccn_tool``, where it names the
      card-internal ``eth0``-``eth7``. Those devices do not exist in the host
      network namespace, so UCX cannot bind to them -- it is a card identifier,
      not a NIC name.
    - On other vendors the field is only ever populated by hand via
      ``resources.gpu_devices``, and filling that in swaps the whole GPU
      detector for the ``Custom`` one, freezing VRAM/model/power discovery.
      That is a worse trade than typing one ``kv_ifname``.

    Known boundaries, both of which land the caller on ``kv_ifname``:

    - ``Worker.ifname`` is the management-plane NIC. On a single-NIC host that
      is the right answer; on a multi-NIC host, or when the KV traffic is meant
      to ride a dedicated fabric, it is not, and only the operator knows which.
    - When ``gpu_type_selector`` is set (the only way to get a gang), the cards
      are assigned by the device plugin *after* the pod binds, so no per-card
      source could have been used at render time anyway.
    """
    # ``Worker.ifname`` is a non-optional ``str``, but pool-provisioned rows are
    # created with "" as a placeholder before the worker ever reports in, so
    # emptiness -- not absence -- is what marks "unknown" on both sources.
    kv_ifname = (getattr(config, "kv_ifname", None) or "").strip()
    if kv_ifname:
        return kv_ifname

    worker_ifname = (getattr(worker, "ifname", None) or "").strip()
    if worker_ifname:
        return worker_ifname

    logger.warning(
        "Cannot derive the KV plane network interface for worker %s: "
        "neither kv_ifname nor a detected worker ifname is available. "
        "Set kv_ifname on that worker to name the interface explicitly.",
        getattr(worker, "name", None) or "<unknown>",
    )
    return None
