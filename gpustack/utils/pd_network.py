from typing import Optional

from gpustack.schemas.models import Model, PDModeEnum


def uses_kv_transfer_ifname(model: Optional[Model]) -> bool:
    """Whether the built-in PD recipe uses a node's KV data interface."""
    disaggregation = getattr(model, "disaggregation", None)
    return getattr(disaggregation, "mode", None) in (
        PDModeEnum.VLLM_NIXL,
        PDModeEnum.SGLANG_NIXL,
    )


def effective_kv_transfer_ifname(
    worker_value: Optional[str],
    local_value: Optional[str],
    cluster_value: Optional[str],
) -> Optional[str]:
    """Choose the first explicit device in Worker, startup, cluster order."""
    return (
        (worker_value or "").strip()
        or (local_value or "").strip()
        or (cluster_value or "").strip()
        or None
    )
