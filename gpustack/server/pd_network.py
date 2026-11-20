from typing import Dict, List, Optional, Sequence

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.clusters import Cluster
from gpustack.schemas.models import Model, ModelInstance, find_role, member_worker_ids
from gpustack.schemas.workers import Worker
from gpustack.utils.pd_network import (
    effective_kv_transfer_ifname,
    uses_kv_transfer_ifname,
)


def kv_transfer_members(
    model: Model, instances: Sequence[ModelInstance]
) -> List[ModelInstance]:
    """Select members whose NIXL devices come from the platform settings."""
    if not uses_kv_transfer_ifname(model):
        return []
    tracked = []
    for instance in instances:
        role_name = getattr(instance, "role", None)
        if role_name not in ("prefill", "decode") or getattr(
            instance, "deleted_at", None
        ):
            continue
        role = find_role(model, role_name)
        env = role.env if role and role.env is not None else getattr(model, "env", None)
        if not ((env or {}).get("UCX_NET_DEVICES") or "").strip():
            tracked.append(instance)
    return tracked


def kv_transfer_ifname_from_settings(settings) -> Optional[str]:
    if isinstance(settings, dict):
        return settings.get("kv_transfer_ifname")
    return getattr(settings, "kv_transfer_ifname", None)


def _startup_kv_transfer_ifname(worker: Worker) -> Optional[str]:
    return kv_transfer_ifname_from_settings(getattr(worker, "status", None))


async def configured_worker_kv_transfer_ifnames(
    session: AsyncSession, cluster_id: Optional[int], workers: Sequence[Worker]
) -> Dict[str, Optional[str]]:
    """Sample one cluster default for all Workers in a placement transaction."""
    cluster = (
        await Cluster.one_by_id(session, cluster_id, refresh=True)
        if cluster_id is not None and workers
        else None
    )
    default = kv_transfer_ifname_from_settings(getattr(cluster, "worker_config", None))
    return {
        str(worker.id): effective_kv_transfer_ifname(
            getattr(worker, "kv_transfer_ifname", None),
            _startup_kv_transfer_ifname(worker),
            default,
        )
        for worker in workers
        if getattr(worker, "deleted_at", None) is None
        and getattr(worker, "cluster_id", cluster_id) == cluster_id
    }


def kv_transfer_ifnames_changed(
    instances: Sequence[ModelInstance],
    workers: Sequence[Worker],
    configured: Dict[str, Optional[str]],
) -> bool:
    workers_by_id = {worker.id: worker for worker in workers}
    for instance in instances:
        snapshot = getattr(instance, "kv_transfer_ifnames", None)
        for worker_id in member_worker_ids(instance):
            key = str(worker_id)
            if key not in configured:
                continue
            if snapshot is None:
                # Placements without a snapshot used the Worker's startup value.
                previous = effective_kv_transfer_ifname(
                    None, _startup_kv_transfer_ifname(workers_by_id[worker_id]), None
                )
            else:
                previous = snapshot.get(key)
            if configured[key] != previous:
                return True
    return False


async def model_kv_transfer_ifnames_stale(
    session: AsyncSession, model: Model, instances: Sequence[ModelInstance]
) -> bool:
    tracked = kv_transfer_members(model, instances)
    worker_ids = {
        worker_id for entry in tracked for worker_id in member_worker_ids(entry)
    }
    if not worker_ids:
        return False
    workers = await Worker.all_by_ids(session, worker_ids, refresh=True)
    configured = await configured_worker_kv_transfer_ifnames(
        session, model.cluster_id, workers
    )
    return kv_transfer_ifnames_changed(tracked, workers, configured)
