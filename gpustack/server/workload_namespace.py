"""Where a tenant's workloads live.

Model instances, cache service instances and benchmark runs are all deployed
by a worker into a Kubernetes namespace. That namespace is **per workload**,
not per worker: a worker DaemonSet is per-cluster and cannot be "in" several
tenants' namespaces at once, while the workloads it runs belong to different
tenants. So the namespace has to be decided where the tenant is known — on
the server, when the row is created — and carried on the row from there.

The resolution matches what the GPU instance path already does
(``gpu_instances/controllers.py``): the *resource's own* owner picks the
namespace, so a model and the GPU instances of the same Org land in the same
namespace and therefore the same quota domain.
"""

import logging
from typing import Any, Iterable, Optional, Set, Tuple

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.config.config import Config
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.gpu_instances.cluster_apis_util import (
    get_namespace_name,
    parse_namespace_name,
    principal_namespace_identifier,
)
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.principals import Principal

logger = logging.getLogger(__name__)


async def resolve_workload_namespace(
    session: AsyncSession,
    owner_principal_id: Optional[int],
    cluster_id: Optional[int] = None,
) -> Optional[str]:
    """The namespace workloads owned by ``owner_principal_id`` are deployed in.

    Returns None when there is no namespace to name — either because the owner
    cannot be resolved, or because the cluster has no such concept. None is not
    an error the caller has to handle: a workload declaring no namespace is
    deployed to the deployer's configured default, exactly where every workload
    went before namespaces were per-tenant. Losing tenant isolation is the
    lesser failure against refusing to deploy at all — but a *resolvable* owner
    failing to resolve is worth a log line, because a workload landing there is
    invisible to the tenant's LocalQueue.

    A Docker cluster returns None because a namespace there is not merely
    unused, it does not exist: recording one would put a fact in the row that
    is false, and the placement-drift check downstream would then report every
    Docker instance as misplaced with nowhere to move it to.
    """
    if owner_principal_id is None:
        return None

    if cluster_id is not None:
        cluster = await Cluster.one_by_id(session, cluster_id)
        if cluster is None or cluster.provider != ClusterProvider.Kubernetes:
            return None

    principal = await Principal.one_by_id(session, owner_principal_id)
    if principal is None:
        logger.warning(
            "Cannot resolve the workload namespace of principal %s: no such "
            "principal. The workload will be deployed to the runtime's default "
            "namespace.",
            owner_principal_id,
        )
        return None

    return get_namespace_name(principal_namespace_identifier(principal))


def placement_drifted(
    instances: Iterable[Any],
    target_namespace: Optional[str],
) -> bool:
    """Whether any of these workloads sits somewhere other than where one
    created now would go.

    This is what an upgrade leaves behind, and leaving it behind is the right
    default: the workloads keep serving, every operation still finds them, and
    they move on the next restart. What is *not* acceptable is that being
    silent, because a Pod in the old namespace holds real accelerators that
    the tenant's queue has no record of — so quota accounting is optimistic by
    exactly those cards until the instance cycles.

    A target of None means there is nowhere to move to (a Docker cluster, or
    an owner that cannot be resolved), so nothing can have drifted from it.
    Reported for a namespace that merely *changed* too — an Org rename moves
    the target while the Pods stay put, which is the same fact.
    """
    if target_namespace is None:
        return False
    return any(
        getattr(instance, "namespace", None) != target_namespace
        for instance in instances
    )


class WorkloadNamespaceEnsurer:
    """Creates a tenant's workload namespace before anything is deployed into it.

    The manifest renderer creates two namespaces, and the GPU instance path
    creates the rest as a side effect of a CR landing. A tenant that has never
    created a GPU instance therefore has no namespace at all, and its first
    model would fail at ``create_namespaced_pod`` with a 404 — a failure that
    reads as "the deployment is broken", not as "a namespace is missing".

    Kept per controller instance rather than per call: the answer is stable
    for the process's lifetime, and the alternative is a Kubernetes read on
    every reconcile of every model. Only *successes* are remembered, so a
    cluster that was briefly unreachable is retried rather than assumed done.
    """

    def __init__(self, config: Config):
        self._config = config
        self._ensured: Set[Tuple[int, str]] = set()

    async def ensure(
        self,
        session: AsyncSession,
        cluster_id: Optional[int],
        namespace: Optional[str],
    ) -> None:
        """Best-effort: log and carry on if the namespace cannot be created.

        Refusing to reconcile would be worse than trying to deploy — the Pod
        creation that follows reports the real reason, with the namespace in
        the message, whereas a controller that bailed out here leaves a model
        stuck with nothing to point at.
        """
        if not namespace or cluster_id is None:
            return
        key = (cluster_id, namespace)
        if key in self._ensured:
            return

        try:
            cluster = await Cluster.one_by_id(session, cluster_id)
            if cluster is None or not cluster.registration_token:
                return
            ops = ClusterOps(
                server_api_port=self._config.get_api_port(),
                cluster_id=cluster.id,
                cluster_registration_token=cluster.registration_token,
                # The identifier the namespace was built from, not the
                # namespace: `ClusterOps` derives `org_namespace` from this,
                # and handing it the finished name would produce
                # `gpustack-gpustack-<org>` for anything reading that instead.
                cluster_owner_principal_identifier=parse_namespace_name(namespace),
            )
            async with ops:
                await ops.create_namespace(namespace)
            self._ensured.add(key)
        except Exception as e:
            logger.warning(
                "Failed to ensure workload namespace %s of cluster %s exists: %s",
                namespace,
                cluster_id,
                e,
            )
