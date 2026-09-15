"""Whether the group can be admitted to its queue as a unit.

The marker answers a question the user otherwise has to infer from a group
that queues forever: gang admission is Kueue's, so a Docker cluster has none
at all, and on Kubernetes a heterogeneous group needs one Workload to draw
podSets from two ClusterQueues, which is not a thing.

Unlike every other degradation, this one is a property of the configuration
rather than of something that happened — true the moment the group exists.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.models import GPUTypeSelector
from gpustack.server.controllers import _no_atomic_admission


def _role(name, pool=None):
    return SimpleNamespace(
        name=name,
        gpu_type_selector=GPUTypeSelector(type=pool) if pool else None,
    )


async def _check(roles, provider=ClusterProvider.Kubernetes, cluster_id=1):
    model = SimpleNamespace(roles=roles, cluster_id=cluster_id)
    cluster = SimpleNamespace(provider=provider)
    with patch(
        "gpustack.schemas.clusters.Cluster.one_by_id",
        return_value=cluster,
    ):
        return await _no_atomic_admission(None, model)


PD = [_role("prefill"), _role("decode"), _role("router")]


@pytest.mark.asyncio
async def test_homogeneous_group_on_kubernetes_has_gang():
    """The one configuration that does get the guarantee."""
    assert await _check(PD) is False


@pytest.mark.asyncio
async def test_same_pool_on_both_roles_is_homogeneous():
    roles = [_role("prefill", "a100-half"), _role("decode", "a100-half")]
    assert await _check(roles) is False


@pytest.mark.asyncio
async def test_docker_is_not_a_milder_version_of_the_same_problem():
    """🔴 No queue at all, so nothing queues forever and nothing is half
    admitted — a group that does not fit fails scheduling with the shortfall
    named. A permanent badge on a deployment behaving as designed is how a
    marker stops being read."""
    assert await _check(PD, provider=ClusterProvider.Docker) is False
    roles = [_role("prefill", "a100"), _role("decode", "h20")]
    assert await _check(roles, provider=ClusterProvider.Docker) is False


@pytest.mark.asyncio
async def test_different_pools_are_two_clusterqueues():
    roles = [_role("prefill", "a100-full"), _role("decode", "h20-half")]
    assert await _check(roles) is True


@pytest.mark.asyncio
async def test_whole_cards_and_a_pool_are_also_two_queues():
    """🔴 `None` is a value, not a gap.

    A role on whole cards and a role on slices of a pool are as much two
    queues as two different pools are. Treating the missing selector as
    "matches anything" would claim a gang for the one mixed shape that most
    looks like it should have one.
    """
    roles = [_role("prefill"), _role("decode", "h20-half")]
    assert await _check(roles) is True


@pytest.mark.asyncio
async def test_role_less_model_is_not_a_group():
    assert await _check(None) is False
    assert await _check([]) is False


@pytest.mark.asyncio
async def test_single_gpu_role_cannot_be_half_admitted():
    """No second role to be admitted without, so nothing is at stake."""
    assert await _check([_role("prefill"), _role("router")]) is False


@pytest.mark.asyncio
async def test_router_never_counts_toward_heterogeneity():
    """It is outside the gang by construction — inside it, Kueue waits for the
    router, the router waits for its peers' addresses, and the peers wait for
    Kueue."""
    roles = [_role("prefill", "a100"), _role("decode", "a100"), _role("router")]
    assert await _check(roles) is False


@pytest.mark.asyncio
async def test_cluster_less_model_has_no_queue():
    assert await _check(PD, cluster_id=None) is False
