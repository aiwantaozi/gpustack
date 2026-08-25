"""Where a workload is deployed, and what happens when nobody decided.

Two questions, and they are answered in different places on purpose:

- *Which namespace* is resolved on the server from the row's owner, once,
  when the row is created. It is then persisted on the row, so the answer
  never moves under a running Pod.
- *Getting there* — create, read, delete, log — is the worker's, and the only
  input it has is that persisted field. A row with no namespace on it must
  keep working, because that is every row that exists before this change and
  their Pods really are in the runtime's default namespace.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.schemas.benchmark import Benchmark
from gpustack.schemas.cache_services import CacheServiceInstance
from gpustack.schemas.models import ModelInstance
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.server.workload_namespace import (
    WorkloadNamespaceEnsurer,
    resolve_workload_namespace,
)

# --- resolving --------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_org_resolves_to_its_own_namespace():
    org = Principal(id=7, kind=PrincipalType.ORG, name="acme")
    with patch.object(Principal, "one_by_id", AsyncMock(return_value=org)):
        assert await resolve_workload_namespace(MagicMock(), 7) == "gpustack-acme"


@pytest.mark.asyncio
async def test_a_user_resolves_by_id_not_by_name():
    # A user's name is a login identifier and may not be a valid namespace
    # label at all, so the identifier is `user-<id>` — same rule the GPU
    # instance path uses, which is what keeps the two in one namespace.
    user = Principal(id=42, kind=PrincipalType.USER, name="alice@example.com")
    with patch.object(Principal, "one_by_id", AsyncMock(return_value=user)):
        assert await resolve_workload_namespace(MagicMock(), 42) == "gpustack-user-42"


@pytest.mark.asyncio
async def test_no_owner_resolves_to_nothing_rather_than_a_guess():
    # None means "let the runtime use its configured default". Inventing a
    # namespace here would point every later read and delete at a namespace
    # the Pod is not in — and a Pod that cannot be deleted holds its cards.
    assert await resolve_workload_namespace(MagicMock(), None) is None


@pytest.mark.asyncio
async def test_a_missing_principal_resolves_to_nothing():
    with patch.object(Principal, "one_by_id", AsyncMock(return_value=None)):
        assert await resolve_workload_namespace(MagicMock(), 999) is None


# --- carrying it on the row -------------------------------------------------


def test_a_model_instance_hands_its_namespace_to_the_deployment():
    mi = ModelInstance(id=1, name="m-abcde", worker_id=3, namespace="gpustack-acme")
    assert mi.get_deployment_metadata(3).namespace == "gpustack-acme"


def test_a_model_instance_without_one_hands_over_nothing():
    """Not a bug to fix later: it is how a pre-upgrade row keeps its Pod
    reachable, since the runtime then reads the namespace it was created in."""
    mi = ModelInstance(id=1, name="m-abcde", worker_id=3)
    assert mi.get_deployment_metadata(3).namespace is None


def test_a_subordinate_workload_shares_the_leaders_namespace():
    """A follower's workload has its own *name* — the namespace is the
    instance's, so the name-mangling must not touch it."""
    from gpustack.schemas.models import (
        DistributedServers,
        ModelInstanceSubordinateWorker,
    )

    mi = ModelInstance(
        id=1,
        name="m-abcde",
        worker_id=3,
        namespace="gpustack-acme",
        distributed_servers=DistributedServers(
            subordinate_workers=[ModelInstanceSubordinateWorker(worker_id=4)]
        ),
    )
    follower = mi.get_deployment_metadata(4)
    assert follower.name == "m-abcde-f0"
    assert follower.namespace == "gpustack-acme"


def test_a_cache_service_instance_hands_its_namespace_to_the_deployment():
    instance = CacheServiceInstance(
        id=11,
        name="svc-a1b2c",
        cache_service_id=5,
        worker_id=1,
        cluster_id=1,
        namespace="gpustack-acme",
    )
    assert instance.get_deployment_metadata().namespace == "gpustack-acme"


def test_a_benchmark_hands_its_namespace_to_the_deployment():
    benchmark = Benchmark(
        name="bench-1",
        model_instance_name="m-abcde",
        namespace="gpustack-acme",
    )
    assert benchmark.get_deployment_metadata().namespace == "gpustack-acme"


# --- ensuring it exists -----------------------------------------------------


def _ensurer(ops):
    ensurer = WorkloadNamespaceEnsurer(SimpleNamespace(get_api_port=lambda: 80))
    return ensurer


@pytest.mark.asyncio
async def test_the_namespace_is_created_before_anything_lands_in_it():
    """A tenant that never created a GPU instance has no namespace, and the
    worker would meet that as a 404 on Pod creation with nothing to do about
    it. So the server creates it first."""
    created = []
    ops = MagicMock()
    ops.__aenter__ = AsyncMock(return_value=ops)
    ops.__aexit__ = AsyncMock(return_value=False)
    ops.create_namespace = AsyncMock(side_effect=lambda n: created.append(n))

    cluster = SimpleNamespace(id=1, registration_token="tok")
    with (
        patch(
            "gpustack.server.workload_namespace.Cluster.one_by_id",
            AsyncMock(return_value=cluster),
        ),
        patch(
            "gpustack.server.workload_namespace.ClusterOps", return_value=ops
        ) as ops_cls,
    ):
        ensurer = _ensurer(ops)
        await ensurer.ensure(MagicMock(), 1, "gpustack-acme")

    assert created == ["gpustack-acme"]
    # The identifier, not the finished namespace: ClusterOps builds
    # `org_namespace` from it, and handing it the name would double the prefix.
    assert ops_cls.call_args.kwargs["cluster_owner_principal_identifier"] == "acme"


@pytest.mark.asyncio
async def test_a_namespace_is_only_created_once_per_process():
    ops = MagicMock()
    ops.__aenter__ = AsyncMock(return_value=ops)
    ops.__aexit__ = AsyncMock(return_value=False)
    ops.create_namespace = AsyncMock()

    cluster = SimpleNamespace(id=1, registration_token="tok")
    with (
        patch(
            "gpustack.server.workload_namespace.Cluster.one_by_id",
            AsyncMock(return_value=cluster),
        ),
        patch("gpustack.server.workload_namespace.ClusterOps", return_value=ops),
    ):
        ensurer = _ensurer(ops)
        await ensurer.ensure(MagicMock(), 1, "gpustack-acme")
        await ensurer.ensure(MagicMock(), 1, "gpustack-acme")

    ops.create_namespace.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failure_is_not_remembered_as_success():
    """A cluster that was briefly unreachable must be retried, or one blip
    permanently convinces this process the namespace is there."""
    ops = MagicMock()
    ops.__aenter__ = AsyncMock(return_value=ops)
    ops.__aexit__ = AsyncMock(return_value=False)
    ops.create_namespace = AsyncMock(side_effect=[RuntimeError("unreachable"), None])

    cluster = SimpleNamespace(id=1, registration_token="tok")
    with (
        patch(
            "gpustack.server.workload_namespace.Cluster.one_by_id",
            AsyncMock(return_value=cluster),
        ),
        patch("gpustack.server.workload_namespace.ClusterOps", return_value=ops),
    ):
        ensurer = _ensurer(ops)
        await ensurer.ensure(MagicMock(), 1, "gpustack-acme")  # must not raise
        await ensurer.ensure(MagicMock(), 1, "gpustack-acme")

    assert ops.create_namespace.await_count == 2


@pytest.mark.asyncio
async def test_nothing_is_created_for_a_workload_with_no_namespace():
    with patch("gpustack.server.workload_namespace.ClusterOps") as ops_cls:
        ensurer = WorkloadNamespaceEnsurer(SimpleNamespace(get_api_port=lambda: 80))
        await ensurer.ensure(MagicMock(), 1, None)
    ops_cls.assert_not_called()


# --- the worker acting on it ------------------------------------------------


def _plan_backend(namespace):
    import gpustack.worker.backends.base as base_module
    from gpustack.worker.backends.custom import CustomServer

    backend = CustomServer.__new__(CustomServer)
    backend._model = SimpleNamespace(gpu_type_selector=None)
    backend._model_instance = SimpleNamespace(
        worker_id=1,
        gpu_indexes=[0],
        gpu_type="cuda",
        distributed_servers=None,
        namespace=namespace,
    )
    backend._worker = SimpleNamespace(id=1, status=SimpleNamespace(gpu_devices=[]))
    backend._config = SimpleNamespace(system_default_container_registry=None)
    backend._fallback_registry = None
    return backend, base_module


def test_every_engine_deploys_into_the_instances_namespace(monkeypatch):
    """Set in the one place all five backends funnel through, so they cannot
    drift apart on where they deploy."""
    from gpustack_runtime.deployer import WorkloadPlan

    backend, base_module = _plan_backend("gpustack-acme")
    monkeypatch.setattr(
        base_module, "transform_workload_plan", lambda _c, w, _f=None: w
    )

    plan = WorkloadPlan(name="w1")
    backend._transform_workload_plan(plan)

    assert plan.namespace == "gpustack-acme"


def test_an_instance_without_a_namespace_leaves_the_runtime_default(monkeypatch):
    from gpustack_runtime.deployer import WorkloadPlan

    backend, base_module = _plan_backend(None)
    monkeypatch.setattr(
        base_module, "transform_workload_plan", lambda _c, w, _f=None: w
    )

    plan = WorkloadPlan(name="w1")
    backend._transform_workload_plan(plan)

    assert plan.namespace is None


# Stopping an instance must delete in *its* namespace — the face that must not
# be missed, since a Pod deleted in the wrong namespace is not deleted at all
# and goes on holding its accelerators. Covered in
# tests/worker/test_serve_manager.py, which has the manager harness.
