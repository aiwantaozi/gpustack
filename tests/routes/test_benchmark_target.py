"""Who a benchmark measures, and where its container runs.

Two facts that used to be one. A run was aimed at a ModelInstance the client
named, and it was deployed on that instance's worker. Under PD neither holds:
a group answers only through its router, so the client cannot name the member
correctly; and the router holds no weights, so its worker is not a machine the
tokenizer can be read from.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes import benchmarks as route
from gpustack.schemas.benchmark import BenchmarkCreate
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    PDModeEnum,
    RoleNameEnum,
    RoleSpec,
)


def _group_model(**kwargs) -> Model:
    return Model(
        id=kwargs.pop("id", 1),
        name=kwargs.pop("name", "g"),
        cluster_id=1,
        replicas=1,
        roles=[
            RoleSpec(name=RoleNameEnum.PREFILL.value, replicas=1),
            RoleSpec(name=RoleNameEnum.DECODE.value, replicas=1),
            RoleSpec(name=RoleNameEnum.ROUTER.value, replicas=1),
        ],
        disaggregation=DisaggregationSpec(mode=PDModeEnum.VLLM_NIXL),
        **kwargs,
    )


def _plain_model(**kwargs) -> Model:
    return Model(
        id=kwargs.pop("id", 1),
        name=kwargs.pop("name", "m"),
        cluster_id=1,
        replicas=1,
        **kwargs,
    )


def _member(name, role=None, id=1, worker_id=10, path="/w", **kwargs) -> ModelInstance:
    return ModelInstance(
        id=id,
        name=name,
        model_id=1,
        model_name="g",
        role=role,
        group_id=kwargs.pop("group_id", "grp" if role else None),
        worker_id=worker_id,
        resolved_path=path,
        state=kwargs.pop("state", ModelInstanceStateEnum.RUNNING),
        **kwargs,
    )


def _group_members():
    return [
        _member("p", RoleNameEnum.PREFILL.value, id=1, worker_id=11),
        _member("d", RoleNameEnum.DECODE.value, id=2, worker_id=12),
        # The router holds nothing: no weights on its worker is the whole point.
        _member("r", RoleNameEnum.ROUTER.value, id=3, worker_id=13, path=None),
    ]


def _with_members(members):
    return patch.object(ModelInstance, "all_by_field", AsyncMock(return_value=members))


class TestEndpoint:
    """`servable_instances()` decides, which makes this its third caller after
    the gateway's upstream registration and the direct proxy."""

    @pytest.mark.asyncio
    async def test_a_group_resolves_to_its_router(self):
        model = _group_model()
        with _with_members(_group_members()):
            endpoint = await route._resolve_target_endpoint(None, model, None)
        assert endpoint.name == "r"

    @pytest.mark.asyncio
    async def test_naming_a_prefill_member_is_refused_with_the_reason(self):
        # It would not fail: a prefill answers 200 after a single token. The
        # run would report latencies for a path no user request takes.
        model = _group_model()
        with _with_members(_group_members()):
            with pytest.raises(BadRequestException) as excinfo:
                await route._resolve_target_endpoint(None, model, "p")
        message = excinfo.value.message
        assert "router" in message
        assert "prefill" in message

    @pytest.mark.asyncio
    async def test_naming_the_router_is_accepted(self):
        model = _group_model()
        with _with_members(_group_members()):
            endpoint = await route._resolve_target_endpoint(None, model, "r")
        assert endpoint.name == "r"

    @pytest.mark.asyncio
    async def test_a_group_without_a_running_router_is_refused(self):
        # Not "fall back to a GPU member": no member of a group serves alone,
        # so there is nothing to measure and saying so is the honest answer.
        model = _group_model()
        members = [m for m in _group_members() if m.role != RoleNameEnum.ROUTER.value]
        with _with_members(members):
            with pytest.raises(BadRequestException) as excinfo:
                await route._resolve_target_endpoint(None, model, None)
        assert "router" in excinfo.value.message

    @pytest.mark.asyncio
    async def test_a_plain_model_resolves_to_its_running_instance(self):
        model = _plain_model()
        with _with_members([_member("mi", None, id=7)]):
            endpoint = await route._resolve_target_endpoint(None, model, None)
        assert endpoint.name == "mi"

    @pytest.mark.asyncio
    async def test_a_named_instance_of_a_plain_model_is_honored(self):
        # The behaviour every existing caller has: naming a replica picks it.
        model = _plain_model()
        members = [_member("a", None, id=1), _member("b", None, id=2)]
        with _with_members(members):
            endpoint = await route._resolve_target_endpoint(None, model, "b")
        assert endpoint.name == "b"

    @pytest.mark.asyncio
    async def test_an_instance_that_is_not_running_keeps_its_old_message(self):
        model = _plain_model()
        stopped = _member("mi", None, id=1, state=ModelInstanceStateEnum.ERROR)
        with _with_members([stopped]):
            with pytest.raises(BadRequestException) as excinfo:
                await route._resolve_target_endpoint(None, model, "mi")
        assert "not in RUNNING state" in excinfo.value.message


class TestPlacement:
    """`--processor` is a host path, so the run has to sit where the weights
    are — which under PD is never the endpoint."""

    @pytest.mark.asyncio
    async def test_a_group_runs_next_to_decode_not_the_router(self):
        model = _group_model()
        members = _group_members()
        router = members[-1]
        with _with_members(members):
            placement = await route._resolve_placement(None, model, router)
        assert placement.name == "d"
        assert placement.worker_id != router.worker_id

    @pytest.mark.asyncio
    async def test_prefill_is_used_when_there_is_no_decode(self):
        # Preference, not a requirement: the load generator is CPU-bound and
        # prefill is the role that spends its CPU on tokenization.
        model = _group_model()
        members = [m for m in _group_members() if m.role != RoleNameEnum.DECODE.value]
        with _with_members(members):
            placement = await route._resolve_placement(None, model, members[-1])
        assert placement.name == "p"

    @pytest.mark.asyncio
    async def test_no_member_holds_weights_falls_back_to_the_endpoint(self):
        model = _group_model()
        members = [
            _member("p", RoleNameEnum.PREFILL.value, id=1, worker_id=11, path=None),
            _member("r", RoleNameEnum.ROUTER.value, id=3, worker_id=13, path=None),
        ]
        with _with_members(members):
            placement = await route._resolve_placement(None, model, members[-1])
        assert placement.name == "r"

    @pytest.mark.asyncio
    async def test_a_plain_model_runs_where_it_always_did(self):
        model = _plain_model()
        instance = _member("mi", None, id=7)
        placement = await route._resolve_placement(None, model, instance)
        assert placement is instance


class TestSnapshot:
    """A group's endpoint is its router, which holds no accelerator: a snapshot
    of it alone reports a run on zero cards."""

    @pytest.mark.asyncio
    async def test_a_group_snapshot_covers_every_member(self):
        model = _group_model()
        members = _group_members()
        router = members[-1]
        with (
            _with_members(members),
            patch.object(
                route,
                "WorkerService",
                lambda _: SimpleNamespace(
                    get_by_id=AsyncMock(return_value=SimpleNamespace(name="w"))
                ),
            ),
            patch.object(route, "create_worker_snapshot", lambda *_: (None, None)),
        ):
            snapshot = await route.get_benchmark_snapshot(None, router, model)
        assert set(snapshot.instances) == {"p", "d", "r"}

    @pytest.mark.asyncio
    async def test_the_generation_measured_is_recorded(self):
        # Two runs of "the same model" measure different things across a
        # config edit; without this the reports cannot be told apart later.
        model = _group_model()
        members = _group_members()
        for m in members:
            # One group_id is one generation, so every member carries it.
            m.spec_digest = "abc123"
        with (
            _with_members(members),
            patch.object(
                route,
                "WorkerService",
                lambda _: SimpleNamespace(
                    get_by_id=AsyncMock(return_value=SimpleNamespace(name="w"))
                ),
            ),
            patch.object(route, "create_worker_snapshot", lambda *_: (None, None)),
        ):
            snapshot = await route.get_benchmark_snapshot(None, members[-1], model)
        assert snapshot.spec_digest == "abc123"

    @pytest.mark.asyncio
    async def test_a_plain_model_snapshot_is_unchanged(self):
        model = _plain_model()
        instance = _member("mi", None, id=7)
        with (
            _with_members([instance]),
            patch.object(
                route,
                "WorkerService",
                lambda _: SimpleNamespace(
                    get_by_id=AsyncMock(return_value=SimpleNamespace(name="w"))
                ),
            ),
            patch.object(route, "create_worker_snapshot", lambda *_: (None, None)),
        ):
            snapshot = await route.get_benchmark_snapshot(None, instance, model)
        assert set(snapshot.instances) == {"mi"}


class TestTargetModel:
    @pytest.mark.asyncio
    async def test_an_instance_name_still_identifies_its_model(self):
        # The instance list page's "run benchmark" action names a member.
        instance = _member("d", RoleNameEnum.DECODE.value, id=2)
        model = _group_model()
        with (
            patch.object(
                ModelInstance, "one_by_field", AsyncMock(return_value=instance)
            ),
            patch.object(Model, "one_by_id", AsyncMock(return_value=model)),
        ):
            resolved = await route._resolve_target_model(
                None, BenchmarkCreate(name="bm", model_instance_name="d")
            )
        assert resolved.name == "g"

    @pytest.mark.asyncio
    async def test_naming_nothing_at_all_is_refused(self):
        with pytest.raises(BadRequestException) as excinfo:
            await route._resolve_target_model(None, BenchmarkCreate(name="bm"))
        assert "model_id" in excinfo.value.message
