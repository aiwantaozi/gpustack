from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from gpustack.scheduler.policy import Allocatable, ModelInstanceScheduleCandidate
from gpustack.scheduler.resource_fit_policy import get_worker_allocatable_resource
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    PlacementStrategyEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import AsyncEngine

MaxScore = 100


@dataclass
class ResourceWeight:
    vram: int = 2
    ram: int = 1


@dataclass
class ModelWeight:
    current: int = 1
    others: int = 0.2


@dataclass
class InferServerTypeWeight:
    server: int = 5
    rpc_server: int = 1  # max rpc server count is 3


class ScaleTypeEnum(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


class PlacementPolicy:
    def __init__(
        self,
        model: Model,
        model_instance: Optional[ModelInstance] = None,
        scale_type: ScaleTypeEnum = ScaleTypeEnum.SCALE_UP,
    ):
        self._engine = get_engine()
        self._model_instance = model_instance
        self._model = model
        self._resource_weight = ResourceWeight()
        self._model_weight = ModelWeight()
        self._infer_server_type_weight = InferServerTypeWeight()
        self._scale_type = scale_type

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidate with placement strategy.
        """

        if self._model.placement_strategy == PlacementStrategyEnum.SPREAD:
            return await self.score_spread(candidates)
        elif self._model.placement_strategy == PlacementStrategyEnum.BINPACK:
            return await self.score_binpack(candidates)
        else:
            raise ValueError(
                f"Invalid placement strategy {self._model.placement_strategy}"
            )

    async def score_binpack(  # noqa: C901
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidates with the binpack strategy.
        """
        for candidate in candidates:
            allocatable = await get_worker_allocatable_resource(
                self._engine, candidate.worker
            )

            final_score = 0
            score = await self._score_binpack(candidate, allocatable, self._scale_type)
            final_score = score

            if candidate.rpc_servers:
                rpc_server_score = await self._score_binpack_rpc_servers(
                    candidate, self._scale_type
                )
                final_score = (
                    score * self._infer_server_type_weight.server
                    + rpc_server_score
                    * len(candidate.rpc_servers)
                    * self._infer_server_type_weight.rpc_server
                ) / (
                    self._infer_server_type_weight.server
                    + self._infer_server_type_weight.rpc_server
                    * len(candidate.rpc_servers)
                )

            candidate.score = final_score

        return candidates

    async def score_spread(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidates with the spread strategy.
        """
        worker_model_instances_count_map = await self._get_worker_model_instance_count()

        for candidate in candidates:
            # level 1: max score, no model instances
            if candidate.worker.id not in worker_model_instances_count_map:
                candidate.score = MaxScore
                continue

            instance_count_map = worker_model_instances_count_map.get(
                candidate.worker.id, {}
            )

            if candidate.gpu_indexes is None:
                candidate.score = await self._score_spread_cpu(
                    instance_count_map.get("cpu", {})
                )
            else:
                candidate.score = await self._score_spread_gpu(
                    candidate, instance_count_map.get("gpu", {})
                )

        return candidates

    async def _get_worker_model_instance_count(self) -> dict:
        """
        Get current model and other models deployed model instance count for each worker/gpu.

        Returns:
            dict: A map of worker id to model instance count.

        Example:
            {
                "worker_1": {
                    "cpu": {"current": 2, "others": 3},
                    "gpu": {
                        0: {"current": 1, "others": 2},
                        1: {"current": 0, "others": 1}
                    }
                },
                "worker_2": {
                    "cpu": {"current": 1, "others": 1},
                    "gpu": {
                        0: {"current": 2, "others": 0}
                    }
                }
            }
        """

        model_id = self._model.id
        model_instances = await get_model_instances(self._engine)

        worker_model_instances_count_map = defaultdict(
            lambda: {
                "cpu": {"current": 0, "others": 0},
                "gpu": defaultdict(lambda: {"current": 0, "others": 0}),
            }
        )

        def update_count(worker_id, gpu_index, is_current_model):
            if gpu_index is not None:
                key = "current" if is_current_model else "others"
                worker_model_instances_count_map[worker_id]["gpu"][gpu_index][key] += 1
            else:
                key = "current" if is_current_model else "others"
                worker_model_instances_count_map[worker_id]["cpu"][key] += 1

        for model_instance in model_instances:
            if self._model_instance and model_instance.id == self._model_instance.id:
                continue

            is_current_model = model_instance.model_id == model_id
            if model_instance.gpu_indexes:
                for gpu_index in model_instance.gpu_indexes:
                    update_count(model_instance.worker_id, gpu_index, is_current_model)
            else:
                update_count(model_instance.worker_id, None, is_current_model)

            if model_instance.distributes_servers:
                for rpc_server in model_instance.distributes_servers:
                    update_count(
                        rpc_server.worker_id, rpc_server.gpu_index, is_current_model
                    )

        return worker_model_instances_count_map

    async def _score_spread_gpu(
        self, candidate: ModelInstanceScheduleCandidate, instance_count_map: dict
    ) -> int:
        score = 0
        worker_current_model_instance_count = sum(
            instance_count_map.get(gpu_index, {}).get("current", 0)
            for gpu_index in instance_count_map.keys()
        )

        worker_other_model_instance_count = sum(
            instance_count_map.get(gpu_index, {}).get("others", 0)
            for gpu_index in instance_count_map.keys()
        )

        if (
            worker_current_model_instance_count == 0
            and worker_other_model_instance_count > 0
        ):
            # level 2: 90 < score < 100, only have other model's instances
            score = 90

            each_gpu_max_score = 10 / len(candidate.gpu_indexes)
            for gpu_index in candidate.gpu_indexes:
                if gpu_index not in instance_count_map:
                    score += each_gpu_max_score / 1
                    continue
                count = instance_count_map.get(gpu_index, {}).get("others", 0)
                score += each_gpu_max_score / (count + 1)

        elif (
            worker_current_model_instance_count > 0
            and worker_other_model_instance_count == 0
        ):
            # level 3: 80 < score < 90, only have current model's instances
            score = 80

            each_gpu_max_score = 10 / len(candidate.gpu_indexes)
            for gpu_index in candidate.gpu_indexes:
                if gpu_index not in instance_count_map:
                    score += each_gpu_max_score / 1
                    continue
                count = instance_count_map.get(gpu_index, {}).get("current", 0)
                score += each_gpu_max_score / (count + 1)

        else:
            # level 4: 70 < score < 80, have both current model's instances and other model's instances
            score = 70

            each_gpu_max_score = 10 / len(candidate.gpu_indexes)
            for gpu_index in candidate.gpu_indexes:
                if gpu_index not in instance_count_map:
                    score += each_gpu_max_score / 1
                    continue
                current_count = instance_count_map.get(gpu_index, {}).get("current", 0)
                others_count = instance_count_map.get(gpu_index, {}).get("others", 0)
                score += each_gpu_max_score / (
                    (current_count + 1) + (others_count + 1) * self._model_weight.others
                )

        return score

    async def _score_spread_cpu(self, instance_count_map: dict) -> int:
        worker_current_model_instance_count = instance_count_map.get("current", 0)

        worker_others_model_instance_count = instance_count_map.get("others", 0)

        score = 0
        if (
            worker_current_model_instance_count == 0
            and worker_others_model_instance_count > 0
        ):
            # level 2: 90 < score < 100, only have other model's instances
            score = 10 / (worker_others_model_instance_count + 1)
            score += 90
        elif (
            worker_current_model_instance_count > 0
            and worker_others_model_instance_count == 0
        ):
            # level 3: 80 < score < 90, only have current model's instances
            score = 10 / (worker_current_model_instance_count + 1)
            score += 80
        else:
            # level 4: 70 < score < 80, have both current model's instances and other model's instances
            score = 10 / (
                worker_current_model_instance_count
                + worker_others_model_instance_count * self._model_weight.others
            )
            score += 70

        return score

    async def _score_binpack_rpc_servers(
        self, candidate: ModelInstanceScheduleCandidate, scale_type: str
    ) -> int:
        if candidate.rpc_servers is None:
            return 0
        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            worker_map = {worker.id: worker for worker in workers}

            score = 0
            for rpc_server in candidate.rpc_servers:
                allocatable = await get_worker_allocatable_resource(
                    self._engine, worker_map.get(rpc_server.worker_id)
                )

                wrap = ModelInstanceScheduleCandidate(
                    worker=rpc_server.worker_id,
                    computed_resource_claim=rpc_server.computed_resource_claim,
                    gpu_indexes=[rpc_server.gpu_index],
                )

                score += await self._score_binpack(wrap, allocatable, scale_type)

            return score

    async def _score_binpack(  # noqa: C901
        self,
        candidate: ModelInstanceScheduleCandidate,
        allocatable: Allocatable,
        scale_type: str,
    ) -> int:
        score = 0
        gpu_count = len(candidate.gpu_indexes) if candidate.gpu_indexes else 0

        def calculate_score(ram_claim, ram_allocatable, vram_claim, vram_allocatable):
            ram_score = (
                ram_claim / ram_allocatable * MaxScore * self._resource_weight.ram
            )
            vram_score = (
                vram_claim / vram_allocatable * MaxScore * self._resource_weight.vram
            )
            return (ram_score + vram_score) / (
                self._resource_weight.ram + self._resource_weight.vram
            )

        if gpu_count == 0:
            if scale_type == ScaleTypeEnum.SCALE_UP:
                score = (
                    candidate.computed_resource_claim.ram / allocatable.ram * MaxScore
                )
            elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                score = (
                    candidate.computed_resource_claim.ram
                    / (allocatable.ram + candidate.computed_resource_claim.ram)
                    * MaxScore
                )
        elif gpu_count == 1:
            if scale_type == ScaleTypeEnum.SCALE_UP:
                score = calculate_score(
                    candidate.computed_resource_claim.ram,
                    allocatable.ram,
                    candidate.computed_resource_claim.vram[candidate.gpu_indexes[0]],
                    allocatable.vram[candidate.gpu_indexes[0]],
                )
            elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                score = calculate_score(
                    candidate.computed_resource_claim.ram,
                    allocatable.ram + candidate.computed_resource_claim.ram,
                    candidate.computed_resource_claim.vram[candidate.gpu_indexes[0]],
                    allocatable.vram[candidate.gpu_indexes[0]]
                    + candidate.computed_resource_claim.vram[candidate.gpu_indexes[0]],
                )
        else:
            for i in candidate.gpu_indexes:
                if scale_type == ScaleTypeEnum.SCALE_UP:
                    result = calculate_score(
                        candidate.computed_resource_claim.ram,
                        allocatable.ram,
                        candidate.computed_resource_claim.vram[i],
                        allocatable.vram[i],
                    )
                elif scale_type == ScaleTypeEnum.SCALE_DOWN:
                    result = calculate_score(
                        candidate.computed_resource_claim.ram,
                        allocatable.ram + candidate.computed_resource_claim.ram,
                        candidate.computed_resource_claim.vram[i],
                        allocatable.vram[i] + candidate.computed_resource_claim.vram[i],
                    )
                if result > score:
                    score = result

        return score


async def get_model_instances(engine: AsyncEngine) -> List[ModelInstance]:
    async with AsyncSession(engine) as session:
        model_instances = await ModelInstance.all(session)
        return model_instances
