from typing import List

from gpustack.scheduler.policy import ModelInstanceScheduleCandidate
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.schemas.workers import WorkerStateEnum

MaxScore = 100


class StatusPolicy:
    def __init__(self):
        pass

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidate with the worker and instance status.
        """

        for candidate in candidates:
            if candidate.worker.state == WorkerStateEnum.NOT_READY:
                candidate.score = 0
                continue

            if candidate.instance.state == ModelInstanceStateEnum.ERROR:
                candidate.score = 0
                continue

            if (
                candidate.worker.state == WorkerStateEnum.READY
                and candidate.instance.state == ModelInstanceStateEnum.RUNNING
            ):
                candidate.score = MaxScore
                continue

            candidate.score = 50

        return candidates
