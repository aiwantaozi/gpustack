import logging
from typing import List

from gpustack.scheduler.policy import ModelInstanceScheduleCandidate
from gpustack.schemas.models import Model

MaxScore = 100

logger = logging.getLogger(__name__)


class OffloadLayerPolicy:
    def __init__(self, model: Model):
        self._model = model

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidate with offload layers.
        """

        logger.debug(
            f"model {self._model.name}, score canidates with offload layer policy"
        )

        for candidate in candidates:
            total_layers = candidate.computed_resource_claim.total_layers
            offload_layers = candidate.computed_resource_claim.offload_layers

            if total_layers == offload_layers:
                candidate.score = MaxScore
            else:
                candidate.score = offload_layers / total_layers * MaxScore

        return candidates
