from typing import List

from gpustack.scheduler.policy import ModelInstanceScheduleCandidate

MaxScore = 100


class OffloadLayerPolicy:
    def __init__(self):
        pass

    async def score(
        self, candidates: List[ModelInstanceScheduleCandidate]
    ) -> List[ModelInstanceScheduleCandidate]:
        """
        Score the candidate with offload layers.
        """

        for candidate in candidates:
            total_layers = candidate.computed_resource_claim.total_layers
            offload_layers = candidate.computed_resource_claim.offload_layers

            if total_layers == offload_layers:
                candidate.score = MaxScore
            else:
                candidate.score = offload_layers / total_layers * MaxScore

        return candidates
