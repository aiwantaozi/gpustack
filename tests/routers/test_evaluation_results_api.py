from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.routes.evaluations import (
    update_evaluation_result,
    update_evaluation_state,
)
from gpustack.schemas.evaluations import (
    EvaluationResult,
    EvaluationStateEnum,
    EvaluationStateUpdate,
    EvaluationTaskUpsert,
)


@pytest.mark.asyncio
async def test_update_evaluation_state_applies_patch(monkeypatch):
    evaluation = MagicMock()
    evaluation.state = EvaluationStateEnum.PENDING
    evaluation.update = AsyncMock()

    monkeypatch.setattr(
        "gpustack.routes.evaluations.Evaluation.one_by_id",
        AsyncMock(return_value=evaluation),
    )

    state_update = EvaluationStateUpdate(
        state=EvaluationStateEnum.RUNNING,
        progress=0.5,
    )

    await update_evaluation_state(MagicMock(), 1, state_update)

    evaluation.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_evaluation_result_upserts_tasks(monkeypatch):
    evaluation = MagicMock()
    evaluation.update = AsyncMock()
    created_tasks = []

    monkeypatch.setattr(
        "gpustack.routes.evaluations.Evaluation.one_by_id",
        AsyncMock(return_value=evaluation),
    )

    async def fake_create(_session, item):
        created_tasks.append(item)
        return item

    monkeypatch.setattr(
        "gpustack.routes.evaluations.EvaluationTask.create",
        fake_create,
    )
    monkeypatch.setattr(
        "gpustack.routes.evaluations.EvaluationTask.all_by_field",
        AsyncMock(return_value=[]),
    )

    payload = EvaluationResult(
        task_count=1,
        sample_count=100,
        tasks=[
            EvaluationTaskUpsert(
                task_name="tinyArc",
                primary_metric_key="acc_norm,none",
                primary_metric_value=0.7359,
                raw_metrics={"acc_norm,none": 0.7359},
            )
        ],
    )

    await update_evaluation_result(MagicMock(), 1, payload)

    assert len(created_tasks) == 1
    assert created_tasks[0].evaluation_id == 1
