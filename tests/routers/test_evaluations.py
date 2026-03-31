from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

from gpustack.routes.evaluations import create_evaluation
from gpustack.schemas.evaluations import EvaluationCreate, EvaluationStateEnum
from gpustack.schemas.models import ModelInstanceStateEnum


@pytest.mark.asyncio
async def test_create_evaluation_populates_runtime_fields(monkeypatch):
    session = MagicMock()
    evaluation_in = EvaluationCreate(
        name="qwen35-quick-check",
        suite_id="quick-check",
        suite_name="Quick Check",
        category="general",
        model_instance_name="qwen3.5-9b-vllm",
    )

    instance = SimpleNamespace(
        id=2,
        name="qwen3.5-9b-vllm",
        model_id=1,
        model_name="qwen3.5-9b",
        cluster_id=3,
        worker_id=4,
        state=ModelInstanceStateEnum.RUNNING,
    )
    model = SimpleNamespace(id=1)

    monkeypatch.setattr(
        "gpustack.routes.evaluations.Evaluation.one_by_field",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "gpustack.routes.evaluations.ModelInstance.one_by_field",
        AsyncMock(return_value=instance),
    )
    monkeypatch.setattr(
        "gpustack.routes.evaluations.Model.one_by_id",
        AsyncMock(return_value=model),
    )
    monkeypatch.setattr(
        "gpustack.routes.evaluations.get_model_runtime_snapshot",
        AsyncMock(
            return_value={
                "instances": {
                    "qwen3.5-9b-vllm": {
                        "id": 2,
                        "name": "qwen3.5-9b-vllm",
                        "computed_resource_claim": None,
                        "ports": [40057],
                    }
                }
            }
        ),
    )

    async def fake_create(_session, mutated):
        mutated.id = 11
        return mutated

    monkeypatch.setattr(
        "gpustack.routes.evaluations.Evaluation.create",
        fake_create,
    )

    evaluation = await create_evaluation(session, evaluation_in)

    assert evaluation.id == 11
    assert evaluation.model_id == 1
    assert evaluation.model_name == "qwen3.5-9b"
    assert evaluation.model_instance_name == "qwen3.5-9b-vllm"
    assert evaluation.cluster_id == 3
    assert evaluation.worker_id == 4
    assert evaluation.state == EvaluationStateEnum.PENDING
    assert evaluation.snapshot.instances["qwen3.5-9b-vllm"].name == "qwen3.5-9b-vllm"
