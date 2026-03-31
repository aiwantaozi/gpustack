from gpustack.schemas.evaluations import (
    Evaluation,
    EvaluationSnapshot,
    EvaluationStateEnum,
    EvaluationTask,
)


def test_evaluation_supports_snapshot():
    evaluation = Evaluation(
        name="qwen35-tinybench",
        suite_id="quick-check",
        suite_name="Quick Check",
        category="general_knowledge_reasoning",
        model_id=1,
        model_name="qwen3.5-9b",
        model_instance_name="qwen3.5-9b-vllm",
        state=EvaluationStateEnum.PENDING,
        snapshot=EvaluationSnapshot(
            instances={},
            workers={},
            gpus={},
        ),
    )

    assert evaluation.snapshot is not None
    assert evaluation.state == EvaluationStateEnum.PENDING


def test_evaluation_task_supports_raw_metrics_and_config_snapshot():
    task = EvaluationTask(
        evaluation_id=1,
        task_name="tinyGSM8k",
        display_name="tinyGSM8k",
        primary_metric_key="exact_match,flexible-extract",
        primary_metric_value=0.9010,
        raw_metrics={
            "exact_match,strict-match": 0.9010,
            "exact_match_stderr,strict-match": "N/A",
            "exact_match,flexible-extract": 0.9010,
            "exact_match_stderr,flexible-extract": "N/A",
        },
        config_snapshot={
            "num_fewshot": 5,
            "output_type": "generate_until",
            "dataset_name": "main",
        },
        task_metadata={"config_source": "lm_eval/tasks/tinyBenchmarks/tinyGSM8k.yaml"},
    )

    assert task.evaluation_id == 1
    assert task.primary_metric_value == 0.9010
    assert task.raw_metrics["exact_match,strict-match"] == 0.9010
    assert task.config_snapshot["num_fewshot"] == 5
    assert task.task_metadata["config_source"].endswith("tinyGSM8k.yaml")
