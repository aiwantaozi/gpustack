from types import SimpleNamespace

from gpustack.schemas.evaluation_suites import EvaluationSuite, EvaluationSuitesPublic
from gpustack.schemas.evaluations import Evaluation, EvaluationSnapshot
from gpustack.worker.evaluation.runner import EvaluationRunner


def _build_runner(evaluation: Evaluation) -> EvaluationRunner:
    return EvaluationRunner(
        clientset=SimpleNamespace(),
        evaluation=evaluation,
        cfg=SimpleNamespace(debug=False, evaluation_dir="/tmp/evaluations"),
    )


def test_runner_uses_suite_default_limit(monkeypatch):
    captured = {}

    def fake_build_lm_eval_command(**kwargs):
        captured["kwargs"] = kwargs
        return ["lm-eval"]

    monkeypatch.setattr(
        "gpustack.worker.evaluation.runner.load_builtin_evaluation_suites",
        lambda: EvaluationSuitesPublic(
            items=[
                EvaluationSuite(
                    id="quick",
                    name="Quick",
                    description="Quick suite",
                    category="preset",
                    tasks=["mmlu_pro"],
                    limit=10,
                    estimated_runtime_level="short",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "gpustack.worker.evaluation.runner.build_lm_eval_command",
        fake_build_lm_eval_command,
    )

    evaluation = Evaluation(
        id=1,
        name="eval-1",
        suite_id="quick",
        suite_name="Quick",
        model_name="Qwen/Qwen3.5-9B",
        model_instance_name="qwen",
        snapshot=EvaluationSnapshot(
            instances={
                "qwen": {
                    "id": 1,
                    "name": "qwen",
                    "worker_ip": "127.0.0.1",
                    "ports": [4000],
                    "resolved_path": "/models/qwen",
                    "backend": "vLLM",
                }
            }
        ),
    )

    runner = _build_runner(evaluation)
    runner._build_command()

    assert captured["kwargs"]["limit"] == 10


def test_runner_prefers_explicit_evaluation_limit(monkeypatch):
    captured = {}

    def fake_build_lm_eval_command(**kwargs):
        captured["kwargs"] = kwargs
        return ["lm-eval"]

    monkeypatch.setattr(
        "gpustack.worker.evaluation.runner.load_builtin_evaluation_suites",
        lambda: EvaluationSuitesPublic(
            items=[
                EvaluationSuite(
                    id="quick",
                    name="Quick",
                    description="Quick suite",
                    category="preset",
                    tasks=["mmlu_pro"],
                    limit=10,
                    estimated_runtime_level="short",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "gpustack.worker.evaluation.runner.build_lm_eval_command",
        fake_build_lm_eval_command,
    )

    evaluation = Evaluation(
        id=1,
        name="eval-1",
        suite_id="quick",
        suite_name="Quick",
        model_name="Qwen/Qwen3.5-9B",
        model_instance_name="qwen",
        limit=3,
        snapshot=EvaluationSnapshot(
            instances={
                "qwen": {
                    "id": 1,
                    "name": "qwen",
                    "worker_ip": "127.0.0.1",
                    "ports": [4000],
                    "resolved_path": "/models/qwen",
                    "backend": "vLLM",
                }
            }
        ),
    )

    runner = _build_runner(evaluation)
    runner._build_command()

    assert captured["kwargs"]["limit"] == 3
