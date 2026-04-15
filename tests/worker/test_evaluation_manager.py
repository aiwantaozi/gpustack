from types import SimpleNamespace
from unittest.mock import MagicMock

from gpustack.schemas.evaluations import Evaluation, EvaluationStateEnum
from gpustack.worker.evaluation_manager import EvaluationManager


def test_sync_evaluation_result_posts_summary_and_tasks(tmp_path, monkeypatch):
    result_file = tmp_path / "1.json"
    result_file.write_text(
        """
        {
          "results": {
            "tinyArc": {
              "alias": "tinyArc",
              "sample_len": 100,
              "acc_norm,none": 0.7359
            }
          },
          "configs": {
            "tinyArc": {
              "dataset_name": "ARC-Challenge",
              "num_fewshot": 25,
              "output_type": "multiple_choice",
              "metadata": {"version": 0}
            }
          }
        }
        """.strip()
    )

    posted = {}

    class FakeHttpClient:
        def post(self, url, json):
            posted[url] = json
            response = MagicMock()
            response.raise_for_status = MagicMock()
            return response

        def patch(self, url, json):
            posted[url] = json
            response = MagicMock()
            response.raise_for_status = MagicMock()
            return response

    manager = EvaluationManager(
        worker_id_getter=lambda: 1,
        clientset_getter=lambda: SimpleNamespace(
            http_client=SimpleNamespace(get_httpx_client=lambda: FakeHttpClient())
        ),
        cfg=SimpleNamespace(log_dir=str(tmp_path), evaluation_dir=str(tmp_path)),
    )
    evaluation = Evaluation(
        id=1,
        name="eval-1",
        suite_id="quick",
        suite_name="Quick",
        state=EvaluationStateEnum.RUNNING,
        worker_id=1,
    )

    manager._sync_evaluation_result(evaluation)

    assert "/evaluations/1/result" in posted
    assert posted["/evaluations/1/result"]["task_count"] == 1
    assert posted["/evaluations/1/result"]["tasks"][0]["task_name"] == "tinyArc"
