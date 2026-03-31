from gpustack.worker.evaluation.result_parser import parse_evaluation_results


def test_parse_evaluation_results_extracts_primary_metrics_and_tasks():
    payload = {
        "results": {
            "tinyArc": {
                "alias": "tinyArc",
                "sample_len": 100,
                "acc_norm,none": 0.7359,
                "acc_norm_stderr,none": "N/A",
            },
            "tinyGSM8k": {
                "alias": "tinyGSM8k",
                "sample_len": 100,
                "exact_match,strict-match": 0.901,
                "exact_match_stderr,strict-match": "N/A",
                "exact_match,flexible-extract": 0.901,
                "exact_match_stderr,flexible-extract": "N/A",
            },
        },
        "group_subtasks": {"tinyBenchmarks": ["tinyArc", "tinyGSM8k"]},
        "configs": {
            "tinyArc": {
                "dataset_name": "ARC-Challenge",
                "num_fewshot": 25,
                "output_type": "multiple_choice",
                "metadata": {"version": 0},
            },
            "tinyGSM8k": {
                "dataset_name": "main",
                "num_fewshot": 5,
                "output_type": "generate_until",
                "metadata": {"version": 0},
            },
        },
    }

    result = parse_evaluation_results(payload)

    assert result.task_count == 2
    assert result.sample_count == 200
    assert len(result.tasks) == 2
    assert result.tasks[0].task_name == "tinyArc"
    assert result.tasks[0].primary_metric_key == "acc_norm,none"
    assert result.tasks[0].primary_metric_value == 0.7359
    assert result.tasks[0].task_group == "tinyBenchmarks"
    assert result.tasks[1].primary_metric_key == "exact_match,strict-match"
    assert result.tasks[1].config_snapshot["num_fewshot"] == 5
