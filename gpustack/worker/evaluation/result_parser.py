from typing import Any, Dict, List, Optional

from gpustack.schemas.evaluations import EvaluationResult, EvaluationTaskUpsert


def _task_groups(group_subtasks: Optional[Dict[str, List[str]]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group_name, task_names in (group_subtasks or {}).items():
        for task_name in task_names:
            mapping[task_name] = group_name
    return mapping


def _metric_keys(raw_metrics: Dict[str, Any]) -> List[str]:
    keys = []
    for key, value in raw_metrics.items():
        if key == "sample_len" or key.endswith("_stderr"):
            continue
        if "," not in key:
            continue
        if isinstance(value, (int, float)):
            keys.append(key)
    return keys


def parse_evaluation_results(payload: Dict[str, Any]) -> EvaluationResult:
    tasks = []
    sample_count = 0
    task_group_map = _task_groups(payload.get("group_subtasks"))
    configs = payload.get("configs") or {}

    for task_name, raw_task_metrics in (payload.get("results") or {}).items():
        if not isinstance(raw_task_metrics, dict):
            continue

        metric_keys = _metric_keys(raw_task_metrics)
        primary_metric_key = metric_keys[0] if metric_keys else None
        primary_metric_value = (
            raw_task_metrics.get(primary_metric_key)
            if primary_metric_key is not None
            else None
        )
        primary_stderr = None
        if primary_metric_key is not None:
            metric_name, filter_name = primary_metric_key.split(",", 1)
            primary_stderr = raw_task_metrics.get(f"{metric_name}_stderr,{filter_name}")

        config_snapshot = configs.get(task_name) or {}
        metadata = config_snapshot.get("metadata") or {}
        task_sample_count = raw_task_metrics.get("sample_len")
        sample_count += task_sample_count or 0

        tasks.append(
            EvaluationTaskUpsert(
                task_name=task_name,
                task_alias=raw_task_metrics.get("alias"),
                task_group=task_group_map.get(task_name),
                display_name=raw_task_metrics.get("alias") or task_name,
                dataset_name=config_snapshot.get("dataset_name"),
                version=(
                    str(metadata.get("version"))
                    if metadata.get("version") is not None
                    else None
                ),
                sample_count=task_sample_count,
                n_shot=config_snapshot.get("num_fewshot"),
                output_type=config_snapshot.get("output_type"),
                primary_metric_key=primary_metric_key,
                primary_metric_value=primary_metric_value,
                primary_stderr=(
                    str(primary_stderr) if primary_stderr is not None else None
                ),
                raw_metrics=raw_task_metrics,
                config_snapshot=config_snapshot,
                task_metadata=metadata,
            )
        )

    return EvaluationResult(
        task_count=len(tasks),
        sample_count=sample_count,
        tasks=tasks,
    )
