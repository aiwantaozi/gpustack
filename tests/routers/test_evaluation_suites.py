from gpustack.routes.evaluation_suites import (
    get_builtin_evaluation_suites_file_path,
    load_builtin_evaluation_suites,
)


def test_load_builtin_evaluation_suites():
    suites = load_builtin_evaluation_suites()

    assert len(suites.items) >= 6

    suite_ids = {suite.id for suite in suites.items}
    assert "quick" in suite_ids
    assert "general" in suite_ids
    assert "all-tasks" in suite_ids

    quick = next(s for s in suites.items if s.id == "quick")
    assert quick.category == "preset"
    assert "mmlu_pro" in quick.tasks
    assert quick.limit == 10
    assert quick.estimated_runtime_level == "short"

    all_tasks = next(s for s in suites.items if s.id == "all-tasks")
    assert "longbench2" in all_tasks.tasks
    assert all_tasks.limit is None


def test_builtin_evaluation_suites_file_path():
    path = get_builtin_evaluation_suites_file_path()

    assert path.endswith("evaluation_suites.yaml")
