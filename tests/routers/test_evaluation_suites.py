from gpustack.routes.evaluation_suites import (
    get_builtin_evaluation_suites_file_path,
    load_builtin_evaluation_suites,
)


def test_load_builtin_evaluation_suites():
    suites = load_builtin_evaluation_suites()

    assert len(suites.items) >= 3

    suite_ids = {suite.id for suite in suites.items}
    assert "quick-check" in suite_ids
    assert "standard-general" in suite_ids
    assert "chinese-general" in suite_ids

    quick_check = next(s for s in suites.items if s.id == "quick-check")
    assert quick_check.category == "general_knowledge_reasoning"
    assert "leaderboard_bbh" in quick_check.tasks
    assert quick_check.estimated_runtime_level == "short"


def test_builtin_evaluation_suites_file_path():
    path = get_builtin_evaluation_suites_file_path()

    assert path.endswith("evaluation_suites.yaml")
