import argparse

from gpustack.cmd.start import parse_args, start_cmd_options


def test_parse_args_supports_evaluation_worker_options():
    parser = argparse.ArgumentParser()
    start_cmd_options(parser)

    args = parser.parse_args(
        [
            "--evaluation-dir",
            "/tmp/evaluations",
            "--evaluation-image-repo",
            "example.com/gpustack/evaluation-runner:test",
            "--evaluation-max-duration-seconds",
            "1800",
            "--data-dir",
            "/tmp/gpustack-test-data",
        ]
    )

    cfg = parse_args(args)

    assert cfg.evaluation_dir == "/tmp/evaluations"
    assert cfg.evaluation_image_repo == "example.com/gpustack/evaluation-runner:test"
    assert cfg.evaluation_max_duration_seconds == 1800
