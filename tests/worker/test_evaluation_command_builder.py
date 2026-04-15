from gpustack.worker.evaluation.command_builder import build_lm_eval_command


def test_build_lm_eval_command_for_default_backend():
    command = build_lm_eval_command(
        backend="vLLM",
        pretrained="qwen3.5-9b-vllm",
        base_url="http://192.168.50.16:40057/v1/completions",
        tokenizer="/data/gpustack_cache/huggingface/Qwen/Qwen3.5-9B",
        tasks=["tinyHellaswag"],
        batch_size=1,
    )

    assert command == [
        "lm-eval",
        "--model",
        "local-completions",
        "--model_args",
        "pretrained=qwen3.5-9b-vllm,base_url=http://192.168.50.16:40057/v1/completions,tokenizer=/data/gpustack_cache/huggingface/Qwen/Qwen3.5-9B",
        "--tasks",
        "tinyHellaswag",
        "--batch_size",
        "1",
        "--confirm-run-unsafe-code",
        "--trust-remote-code",
    ]


def test_build_lm_eval_command_for_sglang_backend():
    command = build_lm_eval_command(
        backend="SGLang",
        pretrained="Qwen/Qwen3.5-9B",
        base_url="http://192.168.50.16:40050/generate",
        tokenizer="/data/gpustack_cache/huggingface/Qwen/Qwen3.5-9B",
        tasks=["tinyHellaswag"],
        batch_size=1,
    )

    assert command == [
        "lm-eval",
        "--model",
        "sglang-generate",
        "--model_args",
        "pretrained=Qwen/Qwen3.5-9B,base_url=http://192.168.50.16:40050/generate,tokenizer=/data/gpustack_cache/huggingface/Qwen/Qwen3.5-9B",
        "--tasks",
        "tinyHellaswag",
        "--batch_size",
        "1",
        "--confirm-run-unsafe-code",
        "--trust-remote-code",
    ]


def test_build_lm_eval_command_with_limit():
    command = build_lm_eval_command(
        backend="vLLM",
        pretrained="qwen3.5-9b-vllm",
        base_url="http://192.168.50.16:40057/v1/completions",
        tokenizer="/data/gpustack_cache/huggingface/Qwen/Qwen3.5-9B",
        tasks=["tinyHellaswag"],
        batch_size=1,
        limit=10,
    )

    assert command[-2:] == ["--limit", "10"]
