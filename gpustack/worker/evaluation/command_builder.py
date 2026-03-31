from typing import Iterable, List, Optional


def _resolve_lm_eval_model(backend: Optional[str]) -> str:
    if backend == "SGLang":
        return "sglang-generate"
    return "local-completions"


def build_lm_eval_command(
    *,
    backend: Optional[str],
    pretrained: str,
    base_url: str,
    tokenizer: str,
    tasks: Iterable[str],
    batch_size: int,
) -> List[str]:
    model = _resolve_lm_eval_model(backend)
    model_args = f"pretrained={pretrained},base_url={base_url},tokenizer={tokenizer}"

    return [
        "lm-eval",
        "--model",
        model,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
    ]
