import gc
import os
import random
import statistics
import argparse
import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


def make_random_prompts(
    vocab_size: int,
    num_prompts: int = 8,
    prompt_len: int = 950,
) -> list[list[int]]:
    return [
        [random.randint(0, vocab_size - 1) for _ in range(prompt_len)]
        for _ in range(num_prompts)
    ]


def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def print_metrics(metrics: dict, label: str) -> None:
    ttfts = metrics["ttft_ms"]
    tpots = metrics["tpot_ms"]
    bar = "─" * 56
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    if ttfts:
        print(
            f"  TTFT (ms): mean={statistics.mean(ttfts):7.1f}  "
            f"p50={statistics.median(ttfts):7.1f}  "
            f"p99={_pct(ttfts, 0.99):7.1f}"
        )
    if tpots:
        print(
            f"  TPOT (ms): mean={statistics.mean(tpots):7.1f}  "
            f"p50={statistics.median(tpots):7.1f}  "
            f"p99={_pct(tpots, 0.99):7.1f}"
        )
    print()


def run_benchmark(
    path: str,
    prompts: list[list[int]],
    sampling_params: SamplingParams,
    enable_chunked_prefill: bool,
    prefill_chunk_size: int = 512,
) -> None:
    tag = (
        f"Chunked Prefill ON  (chunk_size={prefill_chunk_size})"
        if enable_chunked_prefill
        else "Chunked Prefill OFF"
    )
    print(f"\n{'='*56}\n  Initializing — {tag}\n{'='*56}")

    llm = LLM(
        path,
        enforce_eager=True,
        enable_chunked_prefill=enable_chunked_prefill,
        prefill_chunk_size=prefill_chunk_size,
    )
    llm.generate(prompts, sampling_params, use_tqdm=True)
    print_metrics(llm.last_metrics, tag)

    # explicit cleanup so the second run doesn't OOM
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main(model_path: str) -> None:
    path = os.path.expanduser(model_path)
    hf_config = AutoConfig.from_pretrained(path)
    vocab_size = hf_config.vocab_size

    num_prompts = 8
    prompt_len = 1024
    prompts = make_random_prompts(vocab_size, num_prompts, prompt_len)

    print(f"Batch : {num_prompts} requests  (random token IDs)")
    print(f"Tokens: {prompt_len} per prompt")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)

    run_benchmark(path, prompts, sampling_params, enable_chunked_prefill=False)
    run_benchmark(path, prompts, sampling_params, enable_chunked_prefill=True, prefill_chunk_size=512)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Example of using NanoVLLM")
    args.add_argument("--model_path", type=str, default="~/huggingface/Qwen3-0.6B/")
    args = args.parse_args()
    main(args.model_path)
