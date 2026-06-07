import gc
import os
import random
import statistics
import time
import argparse

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams


# ——— config ———

BATCH_SIZES  = [1, 2, 4, 8, 16, 32, 64, 128]
PROMPT_LEN   = 64      # short prompts so decode dominates
OUTPUT_LEN   = 256     # decode tokens per sequence
WARMUP_TOKENS = 4


# ——— helpers ———

def _make_prompts(vocab_size: int, num_seqs: int) -> list[list[int]]:
    random.seed(42)
    return [
        [random.randint(0, vocab_size - 1) for _ in range(PROMPT_LEN)]
        for _ in range(num_seqs)
    ]


def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


# ——— subprocess worker ———

def _worker(
    path: str,
    enforce_eager: bool,
    vocab_size: int,
    num_seqs: int,
    result_queue: mp.Queue,
) -> None:
    prompts = _make_prompts(vocab_size, num_seqs)
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=OUTPUT_LEN)
    ] * num_seqs

    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=4096)

    # absorb first-run CUDA overhead / trigger graph capture
    llm.generate(["warmup"], SamplingParams(temperature=0.6, max_tokens=WARMUP_TOKENS), use_tqdm=False)

    t = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t

    total_input  = PROMPT_LEN * num_seqs
    total_output = OUTPUT_LEN * num_seqs
    metrics = llm.last_metrics

    result_queue.put({
        "elapsed":     elapsed,
        "total_input":  total_input,
        "total_output": total_output,
        "throughput":   total_output / elapsed,
        "ttft_ms":      metrics.get("ttft_ms", []),
        "tpot_ms":      metrics.get("tpot_ms", []),
    })

    llm.exit()


# ——— run a single (batch_size, mode) point ———

def _run_one(
    path: str,
    enforce_eager: bool,
    vocab_size: int,
    num_seqs: int,
) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(path, enforce_eager, vocab_size, num_seqs, q))
    p.start()
    p.join()
    r = q.get()
    gc.collect()
    torch.cuda.empty_cache()
    return r


# ——— main ———

def main(model_path: str) -> None:
    path = os.path.expanduser(model_path)
    hf_config = AutoConfig.from_pretrained(path)
    vocab_size = hf_config.vocab_size

    print(f"Model:  {hf_config.model_type}")
    print(f"Vocab:  {vocab_size}")
    print(f"Prompt: {PROMPT_LEN} tokens/seq  |  Output: {OUTPUT_LEN} tokens/seq")
    print()

    # header
    hdr = (f"{'BS':>5}  {'eager':>8}  {'graph':>8}  "
           f"{'speedup':>9}  {'eager':>8}  {'graph':>8}  "
           f"{'eager':>9}  {'graph':>9}")
    print(hdr)
    print(f"{'':>5}  {'tok/s':>8}  {'tok/s':>8}  "
          f"{'':>9}  {'TPOTms':>8}  {'TPOTms':>8}  "
          f"{'TTFTms':>9}  {'TTFTms':>9}")
    print("─" * len(hdr))

    rows = []
    for bs in BATCH_SIZES:
        print(f"  bs={bs:>3} ...", end=" ", flush=True)

        r_eager = _run_one(path, True,  vocab_size, bs)
        r_graph = _run_one(path, False, vocab_size, bs)

        tp_e  = r_eager["throughput"]
        tp_g  = r_graph["throughput"]
        sp    = tp_g / tp_e if tp_e > 0 else 0
        tpot_e = statistics.mean(r_eager["tpot_ms"]) if r_eager["tpot_ms"] else 0
        tpot_g = statistics.mean(r_graph["tpot_ms"]) if r_graph["tpot_ms"] else 0
        ttft_e = statistics.mean(r_eager["ttft_ms"]) if r_eager["ttft_ms"] else 0
        ttft_g = statistics.mean(r_graph["ttft_ms"]) if r_graph["ttft_ms"] else 0

        print("done")
        print(f"{bs:>5}  {tp_e:>8.0f}  {tp_g:>8.0f}  "
              f"{sp:>8.2f}x  {tpot_e:>8.2f}  {tpot_g:>8.2f}  "
              f"{ttft_e:>9.2f}  {ttft_g:>9.2f}")

        rows.append({
            "bs": bs,
            "tp_eager": tp_e,   "tp_graph": tp_g,   "speedup": sp,
            "tpot_e": tpot_e,   "tpot_g": tpot_g,
            "ttft_e": ttft_e,   "ttft_g": ttft_g,
        })

    # ——— summary ———
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")

    peak = max(rows, key=lambda r: r["speedup"])
    print(f"  Peak speedup        {peak['speedup']:.2f}x  at bs={peak['bs']}")

    if rows:
        avg_speedup = statistics.mean(r["speedup"] for r in rows)
        print(f"  Average speedup     {avg_speedup:.2f}x  across all batch sizes")

    small_bs = [r for r in rows if r["bs"] <= 8]
    large_bs = [r for r in rows if r["bs"] >= 32]
    if small_bs:
        print(f"  Small-batch (1-8)   avg speedup = {statistics.mean(r['speedup'] for r in small_bs):.2f}x")
    if large_bs:
        print(f"  Large-batch (32+)   avg speedup = {statistics.mean(r['speedup'] for r in large_bs):.2f}x")

    print(f"\n  TPOT reduction is most visible at small batch sizes")
    print(f"  where kernel-launch overhead dominates total time.")
    print(f"  At large batch sizes compute becomes the bottleneck.")
    print()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="CUDA Graphs vs Eager — batch-size sweep")
    args.add_argument("--model_path", type=str, default="~/huggingface/Qwen3-0.6B/")
    args = args.parse_args()
    main(args.model_path)
