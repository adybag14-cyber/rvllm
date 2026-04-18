#!/usr/bin/env python3
"""TPU throughput benchmark for Qwen3-8B and Mistral-7B-v0.3 on v6e-1.

Uses manual token-by-token decode with torch_xla.sync() per step.
torch 2.8.0 + torch_xla 2.8.0 + libtpu 0.0.17
"""

import gc
import time
import torch
import torch_xla
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("Mistral-7B-v0.3", "/tmp/mistral-7b"),
    ("Qwen3-8B", "/tmp/qwen3-8b"),
]
BATCH_SIZES = [1, 8, 16, 64]
INPUT_LEN = 16
OUTPUT_LEN = 128
WARMUP_DECODE_STEPS = 4
DTYPE = torch.bfloat16


def p(msg):
    print(msg, flush=True)


@torch.no_grad()
def decode_loop(model, input_ids, attention_mask, n_tokens, device):
    """Prefill + token-by-token decode with sync per step."""
    bsz = input_ids.shape[0]

    # Prefill
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    torch_xla.sync()
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    cur_mask = torch.cat(
        [attention_mask, torch.ones(bsz, 1, dtype=attention_mask.dtype, device=device)],
        dim=1,
    )

    # Decode
    for _ in range(n_tokens - 1):
        out = model(
            input_ids=next_tok,
            attention_mask=cur_mask,
            past_key_values=past,
            use_cache=True,
        )
        torch_xla.sync()
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(bsz, 1, dtype=cur_mask.dtype, device=device)],
            dim=1,
        )

    return n_tokens * bsz


def bench_model(name, path, device):
    p(f"\n{'='*60}")
    p(f"  {name}")
    p(f"{'='*60}")

    p("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.vocab_size

    p("  Loading model (bf16, low_cpu_mem_usage)...")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    torch_xla.sync()
    p("  Model on TPU.")

    results = []

    for bs in BATCH_SIZES:
        p(f"\n  --- BS={bs} ---")
        try:
            ids = torch.randint(100, vocab - 100, (bs, INPUT_LEN), dtype=torch.long, device=device)
            mask = torch.ones_like(ids)

            # Warmup: prefill compile
            p(f"    warmup prefill...")
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, use_cache=True)
            torch_xla.sync()
            p(f"    prefill compiled in {time.perf_counter()-t0:.1f}s")

            # Warmup: decode compile (a few steps)
            p(f"    warmup decode ({WARMUP_DECODE_STEPS} steps)...")
            past = out.past_key_values
            tok = out.logits[:, -1:, :].argmax(dim=-1)
            cmask = torch.cat([mask, torch.ones(bs, 1, dtype=mask.dtype, device=device)], dim=1)
            t0 = time.perf_counter()
            for w in range(WARMUP_DECODE_STEPS):
                with torch.no_grad():
                    out = model(input_ids=tok, attention_mask=cmask, past_key_values=past, use_cache=True)
                torch_xla.sync()
                past = out.past_key_values
                tok = out.logits[:, -1:, :].argmax(dim=-1)
                cmask = torch.cat([cmask, torch.ones(bs, 1, dtype=cmask.dtype, device=device)], dim=1)
            p(f"    decode compiled in {time.perf_counter()-t0:.1f}s")

            del out, past, tok, cmask
            gc.collect()

            # Timed run: fresh input
            ids = torch.randint(100, vocab - 100, (bs, INPUT_LEN), dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            torch_xla.sync()

            p(f"    generating {OUTPUT_LEN} tokens...")
            t0 = time.perf_counter()
            gen = decode_loop(model, ids, mask, OUTPUT_LEN, device)
            elapsed = time.perf_counter() - t0

            tps = gen / elapsed
            p(f"    {gen} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
            results.append((bs, gen, elapsed, tps))

        except Exception as e:
            err = str(e).lower()
            p(f"    Error: {e}")
            if any(k in err for k in ["out of memory", "oom", "alloc", "resource"]):
                p(f"    OOM -- stopping batch sweep")
                gc.collect()
                torch_xla.sync()
                break
            gc.collect()
            torch_xla.sync()
            continue

    p(f"\n  {'='*56}")
    p(f"  RESULTS: {name} | TPU v6e-1 bf16")
    p(f"  Prefill {INPUT_LEN} tok -> Decode {OUTPUT_LEN} tok")
    p(f"  {'='*56}")
    p(f"  {'BS':>4}  {'Total tok':>10}  {'Time(s)':>8}  {'tok/s':>10}")
    p(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*10}")
    for bs, tok, t, tps in results:
        p(f"  {bs:>4}  {tok:>10}  {t:>8.2f}  {tps:>10.1f}")
    p("")

    del model
    gc.collect()
    torch_xla.sync()
    return results


def main():
    device = torch_xla.device()
    p(f"Device: {device}")
    p(f"PyTorch {torch.__version__} + torch_xla {torch_xla.__version__}")
    p(f"bf16, input={INPUT_LEN} tok, output={OUTPUT_LEN} tok")
    p(f"Batch sizes: {BATCH_SIZES}")

    all_results = {}
    for name, path in MODELS:
        all_results[name] = bench_model(name, path, device)

    p("\n" + "=" * 60)
    p("  COMBINED RESULTS -- TPU v6e-1 (bfloat16)")
    p("  PyTorch + torch_xla, manual decode")
    p("=" * 60)
    for name, results in all_results.items():
        p(f"\n  {name}:")
        p(f"  {'BS':>4}  {'tok/s':>10}")
        p(f"  {'-'*4}  {'-'*10}")
        for bs, _, _, tps in results:
            p(f"  {bs:>4}  {tps:>10.1f}")
    p("")
    p("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
