# 15 — validation

## Scope
Numeric parity vs HF, per-layer cosine, perplexity, compute-sanitizer, golden traces, bench gate. All under `tools/`, never linked into runtime crate.

## v2 problems
- Vectorized quantize kernel (`crates/rvllm-v2/src/kernels/quantize.rs`) shipped without a per-output cosine harness — direct path to `CUDA_ERROR_ILLEGAL_ADDRESS` in capture.
- Graph metadata layout drift (`worker.rs` `patch_metadata_decode` vs `forward_greedy_launch`) was undetectable: no fingerprint of `(kernel, arg_ptrs, arg_scalars)` per captured node, so prefill→decode swap silently corrupted `block_tables`.
- "Always-upload commit" produced symptom-free wrong output (grammatical garbage). No reference-token comparison ran in CI; regression merged.
- `engine.rs` had two execution paths — neither had per-layer activation diff vs HF; integration tests checked tok/s only.
- compute-sanitizer never ran in v2 CI; `fa3_v3_decode_gqa_kernel` OOB only surfaced under live H100 load.

## v3 contract

```
tools/parity         # per-layer cosine vs HF
tools/perplexity     # WikiText-2 nats/byte
tools/smoke          # 50-prompt token-equality
tools/bench-gate     # N=1,8,32,128 ±2% vs last green
tools/sanitize       # wraps bench under compute-sanitizer
tools/fingerprint    # diffs captured-graph fingerprints across runs
```

Per-layer parity:
- `parity --model qwen2.5-7b --prompt canonical.txt --dtype {fp8,f16}` runs HF (eager, fp32 accum) and rvllm v3 with layer hooks (agent 09) writing `(layer, tensor) → fp32` to `runs/<sha>/parity/`.
- Cosine threshold: **≥ 0.999** FP8, **≥ 0.9999** f16. Below = hard fail with `(layer, tensor, cos)`.
- Reference traces CHECKED IN at `tools/parity/golden/<model>/<dtype>.safetensors`. Refresh = explicit PR.

End-to-end perplexity:
- `perplexity --model X --corpus wikitext-2 --tokens 10000` reports nats/byte. CI gate: within **0.5%** of HF reference at `tools/perplexity/golden/<model>.json`.

Smoke corpus:
- 50 prompts × 64 output tokens, `temperature=0`. Token equality vs HF greedy. **First divergence allowed only after token 32** for FP8; f16 must match all 64.

compute-sanitizer:
- Every PR runs `--tool memcheck` over **one iteration at every bucket** (N=1,8,32,128). ZERO errors.
- Subset `--tool racecheck` on captured graph for N=32, nightly. Zero races.

Capture validation:
- Each captured graph emits `sha256(kernel || arg_ptrs || arg_scalars)` per node, ordered topo. `fingerprint --baseline runs/<green-sha>` diffs vs new run; drift = CI fail with diverging node.

Bench regression:
- `bench-gate` runs N∈{1,8,32,128}, output_len=512, FP8. Compares vs `runs/<last-green-sha>/bench.json`. **±2% tolerance**, regression blocks merge. Improvement updates baseline post-merge.

## Failure modes
- Cosine below threshold → `Err(ParityFail { layer, tensor, cos })`. Exit non-zero.
- Perplexity drift > 0.5% → `Err(PerplexityDrift { ref, got })`.
- Sanitizer non-zero → propagate report path, exit non-zero.
- Missing golden → `Err(MissingGolden)`. Never auto-generate.
- Fingerprint diff → `Err(GraphDrift { node, old, new })`.
- `debug_assert!(layer_hook.count == hf_count)` in parity tool.

## Test plan
- Unit: cosine fn, fingerprint hash, perplexity accumulator (CPU-only, deterministic).
- Integration: parity on 2-layer toy model; smoke on Qwen2.5-0.5B before 7B.
- CI: H100 per PR, single full sweep (parity+perplexity+smoke+sanitizer+bench-gate). Nightly main: full sweep + racecheck + fingerprint diff vs prior nightly.

## Cross-cutting deps
- agent 06: deterministic loader for canonical FP8 model.
- agent 09: per-layer activation hooks exposing fp32 tensors.
- agent 14: graph-fingerprint API (`Graph::fingerprint() -> [NodeFp]`).
- agent 16: H100 CI runner, SHA-pinned tarball, `runs/<sha>/` artifact upload.
