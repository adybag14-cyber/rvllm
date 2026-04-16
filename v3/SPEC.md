# rvllm v3 — clean rewrite plan

## Why v3

v2 evidence (today's bench):
- Engine deadlocks/crashes on graph replay with `CUDA_ERROR_ILLEGAL_ADDRESS` because:
  - `forward_greedy_launch` (prefill) overwrites `last_meta_offsets` with non-padded layout, then the captured graph (which expects padded offsets) reads garbage block_tables → -10 MB OOB in `fa3_v3_decode_gqa_kernel`.
  - `patch_metadata_decode` skips block_tables on normal page-boundary growth (only checks CoW copies in `diff.block_ops.copies`), silently producing wrong KV reads for any output_len > 64 tokens.
  - `cutlass_fp8_gemm_residual_v0` pairs non-cooperative WS mainloop with cooperative epilogue → crashes only inside captured graph.
  - `max_workspace_size` queries autotuned variants only, missing the fallback `fp8_gemm`/`fp8_gemm_small` workspaces that the dispatch actually calls when the FP8 autotune cache is empty.
- Frankenstein dispatch (autotune → small → default → cuBLASLt → cuBLAS) with implicit assumptions at each layer, so silent regressions are routine.
- Two execution paths (sync `step()` and `step_pipelined()`) that don't share metadata invariants.
- `runner.rs` is 1,432 lines; `layer.rs` is 2,000+ lines. Coupled state, hidden mutation across calls, no enforceable invariants.
- Today's actual numbers (e9c523d91, FP8, output_len=512, H100 SXM):

  | N | tok/s |
  |---|---|
  | 1 | 65.0 |
  | 4 | 259.1 |
  | 8 | 518.3 |
  | 16 | 1,036.1 |
  | 32 | 2,055.8 |
  | 64 | 3,748.0 |
  | 128 | 9,530.7 |

  vs April 15 baseline 19,259 @ N=128 (now half), vs vLLM ~22-29K.

## v3 design tenets

1. **No fallbacks.** Misconfiguration panics with a typed reason, not a silent slow path. Fallback-on-failure is forbidden in compute paths; fallback-by-config is allowed if explicitly requested.
2. **Single execution path.** Graph capture is a transparent property of the runtime. There is exactly one decode codepath; capture wraps it. No "graph vs no-graph" duality.
3. **Invariants enforced at boundaries.** Every public function declares its preconditions and postconditions in the type system or with `debug_assert!`. Callers cannot violate them.
4. **No hidden state mutation.** Buffers used inside captured graphs are immutable references; allocators that can realloc are not allowed inside the capture region.
5. **Rust + Zig.** Rust for orchestration, type-safe FFI, async pipeline. Zig for SIMD-heavy CPU bits where Rust auto-vec is weak (BPE, sampling top-k, metadata pack). C++ remains for CUTLASS templates only.
6. **No runtime introspection of which dispatch path was taken.** The choice is logged at startup; runtime stays on it. Autotune produces a frozen policy file.
7. **Failure surface.** All errors are `Result<T, RvllmError>` with a structured cause. No `String` errors. No `Box<dyn Error>`. No `?` swallowing meaning.
8. **CI-gated correctness.** Every PR runs (a) numeric parity vs HF reference (cosine ≥ 0.999 per layer), (b) end-to-end perplexity within 0.5%, (c) graph-replay smoke test on H100, (d) compute-sanitizer clean.
9. **Explicit ownership.** Crate split mirrors the dependency DAG. No crate depends on a sibling for types. `core` defines types, others use them.
10. **Build artifacts are SHA-pinned.** Every binary, .so, .ptx, autotune.json is named by content hash. Deploys reject any drift.

## Crate layout (target)

```
v3/
  rvllm-core      # error types, IDs, Tensor descriptor, no GPU deps
  rvllm-mem       # HBM allocator, pinned host buffers, KV cache layout
  rvllm-kernels   # PTX/CUBIN/.so wrappers, kernel signatures, no orchestration
  rvllm-cutlass   # FP8 GEMM variants, autotune cache, workspace contracts
  rvllm-attention # FA3 paged decode, FA prefill, GQA support
  rvllm-graph     # CUDA graph capture/replay, bucket pool, validation
  rvllm-runtime   # model load, layer execute, sampling, scheduling
  rvllm-serve     # HTTP/gRPC, OpenAI-compatible API
  rvllm-bench     # bench harness, profile mode, regression gates
  rvllm-deploy    # tarball, vast.ai/runpod, deploy_and_bench
  rvllm-zig       # Zig SIMD: tokenizer, sampling, metadata pack
```

Strict DAG: `core` ← `mem` ← `kernels` ← `cutlass`+`attention`+`graph` ← `runtime` ← `serve`/`bench`. No cycles.

## Agent roster (16)

Each agent writes ONE markdown file at `v3/specs/<NN>-<name>.md`, ≤500 words, focused. Output format below.

| # | Name | Owns | Deliverable |
|---|---|---|---|
| 01 | architecture | crate boundaries, public APIs, dependency DAG, module list per crate | `01-architecture.md` |
| 02 | config | model config, runtime config, validation, no-defaults policy, env vars allowed | `02-config.md` |
| 03 | error-model | typed error enum, structured failure reporting, panic vs Err policy | `03-errors.md` |
| 04 | memory | HBM allocator, no-realloc invariant, KV cache layout, pinned buffers, lifetime rules | `04-memory.md` |
| 05 | concurrency | streams, events, async pipeline, graph-safe vs not, no-aliasing rules | `05-concurrency.md` |
| 06 | model-loader | HF safetensors → GPU layout, FP8 quantization at load, weight ownership, deterministic placement | `06-loader.md` |
| 07 | scheduler | request lifecycle, prefill/decode split, bucket selection, preemption, no metadata coupling | `07-scheduler.md` |
| 08 | metadata | how kernel metadata (positions, context_lens, block_tables, slot_mapping) is packed and uploaded; capture-safe layout | `08-metadata.md` |
| 09 | layer-exec | layer forward as a pure function of (input, weights, kv_state, scratch); no hidden mutation | `09-layer.md` |
| 10 | attention | FA3 paged decode + prefill + GQA dispatch contract; kernel signature, predication rules | `10-attention.md` |
| 11 | cutlass-fp8 | variant catalog, schedule/epilogue compatibility matrix, autotune cache format, workspace contract | `11-cutlass-fp8.md` |
| 12 | fused-kernels | fused-add-norm-quant, RoPE+KV-write, SiLU*mul+quant, LM-head+argmax — kernel signatures, fusion boundaries | `12-fused.md` |
| 13 | sampling | greedy, top-k, top-p, GPU-side, async DtoH with double-buffered pinned argmax, event coordination | `13-sampling.md` |
| 14 | graph-runtime | CUDA graph capture/replay, bucket pool, what's allowed inside capture, validation harness | `14-graph.md` |
| 15 | validation | numeric parity vs HF, per-layer cosine, end-to-end perplexity, compute-sanitizer in CI, golden traces | `15-validation.md` |
| 16 | deploy-bench | SHA-pinned tarball, autotune in CI, kernel build outside the box, deploy-and-bench in one shot | `16-deploy.md` |

## Agent output format

Every spec MD must include:
1. **Scope** — one sentence, what this agent owns.
2. **v2 problems** — 3-5 bullets, file:line references in `crates/rvllm-v2/` showing what broke. Use today's failures (above) as evidence.
3. **v3 contract** — public types, signatures, invariants. Code blocks where helpful.
4. **Failure modes** — what panics, what returns Err, what's a debug_assert.
5. **Test plan** — what unit/integration tests prove the invariant holds.
6. **Cross-cutting deps** — which other agent's MD they depend on (one line each).

Length: ≤500 words. No fluff. Code sketches preferred over prose.

## Build flow after specs

1. Review all 16 specs for conflicts. Reconcile in `v3/IMPL_PLAN.md`.
2. Write crate scaffolds (Cargo.toml, lib.rs stubs).
3. Build bottom-up: core → mem → kernels → cutlass+attention+graph → runtime → bench.
4. Each crate ships with its tests. No crate merges to main without compute-sanitizer pass.
5. First end-to-end milestone: Qwen2.5-7B FP8 N=1 greedy correctness vs HF, then graph capture, then N=128 throughput.
