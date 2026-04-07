# Current Forward Model (April 7, 2026)

This is the current high-level forward-path model for `rvLLM`, not the older pre-fusion trace.

The two important ideas now are:

1. Normal decode and batched execution now converge on one canonical default lane.
2. Legacy single-token decode escapes are no longer part of the default path.

## Current Path Selection

### Batch-1 (`T=1`)

Normal batch-1 decode now defaults to:

```text
BatchedV2
```

The selection order is now:

```text
explicit experimental env paths
-> BatchedV2 (default normal path)
```

Current verified number on H100 / Qwen2.5-7B / `output-len=128`:

- `N=1`: `133.1 tok/s`

### Batched (`T>=2`)

Batched prefill and batched decode use the same normal `BatchedV2` layer stack plus an explicit GEMM policy.

Current default path:

```text
BatchedV2 + GemmStrategy::Hybrid
```

Hybrid means:

```text
QKV        -> cuBLAS / cublasLt
O-proj     -> cuBLAS / cublasLt
GateUp     -> CUTLASS
SiLU       -> fused with GateUp CUTLASS epilogue
Down-proj  -> cuBLAS / cublasLt
```

The old bug was that the runner conceptually wanted this hybrid policy, but the actual enum and dispatch did not encode it cleanly. As a result, QKV could still wander onto CUTLASS just because the shared library was present.

That is fixed now.

## Current Layer Shape

### Default decode

Per layer:

```text
RMSNorm
QKV projection via cuBLAS / cublasLt
RoPE + KV cache write
attention decode
O-proj via cuBLAS / cublasLt
RMSNorm
GateUp + SiLU via CUTLASS
down via cuBLAS / cublasLt
```

### Batched decode / prefill

Per layer:

```text
RMSNorm
QKV via cuBLAS / cublasLt
bias / layout handling as needed
RoPE + cache update
attention backend
O-proj via cuBLAS / cublasLt
residual + RMSNorm
GateUp + SiLU via CUTLASS
down via cuBLAS / cublasLt
```

## Current Benchmark Truth

Same H100, same Qwen2.5-7B snapshot, `output-len=128`, direct engine:

| N | vLLM 0.19.0 | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 167.5 | 132.7 | 0.79x |
| 32 | 4964.2 | 4494.9 | 0.91x |
| 64 | 9312.6 | 8503.4 | 0.91x |
| 96 | 13085.9 | 10530.6 | 0.80x |
| 128 | 16825.3 | 13718.1 | 0.82x |

## What Is Still Behind

- `N=1` decode is still materially behind vLLM.
- `N=32` and `N=64` are improved, but still behind.
- `N=128` is still meaningfully behind.
- the earlier `89f` gate-aux `N=64` win was invalid because that path skipped FFN down-proj.
- `cublasLt` autotune cache behavior is still flaky on some shapes and should fall back more aggressively when a cached algo goes bad.

## Relevant Controls

```bash
RVLLM_BATCHED_GEMM_STRATEGY=cublas|hybrid|cutlass
RVLLM_PERSISTENT=1
RVLLM_PERSISTENT_V2=1
RVLLM_PERSISTENT_V3=1
RVLLM_MEGAKERNEL=1
RVLLM_MEGAKERNEL_V2=1
RVLLM_FP8_WEIGHTS=1
```

## Bottom Line

The current system is no longer “legacy single-token decode by default, batched as a side path.”

It is:

- normal decode and batched execution on `BatchedV2`
- explicit hybrid GEMM policy inside that lane
- experimental persistent and megakernel paths kept separate from the normal path
