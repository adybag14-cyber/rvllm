# Benchmark History

This file starts with the current public benchmark truth, then keeps older numbers only as historical context.

## Current Public Comparison (April 7, 2026)

Model: Qwen2.5-7B f16
GPU: H100 SXM 80GB
Harness: direct engine
Decode length: `output-len=128`

### vLLM 0.19.0 vs rvLLM

| N | vLLM 0.19.0 tok/s | rvLLM tok/s | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 167.5 | 132.7 | 0.79x |
| 32 | 4964.2 | 4494.9 | 0.91x |
| 64 | 9312.6 | 8503.4 | 0.91x |
| 96 | 13085.9 | 10530.6 | 0.80x |
| 128 | 16825.3 | 13718.1 | 0.82x |

### What changed to get here

Two things matter most:

1. **Batch-1 default-path fix**
   - normal `T=1` decode now defaults to the reusable `Batched` path
   - this is still the right architecture change even though current `N=1` is behind vLLM

2. **Batched GEMM policy fix**
   - `GemmStrategy::Hybrid` is now real instead of half-implied
   - current hybrid policy is:
     - QKV: cuBLAS / cublasLt
     - O-proj: cuBLAS / cublasLt
     - GateUp + SiLU: CUTLASS
     - Down-proj: cuBLAS / cublasLt

### Important correction

The earlier `89f`-era "rvLLM beats vLLM at `N=64`" claim is no longer treated as valid.

- the fast `89f` H100 run was real
- but that path was fast because the CUTLASS gate-aux FFN branch skipped the FFN down-projection
- the archived fast H100 CUTLASS library is still kept in the repo for forensic reproducibility

So the current public baseline is the clean current-`main` table above, not the older `9589 tok/s` claim.

### Earlier explicit batched strategy sweep

On the same H100 for `N=64`, `output-len=128`:

| Strategy | tok/s |
|---|---:|
| `cublas` | 7965.6 |
| `hybrid` | 8193.3 |
| `cutlass` | 7830.4 |

That sweep is why `Hybrid` is the current default when CUTLASS is available.

## Current Read of the Gap

- `N=1`: materially behind vLLM
- `N=32`: closer, but still behind
- `N=64`: still behind
- `N=128`: still behind by a wider margin

The biggest remaining work is:

- better single-stream decode
- a correct fast Hopper FFN path that does not skip work
- safer `cublasLt` autotune fallback when cached algos go bad
- more efficiency at `N=64` and `N=128`

## Historical Context

Older measurements below used different harnesses, older vLLM versions, or pre-fix architecture. Keep them as optimization history, not as the current headline.

### Earlier direct-engine comparison vs vLLM 0.6.3

| N | stock vLLM 0.6.3.post1 | rvLLM | rvLLM / vLLM |
|---:|---:|---:|---:|
| 1 | 133.7 | 120.6 | 0.90x |
| 4 | 543.3 | 427.9 | 0.79x |
| 8 | 926.1 | 845.8 | 0.91x |
| 16 | 1934.5 | 1648.9 | 0.85x |
| 32 | 3197.1 | 3170.0 | 0.99x |

### Earlier H100 direct-engine peak

This was a useful optimization waypoint, but not the current apples-to-apples comparison:

| N | rvLLM tok/s |
|---:|---:|
| 128 | 12312 |

### Earlier lifecycle / HTTP numbers

Those runs were useful for separating direct-engine performance from serving-stack overhead, but they were not re-run against `vLLM 0.19.0` and should not be treated as the current public baseline.
