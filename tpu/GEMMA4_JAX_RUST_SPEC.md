# Gemma 4 JAX/Rust TPU Pipeline

## Goal
JAX traces the forward pass and exports StableHLO with SPMD annotations.
Rust loads the artifact and runs inference via PJRT -- no Python at runtime.

## Architecture
```
JAX (one-time) -> jax.export() -> StableHLO .mlir artifact (with SPMD)
                                        |
Rust binary: load .mlir -> PJRT compile (4 devices) -> execute loop
  - Safetensors -> int8 quantize -> shard across devices
  - KV cache allocate + donate each step
  - Token loop: feed token_id, get next_token + updated caches
```

## Current State
- `tpu/harness/gemma4_tpu_infer.py`: working JAX inference, 68 tok/s, TP=4
- `crates/rvllm-xla/src/client.rs`: PJRT client, single-device only
- `crates/rvllm-xla/src/ffi.rs`: PJRT FFI types
- `crates/rvllm-xla/src/bin/infer_tpu.rs`: Llama 3.1 single-device inference (reference)

## Key Dimensions (Gemma 4 31B)
- H=5376, NH=32, INTER=21504, VOCAB=262144, NL=60
- Sliding: q=[MAX_Q=16384,H] k=[MAX_KV=4096,H] (padded from 8192/4096)
- Global: q=[16384,H] k=[4096,H] (padded from 16384/2048)
- Int8 weights + bf16 per-channel scales
- KV cache: [60, max_ctx, 4096] bf16, sharded along dim 2 (TP=4)
- TP=4 mesh, shard matmul weights along output/input dims

## Files to Create/Edit

| File | Agent | Description |
|---|---|---|
| tpu/harness/gemma4_export.py | 1 | JAX export script with SPMD |
| tpu/harness/gen_compile_options.py | 2 | Generate compile_options.pb for 4-device SPMD |
| crates/rvllm-xla/src/client.rs | 3 | Multi-device execute + buffer donation |
| crates/rvllm-xla/src/mesh.rs | 4 | Device mesh / topology |
| crates/rvllm-xla/src/artifact.rs | 5 | Load exported StableHLO artifacts |
| crates/rvllm-xla/src/gemma4_weights.rs | 6 | Safetensors -> int8 sharded buffers |
| crates/rvllm-xla/src/kv_cache.rs | 7 | KV cache allocation + donation |
| crates/rvllm-xla/src/bin/infer_gemma4_tpu.rs | 8 | Main inference binary |
| crates/rvllm-xla/src/lib.rs | 9 | Module wiring |
| tpu/GEMMA4_JAX_RUST_SPEC.md | 10 | This spec (review + update) |

## PJRT Multi-Device Changes (client.rs)
Current: `num_devices: 1` hardcoded in execute call (line 316).
Need:
1. `execute()` takes `num_devices` parameter
2. Support `num_replicas=1, num_partitions=4` in compile options
3. Buffer donation flags in execute (input_buffer_indices to donate)
4. Multiple argument lists (one per device) for sharded inputs

## JAX Export
Use `jax.export.export()` to serialize the forward function:
```python
exported = jax.export.export(forward)(
    token_id, pos, ctx, embed, final_norm, weights, caches,
    cos_s, sin_s, cos_g, sin_g)
exported.save("gemma4_step.mlir")
```
The exported artifact includes SPMD sharding annotations.

## Int8 Weight Loading (Rust)
Same quantization as Python: per-channel absmax/127, store int8 + bf16 scale.
Read safetensors, quantize, create PJRT buffers on correct devices per shard.

## KV Cache
Allocate [60, max_ctx, 4096] bf16 on device, sharded along dim 2.
Each step: donate old caches, get new caches from execute output.
PJRT buffer donation: mark input indices as donated in execute call.
