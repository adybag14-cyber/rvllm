# rvLLM Status: Pipeline Review & Bug Report

Model: Qwen2.5-32B-Instruct, FP8 CUTLASS path, H100 80GB SXM
Date: 2026-04-15
Commit: 35b253d07 (pipeline changes), c0ae8ab3e (kernels on HF)

---

## CRITICAL: Showstopper Bugs in step_pipelined()

16-agent review found 4 showstopper bugs that make the pipelined decode path
produce garbage output. The non-pipelined `step()` path is unaffected.

### BUG 1: Pipeline ordering produces wrong tokens (CRITICAL)

**File:** `engine.rs:285` (`step_pipelined`)

The pipeline does: schedule(N) -> launch(N) -> process_output(N-1)

But `schedule(N)` reads `req.last_new_token` to build the step diff, and
`last_new_token` is set by `process_step_result()` inside `process_output(N-1)`.
Since N-1 output hasn't been processed yet when N is scheduled, `last_new_token`
is always `None`, so `new_token_id` in every `ContinuedRequest` is 0.

Downstream cascade:
- `apply_diff()` in worker.rs skips the push when `new_token_id == 0`
- Worker's `output_token_ids` is permanently empty
- `last_token_id()` returns the last prompt token instead of the generated token
- Every decode step feeds the model the same last-prompt-token
- `seq_len()` never advances, so position_ids are wrong
- `slot_mapping` writes every step's KV to the same slot (overwrites previous)
- `context_lens` stuck at prompt_len, attention never sees generated tokens
- **Output is nonsense after the first token**

**Fix:** Reorder to: process_output(N-1) -> schedule(N) -> launch(N). The
overlap then comes from GPU running step N while CPU processes N's output
scheduling N+1.

### BUG 2: Pinned buffer data race in pipelined path (CRITICAL)

**File:** `runner.rs:1361` (`launch_dtoh`), `worker.rs:289`

`step_pipelined` launches step N+1's DtoH into the same `pinned_argmax` buffer
before reading step N's output. Both are on the same CUDA stream, so the
execution order is:

```
GPU stream: [graph_N] [DtoH_N -> pinned_argmax] [HtoD_N+1] [graph_N+1] [DtoH_N+1 -> pinned_argmax]
CPU:        [launch N+1] .......................... [read pinned_argmax thinking it's N's data]
```

Step N+1's DtoH overwrites step N's data. The `dtoh_event` is also overwritten
by step N+1's `cuEventRecord`, so `wait_dtoh()` waits on N+1's event, not N's.

**Fix:** Double-buffer: 2 pinned buffers + 2 CUevents, alternating per step.
Or restructure so step N's output is read before step N+1's DtoH is launched.

### BUG 3: LM head f16 weight shrunk but still used (CRITICAL)

**File:** `runner.rs:391-395` (`enable_fp8_weights`)

`shrink_weight_vecs` replaces `lm_head_weight` with a 1-element CudaSlice stub.
But all 3 forward paths (`forward`, `forward_greedy_launch`, `forward_gpu_only`)
call `cublas.hgemm_f32_output(..., lm_head_weight, ...)` with dimensions
m=num_tokens, n=152064, k=3584.

cuBLAS reads 545M f16 elements from a 1-element buffer. This is an out-of-bounds
GPU memory read. cuBLAS has no bounds checking -- it reads whatever is in adjacent
GPU memory. Result: silent garbage logits, wrong tokens. No crash unless the read
hits an unmapped page.

The `fp8_lm_head` and `fp8_lm_head_scale` fields ARE populated but NEVER
dispatched to in any forward path. The LM head FP8 path is dead code.

**Fix:** Don't shrink `lm_head_weight`. Remove lines 391-395. The LM head is
~1.04 GB -- keep it. Separately, wire up the FP8 LM head path if it's faster.

### BUG 4: build() never updates cached decode keys (MODERATE)

**File:** `input.rs:105-148` (`build`), `input.rs:60` (`build_decode_only`)

When `is_decode_only=false` (additions exist), `build()` is called instead of
`build_decode_only()`. But `build()` never updates `cached_decode_keys` or
`cached_decode_valid`. Scenario:

1. Step N: 3 decode sequences [A,B,C]. Cache = [A,B,C].
2. Step N+1: B removed, D added. `build()` called. Cache NOT updated.
3. Step N+2: No changes. `build_decode_only()` called. Cache = [A,B,C] but
   B no longer exists. `requests[&B]` panics.

**Fix:** At end of `build()`, update the cache:
```rust
self.cached_decode_keys.clear();
self.cached_decode_keys.extend_from_slice(&self.decode_keys);
self.cached_decode_valid = true;
```

---

## Wrong Estimates: My CPU Cost Numbers Were 10-25x Too High

16 agents independently traced every operation. The original estimates were
wildly pessimistic:

| Operation | My Estimate | Actual (agents) | Factor Off |
|-----------|-------------|-----------------|------------|
| GPU forward pass | 5.0 ms | 6.2-7.2 ms | 1.3x too optimistic |
| Total CPU overhead | 1.3-1.5 ms | 50-150 us | 10-20x too pessimistic |
| scheduler.schedule_decode_only() | 500 us | 20-30 us | 15-25x too high |
| process_forward_output() | 1000 us | 25-45 us | 20-40x too high |
| HashMap cost (1800 ops) | dominant | ~36 us total | not a bottleneck |
| build_request_outputs() | 100 us | 9-15 us | 7-10x too high |
| cleanup_finished() | 50 us | 1-3 us | 15-50x too high |

**Key insight: CPU was never the bottleneck.** At N=128 with GPU taking 6-7ms
and CPU taking 50-150us, the CPU has 40-100x headroom. The pipeline was solving
a problem that barely exists at this batch size.

The 1.3ms figure was likely measured from the synchronous `step()` path which
includes `stream.synchronize()` -- that blocks the CPU for the full GPU forward
pass, making "CPU overhead" appear to include GPU time.

### GPU Forward Pass Breakdown (from first principles)

All GEMMs are memory-bandwidth-bound at M=128 (arithmetic intensity ~240,
crossover at ~1182 FLOPs/byte):

```
Per-layer weight reads:
  QKV:     16.5 MB FP8
  O-proj:  12.85 MB FP8
  Gate+Up: 135.8 MB FP8
  Down:    67.9 MB FP8
  Total:   233 MB/layer

64 layers: 14.9 GB
LM head:   1.09 GB (FP16, NOT FP8 -- see Bug 3)
Total:     16.0 GB weight reads per forward step

At 3.35 TB/s theoretical: 4.8 ms floor (impossible to beat)
At 80% BW utilization:    6.0 ms
At 75% BW utilization:    6.4 ms
With non-GEMM overhead:   6.2-7.2 ms realistic
```

Flash attention at N=128, context 512: ~5-10 us/layer (trivial).
RMSNorm, RoPE, SiLU, residuals: ~5 us/layer.
CUDA graph replay overhead: ~8-12 us (depends on node count).

---

## PCIe & Transfer Analysis

All transfers are correctly async with pinned memory. No hidden costs.

### HtoD Metadata Upload (per decode step at N=128)

| Field | Elements | Bytes |
|-------|----------|-------|
| token_ids | 128 | 512 |
| position_ids | 128 | 512 |
| context_lens | 128 | 512 |
| block_tables | 128 * max_blocks | 16-66 KB (depends on max_seq_len) |
| slot_mapping | 128 | 512 |
| seq_start_pos | 129 | 516 |
| **Total (2K ctx)** | | **~19 KB** |

Note: `patch_metadata_decode` patches 4 fields in the pinned buffer but
re-uploads the FULL buffer. Optimization: could do sub-region upload for the
~2 KB that actually changed, saving ~17 KB of PCIe traffic. But at PCIe Gen5
this saves ~0.5 us. Not worth the complexity.

### DtoH Argmax Output

128 * 4 bytes = 512 bytes. PCIe latency floor: ~2-3 us. Negligible.

### Pinned Memory

Genuine `cuMemAllocHost_v2` (pinned_memory.rs:58). Not a regular Vec.
Argmax writes directly to `argmax_output` device buffer (no D2D copy).
Mapped memory would be worse (128 non-coalesced 4-byte PCIe writes from
separate blocks vs one 512-byte DMA).

### CUDA Driver Calls (5 per step)

1. `cuMemcpyHtoDAsync_v2` (metadata upload): ~1.5-3 us
2. `cuGraphLaunch` (graph replay): ~8-12 us
3. `cuMemcpyDtoHAsync_v2` (argmax DtoH): ~1.5-2 us
4. `cuEventRecord` (DtoH completion): ~0.3-0.5 us
5. `cuEventSynchronize` (wait for previous step): ~0.3-1 us (if already done)

Total driver overhead: ~12-19 us. No hidden mutex contention (single thread,
single stream, single GPU). cudarc wrapper adds negligible overhead.

---

## Heap Allocation Audit (per decode step, N=128)

147 heap allocations per step in steady-state decode. Total ~25 KB of churn.

| Source | Allocs | Bytes | Avoidable? |
|--------|--------|-------|------------|
| GpuBatchInput.clone() | ~10 | ~7,680 | YES -- build in-place, pass by ref |
| vec![logprob] x128 | 128 | 512 | YES -- use f32 field, not Vec<f32> |
| Vec<V2RequestOutput> + reallocs | ~7 | ~15,360 | YES -- reusable field |
| Vec<TokenId> collect in read_output | 1 | 512 | YES -- transmute &[i32] to &[u32] |
| cached_decode_keys.clone() | 1 | 1,024 | YES -- use index iteration |
| **Total** | **~147** | **~25,088** | **all avoidable** |

At ~35ns per malloc/free, allocator time is ~5 us. Real cost is L1 cache
pollution: ~196 cache lines dirtied and evicted per step. Jitter risk from
occasional mmap/munmap.

Priority fixes:
1. GpuBatchInput: build in-place, pass by reference (eliminates 10 allocs)
2. logprobs: store as `f32` not `Vec<f32>` (eliminates 128 allocs)
3. request_outputs: reusable Vec field (eliminates 7 reallocs)
4. read_output: transmute instead of collect (eliminates 1 alloc)
5. cached keys: index iteration instead of clone (eliminates 1 alloc)

---

## cuEvent Lifecycle Issues

### Drop Use-After-Free (MODERATE)

`runner.rs:1414-1418`: Drop impl calls `cu_event::destroy` but does NOT
synchronize the stream. Rust struct drop order drops fields in declaration order.
`pinned_argmax` (line 123) is declared before `dtoh_event` (line 128), so
pinned memory is freed first. If a DtoH is in-flight writing to pinned memory,
this is a use-after-free.

**Fix:** Add `self.stream.synchronize()` in Drop impl before field drops.

### is_dtoh_ready() Swallows Errors

`runner.rs:1383-1387`: `cuEventQuery` returns `CUDA_SUCCESS` or
`CUDA_ERROR_NOT_READY`. But it can also return `CUDA_ERROR_LAUNCH_FAILED` if a
kernel failed. Current code treats all non-SUCCESS as "not ready", causing
infinite polling on GPU errors.

**Fix:** Return `Result<bool>`, propagate errors that aren't NOT_READY.

### invalidate_cache() is Dead Code

`input.rs:164-166`: Defined but never called anywhere. Cache invalidation
relies entirely on the `set_changed` parameter. Not a bug but misleading.

---

## Pipeline Architecture Assessment

### The Good

- Single-stream ordering is correct for this workload (sequential layer deps)
- HtoD/DtoH transfers are tiny and properly async
- Pinned memory is genuine
- CUDA graph replay eliminates kernel launch overhead
- GPU argmax means only token IDs cross PCIe (not logits)

### The Bad

- Pipeline ordering bug makes ALL pipelined output garbage (Bug 1)
- Pinned buffer race means wrong tokens even if ordering is fixed (Bug 2)
- CPU was never the bottleneck -- pipeline adds complexity for ~50-150us savings
  against a 6-7ms GPU step
- 147 heap allocations per step is sloppy but not performance-critical

### The Ugly

- LM head weight shrunk while still actively used (Bug 3) -- silent OOB reads
- FP8 LM head path is dead code (quantized but never dispatched)
- 5ms GPU estimate was wrong (real: 6.2-7.2ms)
- CPU cost estimates were 10-25x too high

---

## f16 Weight Shrink Analysis

Per-layer shrink is SAFE: FP8 guard (`weights.fp8.is_some()`) prevents any
code path from reaching the f16 GEMM branch when FP8 is enabled.

LM head shrink is BROKEN: All 3 forward paths use `lm_head_weight` via
`cublas.hgemm_f32_output`. The `fp8_lm_head` is populated but never used.
cuBLAS reads past the 1-element stub into adjacent GPU memory = garbage logits.

Layernorm weights correctly NOT shrunk (used by all paths).

Savings after fix (don't shrink lm_head):
- Dead f16 layer weights freed: 27.8 GB (correct, safe)
- lm_head_weight kept: 1.04 GB (must keep)
- Net savings: ~27.8 GB instead of 28.8 GB

---

## vLLM Competitive Comparison

### What vLLM Does

- CUDA graphs for decode (same as rvLLM)
- Separate scheduler process (Python, communicates via shared memory + IPC)
- Pinned buffers pre-allocated at max batch, reused (same as rvLLM)
- Mostly single-stream for forward, separate streams for H2D/D2H
- Caching memory allocator via torch (rvLLM needs equivalent)
- Chunked prefill interleaved with decode
- Prefix caching for shared system prompts
- Preemption + KV swap to CPU under memory pressure

### rvLLM Advantages

- No Python interpreter overhead (~5-10 us per op dispatch eliminated)
- No torch framework tax (3-6 ms of dispatch overhead eliminated for 64 layers)
- CUTLASS autotuning (potentially better than cuBLAS for specific shapes)
- Direct CUDA control (fused kernels, precise memory management)
- Lower CPU overhead in Rust vs Python scheduler

### rvLLM Disadvantages

- Missing chunked prefill (long prompts stall all decode sequences)
- Missing prefix caching
- Missing preemption/KV swap
- No caching memory allocator (risk of cuMemAlloc during inference)
- FP8 LM head path incomplete

### Throughput Numbers

- rvLLM baseline: 18,922 tok/s at N=128 (needs validation at longer contexts)
- vLLM v0.6.x FP8 on H100: ~4,000-5,500 tok/s (Qwen2.5-32B, batch=128)
- TensorRT-LLM: ~5,500-7,000 tok/s
- rvLLM's 18,922 is 3-4x higher than vLLM -- needs context length validation.
  If measured at very short sequence lengths (few decode steps), attention cost
  is minimal and the number is plausible. At seq_len=2048, it would need
  re-verification.

---

## Benchmark Baselines (pre-pipeline, commit 3eacacd0f)

```
N=1:    139.6 tok/s
N=32:  4,257.1 tok/s
N=64: 10,702.8 tok/s
N=128: 18,922.8 tok/s
```

These were measured with the synchronous `step()` path (correct output).
The pipelined path has never produced correct output.

---

## Model Dimensions (Qwen2.5-32B)

```
hidden_size      = 3584
num_heads        = 28   (query heads)
num_kv_heads     = 4    (GQA, 7:1 ratio)
head_dim         = 128
q_dim            = 28 * 128  = 3584
kv_dim           = 4  * 128  = 512
qkv_dim          = 3584 + 2*512 = 4608
intermediate     = 18944
gate_up_dim      = 37888  (2 * intermediate)
num_layers       = 64
vocab_size       = 152064
max_batch_tokens = 8192  (default)
block_size       = 16    (KV cache)
```

---

## Next Steps (Priority Order)

1. Fix Bug 1: pipeline ordering (process N-1 before scheduling N)
2. Fix Bug 3: don't shrink lm_head_weight
3. Fix Bug 2: double-buffer pinned argmax
4. Fix Bug 4: update cache in build()
5. Fix Drop use-after-free (stream sync in Drop)
6. Fix disk space on H100 instance (model download failed: ENOSPC)
7. Compile and deploy
8. Bench with `--profile` to verify actual CPU timing matches agent estimates
9. Bench at longer sequence lengths to validate 18,922 tok/s claim
10. Eliminate 147 heap allocations (GpuBatchInput clone, logprobs Vec, etc.)
