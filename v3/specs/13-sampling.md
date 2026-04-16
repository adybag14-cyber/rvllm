# 13 — Sampling

## Scope
GPU-side sampling (greedy / top-k / top-p / categorical). The only DtoH per step is `[N] TokenId` via a double-buffered pinned staging area, gated by per-buffer `CUevent`s and a type-state handle.

## v2 problems
- `runner.rs:1342-1361` `read_graph_output` does `cuMemcpyDtoHAsync_v2` then `stream.synchronize()`, killing async DtoH; called from `engine::step()` while `step_pipelined` uses the correct `launch_dtoh`/`wait_dtoh` (`runner.rs:1366-1406`). Two DtoH paths.
- Token IDs are `i32` end-to-end; `slot_mapping` is also `i32` with `-1` sentinels — confusable, and OOB reads have been misread as token IDs.
- `runner.rs:1320-1335` argmax is on-GPU, but `temperature/top_k/top_p` have no kernel; OpenAI sampling params are silently dropped.
- No per-seq RNG state — non-greedy would round-trip seeds or share one batch seed.
- Logprobs path reads full `[N,vocab]` logits DtoH — the remaining DtoH bottleneck once argmax went on-GPU (SPEC.md table).

## v3 contract

```rust
// rvllm-core
#[repr(transparent)] pub struct TokenId(pub u32);   // always non-negative; NOT i32
pub struct SamplingParams {
    pub temperature: f32,    // 0.0 == greedy
    pub top_k: u32,          // 0 == disabled
    pub top_p: f32,          // 1.0 == disabled
    pub seed: u64,
    pub want_logprobs: u8,   // 0 or k>0
}

// rvllm-runtime::sampling
pub struct SamplerScratch<'a> {
    pub params:      Tensor<'a, SamplingParams>,   // [max_batch], packed
    pub rng_state:   Tensor<'a, u64>,              // [max_batch * 4], cuRAND Philox per seq
    pub argmax_dev:  Tensor<'a, TokenId>,          // [max_batch] device output
    pub topk_dev:    Tensor<'a, f32>,              // [max_batch * max_topk] optional
}
pub struct PinnedTokens { /* [A,B] pinned + [evt0,evt1] + write_idx */ }

pub trait Sampler {
    /// Pure on-device launch. Reads logits + params, writes argmax_dev. No DtoH.
    fn launch(
        logits:  &Tensor<'_, f16>,            // [N, vocab]
        params:  &Tensor<'_, SamplingParams>, // [N]
        rng:     &mut Tensor<'_, u64>,        // [N*4]
        out:     &mut Tensor<'_, TokenId>,    // [N]
        stream:  &ComputeStream,
    ) -> Result<(), RvllmError>;
}

// Type-state handle returned by launch_dtoh; consumed by wait_dtoh.
pub struct DtoHTicket<'p> { _p: &'p PinnedTokens, buf: u8, evt: EventHandle }
impl PinnedTokens {
    pub fn launch_dtoh(&mut self, src: &Tensor<'_, TokenId>, n: usize, s: &ComputeStream)
        -> Result<DtoHTicket<'_>, RvllmError>;
    pub fn wait(&self, t: DtoHTicket<'_>) -> Result<&[TokenId], RvllmError>; // n elements
}
```

### Kernels (signatures owned by agent 12)
- `argmax_kernel(logits[N,V], out[N])` — one block per row, warp-shuffle reduce. Single dispatch when all rows have `temperature==0.0`; lowest-index tie-break.
- `top_k_top_p_sample_kernel(logits, params, rng, out)` — one kernel: scale by `1/temperature`, partial top-k (k≤256), CDF top-p mask, cuRAND Philox draw → argmax of survivors. Updates `rng_state` in place. Branchless via `select`.
- `topk_logprobs_kernel(logits, k, out_topk[N,k])` — only enqueued when any seq sets `want_logprobs>0`.

### DtoH contract
- Per step, ONLY `[N] TokenId` (4·N bytes, ~512 B at N=128) leaves the GPU. `cuMemcpyDtoHAsync_v2` from `argmax_dev` → `pinned[buf]`, then `cuEventRecord` on `evt[buf]`; buffer index flips. Returns a `DtoHTicket` borrowing `&mut PinnedTokens` so the next launch cannot start until `wait` consumes the ticket (mirrors agent 05 `PendingStep`).
- `wait()` calls `cuEventSynchronize`. With `want_logprobs>0` a second async DtoH copies `topk_dev` into its own pinned region with its own event/ticket.
- Logits never DtoH. `read_graph_output` is deleted.

### RNG
Per-seq Philox state lives in HBM (`rng_state[N*4]`), seeded from `SamplingParams.seed` at admission via a tiny init kernel and advanced inside the sample kernel. No CPU round-trip.

## Failure modes
- `temperature<0`, `top_p∉(0,1]`, `top_k>max_topk`, `want_logprobs>max_topk` → `RvllmError::SamplingParams` at admission.
- DtoH or event-sync error → `RvllmError::Cuda`; no silent retry.
- `wait` called twice on same ticket → compile error (consumed by value).
- `n > max_batch` → `debug_assert!`.

## Test plan
- Unit: `argmax_kernel` vs CPU over 1k random rows; lowest-index tie-break.
- Property: `temperature→0` ⇒ `top_k_top_p_sample_kernel == argmax_kernel` byte-exact.
- Determinism: fixed seed ⇒ identical token sequences across 1000 replays.
- Distribution: χ² on 100k draws from a known logits row; p>0.01.
- Compile-fail (`trybuild`): reusing a `DtoHTicket`; reading `pinned[buf]` without `wait`.
- Bench: nsys shows exactly one 4·N-byte `cuMemcpyDtoHAsync_v2` per decode step, zero steady-state stream syncs.
- Regression vs v2: 1000 captured decode steps produce identical greedy token sequences.

## Cross-cutting deps
- 04-memory: `PinnedPool` for `pinned[A,B]`; `Tensor` for `argmax_dev`/`rng_state`/`params`/`topk_dev`, all `GraphSafe`.
- 05-concurrency: `DtoHTicket` is the per-step pinned analogue of `PendingStep`; `step_collect` calls `wait()`.
- 09-layer: LM-head GEMM writes `logits`; `Sampler::launch` is the next launch on the same stream.
- 12-fused: owns the kernel signatures listed above.
- 03-errors: `RvllmError::{SamplingParams, Cuda}`.
