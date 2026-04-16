# 07 — scheduler

## Scope
Owns request queues and the prefill/decode lifecycle. Produces an immutable `BatchPlan` per step. Touches no GPU memory, no metadata buffers, no kernels.

## v2 problems
- `crates/rvllm-v2/src/worker.rs:194-247` — `step()` calls `input_builder.build()` for mixed batches and `build_decode_only()` (path at `input.rs:52`) elsewhere; metadata offsets are reused across the two layouts. Today's `CUDA_ERROR_ILLEGAL_ADDRESS` is rooted here: capture used the padded decode layout, replay path uploaded the unpadded prefill layout.
- `crates/rvllm-v2/src/worker.rs:199` — `is_all_decode` branch downstream of scheduling. Two execution paths (graph replay vs `forward_greedy`) both depend on a runtime flag the scheduler computes implicitly.
- `crates/rvllm-v2/src/integration.rs:1-50` — scheduler, block manager, runner, worker, and tokenizer are wired together inside engine init; no clean seam between "what to run" and "how to run".
- `crates/rvllm-v2/src/scheduler.rs:18-28` — `SchedulerConfig::default()` ships silent magic (`max_num_seqs:256`, `max_prefill_chunk:0`, `block_size:64`); no bucket list anywhere — `padded_batch_size()` is hard-coded in worker.
- `crates/rvllm-v2/src/scheduler.rs:117` — `add_request` mutates queue mid-step via shared `&mut self`; ordering between `add_request` and `step` is undefined under async serve.

## v3 contract

```rust
// rvllm-runtime::scheduler

pub enum ReqState { Queued, Prefilling, Decoding, Finished(FinishReason) }

pub struct Request {
    pub id: ReqId,
    pub prompt: Arc<[TokenId]>,
    pub sampling: SamplingParams,
    pub priority: u32,                 // lower = preempt first
    pub max_new_tokens: u32,
    pub arrival: Instant,
    pub state: ReqState,
    pub generated: SmallVec<[TokenId; 8]>,
}

pub enum BatchPlan {
    Prefill(PrefillPlan),               // step() returns ONE variant only
    Decode(DecodePlan),
    Idle,
}

pub struct PrefillPlan {
    pub req: ReqId,                     // exactly one prefill per step (chunked)
    pub start: u32, pub end: u32,       // [start, end) of the prompt this chunk
    pub block_table: Arc<[BlockId]>,
}

pub struct DecodePlan {
    pub bucket: u32,                    // padded batch size, ∈ RuntimeConfig.decode_buckets
    pub seqs: Arc<[DecodeSeq]>,         // len == num_active; bucket - len = padding slots
    pub max_context: u32,               // monotonic per bucket; metadata sized off this
}

pub struct DecodeSeq {
    pub req: ReqId, pub seq_len: u32,
    pub block_table: Arc<[BlockId]>,
    pub last_token: TokenId,
}

pub struct StepOutput {
    pub req: ReqId,
    pub new_token: Option<TokenId>,     // None for prefill chunks that don't finish prompt
    pub finish: Option<FinishReason>,
    pub freed_blocks: SmallVec<[BlockId; 4]>,
}

pub struct Scheduler { /* opaque */ }

impl Scheduler {
    pub fn new(cfg: &RuntimeConfig, blocks: BlockManager) -> Self;
    pub fn enqueue(&self, req: Request);                        // lock-free MPSC; never blocks
    pub fn schedule(&mut self) -> Result<BatchPlan, RvllmError>; // drains intake at top
    pub fn commit(&mut self, plan: &BatchPlan, outs: &[RawSample]) -> Vec<StepOutput>;
}
```

Bucket selection: `decode_buckets: Vec<u32>` lives in `RuntimeConfig` (default in agent 02: `[1,2,4,8,16,32,48,64,96,128,160,192,256]`). `bucket = decode_buckets.iter().find(|b| **b >= num_active)` — error if none. No silent rounding.

Scheduling rule: drain intake → if any `Queued` request, return `Prefill(...)` of one chunk; else return `Decode(...)` of all `Decoding` reqs padded to chosen bucket. Never mixes.

Preemption: when `BlockManager::block_use() / num_blocks > 0.96`, preempt lowest-priority `Decoding` reqs (re-queue, transition `Decoding → Queued`, free blocks). Repeat until under watermark.

`enqueue` is lock-free; `schedule` snapshots intake at entry — new arrivals after that are picked up next call. No `add_request` mid-step.

## Failure modes
- `bucket` not found → `RvllmError::Scheduler(BucketTooSmall { num_active, max_bucket })`. No silent expand.
- Watermark > 1.0 or invalid bucket list at config build → `ConfigError`.
- `commit` called with `outs.len() != plan.num_sampled()` → panic (invariant we enforce).
- Preemption that cannot satisfy a single new prefill (all reqs preempted, still no room) → `RvllmError::Scheduler(KvExhausted)`.
- `schedule()` never blocks; empty queues + empty active → `BatchPlan::Idle`.

## Test plan
- Unit: bucket selection across `[0..=300]` matches `decode_buckets.find_first_geq`.
- Property: `schedule()` output is exactly one variant; `Prefill` xor `Decode` xor `Idle`.
- Preemption: synthetic block manager at 97% usage → asserts lowest-priority req re-queued, blocks freed match seq length.
- Concurrency: 1k threads `enqueue` while one thread loops `schedule/commit`; final state has every req either finished or queued, no losses.
- Determinism: seeded run produces identical `BatchPlan` sequence given identical arrivals.

## Cross-cutting deps
- 02-config: `decode_buckets`, `kv_watermark`, `max_batch`, `preemption_mode`.
- 03-errors: `SchedulerError { BucketTooSmall, KvExhausted, BadCommit }` variants of `RvllmError`.
- 04-memory: `BlockManager` API (`allocate`, `free`, `block_use`, `cow_if_needed`).
- 08-metadata: consumes `&BatchPlan` only; never sees `Request`.
- 14-graph: dispatches on `BatchPlan` variant; same shape per `(variant, bucket)` so capture is stable.
