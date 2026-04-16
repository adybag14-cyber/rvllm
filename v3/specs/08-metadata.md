# 08 — Metadata

## Scope
Pack and upload per-step kernel metadata (token_ids, positions, context_lens, block_tables, slot_mapping, seq_start_pos) into one device region with capture-safe offsets that depend only on `(bucket, max_blocks)`.

## v2 problems
- `crates/rvllm-v2/src/runner.rs:785,982,1083` — THREE upload paths (`upload_metadata`, `upload_metadata_padded`, `patch_metadata_decode`) producing THREE `PackedMetaOffsets` against ONE captured graph.
- `runner.rs:919` `last_meta_offsets = Some(offsets)` is overwritten by whichever path ran last. Prefill writes non-padded offsets; the captured decode graph (built against padded offsets) reads garbage `block_tables` → `CUDA_ERROR_ILLEGAL_ADDRESS` -10 MB OOB inside `fa3_v3_decode_gqa_kernel`.
- `runner.rs:1112` `patch_metadata_decode` re-uploads `block_tables` only on `block_table_changed` (CoW copies). Normal page-boundary growth is a silent miss; captured kernel reads stale block IDs.
- `runner.rs:878,1034` realloc `meta_packed`/`pinned_meta` mid-step; captured graphs hold the freed pointer.
- Offsets depend on request content (`is_all_decode`, `actual` vs `padded_batch`); two requests at the same bucket yield different offsets.

## v3 contract

```rust
// Computed at engine init for each bucket. FROZEN.
pub struct MetadataLayout {
    pub bucket: usize, pub max_blocks: usize,
    pub token_ids:    Range<usize>, // [bucket]
    pub positions:    Range<usize>, // [bucket]
    pub context_lens: Range<usize>, // [bucket]
    pub block_tables: Range<usize>, // [bucket * max_blocks]
    pub slot_mapping: Range<usize>, // [bucket]
    pub seq_start_pos:Range<usize>, // [bucket + 1]
    pub total_i32:    usize,
}

pub struct MetadataView<'r> {
    pub layout:        &'r MetadataLayout,
    pub token_ids:     Tensor<'r, i32>,
    pub positions:     Tensor<'r, i32>,
    pub context_lens:  Tensor<'r, i32>,
    pub block_tables:  Tensor<'r, i32>,
    pub slot_mapping:  Tensor<'r, i32>,
    pub seq_start_pos: Tensor<'r, i32>,
}

pub struct Metadata { /* layout + slice of meta_packed */ }
impl Metadata {
    pub fn for_bucket(rt: &Runtime, bucket: usize) -> &Metadata; // FIXED offsets
}

// THE ONLY upload entry point. No patch path.
pub fn upload_for_bucket<'r>(
    plan: &BatchPlan, bucket: usize, rt: &'r Runtime,
) -> Result<MetadataView<'r>, RvllmError>;
```

Rules:
- `(bucket, max_blocks) -> offsets` is total. No `is_all_decode` branch, no `actual vs padded` branch. Two requests at same key have byte-identical offsets.
- `pinned_meta` and `meta_packed` sized at engine init for `(max_bucket, max_blocks_global)` where `max_blocks_global = ceil(max_context / kv_block_size) + 1`. Allocated once in `HbmArena` (agent 04). Refuse engine init if it does not fit.
- Every step: fresh full payload via `cuMemcpyHtoDAsync_v2` from pinned to `meta_packed` on the compute stream. Pad token_ids/positions/context_lens/block_tables with 0, slot_mapping with -1, seq_start_pos = `[0,1,…,bucket]`. One memcpy of `layout.total_i32 * 4` bytes.
- Block_tables ALWAYS uploaded fresh, no diff. Worst case (bucket=128, max_blocks=129): 64.5 KB. Total step payload <100 KB.
- Captured graphs bind `Metadata::for_bucket(rt, b)` once; replay reuses the same `&Metadata`.

## Failure modes
- `bucket > max_bucket` → `Err(RvllmError::BucketOob)`.
- `plan.num_seqs > bucket` → `Err(RvllmError::PlanExceedsBucket)`.
- Any seq's used blocks > `max_blocks` → `Err(RvllmError::ContextExceedsBucket)`.
- HtoD failure → `Err(RvllmError::Cuda)`.
- Engine init: pinned alloc fail or worst-case > arena → startup panic `RvllmError::MetaArenaInsufficient`.
- `debug_assert!` captured graph bucket key matches `MetadataLayout` key on replay.

## Test plan
- Unit: `MetadataLayout::compute(bucket, max_blocks)` pure; same inputs ⇒ byte-identical Ranges.
- Property: 1000 random `BatchPlan`s at fixed bucket ⇒ identical layout; all writes inside `[0, total_i32)`.
- Integration: capture at bucket=128, replay 256 steps with varying `num_seqs ∈ [1,128]` and growing context; compute-sanitizer memcheck clean.
- Regression: synthesize v2 failure (prefill then decode against same graph) — v3 must produce identical `block_tables` view both times.
- Bench: log payload bytes/step per bucket; assert <100 KB at bucket=128.

## Cross-cutting deps
- 04-memory: provides `meta_packed` Region in `HbmArena` and pinned staging; sizes from worst-case here.
- 07-scheduler: produces `BatchPlan` (raw arrays + bucket); never owns offsets.
- 14-graph: `CaptureScope` binds `Metadata::for_bucket(rt, b)` once; replay relies on `(bucket, max_blocks)`-determined offsets.
