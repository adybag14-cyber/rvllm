// rvllm-mem — scaffold only. See v3/specs/04-memory.md and 05-concurrency.md.
// Modules to implement in Phase A:
//   pub mod hbm;          // HbmArena (one cuMemAlloc, bump-allocated Regions)
//   pub mod region;       // Region<'a>: offset/len/name into HbmArena
//   pub mod tensor;       // Tensor<'a, T>: shape/dtype view into Region
//   pub mod pinned;       // PinnedPool (cuMemAllocHost), double-buffered
//   pub mod kv_layout;    // [2, num_blocks, block_size, num_kv_heads, head_dim] stride
//   pub mod stream;       // Stream (single compute per worker)
//   pub mod event;        // Event wrapper for DtoH coordination
//   pub mod capture;      // CaptureScope: closure-scoped graph capture
//   pub mod graph_safe;   // unsafe trait GraphSafe — borrow into captured region
