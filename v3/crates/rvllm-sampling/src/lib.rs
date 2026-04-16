// rvllm-sampling — scaffold only. See v3/specs/13-sampling.md.
//   pub mod greedy;       // argmax_kernel launcher
//   pub mod topk_topp;    // combined sampling kernel launcher
//   pub mod dtoh_pinned;  // PinnedTokens double buffer + DtoHTicket<'p> type-state
//   pub mod rng;          // per-seq Philox state in HBM
