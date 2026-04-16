#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// rvllm-core — scaffold only. See v3/specs/{01,02,03}.md for contract.
// Modules to implement in Phase A:
//   pub mod error;   // 03-errors.md: RvllmError enum
//   pub mod ids;     // RequestId, SeqId, BlockId, TokenId newtypes
//   pub mod dtype;   // DType, Shape
//   pub mod config;  // 02-config.md: ModelConfig, RuntimeConfig, builder
//   pub mod env;     // whitelisted RVLLM_* vars
