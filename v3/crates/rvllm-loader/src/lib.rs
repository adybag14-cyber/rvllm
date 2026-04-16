// rvllm-loader — scaffold only. See v3/specs/06-loader.md.
//   pub mod safetensors;  // HF mmap -> CPU staging
//   pub mod fp8_quant;    // GPU-side abs_max + write FP8 + scale, count clamps
//   pub mod placement;    // deterministic HBM placement: same (model, rt, gpu) = same addrs
