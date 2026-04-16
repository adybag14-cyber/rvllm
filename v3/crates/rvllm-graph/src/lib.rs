// rvllm-graph — scaffold only. See v3/specs/14-graph.md.
//   pub mod capture;      // CaptureScope::record closure API (uses rvllm-mem CaptureScope)
//   pub mod replay;       // GraphPool::replay(bucket, scope) -> &Tensor<i32>
//   pub mod pool;         // GraphPool::capture_all(rt, buckets, body)
//   pub mod fingerprint;  // walk captured graph nodes; emit + compare JSON
//   pub mod validate;     // eager-mode numeric ref compare (CI)
