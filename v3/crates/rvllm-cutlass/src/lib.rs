// rvllm-cutlass — scaffold only. See v3/specs/11-cutlass-fp8.md.
//   pub mod variants;        // catalog: VariantId, tile, cluster, schedule pair
//   pub mod plan;            // Fp8GemmPlan (variant + shape + workspace size)
//   pub mod autotune_policy; // policy.json load; missing entry = engine init fail
//   pub mod workspace;       // workspace_bytes(&plan) -> usize
//   pub mod ffi;             // Cutlass::run(plan, args, ws, scope)
