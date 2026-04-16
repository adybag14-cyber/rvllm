// rvllm-runtime — scaffold only. See v3/specs/{07-scheduler,09-layer}.md.
//   pub mod engine;      // Engine::init / step_launch / step_collect
//   pub mod layer_exec;  // forward(input, weights, kv, scratch, meta, out, scope)
//   pub mod scheduler;   // BatchPlan::{Prefill, Decode, Idle}
//   pub mod sched_state; // Request state machine
//   pub mod lifecycle;   // admission, completion, preemption
