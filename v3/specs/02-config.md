# 02 — config

## Scope
`ModelConfig` (loaded from HF `config.json`) and `RuntimeConfig` (built via validating builder). No `Default::default()` for anything load-bearing; whitelisted env vars only.

## v2 problems
- `crates/rvllm-v2/src/integration.rs:73` — `block_size: 64` as silent `Default`. Caused today's `graph_max_blocks` math to mismatch callers assuming 16.
- `crates/rvllm-v2/src/integration.rs:58-90` — `impl Default for V2EngineConfig` ships 17 silent values (`max_model_len: 2048`, `max_num_seqs: 256`, `watermark: 0.04`, `gpu_memory_utilization: 0.90`, …). Forgotten fields filled silently.
- `crates/rvllm-v2/src/integration.rs:103-111` — `from_model_path` uses `unwrap_or(32)`/`unwrap_or(4096)` for missing HF fields. A typo in `config.json` ships a silently wrong model.
- `crates/rvllm-v2/src/bin/bench.rs:141` — `cli.max_model_len.unwrap_or(2048)` silently overrides the model.
- `crates/rvllm-v2/src/scheduler.rs:18`, `worker.rs:46`, `block_manager.rs:14` — three more `Default` impls duplicating magic numbers; updating one desyncs the others.
- `crates/rvllm-v2/src/integration.rs:86` — `RVLLM_NO_GRAPH` read ad-hoc inside a `Default` body. No whitelist; `RVLLM_GEMV_FP8`, `RVLLM_FP8_KV`, `RVLLM_KERNEL_DIR`, `RVLLM_CUTLASS_AUTOTUNE_CACHE`, `RVLLM_PHASE_PROFILE_BATCHES`, `RVLLM_BATCHED_GEMM_STRATEGY` all read straight from `std::env` deep in subsystems. `RVLLM_NOGRAPH=1` (typo) silently ignored.

## v3 contract

```rust
// rvllm-core::config

/// Loaded from HF config.json. Every field REQUIRED. NO Default impl.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture:        ModelArch,    // Qwen2 | Llama3 | …
    pub hidden_size:         usize,
    pub num_layers:          usize,
    pub num_attention_heads: usize,
    pub num_kv_heads:        usize,
    pub head_dim:            usize,        // computed; verified hidden_size == num_attention_heads * head_dim
    pub intermediate_size:   usize,
    pub vocab_size:          usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps:        f32,
    pub rope_theta:          f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype:         DType,        // bf16 | f16
}

impl ModelConfig {
    /// Field-by-field parse. NO `serde(default)`. Missing field → ConfigError::MissingHfField{ name }.
    pub fn load_hf(dir: &Path) -> Result<Self, ConfigError>;
}

/// Runtime knobs. NO Default impl. Build only via RuntimeConfigBuilder.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub device_id:               u32,
    pub max_batch:               u32,   // ≤ 256
    pub max_context:             u32,   // ≤ ModelConfig::max_position_embeddings
    pub kv_block_size:           u32,   // ∈ {16, 32, 64}
    pub num_gpu_blocks:          u32,   // > 0
    pub num_cpu_blocks:          u32,
    pub gpu_memory_utilization:  f32,   // (0.0, 0.95]
    pub fp8_weights:             bool,
    pub fp8_kv_cache:            bool,
    pub graph_capture:           GraphMode,           // Off | Buckets(Vec<u32>)
    pub preemption:              PreemptionMode,
    pub log_level:               LogLevel,            // ONLY field with a Default; cosmetic
}

#[derive(Default)]
pub struct RuntimeConfigBuilder { /* Option<T> per field */ }

impl RuntimeConfigBuilder {
    pub fn build(self, model: &ModelConfig) -> Result<RuntimeConfig, ConfigError>;
}

#[derive(Debug)]
pub enum ConfigError {
    MissingHfField   { name: &'static str, file: PathBuf },
    HfTypeMismatch   { name: &'static str, expected: &'static str },
    MissingField     { name: &'static str },
    InvalidField     { name: &'static str, reason: String },
    UnknownEnvVar    { name: String },
    Inconsistent     { reasons: Vec<String> },  // accumulates ALL invalid fields
}
```

`build()` validates: `kv_block_size ∈ {16,32,64}`, `0 < max_batch ≤ 256`, `0 < max_context ≤ model.max_position_embeddings`, `num_gpu_blocks ≥ ceil(max_batch*max_context/kv_block_size)`, `0.0 < gpu_memory_utilization ≤ 0.95`, `fp8_kv_cache ⇒ kv_block_size ≥ 32`. ALL failures collected into `Inconsistent { reasons }`; never short-circuits.

## Env-var whitelist
Only these read at startup; any other `RVLLM_*` aborts with `UnknownEnvVar`:
```
RVLLM_LOG            // log level
RVLLM_NO_GRAPH=1     // disables graph capture
RVLLM_KERNEL_DIR     // path
RVLLM_AUTOTUNE_CACHE // path
RVLLM_DEVICE         // u32
```
List in `rvllm-core::config::ENV_WHITELIST: &[&str]`. Scanned ONCE in `RuntimeConfig::from_env_and_builder`.

## Failure modes
- HF missing/wrong-typed field → `Err(MissingHfField | HfTypeMismatch)`.
- Builder missing/invalid field → `Err(MissingField | InvalidField)`; multiple → `Inconsistent`.
- Unknown `RVLLM_*` env var → `Err(UnknownEnvVar)`.
- `debug_assert!` only for invariants the builder proved (e.g. `kv_block_size > 0` in hot paths).
- No `panic!` after `Ok(RuntimeConfig)`; partial config never escapes.

## Test plan
- Golden HF round-trip for Qwen2.5-7B, Llama-3.1-8B; mutate one field → `MissingHfField`.
- Builder fuzz: omit/invalidate random fields → `Inconsistent.reasons` lists every offender.
- Whitelist: `RVLLM_NOGRAPH=1` (typo) → startup aborts naming the var.
- Serde round-trip of `RuntimeConfig` for autotune-cache keying.
- `compile_fail`: struct-literal `RuntimeConfig { .. }` forbidden outside builder module (private fields).

## Cross-cutting deps
- 03-errors: `ConfigError` is a variant of `RvllmError`.
- 04-memory: consumes `kv_block_size`, `num_gpu_blocks`.
- 06-loader: consumes `ModelConfig`; verifies `head_dim*num_attention_heads == hidden_size`.
- 07-scheduler: consumes `max_batch`, `max_context`, `preemption`.
- 14-graph: consumes `GraphMode::Buckets`.
- 16-deploy: bakes `RuntimeConfig` SHA into autotune-cache filename.
