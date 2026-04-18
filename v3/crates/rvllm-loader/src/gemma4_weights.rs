//! Gemma 4 weight structures.
//!
//! All layers have IDENTICAL weight shapes regardless of sliding/global type.
//! The dual head_dim (256 sliding vs 512 global) is a runtime reshape.
//!
//! Per-layer extras vs Llama/Qwen:
//!   - 4 norms (input, post_attn, pre_ff, post_ff)
//!   - QK-norm gammas (q_norm [256], k_norm [256])
//!   - layer_scalar [1] (per-layer residual multiplier)
//!
//! Weight shapes (google/gemma-4-31B-it):
//!   q_proj:        [8192, 5376]
//!   k_proj:        [4096, 5376]
//!   v_proj:        [4096, 5376]
//!   o_proj:        [5376, 8192]
//!   gate_proj:     [21504, 5376]
//!   up_proj:       [21504, 5376]
//!   down_proj:     [5376, 21504]
//!   q_norm:        [256]
//!   k_norm:        [256]
//!   layer_scalar:  [1]
//!   *_layernorm:   [5376]

use crate::weights::{F16Weight, Fp8Weight};

#[derive(Debug)]
pub struct Gemma4LayerWeights {
    pub qkv: Fp8Weight,
    pub o_proj: Fp8Weight,
    pub gate_up: Fp8Weight,
    pub down_proj: Fp8Weight,
    pub input_layernorm: F16Weight,
    pub post_attention_layernorm: F16Weight,
    pub pre_feedforward_layernorm: F16Weight,
    pub post_feedforward_layernorm: F16Weight,
    pub q_norm: F16Weight,
    pub k_norm: F16Weight,
    pub layer_scalar: F16Weight,
}

#[derive(Debug)]
pub struct Gemma4LoadedModel {
    pub embedding: F16Weight,
    pub lm_head_fp8: Fp8Weight,
    pub final_norm: F16Weight,
    /// Sliding layers: theta=10000, full rotation (rotary_dim=256)
    pub rope_cos_sliding: F16Weight,
    pub rope_sin_sliding: F16Weight,
    /// Global layers: theta=1M, partial rotation (rotary_dim=128 of head_dim=512)
    pub rope_cos_global: F16Weight,
    pub rope_sin_global: F16Weight,
    pub layers: Vec<Gemma4LayerWeights>,
}
