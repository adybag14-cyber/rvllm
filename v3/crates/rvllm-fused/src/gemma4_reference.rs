//! Pure-Rust f32 reference implementations for Gemma 4 kernels.
//!
//! Same role as `reference.rs`: CI ground truth for cosine tests.

pub const FP8_E4M3_MAX: f32 = 448.0;

/// GELU(tanh)(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_tanh(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    0.5 * x * (1.0 + inner.tanh())
}

/// Fused GELU(tanh)(gate) * up + FP8 per-tensor quantization.
pub fn fused_gelu_mul_fp8_quant_ref(
    gate_up: &[f32],
    intermediate: usize,
    out_fp8: &mut [u8],
    scales: &mut [f32],
) {
    let rows = gate_up.len() / (2 * intermediate);
    assert_eq!(out_fp8.len(), rows * intermediate);
    assert_eq!(scales.len(), rows);

    for row in 0..rows {
        let gate = &gate_up[row * 2 * intermediate..row * 2 * intermediate + intermediate];
        let up = &gate_up[row * 2 * intermediate + intermediate..(row + 1) * 2 * intermediate];

        let mut amax: f32 = 0.0;
        let vals: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let v = gelu_tanh(g) * u;
                amax = amax.max(v.abs());
                v
            })
            .collect();

        let scale = amax.max(1e-12) / FP8_E4M3_MAX;
        let inv = 1.0 / scale;
        scales[row] = scale;

        for (i, &v) in vals.iter().enumerate() {
            out_fp8[row * intermediate + i] = f32_to_fp8_e4m3(v * inv);
        }
    }
}

/// Per-head RMSNorm reference (QK-norm).
pub fn qk_rmsnorm_ref(
    input: &[f32],
    gamma: &[f32],
    eps: f32,
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    assert_eq!(gamma.len(), head_dim);
    for t in 0..num_tokens {
        for h in 0..num_heads {
            let offset = (t * num_heads + h) * head_dim;
            let head = &input[offset..offset + head_dim];
            let out = &mut output[offset..offset + head_dim];

            let ms: f32 = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (o, (x, g)) in out.iter_mut().zip(head.iter().zip(gamma.iter())) {
                *o = x * inv * g;
            }
        }
    }
}

/// Partial RoPE reference: rotate first `rotary_dim` elements, pass rest through.
pub fn partial_rope_ref(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    positions: &[i32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) {
    let half_rotary = rotary_dim / 2;
    for t in 0..num_tokens {
        let pos = positions[t] as usize;
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            for d in 0..half_rotary {
                let c = cos[pos * half_rotary + d];
                let s = sin[pos * half_rotary + d];
                let x0 = x[base + 2 * d];
                let x1 = x[base + 2 * d + 1];
                x[base + 2 * d] = x0 * c - x1 * s;
                x[base + 2 * d + 1] = x0 * s + x1 * c;
            }
        }
    }
}

/// Logit softcap reference: cap * tanh(x / cap)
pub fn logit_softcap_ref(logits: &mut [f32], cap: f32) {
    let inv = 1.0 / cap;
    for v in logits.iter_mut() {
        *v = cap * (*v * inv).tanh();
    }
}

fn f32_to_fp8_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let s: u8 = if v < 0.0 { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return s;
    }
    if a > FP8_E4M3_MAX {
        return s | 0x7e;
    }
    let bits = a.to_bits();
    let exp32 = ((bits >> 23) & 0xff) as i32 - 127;
    let mant32 = bits & 0x7f_ffff;
    let exp8 = exp32 + 7;
    if exp8 <= 0 {
        let shift = 1 - exp8;
        let m = (mant32 | (1 << 23)) >> (21 + shift);
        return s | (m as u8 & 0x07);
    }
    if exp8 >= 0xf {
        return s | 0x7e;
    }
    let m = (mant32 >> 20) as u8 & 0x07;
    s | ((exp8 as u8 & 0x0f) << 3) | m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_tanh_at_zero() {
        assert!((gelu_tanh(0.0)).abs() < 1e-6);
    }

    #[test]
    fn gelu_tanh_positive() {
        let v = gelu_tanh(1.0);
        assert!((v - 0.8412).abs() < 0.01);
    }

    #[test]
    fn gelu_tanh_negative() {
        let v = gelu_tanh(-1.0);
        assert!(v.abs() < 0.2);
    }

    #[test]
    fn softcap_identity_near_zero() {
        let mut logits = vec![0.1, -0.1, 0.0];
        logit_softcap_ref(&mut logits, 30.0);
        assert!((logits[0] - 0.1).abs() < 0.01);
        assert!((logits[1] + 0.1).abs() < 0.01);
    }

    #[test]
    fn softcap_clamps_large() {
        let mut logits = vec![1000.0, -1000.0];
        logit_softcap_ref(&mut logits, 30.0);
        assert!((logits[0] - 30.0).abs() < 0.01);
        assert!((logits[1] + 30.0).abs() < 0.01);
    }

    #[test]
    fn qk_rmsnorm_unit_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        qk_rmsnorm_ref(&input, &gamma, 1e-6, 1, 1, 4, &mut output);
        let rms = (input.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        for (o, x) in output.iter().zip(input.iter()) {
            assert!((o - x / rms).abs() < 1e-4);
        }
    }

    #[test]
    fn partial_rope_leaves_passthrough_unchanged() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cos = vec![1.0, 1.0];
        let sin = vec![0.0, 0.0];
        let positions = vec![0i32];
        partial_rope_ref(&mut x, &cos, &sin, &positions, 1, 1, 8, 4);
        assert!((x[4] - 5.0).abs() < 1e-6);
        assert!((x[5] - 6.0).abs() < 1e-6);
        assert!((x[6] - 7.0).abs() < 1e-6);
        assert!((x[7] - 8.0).abs() < 1e-6);
    }
}
