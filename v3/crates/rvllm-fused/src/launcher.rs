//! Launcher descriptors for the fused kernel set.
//!
//! Each launcher is a plain-old Rust struct that validates shapes,
//! alignment, and device-pointer presence before issuing the CUDA
//! launch. The launch itself is a single `#[cfg(feature = "cuda")]`
//! function that calls into `cudarc` (pending wiring); the validation
//! is pure-Rust and runs everywhere.

use rvllm_core::{Result, RvllmError, SamplingError, SampleCtx};

/// Common alignment rule: FP8 and f16 kernels using uint4 loads require
/// the last dim to be a multiple of 8 halves (for f16) or 16 bytes (for
/// u8). Check at validate time — misalignment here → `Err`, not a silent
/// crash under graph replay.
pub fn require_multiple(got: usize, of: usize, what: &'static str) -> Result<()> {
    if of == 0 || got % of != 0 {
        return Err(RvllmError::Sampling {
            err: SamplingError::InvalidParams {
                reason: format!("{what} must be multiple of {of}, got {got}"),
            },
            ctx: SampleCtx {
                op: "require_multiple",
                stream: 0,
            },
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fused_add_rmsnorm_fp8_quant
// ---------------------------------------------------------------------------

pub struct FusedAddRmsnormFp8QuantLaunch {
    pub num_tokens: u32,
    pub hidden: u32,
    pub eps: f32,
}

impl FusedAddRmsnormFp8QuantLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.hidden as usize, 8, "hidden")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// quantize_fp8_per_token
// ---------------------------------------------------------------------------

pub struct QuantizeFp8PerTokenLaunch {
    pub num_tokens: u32,
    pub dim: u32,
}

impl QuantizeFp8PerTokenLaunch {
    pub fn validate(&self) -> Result<()> {
        // Vectorized uint4/uint2 path requires dim % 8 == 0 (8 halves per
        // uint4 load, 8 fp8 per uint2 store). This was the alignment
        // guard whose absence caused the April 16 ILLEGAL_ADDRESS hunt.
        require_multiple(self.dim as usize, 8, "dim")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        // Upper bound matches the MAX_VEC_PER_THREAD=8 x block=1024 cache.
        if self.dim > 65536 {
            return Err(invalid("dim", "must be <= 65536"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// fused_silu_mul_fp8_quant
// ---------------------------------------------------------------------------

pub struct FusedSiluMulFp8QuantLaunch {
    pub num_tokens: u32,
    pub intermediate: u32,
}

impl FusedSiluMulFp8QuantLaunch {
    pub fn validate(&self) -> Result<()> {
        require_multiple(self.intermediate as usize, 8, "intermediate")?;
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------

pub struct ArgmaxLaunch {
    pub num_tokens: u32,
    pub vocab: u32,
}

impl ArgmaxLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.vocab == 0 {
            return Err(invalid("vocab", "must be > 0"));
        }
        if self.num_tokens == 0 {
            return Err(invalid("num_tokens", "must be > 0"));
        }
        Ok(())
    }

    /// Issue the `argmax_kernel` launch.
    ///
    /// Kernel sig (from spec 12 and v2's argmax.cu):
    ///   `argmax_kernel(logits: *const f32, out: *mut i32, vocab: i32)`
    /// grid = (num_tokens, 1, 1); block = (min(vocab, 1024), 1, 1).
    ///
    /// # Safety
    /// Caller must ensure `logits_ptr` / `out_ptr` are live device
    /// pointers for the kernel's duration (graph capture or eager).
    pub unsafe fn launch(
        &self,
        kernel: rvllm_kernels::KernelFn,
        logits_ptr: u64,
        out_ptr: u64,
        stream: u64,
    ) -> Result<()> {
        self.validate()?;
        let vocab = self.vocab as i32;
        // Kernel args must outlive the launch call — keep bindings alive.
        let mut logits_ptr = logits_ptr;
        let mut out_ptr = out_ptr;
        let mut vocab_arg = vocab;
        let args = [
            (&mut logits_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut out_ptr) as *mut u64 as *mut core::ffi::c_void,
            (&mut vocab_arg) as *mut i32 as *mut core::ffi::c_void,
        ];
        let block_dim = (self.vocab.min(1024), 1, 1);
        let grid = (self.num_tokens, 1, 1);
        crate::launch_raw::launch_raw(kernel, grid, block_dim, 0, stream, &args)
    }
}

// ---------------------------------------------------------------------------
// fused_rope_kv_write
// ---------------------------------------------------------------------------

pub struct FusedRopeKvWriteLaunch {
    pub num_tokens: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub head_dim: u32,
}

impl FusedRopeKvWriteLaunch {
    pub fn validate(&self) -> Result<()> {
        if self.head_dim != 128 {
            return Err(invalid("head_dim", "v3 FA3 path requires head_dim == 128"));
        }
        if self.q_dim % self.head_dim != 0 || self.kv_dim % self.head_dim != 0 {
            return Err(invalid(
                "q_dim/kv_dim",
                "must be a multiple of head_dim",
            ));
        }
        Ok(())
    }
}

fn invalid(field: &'static str, reason: &'static str) -> RvllmError {
    RvllmError::Sampling {
        err: SamplingError::InvalidParams {
            reason: format!("{field}: {reason}"),
        },
        ctx: SampleCtx {
            op: "validate",
            stream: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_rejects_non_multiple_of_8() {
        let l = QuantizeFp8PerTokenLaunch {
            num_tokens: 1,
            dim: 13,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn quant_accepts_power_of_two() {
        let l = QuantizeFp8PerTokenLaunch {
            num_tokens: 1,
            dim: 3584,
        };
        assert!(l.validate().is_ok());
    }

    #[test]
    fn rope_requires_head_dim_128() {
        let l = FusedRopeKvWriteLaunch {
            num_tokens: 1,
            q_dim: 64,
            kv_dim: 64,
            head_dim: 64,
        };
        assert!(l.validate().is_err());
    }

    #[test]
    fn argmax_rejects_zero_vocab() {
        let l = ArgmaxLaunch {
            num_tokens: 32,
            vocab: 0,
        };
        assert!(l.validate().is_err());
    }
}
