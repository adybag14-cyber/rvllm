//! Paged-decode launcher.
//!
//! One query per sequence. Kernel reads context_lens[seq] and walks
//! block_tables[seq, 0..ceil(context_lens/block_size)] to find KV
//! pages. `context_lens[i] == 0` is a valid padded slot; kernel must
//! predicate and never touch block_tables[i,*].

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

/// Parameters for one paged decode launch.
#[derive(Copy, Clone, Debug)]
pub struct PagedDecodeParams {
    pub num_seqs: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub scale: f32,
}

impl PagedDecodeParams {
    pub fn validate(&self) -> Result<()> {
        let ctx = || AttnCtx {
            op: "paged_decode.validate",
            stream: 0,
            num_seqs: self.num_seqs,
            head_dim: self.head_dim,
        };
        if self.head_dim != 128 {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: self.head_dim,
                    required: 128,
                },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::GqaRatioInvalid {
                    num_heads: self.num_heads,
                    num_kv_heads: self.num_kv_heads,
                },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if self.num_seqs == 0 {
            return Err(RvllmError::Attention {
                err: AttentionError::ContextExceedsBucket { context: 0, max: 0 },
                ctx: ctx(),
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        Ok(())
    }
}

/// Launcher. Construction from `&Fa3Kernels` guarantees the .so is
/// loaded and head_dim is 128.
pub struct PagedDecodeLauncher<'a> {
    _fa3: &'a super::Fa3Kernels,
}

impl<'a> PagedDecodeLauncher<'a> {
    pub fn new(fa3: &'a super::Fa3Kernels) -> Self {
        Self { _fa3: fa3 }
    }

    /// Validate params + issue the launch. Under `feature = "cuda"`,
    /// this calls into the FA3 .so; under no-cuda it returns Ok without
    /// launching (so tests on Mac can exercise validation).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        params: PagedDecodeParams,
        _out_ptr: u64,
        _q_ptr: u64,
        _k_cache_ptr: u64,
        _v_cache_ptr: u64,
        _block_tables_ptr: u64,
        _context_lens_ptr: u64,
        _workspace_ptr: u64,
        _stream: u64,
    ) -> Result<()> {
        params.validate()?;
        #[cfg(feature = "cuda")]
        {
            // Pending: dlsym("paged_decode") + call with the args above.
            // For the scaffolding commit we return Ok so callers can
            // plug in. The actual launch is added when rvllm-runtime is
            // wired up in Phase C.
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> PagedDecodeParams {
        PagedDecodeParams {
            num_seqs: 32,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0 / (128f32).sqrt(),
        }
    }

    #[test]
    fn rejects_head_dim_64() {
        let mut p = good();
        p.head_dim = 64;
        assert!(p.validate().is_err());
    }

    #[test]
    fn rejects_gqa_ratio_not_divisible() {
        let mut p = good();
        p.num_heads = 7;
        p.num_kv_heads = 4;
        assert!(p.validate().is_err());
    }

    #[test]
    fn accepts_qwen_shape() {
        assert!(good().validate().is_ok());
    }
}
