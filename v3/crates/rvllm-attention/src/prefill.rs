//! Paged-prefill launcher. Same .so as decode; different entry point.
//!
//! Prefill runs on `num_tokens` query tokens (not one-per-seq). The
//! kernel uses `cu_seqlens_q` / `cu_seqlens_k` to find each request's
//! span in the concatenated tensor.

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

#[derive(Copy, Clone, Debug)]
pub struct PagedPrefillParams {
    pub num_tokens: u32,
    pub num_seqs: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_blocks_total: u32,
    pub scale: f32,
}

impl PagedPrefillParams {
    pub fn validate(&self) -> Result<()> {
        let ctx = || AttnCtx {
            op: "paged_prefill.validate",
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
        Ok(())
    }
}

pub struct PagedPrefillLauncher<'a> {
    _fa3: &'a super::Fa3Kernels,
}

impl<'a> PagedPrefillLauncher<'a> {
    pub fn new(fa3: &'a super::Fa3Kernels) -> Self {
        Self { _fa3: fa3 }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        params: PagedPrefillParams,
        _out_ptr: u64,
        _q_ptr: u64,
        _k_cache_ptr: u64,
        _v_cache_ptr: u64,
        _block_tables_ptr: u64,
        _context_lens_ptr: u64,
        _cu_seqlens_q_ptr: u64,
        _cu_seqlens_k_ptr: u64,
        _workspace_ptr: u64,
        _stream: u64,
    ) -> Result<()> {
        params.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_validates_head_dim() {
        let p = PagedPrefillParams {
            num_tokens: 256,
            num_seqs: 4,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 64, // bad
            block_size: 64,
            max_blocks_per_seq: 33,
            num_blocks_total: 1024,
            scale: 1.0,
        };
        assert!(p.validate().is_err());
    }
}
