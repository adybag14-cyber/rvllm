//! rvllm-attention: FA3 SM90 paged decode + prefill.
//!
//! Two kernels only: `paged_decode` and `paged_prefill`. Both live in
//! `libfa3_kernels.so` which is built from the FlashAttention-3 Hopper
//! source at deploy time. No PTX fallback: engine refuses to start if
//! the `.so` is missing or not in the manifest.
//!
//! The invariants:
//! - `head_dim == 128` hard gate at construction (`AttentionError::UnsupportedHeadDim`)
//! - GQA ratio sanity (`num_heads` divisible by `num_kv_heads`)
//! - context_lens[i] == 0 valid padded-slot marker; kernel must predicate

pub mod decode;
pub mod prefill;

pub use decode::{PagedDecodeLauncher, PagedDecodeParams};
pub use prefill::{PagedPrefillLauncher, PagedPrefillParams};

use rvllm_core::{AttentionError, AttnCtx, Result, RvllmError};

/// Runtime-constructed wrapper around `libfa3_kernels.so`. The wrapper
/// refuses to exist if the .so is missing or its manifest-verified
/// exports don't include the entry points. Callers obtain launchers
/// from the wrapper.
#[derive(Debug)]
pub struct Fa3Kernels {
    /// Path to the .so (diagnostics only).
    pub so_path: std::path::PathBuf,
    // Under `feature = "cuda"`, this holds the dlopen handle and fn
    // pointers. Under no-cuda, it's unit so the crate tests compile.
    #[cfg(feature = "cuda")]
    _lib: libloading::Library,
}

impl Fa3Kernels {
    /// Load the FA3 .so. Called once at engine init from a
    /// `KernelLoader`-produced path. Returns `Err` with explicit
    /// `AttentionError::Fa3SoMissing` if the path does not exist.
    pub fn load(path: std::path::PathBuf, head_dim: u32) -> Result<Self> {
        if !path.exists() {
            return Err(RvllmError::Attention {
                err: AttentionError::Fa3SoMissing { path: path.clone() },
                ctx: AttnCtx {
                    op: "Fa3Kernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        if head_dim != 128 {
            return Err(RvllmError::Attention {
                err: AttentionError::UnsupportedHeadDim {
                    got: head_dim,
                    required: 128,
                },
                ctx: AttnCtx {
                    op: "Fa3Kernels::load",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            });
        }

        #[cfg(feature = "cuda")]
        let _lib = unsafe {
            libloading::Library::new(&path).map_err(|e| RvllmError::Attention {
                err: AttentionError::Fa3SoMissing { path: path.clone() },
                ctx: AttnCtx {
                    op: "dlopen",
                    stream: 0,
                    num_seqs: 0,
                    head_dim,
                },
                bt: std::backtrace::Backtrace::capture(),
            })?
        };

        Ok(Self {
            so_path: path,
            #[cfg(feature = "cuda")]
            _lib,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_so_rejected_at_load() {
        let err = Fa3Kernels::load("/nonexistent/libfa3_kernels.so".into(), 128).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("Fa3SoMissing"));
    }

    #[test]
    fn non_128_head_dim_rejected() {
        // use a real-ish path so the missing-so check doesn't fire first
        let tmp = std::env::temp_dir().join("fa3-fake.so");
        std::fs::write(&tmp, b"fake").unwrap();
        let err = Fa3Kernels::load(tmp.clone(), 64).unwrap_err();
        std::fs::remove_file(&tmp).ok();
        let s = format!("{err}");
        assert!(s.contains("UnsupportedHeadDim"));
    }
}
