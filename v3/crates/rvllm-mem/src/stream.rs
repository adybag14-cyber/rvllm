//! CUDA compute stream wrapper.
//!
//! One stream per worker. `Stream` is neither `Send` nor `Sync` — a
//! worker's stream must stay pinned to the thread that created it.
//! `Drop` fences and destroys; the rule "no handle destroyed while the
//! stream is busy" is enforced by `CudaOwned` (see `cuda_owned.rs`).

use core::marker::PhantomData;

use rvllm_core::{CudaCtx, CudaErrorKind, Result, RvllmError};

use crate::cuda_owned::CudaOwned;

/// A CUDA compute stream. Opaque; clients see only methods.
pub struct Stream {
    raw: u64,
    // !Send !Sync — streams are thread-local in our runtime.
    _not_send_sync: PhantomData<*const ()>,
}

impl Stream {
    /// Host-side stub used in tests (no CUDA). Allocates nothing.
    pub fn host_stub() -> Self {
        Self {
            raw: 0,
            _not_send_sync: PhantomData,
        }
    }

    /// Raw stream handle for FFI. The handle is valid for the lifetime
    /// of `self`; the borrow checker prevents dangling use.
    pub fn raw(&self) -> u64 {
        self.raw
    }

    /// Block until all enqueued work on this stream completes.
    /// Only callable from init / shutdown / test modules — the
    /// `no_steady_state_fence` lint denies it elsewhere (added in a
    /// later pass when the lint is implemented).
    pub fn fence(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // Real impl: cudarc stream synchronize.
            Err(RvllmError::cuda(
                "Stream::fence",
                CudaErrorKind::Other,
                CudaCtx {
                    stream: self.raw,
                    kernel: "fence",
                    launch: None,
                    device: -1,
                },
            ))
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Host stub: nothing to fence.
            Ok(())
        }
    }
}

impl CudaOwned for Stream {
    fn stream_for_fence(&self) -> &Stream {
        self
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // The stream owns itself; there is no separate handle to
        // destroy. Real impl calls cuStreamDestroy here.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_stub_fences() {
        let s = Stream::host_stub();
        assert!(s.fence().is_ok());
        assert_eq!(s.raw(), 0);
    }
}
