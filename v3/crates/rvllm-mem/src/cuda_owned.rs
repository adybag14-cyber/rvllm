//! `CudaOwned`: Drop safety rule re-derived from v2 commit `7c212c13c`.
//!
//! *"Any object owning a CUDA handle must guarantee its stream is idle
//! before destroying the handle."*
//!
//! Encoded as a trait: implementors provide the stream the handle is
//! tied to. The default drop helper (`fence_then_destroy`) can be called
//! from concrete `Drop` impls. This replaces the implicit ordering that
//! v2 encoded with a comment next to each `Drop`.

use crate::stream::Stream;

pub trait CudaOwned {
    /// The stream that synchronizes this handle's completion.
    fn stream_for_fence(&self) -> &Stream;

    /// Helper callable from `Drop`. Fails silently in release (the
    /// handle is destroyed anyway), debug_asserts on fence error.
    fn fence_then_destroy(&mut self) {
        let res = self.stream_for_fence().fence();
        debug_assert!(
            res.is_ok(),
            "CudaOwned: stream fence failed during Drop — destroying anyway"
        );
    }
}
