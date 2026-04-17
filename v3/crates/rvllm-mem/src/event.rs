//! CUDA event wrapper.
//!
//! Events are how the runtime coordinates DtoH double-buffering. Two
//! events per worker (buf0, buf1); `step_launch` records into the
//! outgoing buf's event, `step_collect` synchronizes on the incoming
//! buf's event.

use core::marker::PhantomData;

use rvllm_core::Result;

use crate::cuda_owned::CudaOwned;
use crate::stream::Stream;

pub struct Event<'s> {
    raw: u64,
    stream: &'s Stream,
    _not_send_sync: PhantomData<*const ()>,
}

impl<'s> Event<'s> {
    /// Host-side stub.
    pub fn host_stub(stream: &'s Stream) -> Self {
        Self {
            raw: 0,
            stream,
            _not_send_sync: PhantomData,
        }
    }

    pub fn raw(&self) -> u64 {
        self.raw
    }

    /// Record this event onto the associated stream.
    pub fn record(&mut self) -> Result<()> {
        // Real impl: cuEventRecord(self.raw, stream.raw()).
        Ok(())
    }

    /// Block the CPU until this event has fired.
    pub fn synchronize(&self) -> Result<()> {
        // Real impl: cuEventSynchronize(self.raw).
        Ok(())
    }
}

impl<'s> CudaOwned for Event<'s> {
    fn stream_for_fence(&self) -> &Stream {
        self.stream
    }
}

impl<'s> Drop for Event<'s> {
    fn drop(&mut self) {
        // Fence the associated stream before destroying the event.
        // Real impl: cuStreamSynchronize then cuEventDestroy.
        self.fence_then_destroy();
    }
}
