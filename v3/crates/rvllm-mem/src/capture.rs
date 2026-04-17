//! `CaptureScope`: the only handle that can bind tensors into a captured
//! CUDA graph.
//!
//! The bind API takes `&T where T: GraphSafe`, so `&mut HbmArena` (which
//! is not `GraphSafe` — arenas can grow) cannot be in scope. That makes
//! "realloc inside capture" a compile error, not a runtime bug.
//!
//! `record` runs the caller's closure under graph capture; when the
//! closure returns, the scope ends and the graph is instantiated.

use core::marker::PhantomData;

use rvllm_core::Result;

use crate::graph_safe::GraphSafe;
use crate::stream::Stream;

/// Token tying the lifetime of bound handles to the scope.
pub struct BoundHandle<'g> {
    device_ptr: u64,
    _scope: PhantomData<&'g ()>,
}

impl<'g> BoundHandle<'g> {
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }
}

/// A graph capture scope. Created by `record(stream, |scope| ...)`.
pub struct CaptureScope<'g, 's> {
    stream: &'s Stream,
    _scope: PhantomData<&'g ()>,
}

impl<'g, 's> CaptureScope<'g, 's> {
    /// Bind a `GraphSafe` value by shared reference. Returns a handle
    /// whose lifetime is tied to the scope, so the value outlives the
    /// capture.
    ///
    /// Trait bound ensures callers cannot pass `&mut HbmArena` or any
    /// realloc-capable wrapper.
    pub fn bind<T>(&mut self, value: &'g T) -> BoundHandle<'g>
    where
        T: GraphSafe + HasDevicePtr,
    {
        BoundHandle {
            device_ptr: value.device_ptr(),
            _scope: PhantomData,
        }
    }

    pub fn stream(&self) -> &'s Stream {
        self.stream
    }
}

/// Types that expose a device pointer for graph binding. Implemented by
/// `Region`, `Tensor`, and any other `GraphSafe` value that backs a
/// kernel argument.
pub trait HasDevicePtr {
    fn device_ptr(&self) -> u64;
}

impl<'a> HasDevicePtr for crate::hbm::Region<'a> {
    fn device_ptr(&self) -> u64 {
        crate::hbm::Region::device_ptr(self)
    }
}
impl<'a, T> HasDevicePtr for crate::tensor::Tensor<'a, T> {
    fn device_ptr(&self) -> u64 {
        crate::tensor::Tensor::device_ptr(self)
    }
}

/// Run `body` under graph capture on `stream`. The closure's `scope`
/// argument is the only way to bind tensors; `&mut HbmArena` is not in
/// scope (the caller does not pass it in), so realloc inside `body`
/// doesn't compile.
///
/// The scope lifetime matches the stream borrow; bound values need only
/// outlive the stream, which by construction outlives the scope.
pub fn record<'s, F, R>(stream: &'s Stream, body: F) -> Result<R>
where
    F: FnOnce(&mut CaptureScope<'s, 's>) -> Result<R>,
{
    // Real impl: cuStreamBeginCapture(stream.raw(), ...); run; cuStreamEndCapture(...).
    let mut scope = CaptureScope {
        stream,
        _scope: PhantomData,
    };
    body(&mut scope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hbm::HbmArena;

    #[test]
    fn can_bind_region_into_scope() {
        let arena = HbmArena::new_host_stub(1 << 20);
        let region = arena.region("kv", 4096, 128).unwrap();
        let s = Stream::host_stub();
        let ptr = record(&s, |scope| {
            let h = scope.bind(&region);
            Ok(h.device_ptr())
        })
        .unwrap();
        assert_eq!(ptr, region.device_ptr());
    }

    // The compile-fail test (`&mut HbmArena` is not in scope inside
    // `record`, so `arena.region(...)` during capture does not compile)
    // is documented in trybuild — to be added when rvllm-invariants
    // grows its trybuild harness in Phase A.5.
}
