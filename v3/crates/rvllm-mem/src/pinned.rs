//! Pinned (page-locked) host buffer, and the double-buffer pool used by
//! the DtoH pipeline.
//!
//! Real impl allocates via `cuMemAllocHost`; host stub uses a `Box<[T]>`
//! with the right layout so invariant-level tests pass without CUDA.

use core::marker::PhantomData;

use rvllm_core::{Result, RvllmError};

/// A pinned host buffer of `N` elements of `T`.
pub struct PinnedBuf<T> {
    // Host stub uses Box<[T]>; real impl will replace with a
    // `cuMemAllocHost`-backed raw pointer + len, behind `#[cfg(feature =
    // "cuda")]`.
    data: Box<[T]>,
    _not_send_sync: PhantomData<*const ()>,
}

impl<T: Default + Clone> PinnedBuf<T> {
    /// Allocate a pinned buffer of `len` elements, zero-filled.
    pub fn new(len: usize) -> Result<Self> {
        let data: Box<[T]> = vec![T::default(); len].into_boxed_slice();
        Ok(Self {
            data,
            _not_send_sync: PhantomData,
        })
    }
}

impl<T> PinnedBuf<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    /// Raw host pointer (for `cuMemcpy*`). The runtime's DtoH path uses this.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

/// Double-buffered pinned argmax pool. `[A, B]` buffers, `[ev_A, ev_B]`
/// events owned by the caller; this struct holds just the buffers and
/// the flip-index.
pub struct PinnedPool<T> {
    buffers: [PinnedBuf<T>; 2],
    /// Index of the buffer the NEXT `launch_dtoh` writes into.
    write_idx: u8,
}

impl<T: Default + Clone> PinnedPool<T> {
    pub fn new(len_per_buf: usize) -> Result<Self> {
        Ok(Self {
            buffers: [PinnedBuf::new(len_per_buf)?, PinnedBuf::new(len_per_buf)?],
            write_idx: 0,
        })
    }
}

impl<T> PinnedPool<T> {
    /// Index the next `launch_dtoh` writes into.
    pub fn write_idx(&self) -> usize {
        self.write_idx as usize
    }
    /// Index that the next `step_collect` reads from.
    pub fn read_idx(&self) -> usize {
        1 - self.write_idx as usize
    }
    pub fn write_buf_mut(&mut self) -> &mut PinnedBuf<T> {
        &mut self.buffers[self.write_idx as usize]
    }
    pub fn read_buf(&self) -> &PinnedBuf<T> {
        &self.buffers[1 - self.write_idx as usize]
    }
    /// Call after `cuEventRecord` on the outgoing buffer's event. Flips
    /// the write index so the next launch writes the other buffer.
    pub fn flip(&mut self) {
        self.write_idx = 1 - self.write_idx;
    }
}

// Pinned buffers are not GraphSafe — a captured graph binds the
// *device* pointer of a DtoH copy, not the host pointer. The kernel
// sees zero host pointers directly, so there is nothing to implement.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_flips_between_buffers() {
        let mut p: PinnedPool<i32> = PinnedPool::new(128).unwrap();
        assert_eq!(p.write_idx(), 0);
        assert_eq!(p.read_idx(), 1);
        p.flip();
        assert_eq!(p.write_idx(), 1);
        assert_eq!(p.read_idx(), 0);
    }

    #[test]
    fn buf_is_zero_initialized() {
        let b: PinnedBuf<i32> = PinnedBuf::new(16).unwrap();
        assert_eq!(b.len(), 16);
        assert!(b.as_slice().iter().all(|x| *x == 0));
    }
}
