//! Tensor shape with explicit row-major strides.
//!
//! Fixed-capacity; the runtime only allocates up to 4-D tensors for the
//! decode path (plus the KV cache which we model as 5-D in `rvllm-mem`).

use core::fmt;

/// Maximum rank supported by the core shape. KV-cache layout bumps this
/// to 5 via a dedicated type in `rvllm-mem`.
pub const MAX_RANK: usize = 4;

/// Shape + row-major strides. Rank is stored inline; no heap alloc.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Shape {
    rank: u8,
    dims: [usize; MAX_RANK],
}

impl Shape {
    /// Build from a slice. Panics if `dims.len() > MAX_RANK`.
    pub fn new(dims: &[usize]) -> Self {
        assert!(
            dims.len() <= MAX_RANK,
            "Shape::new: rank {} > MAX_RANK {}",
            dims.len(),
            MAX_RANK
        );
        let mut out = [0usize; MAX_RANK];
        out[..dims.len()].copy_from_slice(dims);
        Self {
            rank: dims.len() as u8,
            dims: out,
        }
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank as usize
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        assert!(i < self.rank as usize, "dim index {i} out of range");
        self.dims[i]
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.dims[..self.rank as usize].iter().product()
    }

    /// Row-major strides in elements.
    pub fn strides(&self) -> [usize; MAX_RANK] {
        let mut s = [0usize; MAX_RANK];
        let r = self.rank as usize;
        if r == 0 {
            return s;
        }
        s[r - 1] = 1;
        for i in (0..r - 1).rev() {
            s[i] = s[i + 1] * self.dims[i + 1];
        }
        s
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dbg = f.debug_list();
        for i in 0..self.rank as usize {
            dbg.entry(&self.dims[i]);
        }
        dbg.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_and_strides() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(&s.strides()[..3], &[12, 4, 1]);
    }

    #[test]
    fn zero_rank() {
        let s = Shape::new(&[]);
        assert_eq!(s.rank(), 0);
        // numel of empty product is 1 (scalar)
        assert_eq!(s.numel(), 1);
    }

    #[test]
    #[should_panic]
    fn rank_overflow_panics() {
        let _ = Shape::new(&[1; MAX_RANK + 1]);
    }
}
