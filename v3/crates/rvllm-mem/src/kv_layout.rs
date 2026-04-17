//! KV-cache layout: `[2, num_blocks, block_size, num_kv_heads, head_dim]`.
//!
//! K at offset 0, V at `num_blocks * block_size * num_kv_heads * head_dim`.
//! This matches the FA3 paged-decode page-table descriptor byte-for-byte,
//! so the kernel never does pointer math beyond `block_table[i]`.

use rvllm_core::DType;

/// Per-layer KV layout. One instance per model.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvLayout {
    pub num_blocks: u32,
    pub block_size: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
}

impl KvLayout {
    /// Bytes per block (K *or* V, one of the two).
    pub const fn block_bytes(&self) -> usize {
        (self.block_size as usize)
            * (self.num_kv_heads as usize)
            * (self.head_dim as usize)
            * self.dtype.bytes()
    }

    /// Bytes for one layer (K + V combined). The factor-of-two is the
    /// first axis of the layout.
    pub const fn layer_bytes(&self) -> usize {
        2 * (self.num_blocks as usize) * self.block_bytes()
    }

    /// Byte offset of the start of V within a layer (K starts at 0).
    pub const fn v_offset(&self) -> usize {
        (self.num_blocks as usize) * self.block_bytes()
    }

    /// Row-major strides in elements (not bytes), by axis index:
    /// `[K_or_V, block, token_in_block, kv_head, head_dim]`.
    pub fn strides(&self) -> [usize; 5] {
        let d = self.head_dim as usize;
        let h = self.num_kv_heads as usize;
        let b = self.block_size as usize;
        let n = self.num_blocks as usize;
        [n * b * h * d, b * h * d, h * d, d, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_kv() -> KvLayout {
        KvLayout {
            num_blocks: 1024,
            block_size: 64,
            num_kv_heads: 4,
            head_dim: 128,
            dtype: DType::F16,
        }
    }

    #[test]
    fn sizes_round_trip() {
        let l = qwen_kv();
        // One block: 64 tokens * 4 heads * 128 dim * 2 bytes = 65536
        assert_eq!(l.block_bytes(), 64 * 4 * 128 * 2);
        // One layer: 2 * 1024 blocks * 65536 bytes = 128 MiB
        assert_eq!(l.layer_bytes(), 2 * 1024 * 65536);
        assert_eq!(l.v_offset(), 1024 * 65536);
    }

    #[test]
    fn strides_are_row_major() {
        let l = qwen_kv();
        let s = l.strides();
        // fastest axis is head_dim
        assert_eq!(s[4], 1);
        assert_eq!(s[3], 128);
        assert_eq!(s[2], 128 * 4);
        assert_eq!(s[1], 128 * 4 * 64);
        assert_eq!(s[0], 128 * 4 * 64 * 1024);
    }
}
