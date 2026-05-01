use crate::cuda::context::CudaContext;
use crate::cuda::error::CudaError;
use crate::cuda::error::Result;
use crate::cuda::kernels::{launch_flash_attention_3_gqa, launch_reshape_and_cache_f16};
use crate::cuda::{BlockId, BlockTable, PagedBlockAllocator};
use candle_nn::kv_cache::Cache;
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, LaunchConfig};
use cudarc::nvrtc::Ptx;
use half::f16;

/*
PagedSession:
Manages KV block allocation
*/

pub enum CacheKind {
    Standard,                   // single flat KV cache
    Forked { base_len: usize }, // has bCache prefix of base_len tokens
}

pub struct PagedSession {
    pub session_id: u64,
    pub block_table: BlockTable,
    pub seq_len: usize,
    pub cache_kind: CacheKind,
}

impl PagedSession {
    pub fn new(session_id: u64) -> Self {
        let block_table = BlockTable::new();
        let cache_kind = CacheKind::Standard;

        Self {
            session_id: session_id,
            block_table: block_table,
            cache_kind: cache_kind,
            seq_len: 0,
        }
    }

    pub fn allocate_blocks(
        &mut self,
        alloc: &mut PagedBlockAllocator,
        num_new_tokens: usize,
    ) -> Result<()> {
        let current_blocks_held = self.block_table.num_logical_blocks();
        let blocks_needed =
            (self.seq_len + num_new_tokens + alloc.block_size - 1) / alloc.block_size;

        let blocks_to_alloc = blocks_needed - current_blocks_held;
        let new_blocks = alloc.alloc_n(blocks_to_alloc)?;
        for id in new_blocks {
            self.block_table.push_block(id);
        }

        self.seq_len += num_new_tokens;
        Ok(())
    }

    pub fn free_blocks(&mut self, alloc: &mut PagedBlockAllocator) -> Result<()> {
        for &phys in &self.block_table.physical {
            alloc.dec_ref(BlockId(phys as usize))?;
        }

        Ok(())
    }

    pub fn slot_for_token(&self, token_pos: usize, block_size: usize) -> Option<i32> {
        let slot = self.block_table.slot_for(token_pos, block_size);
        slot
    }

    pub fn is_forked(&self) -> bool {
        matches!(self.cache_kind, CacheKind::Forked { .. })
    }

    pub fn base_len(&self) -> usize {
        match self.cache_kind {
            CacheKind::Standard => 0,
            CacheKind::Forked { base_len } => base_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::allocator::PagedBlockAllocator;
    use crate::cuda::context::CudaContext;
    use std::sync::Arc;

    fn try_alloc() -> Option<PagedBlockAllocator> {
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok()?;
        let ctx = CudaContext::new(0, &ptx).ok()?;

        PagedBlockAllocator::new(&ctx, 16, 4, 2, 32).ok()
    }

    #[test]
    fn test_new_session_is_empty() {
        let s = PagedSession::new(1);
        assert_eq!(s.session_id, 1);
        assert_eq!(s.seq_len, 0);
        assert_eq!(s.block_table.num_logical_blocks(), 0);
        assert!(!s.is_forked());
    }

    #[test]
    fn test_allocate_single_block() {
        let Some(mut alloc) = try_alloc() else { return };
        let mut s = PagedSession::new(1);
        s.allocate_blocks(&mut alloc, 1).unwrap();
        assert_eq!(s.block_table.num_logical_blocks(), 1);
        assert_eq!(s.seq_len, 1);
    }

    #[test]
    fn test_allocate_spans_two_blocks() {
        let Some(mut alloc) = try_alloc() else { return };
        // block_size = 4, so 5 tokens needs 2 blocks
        let mut s = PagedSession::new(1);
        s.allocate_blocks(&mut alloc, 5).unwrap();
        assert_eq!(s.block_table.num_logical_blocks(), 2);
        assert_eq!(s.seq_len, 5);
    }

    #[test]
    fn test_allocate_incremental() {
        let Some(mut alloc) = try_alloc() else { return };
        // block_size = 4
        // first call: 3 tokens -> 1 block
        // second call: 2 more tokens -> now 5 total -> needs 2 blocks, allocates 1 more
        let mut s = PagedSession::new(1);
        s.allocate_blocks(&mut alloc, 3).unwrap();
        assert_eq!(s.block_table.num_logical_blocks(), 1);
        s.allocate_blocks(&mut alloc, 2).unwrap();
        assert_eq!(s.block_table.num_logical_blocks(), 2);
        assert_eq!(s.seq_len, 5);
    }

    #[test]
    fn test_free_returns_blocks_to_pool() {
        let Some(mut alloc) = try_alloc() else { return };
        let free_before = alloc.num_free();
        let mut s = PagedSession::new(1);
        s.allocate_blocks(&mut alloc, 5).unwrap(); // 2 blocks
        assert_eq!(alloc.num_free(), free_before - 2);
        s.free_blocks(&mut alloc).unwrap();
        assert_eq!(alloc.num_free(), free_before);
    }

    #[test]
    fn test_slot_for_token_correct() {
        let Some(mut alloc) = try_alloc() else { return };
        // block_size = 4
        let mut s = PagedSession::new(1);
        s.allocate_blocks(&mut alloc, 6).unwrap();
        // token 0 is in block 0, offset 0
        // slot = physical_block_id * 4 + 0
        let slot0 = s.slot_for_token(0, alloc.block_size).unwrap();
        // token 4 is in block 1, offset 0
        let slot4 = s.slot_for_token(4, alloc.block_size).unwrap();
        // token 5 is in block 1, offset 1
        let slot5 = s.slot_for_token(5, alloc.block_size).unwrap();
        assert_eq!(slot5, slot4 + 1);
        // slots in different blocks should not be adjacent
        assert_ne!(slot0 / 4, slot4 / 4);
    }

    #[test]
    fn test_slot_for_unallocated_token_returns_none() {
        let Some(mut alloc) = try_alloc() else { return };
        let mut s = PagedSession::new(1);
        s.allocate_blocks(&mut alloc, 2).unwrap();
        // token 100 is way beyond allocated range
        assert!(s.slot_for_token(100, alloc.block_size).is_none());
    }

    #[test]
    fn test_is_forked_false_by_default() {
        let s = PagedSession::new(42);
        assert!(!s.is_forked());
        assert_eq!(s.base_len(), 0);
    }

    #[test]
    fn test_is_forked_true_after_fork() {
        let mut s = PagedSession::new(1);
        s.cache_kind = CacheKind::Forked { base_len: 16 };
        assert!(s.is_forked());
        assert_eq!(s.base_len(), 16);
    }

    #[test]
    fn test_allocate_exact_block_boundary() {
        let Some(mut alloc) = try_alloc() else { return };
        // block_size = 4, allocating exactly 4 tokens should give exactly 1 block
        let mut s = PagedSession::new(1);
        s.allocate_blocks(&mut alloc, 4).unwrap();
        assert_eq!(s.block_table.num_logical_blocks(), 1);
        // allocating 1 more should spill into block 2
        s.allocate_blocks(&mut alloc, 1).unwrap();
        assert_eq!(s.block_table.num_logical_blocks(), 2);
    }

    #[test]
    fn test_free_empty_session_is_noop() {
        let Some(mut alloc) = try_alloc() else { return };
        let free_before = alloc.num_free();
        let mut s = PagedSession::new(1);
        s.free_blocks(&mut alloc).unwrap();
        assert_eq!(alloc.num_free(), free_before);
    }

    #[test]
    fn test_out_of_blocks_returns_error() {
        let Some(mut alloc) = try_alloc() else { return };
        // alloc has 16 blocks * 4 block_size = 64 slots
        // requesting more than available should fail
        let mut s = PagedSession::new(1);
        assert!(s.allocate_blocks(&mut alloc, 1000).is_err());
    }
}
