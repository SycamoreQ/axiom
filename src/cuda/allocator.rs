use crate::cuda::context::CudaContext;
use crate::cuda::error::{CudaError, Result};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use half::f16;
use std::collections::VecDeque;

/*  PagedBlockAllocator
Manages a fixed GPU memory pool divided into equal-sized blocks.
Each block holds [block_size, num_kv_heads, head_dim] f16 elements
for both K and V (two separate pools, same layout).
Block indices are physical — they index directly into the K/V cache
tensors passed to attention kernels. The block table (per-sequence
mapping from logical to physical blocks) is managed by the caller.
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockId(pub usize);

#[derive(Debug)]
struct BlockMeta {
    ref_count: usize,
}

pub struct PagedBlockAllocator {
    // GPU memory pools
    pub k_cache: CudaSlice<f16>, // [num_blocks, block_size, num_kv_heads, head_dim]
    pub v_cache: CudaSlice<f16>, // same layout

    // Block metadata (host-side)
    blocks: Vec<BlockMeta>,
    free_list: VecDeque<BlockId>,

    // Geometry
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,

    // Elements per block (block_size * num_kv_heads * head_dim)
    elems_per_block: usize,
}

impl PagedBlockAllocator {
    pub fn new(
        ctx: &CudaContext,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let elems_per_block = block_size * num_kv_heads * head_dim;
        let total_elems = num_blocks * elems_per_block;

        // f16 does not implement cudarc's ValidAsZeroBits/DeviceRepr unless the
        // cudarc "f16" feature is enabled. Two options:
        //   A) Add `features = ["f16"]` to cudarc in Cargo.toml  ← preferred
        //   B) Allocate as u16 (same bit-width) and transmute the owned slice.
        //
        // We use option B here so no Cargo change is required.
        // Safety: CudaSlice<u16> and CudaSlice<f16> have identical memory layout
        // (both are 2 bytes per element, no padding). All-zero bits = f16 +0.0.
        let k_cache: CudaSlice<f16> = unsafe {
            let raw: CudaSlice<u16> = ctx
                .stream()
                .alloc_zeros::<u16>(total_elems)
                .map_err(CudaError::Driver)?;
            std::mem::transmute(raw)
        };
        let v_cache: CudaSlice<f16> = unsafe {
            let raw: CudaSlice<u16> = ctx
                .stream()
                .alloc_zeros::<u16>(total_elems)
                .map_err(CudaError::Driver)?;
            std::mem::transmute(raw)
        };

        let blocks: Vec<BlockMeta> = (0..num_blocks)
            .map(|_| BlockMeta { ref_count: 0 })
            .collect();

        let free_list: VecDeque<BlockId> = (0..num_blocks).map(BlockId).collect();

        Ok(Self {
            k_cache,
            v_cache,
            blocks,
            free_list,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            elems_per_block,
        })
    }

    //Allocate one free block. Sets ref_count to 1.
    pub fn alloc(&mut self) -> Result<BlockId> {
        let id = self.free_list.pop_front().ok_or(CudaError::OutOfBlocks {
            requested: 1,
            available: 0,
        })?;
        self.blocks[id.0].ref_count = 1;
        Ok(id)
    }

    //Allocate n blocks at once. Rolls back on partial failure.
    pub fn alloc_n(&mut self, n: usize) -> Result<Vec<BlockId>> {
        if self.free_list.len() < n {
            return Err(CudaError::OutOfBlocks {
                requested: n,
                available: self.free_list.len(),
            });
        }
        let ids: Vec<BlockId> = (0..n)
            .map(|_| {
                let id = self.free_list.pop_front().unwrap();
                self.blocks[id.0].ref_count = 1;
                id
            })
            .collect();
        Ok(ids)
    }

    //Increment ref_count — used when bCache shares a block with a fork.
    pub fn inc_ref(&mut self, id: BlockId) -> Result<()> {
        let meta = self
            .blocks
            .get_mut(id.0)
            .ok_or(CudaError::InvalidBlock(id.0))?;
        if meta.ref_count == 0 {
            return Err(CudaError::InvalidBlock(id.0));
        }
        meta.ref_count += 1;
        Ok(())
    }

    //Decrement ref_count. Returns block to free list when count reaches 0.
    pub fn dec_ref(&mut self, id: BlockId) -> Result<()> {
        let meta = self
            .blocks
            .get_mut(id.0)
            .ok_or(CudaError::InvalidBlock(id.0))?;
        if meta.ref_count == 0 {
            return Err(CudaError::InvalidBlock(id.0));
        }
        meta.ref_count -= 1;
        if meta.ref_count == 0 {
            self.free_list.push_back(id);
        }
        Ok(())
    }

    //Returns true if this block is shared (ref_count > 1).
    //ForkKV copy-on-write check: before writing to a block, the caller
    //must check is_shared and copy if true.
    pub fn is_shared(&self, id: BlockId) -> bool {
        self.blocks.get(id.0).map_or(false, |m| m.ref_count > 1)
    }

    pub fn ref_count(&self, id: BlockId) -> usize {
        self.blocks.get(id.0).map_or(0, |m| m.ref_count)
    }

    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    pub fn num_used(&self) -> usize {
        self.num_blocks - self.free_list.len()
    }

    //Byte offset of block `id` within the flat k_cache / v_cache buffer.
    //Useful for manual pointer arithmetic if needed.
    pub fn block_offset_elems(&self, id: BlockId) -> usize {
        id.0 * self.elems_per_block
    }
}

// BlockTable
// Per-sequence logical-to-physical block mapping.
// Logical block i covers token positions [i*block_size .. (i+1)*block_size).
// Passed to attention kernels as a flat &[i32] slice, one entry per
// logical block. The kernel indexes it as block_tables[seq * max_blocks + i].
// The i32 type matches what the CUDA kernels expect (const int*).

#[derive(Debug, Clone)]
pub struct BlockTable {
    pub physical: Vec<i32>, // logical -> physical block id
    pub seq_len: usize,     // current number of tokens written
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            physical: Vec::new(),
            seq_len: 0,
        }
    }

    pub fn num_logical_blocks(&self) -> usize {
        self.physical.len()
    }

    pub fn push_block(&mut self, id: BlockId) {
        self.physical.push(id.0 as i32)
    }

    pub fn logical_block_for(&self, token_pos: usize, block_size: usize) -> usize {
        token_pos / block_size
    }

    pub fn physical_for(&self, token_pos: usize, block_size: usize) -> Option<BlockId> {
        let logical = self.logical_block_for(token_pos, block_size);
        self.physical.get(logical).map(|&p| BlockId(p as usize))
    }

    //Slot index within the cache tensor for a given token position.
    //slot = physical_block * block_size + (token_pos % block_size)
    //This matches the slot_mapping layout expected by reshape_and_cache_f16.
    pub fn slot_for(&self, token_pos: usize, block_size: usize) -> Option<i32> {
        let phys = self.physical_for(token_pos, block_size)?;
        let offset = token_pos % block_size;
        Some((phys.0 * block_size + offset) as i32)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::context::CudaContext;

    // Most allocator tests require a real CUDA device.
    // Guard with an availability check so CI without GPU still passes.
    fn try_ctx() -> Option<CudaContext> {
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok()?;
        CudaContext::new(0, &ptx).ok()
    }

    #[test]
    fn test_alloc_and_free() {
        let Some(ctx) = try_ctx() else { return };
        let mut alloc = PagedBlockAllocator::new(&ctx, 8, 16, 4, 64).unwrap();
        assert_eq!(alloc.num_free(), 8);

        let b0 = alloc.alloc().unwrap();
        let b1 = alloc.alloc().unwrap();
        assert_eq!(alloc.num_free(), 6);
        assert_ne!(b0.0, b1.0);

        alloc.dec_ref(b0).unwrap();
        assert_eq!(alloc.num_free(), 7);
    }

    #[test]
    fn test_out_of_blocks() {
        let Some(ctx) = try_ctx() else { return };
        let mut alloc = PagedBlockAllocator::new(&ctx, 2, 16, 4, 64).unwrap();
        alloc.alloc().unwrap();
        alloc.alloc().unwrap();
        assert!(alloc.alloc().is_err());
    }

    #[test]
    fn test_ref_counting() {
        let Some(ctx) = try_ctx() else { return };
        let mut alloc = PagedBlockAllocator::new(&ctx, 4, 16, 4, 64).unwrap();
        let b = alloc.alloc().unwrap();
        alloc.inc_ref(b).unwrap();
        assert!(alloc.is_shared(b));
        assert_eq!(alloc.ref_count(b), 2);

        alloc.dec_ref(b).unwrap();
        assert!(!alloc.is_shared(b));

        alloc.dec_ref(b).unwrap();
        assert_eq!(alloc.num_free(), 4); // back in free list
    }

    #[test]
    fn test_alloc_n_rollback() {
        let Some(ctx) = try_ctx() else { return };
        let mut alloc = PagedBlockAllocator::new(&ctx, 3, 16, 4, 64).unwrap();
        assert!(alloc.alloc_n(5).is_err()); // more than available
        assert_eq!(alloc.num_free(), 3); // nothing consumed
    }

    #[test]
    fn test_block_table_slot() {
        let mut table = BlockTable::new();
        table.push_block(BlockId(7));
        table.push_block(BlockId(3));
        let block_size = 16;
        // token 0 -> block 7, offset 0 -> slot 7*16+0 = 112
        assert_eq!(table.slot_for(0, block_size), Some(112));
        // token 17 -> logical block 1 = physical block 3, offset 1 -> slot 3*16+1 = 49
        assert_eq!(table.slot_for(17, block_size), Some(49));
        // token 64 -> logical block 4, not allocated
        assert_eq!(table.slot_for(64, block_size), None);
    }

    #[test]
    fn test_block_offset_elems() {
        let Some(ctx) = try_ctx() else { return };
        let alloc = PagedBlockAllocator::new(&ctx, 4, 16, 4, 64).unwrap();
        // elems_per_block = 16 * 4 * 64 = 4096
        assert_eq!(alloc.block_offset_elems(BlockId(0)), 0);
        assert_eq!(alloc.block_offset_elems(BlockId(1)), 4096);
        assert_eq!(alloc.block_offset_elems(BlockId(3)), 12288);
    }
}