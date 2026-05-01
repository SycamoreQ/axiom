use tracing::Instrument;

use crate::core::backend::Backend;
use crate::cuda::allocator::{BlockId, BlockTable, PagedBlockAllocator};
use crate::cuda::context::CudaContext;
use crate::cuda::error::Result;
use crate::cuda::kernels::launch_copy_blocks_f16;
use crate::cuda::paged_session::{CacheKind, PagedSession};
use crate::inference::session::{Session, SessionId};
use crate::kv_cache::manager::KVConfig;
use crate::kv_cache::manager::KVManager;
use cudarc::driver::{CudaFunction, CudaModule, CudaStream};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::alloc::alloc;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub struct ForkManager<B: Backend> {
    kv_manager: KVManager<B>,
    alloc: Arc<Mutex<PagedBlockAllocator>>,
    sessions: HashMap<u64, PagedSession>,
}

impl<B: Backend> ForkManager<B> {
    pub fn new(
        alloc: Arc<Mutex<PagedBlockAllocator>>,
        max_base_entries: usize,
        max_residual_entries: usize,
    ) -> Self {
        let kv_manager = KVManager::new(KVConfig {
            max_base_entries,
            max_residual_entries,
        });

        Self {
            kv_manager,
            alloc,
            sessions: HashMap::new(),
        }
    }

    pub fn register_session(&mut self, session_id: u64) -> Result<()> {
        if self.sessions.contains_key(&session_id) {
            return Ok(());
        }
        let new_session = PagedSession::new(session_id);
        self.sessions.insert(session_id, new_session);

        Ok(())
    }

    pub fn fork_session(&mut self, parent_id: u64, child_id: u64, ctx: &CudaContext) -> Result<()> {
        let (parent_blocks, parent_seq_len) = {
            let parent = self
                .sessions
                .get(&parent_id)
                .ok_or(CudaError::InvalidSession(parent_id))?;
            (parent.block_table.physical.clone(), parent.seq_len)
        };

        {
            let mut allocator = self
                .alloc
                .lock()
                .map_err(|_| CudaError::Internal("Allocator mutex poisoned".to_string()))?;

            for &phys_id in &parent_blocks {
                // Convert i32 back to BlockId to call the allocator method
                allocator.inc_ref(BlockId(phys_id as usize))?;
            }
        };

        let child_session = PagedSession {
            session_id: child_id,
            block_table: BlockTable {
                physical: parent_blocks,
                seq_len: parent_seq_len,
            },
            seq_len: parent_seq_len,
            cache_kind: CacheKind::Forked {
                base_len: parent_seq_len,
            },
        };

        self.sessions.insert(child_id, child_session);

        Ok(())
    }

    pub fn copy_on_write(
        &mut self,
        session_id: u64,
        block_idx: usize,
        ctx: &CudaContext,
    ) -> Result<()> {
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or(CudaError::InvalidSession(session_id))?;

        let old_phys_id = session
            .block_table
            .physical
            .get(block_idx)
            .map(|&id| BlockId(id as usize))
            .ok_or(CudaError::Internal("Block index out of bounds".into()))?;

        let mut allocator = self
            .alloc
            .lock()
            .map_err(|_| CudaError::Internal("Allocator mutex poisoned".into()))?;

        if allocator.is_shared(old_phys_id) {
            let new_phys_id = allocator.alloc()?;

            let mapping: Vec<i64> = vec![old_phys_id.0 as i64, new_phys_id.0 as i64];
            let mapping_buf = ctx
                .device()
                .htod_sync_copy(&mapping)
                .map_err(CudaError::Driver)?;

            launch_copy_blocks_f16(
                ctx,
                &mut allocator.k_cache.slice_mut(..),
                &mut allocator.v_cache.slice_mut(..),
                &mapping_buf.slice(..),
                1, // num_pairs
                allocator.block_size,
                allocator.num_kv_heads,
                allocator.head_dim,
            )?;

            session.block_table.physical[block_idx] = new_phys_id.0 as i32;
            allocator.dec_ref(old_phys_id)?;
        }

        Ok(())
    }

    pub fn free_session(&mut self, session_id: u64) -> Result<()> {
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or(CudaError::InvalidSession(session_id))?;

        let mut allocator = self
            .alloc
            .lock()
            .map_err(|_| CudaError::Internal("Allocator mutex poisoned".into()))?;

        session.free_blocks(&mut *allocator);
        Ok(())
    }

    pub fn get_session(&self, session_id: u64) -> Option<&PagedSession> {
        self.sessions.get(&session_id)
    }
    pub fn get_session_mut(&mut self, session_id: u64) -> Option<&mut PagedSession> {
        self.sessions.get_mut(&session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::cuda::allocator::PagedBlockAllocator;
    use crate::cuda::context::CudaContext;
    use std::sync::{Arc, Mutex};

    fn try_setup() -> Option<(ForkManager<CandleBackend>, Arc<Mutex<PagedBlockAllocator>>)> {
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok()?;
        let ctx = CudaContext::new(0, &ptx).ok()?;
        let alloc = PagedBlockAllocator::new(ctx, 32, 4, 2, 32).ok()?;
        let alloc = Arc::new(Mutex::new(alloc));
        let fm = ForkManager::<CandleBackend>::new(Arc::clone(&alloc), 100, 400);
        Some((fm, alloc))
    }

    #[test]
    fn test_register_session() {
        let Some((mut fm, _)) = try_setup() else {
            return;
        };
        fm.register_session(1).unwrap();
        assert!(fm.get_session(1).is_some());
    }

    #[test]
    fn test_register_session_twice_is_noop() {
        let Some((mut fm, _)) = try_setup() else {
            return;
        };
        fm.register_session(1).unwrap();
        fm.register_session(1).unwrap();
        assert!(fm.get_session(1).is_some());
    }

    #[test]
    fn test_fork_inherits_parent_blocks() {
        let Some((mut fm, alloc)) = try_setup() else {
            return;
        };
        fm.register_session(1).unwrap();
        {
            let mut a = alloc.lock().unwrap();
            fm.get_session_mut(1)
                .unwrap()
                .allocate_blocks(&mut a, 6)
                .unwrap();
        }
        let ctx_ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok();
        let Some(ptx) = ctx_ptx else { return };
        let ctx = crate::cuda::context::CudaContext::new(0, &ptx).ok();
        let Some(ctx) = ctx else { return };
        fm.fork_session(1, 2, &ctx).unwrap();
        let parent_blocks = fm.get_session(1).unwrap().block_table.physical.clone();
        let child_blocks = fm.get_session(2).unwrap().block_table.physical.clone();
        assert_eq!(parent_blocks, child_blocks);
    }

    #[test]
    fn test_fork_increments_ref_count() {
        let Some((mut fm, alloc)) = try_setup() else {
            return;
        };
        fm.register_session(1).unwrap();
        {
            let mut a = alloc.lock().unwrap();
            fm.get_session_mut(1)
                .unwrap()
                .allocate_blocks(&mut a, 4)
                .unwrap();
        }
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok();
        let Some(ptx) = ptx else { return };
        let ctx = crate::cuda::context::CudaContext::new(0, &ptx).ok();
        let Some(ctx) = ctx else { return };
        fm.fork_session(1, 2, &ctx).unwrap();
        let a = alloc.lock().unwrap();
        let block_id = crate::cuda::allocator::BlockId(
            fm.get_session(1).unwrap().block_table.physical[0] as usize,
        );
        assert_eq!(a.ref_count(block_id), 2);
    }

    #[test]
    fn test_free_parent_after_fork_keeps_child() {
        let Some((mut fm, alloc)) = try_setup() else {
            return;
        };
        fm.register_session(1).unwrap();
        {
            let mut a = alloc.lock().unwrap();
            fm.get_session_mut(1)
                .unwrap()
                .allocate_blocks(&mut a, 4)
                .unwrap();
        }
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok();
        let Some(ptx) = ptx else { return };
        let ctx = crate::cuda::context::CudaContext::new(0, &ptx).ok();
        let Some(ctx) = ctx else { return };
        fm.fork_session(1, 2, &ctx).unwrap();
        fm.free_session(1).unwrap();
        // child blocks should still have ref_count 1
        let a = alloc.lock().unwrap();
        let block_id = crate::cuda::allocator::BlockId(
            fm.get_session(2).unwrap().block_table.physical[0] as usize,
        );
        assert_eq!(a.ref_count(block_id), 1);
    }

    #[test]
    fn test_free_session_unknown_id_returns_err() {
        let Some((mut fm, _)) = try_setup() else {
            return;
        };
        assert!(fm.free_session(99).is_err());
    }

    #[test]
    fn test_get_session_unknown_returns_none() {
        let Some((fm, _)) = try_setup() else { return };
        assert!(fm.get_session(42).is_none());
    }

    #[test]
    fn test_fork_unknown_parent_returns_err() {
        let Some((mut fm, _)) = try_setup() else {
            return;
        };
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok();
        let Some(ptx) = ptx else { return };
        let ctx = crate::cuda::context::CudaContext::new(0, &ptx).ok();
        let Some(ctx) = ctx else { return };
        assert!(fm.fork_session(99, 1, &ctx).is_err());
    }
}
