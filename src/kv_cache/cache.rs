use crate::core::backend::Backend;

/*
Disaggregated cache blocks , this is where the bcache and rcache structs live
*/

// A single cached block — one layer's K and V tensors
pub struct CacheBlock<B: Backend> {
    pub k: B::Tensor,
    pub v: B::Tensor,
    pub layer_idx: usize,
}

// Base cache — shared across all agents processing the same context
pub struct BaseCache<B: Backend> {
    pub blocks: Vec<CacheBlock<B>>, // one per layer
    pub token_ids: Vec<u32>,        // the sequence this cache covers
    pub ref_count: usize,           // how many agents share this
}

impl<B: Backend> BaseCache<B> {
    pub fn new(token_ids: Vec<u32>, blocks: Vec<CacheBlock<B>>) -> Self {
        Self {
            blocks,
            token_ids,
            ref_count: 1,
        }
    }

    pub fn increment_ref(&mut self) {
        self.ref_count += 1;
    }
    pub fn decrement_ref(&mut self) {
        self.ref_count = self.ref_count.saturating_sub(1);
    }
    pub fn is_shared(&self) -> bool {
        self.ref_count > 1
    }
    pub fn seq_len(&self) -> usize {
        self.token_ids.len()
    }
}

// Residual cache — unique per agent, stores xA_i (low-rank projection)
// Much smaller than base cache due to low-rank r << n
pub struct ResidualCache<B: Backend> {
    pub blocks: Vec<CacheBlock<B>>,
    pub agent_id: String,
    pub adapter_id: String,
}

impl<B: Backend> ResidualCache<B> {
    pub fn new(agent_id: String, adapter_id: String, blocks: Vec<CacheBlock<B>>) -> Self {
        Self {
            blocks,
            agent_id,
            adapter_id,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }
}
