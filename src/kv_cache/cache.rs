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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_block(layer_idx: usize) -> CacheBlock<CandleBackend> {
        use crate::core::backend::CandleTensor;
        CacheBlock {
            k: CandleTensor::zeros(&Shape::new(&[2, 4, 16]), DType::F32, &cpu()).unwrap(),
            v: CandleTensor::zeros(&Shape::new(&[2, 4, 16]), DType::F32, &cpu()).unwrap(),
            layer_idx,
        }
    }

    // ── CacheBlock ──

    #[test]
    fn test_cache_block_layer_idx() {
        let block = make_block(3);
        assert_eq!(block.layer_idx, 3);
    }

    // ── BaseCache ──

    #[test]
    fn test_base_cache_new() {
        let blocks = vec![make_block(0), make_block(1)];
        let cache = BaseCache::<CandleBackend>::new(vec![1, 2, 3], blocks);
        assert_eq!(cache.ref_count, 1);
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.blocks.len(), 2);
    }

    #[test]
    fn test_base_cache_increment_ref() {
        let mut cache = BaseCache::<CandleBackend>::new(vec![1], vec![make_block(0)]);
        cache.increment_ref();
        assert_eq!(cache.ref_count, 2);
        assert!(cache.is_shared());
    }

    #[test]
    fn test_base_cache_decrement_ref() {
        let mut cache = BaseCache::<CandleBackend>::new(vec![1], vec![make_block(0)]);
        cache.increment_ref();
        cache.decrement_ref();
        assert_eq!(cache.ref_count, 1);
        assert!(!cache.is_shared());
    }

    #[test]
    fn test_base_cache_decrement_ref_saturates_at_zero() {
        let mut cache = BaseCache::<CandleBackend>::new(vec![1], vec![make_block(0)]);
        cache.decrement_ref();
        cache.decrement_ref(); // should not underflow
        assert_eq!(cache.ref_count, 0);
    }

    #[test]
    fn test_base_cache_is_shared_false_initially() {
        let cache = BaseCache::<CandleBackend>::new(vec![1, 2], vec![make_block(0)]);
        assert!(!cache.is_shared());
    }

    #[test]
    fn test_base_cache_seq_len() {
        let cache = BaseCache::<CandleBackend>::new(vec![10, 20, 30, 40], vec![]);
        assert_eq!(cache.seq_len(), 4);
    }

    #[test]
    fn test_base_cache_token_ids_stored() {
        let tokens = vec![1u32, 2, 3];
        let cache = BaseCache::<CandleBackend>::new(tokens.clone(), vec![]);
        assert_eq!(cache.token_ids, tokens);
    }

    // ── ResidualCache ──

    #[test]
    fn test_residual_cache_new() {
        let cache = ResidualCache::<CandleBackend>::new(
            "agent-1".to_string(),
            "adapter-A".to_string(),
            vec![make_block(0), make_block(1)],
        );
        assert_eq!(cache.agent_id, "agent-1");
        assert_eq!(cache.adapter_id, "adapter-A");
        assert_eq!(cache.num_layers(), 2);
    }

    #[test]
    fn test_residual_cache_num_layers_empty() {
        let cache = ResidualCache::<CandleBackend>::new("a".to_string(), "b".to_string(), vec![]);
        assert_eq!(cache.num_layers(), 0);
    }
}
