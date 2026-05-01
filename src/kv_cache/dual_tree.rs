use crate::core::backend::Backend;
use crate::kv_cache::cache::{BaseCache, ResidualCache};
use crate::kv_cache::radix::RadixTree;

pub struct ForkResult<B: Backend> {
    pub base_cache: Option<BaseCache<B>>, // None = complete miss, needs recompute
    pub base_hit_len: usize,              // how many tokens matched in base tree
}

pub struct DualRadixTree<B: Backend> {
    base_tree: RadixTree<BaseCache<B>>,
    residual_tree: RadixTree<ResidualCache<B>>,
}

fn residual_key(tokens: &[u32], agent_id: &str) -> Vec<u32> {
    let mut key = tokens.to_vec();
    key.push(u32::MAX);
    key.extend(agent_id.bytes().map(|b| b as u32));
    key
}

impl<B: Backend> DualRadixTree<B> {
    pub fn new() -> Self {
        Self {
            base_tree: RadixTree::new(),
            residual_tree: RadixTree::new(),
        }
    }

    pub fn lookup_base(&self, tokens: &[u32]) -> Option<&BaseCache<B>> {
        self.base_tree.get(tokens)
    }
    pub fn prefix_match_base(&self, tokens: &[u32]) -> (usize, Option<&BaseCache<B>>) {
        let (cache, matched_len) = self.base_tree.prefix_match(tokens);
        (matched_len, cache)
    }

    pub fn lookup_residual(&self, tokens: &[u32], agent_id: &str) -> Option<&ResidualCache<B>> {
        let key = residual_key(tokens, agent_id);
        self.residual_tree.get(&key)
    }

    pub fn remove_residual(&mut self, tokens: &[u32], agent_id: &str) -> Option<ResidualCache<B>> {
        let key = residual_key(tokens, agent_id);
        self.residual_tree.remove(&key)
    }

    pub fn insert_base(&mut self, tokens: &[u32], cache: BaseCache<B>) {
        self.base_tree.insert(tokens, cache);
    }
    pub fn insert_residual(&mut self, tokens: &[u32], agent_id: &str, cache: ResidualCache<B>) {
        let key = residual_key(tokens, agent_id);
        self.residual_tree.insert(&key, cache);
    }

    pub fn remove_base(&mut self, tokens: &[u32]) -> Option<BaseCache<B>> {
        self.base_tree.remove(tokens)
    }

    // fork semantics — Step 1 of ForkKV paper Figure 9
    // inherits base cache via prefix match, allocates slot for residual
    pub fn fork(&self, tokens: &[u32], _new_agent_id: &str, _adapter_id: &str) -> ForkResult<B> {
        let (hit_len, _cache) = self.prefix_match_base(tokens);
        ForkResult {
            base_cache: None,
            base_hit_len: hit_len,
        }
    }

    pub fn evict_base_lru(&mut self) -> Option<BaseCache<B>> {
        self.base_tree.evict_lru()
    }
    pub fn evict_residual_lru(&mut self) -> Option<ResidualCache<B>> {
        self.residual_tree.evict_lru()
    }
}

// dual_tree.rs tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;
    use crate::kv_cache::cache::{BaseCache, CacheBlock, ResidualCache};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_block() -> CacheBlock<CandleBackend> {
        use crate::core::backend::CandleTensor;
        CacheBlock {
            k: CandleTensor::zeros(&Shape::new(&[1, 4, 16]), DType::F32, &cpu()).unwrap(),
            v: CandleTensor::zeros(&Shape::new(&[1, 4, 16]), DType::F32, &cpu()).unwrap(),
            layer_idx: 0,
        }
    }

    fn make_base_cache(tokens: Vec<u32>) -> BaseCache<CandleBackend> {
        BaseCache::new(tokens, vec![make_block()])
    }

    fn make_residual_cache(agent_id: &str) -> ResidualCache<CandleBackend> {
        ResidualCache::new(
            agent_id.to_string(),
            "adapter-A".to_string(),
            vec![make_block()],
        )
    }

    #[test]
    fn test_new_dual_tree() {
        let tree = DualRadixTree::<CandleBackend>::new();
        assert!(tree.lookup_base(&[1, 2, 3]).is_none());
    }

    #[test]
    fn test_insert_and_lookup_base() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_base(&[1, 2, 3], make_base_cache(vec![1, 2, 3]));
        assert!(tree.lookup_base(&[1, 2, 3]).is_some());
    }

    #[test]
    fn test_lookup_base_miss() {
        let tree = DualRadixTree::<CandleBackend>::new();
        assert!(tree.lookup_base(&[9, 9, 9]).is_none());
    }

    #[test]
    fn test_insert_and_lookup_residual() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_residual(&[1, 2, 3], "agent-1", make_residual_cache("agent-1"));
        assert!(tree.lookup_residual(&[1, 2, 3], "agent-1").is_some());
    }

    #[test]
    fn test_residual_different_agents_same_tokens() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_residual(&[1, 2], "agent-1", make_residual_cache("agent-1"));
        tree.insert_residual(&[1, 2], "agent-2", make_residual_cache("agent-2"));
        assert!(tree.lookup_residual(&[1, 2], "agent-1").is_some());
        assert!(tree.lookup_residual(&[1, 2], "agent-2").is_some());
    }

    #[test]
    fn test_residual_agent_isolation() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_residual(&[1, 2], "agent-1", make_residual_cache("agent-1"));
        // agent-2 should not see agent-1's cache
        assert!(tree.lookup_residual(&[1, 2], "agent-2").is_none());
    }

    #[test]
    fn test_prefix_match_base_exact() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_base(&[1, 2, 3], make_base_cache(vec![1, 2, 3]));
        let (len, cache) = tree.prefix_match_base(&[1, 2, 3]);
        assert_eq!(len, 3);
        assert!(cache.is_some());
    }

    #[test]
    fn test_prefix_match_base_partial() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_base(&[1, 2, 3], make_base_cache(vec![1, 2, 3]));
        let (len, cache) = tree.prefix_match_base(&[1, 2, 3, 4, 5]);
        assert_eq!(len, 3);
        assert!(cache.is_some());
    }

    #[test]
    fn test_prefix_match_base_miss() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        let (len, cache) = tree.prefix_match_base(&[9, 9, 9]);
        assert_eq!(len, 0);
        assert!(cache.is_none());
    }

    #[test]
    fn test_remove_base() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_base(&[1, 2, 3], make_base_cache(vec![1, 2, 3]));
        let removed = tree.remove_base(&[1, 2, 3]);
        assert!(removed.is_some());
        assert!(tree.lookup_base(&[1, 2, 3]).is_none());
    }

    #[test]
    fn test_remove_residual() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_residual(&[1, 2], "agent-1", make_residual_cache("agent-1"));
        let removed = tree.remove_residual(&[1, 2], "agent-1");
        assert!(removed.is_some());
        assert!(tree.lookup_residual(&[1, 2], "agent-1").is_none());
    }

    #[test]
    fn test_fork_full_hit() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_base(&[1, 2, 3], make_base_cache(vec![1, 2, 3]));
        let result = tree.fork(&[1, 2, 3], "new-agent", "adapter-A");
        assert_eq!(result.base_hit_len, 3);
    }

    #[test]
    fn test_fork_miss() {
        let tree = DualRadixTree::<CandleBackend>::new();
        let result = tree.fork(&[1, 2, 3], "new-agent", "adapter-A");
        assert_eq!(result.base_hit_len, 0);
    }

    #[test]
    fn test_evict_base_lru() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_base(&[1], make_base_cache(vec![1]));
        let evicted = tree.evict_base_lru();
        assert!(evicted.is_some());
        assert!(tree.lookup_base(&[1]).is_none());
    }

    #[test]
    fn test_evict_residual_lru() {
        let mut tree = DualRadixTree::<CandleBackend>::new();
        tree.insert_residual(&[1], "agent-1", make_residual_cache("agent-1"));
        let evicted = tree.evict_residual_lru();
        assert!(evicted.is_some());
    }
}
