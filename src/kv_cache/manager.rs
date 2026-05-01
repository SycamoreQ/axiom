
use crate::core::backend::Backend;
use crate::kv_cache::cache::{BaseCache, ResidualCache};
use crate::kv_cache::dual_tree::{DualRadixTree, ForkResult};

pub struct KVConfig {
    pub max_base_entries: usize, // evict when base tree exceeds this
    pub max_residual_entries: usize,
}

impl Default for KVConfig {
    fn default() -> Self {
        Self {
            max_base_entries: 1000,
            max_residual_entries: 4000,
        }
    }
}

pub enum EvictionPolicy {
    Lru,
}

pub struct CacheStats {
    pub base_entries: usize,
    pub residual_entries: usize,
}

pub struct KVManager<B: Backend> {
    tree: DualRadixTree<B>,
    config: KVConfig,
    base_count: usize,
    residual_count: usize,
}

impl<B: Backend> KVManager<B> {
    pub fn new(config: KVConfig) -> Self {
        Self {
            tree: DualRadixTree::new(),
            config: config,
            base_count: 0,
            residual_count: 0,
        }
    }
    pub fn fork_agent(&self, tokens: &[u32], agent_id: &str, adapter_id: &str) -> ForkResult<B> {
        self.tree.fork(tokens, agent_id, adapter_id)
    }
    pub fn store_base(&mut self, tokens: &[u32], cache: BaseCache<B>) {
        self.tree.insert_base(tokens, cache);
        self.base_count += 1;
        self.evict_if_needed();
    }

    pub fn store_residual(&mut self, tokens: &[u32], agent_id: &str, cache: ResidualCache<B>) {
        self.tree.insert_residual(tokens, agent_id, cache);
        self.residual_count += 1;
        self.evict_if_needed();
    }
    pub fn lookup_base(&self, tokens: &[u32]) -> Option<&BaseCache<B>> {
        self.tree.lookup_base(tokens)
    }
    pub fn lookup_residual(&self, tokens: &[u32], agent_id: &str) -> Option<&ResidualCache<B>> {
        self.tree.lookup_residual(tokens, agent_id)
    }
    pub fn evict_if_needed(&mut self) {
        while self.base_count > self.config.max_base_entries {
            if self.tree.evict_base_lru().is_some() {
                self.base_count -= 1;
            } else {
                break;
            }
        }
        while self.residual_count > self.config.max_residual_entries {
            if self.tree.evict_residual_lru().is_some() {
                self.residual_count -= 1;
            } else {
                break;
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            base_entries: self.base_count,
            residual_entries: self.residual_count,
        }
    }
}

// manager.rs tests
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

    fn make_base(tokens: Vec<u32>) -> BaseCache<CandleBackend> {
        BaseCache::new(tokens, vec![make_block()])
    }

    fn make_residual(agent_id: &str) -> ResidualCache<CandleBackend> {
        ResidualCache::new(
            agent_id.to_string(),
            "adapter-A".to_string(),
            vec![make_block()],
        )
    }

    fn make_manager() -> KVManager<CandleBackend> {
        KVManager::new(KVConfig::default())
    }

    #[test]
    fn test_new_manager_stats_zero() {
        let mgr = make_manager();
        let stats = mgr.stats();
        assert_eq!(stats.base_entries, 0);
        assert_eq!(stats.residual_entries, 0);
    }

    #[test]
    fn test_store_and_lookup_base() {
        let mut mgr = make_manager();
        mgr.store_base(&[1, 2, 3], make_base(vec![1, 2, 3]));
        assert!(mgr.lookup_base(&[1, 2, 3]).is_some());
    }

    #[test]
    fn test_store_base_increments_count() {
        let mut mgr = make_manager();
        mgr.store_base(&[1], make_base(vec![1]));
        mgr.store_base(&[2], make_base(vec![2]));
        assert_eq!(mgr.stats().base_entries, 2);
    }

    #[test]
    fn test_store_and_lookup_residual() {
        let mut mgr = make_manager();
        mgr.store_residual(&[1, 2], "agent-1", make_residual("agent-1"));
        assert!(mgr.lookup_residual(&[1, 2], "agent-1").is_some());
    }

    #[test]
    fn test_store_residual_increments_count() {
        let mut mgr = make_manager();
        mgr.store_residual(&[1], "agent-1", make_residual("agent-1"));
        mgr.store_residual(&[1], "agent-2", make_residual("agent-2"));
        assert_eq!(mgr.stats().residual_entries, 2);
    }

    #[test]
    fn test_lookup_base_miss() {
        let mgr = make_manager();
        assert!(mgr.lookup_base(&[9, 9]).is_none());
    }

    #[test]
    fn test_lookup_residual_miss() {
        let mgr = make_manager();
        assert!(mgr.lookup_residual(&[1, 2], "ghost").is_none());
    }

    #[test]
    fn test_fork_agent_hit() {
        let mut mgr = make_manager();
        mgr.store_base(&[1, 2, 3], make_base(vec![1, 2, 3]));
        let result = mgr.fork_agent(&[1, 2, 3], "new-agent", "adapter-A");
        assert_eq!(result.base_hit_len, 3);
    }

    #[test]
    fn test_fork_agent_miss() {
        let mgr = make_manager();
        let result = mgr.fork_agent(&[1, 2, 3], "new-agent", "adapter-A");
        assert_eq!(result.base_hit_len, 0);
    }

    #[test]
    fn test_evict_if_needed_base() {
        let mut mgr = KVManager::new(KVConfig {
            max_base_entries: 2,
            max_residual_entries: 1000,
        });
        mgr.store_base(&[1], make_base(vec![1]));
        mgr.store_base(&[2], make_base(vec![2]));
        mgr.store_base(&[3], make_base(vec![3])); // triggers eviction
        assert!(mgr.stats().base_entries <= 2);
    }

    #[test]
    fn test_evict_if_needed_residual() {
        let mut mgr = KVManager::new(KVConfig {
            max_base_entries: 1000,
            max_residual_entries: 1,
        });
        mgr.store_residual(&[1], "agent-1", make_residual("agent-1"));
        mgr.store_residual(&[2], "agent-2", make_residual("agent-2")); // triggers eviction
        assert!(mgr.stats().residual_entries <= 1);
    }

    #[test]
    fn test_stats_reflects_current_state() {
        let mut mgr = make_manager();
        mgr.store_base(&[1], make_base(vec![1]));
        mgr.store_residual(&[1], "a", make_residual("a"));
        let stats = mgr.stats();
        assert_eq!(stats.base_entries, 1);
        assert_eq!(stats.residual_entries, 1);
    }
}
