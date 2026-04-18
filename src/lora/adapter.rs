use crate::core::backend::Backend;
use crate::lora::loader::LoadedAdapter;
use std::collections::HashMap;

pub struct AdapterRegistry<B: Backend> {
    adapters: HashMap<String, LoadedAdapter<B>>,
}

impl<B: Backend> AdapterRegistry<B> {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }
    pub fn register(&mut self, adapter: LoadedAdapter<B>) {
        self.adapters.insert(adapter.meta.id.to_owned(), adapter);
    }
    pub fn get(&self, id: &str) -> Option<&LoadedAdapter<B>> {
        self.adapters.get(id)
    }
    pub fn remove(&mut self, id: &str) -> Option<LoadedAdapter<B>> {
        self.adapters.remove(id)
    }
    pub fn list(&self) -> Vec<&str> {
        self.adapters.keys().map(|k| k.as_str()).collect()
    }
    pub fn contains(&self, id: &str) -> bool {
        self.adapters.contains_key(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::lora::config::{AdapterMeta, LoraConfig};
    use crate::lora::loader::LoadedAdapter;
    use std::collections::HashMap;

    fn make_adapter(id: &str) -> LoadedAdapter<CandleBackend> {
        LoadedAdapter {
            meta: AdapterMeta {
                id: id.to_string(),
                base_model: "llama-3-8b".to_string(),
                config: LoraConfig::default(),
            },
            weights: HashMap::new(),
        }
    }

    #[test]
    fn test_new_registry_is_empty() {
        let registry = AdapterRegistry::<CandleBackend>::new();
        assert!(registry.list().is_empty());
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_register_adapter() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("adapter-1"));
        assert!(registry.contains("adapter-1"));
    }

    #[test]
    fn test_register_multiple_adapters() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("a1"));
        registry.register(make_adapter("a2"));
        registry.register(make_adapter("a3"));
        assert_eq!(registry.list().len(), 3);
        assert!(registry.contains("a1"));
        assert!(registry.contains("a2"));
        assert!(registry.contains("a3"));
    }

    #[test]
    fn test_get_existing_adapter() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("my-adapter"));
        let got = registry.get("my-adapter");
        assert!(got.is_some());
        assert_eq!(got.unwrap().meta.id, "my-adapter");
    }

    #[test]
    fn test_get_missing_adapter() {
        let registry = AdapterRegistry::<CandleBackend>::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_remove_adapter() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("to-remove"));
        let removed = registry.remove("to-remove");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().meta.id, "to-remove");
        assert!(!registry.contains("to-remove"));
    }

    #[test]
    fn test_remove_missing_returns_none() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        assert!(registry.remove("ghost").is_none());
    }

    #[test]
    fn test_contains_true() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("exists"));
        assert!(registry.contains("exists"));
    }

    #[test]
    fn test_contains_false() {
        let registry = AdapterRegistry::<CandleBackend>::new();
        assert!(!registry.contains("missing"));
    }

    #[test]
    fn test_list_returns_all_ids() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("x"));
        registry.register(make_adapter("y"));
        let mut ids = registry.list();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn test_register_overwrites_existing() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("dup"));
        registry.register(make_adapter("dup")); // overwrite
        assert_eq!(registry.list().len(), 1);
    }

    #[test]
    fn test_remove_then_reregister() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("a"));
        registry.remove("a");
        assert!(!registry.contains("a"));
        registry.register(make_adapter("a"));
        assert!(registry.contains("a"));
    }

    #[test]
    fn test_registry_base_model_stored() {
        let mut registry = AdapterRegistry::<CandleBackend>::new();
        registry.register(make_adapter("a1"));
        let adapter = registry.get("a1").unwrap();
        assert_eq!(adapter.meta.base_model, "llama-3-8b");
    }
}
