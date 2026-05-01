use std::{collections::HashMap, time::Instant};


/*
A radix tree keyed by token sequences. Enables O(prefix_len) lookup for shared prefix matching. This is the main
data structure both caches are present in.
*/

pub struct RadixNode<V> {
    pub children: HashMap<u32, RadixNode<V>>,
    pub value: Option<V>,
    pub last_accessed: std::time::Instant, // for LRU eviction
}

pub struct RadixTree<V> {
    root: RadixNode<V>,
}

impl<V> RadixTree<V> {
    pub fn new() -> Self {
        let node = RadixNode {
            children: HashMap::new(),
            value: None,
            last_accessed: Instant::now(),
        };
        Self { root: node }
    }
    pub fn insert(&mut self, tokens: &[u32], value: V) {
        let mut current = &mut self.root;

        for &token in tokens {
            current = current.children.entry(token).or_insert_with(|| RadixNode {
                children: HashMap::new(),
                value: None,
                last_accessed: Instant::now(),
            });
        }

        current.value = Some(value);
        current.last_accessed = Instant::now();
    }

    pub fn get(&self, tokens: &[u32]) -> Option<&V> {
        let mut current = &self.root;

        for token in tokens {
            current = current.children.get(token)?;
        }

        current.value.as_ref()
    }

    pub fn prefix_match(&self, tokens: &[u32]) -> (Option<&V>, usize) {
        let mut current = &self.root;
        let mut matched_len = 0;
        let mut last_value = None;
        let mut best_matched_len = 0;

        for &token in tokens {
            if let Some(next_node) = current.children.get(&token) {
                current = next_node;
                matched_len += 1;

                if let Some(val) = &current.value {
                    last_value = Some(val);
                    best_matched_len = matched_len;
                }
            } else {
                break;
            }
        }

        (last_value, best_matched_len)
    }

    pub fn remove(&mut self, tokens: &[u32]) -> Option<V> {
        let mut current = &mut self.root;

        for token in tokens {
            current = current.children.get_mut(token)?;
        }

        current.value.take()
    }

    pub fn evict_lru(&mut self) -> Option<V> {
        let mut oldest_path: Option<Vec<u32>> = None;
        let mut oldest_elapsed = std::time::Duration::ZERO;

        fn walk<V>(
            node: &RadixNode<V>,
            path: &mut Vec<u32>,
            oldest_path: &mut Option<Vec<u32>>,
            oldest_elapsed: &mut std::time::Duration,
        ) {
            if node.value.is_some() {
                let elapsed = node.last_accessed.elapsed();
                if elapsed > *oldest_elapsed {
                    *oldest_elapsed = elapsed;
                    *oldest_path = Some(path.clone());
                }
            }

            for (&token, child) in &node.children {
                path.push(token);
                walk(child, path, oldest_path, oldest_elapsed);
                path.pop();
            }
        }

        walk(
            &self.root,
            &mut vec![],
            &mut oldest_path,
            &mut oldest_elapsed,
        );

        if let Some(path) = oldest_path {
            self.remove(&path)
        } else {
            None
        }
    }
}

// radix.rs tests
#[cfg(test)]
mod tests {
    use super::*;

    // ── insert + get ──

    #[test]
    fn test_insert_and_get() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        assert_eq!(tree.get(&[1, 2, 3]), Some(&42));
    }

    #[test]
    fn test_get_missing_returns_none() {
        let tree: RadixTree<u32> = RadixTree::new();
        assert_eq!(tree.get(&[1, 2, 3]), None);
    }

    #[test]
    fn test_get_partial_path_returns_none() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        // [1, 2] exists as a node but has no value
        assert_eq!(tree.get(&[1, 2]), None);
    }

    #[test]
    fn test_insert_multiple_sequences() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 10);
        tree.insert(&[1, 2, 4], 20);
        tree.insert(&[5, 6], 30);
        assert_eq!(tree.get(&[1, 2, 3]), Some(&10));
        assert_eq!(tree.get(&[1, 2, 4]), Some(&20));
        assert_eq!(tree.get(&[5, 6]), Some(&30));
    }

    #[test]
    fn test_insert_overwrites_existing() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2], 10);
        tree.insert(&[1, 2], 99);
        assert_eq!(tree.get(&[1, 2]), Some(&99));
    }

    #[test]
    fn test_insert_empty_tokens() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[], 7);
        assert_eq!(tree.get(&[]), Some(&7));
    }

    // ── prefix_match ──

    #[test]
    fn test_prefix_match_exact() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        let (val, len) = tree.prefix_match(&[1, 2, 3]);
        assert_eq!(val, Some(&42));
        assert_eq!(len, 3);
    }

    #[test]
    fn test_prefix_match_partial() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        // query longer than stored — should still match at [1,2,3]
        let (val, len) = tree.prefix_match(&[1, 2, 3, 4, 5]);
        assert_eq!(val, Some(&42));
        assert_eq!(len, 3);
    }

    #[test]
    fn test_prefix_match_no_match() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        let (val, len) = tree.prefix_match(&[9, 9, 9]);
        assert_eq!(val, None);
        assert_eq!(len, 0);
    }

    #[test]
    fn test_prefix_match_returns_deepest_value() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2], 10);
        tree.insert(&[1, 2, 3], 20);
        // query [1, 2, 3, 4] should match at depth 3
        let (val, len) = tree.prefix_match(&[1, 2, 3, 4]);
        assert_eq!(val, Some(&20));
        assert_eq!(len, 3);
    }

    #[test]
    fn test_prefix_match_intermediate_node_with_value() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1], 5);
        tree.insert(&[1, 2, 3], 15);
        let (val, len) = tree.prefix_match(&[1, 2]);
        // [1] has a value, [1,2] does not — best match is at depth 1
        assert_eq!(val, Some(&5));
        assert_eq!(len, 1);
    }

    #[test]
    fn test_prefix_match_empty_tree() {
        let tree: RadixTree<u32> = RadixTree::new();
        let (val, len) = tree.prefix_match(&[1, 2, 3]);
        assert_eq!(val, None);
        assert_eq!(len, 0);
    }

    // ── remove ──

    #[test]
    fn test_remove_existing() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        let removed = tree.remove(&[1, 2, 3]);
        assert_eq!(removed, Some(42));
        assert_eq!(tree.get(&[1, 2, 3]), None);
    }

    #[test]
    fn test_remove_missing_returns_none() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        assert_eq!(tree.remove(&[1, 2, 3]), None);
    }

    #[test]
    fn test_remove_does_not_affect_siblings() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 10);
        tree.insert(&[1, 2, 4], 20);
        tree.remove(&[1, 2, 3]);
        assert_eq!(tree.get(&[1, 2, 4]), Some(&20));
    }

    #[test]
    fn test_remove_parent_does_not_affect_child() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2], 10);
        tree.insert(&[1, 2, 3], 20);
        tree.remove(&[1, 2]);
        assert_eq!(tree.get(&[1, 2, 3]), Some(&20));
        assert_eq!(tree.get(&[1, 2]), None);
    }

    // ── evict_lru ──

    #[test]
    fn test_evict_lru_empty_tree_returns_none() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        assert_eq!(tree.evict_lru(), None);
    }

    #[test]
    fn test_evict_lru_single_entry() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1, 2, 3], 42);
        let evicted = tree.evict_lru();
        assert_eq!(evicted, Some(42));
        assert_eq!(tree.get(&[1, 2, 3]), None);
    }

    #[test]
    fn test_evict_lru_removes_oldest() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1], 10);
        std::thread::sleep(std::time::Duration::from_millis(5));
        tree.insert(&[2], 20);
        // [1] was inserted first so it is the oldest
        let evicted = tree.evict_lru();
        assert_eq!(evicted, Some(10));
        // [2] should still be present
        assert_eq!(tree.get(&[2]), Some(&20));
    }

    #[test]
    fn test_evict_lru_twice() {
        let mut tree: RadixTree<u32> = RadixTree::new();
        tree.insert(&[1], 10);
        std::thread::sleep(std::time::Duration::from_millis(5));
        tree.insert(&[2], 20);
        tree.evict_lru(); // removes [1]
        tree.evict_lru(); // removes [2]
        assert_eq!(tree.get(&[1]), None);
        assert_eq!(tree.get(&[2]), None);
    }
}
