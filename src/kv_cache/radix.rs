use std::{collections::HashMap, hash::Hash, time::Instant};

use tracing::Instrument;
use tracing_subscriber::fmt::time;
use unix_ts::Timestamp;

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
