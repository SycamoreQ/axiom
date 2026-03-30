use crate::tokenizer::pretokenize::*;
use crate::tokenizer::vocab::{TokenID, Vocab};
use serde_json::*;
use std::collections::*;
use std::sync::*;

/*
where text actually becomes token IDs.

What it receives and what it returns
It receives a Vec<PreToken> from pretokenize.rs and returns a Vec<TokenID>.

PreToken::Special(id) — already resolved, just emit the ID directly
PreToken::Text(s) — this is a byte-mapped string that needs to go through the merge loop

Efficient O(n log n) — use a priority queue (BinaryHeap).
The insight is that after merging a pair at position i, only the pairs involving position i-1 and i+1 change —
everything else stays the same. So you only need to update two entries in the heap, not rescan everything.

This is a todo for later , for now tests pass in the O(n^2) algorithm which is in encode_chunks()
 */

#[derive(Debug)]
pub enum MergeMode {
    Score,
    Rank,
}

#[derive(Debug, PartialEq)]
pub enum MergePriority {
    Score(f32),  // higher is better — natural float ordering
    Rank(usize), // lower is better — needs Reverse ordering
}

impl Ord for MergePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (MergePriority::Score(a), MergePriority::Score(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
            (MergePriority::Rank(a), MergePriority::Rank(b)) => {
                // lower rank = higher priority so reverse the ordering
                b.cmp(a)
            }
            _ => std::cmp::Ordering::Equal,
        }
    }
}

impl Eq for MergePriority {}

impl PartialOrd for MergePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct MergeEntry {
    priority: MergePriority, // score or rank — used for ordering
    left: usize,             // index into token array
    right: usize,            // index into token array
    left_gen: u32,           // generation of left token when this entry was created
    right_gen: u32,          // generation of right token when this entry was created
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}
impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for MergeEntry {}

impl PartialEq for MergeEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

#[derive(Debug)]
struct TokenState {
    token: String,       // current token string
    id: Option<TokenID>, // resolved ID if known
    gen: u32,            // incremented when this slot is invalidated by a merge
    active: bool,        // false if this slot has been merged away
}

pub struct Bpe {
    vocab: Arc<Vocab>,
    merges: HashMap<(String, String), usize>, // rank mode
    mode: MergeMode,
}

impl Bpe {
    pub fn new(vocab: Arc<Vocab>, merges: Vec<(String, String)>, mode: MergeMode) -> Self {
        let merges_map = merges
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank))
            .collect();

        Self {
            vocab,
            merges: merges_map,
            mode,
        }
    }

    fn encode_chunk(&self, s: &str) -> Vec<TokenID> {
        //initialize one TokenState per character
        let mut states: Vec<TokenState> = s
            .chars()
            .map(|c| {
                let token = c.to_string();
                let id = self.vocab.token_to_id(&token);
                TokenState {
                    token,
                    id,
                    gen: 0,
                    active: true,
                }
            })
            .collect();

        loop {
            let active_indices: Vec<usize> = states
                .iter()
                .enumerate()
                .filter(|(_, s)| s.active)
                .map(|(i, _)| i)
                .collect();

            if active_indices.len() < 2 {
                break; // nothing left to merge
            }

            let mut best: Option<(usize, usize, MergePriority)> = None;

            // scan all adjacent active pairs
            for window in active_indices.windows(2) {
                let (li, ri) = (window[0], window[1]);
                let left = &states[li].token;
                let right = &states[ri].token;

                match self.mode {
                    MergeMode::Rank => {
                        if let Some(&rank) = self.merges.get(&(left.clone(), right.clone())) {
                            let priority = MergePriority::Rank(rank);
                            let is_better = match &best {
                                None => true,
                                Some((_, _, current)) => priority > *current,
                            };
                            if is_better {
                                best = Some((li, ri, priority));
                            }
                        }
                    }
                    MergeMode::Score => {
                        let concat = format!("{}{}", left, right);
                        if let Some(id) = self.vocab.token_to_id(&concat) {
                            if let Some(score) = self.vocab.score(id) {
                                let priority = MergePriority::Score(score);
                                let is_better = match &best {
                                    None => true,
                                    Some((_, _, current)) => priority > *current,
                                };
                                if is_better {
                                    best = Some((li, ri, priority));
                                }
                            }
                        }
                    }
                }
            }

            // perform the best merge or break if none found
            match best {
                None => break,
                Some((li, ri, _)) => {
                    let merged = format!("{}{}", states[li].token, states[ri].token);
                    let id = self.vocab.token_to_id(&merged);
                    states[li].token = merged;
                    states[li].id = id;
                    states[li].gen += 1;
                    states[ri].active = false;
                    states[ri].gen += 1;
                }
            }
        }

        //collect final token IDs from active states
        states
            .iter()
            .filter(|s| s.active)
            .flat_map(|s| {
                match s.id {
                    Some(id) => vec![id],
                    None => {
                        // byte fallback — unknown token, encode each byte individually
                        s.token
                            .bytes()
                            .filter_map(|b| self.vocab.token_to_id(&(b as char).to_string()))
                            .collect()
                    }
                }
            })
            .collect()
    }

    pub fn encode(&self, pretokens: Vec<PreToken>) -> Vec<TokenID> {
        pretokens
            .into_iter()
            .flat_map(|pt| match pt {
                PreToken::Special(id) => vec![id],
                PreToken::Text(s) => self.encode_chunk(&s),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::loader::{LoadedTokenizer, Loader};
    use crate::tokenizer::pretokenize::pretokenize;
    use std::sync::Arc;

    const TOKENIZER_PATH: &str = "testdata/tokenizer.json";

    fn make_bpe() -> Bpe {
        let loader = Loader {
            file: TOKENIZER_PATH.to_string(),
        };
        match loader.load().expect("failed to load tokenizer") {
            LoadedTokenizer::HfVocab(vocab, merges) => {
                Bpe::new(Arc::new(vocab), merges, MergeMode::Rank)
            }
            _ => panic!("expected HfVocab"),
        }
    }

    fn encode_string(bpe: &Bpe, s: &str) -> Vec<TokenID> {
        let vocab = &bpe.vocab;
        let pretokens = pretokenize(s, vocab);
        bpe.encode(pretokens)
    }

    #[test]
    fn test_hello_world() {
        let bpe = make_bpe();
        assert_eq!(encode_string(&bpe, "Hello world"), vec![9906, 1917]);
    }

    #[test]
    fn test_contraction() {
        let bpe = make_bpe();
        assert_eq!(
            encode_string(&bpe, "don't you think?"),
            vec![15357, 956, 499, 1781, 30]
        );
    }

    #[test]
    fn test_pangram() {
        let bpe = make_bpe();
        assert_eq!(
            encode_string(&bpe, "The quick brown fox jumps over the lazy dog"),
            vec![791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679]
        );
    }

    #[test]
    fn test_multiple_spaces() {
        let bpe = make_bpe();
        assert_eq!(
            encode_string(&bpe, "hello     world"),
            vec![15339, 257, 1917]
        );
    }

    #[test]
    fn test_digits() {
        let bpe = make_bpe();
        assert_eq!(encode_string(&bpe, "123456789"), vec![4513, 10961, 16474]);
    }

    #[test]
    fn test_newline() {
        let bpe = make_bpe();
        assert_eq!(encode_string(&bpe, "Hello\nworld"), vec![9906, 198, 14957]);
    }

    #[test]
    fn test_empty_string() {
        let bpe = make_bpe();
        assert_eq!(encode_string(&bpe, ""), vec![] as Vec<TokenID>);
    }

    #[test]
    fn test_special_token_passthrough() {
        let bpe = make_bpe();
        let pretokens = pretokenize("<|begin_of_text|>Hello", &bpe.vocab);
        let ids = bpe.encode(pretokens);
        // first ID must be the BOS token
        assert_eq!(ids[0], 128000);
        // followed by Hello
        assert_eq!(ids[1], 9906);
    }

    #[test]
    fn test_round_trip_token_count() {
        // number of tokens must match HuggingFace exactly for all test strings
        let bpe = make_bpe();
        let cases = vec![
            ("Hello world", 2),
            ("don't you think?", 5),
            ("123456789", 3),
            ("Hello\nworld", 3),
        ];
        for (s, expected_count) in cases {
            let ids = encode_string(&bpe, s);
            assert_eq!(
                ids.len(),
                expected_count,
                "token count mismatch for {:?}: got {} expected {}",
                s,
                ids.len(),
                expected_count
            );
        }
    }
}
