use std::collections::*;

/*
The reference scatters vocab concerns across three separate HashMap fields on the tokenizer struct.
In Axiom, Vocab is a first-class type that owns all of this and exposes a clean interface.
Everything downstream — BPE, the loader, the tokenizer — talks to Vocab through methods, never touches its fields directly.
*/

pub type TokenID = usize;

pub struct Vocab {
    tokens: Vec<String>,
    token_ids: HashMap<String, TokenID>,
    scores: Option<Vec<f32>>,
    special_tokens: HashMap<String, TokenID>,
    special_token_strings: Vec<String>, // sorted longest-first
    special_token_ids: HashSet<TokenID>,
    bos_id: Option<TokenID>,
    eos_id: Option<TokenID>,
    pad_id: Option<TokenID>,
    unk_id: Option<TokenID>,
}

impl Vocab {
    pub fn new(
        tokens: Vec<String>,
        scores: Option<Vec<f32>>,
        special_tokens: Vec<(String, TokenID)>,
        bos_id: Option<TokenID>,
        eos_id: Option<TokenID>,
        pad_id: Option<TokenID>,
        unk_id: Option<TokenID>,
    ) -> Self {
        let token_ids: HashMap<String, TokenID> = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        let special_tokens_map: HashMap<String, TokenID> = special_tokens //
            .into_iter()
            .collect();

        let mut special_token_strings: Vec<String> = special_tokens_map.keys().cloned().collect();
        special_token_strings.sort_by_key(|k| std::cmp::Reverse(k.len()));

        let special_token_ids: HashSet<TokenID> = special_tokens_map.values().copied().collect();

        if let Some(ref s) = scores {
            assert_eq!(s.len(), tokens.len(), "scores and tokens length mismatch");
        }

        Self {
            tokens,
            token_ids,
            scores,
            special_tokens: special_tokens_map,
            special_token_strings,
            special_token_ids,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
        }
    }

    pub fn find_special_prefix(&self, text: &str) -> Option<(&str, TokenID)> {
        for s in &self.special_token_strings {
            if text.starts_with(s.as_str()) {
                let id = self.special_tokens[s];
                return Some((s.as_str(), id));
            }
        }
        None
    }

    pub fn token_to_id(&self, token: &str) -> Option<TokenID> {
        if let Some(&id) = self.special_tokens.get(token) {
            return Some(id);
        }
        self.token_ids.get(token).copied()
    }

    pub fn id_to_token(&self, id: TokenID) -> Option<&str> {
        self.tokens.get(id).map(|s| s.as_str())
    }

    pub fn score(&self, id: TokenID) -> Option<f32> {
        self.scores.as_ref()?.get(id).copied()
    }

    pub fn is_special(&self, id: TokenID) -> bool {
        self.special_token_ids.contains(&id)
    }

    pub fn bos_id(&self) -> Option<TokenID> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<TokenID> {
        self.eos_id
    }
    pub fn pad_id(&self) -> Option<TokenID> {
        self.pad_id
    }

    pub fn unk_id(&self) -> Option<TokenID> {
        self.unk_id
    }

    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    pub fn special_token_strings(&self) -> &[String] {
        &self.special_token_strings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vocab() -> Vocab {
        let tokens = vec![
            "<unk>".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "▁hello".to_string(),
            "▁world".to_string(),
            "hell".to_string(),
        ];
        let scores = Some(vec![0.0, -1.0, -2.0, -0.5, -1.5, -0.8]);
        let special_tokens = vec![
            ("<|bos|>".to_string(), 100),
            ("<|eos|>".to_string(), 101),
            ("<|pad|>".to_string(), 102),
            ("<|end_of_text|>".to_string(), 103),
            ("<|end|>".to_string(), 104),
        ];
        Vocab::new(
            tokens,
            scores,
            special_tokens,
            Some(100),
            Some(101),
            Some(102),
            Some(0),
        )
    }

    #[test]
    fn test_size() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.size(), 6);
    }

    #[test]
    fn test_forward_lookup_regular() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.token_to_id("hello"), Some(1));
        assert_eq!(vocab.token_to_id("world"), Some(2));
        assert_eq!(vocab.token_to_id("▁hello"), Some(3));
    }

    #[test]
    fn test_forward_lookup_missing() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.token_to_id("notinvocab"), None);
    }

    #[test]
    fn test_reverse_lookup() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.id_to_token(1), Some("hello"));
        assert_eq!(vocab.id_to_token(3), Some("▁hello"));
    }

    #[test]
    fn test_reverse_lookup_out_of_bounds() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.id_to_token(9999), None);
    }

    #[test]
    fn test_special_token_priority() {
        // "hello" exists in both main vocab (id=1) and we add it as special (id=99)
        let tokens = vec!["hello".to_string(), "world".to_string()];
        let special_tokens = vec![("hello".to_string(), 99)];
        let vocab = Vocab::new(tokens, None, special_tokens, None, None, None, None);
        // special must win
        assert_eq!(vocab.token_to_id("hello"), Some(99));
    }

    #[test]
    fn test_special_lookup_direct() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.token_to_id("<|bos|>"), Some(100));
        assert_eq!(vocab.token_to_id("<|eos|>"), Some(101));
        assert_eq!(vocab.token_to_id("<|pad|>"), Some(102));
    }

    #[test]
    fn test_is_special_true() {
        let vocab = make_test_vocab();
        assert!(vocab.is_special(100));
        assert!(vocab.is_special(101));
        assert!(vocab.is_special(103));
    }

    #[test]
    fn test_is_special_false() {
        let vocab = make_test_vocab();
        assert!(!vocab.is_special(1));
        assert!(!vocab.is_special(2));
    }

    #[test]
    fn test_score_present() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.score(1), Some(-1.0));
        assert_eq!(vocab.score(0), Some(0.0));
    }

    #[test]
    fn test_score_out_of_bounds() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.score(9999), None);
    }

    #[test]
    fn test_score_absent() {
        let tokens = vec!["hello".to_string()];
        let vocab = Vocab::new(tokens, None, vec![], None, None, None, None);
        assert_eq!(vocab.score(0), None);
    }

    #[test]
    fn test_sentinel_ids() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.bos_id(), Some(100));
        assert_eq!(vocab.eos_id(), Some(101));
        assert_eq!(vocab.pad_id(), Some(102));
        assert_eq!(vocab.unk_id(), Some(0));
    }

    #[test]
    fn test_sentinel_ids_absent() {
        let tokens = vec!["hello".to_string()];
        let vocab = Vocab::new(tokens, None, vec![], None, None, None, None);
        assert_eq!(vocab.bos_id(), None);
        assert_eq!(vocab.eos_id(), None);
        assert_eq!(vocab.pad_id(), None);
        assert_eq!(vocab.unk_id(), None);
    }

    #[test]
    fn test_find_special_prefix_longest_first() {
        let vocab = make_test_vocab();
        // "<|end_of_text|>" and "<|end|>" both match — longer must win
        let result = vocab.find_special_prefix("<|end_of_text|> some text");
        assert_eq!(result, Some(("<|end_of_text|>", 103)));
    }

    #[test]
    fn test_find_special_prefix_short_match() {
        let vocab = make_test_vocab();
        // only "<|eos|>" matches here
        let result = vocab.find_special_prefix("<|eos|> continuing text");
        assert_eq!(result, Some(("<|eos|>", 101)));
    }

    #[test]
    fn test_find_special_prefix_no_match() {
        let vocab = make_test_vocab();
        assert_eq!(vocab.find_special_prefix("plain text no special"), None);
    }

    #[test]
    fn test_special_token_strings_sorted() {
        let vocab = make_test_vocab();
        let strings = vocab.special_token_strings();
        // verify sorted longest to shortest
        let lengths: Vec<usize> = strings.iter().map(|s| s.len()).collect();
        let mut sorted = lengths.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(lengths, sorted);
    }
}
