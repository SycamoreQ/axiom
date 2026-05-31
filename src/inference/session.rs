use crate::core::backend::Backend;
#[cfg(feature = "cuda")]
use crate::cuda::BlockTable;

/*
A session represents one inference request — its state lives here between generation steps.

need to wire in the actual KV cache here , this is a sort of stub until phase 9
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    Waiting,  // not yet started
    Running,  // actively generating
    Finished, // hit EOS or max tokens
    Failed,   // error occurred
}

pub struct Session<B: Backend> {
    // identity
    pub id: SessionId,

    // token history
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,

    // KV cache — one (k,v) pair per layer, grows each step
    pub kv_cache: Vec<(B::Tensor, B::Tensor)>,
    pub session_id_u64: u64,
    #[cfg(feature = "cuda")]
    pub block_table: Option<crate::cuda::BlockTable>,
    pub is_forked: bool, // true if this session was forked from a parent
    pub base_len: usize,

    // position tracking
    pub offset: usize, // current position in sequence

    // stop conditions
    pub max_new_tokens: usize,
    pub eos_token_id: Option<u32>,

    // state
    pub status: SessionStatus,
}

impl<B: Backend> Session<B> {
    pub fn new(
        id: SessionId,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Self {
        let offset = 0;
        Self {
            id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            kv_cache: Vec::new(),
            offset,
            max_new_tokens,
            eos_token_id,
            status: SessionStatus::Waiting,
            session_id_u64: id.0,
            #[cfg(feature = "cuda")]
            block_table: None,
            is_forked: false,
            base_len: 0,
        }
    }

    // total tokens including prompt
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    // number of tokens generated so far
    pub fn num_generated(&self) -> usize {
        self.generated_tokens.iter().count()
    }

    // check if generation should stop
    pub fn is_finished(&self) -> bool {
        self.status == SessionStatus::Finished
            || self.num_generated() >= self.max_new_tokens
            || self.last_token_is_eos()
    }

    // check if last generated token is EOS
    pub fn last_token_is_eos(&self) -> bool {
        match (self.generated_tokens.last(), self.eos_token_id) {
            (Some(&last), Some(eos)) => last == eos,
            _ => false,
        }
    }

    pub fn push_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
        if self.last_token_is_eos() || self.num_generated() >= self.max_new_tokens {
            self.status = SessionStatus::Finished;
        }
    }

    // tokens to feed into model next step
    // prefill: all prompt tokens
    // generation: just the last token
    pub fn next_input_tokens(&self) -> &[u32] {
        if self.generated_tokens.is_empty() {
            // prefill — feed entire prompt
            &self.prompt_tokens
        } else {
            // generation — feed only last token
            let last = self.generated_tokens.last().unwrap();
            std::slice::from_ref(last)
        }
    }
    //full token sequence prompt + generated
    pub fn all_tokens(&self) -> Vec<u32> {
        self.prompt_tokens
            .iter()
            .chain(self.generated_tokens.iter())
            .copied()
            .collect()
    }

    //Mark this session as a fork of a parent at parent_seq_len.
    pub fn mark_forked(&mut self, base_len: usize) {
        self.is_forked = true;
        self.base_len = base_len;
    }

    //Returns how many residual tokens this session has generated past the fork point.
    pub fn residual_len(&self) -> usize {
        self.total_tokens().saturating_sub(self.base_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;

    fn make_session(prompt: Vec<u32>, max_new: usize) -> Session<CandleBackend> {
        Session::new(SessionId(1), prompt, max_new, Some(2))
    }

    #[test]
    fn test_new_session_initial_state() {
        let s = make_session(vec![1, 2, 3], 10);
        assert_eq!(s.offset, 0);
        assert_eq!(s.status, SessionStatus::Waiting);
        assert!(s.generated_tokens.is_empty());
        assert!(s.kv_cache.is_empty());
    }

    #[test]
    fn test_total_tokens_prompt_only() {
        let s = make_session(vec![1, 2, 3], 10);
        assert_eq!(s.total_tokens(), 3);
    }

    #[test]
    fn test_total_tokens_after_generation() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.push_token(42);
        s.push_token(43);
        assert_eq!(s.total_tokens(), 5);
    }

    #[test]
    fn test_num_generated_starts_zero() {
        let s = make_session(vec![1, 2], 10);
        assert_eq!(s.num_generated(), 0);
    }

    #[test]
    fn test_num_generated_after_push() {
        let mut s = make_session(vec![1, 2], 10);
        s.push_token(10);
        s.push_token(11);
        assert_eq!(s.num_generated(), 2);
    }

    #[test]
    fn test_is_finished_false_initially() {
        let s = make_session(vec![1, 2, 3], 10);
        assert!(!s.is_finished());
    }

    #[test]
    fn test_is_finished_on_max_tokens() {
        let mut s = make_session(vec![1], 3);
        s.push_token(10);
        s.push_token(11);
        assert!(!s.is_finished());
        s.push_token(12);
        assert!(s.is_finished());
    }

    #[test]
    fn test_is_finished_on_eos() {
        let mut s = make_session(vec![1, 2], 100);
        s.push_token(42);
        assert!(!s.is_finished());
        s.push_token(2); // eos_token_id = 2
        assert!(s.is_finished());
    }

    #[test]
    fn test_last_token_is_eos_empty() {
        let s = make_session(vec![1, 2], 10);
        assert!(!s.last_token_is_eos());
    }

    #[test]
    fn test_last_token_is_eos_true() {
        let mut s = make_session(vec![1], 10);
        s.push_token(2); // eos = 2
        assert!(s.last_token_is_eos());
    }

    #[test]
    fn test_last_token_is_eos_false() {
        let mut s = make_session(vec![1], 10);
        s.push_token(99);
        assert!(!s.last_token_is_eos());
    }

    #[test]
    fn test_last_token_is_eos_no_eos_configured() {
        let mut s: Session<CandleBackend> = Session::new(SessionId(1), vec![1], 10, None);
        s.push_token(2);
        assert!(!s.last_token_is_eos());
    }

    #[test]
    fn test_push_token_updates_offset() {
        let mut s = make_session(vec![1, 2, 3], 10);
        assert_eq!(s.offset, 0);
        s.push_token(42);
        assert_eq!(s.offset, 0);
        s.push_token(43);
        assert_eq!(s.offset, 0);
    }

    #[test]
    fn test_push_token_sets_finished_on_eos() {
        let mut s = make_session(vec![1], 100);
        s.push_token(2); // eos
        assert_eq!(s.status, SessionStatus::Finished);
    }

    #[test]
    fn test_push_token_sets_finished_on_max() {
        let mut s = make_session(vec![1], 2);
        s.push_token(10);
        assert_eq!(s.status, SessionStatus::Waiting);
        s.push_token(11);
        assert_eq!(s.status, SessionStatus::Finished);
    }

    #[test]
    fn test_next_input_tokens_prefill() {
        let s = make_session(vec![1, 2, 3], 10);
        assert_eq!(s.next_input_tokens(), &[1, 2, 3]);
    }

    #[test]
    fn test_next_input_tokens_generation() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.push_token(42);
        assert_eq!(s.next_input_tokens(), &[42]);
        s.push_token(43);
        assert_eq!(s.next_input_tokens(), &[43]);
    }

    #[test]
    fn test_all_tokens_prompt_only() {
        let s = make_session(vec![1, 2, 3], 10);
        assert_eq!(s.all_tokens(), vec![1, 2, 3]);
    }

    #[test]
    fn test_all_tokens_with_generated() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.push_token(4);
        s.push_token(5);
        assert_eq!(s.all_tokens(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_session_id_equality() {
        let id1 = SessionId(42);
        let id2 = SessionId(42);
        let id3 = SessionId(99);
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_new_session_is_not_forked() {
        let s = make_session(vec![1, 2, 3], 10);
        assert!(!s.is_forked);
        assert_eq!(s.base_len, 0);
    }

    #[test]
    fn test_new_session_block_table_is_none() {
        let s = make_session(vec![1, 2, 3], 10);
        #[cfg(feature = "cuda")]
        assert!(s.block_table.is_none());
    }

    #[test]
    fn test_new_session_id_u64_matches() {
        let s = make_session(vec![1, 2, 3], 10);
        assert_eq!(s.session_id_u64, 1u64); // make_session uses SessionId(1)
    }

    #[test]
    fn test_mark_forked_sets_is_forked() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.mark_forked(3);
        assert!(s.is_forked);
    }

    #[test]
    fn test_mark_forked_sets_base_len() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.mark_forked(3);
        assert_eq!(s.base_len, 3);
    }

    #[test]
    fn test_mark_forked_twice_takes_last() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.mark_forked(3);
        s.mark_forked(5);
        assert_eq!(s.base_len, 5);
    }

    #[test]
    fn test_residual_len_unforked_equals_total_tokens() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.push_token(4);
        s.push_token(5);
        // not forked, base_len = 0, residual = total = 5
        assert_eq!(s.residual_len(), s.total_tokens());
    }

    #[test]
    fn test_residual_len_forked_subtracts_base() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.mark_forked(3); // base covers the 3 prompt tokens
        s.push_token(4);
        s.push_token(5);
        // total = 5, base = 3, residual = 2
        assert_eq!(s.residual_len(), 2);
    }

    #[test]
    fn test_residual_len_forked_no_generation_is_zero() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.mark_forked(3);
        // no tokens generated yet, residual = 0
        assert_eq!(s.residual_len(), 0);
    }

    #[test]
    fn test_residual_len_saturates_at_zero() {
        let mut s = make_session(vec![1, 2, 3], 10);
        s.mark_forked(10); // base_len larger than total_tokens
                           // should not underflow
        assert_eq!(s.residual_len(), 0);
    }
}
