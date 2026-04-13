use crate::core::backend::Backend;
use crate::inference::session::{Session, SessionId, SessionStatus};
/*
manages multiple concurrent requests. No continuous batching yet (that's a Phase 9 optimization), just a queue of sessions.
*/

pub struct Batch<B: Backend> {
    pub sessions: Vec<Session<B>>,
    max_batch_size: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    #[error("batch is full (max {0} sessions)")]
    Full(usize),
    #[error("session {0:?} not found")]
    NotFound(SessionId),
}

impl<B: Backend> Batch<B> {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            sessions: Vec::new(),
            max_batch_size,
        }
    }
    pub fn add(&mut self, session: Session<B>) -> Result<(), BatchError> {
        if self.is_full() {
            return Err(BatchError::Full(self.max_batch_size));
        }
        self.sessions.push(session);
        Ok(())
    }

    pub fn is_full(&self) -> bool {
        self.sessions.len() == self.max_batch_size
    }
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
    pub fn len(&self) -> usize {
        self.sessions.len()
    }
    pub fn active_sessions(&self) -> Vec<&Session<B>> {
        self.sessions
            .iter()
            .filter(|s| s.status == SessionStatus::Running)
            .collect()
    }

    pub fn finished_sessions(&self) -> Vec<&Session<B>> {
        self.sessions
            .iter()
            .filter(|s| s.status == SessionStatus::Finished)
            .collect()
    }

    pub fn drain_finished(&mut self) -> Vec<Session<B>> {
        let mut finished = Vec::new();
        let mut i = 0;

        while i < self.sessions.len() {
            if self.sessions[i].is_finished() || self.sessions[i].status == SessionStatus::Failed {
                finished.push(self.sessions.remove(i));
            } else {
                i += 1;
            }
        }
        finished
    }
    pub fn session_mut(&mut self, id: SessionId) -> Option<&mut Session<B>> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::inference::session::{Session, SessionId, SessionStatus};

    fn make_session(id: u64, max_new: usize) -> Session<CandleBackend> {
        Session::new(SessionId(id), vec![1u32, 2, 3], max_new, Some(99))
    }

    #[test]
    fn test_new_batch_is_empty() {
        let batch = Batch::<CandleBackend>::new(4);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_add_session() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_add_up_to_max() {
        let mut batch = Batch::<CandleBackend>::new(3);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        batch.add(make_session(3, 10)).unwrap();
        assert!(batch.is_full());
    }

    #[test]
    fn test_add_over_max_fails() {
        let mut batch = Batch::<CandleBackend>::new(2);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        let result = batch.add(make_session(3, 10));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BatchError::Full(2)));
    }

    #[test]
    fn test_is_full_false_when_not_full() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        assert!(!batch.is_full());
    }

    #[test]
    fn test_active_sessions_empty_when_all_waiting() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        // status starts as Waiting not Running
        assert_eq!(batch.active_sessions().len(), 0);
    }

    #[test]
    fn test_active_sessions_counts_running() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        batch.session_mut(SessionId(1)).unwrap().status = SessionStatus::Running;
        assert_eq!(batch.active_sessions().len(), 1);
    }

    #[test]
    fn test_finished_sessions_empty_initially() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        assert_eq!(batch.finished_sessions().len(), 0);
    }

    #[test]
    fn test_finished_sessions_counts_finished() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        batch.session_mut(SessionId(1)).unwrap().status = SessionStatus::Finished;
        assert_eq!(batch.finished_sessions().len(), 1);
    }

    #[test]
    fn test_drain_finished_removes_finished() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        batch.session_mut(SessionId(1)).unwrap().status = SessionStatus::Finished;
        let drained = batch.drain_finished();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].id, SessionId(1));
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_drain_finished_removes_failed() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        batch.session_mut(SessionId(1)).unwrap().status = SessionStatus::Failed;
        let drained = batch.drain_finished();
        assert_eq!(drained.len(), 1);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_drain_finished_keeps_running() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        batch.session_mut(SessionId(1)).unwrap().status = SessionStatus::Running;
        batch.session_mut(SessionId(2)).unwrap().status = SessionStatus::Finished;
        let drained = batch.drain_finished();
        assert_eq!(drained.len(), 1);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.sessions[0].id, SessionId(1));
    }

    #[test]
    fn test_session_mut_found() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        let s = batch.session_mut(SessionId(1));
        assert!(s.is_some());
    }

    #[test]
    fn test_session_mut_not_found() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        let s = batch.session_mut(SessionId(99));
        assert!(s.is_none());
    }

    #[test]
    fn test_drain_empty_batch() {
        let mut batch = Batch::<CandleBackend>::new(4);
        let drained = batch.drain_finished();
        assert!(drained.is_empty());
    }

    #[test]
    fn test_len_after_drain() {
        let mut batch = Batch::<CandleBackend>::new(4);
        batch.add(make_session(1, 10)).unwrap();
        batch.add(make_session(2, 10)).unwrap();
        batch.add(make_session(3, 10)).unwrap();
        batch.session_mut(SessionId(2)).unwrap().status = SessionStatus::Finished;
        batch.drain_finished();
        assert_eq!(batch.len(), 2);
    }
}
