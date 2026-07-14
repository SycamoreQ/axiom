use crate::core::backend::Backend;
use crate::core::device::Device;
#[cfg(feature = "cuda")]
use crate::cuda::{fork_manager::ForkManager, CudaContext};
use crate::inference::batch::Batch;
use crate::inference::generator::Generator;
use crate::inference::sampler::{Sampler, SamplerConfig};
use crate::inference::session::{Session, SessionId, SessionStatus};
use crate::model::model::LlamaModel;
use crate::tokenizer::tokenizer::{EncodeOptions, Tokenizer};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

/*
 top-level public API for the inference layer.
*/

pub struct Engine<B: Backend> {
    generator: Generator<B>,
    pub batch: Batch<B>,
    next_session_id: u64,
    #[cfg(feature = "cuda")]
    pub fork_manager: Option<Arc<Mutex<ForkManager<B>>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("batch error: {0}")]
    Batch(#[from] crate::inference::batch::BatchError),
    #[error("generator error: {0}")]
    Generator(#[from] crate::inference::generator::GeneratorError),
    #[error("session {0:?} not found")]
    SessionNotFound(SessionId),
}

impl<B: Backend> Engine<B> {
    pub fn new(
        model: LlamaModel<B>,
        tokenizer: Tokenizer,
        sampler_config: SamplerConfig,
        max_batch_size: usize,
        device: Device,
    ) -> Self {
        let sampler = Sampler::new(sampler_config);
        let generator = Generator::new(model, tokenizer, sampler, device);
        let batch = Batch::new(max_batch_size);

        Self {
            generator,
            batch,
            next_session_id: 0,
            #[cfg(feature = "cuda")]
            fork_manager: None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn with_fork_manager(mut self, fork_manager: Arc<Mutex<ForkManager<B>>>) -> Self {
        self.fork_manager = Some(fork_manager);
        self
    }

    pub fn submit(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<SessionId, EngineError> {
        let id = SessionId(self.next_session_id);
        self.next_session_id += 1;

        let mut session = Session::new(id, prompt_tokens, max_new_tokens, eos_token_id);
        session.status = SessionStatus::Running;
        session.session_id_u64 = id.0;

        #[cfg(feature = "cuda")]
        if let Some(fm) = &self.fork_manager {
            fm.lock().unwrap().free_session(id.0);
        }

        self.batch.add(session)?;
        Ok(id)
    }

    pub fn submit_text(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        opts: EncodeOptions,
    ) -> Result<SessionId, EngineError> {
        let prompt_ids: Vec<u32> = self
            .generator
            .tokenizer()
            .encode(prompt, opts)
            .iter()
            .map(|&id| id as u32)
            .collect();

        eprintln!("full prompt tokens: {:?}", prompt_ids);
        for &id in &prompt_ids {
            eprintln!(
                "  {} -> {:?}",
                id,
                self.generator.tokenizer().decode(&[id as usize])
            );
        }

        let eos_id = self.generator.tokenizer().eos_id().map(|id| id as u32);
        self.submit(prompt_ids, max_new_tokens, eos_id)
    }

    pub fn step(&mut self) -> Result<Vec<(SessionId, u32)>, EngineError> {
        let active_ids: Vec<SessionId> =
            self.batch.active_sessions().iter().map(|s| s.id).collect();
        let mut results = Vec::new();

        for id in active_ids {
            let session = self.batch.session_mut(id).unwrap();
            let token = self.generator.step(session)?;
            results.push((id, token));
        }
        Ok(results)
    }

    pub fn run_to_completion(&mut self) -> Result<(), EngineError> {
        while !self.batch.active_sessions().is_empty() {
            self.step()?;
        }
        Ok(())
    }

    pub fn get_output(&self, id: SessionId) -> Option<&[u32]> {
        self.batch
            .sessions
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.generated_tokens.as_slice())
    }

    pub fn decode_output(&self, id: SessionId) -> Option<String> {
        let tokens = self.get_output(id)?;
        let token_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        Some(self.generator.tokenizer().decode(&token_ids))
    }

    pub fn drain_finished(&mut self) -> Vec<(SessionId, Vec<u32>)> {
        let finished = self.batch.drain_finished();
        let mut results = Vec::new();

        for s in finished {
            #[cfg(feature = "cuda")]
            if let Some(fm) = &self.fork_manager {
                fm.lock().unwrap().free_session(s.id.0);
            }
            results.push((s.id, s.generated_tokens));
        }
        results
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.generator.tokenizer()
    }

    #[cfg(feature = "cuda")]
    pub fn fork_session(
        &mut self,
        parent_id: SessionId,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        _ctx: &CudaContext,
    ) -> Result<SessionId, EngineError> {
        let parent_tokens = self
            .batch
            .sessions
            .iter()
            .find(|s| s.id == parent_id)
            .ok_or(EngineError::SessionNotFound(parent_id))?
            .all_tokens();

        let child_id = SessionId(self.next_session_id);
        self.next_session_id += 1;

        let mut child_session = Session::new(
            child_id,
            parent_tokens.clone(),
            max_new_tokens,
            eos_token_id,
        );
        child_session.session_id_u64 = child_id.0;
        child_session.status = SessionStatus::Running;

        if let Some(fm) = &self.fork_manager {
            fm.lock()
                .unwrap()
                .fork_session(parent_id.0, child_id.0, _ctx);
        }

        let base_len = child_session.prompt_tokens.len();
        child_session.mark_forked(base_len);
        self.batch.add(child_session)?;
        Ok(child_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::model::config::ModelConfig;

    fn cpu() -> Device {
        Device::Cpu
    }

    const TOKENIZER_PATH: &str = "testdata/tokenizer.json";

    fn make_config(vocab_size: usize) -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
            rope_freqs: None,
            rope_scaling: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            num_shared_experts: None,
            expert_interval: None,
            prefetch_threshold: None,
            torch_dtype: "float32".to_string(),
            architectures: None,
            model_type: Some("llama".to_string()),
        }
    }

    fn make_engine(vocab_size: usize) -> Engine<CandleBackend> {
        let config = make_config(vocab_size);
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let sampler_config = SamplerConfig {
            temperature: 0.0,
            seed: Some(42),
            max_new_tokens: 10,
            ..Default::default()
        };
        Engine::new(model, tokenizer, sampler_config, 4, cpu())
    }

    #[test]
    fn test_engine_construction() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let engine = make_engine(1000);
        assert_eq!(engine.next_session_id, 0);
        assert!(engine.batch.is_empty());
    }

    #[test]
    fn test_submit_returns_session_id() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        let id = engine.submit(vec![1, 2, 3], 5, None).unwrap();
        assert_eq!(id, SessionId(0));
        assert_eq!(engine.next_session_id, 1);
    }

    #[test]
    fn test_submit_increments_id() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        let id1 = engine.submit(vec![1], 5, None).unwrap();
        let id2 = engine.submit(vec![2], 5, None).unwrap();
        assert_eq!(id1, SessionId(0));
        assert_eq!(id2, SessionId(1));
    }

    #[test]
    fn test_submit_adds_to_batch() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        engine.submit(vec![1, 2], 5, None).unwrap();
        assert_eq!(engine.batch.len(), 1);
    }

    #[test]
    fn test_submit_over_capacity_fails() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        for _ in 0..4 {
            engine.submit(vec![1], 5, None).unwrap();
        }
        assert!(engine.submit(vec![3], 5, None).is_err());
    }

    #[test]
    fn test_step_produces_token() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        engine.submit(vec![1, 2, 3], 5, None).unwrap();
        let results = engine.step().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, SessionId(0));
    }

    #[test]
    fn test_get_output_after_step() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        let id = engine.submit(vec![1, 2, 3], 5, None).unwrap();
        engine.step().unwrap();
        let output = engine.get_output(id);
        assert!(output.is_some());
        assert_eq!(output.unwrap().len(), 1);
    }

    #[test]
    fn test_run_to_completion() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        engine.submit(vec![1, 2], 10, Some(0)).unwrap();
        engine.run_to_completion().unwrap();
        assert_eq!(engine.batch.finished_sessions().len(), 1);
    }

    #[test]
    fn test_drain_finished() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        engine.submit(vec![1], 3, Some(0)).unwrap();
        engine.run_to_completion().unwrap();
        let drained = engine.drain_finished();
        assert_eq!(drained.len(), 1);
        assert!(engine.batch.is_empty());
    }

    #[test]
    fn test_engine_session_id_u64_set_on_submit() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        let id = engine.submit(vec![1, 2, 3], 5, None).unwrap();
        let session = engine.batch.sessions.iter().find(|s| s.id == id).unwrap();
        assert_eq!(session.session_id_u64, 0u64);
    }

    #[test]
    fn test_submit_session_not_forked_by_default() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        let id = engine.submit(vec![1, 2, 3], 5, None).unwrap();
        let session = engine.batch.sessions.iter().find(|s| s.id == id).unwrap();
        assert!(!session.is_forked);
        assert_eq!(session.base_len, 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fork_session_creates_child() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        // Attempt to load PTX and initialize CUDA context
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok();
        let Some(ptx_str) = ptx else { return };
        let ctx_res = crate::cuda::CudaContext::new(0, &ptx_str);
        let Ok(ctx) = ctx_res else { return };

        let alloc = crate::cuda::PagedBlockAllocator::new(&ctx, 32, 4, 2, 32).unwrap();
        let alloc = Arc::new(Mutex::new(alloc));
        let fm = Arc::new(Mutex::new(ForkManager::<CandleBackend>::new(
            Arc::clone(&alloc),
            100,
            400,
        )));

        let mut engine = make_engine(1000).with_fork_manager(fm);
        let parent_id = engine.submit(vec![1, 2, 3], 10, None).unwrap();
        let child_id = engine.fork_session(parent_id, 10, None, &ctx).unwrap();

        assert_ne!(parent_id, child_id);
        let child = engine
            .batch
            .sessions
            .iter()
            .find(|s| s.id == child_id)
            .unwrap();
        assert!(child.is_forked);
        assert_eq!(child.base_len, 3);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fork_session_unknown_parent_returns_err() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let ptx = std::fs::read_to_string(env!("AXIOM_KERNELS_PTX")).ok();
        let Some(ptx_str) = ptx else { return };
        let ctx_res = crate::cuda::CudaContext::new(0, &ptx_str);
        let Ok(ctx) = ctx_res else { return };

        let alloc = crate::cuda::PagedBlockAllocator::new(&ctx, 32, 4, 2, 32).unwrap();
        let alloc = Arc::new(Mutex::new(alloc));
        let fm = Arc::new(Mutex::new(ForkManager::<CandleBackend>::new(
            Arc::clone(&alloc),
            100,
            400,
        )));

        let mut engine = make_engine(1000).with_fork_manager(fm);
        let result = engine.fork_session(SessionId(99), 10, None, &ctx);
        assert!(result.is_err());
    }
}
