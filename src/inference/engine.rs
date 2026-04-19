use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::inference::batch::Batch;
use crate::inference::generator::Generator;
use crate::inference::sampler::{Sampler, SamplerConfig};
use crate::inference::session::{Session, SessionId, SessionStatus};
use crate::kv_cache::manager::KVManager;
use crate::model::model::LlamaModel;
use crate::tokenizer::tokenizer::{EncodeOptions, Tokenizer};

/*
 top-level public API for the inference layer. Everything above wires together here.
*/

pub struct Engine<B: Backend> {
    generator: Generator<B>,
    batch: Batch<B>,
    next_session_id: u64,
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
        let next_session_id = 0;

        Self {
            generator: generator,
            batch: batch,
            next_session_id: next_session_id,
        }
    }

    // submit a new request, returns session id
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
        self.batch.add(session)?;
        Ok(id)
    }

    // submit from raw text
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
        let eos_id = self.generator.tokenizer().eos_id().map(|id| id as u32);
        self.submit(prompt_ids, max_new_tokens, eos_id)
    }

    // run one full generation step across all active sessions
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

    // run until all sessions finish
    pub fn run_to_completion(&mut self) -> Result<(), EngineError> {
        loop {
            let active_ids: Vec<SessionId> =
                self.batch.active_sessions().iter().map(|s| s.id).collect();
            if active_ids.is_empty() {
                break;
            }
            for id in active_ids {
                let session = self.batch.session_mut(id).unwrap();
                self.generator.step(session)?;
            }
        }
        Ok(())
    }

    // get generated tokens for a session
    pub fn get_output(&self, id: SessionId) -> Option<&[u32]> {
        self.batch
            .sessions
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.generated_tokens.as_slice())
    }

    // decode output to string
    pub fn decode_output(&self, id: SessionId) -> Option<String> {
        let tokens = self.get_output(id)?;
        let token_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        Some(self.generator.tokenizer().decode(&token_ids))
    }

    // drain finished sessions and return their outputs
    pub fn drain_finished(&mut self) -> Vec<(SessionId, Vec<u32>)> {
        self.batch
            .drain_finished()
            .into_iter()
            .map(|s| (s.id, s.generated_tokens))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::inference::sampler::SamplerConfig;
    use crate::model::config::ModelConfig;
    use crate::tokenizer::tokenizer::Tokenizer;

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
        let model = crate::model::model::LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
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
        let id = engine.submit(vec![1u32, 2, 3], 5, None).unwrap();
        assert_eq!(id, SessionId(0));
        assert_eq!(engine.next_session_id, 1);
    }

    #[test]
    fn test_submit_increments_id() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        let id1 = engine.submit(vec![1u32], 5, None).unwrap();
        let id2 = engine.submit(vec![2u32], 5, None).unwrap();
        assert_eq!(id1, SessionId(0));
        assert_eq!(id2, SessionId(1));
    }

    #[test]
    fn test_submit_adds_to_batch() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        engine.submit(vec![1u32, 2], 5, None).unwrap();
        assert_eq!(engine.batch.len(), 1);
    }

    #[test]
    fn test_submit_over_capacity_fails() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let config = make_config(1000);
        let model = crate::model::model::LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let mut engine = Engine::new(model, tokenizer, SamplerConfig::default(), 2, cpu());
        engine.submit(vec![1u32], 5, None).unwrap();
        engine.submit(vec![2u32], 5, None).unwrap();
        assert!(engine.submit(vec![3u32], 5, None).is_err());
    }

    #[test]
    fn test_step_produces_token() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        engine.submit(vec![1u32, 2, 3], 5, None).unwrap();
        let results = engine.step().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, SessionId(0));
        assert!((results[0].1 as usize) < 1000);
    }

    #[test]
    fn test_get_output_after_step() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        let id = engine.submit(vec![1u32, 2, 3], 5, None).unwrap();
        engine.step().unwrap();
        let output = engine.get_output(id);
        assert!(output.is_some());
        assert_eq!(output.unwrap().len(), 1);
    }

    #[test]
    fn test_get_output_not_found() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let engine = make_engine(1000);
        assert!(engine.get_output(SessionId(99)).is_none());
    }

    #[test]
    fn test_run_to_completion() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        // eos=0, greedy with zero weights picks 0 → stops immediately
        engine.submit(vec![1u32, 2], 10, Some(0)).unwrap();
        engine.run_to_completion().unwrap();
        let finished = engine.batch.finished_sessions();
        assert_eq!(finished.len(), 1);
    }

    #[test]
    fn test_drain_finished() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let mut engine = make_engine(1000);
        engine.submit(vec![1u32], 3, Some(0)).unwrap();
        engine.run_to_completion().unwrap();
        let drained = engine.drain_finished();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, SessionId(0));
        assert!(engine.batch.is_empty());
    }

    #[test]
    fn test_submit_text() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let tok = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let vocab_size = tok.vocab().size();
        let config = make_config(vocab_size);
        let model = crate::model::model::LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut engine = Engine::new(model, tok, SamplerConfig::default(), 4, cpu());
        let id = engine.submit_text("Hello", 5, EncodeOptions::default());
        assert!(id.is_ok());
    }

    #[test]
    fn test_decode_output() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let tok = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let vocab_size = tok.vocab().size();
        let config = make_config(vocab_size);
        let model = crate::model::model::LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut engine = Engine::new(
            model,
            tok,
            SamplerConfig {
                temperature: 0.0,
                ..Default::default()
            },
            4,
            cpu(),
        );
        let id = engine.submit(vec![1u32, 2], 3, Some(0)).unwrap();
        engine.run_to_completion().unwrap();
        let decoded = engine.decode_output(id);
        assert!(decoded.is_some());
    }
}
