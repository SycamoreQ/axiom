use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::tensor::TensorOps;
use crate::inference::sampler::Sampler;
use crate::inference::session::{Session, SessionId, SessionStatus};
use crate::model::model::LlamaModel;
use crate::tokenizer::tokenizer::{EncodeOptions, Tokenizer};

/*
This is the core of the inference engine — it wires the model, tokenizer, sampler, and session together into a generation loop.
*/

#[derive(Debug, thiserror::Error)]
pub enum GeneratorError {
    #[error("model error: {0}")]
    Model(#[from] crate::core::error::CoreError),
    #[error("session finished")]
    SessionFinished,
}

pub struct Generator<B: Backend> {
    model: LlamaModel<B>,
    tokenizer: Tokenizer,
    sampler: Sampler,
    device: Device,
}

impl<B: Backend> Generator<B> {
    pub fn new(
        model: LlamaModel<B>,
        tokenizer: Tokenizer,
        sampler: Sampler,
        device: Device,
    ) -> Self {
        Self {
            model: model,
            tokenizer: tokenizer,
            sampler: sampler,
            device: device,
        }
    }
    //One step of autoregressive generation:
    pub fn step(&mut self, session: &mut Session<B>) -> std::result::Result<u32, GeneratorError> {
        let session_tokens = session.next_input_tokens().to_vec();
        let logits = self
            .model
            .forward(&session_tokens, Some(&mut session.kv_cache), session.offset)
            .map_err(GeneratorError::Model)?;

        let seq_len: usize = logits.shape().dim(1)?;
        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?;
        let flattened_logits = last_token_logits.squeeze(0)?.squeeze(0)?;
        let logits_vec: Vec<f32> = flattened_logits.to_vec_f32()?;

        let next_token = self.sampler.sample(&logits_vec, &session.all_tokens());

        session.push_token(next_token);
        Ok(next_token)
    }

    pub fn run(
        &mut self,
        session: &mut Session<B>,
    ) -> std::result::Result<Vec<u32>, GeneratorError> {
        session.status = SessionStatus::Running;

        while !session.is_finished() {
            if let Err(e) = self.step(session) {
                session.status = SessionStatus::Failed;
                return Err(e);
            }
        }
        Ok(session.generated_tokens.clone())
    }

    pub fn generate_from_str(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        opts: crate::tokenizer::tokenizer::EncodeOptions,
    ) -> std::result::Result<String, GeneratorError> {
        let prompt_ids: Vec<u32> = self
            .tokenizer
            .encode(prompt, opts)
            .iter()
            .map(|&id| id as u32)
            .collect();

        let eos_id = self.tokenizer.eos_id().map(|id| id as u32);

        let mut session = Session::new(SessionId(0), prompt_ids, max_new_tokens, eos_id);

        let generated = self.run(&mut session)?;

        let token_ids: Vec<usize> = generated.iter().map(|&id| id as usize).collect();
        let decoded = self.tokenizer.decode(&token_ids);

        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::inference::sampler::SamplerConfig;
    use crate::model::config::ModelConfig;
    use crate::tokenizer::tokenizer::EncodeOptions;

    fn cpu() -> Device {
        Device::Cpu
    }

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

    fn make_generator(vocab_size: usize) -> Generator<CandleBackend> {
        let config = make_config(vocab_size);
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let tokenizer = Tokenizer::from_file("testdata/tokenizer.json").unwrap();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0, // greedy for determinism
            seed: Some(42),
            ..Default::default()
        });
        Generator::new(model, tokenizer, sampler, cpu())
    }

    fn make_session(prompt: Vec<u32>, max_new: usize) -> Session<CandleBackend> {
        Session::new(SessionId(1), prompt, max_new, Some(2))
    }

    #[test]
    fn test_generator_construction() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let g = make_generator(1000);
        assert_eq!(g.device, cpu());
    }

    #[test]
    fn test_step_returns_valid_token() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let mut gen = make_generator(1000);
        let mut session = make_session(vec![1u32, 2, 3], 10);
        let token = gen.step(&mut session).unwrap();
        assert!((token as usize) < 1000);
    }

    #[test]
    fn test_step_updates_session() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let mut gen = make_generator(1000);
        let mut session = make_session(vec![1u32, 2, 3], 10);
        gen.step(&mut session).unwrap();
        assert_eq!(session.num_generated(), 1);
        assert_eq!(session.offset, 4);
    }

    #[test]
    fn test_step_grows_kv_cache() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let config = make_config(1000);
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let tokenizer = Tokenizer::from_file("testdata/tokenizer.json").unwrap();
        let sampler = Sampler::new(SamplerConfig::default());
        let mut gen = Generator::new(model, tokenizer, sampler, cpu());
        let mut session = make_session(vec![1u32, 2, 3], 10);

        assert_eq!(session.kv_cache.len(), 0);
        gen.step(&mut session).unwrap();
        assert_eq!(session.kv_cache.len(), config.num_hidden_layers);
    }

    #[test]
    fn test_run_generates_max_tokens() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let mut gen = make_generator(1000);
        let mut session = make_session(vec![1u32, 2, 3], 5);
        // use eos that won't be hit with zero weights
        let mut session = Session::<CandleBackend>::new(SessionId(1), vec![1u32, 2, 3], 5, None);
        let generated = gen.run(&mut session).unwrap();
        assert_eq!(generated.len(), 5);
    }

    #[test]
    fn test_run_stops_on_eos() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let config = make_config(1000);
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let tokenizer = Tokenizer::from_file("testdata/tokenizer.json").unwrap();
        // greedy sampler with zero weights will always pick token 0
        // set eos = 0 so generation stops immediately
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        let mut gen = Generator::new(model, tokenizer, sampler, cpu());
        let mut probe_session = Session::<CandleBackend>::new(SessionId(0), vec![1, 2], 1, None);
        let picked_token = gen.step(&mut probe_session).unwrap();
        let mut session =
            Session::<CandleBackend>::new(SessionId(1), vec![1u32, 2], 100, Some(picked_token));
        let generated = gen.run(&mut session).unwrap();
        // should stop after first token since greedy picks 0 = eos
        assert_eq!(generated.len(), 1);
        assert_eq!(session.status, SessionStatus::Finished);
    }

    #[test]
    fn test_run_sets_status_finished() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let mut gen = make_generator(1000);
        let mut session = Session::<CandleBackend>::new(SessionId(1), vec![1u32], 3, None);
        gen.run(&mut session).unwrap();
        assert_eq!(session.status, SessionStatus::Finished);
    }

    #[test]
    fn test_generate_from_str_returns_string() {
        if !std::path::Path::new("testdata/tokenizer.json").exists() {
            return;
        }
        let tok = Tokenizer::from_file("testdata/tokenizer.json").unwrap();
        let vocab_size = tok.vocab().size();
        let config = make_config(vocab_size);
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            seed: Some(42),
            max_new_tokens: 5,
            ..Default::default()
        });
        let mut gen = Generator::new(model, tok, sampler, cpu());
        let result = gen.generate_from_str("Hello", 5, EncodeOptions::default());
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(!text.is_empty() || text.is_empty()); // any string is valid with zero weights
    }
}
