use crate::core::backend::Backend;
use crate::core::error::Result;
use crate::core::tensor::TensorOps;
use crate::inference::sampler::Sampler;
use crate::inference::session::Session;
use crate::model::model::LlamaModel;

pub struct DraftModel<B: Backend> {
    model: LlamaModel<B>,
    sampler: Sampler,
    gamma: usize,
    pub kv_cache: Vec<(B::Tensor, B::Tensor)>,
    pub offset: usize,
}

impl<B: Backend> DraftModel<B> {
    pub fn new(model: LlamaModel<B>, sampler: Sampler, gamma: usize) -> Self {
        Self {
            model,
            sampler,
            gamma,
            kv_cache: Vec::new(),
            offset: 0,
        }
    }

    pub fn gamma(&self) -> usize {
        self.gamma
    }

    pub fn model(&self) -> &LlamaModel<B> {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut LlamaModel<B> {
        &mut self.model
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.offset = 0;
    }

    pub fn catch_up_to(&mut self, session: &Session<B>) -> Result<()> {
        let max_seq_len = session.prompt_tokens.len() + session.max_new_tokens;
        if self.kv_cache.is_empty() && !session.prompt_tokens.is_empty() {
            self.model.forward(
                &session.prompt_tokens,
                Some(&mut self.kv_cache),
                0,
                max_seq_len,
            )?;
            self.offset = session.prompt_tokens.len();
        }

        let target_len = session.prompt_tokens.len() + session.generated_tokens.len().saturating_sub(1);
        if self.offset < target_len {
            let start = self.offset.saturating_sub(session.prompt_tokens.len());
            let end = target_len - session.prompt_tokens.len();
            if start < end && end <= session.generated_tokens.len() {
                let missing = &session.generated_tokens[start..end];
                self.model.forward(
                    missing,
                    Some(&mut self.kv_cache),
                    self.offset,
                    max_seq_len,
                )?;
                self.offset += missing.len();
            }
        }
        Ok(())
    }

    pub fn rollback_to(&mut self, new_offset: usize) -> Result<()> {
        for (k, v) in &mut self.kv_cache {
            if k.shape().dims().len() > 1 && k.shape().dims()[1] > new_offset {
                *k = k.narrow(1, 0, new_offset)?;
                *v = v.narrow(1, 0, new_offset)?;
            }
        }
        self.offset = new_offset;
        Ok(())
    }

    /// Generate `gamma` draft tokens using the draft model.
    ///
    /// Uses the draft model's own KV cache. Returns a list of (token_id, logits) pairs
    /// for verification by the target model.
    pub fn draft(&mut self, session: &mut Session<B>) -> Result<Vec<(u32, Vec<f32>)>>
    where
        B::Tensor: Clone,
    {
        if self.gamma == 0 {
            return Ok(Vec::new());
        }

        self.catch_up_to(session)?;

        let mut current_input = if let Some(&last) = session.generated_tokens.last() {
            vec![last]
        } else if let Some(&last) = session.prompt_tokens.last() {
            vec![last]
        } else {
            return Ok(Vec::new());
        };

        let mut draft_offset = self.offset;
        let mut results: Vec<(u32, Vec<f32>)> = Vec::new();
        let max_seq_len = session.prompt_tokens.len() + session.max_new_tokens;

        for _i in 0..self.gamma {
            let logits_tensor = self.model.forward(
                &current_input,
                Some(&mut self.kv_cache),
                draft_offset,
                max_seq_len,
            )?;

            let seq_len = logits_tensor.shape().dims()[1];
            let last_logits_tensor = logits_tensor
                .narrow(1, seq_len - 1, 1)?
                .squeeze(0)?
                .squeeze(0)?;

            let last_logits_vec = last_logits_tensor.to_vec_f32()?;

            // Build the all-tokens list including draft tokens for repetition penalty
            let mut all_tokens = session.all_tokens();
            for (t, _) in &results {
                all_tokens.push(*t);
            }

            let next_token = self.sampler.sample(&last_logits_vec, &all_tokens);
            results.push((next_token, last_logits_vec));

            draft_offset += current_input.len();
            self.offset = draft_offset;
            current_input = vec![next_token];

            if let Some(eos_id) = session.eos_token_id {
                if next_token == eos_id {
                    break;
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::inference::sampler::SamplerConfig;
    use crate::inference::session::SessionId;
    use crate::model::config::ModelConfig;

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
            lazy_moe: false,
            head_dim_override: None,
        }
    }

    fn make_draft(gamma: usize) -> DraftModel<CandleBackend> {
        let model = LlamaModel::<CandleBackend>::new(&make_config(256), &cpu()).unwrap();
        let sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        });
        DraftModel::new(model, sampler, gamma)
    }

    fn make_session() -> Session<CandleBackend> {
        Session::new(SessionId(0), vec![1u32, 2, 3], 64, None)
    }

    #[test]
    fn test_draft_returns_gamma_tokens() {
        let mut draft = make_draft(4);
        let mut session = make_session();
        let pairs = draft.draft(&mut session).unwrap();
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn test_draft_logits_have_correct_vocab_size() {
        let mut draft = make_draft(3);
        let mut session = make_session();
        let pairs = draft.draft(&mut session).unwrap();
        for (_, logits) in &pairs {
            assert_eq!(logits.len(), 256);
        }
    }

    #[test]
    fn test_draft_zero_gamma_returns_empty() {
        let mut draft = make_draft(0);
        let mut session = make_session();
        let pairs = draft.draft(&mut session).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_draft_does_not_modify_session() {
        let mut draft = make_draft(3);
        let mut session = make_session();
        let offset_before = session.offset;
        let generated_before = session.generated_tokens.len();
        draft.draft(&mut session).unwrap();
        assert_eq!(session.offset, offset_before);
        assert_eq!(session.generated_tokens.len(), generated_before);
    }

    #[test]
    fn test_draft_token_ids_in_vocab_range() {
        let mut draft = make_draft(4);
        let mut session = make_session();
        let pairs = draft.draft(&mut session).unwrap();
        for (token, _) in &pairs {
            assert!((*token as usize) < 256, "token {} out of vocab", token);
        }
    }

    #[test]
    fn test_gamma_accessor() {
        let draft = make_draft(5);
        assert_eq!(draft.gamma(), 5);
    }
}
