use crate::core::backend::Backend;
use crate::core::error::Result;
use crate::core::tensor::TensorOps;
use crate::inference::draft::DraftModel;
use crate::inference::sampler::Sampler;
use crate::inference::session::{self, Session, SessionStatus};
use crate::model::model::LlamaModel;
use candle_core::Tensor;
use rand::thread_rng;
use rand::Rng;

pub struct SpeculativeDecoder<B: Backend> {
    target: LlamaModel<B>,
    draft: DraftModel<B>,
    target_sampler: Sampler,
    accepted_total: usize, // lifetime stats
    drafted_total: usize,
}

impl<B: Backend> SpeculativeDecoder<B> {
    pub fn new(target: LlamaModel<B>, draft: DraftModel<B>, target_sampler: Sampler) -> Self {
        Self {
            target,
            draft,
            target_sampler,
            accepted_total: 0,
            drafted_total: 0,
        }
    }

    pub fn step(&mut self, session: &mut Session<B>) -> Result<Vec<u32>> {
        let drafted_pairs = self.draft.draft(session)?;
        self.drafted_total += drafted_pairs.len(); // Track total tokens drafted

        let tokens_to_verify: Vec<u32> = drafted_pairs.iter().map(|(t, _)| *t).collect();

        // Batched forward pass on target model
        let target_logits_tensor = self.target.forward(
            &tokens_to_verify,
            Some(&mut session.kv_cache),
            session.offset,
        )?;

        let seq_len = target_logits_tensor.shape().dims()[1];
        let last_logits_tensor = target_logits_tensor
            .narrow(1, seq_len - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?;

        let last_logits_vec = last_logits_tensor.to_vec_f32()?;

        let mut accepted_tokens = Vec::new();

        for i in 0..drafted_pairs.len() {
            let (draft_token, draft_logits) = &drafted_pairs[i];
            let pos_logits = target_logits_tensor
                .narrow(1, i, 1)?
                .squeeze(0)?
                .squeeze(0)?
                .to_vec_f32()?;
            let p_target = Sampler::softmax(&pos_logits);
            let p_draft = Sampler::softmax(draft_logits);

            if self.check_acceptance(*draft_token, &p_target, &p_draft) {
                accepted_tokens.push(*draft_token);
            } else {
                // Found a rejection: sample correction and stop this step
                let correction = self.sample_correction_dist(&p_target, &p_draft);
                accepted_tokens.push(correction);

                self.accepted_total += accepted_tokens.len();
                for &t in &accepted_tokens {
                    session.push_token(t);
                }
                return Ok(accepted_tokens);
            }
        }

        let bonus_pos = target_logits_tensor
            .narrow(1, drafted_pairs.len(), 1)?
            .squeeze(0)?
            .squeeze(0)?
            .to_vec_f32()?;
        let last_p_target = Sampler::softmax(&bonus_pos);
        let prev_token = 0;
        let bonus_token = self.target_sampler.sample(&last_p_target, &[prev_token]);
        accepted_tokens.push(bonus_token);

        self.accepted_total += accepted_tokens.len();
        for &t in &accepted_tokens {
            session.push_token(t);
        }
        Ok(accepted_tokens)
    }

    pub fn step_to_completion(&mut self, session: &mut Session<B>) -> Result<Vec<u32>> {
        let mut all_generated = Vec::new();

        while !session.is_finished() {
            let mut new_tokens = self.step(session)?;
            all_generated.append(&mut new_tokens);

            // Safety break if we exceed max tokens outside of the session check
            if all_generated.len() >= session.max_new_tokens as usize {
                break;
            }
        }

        Ok(all_generated)
    }

    pub fn reset_stats(&mut self) {
        self.accepted_total = 0;
        self.drafted_total = 0;
    }

    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted_total == 0 {
            return 0.0;
        }
        self.accepted_total as f32 / self.drafted_total as f32
    }

    fn check_acceptance(&self, token: u32, p_target: &[f32], p_draft: &[f32]) -> bool {
        let mut rng = thread_rng();
        let r: f32 = rng.gen();

        let target_prob = p_target[token as usize];
        let draft_prob = p_draft[token as usize];

        acceptance_criterion(target_prob, draft_prob, r)
    }

    //Samples a correction token from the adjusted distribution when a draft is rejected.
    fn sample_correction_dist(&mut self, p_target: &[f32], p_draft: &[f32]) -> u32 {
        let adj_probs = adjusted_distribution(p_target, p_draft);
        let context = vec![0u32];
        self.target_sampler.sample(&adj_probs, &context)
    }
}

//Determines if a drafted token is statistically valid.
pub fn acceptance_criterion(p_target: f32, p_draft: f32, r: f32) -> bool {
    r < (p_target / p_draft).min(1.0)
}

//This is used to sample a replacement token when a draft is rejected.
pub fn adjusted_distribution(target_probs: &[f32], draft_probs: &[f32]) -> Vec<f32> {
    let mut adj: Vec<f32> = target_probs
        .iter()
        .zip(draft_probs.iter())
        .map(|(p_t, p_d)| (p_t - p_d).max(0.0))
        .collect();

    let sum: f32 = adj.iter().sum();

    if sum > 0.0 {
        for v in adj.iter_mut() {
            *v /= sum;
        }
    } else {
        return target_probs.to_vec();
    }

    adj
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::inference::draft::DraftModel;
    use crate::inference::sampler::{Sampler, SamplerConfig};
    use crate::inference::session::SessionId;
    use crate::model::config::ModelConfig;
    use crate::model::model::LlamaModel;

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

    fn make_decoder() -> SpeculativeDecoder<CandleBackend> {
        let target = LlamaModel::<CandleBackend>::new(&make_config(256), &cpu()).unwrap();
        let draft_model = LlamaModel::<CandleBackend>::new(&make_config(256), &cpu()).unwrap();
        let draft_sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            seed: Some(1),
            ..Default::default()
        });
        let target_sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            seed: Some(2),
            ..Default::default()
        });
        let draft = DraftModel::new(draft_model, draft_sampler, 4);
        SpeculativeDecoder::new(target, draft, target_sampler)
    }

    fn make_session(eos: Option<u32>) -> Session<CandleBackend> {
        Session::new(SessionId(0), vec![1u32, 2, 3], 32, eos)
    }

    #[test]
    fn test_acceptance_rate_zero_initially() {
        let dec = make_decoder();
        assert_eq!(dec.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_acceptance_rate_after_manual_set() {
        let mut dec = make_decoder();
        dec.accepted_total = 3;
        dec.drafted_total = 4;
        let rate = dec.acceptance_rate();
        assert!((rate - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_reset_stats() {
        let mut dec = make_decoder();
        dec.accepted_total = 10;
        dec.drafted_total = 20;
        dec.reset_stats();
        assert_eq!(dec.accepted_total, 0);
        assert_eq!(dec.drafted_total, 0);
        assert_eq!(dec.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_acceptance_criterion_accepts_higher_prob() {
        assert!(acceptance_criterion(1.0, 0.5, 0.0));
    }

    #[test]
    fn test_acceptance_criterion_always_accepts_when_target_geq_draft() {
        assert!(acceptance_criterion(0.8, 0.4, 0.99));
    }

    #[test]
    fn test_acceptance_criterion_rejects_low_prob() {
        assert!(!acceptance_criterion(0.1, 0.9, 0.5));
    }

    #[test]
    fn test_acceptance_criterion_boundary() {
        // r == ratio exactly -> should reject (r < ratio is false)
        let ratio = 0.1f32 / 0.9f32;
        assert!(!acceptance_criterion(0.1, 0.9, ratio));
    }

    #[test]
    fn test_adjusted_distribution_sums_to_one() {
        let p_target = vec![0.4, 0.3, 0.2, 0.1];
        let p_draft = vec![0.1, 0.2, 0.3, 0.4];
        let adj = adjusted_distribution(&p_target, &p_draft);
        let sum: f32 = adj.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum was {}", sum);
    }

    #[test]
    fn test_adjusted_distribution_clips_negative() {
        let p_target = vec![0.1, 0.9];
        let p_draft = vec![0.9, 0.1];
        let adj = adjusted_distribution(&p_target, &p_draft);
        assert_eq!(adj[0], 0.0);
        assert!((adj[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adjusted_distribution_all_zero_draft() {
        let p_target = vec![0.25, 0.25, 0.25, 0.25];
        let p_draft = vec![0.0, 0.0, 0.0, 0.0];
        let adj = adjusted_distribution(&p_target, &p_draft);
        for &v in &adj {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_step_returns_nonempty_tokens() {
        let mut dec = make_decoder();
        let mut session = make_session(None);
        let tokens = dec.step(&mut session).unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_step_tokens_in_vocab_range() {
        let mut dec = make_decoder();
        let mut session = make_session(None);
        let tokens = dec.step(&mut session).unwrap();
        for &t in &tokens {
            assert!((t as usize) < 256, "token {} out of vocab", t);
        }
    }

    #[test]
    fn test_step_updates_drafted_total() {
        let mut dec = make_decoder();
        let mut session = make_session(None);
        dec.step(&mut session).unwrap();
        assert!(dec.drafted_total > 0);
    }

    #[test]
    fn test_step_to_completion_finishes_session() {
        let mut dec = make_decoder();
        // eos=0, zero-weight model always picks 0 -> stops immediately
        let mut session = make_session(Some(0));
        dec.step_to_completion(&mut session).unwrap();
        assert!(session.is_finished());
    }

    #[test]
    fn test_step_to_completion_respects_max_tokens() {
        let mut dec = make_decoder();
        let mut session = Session::new(SessionId(0), vec![1u32, 2], 5, None);
        dec.step_to_completion(&mut session).unwrap();
        assert!(session.num_generated() <= 5);
    }
}
