use crate::core::backend::Backend;
use crate::core::error::Result;
use crate::core::tensor::TensorOps;
use crate::inference::draft::DraftModel;
use crate::inference::sampler::Sampler;
use crate::inference::session::Session;
use crate::model::model::LlamaModel;
use rand::Rng;

//Accept or reject a draft token using the speculative sampling criterion.
pub fn acceptance_criterion(p_target: f32, p_draft: f32, r: f32) -> bool {
    if p_draft <= 0.0 {
        return true; // avoid division by zero — accept if draft had zero mass
    }
    r < (p_target / p_draft).min(1.0)
}

//Compute the correction distribution: max(0, p_target - p_draft), renormalized.
//Sampled on rejection to maintain the target distribution in expectation.
pub fn adjusted_distribution(target_probs: &[f32], draft_probs: &[f32]) -> Vec<f32> {
    let mut diff: Vec<f32> = target_probs
        .iter()
        .zip(draft_probs.iter())
        .map(|(t, d)| (t - d).max(0.0))
        .collect();

    let sum: f32 = diff.iter().sum();
    if sum > 0.0 {
        for v in diff.iter_mut() {
            *v /= sum;
        }
    } else {
        let n = diff.len() as f32;
        for v in diff.iter_mut() {
            *v = 1.0 / n;
        }
    }
    diff
}

//Weighted sampling from a probability vector.
pub fn sample_from_distribution(probs: &[f32], rng: &mut rand::rngs::StdRng) -> u32 {
    use rand::distributions::{Distribution, WeightedIndex};
    let dist = WeightedIndex::new(probs).expect("invalid probability distribution");
    dist.sample(rng) as u32
}

/*
SpeculativeDecoder: main loop which takes the draft tokens and does a forward pass
*/

pub struct SpeculativeDecoder<B: Backend> {
    target: LlamaModel<B>,
    draft: DraftModel<B>,
    target_sampler: Sampler,
    pub accepted_total: usize,
    pub drafted_total: usize,
}

impl<B: Backend> SpeculativeDecoder<B>
where
    B::Tensor: Clone,
{
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
        //draft
        let drafted_pairs = self.draft.draft(session)?;
        self.drafted_total += drafted_pairs.len();
        let next_tokens = session.next_input_tokens().to_vec();
        let tokens_to_verify: Vec<u32> = drafted_pairs.iter().map(|(t, _)| *t).collect();

        if drafted_pairs.is_empty() {
            // gamma == 0: just sample one token from target
            let logits = self
                .target
                .forward(&tokens_to_verify, None, session.offset)?;
            let seq_len = logits.shape().dim(1)?;
            let last = logits.narrow(1, seq_len - 1, 1)?.squeeze(0)?.squeeze(0)?;
            let logits_vec = last.to_vec_f32()?;
            let token = self
                .target_sampler
                .sample(&logits_vec, &session.all_tokens());
            session.push_token(token);
            return Ok(vec![token]);
        }

        let tokens_to_verify: Vec<u32> = drafted_pairs.iter().map(|(t, _)| *t).collect();

        //batched target forward over all draft tokens
        let target_logits_tensor = self
            .target
            .forward(&tokens_to_verify, None, session.offset)?;

        let gamma = drafted_pairs.len();
        let mut accepted_tokens: Vec<u32> = Vec::new();
        let mut rng = rand::thread_rng();

        //accept/reject loop
        for i in 0..gamma {
            let (draft_token, ref draft_logits) = drafted_pairs[i];

            // extract target logits at position i
            let pos_logits = target_logits_tensor
                .narrow(1, i, 1)?
                .squeeze(0)?
                .squeeze(0)?
                .to_vec_f32()?;
            let p_target = Sampler::softmax(&pos_logits);
            let p_draft = Sampler::softmax(draft_logits);

            let r: f32 = rng.gen();
            if acceptance_criterion(
                p_target[draft_token as usize],
                p_draft[draft_token as usize],
                r,
            ) {
                accepted_tokens.push(draft_token);
            } else {
                // rejection: sample correction token
                let adj = adjusted_distribution(&p_target, &p_draft);
                let correction = self.target_sampler.sample(&adj, &session.all_tokens());
                accepted_tokens.push(correction);

                self.accepted_total += accepted_tokens.len();
                for &t in &accepted_tokens {
                    session.push_token(t);
                }
                return Ok(accepted_tokens);
            }
        }

        //all accepted — sample bonus token from target at position gamma
        let bonus_logits = target_logits_tensor
            .narrow(1, gamma - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?
            .to_vec_f32()?;
        let bonus_token = self
            .target_sampler
            .sample(&bonus_logits, &session.all_tokens());
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
        }
        Ok(all_generated)
    }

    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted_total == 0 {
            return 0.0;
        }
        self.accepted_total as f32 / self.drafted_total as f32
    }

    pub fn reset_stats(&mut self) {
        self.accepted_total = 0;
        self.drafted_total = 0;
    }
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

    // --- free function tests ---

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
        let ratio = 0.1f32 / 0.9f32;
        assert!(!acceptance_criterion(0.1, 0.9, ratio));
    }

    #[test]
    fn test_acceptance_criterion_zero_draft_always_accepts() {
        assert!(acceptance_criterion(0.5, 0.0, 0.99));
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

    // --- decoder tests ---

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
    fn test_step_updates_session_offset() {
        let mut dec = make_decoder();
        let mut session = make_session(None);
        let offset_before = session.offset;
        dec.step(&mut session).unwrap();
        assert!(session.offset > offset_before);
    }

    #[test]
    fn test_step_to_completion_finishes_session() {
        let mut dec = make_decoder();
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
