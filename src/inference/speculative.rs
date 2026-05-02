use crate::core::backend::Backend;
use crate::core::error::Result;
use crate::inference::draft::DraftModel;
use crate::inference::sampler::Sampler;
use crate::inference::session::{self, Session, SessionStatus};
use crate::model::model::LlamaModel;

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
        let target_logits_tensor =
            self.target
                .forward(&tokens_to_verify, session.kv_cache, session.offset)?;
        let target_logits_vecs = self.tensor_to_vec_of_vecs(target_logits_tensor)?;

        let mut accepted_tokens = Vec::new();

        for i in 0..drafted_pairs.len() {
            let (draft_token, draft_logits) = &drafted_pairs[i];
            let p_target = self.softmax(&target_logits_vecs[i]);
            let p_draft = self.softmax(draft_logits);

            if self.check_acceptance(*draft_token, &p_target, &p_draft) {
                accepted_tokens.push(*draft_token);
            } else {
                // Found a rejection: sample correction and stop this step
                let correction = self.sample_correction_dist(&p_target, &p_draft);
                accepted_tokens.push(correction);

                self.accepted_total += accepted_tokens.len();
                session.update(accepted_tokens.len());
                return Ok(accepted_tokens);
            }
        }

        let last_p_target = self.softmax(&target_logits_vecs.last().unwrap());
        let bonus_token = self.sampler.sample_raw_probs(&last_p_target);
        accepted_tokens.push(bonus_token);

        self.accepted_total += accepted_tokens.len();
        session.update(accepted_tokens.len());
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

    pub fn acceptance_rate(&self) -> f32 {
        if self.drafted_total == 0 {
            return 0.0;
        }
        self.accepted_total as f32 / self.drafted_total as f32
    }

    //Returns true if the draft token is accepted based on the target model's distribution
    fn check_acceptance(&self, token: u32, p_target: &[f32], p_draft: &[f32]) -> bool {
        let mut rng = rand::thread_rng();
        let p_t = p_target[token as usize];
        let p_d = p_draft[token as usize];

        if p_t >= p_d {
            true
        } else {
            let r: f32 = rng.gen();
            r < (p_t / p_d)
        }
    }

    //Standard speculative correction: samples from (P_target - P_draft)+ normalized
    fn sample_correction_dist(&self, p_target: &[f32], p_draft: &[f32]) -> u32 {
        let mut diff: Vec<f32> = p_target
            .iter()
            .zip(p_draft.iter())
            .map(|(t, d)| (t - d).max(0.0))
            .collect();

        let sum: f32 = diff.iter().sum();
        for val in diff.iter_mut() {
            *val /= sum;
        }

        self.sampler.sample_raw_probs(&diff)
    }

    pub fn reset_stats(&mut self) {
        self.accepted_total = 0;
        self.drafted_total = 0;
    }
}
