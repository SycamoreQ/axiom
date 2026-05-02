use std::os::raw;

use crate::core::backend::Backend;
use crate::core::error::Result;
use crate::inference::sampler::Sampler;
use crate::inference::session::Session;
use crate::model::model::LlamaModel;

pub struct DraftModel<B: Backend> {
    model: LlamaModel<B>,
    sampler: Sampler,
    gamma: usize,
}

impl<B: Backend> DraftModel<B> {
    pub fn new(model: LlamaModel<B>, sampler: Sampler, gamma: usize) -> Self {
        Self {
            model,
            sampler,
            gamma,
        }
    }

    pub fn draft(&mut self, session: &mut Session<B>) -> Result<Vec<(u32, Vec<f32>)>> {
        let draft_offset = session.offset;
        let mut draft_cache = session.kv_cache.clone()?;
        let mut drafted_pairs = Vec::new();

        for i in 0..self.gamma - 1 {
            if (i == 0) {
                let raw_logits =
                    self.model
                        .forward(session.next_input_tokens(), draft_cache, draft_offset)?;
            } else {
                let next_token = self.sampler.sample(raw_logits.0, &session.generated_tokens);

                let raw_logits = self
                    .model
                    .forward(&[next_token], draft_cache, draft_offset)?;

                drafted_pairs.push((next_token, last_logits));
                draft_offset += current_input.len();
                current_input = vec![next_token]
            }
        }

        Ok(drafted_pairs)
    }

    pub fn gamma(&self) -> usize {
        self.gamma
    }
}
