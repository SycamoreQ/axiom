use rand::prelude::*;
use std::collections::HashSet;

/*
The sampler takes logits from the model and returns the next token ID. This is where all the sampling strategies live.
*/

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,        // 1.0 = no change, <1.0 = sharper, >1.0 = flatter
    pub top_k: Option<usize>,    // keep only top k tokens before sampling
    pub top_p: Option<f32>,      // nucleus sampling — keep tokens summing to p
    pub repetition_penalty: f32, // 1.0 = no penalty, >1.0 = penalize repeats
    pub max_new_tokens: usize,
    pub seed: Option<u64>,
    pub vocab_size: Option<usize>,
}

pub struct Sampler {
    config: SamplerConfig,
    rng: rand::rngs::StdRng,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            max_new_tokens: 256,
            seed: None,
            vocab_size: None,
        }
    }
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    pub fn sample(&mut self, logits: &[f32], previous_tokens: &[u32]) -> u32 {
        let mut logits_vec = logits.to_vec();

        if let Some(vocab_size) = self.config.vocab_size {
            if logits_vec.len() > vocab_size {
                for v in &mut logits_vec[vocab_size..] {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        //Repetition Penalty (Do this first on raw logits)
        if self.config.repetition_penalty != 1.0 {
            Self::apply_repetition_penalty(
                &mut logits_vec,
                previous_tokens,
                self.config.repetition_penalty,
            );
        }

        //Check for Greedy Shortcut
        // If temperature is effectively 0, just pick the max and save compute.
        if self.config.temperature < 1e-5 {
            return Self::greedy(&logits_vec);
        }

        //Temperature Scaling
        if self.config.temperature != 1.0 {
            Self::apply_temperature(&mut logits_vec, self.config.temperature);
        }

        //Top-K Filtering
        if let Some(k) = self.config.top_k {
            Self::apply_top_k(&mut logits_vec, k);
        }

        //Convert to Probabilities
        let mut probs = Self::softmax(&logits_vec);

        //Top-P (Nucleus) Filtering
        if let Some(p) = self.config.top_p {
            Self::apply_top_p(&mut probs, p);
        }

        //Final Stochastic Sample
        Self::sample_from_probs(&probs, &mut self.rng)
    }

    fn greedy(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    fn apply_temperature(logits: &mut Vec<f32>, temperature: f32) {
        let inv_temp = 1.0 / temperature;
        for l in logits.iter_mut() {
            *l *= inv_temp;
        }
    }

    fn apply_top_k(logits: &mut Vec<f32>, k: usize) {
        let k = k.clamp(1, logits.len());
        let mut indices: Vec<usize> = (0..logits.len()).collect();

        indices.select_nth_unstable_by(k - 1, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

        let threshold = logits[indices[k - 1]];
        for l in logits.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn apply_repetition_penalty(logits: &mut Vec<f32>, previous: &[u32], penalty: f32) {
        let mut seen = HashSet::new();
        for &id in previous {
            if seen.insert(id) {
                let idx = id as usize;
                if idx < logits.len() {
                    // If logit is positive, divide by penalty. If negative, multiply.
                    if logits[idx] > 0.0 {
                        logits[idx] /= penalty;
                    } else {
                        logits[idx] *= penalty;
                    }
                }
            }
        }
    }

    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    fn apply_top_p(probs: &mut Vec<f32>, p: f32) {
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = indexed_probs.len();

        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out the probabilities outside the nucleus
        let mut new_probs = vec![0.0; probs.len()];
        let mut new_sum = 0.0;
        for i in 0..cutoff_idx {
            let (idx, prob) = indexed_probs[i];
            new_probs[idx] = prob;
            new_sum += prob;
        }

        for pr in new_probs.iter_mut() {
            *pr /= new_sum;
        }
        *probs = new_probs;
    }

    fn sample_from_probs(probs: &[f32], rng: &mut rand::rngs::StdRng) -> u32 {
        use rand::distributions::{Distribution, WeightedIndex};
        let dist = WeightedIndex::new(probs).expect("Invalid probability distribution");
        dist.sample(rng) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sampler(config: SamplerConfig) -> Sampler {
        Sampler::new(config)
    }

    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    fn peaked_logits(n: usize, peak: usize) -> Vec<f32> {
        let mut v = vec![0.0; n];
        v[peak] = 100.0;
        v
    }

    #[test]
    fn test_greedy_picks_max() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        assert_eq!(Sampler::greedy(&logits), 2);
    }

    #[test]
    fn test_greedy_single_element() {
        assert_eq!(Sampler::greedy(&[1.0]), 0);
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let mut sampler = make_sampler(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        let logits = peaked_logits(100, 42);
        let token = sampler.sample(&logits, &[]);
        assert_eq!(token, 42);
    }

    #[test]
    fn test_temperature_scaling_sharpens() {
        // low temperature should consistently pick the peak
        let mut sampler = make_sampler(SamplerConfig {
            temperature: 0.01,
            seed: Some(42),
            ..Default::default()
        });
        let logits = peaked_logits(100, 7);
        for _ in 0..10 {
            assert_eq!(sampler.sample(&logits, &[]), 7);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = Sampler::softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum: {}", sum);
    }

    #[test]
    fn test_softmax_all_positive() {
        let logits = vec![-1.0, 0.0, 1.0, 2.0];
        let probs = Sampler::softmax(&logits);
        for p in &probs {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_softmax_preserves_order() {
        let logits = vec![1.0, 3.0, 2.0];
        let probs = Sampler::softmax(&logits);
        assert!(probs[1] > probs[2]);
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_top_k_keeps_k_tokens() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        Sampler::apply_top_k(&mut logits, 2);
        let finite_count = logits.iter().filter(|&&l| l.is_finite()).count();
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn test_top_k_keeps_highest() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        Sampler::apply_top_k(&mut logits, 2);
        // indices 1 (5.0) and 3 (4.0) should survive
        assert!(logits[1].is_finite());
        assert!(logits[3].is_finite());
        assert!(logits[0].is_infinite());
        assert!(logits[2].is_infinite());
        assert!(logits[4].is_infinite());
    }

    #[test]
    fn test_top_k_full_vocab_unchanged() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        Sampler::apply_top_k(&mut logits, 3);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_sums_to_one() {
        let mut probs = vec![0.4, 0.3, 0.2, 0.1];
        Sampler::apply_top_p(&mut probs, 0.9);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "top_p probs sum: {}", sum);
    }

    #[test]
    fn test_top_p_full_nucleus() {
        // p=1.0 keeps everything
        let mut probs = vec![0.4, 0.3, 0.2, 0.1];
        let original = probs.clone();
        Sampler::apply_top_p(&mut probs, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_p_small_nucleus() {
        // p=0.5 with [0.4, 0.3, 0.2, 0.1] — keeps first two (0.4+0.3=0.7 >= 0.5)
        let mut probs = vec![0.4, 0.3, 0.2, 0.1];
        Sampler::apply_top_p(&mut probs, 0.5);
        // zeroed out tokens
        assert_eq!(probs[2], 0.0);
        assert_eq!(probs[3], 0.0);
    }

    #[test]
    fn test_repetition_penalty_positive_logit() {
        let mut logits = vec![0.0, 2.0, 0.0, 0.0];
        Sampler::apply_repetition_penalty(&mut logits, &[1u32], 2.0);
        // positive logit divided by penalty
        assert!((logits[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        let mut logits = vec![0.0, -2.0, 0.0, 0.0];
        Sampler::apply_repetition_penalty(&mut logits, &[1u32], 2.0);
        // negative logit multiplied by penalty
        assert!((logits[1] - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_no_effect_unseen() {
        let mut logits = vec![1.0, 2.0, 3.0];
        Sampler::apply_repetition_penalty(&mut logits, &[5u32], 2.0);
        // token 5 is out of range — no change
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_repetition_penalty_deduplicates() {
        // token appearing twice in history should only be penalized once
        let mut logits = vec![0.0, 4.0, 0.0];
        Sampler::apply_repetition_penalty(&mut logits, &[1u32, 1u32], 2.0);
        assert!((logits[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_with_seed_is_deterministic() {
        let config = SamplerConfig {
            temperature: 1.0,
            seed: Some(42),
            ..Default::default()
        };
        let logits = uniform_logits(100);

        let mut s1 = Sampler::new(config.clone());
        let mut s2 = Sampler::new(config);

        let results1: Vec<u32> = (0..5).map(|_| s1.sample(&logits, &[])).collect();
        let results2: Vec<u32> = (0..5).map(|_| s2.sample(&logits, &[])).collect();

        assert_eq!(results1, results2);
    }

    #[test]
    fn test_sample_returns_valid_token_id() {
        let vocab_size = 1000;
        let mut sampler = make_sampler(SamplerConfig {
            seed: Some(0),
            ..Default::default()
        });
        let logits = uniform_logits(vocab_size);
        for _ in 0..20 {
            let token = sampler.sample(&logits, &[]);
            assert!((token as usize) < vocab_size);
        }
    }

    #[test]
    fn test_top_k_with_sampling() {
        let mut sampler = make_sampler(SamplerConfig {
            top_k: Some(3),
            seed: Some(1),
            ..Default::default()
        });
        let mut logits = vec![f32::NEG_INFINITY; 100];
        logits[10] = 5.0;
        logits[20] = 4.0;
        logits[30] = 3.0;
        // only tokens 10, 20, 30 are valid
        for _ in 0..20 {
            let token = sampler.sample(&logits, &[]);
            assert!(
                token == 10 || token == 20 || token == 30,
                "unexpected token: {}",
                token
            );
        }
    }
}
