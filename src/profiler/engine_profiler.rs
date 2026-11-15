use crate::profiler::metrics::{
    InferenceProfile, LatencyStats, MemoryStats, SpeculativeStats, StepTiming, TokenTiming,
};
use crate::profiler::system::{ResourceSnapshot, SystemMonitor};
use std::time::Instant;

pub struct EngineProfiler {
    model_path: String,
    device_name: String,
    start_snapshot: ResourceSnapshot,
    start_instant: Instant,
    tokenization_duration_ms: f64,
    prompt_tokens: usize,
    step_start: Option<Instant>,
    token_timings: Vec<TokenTiming>,
    step_timings: Vec<StepTiming>,
    step_counter: usize,
}

impl EngineProfiler {
    pub fn new(model_path: impl Into<String>, device_name: impl Into<String>) -> Self {
        let start_snapshot = SystemMonitor::snapshot();
        let start_instant = Instant::now();

        Self {
            model_path: model_path.into(),
            device_name: device_name.into(),
            start_snapshot,
            start_instant,
            tokenization_duration_ms: 0.0,
            prompt_tokens: 0,
            step_start: None,
            token_timings: Vec::new(),
            step_timings: Vec::new(),
            step_counter: 0,
        }
    }

    pub fn record_tokenization(&mut self, prompt_tokens: usize, duration: std::time::Duration) {
        self.prompt_tokens = prompt_tokens;
        self.tokenization_duration_ms = duration.as_secs_f64() * 1000.0;
    }

    pub fn start_step(&mut self) {
        self.step_start = Some(Instant::now());
    }

    pub fn record_batch_step(
        &mut self,
        tokens: &[(u32, String)],
        draft_tokens_count: Option<usize>,
    ) {
        if let Some(start) = self.step_start.take() {
            let step_duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            let is_prefill = self.step_counter == 0;
            let num_tokens = tokens.len().max(1);
            let per_token_latency_ms = step_duration_ms / num_tokens as f64;

            self.step_timings.push(StepTiming {
                step: self.step_counter,
                duration_ms: step_duration_ms,
                tokens_emitted: tokens.len(),
                is_prefill,
                draft_tokens_count,
            });

            for (token_id, token_str) in tokens {
                self.token_timings.push(TokenTiming {
                    step: self.step_counter,
                    token_id: *token_id,
                    token_str: token_str.clone(),
                    latency_ms: per_token_latency_ms,
                    is_prefill,
                    draft_tokens_count,
                });
            }

            self.step_counter += 1;
        }
    }

    pub fn end_step(
        &mut self,
        token_id: u32,
        token_str: String,
        draft_tokens_count: Option<usize>,
    ) {
        self.record_batch_step(&[(token_id, token_str)], draft_tokens_count);
    }

    pub fn finish(self, speculative: Option<SpeculativeStats>) -> InferenceProfile {
        let end_snapshot = SystemMonitor::snapshot();
        let resource_diff = SystemMonitor::diff(&self.start_snapshot, &end_snapshot);

        let generated_tokens = self.token_timings.len();
        let total_tokens = self.prompt_tokens + generated_tokens;

        let ttft_ms = self
            .step_timings
            .first()
            .map(|t| t.duration_ms)
            .unwrap_or(0.0);

        let prefill_throughput_tok_s = if ttft_ms > 0.0 {
            (self.prompt_tokens as f64) / (ttft_ms / 1000.0)
        } else {
            0.0
        };

        // Decode timings: all tokens after prefill step
        let prefill_token_count = self
            .step_timings
            .first()
            .map(|s| s.tokens_emitted)
            .unwrap_or(1);

        let decode_latencies_ms: Vec<f64> = self
            .token_timings
            .iter()
            .skip(prefill_token_count)
            .map(|t| t.latency_ms)
            .collect();

        let itl_stats = LatencyStats::compute(&decode_latencies_ms);

        let decode_step_latencies_ms: Vec<f64> = self
            .step_timings
            .iter()
            .skip(1)
            .map(|s| s.duration_ms)
            .collect();
        let step_latency_stats = LatencyStats::compute(&decode_step_latencies_ms);

        let decode_time_ms: f64 = decode_step_latencies_ms.iter().sum();
        let decode_throughput_tok_s = if decode_time_ms > 0.0 {
            (decode_latencies_ms.len() as f64) / (decode_time_ms / 1000.0)
        } else {
            0.0
        };

        let total_inference_time_ms: f64 = self.step_timings.iter().map(|t| t.duration_ms).sum();
        let end_to_end_time_ms = self.start_instant.elapsed().as_secs_f64() * 1000.0;

        let overall_throughput_tok_s = if total_inference_time_ms > 0.0 {
            (generated_tokens as f64) / (total_inference_time_ms / 1000.0)
        } else {
            0.0
        };

        #[cfg(feature = "metal")]
        let (pool_cap, pool_peak, pool_active) = {
            if let Some(state) = crate::metal::state::global_metal_state() {
                let stats = state.alloc.stats();
                (
                    Some(stats.capacity_bytes as f64 / (1024.0 * 1024.0)),
                    Some(stats.peak_offset as f64 / (1024.0 * 1024.0)),
                    Some(stats.active_bytes as f64 / (1024.0 * 1024.0)),
                )
            } else {
                (None, None, None)
            }
        };

        let memory = MemoryStats {
            host_initial_rss_mb: self.start_snapshot.rss_bytes as f64 / (1024.0 * 1024.0),
            host_peak_rss_mb: resource_diff.peak_rss_mb,
            host_delta_rss_mb: resource_diff.delta_rss_mb,
            #[cfg(feature = "metal")]
            metal_pool_capacity_mb: pool_cap,
            #[cfg(feature = "metal")]
            metal_pool_peak_mb: pool_peak,
            #[cfg(feature = "metal")]
            metal_pool_active_mb: pool_active,
        };

        InferenceProfile {
            model_path: self.model_path,
            device: self.device_name,
            prompt_tokens: self.prompt_tokens,
            generated_tokens,
            total_tokens,
            tokenization_time_ms: self.tokenization_duration_ms,
            ttft_ms,
            prefill_throughput_tok_s,
            decode_time_ms,
            decode_throughput_tok_s,
            total_inference_time_ms,
            end_to_end_time_ms,
            overall_throughput_tok_s,
            itl_stats,
            step_latency_stats,
            speculative,
            resource_diff,
            memory,
            token_timings: self.token_timings,
            step_timings: self.step_timings,
            multi_run: None,
        }
    }
}
