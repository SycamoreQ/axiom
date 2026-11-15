use crate::profiler::system::ResourceDiff;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: usize,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub std_dev_ms: f64,
    pub tokens_per_sec: f64,
}

impl LatencyStats {
    pub fn compute(latencies_ms: &[f64]) -> Self {
        if latencies_ms.is_empty() {
            return Self {
                count: 0,
                mean_ms: 0.0,
                median_ms: 0.0,
                p90_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                std_dev_ms: 0.0,
                tokens_per_sec: 0.0,
            };
        }

        let mut sorted = latencies_ms.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean_ms = sum / count as f64;

        let percentile = |p: f64| -> f64 {
            if count == 1 {
                return sorted[0];
            }
            let idx = (p * (count - 1) as f64).round() as usize;
            sorted[idx.min(count - 1)]
        };

        let median_ms = percentile(0.50);
        let p90_ms = percentile(0.90);
        let p95_ms = percentile(0.95);
        let p99_ms = percentile(0.99);
        let min_ms = sorted[0];
        let max_ms = sorted[count - 1];

        let variance: f64 =
            sorted.iter().map(|&x| (x - mean_ms).powi(2)).sum::<f64>() / count as f64;
        let std_dev_ms = variance.sqrt();

        let total_time_sec = sum / 1000.0;
        let tokens_per_sec = if total_time_sec > 0.0 {
            count as f64 / total_time_sec
        } else {
            0.0
        };

        Self {
            count,
            mean_ms,
            median_ms,
            p90_ms,
            p95_ms,
            p99_ms,
            min_ms,
            max_ms,
            std_dev_ms,
            tokens_per_sec,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTiming {
    pub step: usize,
    pub token_id: u32,
    pub token_str: String,
    pub latency_ms: f64,
    pub is_prefill: bool,
    pub draft_tokens_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepTiming {
    pub step: usize,
    pub duration_ms: f64,
    pub tokens_emitted: usize,
    pub is_prefill: bool,
    pub draft_tokens_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeStats {
    pub drafted_total: usize,
    pub accepted_total: usize,
    pub acceptance_rate_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub host_initial_rss_mb: f64,
    pub host_peak_rss_mb: f64,
    pub host_delta_rss_mb: f64,
    #[cfg(feature = "metal")]
    pub metal_pool_capacity_mb: Option<f64>,
    #[cfg(feature = "metal")]
    pub metal_pool_peak_mb: Option<f64>,
    #[cfg(feature = "metal")]
    pub metal_pool_active_mb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRunSummary {
    pub num_runs: usize,
    pub ttft_ms_min: f64,
    pub ttft_ms_median: f64,
    pub ttft_ms_max: f64,
    pub ttft_ms_std_dev: f64,
    pub decode_tok_s_min: f64,
    pub decode_tok_s_median: f64,
    pub decode_tok_s_max: f64,
    pub decode_tok_s_std_dev: f64,
    pub total_ms_min: f64,
    pub total_ms_median: f64,
    pub total_ms_max: f64,
    pub total_ms_std_dev: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceProfile {
    pub model_path: String,
    pub device: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_tokens: usize,

    // Latency & throughput milestones
    pub tokenization_time_ms: f64,
    pub ttft_ms: f64, // Time to first token (prefill latency)
    pub prefill_throughput_tok_s: f64,
    pub decode_time_ms: f64,
    pub decode_throughput_tok_s: f64,
    pub total_inference_time_ms: f64,
    pub end_to_end_time_ms: f64,
    pub overall_throughput_tok_s: f64,

    // Detailed decode ITL distributions
    pub itl_stats: LatencyStats,
    pub step_latency_stats: LatencyStats,

    // Speculative stats
    pub speculative: Option<SpeculativeStats>,

    // Resources & CPU usage
    pub resource_diff: ResourceDiff,
    pub memory: MemoryStats,

    // Per-token breakdown
    pub token_timings: Vec<TokenTiming>,
    // Per-step breakdown
    pub step_timings: Vec<StepTiming>,

    // Multi-run statistics (if multiple runs were evaluated)
    pub multi_run: Option<MultiRunSummary>,
}
