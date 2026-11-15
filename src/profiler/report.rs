use crate::profiler::metrics::InferenceProfile;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub struct ProfilerReport;

impl ProfilerReport {
    pub fn print_summary(profile: &InferenceProfile) {
        println!();
        println!("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓");
        println!("┃                        AXIOM INFERENCE PROFILE REPORT                    ┃");
        println!("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");
        println!(
            "┃  Model           : {:<52} ┃",
            truncate_str(&profile.model_path, 52)
        );
        println!(
            "┃  Device          : {:<52} ┃",
            truncate_str(&profile.device, 52)
        );
        println!(
            "┃  Prompt Tokens   : {:<12} Output Tokens: {:<12} Total: {:<7} ┃",
            profile.prompt_tokens, profile.generated_tokens, profile.total_tokens
        );
        println!("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");
        println!("┃  LATENCY & THROUGHPUT METRICS                                            ┃");
        println!("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");
        println!(
            "┃  Tokenization    : {:>8.2} ms                                            ┃",
            profile.tokenization_time_ms
        );
        println!(
            "┃  Prefill (TTFT)  : {:>8.2} ms  ({:>7.1} tok/s)                          ┃",
            profile.ttft_ms, profile.prefill_throughput_tok_s
        );
        println!(
            "┃  Decode Time     : {:>8.2} ms  ({:>7.1} tok/s)                          ┃",
            profile.decode_time_ms, profile.decode_throughput_tok_s
        );
        println!(
            "┃  Overall Gen     : {:>8.2} ms  ({:>7.1} tok/s)                          ┃",
            profile.total_inference_time_ms, profile.overall_throughput_tok_s
        );
        println!(
            "┃  End-to-End      : {:>8.2} ms                                            ┃",
            profile.end_to_end_time_ms
        );
        println!("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");
        println!("┃  INTER-TOKEN LATENCY (ITL / TPOT) DISTRIBUTION                           ┃");
        println!("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");
        println!(
            "┃  Mean: {:>6.2} ms  │  Median (P50): {:>6.2} ms  │  P90: {:>6.2} ms         ┃",
            profile.itl_stats.mean_ms, profile.itl_stats.median_ms, profile.itl_stats.p90_ms
        );
        println!(
            "┃  P95:  {:>6.2} ms  │  P99:          {:>6.2} ms  │  Min: {:>6.2} ms         ┃",
            profile.itl_stats.p95_ms, profile.itl_stats.p99_ms, profile.itl_stats.min_ms
        );
        println!(
            "┃  Max:  {:>6.2} ms  │  Std Dev:      {:>6.2} ms                            ┃",
            profile.itl_stats.max_ms, profile.itl_stats.std_dev_ms
        );

        if let Some(ref spec) = profile.speculative {
            println!(
                "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫"
            );
            println!(
                "┃  SPECULATIVE DECODING METRICS                                            ┃"
            );
            println!(
                "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫"
            );
            println!(
                "┃  Tokens Drafted  : {:<10} Tokens Accepted: {:<10}                     ┃",
                spec.drafted_total, spec.accepted_total
            );
            println!(
                "┃  Acceptance Rate : {:>6.1}%                                               ┃",
                spec.acceptance_rate_pct
            );
        }

        println!("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");
        println!("┃  RESOURCE & MEMORY UTILIZATION                                           ┃");
        println!("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫");
        println!(
            "┃  Host Peak RSS   : {:>8.2} MB  │  Delta RSS    : {:>8.2} MB              ┃",
            profile.memory.host_peak_rss_mb, profile.memory.host_delta_rss_mb
        );
        #[cfg(feature = "metal")]
        if let (Some(cap), Some(peak), Some(active)) = (
            profile.memory.metal_pool_capacity_mb,
            profile.memory.metal_pool_peak_mb,
            profile.memory.metal_pool_active_mb,
        ) {
            println!(
                "┃  Metal Pool Cap  : {:>8.2} MB  │  Pool Peak    : {:>8.2} MB (active: {:>5.1}M)┃",
                cap, peak, active
            );
        }
        println!(
            "┃  CPU Time (User) : {:>8.2} ms  │  CPU (System) : {:>8.2} ms              ┃",
            profile.resource_diff.user_cpu_ms, profile.resource_diff.system_cpu_ms
        );
        println!(
            "┃  CPU Utilization : {:>8.1} %                                              ┃",
            profile.resource_diff.cpu_utilization_pct
        );
        println!("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        println!();
    }

    pub fn export_json(profile: &InferenceProfile, path: impl AsRef<Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(profile)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn compare(name_a: &str, a: &InferenceProfile, name_b: &str, b: &InferenceProfile) {
        println!();
        println!("==========================================================================");
        println!("                       PERFORMANCE COMPARISON REPORT                      ");
        println!("==========================================================================");
        println!(
            "Metric                           | {:<18} | {:<18} | Speedup/Delta",
            name_a, name_b
        );
        println!("---------------------------------+--------------------+--------------------+--------------");

        let print_row_time = |label: &str, val_a: f64, val_b: f64| {
            let speedup = if val_b > 0.0 { val_a / val_b } else { 0.0 };
            println!(
                "{:<32} | {:>15.2} ms | {:>15.2} ms | {:>6.2}x ({:+.1}%)",
                label,
                val_a,
                val_b,
                speedup,
                ((val_a - val_b) / val_a) * 100.0
            );
        };

        let print_row_thru = |label: &str, val_a: f64, val_b: f64| {
            let ratio = if val_a > 0.0 { val_b / val_a } else { 0.0 };
            println!(
                "{:<32} | {:>13.1} tok/s | {:>13.1} tok/s | {:>6.2}x ({:+.1}%)",
                label,
                val_a,
                val_b,
                ratio,
                ((val_b - val_a) / val_a) * 100.0
            );
        };

        print_row_time("Prefill Latency (TTFT)", a.ttft_ms, b.ttft_ms);
        print_row_thru(
            "Prefill Throughput",
            a.prefill_throughput_tok_s,
            b.prefill_throughput_tok_s,
        );
        print_row_time("Decode Time", a.decode_time_ms, b.decode_time_ms);
        print_row_thru(
            "Decode Throughput",
            a.decode_throughput_tok_s,
            b.decode_throughput_tok_s,
        );
        print_row_time(
            "Mean Inter-Token Latency (ITL)",
            a.itl_stats.mean_ms,
            b.itl_stats.mean_ms,
        );
        print_row_time(
            "Median ITL (P50)",
            a.itl_stats.median_ms,
            b.itl_stats.median_ms,
        );
        print_row_time("P90 ITL", a.itl_stats.p90_ms, b.itl_stats.p90_ms);
        print_row_time("P99 ITL", a.itl_stats.p99_ms, b.itl_stats.p99_ms);
        print_row_thru(
            "Overall Generation Throughput",
            a.overall_throughput_tok_s,
            b.overall_throughput_tok_s,
        );

        println!("---------------------------------+--------------------+--------------------+--------------");
        println!(
            "{:<32} | {:>15.2} MB | {:>15.2} MB | {:>+.2} MB",
            "Host Peak RSS",
            a.memory.host_peak_rss_mb,
            b.memory.host_peak_rss_mb,
            b.memory.host_peak_rss_mb - a.memory.host_peak_rss_mb
        );
        println!(
            "{:<32} | {:>17.1}% | {:>17.1}% | {:>+.1}%",
            "CPU Utilization",
            a.resource_diff.cpu_utilization_pct,
            b.resource_diff.cpu_utilization_pct,
            b.resource_diff.cpu_utilization_pct - a.resource_diff.cpu_utilization_pct
        );
        println!("==========================================================================");
        println!();
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("...{}", &s[s.len() - (max_len - 3)..])
    } else {
        s.to_string()
    }
}
