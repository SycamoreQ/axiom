#[cfg(not(feature = "metal"))]
use axiom::core::backend::CandleBackend;
#[cfg(feature = "metal")]
use axiom::core::backend::MetalBackend;
use axiom::core::device::Device;
use axiom::inference::draft::DraftModel;
use axiom::inference::engine::Engine;
use axiom::inference::sampler::{Sampler, SamplerConfig};
use axiom::profiler::{EngineProfiler, ProfilerReport};
use axiom::tokenizer::tokenizer::{EncodeOptions, Tokenizer};
use axiom::weights::loader::{load_from_gguf, load_from_gguf_qwen3moe};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn print_usage(prog: &str) {
    println!("Usage: {} [OPTIONS] <MODEL_GGUF> [TOKENIZER_JSON]", prog);
    println!();
    println!("Options:");
    println!("  --prompt <TEXT>        Prompt text to evaluate (default: sample prompt)");
    println!("  --max-tokens <N>       Number of new tokens to generate (default: 32)");
    println!("  --temperature <T>      Sampling temperature (default: 0.0)");
    println!("  --draft <MODEL_GGUF>   Draft model path for speculative decoding");
    println!("  --gamma <N>            Draft speculative lookahead tokens (default: 4)");
    println!("  --warmup <N>           Number of warmup passes before profiling (default: 1)");
    println!("  --runs <N>             Number of repeated profiling trials (default: 1)");
    println!("  --json <PATH>          Export profiling report to JSON file (for vLLM comparison)");
    println!("  --trace <PATH>         Export Chrome / Perfetto trace to file");
    println!("  --quiet, -q            Suppress streaming token text");
    println!("  --help, -h             Show this help message");
}

fn compute_stats(mut values: Vec<f64>) -> (f64, f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    let min = values[0];
    let max = values[n - 1];
    let median = if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    };
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = var.sqrt();
    (min, median, max, mean, std_dev)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let prog = &args[0];

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage(prog);
        return;
    }

    let mut gguf_path: Option<String> = None;
    let mut tokenizer_path = "testdata/tokenizer.json".to_string();
    let mut prompt =
        "Explain the fundamental principles of quantum computing in three clear bullet points."
            .to_string();
    let mut max_new_tokens: usize = 32;
    let mut temperature: f32 = 0.0;
    let mut draft_path: Option<String> = None;
    let mut gamma: usize = 4;
    let mut warmup: usize = 1;
    let mut runs: usize = 1;
    let mut json_out: Option<String> = None;
    #[allow(unused_mut)]
    let mut trace_out: Option<String> = None;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    i += 1;
                }
            }
            "--max-tokens" | "-n" => {
                if i + 1 < args.len() {
                    max_new_tokens = args[i + 1].parse().unwrap_or(32);
                    i += 1;
                }
            }
            "--temperature" | "-t" => {
                if i + 1 < args.len() {
                    temperature = args[i + 1].parse().unwrap_or(0.0);
                    i += 1;
                }
            }
            "--draft" => {
                if i + 1 < args.len() {
                    draft_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--gamma" => {
                if i + 1 < args.len() {
                    gamma = args[i + 1].parse().unwrap_or(4);
                    i += 1;
                }
            }
            "--warmup" => {
                if i + 1 < args.len() {
                    warmup = args[i + 1].parse().unwrap_or(1);
                    i += 1;
                }
            }
            "--runs" => {
                if i + 1 < args.len() {
                    runs = args[i + 1].parse().unwrap_or(1);
                    i += 1;
                }
            }
            "--json" => {
                if i + 1 < args.len() {
                    json_out = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--trace" => {
                if i + 1 < args.len() {
                    trace_out = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--quiet" | "-q" => {
                quiet = true;
            }
            val if !val.starts_with('-') => {
                if gguf_path.is_none() {
                    gguf_path = Some(val.to_string());
                } else {
                    tokenizer_path = val.to_string();
                }
            }
            other => {
                eprintln!("Unknown option: {}", other);
                print_usage(prog);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let gguf_path = match gguf_path {
        Some(p) => p,
        None => {
            eprintln!("Error: model GGUF path is required.");
            print_usage(prog);
            std::process::exit(1);
        }
    };

    if tokenizer_path == "testdata/tokenizer.json" && gguf_path.to_lowercase().contains("llama") {
        if Path::new("testdata/tokenizer_llama.json").exists() {
            tokenizer_path = "testdata/tokenizer_llama.json".to_string();
        }
    }

    #[cfg(feature = "profile")]
    let _trace_guard = if let Some(ref tpath) = trace_out {
        println!(
            "Tracing enabled: writing Perfetto/Chrome trace to {}",
            tpath
        );
        Some(axiom::profiler::init_chrome_tracing(tpath))
    } else {
        None
    };
    #[cfg(not(feature = "profile"))]
    if trace_out.is_some() {
        println!("Note: trace output requires compiling with `--features profile`");
    }

    println!("============================================================");
    println!("              AXIOM INFERENCE ENGINE PROFILER               ");
    println!("============================================================");
    println!("Model:      {}", gguf_path);
    if let Some(ref d) = draft_path {
        println!("Draft:      {} (gamma: {})", d, gamma);
    }
    println!("Tokenizer:  {}", tokenizer_path);
    println!("Tokens:     target {} tokens", max_new_tokens);
    println!("Sampling:   temperature={}", temperature);
    println!("Trials:     {} run(s) (warmup: {})", runs, warmup);

    #[cfg(feature = "metal")]
    let device_name = "Metal (Apple Silicon GPU)";
    #[cfg(not(feature = "metal"))]
    let device_name = "CPU (Candle)";
    println!("Device:     {}", device_name);
    println!("------------------------------------------------------------");

    // Initialize Tokenizer
    let tok_load_start = Instant::now();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");
    println!(
        "Tokenizer loaded in {:.2}ms (vocab size: {})",
        tok_load_start.elapsed().as_secs_f64() * 1000.0,
        tokenizer.vocab().size()
    );

    // Initialize Engine & Model
    #[cfg(feature = "metal")]
    let (mut engine, device_str) = {
        let pool_size = 1024usize * 1024 * 1024; // 1 GB pool
        axiom::metal::state::init_global_metal_state(pool_size)
            .expect("failed to initialize Metal state");

        let device = Device::Metal(0);
        print!("Loading target model into Metal memory... ");
        std::io::stdout().flush().unwrap();
        let load_start = Instant::now();

        let mut model = if gguf_path.contains("moe")
            || gguf_path.contains("MoE")
            || gguf_path.contains("A3B")
        {
            load_from_gguf_qwen3moe::<MetalBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        } else {
            load_from_gguf::<MetalBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        };
        model
            .prepare_metal()
            .expect("failed to prepare metal weights");

        println!("done in {:.2}s", load_start.elapsed().as_secs_f64());

        let vocab_size = model.config().vocab_size;
        let sampler_config = SamplerConfig {
            temperature,
            top_p: Some(0.9),
            top_k: Some(50),
            seed: Some(42),
            max_new_tokens,
            repetition_penalty: 1.1,
            vocab_size: Some(vocab_size),
            no_repeat_ngram_size: Some(3),
        };

        let mut engine =
            Engine::<MetalBackend>::new(model, tokenizer, sampler_config, 1, device.clone());

        if let Some(ref dpath) = draft_path {
            print!("Loading draft model from {}... ", dpath);
            std::io::stdout().flush().unwrap();
            let d_load_start = Instant::now();
            let mut draft_model =
                if dpath.contains("moe") || dpath.contains("MoE") || dpath.contains("A3B") {
                    load_from_gguf_qwen3moe::<MetalBackend>(Path::new(dpath), &device)
                        .expect("failed to load draft model")
                } else {
                    load_from_gguf::<MetalBackend>(Path::new(dpath), &device)
                        .expect("failed to load draft model")
                };
            draft_model
                .prepare_metal()
                .expect("failed to prepare draft metal weights");
            println!("done in {:.2}s", d_load_start.elapsed().as_secs_f64());

            let draft_sampler = Sampler::new(SamplerConfig {
                temperature,
                top_p: Some(0.9),
                top_k: Some(50),
                seed: Some(42),
                max_new_tokens,
                repetition_penalty: 1.1,
                vocab_size: Some(draft_model.config().vocab_size),
                no_repeat_ngram_size: Some(3),
            });
            let draft = DraftModel::new(draft_model, draft_sampler, gamma);
            engine = engine.with_draft_model(draft);
        }

        (engine, "Apple Silicon Metal")
    };

    #[cfg(not(feature = "metal"))]
    let (mut engine, device_str) = {
        let device = Device::Cpu;
        print!("Loading target model into CPU memory... ");
        std::io::stdout().flush().unwrap();
        let load_start = Instant::now();

        let model = if gguf_path.contains("moe")
            || gguf_path.contains("MoE")
            || gguf_path.contains("A3B")
        {
            load_from_gguf_qwen3moe::<CandleBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        } else {
            load_from_gguf::<CandleBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        };
        println!("done in {:.2}s", load_start.elapsed().as_secs_f64());

        let vocab_size = model.config().vocab_size;
        let sampler_config = SamplerConfig {
            temperature,
            top_p: Some(0.9),
            top_k: Some(50),
            seed: Some(42),
            max_new_tokens,
            repetition_penalty: 1.1,
            vocab_size: Some(vocab_size),
            no_repeat_ngram_size: Some(3),
        };

        let mut engine =
            Engine::<CandleBackend>::new(model, tokenizer, sampler_config, 1, device.clone());

        if let Some(ref dpath) = draft_path {
            print!("Loading draft model from {}... ", dpath);
            std::io::stdout().flush().unwrap();
            let draft_model =
                if dpath.contains("moe") || dpath.contains("MoE") || dpath.contains("A3B") {
                    load_from_gguf_qwen3moe::<CandleBackend>(Path::new(dpath), &device)
                        .expect("failed to load draft model")
                } else {
                    load_from_gguf::<CandleBackend>(Path::new(dpath), &device)
                        .expect("failed to load draft model")
                };
            println!("done");

            let draft_sampler = Sampler::new(SamplerConfig {
                temperature,
                top_p: Some(0.9),
                top_k: Some(50),
                seed: Some(42),
                max_new_tokens,
                repetition_penalty: 1.1,
                vocab_size: Some(draft_model.config().vocab_size),
                no_repeat_ngram_size: Some(3),
            });
            let draft = DraftModel::new(draft_model, draft_sampler, gamma);
            engine = engine.with_draft_model(draft);
        }

        (engine, "CPU (Candle)")
    };

    // Prompt formatting & Tokenization
    let im_end_id: Option<u32> = engine
        .tokenizer()
        .vocab()
        .token_to_id("<|im_end|>")
        .map(|id| id as u32);
    let eot_id: Option<u32> = engine
        .tokenizer()
        .vocab()
        .token_to_id("<|eot_id|>")
        .map(|id| id as u32);
    let eos_id: Option<u32> = engine.tokenizer().eos_id().map(|id| id as u32);

    let formatted_prompt = if im_end_id.is_some() {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        )
    } else if eot_id.is_some() {
        format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            prompt
        )
    } else {
        prompt.clone()
    };

    if warmup > 0 {
        print!("Running {} warmup pass(es)... ", warmup);
        std::io::stdout().flush().unwrap();
        let w_start = Instant::now();
        for _ in 0..warmup {
            let session_id = engine
                .submit_text(
                    &formatted_prompt,
                    max_new_tokens,
                    EncodeOptions {
                        add_bos: false,
                        add_eos: false,
                    },
                )
                .expect("failed to submit warmup prompt");

            let mut w_generated = 0;
            loop {
                let results = engine.step().expect("warmup step failed");
                let mut should_stop = false;
                for (sid, token) in &results {
                    if *sid == session_id {
                        let t = *token as u32;
                        w_generated += 1;
                        if Some(t) == im_end_id || Some(t) == eot_id || Some(t) == eos_id {
                            should_stop = true;
                            break;
                        }
                    }
                }
                if should_stop
                    || engine.batch.active_sessions().is_empty()
                    || w_generated >= max_new_tokens
                {
                    break;
                }
            }
            engine.reset();
        }
        println!("done in {:.2}ms", w_start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut profiles: Vec<axiom::profiler::InferenceProfile> = Vec::new();

    for r in 0..runs {
        if runs > 1 && !quiet {
            println!("\n>>> Trial {} / {} <<<", r + 1, runs);
        }

        let mut profiler = EngineProfiler::new(&gguf_path, device_str);

        let tok_start = Instant::now();
        let prompt_tokens = engine.tokenizer().encode(
            &formatted_prompt,
            EncodeOptions {
                add_bos: false,
                add_eos: false,
            },
        );
        let tok_duration = tok_start.elapsed();
        profiler.record_tokenization(prompt_tokens.len(), tok_duration);

        let session_id = engine
            .submit_text(
                &formatted_prompt,
                max_new_tokens,
                EncodeOptions {
                    add_bos: false,
                    add_eos: false,
                },
            )
            .expect("failed to submit prompt");

        if !quiet && (runs == 1 || r == 0) {
            println!();
            println!("Prompt: \"{}\"", prompt);
            println!("Output Generation:");
            print!("  ");
            std::io::stdout().flush().unwrap();
        }

        let mut total_generated = 0;
        let mut stop_reason = None;

        loop {
            profiler.start_step();
            let results = engine.step().expect("step failed");

            let mut step_tokens = Vec::new();
            for (sid, token) in &results {
                if *sid == session_id {
                    let t = *token as u32;
                    let text = engine.tokenizer().decode(&[*token as usize]);
                    if !quiet && (runs == 1 || r == 0) {
                        print!("{}", text);
                        std::io::stdout().flush().unwrap();
                    }

                    step_tokens.push((*token, text));
                    total_generated += 1;

                    if Some(t) == im_end_id || Some(t) == eot_id || Some(t) == eos_id {
                        stop_reason = Some("EOS / Stop Token");
                        break;
                    }
                }
            }

            if !step_tokens.is_empty() {
                profiler.record_batch_step(&step_tokens, None);
            }

            if stop_reason.is_some()
                || engine.batch.active_sessions().is_empty()
                || total_generated >= max_new_tokens
            {
                if !quiet && (runs == 1 || r == 0) {
                    if let Some(reason) = stop_reason {
                        println!("\n  [{}]", reason);
                    }
                }
                break;
            }
        }

        let speculative_stats = if engine.draft_model().is_some() {
            Some(axiom::profiler::metrics::SpeculativeStats {
                drafted_total: engine.drafted_total,
                accepted_total: engine.accepted_total,
                acceptance_rate_pct: (engine.acceptance_rate() * 100.0) as f64,
            })
        } else {
            None
        };

        let profile = profiler.finish(speculative_stats);
        profiles.push(profile);
        engine.reset();
    }

    let final_profile = if runs > 1 {
        let ttfts: Vec<f64> = profiles.iter().map(|p| p.ttft_ms).collect();
        let decode_tok_s: Vec<f64> = profiles.iter().map(|p| p.decode_throughput_tok_s).collect();
        let total_times: Vec<f64> = profiles.iter().map(|p| p.total_inference_time_ms).collect();

        let (ttft_min, ttft_med, ttft_max, _, ttft_sd) = compute_stats(ttfts);
        let (dec_min, dec_med, dec_max, _, dec_sd) = compute_stats(decode_tok_s);
        let (tot_min, tot_med, tot_max, _, tot_sd) = compute_stats(total_times);

        let summary = axiom::profiler::metrics::MultiRunSummary {
            num_runs: runs,
            ttft_ms_min: ttft_min,
            ttft_ms_median: ttft_med,
            ttft_ms_max: ttft_max,
            ttft_ms_std_dev: ttft_sd,
            decode_tok_s_min: dec_min,
            decode_tok_s_median: dec_med,
            decode_tok_s_max: dec_max,
            decode_tok_s_std_dev: dec_sd,
            total_ms_min: tot_min,
            total_ms_median: tot_med,
            total_ms_max: tot_max,
            total_ms_std_dev: tot_sd,
        };

        // Pick median run based on decode_throughput_tok_s closest to dec_med
        let mut sorted_profiles = profiles;
        sorted_profiles.sort_by(|a, b| {
            let diff_a = (a.decode_throughput_tok_s - dec_med).abs();
            let diff_b = (b.decode_throughput_tok_s - dec_med).abs();
            diff_a.partial_cmp(&diff_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut med_profile = sorted_profiles.into_iter().next().unwrap();
        med_profile.multi_run = Some(summary);
        med_profile
    } else {
        profiles.into_iter().next().unwrap()
    };

    // Print the full structured report
    ProfilerReport::print_summary(&final_profile);

    // Save JSON if requested
    if let Some(ref json_path) = json_out {
        if let Err(e) = ProfilerReport::export_json(&final_profile, json_path) {
            eprintln!("Failed to export JSON report to {}: {}", json_path, e);
        } else {
            println!("Exported profile report to JSON: {}", json_path);
        }
    }
}
