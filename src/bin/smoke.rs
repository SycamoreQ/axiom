#[cfg(not(feature = "metal"))]
use axiom::core::backend::CandleBackend;
#[cfg(feature = "metal")]
use axiom::core::backend::MetalBackend;
use axiom::core::device::Device;
use axiom::inference::engine::Engine;
use axiom::inference::draft::DraftModel;
use axiom::inference::sampler::{Sampler, SamplerConfig};
use axiom::tokenizer::tokenizer::{EncodeOptions, Tokenizer};
use axiom::weights::loader::load_from_gguf;
use axiom::weights::loader::load_from_gguf_qwen3moe;
use std::io::Write;
use std::path::Path;

fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let quiet = raw_args.iter().any(|s| s == "--quiet" || s == "-q");
    let positional: Vec<String> = raw_args.into_iter().filter(|s| !s.starts_with('-')).collect();

    let gguf_path = positional
        .get(0)
        .cloned()
        .unwrap_or_else(|| "testdata/tinyllama.gguf".to_string());
    let tokenizer_path = positional
        .get(1)
        .cloned()
        .unwrap_or_else(|| "testdata/tokenizer.json".to_string());
    let prompt = positional
        .get(2)
        .cloned()
        .unwrap_or_else(|| "The quick brown fox".to_string());
    let max_new_tokens: usize = positional
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let temperature: f32 = positional
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let draft_path = positional
        .get(5)
        .cloned()
        .filter(|s| Path::new(s).exists())
        .or_else(|| std::env::var("AXIOM_DRAFT_MODEL").ok().filter(|s| Path::new(s).exists()));
    let gamma: usize = positional
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    if !quiet {
        println!("Axiom Inference Engine");
        println!("Model    : {}", gguf_path);
        if let Some(ref d) = draft_path {
            println!("Draft    : {} (gamma: {})", d, gamma);
        }
        println!("Tokenizer: {}", tokenizer_path);
        println!("Prompt   : {:?}", prompt);
        println!("Max new  : {}", max_new_tokens);

        #[cfg(feature = "metal")]
        println!("Backend  : Metal (Apple Silicon)");
        #[cfg(not(feature = "metal"))]
        println!("Backend  : CPU (Candle)");
        println!("---");
        std::io::stdout().flush().unwrap();
    }

    // tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");
    println!("Tokenizer: ok (vocab {})", tokenizer.vocab().size());
    // model + engine — backend selected at compile time
    #[cfg(feature = "metal")]
    let engine = {
        print!("Initializing Metal... ");
        std::io::stdout().flush().unwrap();

        let pool_size = 512usize * 1024 * 1024;
        println!("Metal pool: {} MB", pool_size / 1024 / 1024);
        axiom::metal::state::init_global_metal_state(pool_size)
            .expect("failed to initialize Metal state");
        println!("ok");

        let device = Device::Metal(0);
        print!("Loading model... ");
        std::io::stdout().flush().unwrap();

        let mut model = if gguf_path.contains("moe") || gguf_path.contains("MoE") || gguf_path.contains("A3B") {
            load_from_gguf_qwen3moe::<MetalBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        } else {
            load_from_gguf::<MetalBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        };
        model
            .prepare_metal()
            .expect("failed to prepare metal weights");

        println!("ok");

        let vocab_size = model.config().vocab_size;
        let sampler_config = SamplerConfig {
            temperature,
            top_p: Some(0.9),
            top_k: Some(50),
            seed: Some(42),
            max_new_tokens,
            repetition_penalty: 1.0,
            vocab_size: Some(vocab_size),
            no_repeat_ngram_size: Some(3),
        };

        let mut engine = Engine::<MetalBackend>::new(model, tokenizer, sampler_config, 1, device.clone());
        if let Some(ref dpath) = draft_path {
            print!("Loading draft model from {}... ", dpath);
            std::io::stdout().flush().unwrap();
            let mut draft_model = if dpath.contains("moe") || dpath.contains("MoE") || dpath.contains("A3B") {
                load_from_gguf_qwen3moe::<MetalBackend>(Path::new(dpath), &device)
                    .expect("failed to load draft model")
            } else {
                load_from_gguf::<MetalBackend>(Path::new(dpath), &device)
                    .expect("failed to load draft model")
            };
            draft_model.prepare_metal().expect("failed to prepare draft metal weights");
            println!("ok");
            let draft_sampler = Sampler::new(SamplerConfig {
                temperature,
                top_p: Some(0.9),
                top_k: Some(50),
                seed: Some(42),
                max_new_tokens,
                repetition_penalty: 1.0,
                vocab_size: Some(draft_model.config().vocab_size),
                no_repeat_ngram_size: Some(3),
            });
            let draft = DraftModel::new(draft_model, draft_sampler, gamma);
            engine = engine.with_draft_model(draft);
        }
        engine
    };

    #[cfg(not(feature = "metal"))]
    let engine = {
        let device = Device::Cpu;
        print!("Loading model... ");
        std::io::stdout().flush().unwrap();

        let model = if gguf_path.contains("moe") || gguf_path.contains("MoE") || gguf_path.contains("A3B") {
            load_from_gguf_qwen3moe::<CandleBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        } else {
            load_from_gguf::<CandleBackend>(Path::new(&gguf_path), &device)
                .expect("failed to load model")
        };
        println!("ok");

        let vocab_size = model.config().vocab_size;
        let sampler_config = SamplerConfig {
            temperature: 0.0,
            top_p: Some(0.9),
            top_k: Some(50),
            seed: Some(42),
            max_new_tokens,
            repetition_penalty: 1.3,
            vocab_size: Some(vocab_size),
            no_repeat_ngram_size: Some(3),
        };

        let mut engine = Engine::<CandleBackend>::new(model, tokenizer, sampler_config, 1, device.clone());
        if let Some(ref dpath) = draft_path {
            print!("Loading draft model from {}... ", dpath);
            std::io::stdout().flush().unwrap();
            let draft_model = if dpath.contains("moe") || dpath.contains("MoE") || dpath.contains("A3B") {
                load_from_gguf_qwen3moe::<CandleBackend>(Path::new(dpath), &device)
                    .expect("failed to load draft model")
            } else {
                load_from_gguf::<CandleBackend>(Path::new(dpath), &device)
                    .expect("failed to load draft model")
            };
            println!("ok");
            let draft_sampler = Sampler::new(SamplerConfig {
                temperature: 0.0,
                top_p: Some(0.9),
                top_k: Some(50),
                seed: Some(42),
                max_new_tokens,
                repetition_penalty: 1.3,
                vocab_size: Some(draft_model.config().vocab_size),
                no_repeat_ngram_size: Some(3),
            });
            let draft = DraftModel::new(draft_model, draft_sampler, gamma);
            engine = engine.with_draft_model(draft);
        }
        engine
    };

    let mut engine = engine;

    let im_end_id: Option<u32> = engine
        .tokenizer()
        .vocab()
        .token_to_id("<|im_end|>")
        .map(|id| id as u32);
    let eos_id: Option<u32> = engine.tokenizer().eos_id().map(|id| id as u32);

    let formatted_prompt = if im_end_id.is_some() {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        )
    } else {
        prompt.to_string()
    };

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

    println!("\nOutput:");
    print!("  ");
    std::io::stdout().flush().unwrap();

    let mut steps = 0;
    let start = std::time::Instant::now();
    let mut stop_reason = None;

    loop {
        let results = engine.step().expect("step failed");
        for (sid, token) in &results {
            if *sid == session_id {
                let t = *token as u32;
                if Some(t) == im_end_id || Some(t) == eos_id {
                    stop_reason = Some("Stop token generated");
                    break;
                }
                let text = engine.tokenizer().decode(&[*token as usize]);
                print!("{}", text);
                std::io::stdout().flush().unwrap();
                steps += 1;
            }
        }

        if stop_reason.is_some()
            || engine.batch.active_sessions().is_empty()
            || steps >= max_new_tokens
        {
            if let Some(reason) = stop_reason {
                println!("\n\n[{}]", reason);
            }
            break; // Breaks the outer 'loop'
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("---");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        steps,
        elapsed.as_secs_f64(),
        steps as f64 / elapsed.as_secs_f64()
    );
    if engine.draft_model().is_some() {
        println!(
            "Speculative Acceptance: {:.1}% ({}/{} draft tokens accepted)",
            engine.acceptance_rate() * 100.0,
            engine.accepted_total,
            engine.drafted_total,
        );
    }
}
