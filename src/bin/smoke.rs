#[cfg(not(feature = "metal"))]
use axiom::core::backend::CandleBackend;
#[cfg(feature = "metal")]
use axiom::core::backend::MetalBackend;
use axiom::core::device::Device;
use axiom::core::tensor::TensorOps;
use axiom::inference::engine::Engine;
use axiom::inference::sampler::SamplerConfig;
use axiom::tokenizer::tokenizer::{EncodeOptions, Tokenizer};
use axiom::weights::loader::load_from_gguf;
use axiom::weights::loader::load_from_gguf_qwen3moe;
use std::io::Write;
use std::path::Path;

fn main() {
    let gguf_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "testdata/tinyllama.gguf".to_string());
    let tokenizer_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "testdata/tokenizer.json".to_string());
    let prompt = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "The quick brown fox".to_string());
    let max_new_tokens: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let temperature: f32 = std::env::args()
        .nth(5)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    println!("Axiom Inference Engine");
    println!("Model    : {}", gguf_path);
    println!("Tokenizer: {}", tokenizer_path);
    println!("Prompt   : {:?}", prompt);
    println!("Max new  : {}", max_new_tokens);

    #[cfg(feature = "metal")]
    println!("Backend  : Metal (Apple Silicon)");
    #[cfg(not(feature = "metal"))]
    println!("Backend  : CPU (Candle)");
    println!("---");
    std::io::stdout().flush().unwrap();

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

        let mut model = load_from_gguf_qwen3moe::<MetalBackend>(Path::new(&gguf_path), &device)
            .expect("failed to load model");
        model
            .prepare_metal()
            .expect("failed to prepare metal weights");

        let vocab_size = model.config().vocab_size;
        let hidden = model.config().hidden_size;

        // pre-transpose, canonical [vocab, hidden] — row for a token is contiguous
        let raw = model.lm_head.weight().to_vec_f32().unwrap();
        // post-transpose+contiguous, [hidden, vocab] — same token is now a strided column
        let prepared = model
            .metal_lm_head_weight
            .as_ref()
            .unwrap()
            .to_vec_f32()
            .unwrap();

        println!("Ok");

        let embd = model.embedding.weight().to_vec_f32().unwrap();
        println!("token_embd[0..10]: {:?}", &embd[..10]);
        let n = embd.len();
        println!("token_embd[last 10]: {:?}", &embd[n - 10..]);

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

        Engine::<MetalBackend>::new(model, tokenizer, sampler_config, 1, device)
    };

    #[cfg(not(feature = "metal"))]
    let engine = {
        let device = Device::Cpu;
        print!("Loading model... ");
        std::io::stdout().flush().unwrap();

        let model = load_from_gguf::<CandleBackend>(Path::new(&gguf_path), &device)
            .expect("failed to load model");
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

        Engine::<CandleBackend>::new(model, tokenizer, sampler_config, 1, device)
    };

    let mut engine = engine;

    let im_end_id: u32 = engine
        .tokenizer()
        .encode(
            "<|im_end|>",
            EncodeOptions {
                add_bos: false,
                add_eos: false,
            },
        )
        .first()
        .copied()
        .expect("<|im_end|> not found in tokenizer vocab") as u32;

    let formatted_prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );

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
                if t == im_end_id {
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
}
