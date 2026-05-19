use axiom::core::backend::CandleBackend;
use axiom::core::device::Device;
use axiom::inference::engine::Engine;
use axiom::inference::sampler::SamplerConfig;
use axiom::tokenizer::tokenizer::{EncodeOptions, Tokenizer};
use axiom::weights::loader::load_from_gguf;
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

    println!("Axiom Smoke Test");
    println!("Model    : {}", gguf_path);
    println!("Tokenizer: {}", tokenizer_path);
    println!("Prompt   : {:?}", prompt);
    println!("Max new  : {}", max_new_tokens);
    println!("---");

    // ── tokenizer ──
    print!("Loading tokenizer... ");
    std::io::stdout().flush().unwrap();
    let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");
    println!("ok  (vocab {})", tokenizer.vocab().size());

    // ── model ──
    print!("Loading model (this takes a few seconds)... ");
    std::io::stdout().flush().unwrap();
    let device = Device::Cpu;
    let model = load_from_gguf::<CandleBackend>(Path::new(&gguf_path), &device)
        .expect("failed to load model");
    println!("ok");

    let vocab_size = tokenizer.vocab().size();

    // ── engine ──
    let sampler_config = SamplerConfig {
        temperature: 0.5,
        top_p: Some(0.9),
        top_k: Some(50),
        seed: Some(42),
        max_new_tokens,
        repetition_penalty: 1.4,
        vocab_size: Some(vocab_size),
    };

    let mut engine = Engine::new(model, tokenizer, sampler_config, 1, device);

    // ── encode prompt ──
    let session_id = engine
        .submit_text(
            &prompt,
            max_new_tokens,
            EncodeOptions {
                add_bos: true,
                add_eos: false,
            },
        )
        .expect("failed to submit prompt");

    println!("Output   : {}", prompt);
    print!("          ");
    std::io::stdout().flush().unwrap();

    // ── generation loop ──
    let mut steps = 0;
    let start = std::time::Instant::now();

    loop {
        let results = engine.step().expect("step failed");

        for (sid, token) in &results {
            if *sid == session_id {
                let text = engine.tokenizer().decode(&[*token as usize]);
                std::io::stdout().flush().unwrap();
                steps += 1;
            }
        }

        // stop if batch is empty
        if engine.batch.active_sessions().is_empty() {
            break;
        }

        // safety stop
        if steps >= max_new_tokens {
            break;
        }
    }

    let elapsed = start.elapsed();
    let tok_per_sec = steps as f64 / elapsed.as_secs_f64();

    println!();
    println!("---");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        steps,
        elapsed.as_secs_f64(),
        tok_per_sec
    );

    // ── full decoded output ──
    println!();
    println!("Full output");
    let full = engine.decode_output(session_id).unwrap_or_default();
    println!("{}{}", prompt, full);
}
