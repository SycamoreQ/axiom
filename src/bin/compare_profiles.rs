use axiom::profiler::metrics::InferenceProfile;
use axiom::profiler::ProfilerReport;
use std::fs::File;
use std::path::Path;

fn load_profile(path: &str) -> InferenceProfile {
    let file = File::open(Path::new(path)).unwrap_or_else(|e| {
        eprintln!("Failed to open {}: {}", path, e);
        std::process::exit(1);
    });
    serde_json::from_reader(file).unwrap_or_else(|e| {
        eprintln!("Failed to parse JSON in {}: {}", path, e);
        std::process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <PROFILE_A_JSON> <PROFILE_B_JSON> [LABEL_A] [LABEL_B]", args[0]);
        println!("Example: {} axiom_qwen.json vllm_qwen.json 'Axiom (Metal)' 'vLLM (Baseline)'", args[0]);
        std::process::exit(1);
    }

    let file_a = &args[1];
    let file_b = &args[2];
    let label_a = args.get(3).map(|s| s.as_str()).unwrap_or(file_a);
    let label_b = args.get(4).map(|s| s.as_str()).unwrap_or(file_b);

    let profile_a = load_profile(file_a);
    let profile_b = load_profile(file_b);

    ProfilerReport::compare(label_a, &profile_a, label_b, &profile_b);
}

