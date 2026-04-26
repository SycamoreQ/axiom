use std::env;
use std::path::PathBuf;
use std::process::Command;

fn cu_sources() -> Vec<PathBuf> {
    let entries: Vec<PathBuf> = std::fs::read_dir("kernel/src")
        .unwrap()
        .filter_map(|res| res.ok())
        .filter(|entry| entry.path().is_file())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "cu"))
        .map(|entry| entry.path())
        .collect();
    entries
}

fn nvcc_path() -> PathBuf {
    if let Ok(val) = env::var("CUDACXX") {
        PathBuf::from(val)
    } else {
        PathBuf::from("nvcc")
    }
}

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    let out_dir = env::var("OUT_DIR").unwrap();
    let ptx_path = PathBuf::from(&out_dir).join("axiom_kernels.ptx");
    let sources = cu_sources();
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
    let mut command = Command::new(nvcc_path);
    command
        .arg("-ptx")
        .arg(format!("-arch={}", arch))
        .arg("-code=sm_80,sm_89,sm_90,sm_100")
        .arg("--expt-relaxed-constexpr")
        .arg("-o")
        .arg(&ptx_path);
    for source in &sources {
        command.arg(source);
    }
    let output = command.output().expect("Failed to execute nvcc");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("nvcc compilation failed:\n{}", stderr);
    }
    println!("cargo:rustc-env=AXIOM_KERNELS_PTX={}", ptx_path.display());
    for source in &sources {
        println!("cargo:rerun-if-changed={}", source);
    }
}
