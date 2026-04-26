use std::env;
use std::path::PathBuf;
use std::process::Command;

fn cu_sources() -> Vec<PathBuf> {
    let entries: Vec<PathBuf> = std::fs::read_dir("kernels/src")
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
    Ok(val)
}

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    let out_dir = env::var("OUT_DIR").unwrap();
    let ptx_path = PathBuf::from(&out_dir).join("axiom_kernels.ptx");
    let sources = vec![
        "kernels/src/fused_residual_rms_[norm.cu](http://norm.cu)",
        "kernels/src/flash_attention_[4.cu](http://4.cu)",
        "kernels/src/residual_[attention.cu](http://attention.cu)",
        "kernels/src/fused_[attention.cu](http://attention.cu)"
        "kernels/src/argmax_[f16.cu](http://f16.cu)"
        "kernels/src/copy_blocks_[f16.cu](http://f16.cu)"
        "kernels/src/flash_attention_[3.cu](http://3.cu)"
        "kernels/src/rotary_embedding_[f16.cu](http://f16.cu)"
        "kernels/src/rms_[norm.cu](http://norm.cu)"
        "kernels/src/reshape_and_cache_[f16.cu](http://f16.cu)"
    ];
    let mut command = Command::new("nvcc");
    command
        .arg("-ptx")
        .arg("-arch=sm_89")
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
