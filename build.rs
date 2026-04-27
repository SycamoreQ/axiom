use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn cu_sources() -> Vec<PathBuf> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir("kernel/src")
        .unwrap()
        .filter_map(|res| res.ok())
        .filter(|entry| entry.path().is_file())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "cu"))
        .map(|entry| entry.path())
        .collect();
    entries.sort();
    entries.dedup();
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
    let out_path = PathBuf::from(&out_dir);
    let final_ptx = out_path.join("axiom_kernels.ptx");

    let sources = cu_sources();

    // Convert sm_XX → compute_XX for PTX virtual arch
    let arch_raw = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
    let virtual_arch = if arch_raw.starts_with("sm_") {
        arch_raw.replace("sm_", "compute_")
    } else {
        arch_raw
    };

    let cuda_include =
        env::var("CUDA_INCLUDE").unwrap_or_else(|_| "/usr/local/cuda/include".to_string());

    let mut all_ptx = String::new();
    let mut first = true;

    for source in &sources {
        let stem = source.file_stem().unwrap().to_string_lossy();
        let ptx_out = out_path.join(format!("{}.ptx", stem));

        let mut command = Command::new(nvcc_path());
        command
            .arg("-ptx")
            .arg(format!("-arch={}", virtual_arch))
            .arg(format!("-I{}", cuda_include))
            .arg("--expt-relaxed-constexpr")
            .arg("-o")
            .arg(&ptx_out)
            .arg(source);

        let output = command.output().expect("Failed to execute nvcc");
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "nvcc compilation failed for {}:\n{}",
                source.display(),
                stderr
            );
        }

        let ptx_text = fs::read_to_string(&ptx_out)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", ptx_out.display(), e));

        if first {
            // Keep full first file including its .version/.target/.address_size header
            all_ptx.push_str(&ptx_text);
            first = false;
        } else {
            // Strip header from subsequent files:
            // skip leading comments, .version, .target, .address_size lines
            let body: String = ptx_text
                .lines()
                .skip_while(|l| {
                    let t = l.trim();
                    t.is_empty()
                        || t.starts_with("//")
                        || t.starts_with(".version")
                        || t.starts_with(".target")
                        || t.starts_with(".address_size")
                })
                .collect::<Vec<_>>()
                .join("\n");
            all_ptx.push('\n');
            all_ptx.push_str(&body);
        }

        all_ptx.push('\n');
    }

    fs::write(&final_ptx, &all_ptx).unwrap_or_else(|e| panic!("Failed to write final PTX: {}", e));

    println!("cargo:rustc-env=AXIOM_KERNELS_PTX={}", final_ptx.display());

    for source in &sources {
        println!("cargo:rerun-if-changed={}", source.display());
    }
}
