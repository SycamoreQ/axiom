fn main() {
    // only compile CUDA when the cuda feature is enabled
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        cc::Build::new()
            .cuda(true)
            .flag("-arch=sm_80") // Ampere — change for your GPU
            .flag("--expt-relaxed-constexpr")
            .flag("-O3")
            .file("kernels/src/fused_rms_norm.cu")
            .file("kernels/src/flash_attention_4.cu")
            .file("kernels/src/residual_attention.cu")
            .compile("axiom_kernels");

        println!("cargo:rerun-if-changed=kernels/src/fused_rms_norm.cu");
        println!("cargo:rerun-if-changed=kernels/src/flash_attention_4.cu");
        println!("cargo:rerun-if-changed=kernels/src/residual_attention.cu");
    }
}
