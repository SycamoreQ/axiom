use axiom::weights::gguf::GgufFile;
use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inspect_gguf <path.gguf>");
    let gguf = GgufFile::from_file(Path::new(&path)).expect("failed to parse gguf");

    println!("=== architecture ===");
    println!("{:?}", gguf.metadata.get("general.architecture"));
    println!("{:?}", gguf.tensors.contains_key("output.weight"));

    println!("\n=== moe-relevant metadata ===");
    let mut keys: Vec<&String> = gguf
        .metadata
        .keys()
        .filter(|k| k.starts_with("qwen") || k.starts_with("general.") || k.starts_with("llama."))
        .collect();
    keys.sort();
    for k in keys {
        println!("{k} = {:?}", gguf.metadata[k]);
    }

    println!("\n=== non-blk tensors ===");
    for (n, t) in &gguf.tensors {
        if !n.starts_with("blk.") {
            println!("{n}: {:?} ({:?})", t.shape, t.dtype);
        }
    }

    if let Some(bias_data) = gguf.get_tensor_data("blk.0.attn_q.bias") {
        let floats: Vec<f32> = bias_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        println!("blk.0.attn_q.bias (len {}): {:?}", floats.len(), &floats[..10]);
    }
}
