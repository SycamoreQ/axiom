use axiom::weights::gguf::GgufFile;
use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: inspect_gguf <path.gguf>");
    let gguf = GgufFile::from_file(Path::new(&path)).expect("failed to parse gguf");

    println!("=== architecture ===");
    println!("{:?}", gguf.metadata.get("general.architecture"));

    println!("\n=== moe-relevant metadata ===");
    let mut keys: Vec<&String> = gguf
        .metadata
        .keys()
        .filter(|k| k.starts_with("qwen3moe.") || k.starts_with("general."))
        .collect();
    keys.sort();
    for k in keys {
        println!("{k} = {:?}", gguf.metadata[k]);
    }

    println!("\n=== layer-0 tensor names/shapes ===");
    let mut names: Vec<&String> = gguf
        .tensors
        .keys()
        .filter(|n| n.starts_with("blk.0."))
        .collect();
    names.sort();
    for n in names {
        let t = &gguf.tensors[n];
        println!("{n}: {:?} ({:?})", t.shape, t.dtype);
    }
}
