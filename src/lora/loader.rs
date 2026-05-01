use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::lora::config::AdapterMeta;
use crate::weights::gguf::GgufFile;
use crate::weights::loader::{self, LoaderError};
use std::collections::HashMap;
use std::path::Path;

/*
Loads LoRA adapter weights from a file and returns tensors keyed by module name.
*/

pub struct LoadedAdapter<B: Backend> {
    pub meta: AdapterMeta,
    // maps "blk.{i}.{module}" → (lora_a, lora_b)
    pub weights: HashMap<String, (B::Tensor, B::Tensor)>,
}

pub fn load_adapter_from_gguf<B: Backend>(
    path: &Path,
    device: &Device,
) -> Result<LoadedAdapter<B>, LoaderError> {
    let gguf = GgufFile::from_file(path)?;
    let lora_config = crate::lora::config::LoraConfig::default();
    let meta = crate::lora::config::AdapterMeta {
        id: gguf
            .get_string("adapter.id")
            .unwrap_or("unknown")
            .to_string(),
        base_model: gguf
            .get_string("general.base_model")
            .or_else(|| gguf.get_string("general.architecture"))
            .unwrap_or("unknown")
            .to_string(),
        config: lora_config,
    };

    let mut paired_tensors: HashMap<String, (Option<B::Tensor>, Option<B::Tensor>)> =
        HashMap::new();

    for (name, info) in &gguf.tensors {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() < 3 {
            continue;
        }

        let base_name = parts[..parts.len() - 2].join(".");
        let lora_type = parts[parts.len() - 2];

        let entry = paired_tensors.entry(base_name).or_insert((None, None));
        let raw_data = gguf
            .get_tensor_data(name)
            .ok_or(LoaderError::MissingTensor("tensor missing".to_owned()))?;
        let tensor = loader::load_bytes_as_tensor::<B>(raw_data, info, device)?; // Reuse your existing loader helper

        match lora_type {
            "lora_a" => entry.0 = Some(tensor),
            "lora_b" => entry.1 = Some(tensor),
            _ => continue,
        }
    }

    // 3. Finalize into LoadedAdapter
    let mut weights = HashMap::new();
    for (name, (a, b)) in paired_tensors {
        if let (Some(a_tensor), Some(b_tensor)) = (a, b) {
            weights.insert(name, (a_tensor, b_tensor));
        }
    }

    Ok(LoadedAdapter { meta, weights })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_loaded_adapter_empty_weights() {
        let meta = crate::lora::config::AdapterMeta {
            id: "test".to_string(),
            base_model: "llama".to_string(),
            config: crate::lora::config::LoraConfig::default(),
        };
        let adapter = LoadedAdapter::<CandleBackend> {
            meta,
            weights: HashMap::new(),
        };
        assert!(adapter.weights.is_empty());
        assert_eq!(adapter.meta.id, "test");
    }

    #[test]
    fn test_loaded_adapter_weight_lookup() {
        use crate::core::backend::CandleTensor;
        use crate::core::dtype::DType;
        use crate::core::shape::Shape;
        use crate::core::tensor::TensorOps;

        let meta = crate::lora::config::AdapterMeta {
            id: "test".to_string(),
            base_model: "llama".to_string(),
            config: crate::lora::config::LoraConfig::default(),
        };
        let mut weights = HashMap::new();
        let a = CandleTensor::zeros(&Shape::new(&[8, 32]), DType::F32, &cpu()).unwrap();
        let b = CandleTensor::zeros(&Shape::new(&[64, 8]), DType::F32, &cpu()).unwrap();
        weights.insert("blk.0.attn_q".to_string(), (a, b));

        let adapter = LoadedAdapter::<CandleBackend> { meta, weights };
        assert!(adapter.weights.contains_key("blk.0.attn_q"));
        assert!(!adapter.weights.contains_key("blk.0.attn_v"));
    }

    #[test]
    fn test_load_adapter_from_real_gguf() {
        let path = Path::new("testdata/tinyllama-lora.gguf");
        if !path.exists() {
            return;
        }
        let result = load_adapter_from_gguf::<CandleBackend>(path, &cpu());
        assert!(result.is_ok(), "load failed: {:?}", result.err());
        let adapter = result.unwrap();
        assert!(!adapter.meta.id.is_empty());
    }
}
