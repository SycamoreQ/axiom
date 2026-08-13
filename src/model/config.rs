use crate::core::dtype::DType;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

/*
APIs for reading config files of models.
 */

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing field: {0}")]
    MissingField(&'static str),
    #[error("invalid dtype: {0}")]
    InvalidDtype(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    //core dimensions
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // GQA — if equal to num_attention_heads, standard MHA
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    //norms and activations
    pub rms_norm_eps: f64,
    pub hidden_act: String, // "silu", "gelu"

    //RoPE
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,

    //MoE
    #[serde(default)]
    pub num_local_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub num_shared_experts: Option<usize>,
    #[serde(default)]
    pub expert_interval: Option<usize>, // every N layers is MoE, None means all layers
    #[serde(default)]
    pub prefetch_threshold: Option<f32>, // confidence threshold for pre-gating

    #[serde(default)]
    pub rope_freqs: Option<Vec<f32>>,
    //dtype
    #[serde(default = "default_dtype")]
    pub torch_dtype: String, // "float32", "float16", "bfloat16"

    //architecture tag
    #[serde(default)]
    pub architectures: Option<Vec<String>>, // e.g. ["LlamaForCausalLM"]
    #[serde(default)]
    pub model_type: Option<String>, // e.g. "llama", "mistral", "deepseek"

    pub lazy_moe: bool,
    pub head_dim_override: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type")]
    pub scaling_type: String, // "linear", "dynamic", "yarn"
    pub factor: f64,
}

fn default_dtype() -> String {
    "float32".to_string()
}

impl ModelConfig {
    // load from a config.json file
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let file = File::open(path)?;
        let config: Self = serde_json::from_reader(file)?;
        Ok(config)
    }

    // load from a model directory — looks for config.json inside
    pub fn from_dir(dir: &Path) -> Result<Self, ConfigError> {
        let config_path = dir.join("config.json");
        Self::from_file(&config_path)
    }

    // derived properties
    pub fn is_moe(&self) -> bool {
        matches!(self.num_local_experts, Some(n) if n > 1)
    }

    // hidden_size / num_attention_heads
    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
    // num_key_value_heads != num_attention_heads
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads != self.num_attention_heads
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if !self.is_moe() {
            return false;
        }
        match self.expert_interval {
            None => true, // all layers are MoE
            Some(n) => layer_idx % n == 0,
        }
    }

    pub fn dtype(&self) -> Result<DType, ConfigError> {
        match self.torch_dtype.as_str() {
            "float32" | "float" => Ok(DType::F32),
            "float16" | "half" => Ok(DType::F16),
            "bfloat16" => Ok(DType::BF16),
            other => Err(ConfigError::InvalidDtype(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn make_dense_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            vocab_size: 128256,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 500000.0,
            rope_freqs: None,
            rope_scaling: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            num_shared_experts: None,
            expert_interval: None,
            prefetch_threshold: None,
            torch_dtype: "bfloat16".to_string(),
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            model_type: Some("llama".to_string()),
            head_dim_override: None,
        }
    }

    fn make_moe_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 512,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            intermediate_size: 2048,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
            rope_freqs: None,
            rope_scaling: None,
            num_local_experts: Some(8),
            num_experts_per_tok: Some(2),
            num_shared_experts: Some(2),
            expert_interval: Some(2),
            prefetch_threshold: Some(0.3),
            torch_dtype: "bfloat16".to_string(),
            architectures: None,
            model_type: Some("deepseek".to_string()),
            head_dim_override: None,
        }
    }

    //is_moe

    #[test]
    fn test_is_moe_false_for_dense() {
        assert!(!make_dense_config().is_moe());
    }

    #[test]
    fn test_is_moe_true_for_moe() {
        assert!(make_moe_config().is_moe());
    }

    #[test]
    fn test_is_moe_false_for_single_expert() {
        let mut c = make_moe_config();
        c.num_local_experts = Some(1);
        assert!(!c.is_moe());
    }

    #[test]
    fn test_is_moe_false_for_none() {
        let mut c = make_moe_config();
        c.num_local_experts = None;
        assert!(!c.is_moe());
    }

    //head_dim

    #[test]
    fn test_head_dim() {
        let c = make_dense_config();
        assert_eq!(c.head_dim(), 4096 / 32);
    }

    //num_kv_groups

    #[test]
    fn test_num_kv_groups() {
        let c = make_dense_config();
        assert_eq!(c.num_kv_groups(), 32 / 8);
    }

    #[test]
    fn test_num_kv_groups_no_gqa() {
        let mut c = make_dense_config();
        c.num_key_value_heads = c.num_attention_heads;
        assert_eq!(c.num_kv_groups(), 1);
    }

    // --- is_gqa ---

    #[test]
    fn test_is_gqa_true() {
        assert!(make_dense_config().is_gqa());
    }

    #[test]
    fn test_is_gqa_false() {
        let mut c = make_dense_config();
        c.num_key_value_heads = c.num_attention_heads;
        assert!(!c.is_gqa());
    }

    // --- is_moe_layer ---

    #[test]
    fn test_is_moe_layer_dense_always_false() {
        let c = make_dense_config();
        for i in 0..32 {
            assert!(!c.is_moe_layer(i));
        }
    }

    #[test]
    fn test_is_moe_layer_with_interval() {
        let c = make_moe_config(); // expert_interval = 2
        assert!(c.is_moe_layer(0));
        assert!(!c.is_moe_layer(1));
        assert!(c.is_moe_layer(2));
        assert!(!c.is_moe_layer(3));
        assert!(c.is_moe_layer(4));
    }

    #[test]
    fn test_is_moe_layer_no_interval_all_moe() {
        let mut c = make_moe_config();
        c.expert_interval = None;
        for i in 0..8 {
            assert!(c.is_moe_layer(i));
        }
    }

    //dtype

    #[test]
    fn test_dtype_bfloat16() {
        assert_eq!(make_dense_config().dtype().unwrap(), DType::BF16);
    }

    #[test]
    fn test_dtype_float32() {
        let mut c = make_dense_config();
        c.torch_dtype = "float32".to_string();
        assert_eq!(c.dtype().unwrap(), DType::F32);
    }

    #[test]
    fn test_dtype_float16() {
        let mut c = make_dense_config();
        c.torch_dtype = "float16".to_string();
        assert_eq!(c.dtype().unwrap(), DType::F16);
    }

    #[test]
    fn test_dtype_aliases() {
        let mut c = make_dense_config();
        c.torch_dtype = "float".to_string();
        assert_eq!(c.dtype().unwrap(), DType::F32);
        c.torch_dtype = "half".to_string();
        assert_eq!(c.dtype().unwrap(), DType::F16);
    }

    #[test]
    fn test_dtype_invalid() {
        let mut c = make_dense_config();
        c.torch_dtype = "int8".to_string();
        assert!(c.dtype().is_err());
    }

    //file loading

    #[test]
    fn test_from_file_llama3() {
        let path = Path::new("testdata/config.json");
        if !path.exists() {
            return;
        }
        let config = ModelConfig::from_file(path).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert!(!config.is_moe());
        assert!(config.is_gqa());
    }

    #[test]
    fn test_from_dir() {
        let path = Path::new("testdata");
        if !path.join("config.json").exists() {
            return;
        }
        let config = ModelConfig::from_dir(path).unwrap();
        assert!(config.hidden_size > 0);
    }

    #[test]
    fn test_from_file_moe_json() {
        let path = Path::new("testdata/moe_config.json");
        if !path.exists() {
            return;
        }
        let config = ModelConfig::from_file(path).unwrap();
        assert!(config.is_moe());
        assert_eq!(config.num_local_experts, Some(8));
        assert_eq!(config.num_shared_experts, Some(2));
        assert_eq!(config.expert_interval, Some(2));
    }

    //rope scaling

    #[test]
    fn test_rope_scaling_none() {
        assert!(make_dense_config().rope_scaling.is_none());
    }

    #[test]
    fn test_rope_scaling_present() {
        let mut c = make_dense_config();
        c.rope_scaling = Some(RopeScaling {
            scaling_type: "linear".to_string(),
            factor: 8.0,
        });
        let rs = c.rope_scaling.unwrap();
        assert_eq!(rs.scaling_type, "linear");
        assert_eq!(rs.factor, 8.0);
    }
}
