/*
Describes LoRA adapter
*/

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraConfig {
    pub rank: usize,                 // r — the low-rank dimension
    pub alpha: f32,                  // scaling factor, usually equals rank
    pub dropout: f32,                // 0.0 at inference
    pub target_modules: Vec<String>, // e.g. ["q_proj", "v_proj"]
}

impl LoraConfig {
    pub fn new(rank: usize, alpha: f32, target_modules: Vec<String>) -> Self {
        Self {
            rank: rank,
            alpha: alpha,
            dropout: 0.0,
            target_modules: target_modules,
        }
    }

    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    pub fn targets(&self, module_name: &str) -> bool {
        self.target_modules.contains(&module_name.to_owned())
    }
}

impl Default for LoraConfig {
    fn default() -> Self {
        let target_modules: Vec<String> = vec![
            "q_proj".to_string(),
            "v_proj".to_string(),
            "k_proj".to_string(),
            "o_proj".to_string(),
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "down_proj".to_string(),
        ];
        Self {
            rank: 16,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: target_modules,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdapterMeta {
    pub id: String,         // unique adapter identifier
    pub base_model: String, // e.g. "llama-3-8b"
    pub config: LoraConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_new() {
        let config = LoraConfig::new(8, 16.0, vec!["q_proj".to_string()]);
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.dropout, 0.0);
        assert_eq!(config.target_modules.len(), 1);
    }

    #[test]
    fn test_lora_config_scaling() {
        let config = LoraConfig::new(16, 16.0, vec![]);
        assert!((config.scaling() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lora_config_scaling_half() {
        let config = LoraConfig::new(16, 8.0, vec![]);
        assert!((config.scaling() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_lora_config_targets_true() {
        let config = LoraConfig::new(16, 16.0, vec!["q_proj".to_string(), "v_proj".to_string()]);
        assert!(config.targets("q_proj"));
        assert!(config.targets("v_proj"));
    }

    #[test]
    fn test_lora_config_targets_false() {
        let config = LoraConfig::new(16, 16.0, vec!["q_proj".to_string()]);
        assert!(!config.targets("k_proj"));
        assert!(!config.targets("ffn_gate"));
    }

    #[test]
    fn test_lora_config_default_rank() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.dropout, 0.0);
    }

    #[test]
    fn test_lora_config_default_targets_all_attention() {
        let config = LoraConfig::default();
        assert!(config.targets("q_proj"));
        assert!(config.targets("v_proj"));
        assert!(config.targets("k_proj"));
        assert!(config.targets("o_proj"));
    }

    #[test]
    fn test_lora_config_serialization() {
        let config = LoraConfig::new(8, 16.0, vec!["q_proj".to_string()]);
        let json = serde_json::to_string(&config).unwrap();
        let back: LoraConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.rank, 8);
        assert_eq!(back.alpha, 16.0);
    }

    #[test]
    fn test_adapter_meta_fields() {
        let meta = AdapterMeta {
            id: "adapter-1".to_string(),
            base_model: "llama-3-8b".to_string(),
            config: LoraConfig::default(),
        };
        assert_eq!(meta.id, "adapter-1");
        assert_eq!(meta.base_model, "llama-3-8b");
        assert_eq!(meta.config.rank, 16);
    }
}
