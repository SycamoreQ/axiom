#[cfg(test)]
mod integration_tests {
    use crate::core::backend::CandleBackend;
    use crate::core::device::Device;
    use crate::core::tensor::TensorOps;
    use crate::model::config::ModelConfig;
    use crate::model::model::LlamaModel;
    use crate::tokenizer::tokenizer::{EncodeOptions, Tokenizer};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_small_config(vocab_size: usize) -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
            rope_freqs: None,
            rope_scaling: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            num_shared_experts: None,
            expert_interval: None,
            prefetch_threshold: None,
            torch_dtype: "float32".to_string(),
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
            model_type: Some("llama".to_string()),
        }
    }

    const TOKENIZER_PATH: &str = "testdata/tokenizer.json";

    // --- Tokenizer integration ---

    #[test]
    fn test_tokenizer_loads() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let tok = Tokenizer::from_file(TOKENIZER_PATH);
        assert!(tok.is_ok());
    }

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let tok = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let cases = vec!["Hello world", "The quick brown fox", "don't you think?"];
        for s in cases {
            let ids = tok.encode(s, EncodeOptions::default());
            let decoded = tok.decode(&ids);
            assert_eq!(decoded, s, "round-trip failed for {:?}", s);
        }
    }

    #[test]
    fn test_tokenizer_bos_eos() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }
        let tok = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let ids = tok.encode(
            "Hello",
            EncodeOptions {
                add_bos: true,
                add_eos: true,
            },
        );
        assert_eq!(ids[0], tok.bos_id().unwrap());
        assert_eq!(*ids.last().unwrap(), tok.eos_id().unwrap());
    }

    // --- Model integration ---

    #[test]
    fn test_model_forward_correct_vocab_logits() {
        let config = make_small_config(1000);
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let token_ids = vec![1u32, 2, 3];
        let logits = model.forward(&token_ids, None, 0).unwrap();
        // logits shape [1, seq_len, vocab_size]
        assert_eq!(logits.shape().dim(0).unwrap(), 1);
        assert_eq!(logits.shape().dim(1).unwrap(), 3);
        assert_eq!(logits.shape().dim(2).unwrap(), 1000);
    }

    #[test]
    fn test_model_autoregressive_generation() {
        // simulate 4 steps of autoregressive generation
        let config = make_small_config(1000);
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache = Vec::new();

        // prefill with prompt
        let prompt = vec![1u32, 2, 3];
        let logits = model.forward(&prompt, Some(&mut cache), 0).unwrap();
        assert_eq!(logits.shape().dim(1).unwrap(), 3);
        assert_eq!(cache.len(), config.num_hidden_layers);

        // generate 3 more tokens one at a time
        let mut offset = prompt.len();
        for _ in 0..3 {
            let next_token = vec![42u32];
            let logits = model
                .forward(&next_token, Some(&mut cache), offset)
                .unwrap();
            assert_eq!(logits.shape().dim(1).unwrap(), 1);
            assert_eq!(logits.shape().dim(2).unwrap(), 1000);
            offset += 1;
        }
    }

    #[test]
    fn test_model_kv_cache_sequence_grows() {
        let config = make_small_config(1000);
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache = Vec::new();

        model
            .forward(&[1u32, 2, 3, 4], Some(&mut cache), 0)
            .unwrap();
        let initial_seq = cache[0].0.shape().dim(1).unwrap();

        model.forward(&[5u32], Some(&mut cache), 4).unwrap();
        let grown_seq = cache[0].0.shape().dim(1).unwrap();

        assert_eq!(grown_seq, initial_seq + 1);
    }

    #[test]
    fn test_model_no_cache_stateless() {
        // running without cache twice should give same shaped output
        let config = make_small_config(1000);
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let tokens = vec![1u32, 2, 3];

        let out1 = model.forward(&tokens, None, 0).unwrap();
        let out2 = model.forward(&tokens, None, 0).unwrap();

        assert_eq!(out1.shape(), out2.shape());
    }

    // --- Full pipeline: tokenizer → model ---

    #[test]
    fn test_full_pipeline_tokenizer_to_model() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }

        let tok = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let ids = tok.encode("Hello world", EncodeOptions::default());
        assert!(!ids.is_empty());

        // model vocab must be at least as large as tokenizer vocab
        let vocab_size = tok.vocab().size();
        let config = make_small_config(vocab_size);
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();

        let ids_u32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let logits = model.forward(&ids_u32, None, 0).unwrap();

        assert_eq!(logits.shape().dim(2).unwrap(), vocab_size);
    }

    #[test]
    fn test_full_pipeline_greedy_next_token() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }

        let tok = Tokenizer::from_file(TOKENIZER_PATH).unwrap();

        // vocab size must cover all token IDs including special tokens
        // LLaMA 3 BOS is 128000 so vocab must be at least 128001
        let bos_id = tok.bos_id().unwrap_or(0);
        let vocab_size = tok.vocab().size().max(bos_id + 1);

        let ids = tok.encode(
            "Hello",
            EncodeOptions {
                add_bos: true,
                ..Default::default()
            },
        );

        let config = make_small_config(vocab_size);
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();

        let ids_u32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let logits = model.forward(&ids_u32, None, 0).unwrap();

        let seq_len = logits.shape().dim(1).unwrap();
        let last_logits = logits.narrow(1, seq_len - 1, 1).unwrap();

        assert_eq!(last_logits.shape().dim(1).unwrap(), 1);
        assert_eq!(last_logits.shape().dim(2).unwrap(), vocab_size);
    }

    #[test]
    fn test_full_pipeline_with_kv_cache() {
        if !std::path::Path::new(TOKENIZER_PATH).exists() {
            return;
        }

        let tok = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let prompt = "The quick brown fox";
        let ids = tok.encode(prompt, EncodeOptions::default());
        let vocab_size = tok.vocab().size();

        let config = make_small_config(vocab_size);
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache = Vec::new();

        let ids_u32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let offset = ids_u32.len();

        // prefill
        model.forward(&ids_u32, Some(&mut cache), 0).unwrap();
        assert_eq!(cache.len(), config.num_hidden_layers);

        // one generation step
        let next = vec![42u32];
        let logits = model.forward(&next, Some(&mut cache), offset).unwrap();
        assert_eq!(logits.shape().dim(1).unwrap(), 1);
        assert_eq!(logits.shape().dim(2).unwrap(), vocab_size);
    }
}
