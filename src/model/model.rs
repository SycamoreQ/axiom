use crate::core::backend::Backend;
use crate::core::backend::MetalTensor;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::CoreError;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::block::Block;
use crate::model::block::FeedForwardLayer;
use crate::model::config::ModelConfig;
use crate::model::embedding::Embedding;
use crate::model::linear::Linear;
use crate::model::norm::RmsNorm;

pub struct LlamaModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    config: ModelConfig,
    metal_lm_head_weight: Option<B::Tensor>,
}

impl<B: Backend> LlamaModel<B> {
    pub fn new(config: &ModelConfig, device: &Device) -> Result<Self> {
        let embedding = Embedding::new(B::Tensor::zeros(
            &Shape::new(&[config.vocab_size, config.hidden_size]),
            DType::F32,
            device,
        )?);

        let blocks: Vec<Block<B>> = (0..config.num_hidden_layers)
            .map(|layer_idx| Block::new(config, layer_idx, device))
            .collect::<Result<Vec<_>>>()?;

        let norm = RmsNorm::new(
            B::Tensor::ones(&Shape::new(&[config.hidden_size]), DType::F32, device)?,
            config.rms_norm_eps as f32,
        );

        let lm_head = Linear::new(
            B::Tensor::zeros(
                &Shape::new(&[config.vocab_size, config.hidden_size]),
                DType::F32,
                device,
            )?,
            None,
        );

        Ok(Self {
            embedding,
            blocks,
            norm,
            lm_head,
            config: config.clone(),
            metal_lm_head_weight: Option::None,
        })
    }

    pub fn prepare_metal(&mut self) -> Result<()> {
        for block in self.blocks.iter_mut() {
            block.prepare_metal()?;
        }
        self.metal_lm_head_weight = Some(self.lm_head.weight().transpose(0, 1)?.contiguous()?);
        Ok(())
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<B::Tensor> {
        if seq_len == 1 {
            return B::Tensor::from_slice(&[0.0f32], &Shape::new(&[1, 1, 1, 1]), device);
        }
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        B::Tensor::from_slice(&mask, &Shape::new(&[1, 1, seq_len, seq_len]), device)
    }

    pub fn forward(
        &mut self,
        token_ids: &[u32],
        mut kv_cache: Option<&mut Vec<(B::Tensor, B::Tensor)>>,
        offset: usize,
    ) -> Result<B::Tensor> {
        // Use the Metal fast path only for single‑token decode (seq_len == 1)
        if self.embedding.weight().device().is_metal() && token_ids.len() == 1 {
            #[cfg(feature = "metal")]
            return self.forward_metal(token_ids, kv_cache, offset);
            #[cfg(not(feature = "metal"))]
            return Err(CoreError::Internal("Metal feature not enabled".into()));
        }

        let seq_len = token_ids.len();
        let x = self.embedding.forward(token_ids)?;
        let mut x = x.unsqueeze(0)?;
        let device = x.device().clone();
        let mask = self.causal_mask(seq_len, &device)?;
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let cache = kv_cache
                .as_deref()
                .and_then(|v| v.get(i).map(|(k, v)| (k, v)));
            let mask_ref = if token_ids.len() > 1 {
                Some(&mask)
            } else {
                None
            };
            let (block_out, new_k, new_v) = block.forward(&x, mask_ref, cache, offset)?;
            x = block_out;
            if let Some(ref mut cache) = kv_cache {
                if i < cache.len() {
                    cache[i] = (new_k, new_v);
                } else {
                    cache.push((new_k, new_v));
                }
            }
        }

        let x_pre_norm = x.to_vec_f32()?;
        let w_norm = self.norm.weight().to_vec_f32()?;
        let eps = self.norm.eps();

        let mut normed = vec![0.0f32; x_pre_norm.len()];
        for i in 0..x_pre_norm.len() {
            let row_start = (i / hidden_size) * hidden_size;
            let row = &x_pre_norm[row_start..row_start + hidden_size];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32;
            let rms = (ms + eps).sqrt();
            normed[i] = x_pre_norm[i] * w_norm[i % hidden_size] / rms;
        }

        let normed_tensor = B::Tensor::from_slice(&normed, &x.shape(), &device)?;
        if normed.iter().any(|&x| x.is_nan()) {
            println!("NaN detected in Post Final Norm!");
        } else {
            let preview_len = 10.min(normed.len());
        }

        let w_lm_head = self.lm_head.weight().to_vec_f32()?;
        let mut logits_cpu = vec![0.0f32; seq_len * vocab_size];
        for pos in 0..seq_len {
            let hidden_start = pos * hidden_size;
            let hidden_row = &normed[hidden_start..hidden_start + hidden_size];
            let offset_logits = pos * vocab_size;
            for v in 0..vocab_size {
                let mut sum = 0.0;
                for d in 0..hidden_size {
                    sum += hidden_row[d] * w_lm_head[v * hidden_size + d];
                }
                logits_cpu[offset_logits + v] = sum;
            }
        }

        let logits_shape = Shape::new(&[1, seq_len, vocab_size]);
        let logits = B::Tensor::from_slice(&logits_cpu, &logits_shape, &device)?;

        Ok(logits)
    }

    #[cfg(feature = "metal")]
    fn forward_metal(
        &mut self,
        token_ids: &[u32],
        mut kv_cache: Option<&mut Vec<(B::Tensor, B::Tensor)>>,
        offset: usize,
    ) -> Result<B::Tensor> {
        use crate::core::tensor::TensorOps;
        use crate::metal::runner::MetalRunner;
        use crate::metal::state::global_metal_state;

        let seq_len = token_ids.len();

        let hidden = self.config.hidden_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv_heads = self.config.num_key_value_heads;
        let head_dim = hidden / n_heads;
        let theta = self.config.rope_theta as f32;
        let eps = self.config.rms_norm_eps as f32;

        let state = global_metal_state()
            .ok_or_else(|| CoreError::Internal("Metal state not initialized".into()))?;
        let alloc = state
            .alloc
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut runner = MetalRunner::new(&state, &alloc)?;

        let x_emb = self.embedding.forward(token_ids)?;
        let mut x = x_emb.unsqueeze(0)?; // [1, seq_len, hidden]

        for (i, block) in self.blocks.iter_mut().enumerate() {
            // ---- RMS norm ----
            let norm_out = B::Tensor::zeros_like(&x)?;
            runner.rms_norm(
                x.as_metal()
                    .ok_or_else(|| CoreError::Internal("x not Metal".into()))?,
                block
                    .attn_norm
                    .weight()
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("attn_norm not Metal".into()))?,
                norm_out
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("norm_out not Metal".into()))?,
                eps,
            )?;

            // ---- Q, K, V projections — cached, pre-transposed weights ----
            let norm_2d = norm_out.reshape(&Shape::new(&[seq_len, hidden]))?;

            let q_weight = block.attn.metal_q_weight.as_ref().ok_or_else(|| {
                CoreError::Internal(
                    "metal weights not prepared — call model.prepare_metal()".into(),
                )
            })?;
            let k_weight = block.attn.metal_k_weight.as_ref().ok_or_else(|| {
                CoreError::Internal(
                    "metal weights not prepared — call model.prepare_metal()".into(),
                )
            })?;
            let v_weight = block.attn.metal_v_weight.as_ref().ok_or_else(|| {
                CoreError::Internal(
                    "metal weights not prepared — call model.prepare_metal()".into(),
                )
            })?;

            let q_2d = B::Tensor::zeros(&Shape::new(&[seq_len, hidden]), x.dtype(), x.device())?;
            runner.broadcast_matmul(
                norm_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("norm_2d not Metal".into()))?,
                q_weight
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("q_weight not Metal".into()))?,
                q_2d.as_metal()
                    .ok_or_else(|| CoreError::Internal("q_2d not Metal".into()))?,
            )?;

            let k_2d = B::Tensor::zeros(
                &Shape::new(&[seq_len, n_kv_heads * head_dim]),
                x.dtype(),
                x.device(),
            )?;
            runner.broadcast_matmul(
                norm_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("norm_2d not Metal".into()))?,
                k_weight
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("k_weight not Metal".into()))?,
                k_2d.as_metal()
                    .ok_or_else(|| CoreError::Internal("k_2d not Metal".into()))?,
            )?;

            let v_2d = B::Tensor::zeros(
                &Shape::new(&[seq_len, n_kv_heads * head_dim]),
                x.dtype(),
                x.device(),
            )?;
            runner.broadcast_matmul(
                norm_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("norm_2d not Metal".into()))?,
                v_weight
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("v_weight not Metal".into()))?,
                v_2d.as_metal()
                    .ok_or_else(|| CoreError::Internal("v_2d not Metal".into()))?,
            )?;

            let q = q_2d.reshape(&Shape::new(&[1, seq_len, n_heads, head_dim]))?;
            let k = k_2d.reshape(&Shape::new(&[1, seq_len, n_kv_heads, head_dim]))?;
            let v = v_2d.reshape(&Shape::new(&[1, seq_len, n_kv_heads, head_dim]))?;

            runner.flush()?;

            let q_data = q.to_vec_f32()?;
            let mut q_rope = B::Tensor::from_slice(&q_data, q.shape(), q.device())?;
            runner.rope(
                q_rope
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("q_rope not Metal".into()))?,
                seq_len as u32,
                n_heads as u32,
                head_dim as u32,
                theta,
                offset as u32,
            )?;

            let k_data = k.to_vec_f32()?;
            let mut k_rope = B::Tensor::from_slice(&k_data, k.shape(), k.device())?;
            runner.rope(
                k_rope
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("k_rope not Metal".into()))?,
                seq_len as u32,
                n_kv_heads as u32,
                head_dim as u32,
                theta,
                offset as u32,
            )?;

            runner.flush()?;

            if let Some(ref mut cache) = kv_cache {
                if i < cache.len() {
                    let (k_cache, v_cache) = &mut cache[i];
                    let new_k = B::Tensor::cat(&[k_cache, &k_rope], 1)?;
                    let new_v = B::Tensor::cat(&[v_cache, &v], 1)?;
                    *k_cache = new_k;
                    *v_cache = new_v;
                } else {
                    cache.push((k_rope.clone(), v.clone()));
                }
            }

            let (k_for_attn, v_for_attn) = if let Some(ref mut cache) = kv_cache {
                if i < cache.len() {
                    let (k_ref, v_ref) = &cache[i];
                    (Some(k_ref), Some(v_ref))
                } else {
                    (Some(&k_rope), Some(&v))
                }
            } else {
                (Some(&k_rope), Some(&v))
            };

            let q_last = q_rope.narrow(1, seq_len - 1, 1)?.squeeze(1)?.squeeze(0)?;
            let scores = B::Tensor::zeros(&Shape::new(&[n_heads, seq_len]), x.dtype(), x.device())?;

            if let (Some(k_ref), Some(v_ref)) = (k_for_attn, v_for_attn) {
                runner.attention_qk(
                    q_last
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("q_last not Metal".into()))?,
                    k_ref
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("k_ref not Metal".into()))?,
                    scores
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("scores not Metal".into()))?,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    seq_len as u32,
                    offset as u32,
                )?;

                let probs = B::Tensor::zeros_like(&scores)?;
                runner.softmax(
                    scores
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("scores not Metal".into()))?,
                    probs
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("probs not Metal".into()))?,
                )?;

                let attn_out =
                    B::Tensor::zeros(&Shape::new(&[n_heads, head_dim]), x.dtype(), x.device())?;
                runner.attention_pv(
                    probs
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("probs not Metal".into()))?,
                    v_ref
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("v_ref not Metal".into()))?,
                    attn_out
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("attn_out not Metal".into()))?,
                    n_heads as u32,
                    n_kv_heads as u32,
                    seq_len as u32,
                    head_dim as u32,
                    offset as u32,
                )?;

                let attn_reshaped = attn_out.reshape(&Shape::new(&[1, 1, hidden]))?;

                let o_weight = block.attn.metal_o_weight.as_ref().ok_or_else(|| {
                    CoreError::Internal(
                        "metal weights not prepared — call model.prepare_metal()".into(),
                    )
                })?;
                let attn_proj =
                    B::Tensor::zeros(&Shape::new(&[1, 1, hidden]), x.dtype(), x.device())?;
                runner.broadcast_matmul(
                    attn_reshaped
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("attn_reshaped not Metal".into()))?,
                    o_weight
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("o_weight not Metal".into()))?,
                    attn_proj
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("attn_proj not Metal".into()))?,
                )?;

                let x_new = B::Tensor::zeros_like(&x)?;
                runner.add(
                    x.as_metal()
                        .ok_or_else(|| CoreError::Internal("x not Metal".into()))?,
                    attn_proj
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("attn_proj not Metal".into()))?,
                    x_new
                        .as_metal()
                        .ok_or_else(|| CoreError::Internal("x_new not Metal".into()))?,
                )?;
                x = x_new;
            } else {
                return Err(CoreError::Internal("No K/V available for attention".into()));
            }

            let ffn_norm_out = B::Tensor::zeros_like(&x)?;
            runner.rms_norm(
                x.as_metal()
                    .ok_or_else(|| CoreError::Internal("x not Metal".into()))?,
                block
                    .ffn_norm
                    .weight()
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("ffn_norm not Metal".into()))?,
                ffn_norm_out
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("ffn_norm_out not Metal".into()))?,
                eps,
            )?;

            let ffn_2d = ffn_norm_out.reshape(&Shape::new(&[seq_len, hidden]))?;

            let ff = match &block.ffn {
                FeedForwardLayer::Dense(ff) => ff,
                FeedForwardLayer::Moe(_) => {
                    return Err(CoreError::Internal(
                        "MoE not supported on the Metal fast path yet".into(),
                    ))
                }
            };
            let gate_weight = ff.metal_gate_weight.as_ref().ok_or_else(|| {
                CoreError::Internal(
                    "metal weights not prepared — call model.prepare_metal()".into(),
                )
            })?;
            let up_weight = ff.metal_up_weight.as_ref().ok_or_else(|| {
                CoreError::Internal(
                    "metal weights not prepared — call model.prepare_metal()".into(),
                )
            })?;
            let down_weight = ff.metal_down_weight.as_ref().ok_or_else(|| {
                CoreError::Internal(
                    "metal weights not prepared — call model.prepare_metal()".into(),
                )
            })?;

            let gate_2d = B::Tensor::zeros(
                &Shape::new(&[seq_len, self.config.intermediate_size]),
                x.dtype(),
                x.device(),
            )?;
            runner.broadcast_matmul(
                ffn_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("ffn_2d not Metal".into()))?,
                gate_weight
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("gate_weight not Metal".into()))?,
                gate_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("gate_2d not Metal".into()))?,
            )?;

            let up_2d = B::Tensor::zeros(
                &Shape::new(&[seq_len, self.config.intermediate_size]),
                x.dtype(),
                x.device(),
            )?;
            runner.broadcast_matmul(
                ffn_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("ffn_2d not Metal".into()))?,
                up_weight
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("up_weight not Metal".into()))?,
                up_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("up_2d not Metal".into()))?,
            )?;

            let swiglu_out = B::Tensor::zeros(
                &Shape::new(&[seq_len, self.config.intermediate_size]),
                x.dtype(),
                x.device(),
            )?;
            runner.swiglu(
                gate_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("gate_2d not Metal".into()))?,
                up_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("up_2d not Metal".into()))?,
                swiglu_out
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("swiglu_out not Metal".into()))?,
                (seq_len * self.config.intermediate_size) as u32,
            )?;

            let down_2d = B::Tensor::zeros(&Shape::new(&[seq_len, hidden]), x.dtype(), x.device())?;
            runner.broadcast_matmul(
                swiglu_out
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("swiglu_out not Metal".into()))?,
                down_weight
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("down_weight not Metal".into()))?,
                down_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("down_2d not Metal".into()))?,
            )?;

            let x_new2 = B::Tensor::zeros_like(&x)?;
            runner.add(
                x.as_metal()
                    .ok_or_else(|| CoreError::Internal("x not Metal".into()))?,
                down_2d
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("down_2d not Metal".into()))?,
                x_new2
                    .as_metal()
                    .ok_or_else(|| CoreError::Internal("x_new2 not Metal".into()))?,
            )?;
            x = x_new2;
        }

        let final_norm = B::Tensor::zeros_like(&x)?;
        runner.rms_norm(
            x.as_metal()
                .ok_or_else(|| CoreError::Internal("x not Metal".into()))?,
            self.norm
                .weight()
                .as_metal()
                .ok_or_else(|| CoreError::Internal("norm weight not Metal".into()))?,
            final_norm
                .as_metal()
                .ok_or_else(|| CoreError::Internal("final_norm not Metal".into()))?,
            eps,
        )?;

        let final_2d = final_norm.reshape(&Shape::new(&[seq_len, hidden]))?;
        let logits_2d = B::Tensor::zeros(
            &Shape::new(&[seq_len, self.config.vocab_size]),
            x.dtype(),
            x.device(),
        )?;
        let lm_weight = self.metal_lm_head_weight.as_ref().ok_or_else(|| {
            CoreError::Internal("metal weights not prepared — call model.prepare_metal()".into())
        })?;
        runner.broadcast_matmul(
            final_2d
                .as_metal()
                .ok_or_else(|| CoreError::Internal("final_2d not Metal".into()))?,
            lm_weight
                .as_metal()
                .ok_or_else(|| CoreError::Internal("lm_weight not Metal".into()))?,
            logits_2d
                .as_metal()
                .ok_or_else(|| CoreError::Internal("logits_2d not Metal".into()))?,
        )?;

        let logits = logits_2d.reshape(&Shape::new(&[1, seq_len, self.config.vocab_size]))?;

        runner.finish()?;
        Ok(logits)
    }

    pub fn set_tensor(
        &mut self,
        kind: &crate::weights::loader::LlamaTensor,
        tensor: B::Tensor,
    ) -> crate::core::error::Result<()> {
        use crate::model::embedding::Embedding;
        use crate::model::linear::Linear;
        use crate::model::norm::RmsNorm;
        use crate::weights::loader::{BlockLayer, LlamaTensor};

        match kind {
            LlamaTensor::TokenEmbd => {
                // Shape is now correctly row-major at the source (gguf.rs
                // reverses ne[] on parse), so no orientation heuristic is
                // needed or safe here anymore — trust the shape as-is.
                self.embedding = Embedding::new(tensor);
            }
            LlamaTensor::OutputNorm => {
                let eps = self.norm.eps();
                self.norm = RmsNorm::new(tensor, eps);
            }
            LlamaTensor::Output => {
                self.lm_head = Linear::new(tensor, None);
            }
            LlamaTensor::Block(i, layer) => {
                let block = self.blocks.get_mut(*i).ok_or_else(|| {
                    crate::core::error::CoreError::Internal(format!(
                        "block index {} out of range",
                        i
                    ))
                })?;
                match layer {
                    BlockLayer::AttnNorm => block.set_attn_norm(tensor),
                    BlockLayer::AttnQ => block.set_attn_q(tensor),
                    BlockLayer::AttnK => block.set_attn_k(tensor)?,
                    BlockLayer::AttnV => block.set_attn_v(tensor)?,
                    BlockLayer::AttnOutput => block.set_attn_o(tensor),
                    BlockLayer::FfnNorm => block.set_ffn_norm(tensor),
                    BlockLayer::FfnGate => block.set_ffn_gate(tensor),
                    BlockLayer::FfnUp => block.set_ffn_up(tensor),
                    BlockLayer::FfnDown => block.set_ffn_down(tensor),
                }
            }
        }
        Ok(())
    }
}

// Tests section remains identical to preserve test safety...
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::{CandleBackend, CandleTensor};
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;
    use crate::model::config::ModelConfig;

    fn cpu() -> Device {
        Device::Cpu
    }

    pub(super) fn make_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            intermediate_size: 128,
            vocab_size: 256,
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
            architectures: None,
            model_type: Some("llama".to_string()),
        }
    }

    pub(super) fn make_moe_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
            rope_freqs: None,
            rope_scaling: None,
            num_local_experts: Some(4),
            num_experts_per_tok: Some(2),
            num_shared_experts: Some(1),
            expert_interval: Some(1),
            prefetch_threshold: Some(0.3),
            torch_dtype: "float32".to_string(),
            architectures: None,
            model_type: Some("deepseek".to_string()),
        }
    }

    #[test]
    fn test_model_construction() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu());
        assert!(model.is_ok());
    }

    #[test]
    fn test_moe_model_construction() {
        let config = make_moe_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu());
        assert!(model.is_ok());
    }

    #[test]
    fn test_block_count() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        assert_eq!(model.blocks.len(), config.num_hidden_layers);
    }

    #[test]
    fn test_forward_output_shape() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let token_ids = vec![1u32, 2, 3, 4];
        let logits = model.forward(&token_ids, None, 0).unwrap();
        assert_eq!(logits.shape().dim(0).unwrap(), 1);
        assert_eq!(logits.shape().dim(1).unwrap(), 4);
        assert_eq!(logits.shape().dims().last().unwrap(), &256);
    }

    #[test]
    fn test_forward_single_token() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let logits = model.forward(&[42u32], None, 0).unwrap();
        assert_eq!(logits.shape().dim(0).unwrap(), 1);
        assert_eq!(logits.shape().dim(1).unwrap(), 1);
        assert_eq!(logits.shape().dims().last().unwrap(), &256);
    }

    #[test]
    fn test_causal_mask_shape() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let device = cpu();
        let mask = model.causal_mask(4, &device).unwrap();
        assert_eq!(mask.shape(), &Shape::new(&[1, 1, 4, 4]));
    }

    #[test]
    fn test_causal_mask_values() {
        let config = make_config();
        let model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let device = cpu();
        let mask = model.causal_mask(3, &device).unwrap();
        let flat = mask.to_vec_f32().unwrap();
        assert_eq!(flat[0], 0.0);
        assert!(flat[1].is_infinite() && flat[1] < 0.0);
        assert_eq!(flat[3], 0.0);
        assert_eq!(flat[4], 0.0);
        assert!(flat[5].is_infinite() && flat[5] < 0.0);
    }

    #[test]
    fn test_forward_with_kv_cache() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache = Vec::new();
        let _ = model
            .forward(&[1u32, 2, 3, 4], Some(&mut cache), 0)
            .unwrap();
        assert_eq!(cache.len(), config.num_hidden_layers);
        let logits2 = model.forward(&[5u32], Some(&mut cache), 4).unwrap();
        assert_eq!(logits2.shape().dim(0).unwrap(), 1);
    }

    #[test]
    fn test_forward_kv_cache_grows() {
        let config = make_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let mut cache = Vec::new();
        model
            .forward(&[1u32, 2, 3, 4], Some(&mut cache), 0)
            .unwrap();
        let first_k_seq = cache[0].0.shape().dim(1).unwrap();
        model.forward(&[5u32], Some(&mut cache), 4).unwrap();
        let second_k_seq = cache[0].0.shape().dim(1).unwrap();
        assert_eq!(second_k_seq, first_k_seq + 1);
    }

    #[test]
    fn test_moe_model_forward() {
        let config = make_moe_config();
        let mut model = LlamaModel::<CandleBackend>::new(&config, &cpu()).unwrap();
        let logits = model.forward(&[1u32, 2, 3], None, 0).unwrap();
        assert_eq!(logits.shape().dims().last().unwrap(), &256);
    }
}

#[cfg(feature = "metal")]
#[cfg(test)]
mod metal_tests {
    use super::*;
    use crate::core::backend::MetalBackend;
    use crate::core::device::Device;
    use crate::core::tensor::TensorOps;

    fn ensure_metal_device() -> Device {
        if crate::metal::state::global_metal_state().is_none() {
            let _ = crate::metal::state::init_global_metal_state(512 * 1024 * 1024);
        }
        Device::Metal(0)
    }

    fn make_config() -> ModelConfig {
        super::tests::make_config()
    }

    #[test]
    fn test_metal_model_construction() {
        let device = ensure_metal_device();
        let config = make_config();
        let model = LlamaModel::<MetalBackend>::new(&config, &device);
        assert!(model.is_ok());
    }

    #[test]
    fn test_metal_forward_single_token() {
        let device = ensure_metal_device();
        let config = make_config();
        let mut model = LlamaModel::<MetalBackend>::new(&config, &device).unwrap();
        let mut cache = Vec::new();
        let logits = model.forward(&[42u32], Some(&mut cache), 0).unwrap();
        assert_eq!(logits.shape().dim(0).unwrap(), 1);
        assert_eq!(logits.shape().dim(1).unwrap(), 1);
    }

    #[test]
    fn test_metal_logits_are_finite() {
        let device = ensure_metal_device();
        let config = make_config();
        let mut model = LlamaModel::<MetalBackend>::new(&config, &device).unwrap();
        let mut cache = Vec::new();
        let logits = model.forward(&[42u32], Some(&mut cache), 0).unwrap();
        let values = logits.to_vec_f32().unwrap();
        assert!(!values.iter().any(|x| x.is_nan()));
        assert!(!values.iter().any(|x| x.is_infinite()));
    }
}
