use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::config::ModelConfig;
use crate::model::linear::Linear;
use crate::model::rope::RotaryEmbedding;

pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rope: RotaryEmbedding<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32, // 1/sqrt(head_dim) — precomputed
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &ModelConfig, device: &Device) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let make_linear = |out: usize, inp: usize| -> Result<Linear<B>> {
            let w = B::Tensor::zeros(&Shape::new(&[out, inp]), DType::F32, device)?;
            Ok(Linear::new(w, None))
        };

        let q_proj = make_linear(num_heads * head_dim, hidden)?;
        let k_proj = make_linear(num_kv_heads * head_dim, hidden)?;
        let v_proj = make_linear(num_kv_heads * head_dim, hidden)?;
        let o_proj = make_linear(hidden, num_heads * head_dim)?;
        let rope = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        })
    }

    // (output, new_k, new_v)
    pub fn forward(
        &self,
        x: &B::Tensor,                              // [batch, seq_len, hidden_size]
        mask: Option<&B::Tensor>,                   // [batch, 1, seq_len, seq_len] causal mask
        kv_cache: Option<(&B::Tensor, &B::Tensor)>, // (past_k, past_v)
        offset: usize,
    ) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
        let batch = x.shape().dim(0)?;
        let seq_len = x.shape().dim(1)?;

        let q = self.q_proj.forward(x)?.reshape(&Shape::new(&[
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ]))?;
        let k = self.k_proj.forward(x)?.reshape(&Shape::new(&[
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]))?;
        let v = self.v_proj.forward(x)?.reshape(&Shape::new(&[
            batch,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ]))?;

        let q = self.rope.forward(&q, offset)?;
        let k = self.rope.forward(&k, offset)?;

        let (mut k, mut v) = match kv_cache {
            Some((past_k, past_v)) => {
                let k = B::Tensor::cat(&[past_k, &k], 1)?;
                let v = B::Tensor::cat(&[past_v, &v], 1)?;
                (k, v)
            }
            None => (k, v),
        };

        let k_cache = k.clone();
        let v_cache = v.clone();

        let repeat_factor = self.num_heads / self.num_kv_heads;
        if repeat_factor > 1 {
            k = k.repeat(&Shape::new(&[1, 1, repeat_factor, 1]))?;
            v = v.repeat(&Shape::new(&[1, 1, repeat_factor, 1]))?;
        }

        // transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // scaled dot product attention
        let scores = q
            .broadcast_matmul(&k.transpose(2, 3)?)?
            .scale(self.scale as f64)?;

        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };

        let scores_vec = scores.to_vec_f32()?;
        let scores_max = scores_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let weights = scores.softmax(3)?;
        let out = weights.broadcast_matmul(&v)?;

        // transpose back and reshape
        let out = out.transpose(1, 2)?.contiguous()?.reshape(&Shape::new(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ]))?;

        let out = self.o_proj.forward(&out)?;

        let out_vec = out.to_vec_f32()?;
        let out_max = out_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        use std::sync::atomic::{AtomicUsize, Ordering};
        static ATTN_CALL: AtomicUsize = AtomicUsize::new(0);
        let call_idx = ATTN_CALL.fetch_add(1, Ordering::Relaxed);
        if call_idx < 2 {}

        Ok((out, k_cache, v_cache)) // (output, new_k, new_v)
    }

    pub fn set_q_proj(&mut self, l: Linear<B>) {
        self.q_proj = l;
    }
    pub fn set_k_proj(&mut self, l: Linear<B>) {
        self.k_proj = l;
    }
    pub fn set_v_proj(&mut self, l: Linear<B>) {
        self.v_proj = l;
    }
    pub fn set_o_proj(&mut self, l: Linear<B>) {
        self.o_proj = l;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleBackend;
    use crate::core::backend::CandleTensor;
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;
    use crate::model::config::ModelConfig;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 128,
            vocab_size: 1000,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            rope_theta: 10000.0,
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

    #[test]
    fn test_attention_output_shape() {
        let config = make_config();
        let attn = Attention::<CandleBackend>::new(&config, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 8, 64]), DType::F32, &cpu()).unwrap();
        let (out, k, v) = attn.forward(&x, None, None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 8, 64]));
        assert_eq!(k.shape(), &Shape::new(&[1, 8, 2, 16]));
        assert_eq!(v.shape(), &Shape::new(&[1, 8, 2, 16]));
    }

    #[test]
    fn test_attention_with_mask() {
        let config = make_config();
        let attn = Attention::<CandleBackend>::new(&config, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &cpu()).unwrap();
        let mask = CandleTensor::zeros(&Shape::new(&[1, 1, 4, 4]), DType::F32, &cpu()).unwrap();
        let (out, _, _) = attn.forward(&x, Some(&mask), None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }

    #[test]
    fn test_attention_with_kv_cache() {
        let config = make_config();
        let attn = Attention::<CandleBackend>::new(&config, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &cpu()).unwrap();

        // first forward — no cache
        let (_, k, v) = attn.forward(&x, None, None, 0).unwrap();

        // second forward — with cache, single new token
        let x2 = CandleTensor::zeros(&Shape::new(&[1, 1, 64]), DType::F32, &cpu()).unwrap();
        let (out2, k2, v2) = attn.forward(&x2, None, Some((&k, &v)), 4).unwrap();
        assert_eq!(out2.shape(), &Shape::new(&[1, 1, 64]));
        // kv cache should now have 5 positions
        assert_eq!(k2.shape().dim(1).unwrap(), 5);
        assert_eq!(v2.shape().dim(1).unwrap(), 5);
    }

    #[test]
    fn test_attention_gqa_mha_equiv() {
        // when num_kv_heads == num_heads, GQA is equivalent to MHA
        let mut config = make_config();
        config.num_key_value_heads = config.num_attention_heads;
        let attn = Attention::<CandleBackend>::new(&config, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 64]), DType::F32, &cpu()).unwrap();
        let (out, _, _) = attn.forward(&x, None, None, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 64]));
    }
}
