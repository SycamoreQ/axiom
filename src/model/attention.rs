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
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    rope: RotaryEmbedding<B>,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f64,
    head_dim: usize,
    scale: f32, // 1/sqrt(head_dim) — precomputed
    pub metal_q_weight: Option<B::Tensor>,
    pub metal_k_weight: Option<B::Tensor>,
    pub metal_v_weight: Option<B::Tensor>,
    pub metal_o_weight: Option<B::Tensor>,
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
            config.rope_freqs.as_deref(),
            device,
        )?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_theta = config.rope_theta;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            scale,
            metal_q_weight: Option::None,
            metal_k_weight: Option::None,
            metal_v_weight: Option::None,
            metal_o_weight: Option::None,
        })
    }

    pub fn prepare_metal_weights(&mut self) -> Result<()> {
        self.metal_q_weight = Some(self.q_proj.weight().transpose(0, 1)?.contiguous()?);
        self.metal_k_weight = Some(self.k_proj.weight().transpose(0, 1)?.contiguous()?);
        self.metal_v_weight = Some(self.v_proj.weight().transpose(0, 1)?.contiguous()?);
        self.metal_o_weight = Some(self.o_proj.weight().transpose(0, 1)?.contiguous()?);
        Ok(())
    }

    pub fn apply_cpu_rope(
        x: &B::Tensor,
        offset: usize,
        theta: f64,
        head_dim: usize,
    ) -> Result<B::Tensor> {
        let seq_len = x.shape().dim(1)?;
        let n_heads = x.shape().dim(2)?;
        let data = x.to_vec_f32()?;
        let mut out = vec![0.0f32; data.len()];
        for token in 0..seq_len {
            for head in 0..n_heads {
                let idx = (token * n_heads + head) * head_dim;
                for i in 0..head_dim / 2 {
                    let freq = 1.0f64 / theta.powf((2 * i) as f64 / head_dim as f64);
                    let angle = (offset + token) as f64 * freq;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let sin_a = sin_a as f32;
                    let cos_a = cos_a as f32;
                    let x0 = data[idx + 2 * i];
                    let x1 = data[idx + 2 * i + 1];
                    out[idx + 2 * i] = x0 * cos_a - x1 * sin_a;
                    out[idx + 2 * i + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
        B::Tensor::from_slice(&out, &x.shape(), &x.device())
    }

    pub fn forward(
        &self,
        x: &B::Tensor,
        mask: Option<&B::Tensor>,
        kv_cache: Option<(&B::Tensor, &B::Tensor)>,
        offset: usize,
    ) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
        let batch = x.shape().dim(0)?;
        let seq_len = x.shape().dim(1)?;
        let q_pre_reshape = self.q_proj.forward(x)?;

        let q = q_pre_reshape.reshape(&Shape::new(&[
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

        let q = q.rope(offset, self.rope_theta, self.head_dim)?;
        let k = k.rope(offset, self.rope_theta, self.head_dim)?;

        let (mut k, mut v) = match kv_cache {
            Some((past_k, past_v)) => (
                B::Tensor::cat(&[past_k, &k], 1)?,
                B::Tensor::cat(&[past_v, &v], 1)?,
            ),
            None => (k, v),
        };
        let k_cache = k.clone();
        let v_cache = v.clone();

        let repeat_factor = self.num_heads / self.num_kv_heads;
        if repeat_factor > 1 {
            let (b, s, kvh, d) = (
                k.shape().dim(0)?,
                k.shape().dim(1)?,
                k.shape().dim(2)?,
                k.shape().dim(3)?,
            );
            k = k
                .unsqueeze(3)?
                .repeat(&Shape::new(&[1, 1, 1, repeat_factor, 1]))?
                .reshape(&Shape::new(&[b, s, kvh * repeat_factor, d]))?;
            v = v
                .unsqueeze(3)?
                .repeat(&Shape::new(&[1, 1, 1, repeat_factor, 1]))?
                .reshape(&Shape::new(&[b, s, kvh * repeat_factor, d]))?;
        }

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scores = q
            .broadcast_matmul(&k.transpose(2, 3)?.contiguous()?)?
            .scale(self.scale as f64)?;

        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };

        let weights = scores.softmax(3)?;

        let out = weights.contiguous()?.broadcast_matmul(&v)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape(&Shape::new(&[
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ]))?;

        let out = self.o_proj.forward(&out)?;

        Ok((out, k_cache, v_cache))
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

    pub(super) fn make_config() -> ModelConfig {
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

        let (_, k, v) = attn.forward(&x, None, None, 0).unwrap();

        let x2 = CandleTensor::zeros(&Shape::new(&[1, 1, 64]), DType::F32, &cpu()).unwrap();
        let (out2, k2, v2) = attn.forward(&x2, None, Some((&k, &v)), 4).unwrap();
        assert_eq!(out2.shape(), &Shape::new(&[1, 1, 64]));
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

    #[test]
    fn test_apply_cpu_rope_matches_llama_cpp_ground_truth() {
        let head_dim = 64usize;
        let mut data = vec![0.0f32; 2 * head_dim]; // seq_len=2, n_heads=1
        data[head_dim] = 0.4288; // position 1, channel 0 (pre-RoPE Q)
        data[head_dim + 1] = -0.2099; // position 1, channel 1 (pre-RoPE Q)
        data[head_dim + 2] = 1.1587; // position 1, channel 2

        let x = CandleTensor::from_slice(&data, &Shape::new(&[1, 2, 1, head_dim]), &cpu()).unwrap();
        let out = Attention::<CandleBackend>::apply_cpu_rope(&x, 0, 500000.0, head_dim).unwrap();
        let out = out.to_vec_f32().unwrap();

        assert!(
            (out[head_dim] - 0.4083).abs() < 1e-3,
            "channel 0 @ pos 1: got {}, want 0.4083",
            out[head_dim]
        );
        assert!(
            (out[head_dim + 1] - 0.2474).abs() < 1e-3,
            "channel 1 @ pos 1: got {}, want 0.2474",
            out[head_dim + 1]
        );
    }

    #[test]
    fn test_rope_trait_method_matches_llama_cpp_ground_truth() {
        // Same as test_apply_cpu_rope_matches_llama_cpp_ground_truth, but
        // exercising the actual `.rope()` trait method used in
        // Attention::forward now, not the underlying free function directly
        // — confirms the Metal migration's wiring, not just the math.
        let head_dim = 64usize;
        let mut data = vec![0.0f32; 2 * head_dim];
        data[head_dim] = 0.4288;
        data[head_dim + 1] = -0.2099;
        data[head_dim + 2] = 1.1587;

        let x = CandleTensor::from_slice(&data, &Shape::new(&[1, 2, 1, head_dim]), &cpu()).unwrap();
        let out = x.rope(0, 500000.0, head_dim).unwrap();
        let out = out.to_vec_f32().unwrap();

        assert!((out[head_dim] - 0.4083).abs() < 1e-3);
        assert!((out[head_dim + 1] - 0.2474).abs() < 1e-3);
    }

    #[test]
    fn test_rope_offset_shifts_position() {
        // A nonzero `offset` (as used during incremental decode with a
        // growing KV cache) must produce a different rotation than offset=0
        // — this is the exact bug that was silently present in the unused
        // Metal rope kernel before the offset parameter was added to it.
        let head_dim = 8usize;
        let mut data = vec![0.0f32; head_dim];
        data[0] = 1.0;
        data[1] = 0.5;

        let x = CandleTensor::from_slice(&data, &Shape::new(&[1, 1, 1, head_dim]), &cpu()).unwrap();
        let out_offset_0 = x.rope(0, 10000.0, head_dim).unwrap().to_vec_f32().unwrap();
        let out_offset_5 = x.rope(5, 10000.0, head_dim).unwrap().to_vec_f32().unwrap();

        assert!(
            (out_offset_0[0] - out_offset_5[0]).abs() > 1e-4,
            "offset=0 and offset=5 should rotate differently, got {} vs {}",
            out_offset_0[0],
            out_offset_5[0]
        );
    }
}
