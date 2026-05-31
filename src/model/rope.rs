use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::error::CoreError;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;

/*
Instead of adding positional embeddings to the input, it encodes position by rotating query and key vectors in complex space before attention.
The rotation angle depends on both the position and the dimension index.

RoPE you use broadcast_mul because you're doing elementwise scaling —
each element of the query/key vector gets multiplied by its corresponding cos or sin value. There's no matrix multiplication happening.
 */

pub struct RotaryEmbedding<B: Backend> {
    cos: B::Tensor, // [max_seq_len, head_dim/2]
    sin: B::Tensor, // [max_seq_len, head_dim/2]
    head_dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let mut freqs: Vec<f32> = Vec::new();
        for i in 0..head_dim / 2 {
            freqs.push((1.0 / rope_theta.powf((2 * i) as f64 / head_dim as f64)) as f32);
        }

        let mut angles: Vec<f32> = Vec::new();
        for pos in 0..max_seq_len {
            for freq in &freqs {
                angles.push(pos as f32 * freq);
            }
        }
        // angles is now [max_seq_len * head_dim/2] flattened
        let angle_tensor =
            B::Tensor::from_slice(&angles, &Shape::new(&[max_seq_len, head_dim / 2]), device)?;
        let cos = angle_tensor.cos()?;
        let sin = angle_tensor.sin()?;

        Ok(Self { cos, sin, head_dim })
    }

    pub fn forward(
        &self,
        x: &B::Tensor, // [batch, seq_len, num_heads, head_dim]
        offset: usize,
    ) -> Result<B::Tensor> {
        let seq_len = x.shape().dim(1)?;
        let end = offset + seq_len;
        let max_seq_len = self.cos.shape().dim(0)?;
        if end > max_seq_len {
            return Err(CoreError::OutOfBounds {
                op: "rope_slice",
                index: end,
                size: max_seq_len,
            });
        }

        // sliced_cos/sin are [seq_len, head_dim/2]
        // need to unsqueeze to [1, seq_len, 1, head_dim/2] for broadcast
        let sliced_cos = self
            .cos
            .narrow(0, offset, seq_len)?
            .unsqueeze(0)? // [1, seq_len, head_dim/2]
            .unsqueeze(2)?; // [1, seq_len, 1, head_dim/2]

        let sliced_sin = self
            .sin
            .narrow(0, offset, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(2)?;

        let chunks = x.chunk(2, 3)?;
        let x1 = &chunks[0];
        let x2 = &chunks[1];

        let out_even = x1
            .broadcast_mul(&sliced_cos)?
            .sub(&x2.broadcast_mul(&sliced_sin)?)?;
        let out_odd = x1
            .broadcast_mul(&sliced_sin)?
            .add(&x2.broadcast_mul(&sliced_cos)?)?;
        B::Tensor::cat(&[&out_even, &out_odd], x.rank() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::{CandleBackend, CandleTensor};
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_rope(head_dim: usize, max_seq_len: usize) -> RotaryEmbedding<CandleBackend> {
        RotaryEmbedding::new(head_dim, max_seq_len, 10000.0, &cpu()).unwrap()
    }

    #[test]
    fn test_cos_sin_shape() {
        let rope = make_rope(64, 128);
        assert_eq!(rope.cos.shape(), &Shape::new(&[128, 32]));
        assert_eq!(rope.sin.shape(), &Shape::new(&[128, 32]));
    }

    #[test]
    fn test_forward_shape_preserved() {
        let rope = make_rope(64, 128);
        // [batch, seq_len, num_heads, head_dim]
        let x = CandleTensor::ones(&Shape::new(&[1, 8, 4, 64]), DType::F32, &cpu()).unwrap();
        let out = rope.forward(&x, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 8, 4, 64]));
    }

    #[test]
    fn test_forward_with_offset() {
        let rope = make_rope(64, 128);
        let x = CandleTensor::ones(&Shape::new(&[1, 4, 4, 64]), DType::F32, &cpu()).unwrap();
        // offset=10 means we're at position 10 in the sequence
        let out = rope.forward(&x, 10).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 4, 64]));
    }

    #[test]
    fn test_forward_single_token() {
        // typical generation step — one token at a time
        let rope = make_rope(64, 128);
        let x = CandleTensor::ones(&Shape::new(&[1, 1, 4, 64]), DType::F32, &cpu()).unwrap();
        let out = rope.forward(&x, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 1, 4, 64]));
    }

    #[test]
    fn test_forward_offset_out_of_bounds() {
        let rope = make_rope(64, 32);
        let x = CandleTensor::ones(&Shape::new(&[1, 8, 4, 64]), DType::F32, &cpu()).unwrap();
        // offset=30 + seq_len=8 = 38 > max_seq_len=32 — should error
        assert!(rope.forward(&x, 30).is_err());
    }

    #[test]
    fn test_forward_full_sequence() {
        let rope = make_rope(32, 64);
        let x = CandleTensor::ones(&Shape::new(&[2, 64, 8, 32]), DType::F32, &cpu()).unwrap();
        let out = rope.forward(&x, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[2, 64, 8, 32]));
    }

    #[test]
    fn test_different_rope_theta() {
        // LLaMA 3 uses 500000.0
        let rope = RotaryEmbedding::<CandleBackend>::new(64, 128, 500000.0, &cpu()).unwrap();
        let x = CandleTensor::ones(&Shape::new(&[1, 4, 4, 64]), DType::F32, &cpu()).unwrap();
        let out = rope.forward(&x, 0).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 4, 4, 64]));
    }

    #[test]
    fn test_head_dim_stored() {
        let rope = make_rope(128, 64);
        assert_eq!(rope.head_dim, 128);
    }
}
