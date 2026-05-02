use crate::core::backend::Backend;
use crate::core::backend::CandleBackend;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;

pub struct Embedding<B: Backend> {
    weight: B::Tensor, // [vocab_size, hidden_size]
    vocab_size: usize,
    hidden_size: usize,
}

impl<B: Backend> Embedding<B> {
    pub fn new(weight: B::Tensor) -> Self {
        let vocab_size = weight.shape().dim(0).unwrap_or(0);
        let hidden_size = weight.shape().dim(1).unwrap_or(0);
        Self {
            weight,
            vocab_size,
            hidden_size,
        }
    }

    pub fn forward(&self, token_ids: &[u32]) -> Result<B::Tensor> {
        let device = self.weight.device().clone();
        let ids_tensor =
            B::Tensor::from_u32_slice(token_ids, &Shape::new(&[token_ids.len()]), &device)?;
        self.weight.index_select(&ids_tensor, 0)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::backend::CandleTensor;
    use crate::core::device::Device;
    use crate::core::dtype::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::TensorOps;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_embedding(vocab_size: usize, hidden_size: usize) -> Embedding<CandleBackend> {
        let weight =
            CandleTensor::zeros(&Shape::new(&[vocab_size, hidden_size]), DType::F32, &cpu())
                .unwrap();
        Embedding::new(weight)
    }

    #[test]
    fn test_vocab_size() {
        let e = make_embedding(128, 64);
        assert_eq!(e.vocab_size(), 128);
    }

    #[test]
    fn test_hidden_size() {
        let e = make_embedding(128, 64);
        assert_eq!(e.hidden_size(), 64);
    }

    #[test]
    fn test_forward_single_token_shape() {
        let e = make_embedding(128, 64);
        let out = e.forward(&[0u32]).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, 64]));
    }

    #[test]
    fn test_forward_multiple_tokens_shape() {
        let e = make_embedding(128, 64);
        let out = e.forward(&[0u32, 1, 2, 3]).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[4, 64]));
    }

    #[test]
    fn test_forward_seq_len_matches_input() {
        let e = make_embedding(1000, 512);
        let token_ids: Vec<u32> = (0..16).collect();
        let out = e.forward(&token_ids).unwrap();
        assert_eq!(out.shape().dim(0).unwrap(), 16);
        assert_eq!(out.shape().dim(1).unwrap(), 512);
    }

    #[test]
    fn test_forward_known_values() {
        // build a weight table where row i = [i as f32; hidden]
        let vocab_size = 4;
        let hidden_size = 3;
        let weight_data: Vec<f32> = (0..vocab_size)
            .flat_map(|i| vec![i as f32; hidden_size])
            .collect();
        let weight = CandleTensor::from_slice(
            &weight_data,
            &Shape::new(&[vocab_size, hidden_size]),
            &cpu(),
        )
        .unwrap();
        let e: Embedding<CandleBackend> = Embedding::new(weight);

        // looking up token 2 should give [2.0, 2.0, 2.0]
        let out = e.forward(&[2u32]).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[1, hidden_size]));
    }

    #[test]
    fn test_forward_out_of_order_tokens() {
        let e = make_embedding(128, 32);
        let out = e.forward(&[5u32, 1, 99, 0]).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[4, 32]));
    }

    #[test]
    fn test_weight_shape() {
        let e = make_embedding(256, 128);
        assert_eq!(e.weight.shape(), &Shape::new(&[256, 128]));
    }
}
