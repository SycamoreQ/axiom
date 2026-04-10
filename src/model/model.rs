use crate::core::backend::{Backend, CandleTensor};
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::core::tensor::{TopKLastDimOp, TopKOutput};
use crate::model::attention::Attention;
use crate::model::block::Block;
use crate::model::config::ModelConfig;
use crate::model::embedding::Embedding;
use crate::model::linear::Linear;
use crate::model::norm::RmsNorm;

pub struct LlamaModel<B: Backend> {
    embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    norm: RmsNorm<B>,
    lm_head: Linear<B>,
    config: ModelConfig,
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
            None, // no bias
        ); // [vocab_size , hidden_size]

        Ok(Self {
            embedding: embedding,
            blocks,
            norm: norm,
            lm_head,
            config: config.clone(),
        })
    }

    fn causal_mask(&self, seq_len: usize, device: &Device) -> Result<B::Tensor> {
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
        kv_cache: Option<&mut Vec<(B::Tensor, B::Tensor)>>,
        offset: usize,
    ) -> Result<B::Tensor> {
        let seq_len = token_ids.len();
        let mut embedded_tokens = self.embedding.forward(token_ids)?;

        let device = embedded_tokens.device();
        let causal_mask = self.causal_mask(seq_len, device)?;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let block_cache = kv_cache.as_ref().map(|v| {
                let (k, v) = &v[i];
                (k, v)
            });
            let (block_out, logits, aux_loss) =
                block.forward(&embedded_tokens, Some(&causal_mask), block_cache, offset)?;

            embedded_tokens = block_out;
        }

        let final_logits = self.lm_head.forward(&embedded_tokens)?;

        Ok(final_logits)
    }
}
