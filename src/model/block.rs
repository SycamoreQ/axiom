use crate::core::backend::{Backend, CandleTensor};
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::core::tensor::{TopKLastDimOp, TopKOutput};
use crate::model::attention::Attention;
use crate::model::config::ModelConfig;
use crate::model::feedforward::FeedForward;
use crate::model::linear::Linear;
use crate::model::moe::MoeLayer;
use crate::model::norm::RmsNorm;

pub enum FeedForwardLayer<B: Backend> {
    Dense(FeedForward<B>),
    Moe(MoeLayer<B>),
}

pub struct Block<B: Backend> {
    attn_norm: RmsNorm<B>,
    attn: Attention<B>,
    ffn_norm: RmsNorm<B>,
    ffn: FeedForwardLayer<B>,
    layer_idx: usize,
}

impl<B: Backend> Block<B> {
    pub fn new(config: &ModelConfig, layer_idx: usize, device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let attn_norm = RmsNorm::new(
            B::Tensor::ones(&Shape::new(&[hidden_size]), DType::F32, device)?,
            config.rms_norm_eps as f32,
        );

        let ffn_norm = RmsNorm::new(
            B::Tensor::ones(&Shape::new(&[hidden_size]), DType::F32, device)?,
            config.rms_norm_eps as f32,
        );

        let attn = Attention::new(config, device)?;

        let ffn = if config.is_moe_layer(layer_idx) {
            FeedForwardLayer::Moe(MoeLayer::new(config, None, device)?)
        } else {
            FeedForwardLayer::Dense(FeedForward::new(config, device)?)
        };

        Ok(Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
            layer_idx,
        })
    }

    pub fn forward(
        &mut self,
        x: &B::Tensor,
        mask: Option<&B::Tensor>,
        kv_cache: Option<(&B::Tensor, &B::Tensor)>,
        offset: usize,
    ) -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
        //attention with pre-norm and residual
        let h = self.attn_norm.forward(x)?;
        let (attn_out, new_k, new_v) = self.attn.forward(&h, mask, kv_cache, offset)?;
        let x = x.add(&attn_out)?;

        //ffn with pre-norm and residual
        let h = self.ffn_norm.forward(&x)?;
        let ffn_out = match &mut self.ffn {
            FeedForwardLayer::Dense(ff) => ff.forward(&h)?,
            FeedForwardLayer::Moe(moe) => moe.forward(&h, offset)?.hidden_states,
        };
        let x = x.add(&ffn_out)?;

        Ok((x, new_k, new_v))
    }
}
