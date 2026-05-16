use candle_nn::ops::softmax;

use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;

/*
 Auxiliary losses for MoE load balancing.
 I have referenced Switch Transformer (Fedus et al. 2021), DeepSeek-V2 paper for the base
 impl . Advancements will be applied
*/

#[derive(Debug)]
pub struct MoeLossOutput<B: Backend> {
    //The load balancing loss scalar — add to main loss during training.
    //None during pure inference.
    pub load_balance_loss: Option<B::Tensor>,
    //Z-loss term that penalizes large router logits for stability.
    //None during pure inference.
    pub z_loss: Option<B::Tensor>,
    pub total_aux_loss: Option<B::Tensor>,
}

#[derive(Debug, Clone)]
pub struct AuxLossConfig {
    //Weight for load balancing loss
    pub load_balance_alpha: f32,
    //Weight for z-loss
    pub z_loss_beta: f32,
    //Whether to compute losses at all. False during inference.
    pub enabled: bool,
}

impl Default for AuxLossConfig {
    fn default() -> Self {
        Self {
            load_balance_alpha: 1e-3,
            z_loss_beta: 1e-3,
            enabled: false, // off by default — inference mode
        }
    }
}

impl AuxLossConfig {
    pub fn training() -> Self {
        Self {
            load_balance_alpha: 1e-3,
            z_loss_beta: 1e-3,
            enabled: true,
        }
    }
    pub fn inference() -> Self {
        Self::default()
    }
}

//Compute the Switch Transformer load balancing loss.
//loss = alpha * num_experts * sum_i( f_i * P_i )
pub fn load_balance_loss<B: Backend>(
    router_logits: &B::Tensor,  // [T, E]
    expert_indices: &B::Tensor, // [T, K]
    num_experts: usize,
    device: &Device,
) -> Result<B::Tensor> {
    let probs = router_logits.softmax(1)?;
    let mean_routing_probs = probs.mean(0)?.squeeze(0)?;

    let indices_vec = expert_indices.to_vec_u32()?;
    let mut counts = vec![0.0f32; num_experts];

    for &idx in &indices_vec {
        if (idx as usize) < num_experts {
            counts[idx as usize] += 1.0;
        }
    }

    let total_assignments = expert_indices.numel() as f32;
    for c in counts.iter_mut() {
        *c /= total_assignments;
    }
    let frac_experts =
        B::Tensor::from_slice(&counts, &Shape::new(&[num_experts]), router_logits.device())?; // [3 , 3 , .. , 3]

    let product = frac_experts.mul(&mean_routing_probs)?;

    let flattened = product.reshape(&Shape::new(&[product.numel()]))?;
    let sum_fp = flattened.sum(0)?.squeeze(0)?;

    let loss = sum_fp.scale(num_experts as f64)?;

    Ok(loss)
}

pub fn z_loss<B: Backend>(
    router_logits: &B::Tensor, // [T, E]
    device: &Device,
) -> Result<B::Tensor> {
    let exp_logits = router_logits.exp()?;
    let sum_exp = exp_logits.sum(1)?;
    let log_z = sum_exp.log()?.squeeze(1)?;
    let squared = log_z.mul(&log_z)?;
    let loss = squared.mean(0)?.squeeze(0)?;
    Ok(loss)
}

//Combine load balance loss and z-loss with their weights.
pub fn compute_aux_loss<B: Backend>(
    router_logits: &B::Tensor,
    expert_indices: &B::Tensor,
    num_experts: usize,
    config: &AuxLossConfig,
    device: &Device,
) -> Result<MoeLossOutput<B>> {
    if !config.enabled {
        return Ok(MoeLossOutput {
            load_balance_loss: None,
            z_loss: None,
            total_aux_loss: None,
        });
    }

    let lb_loss = load_balance_loss::<B>(router_logits, expert_indices, num_experts, device)?;
    let zl_loss = z_loss::<B>(router_logits, device)?;

    let lb_scaled = lb_loss.clone().scale(config.load_balance_alpha as f64)?;
    let zl_scaled = zl_loss.clone().scale(config.z_loss_beta as f64)?;
    let total = lb_scaled.add(&zl_scaled)?;

    Ok(MoeLossOutput {
        load_balance_loss: Some(lb_loss),
        z_loss: Some(zl_loss),
        total_aux_loss: Some(total),
    })
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

    #[test]
    fn test_aux_loss_config_default_disabled() {
        let config = AuxLossConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_aux_loss_config_training_enabled() {
        let config = AuxLossConfig::training();
        assert!(config.enabled);
        assert!((config.load_balance_alpha - 1e-3).abs() < 1e-6);
        assert!((config.z_loss_beta - 1e-3).abs() < 1e-6);
    }

    #[test]
    fn test_aux_loss_config_inference_disabled() {
        let config = AuxLossConfig::inference();
        assert!(!config.enabled);
    }

    #[test]
    fn test_compute_aux_loss_inference_returns_none() {
        let config = AuxLossConfig::inference();
        let logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let indices =
            CandleTensor::from_u32_slice(&[0, 1, 0, 1, 2, 3, 2, 3], &Shape::new(&[4, 2]), &cpu())
                .unwrap();
        let result =
            compute_aux_loss::<CandleBackend>(&logits, &indices, 4, &config, &cpu()).unwrap();
        assert!(result.load_balance_loss.is_none());
        assert!(result.z_loss.is_none());
        assert!(result.total_aux_loss.is_none());
    }

    #[test]
    fn test_load_balance_loss_scalar_shape() {
        let logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let indices =
            CandleTensor::from_u32_slice(&[0, 1, 0, 1, 2, 3, 2, 3], &Shape::new(&[4, 2]), &cpu())
                .unwrap();
        let loss = load_balance_loss::<CandleBackend>(&logits, &indices, 4, &cpu()).unwrap();
        // scalar — rank 0 or shape [1]
        assert!(loss.numel() == 1);
    }

    #[test]
    fn test_load_balance_loss_uniform_routing_is_low() {
        // when all experts get equal tokens, loss should be at its minimum
        // uniform logits → uniform P_i
        // uniform routing → uniform f_i
        // loss = num_experts * sum(f_i * P_i) = num_experts * (1/E * 1/E * E) = 1/E
        let logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        // each token goes to a different expert pair
        let indices =
            CandleTensor::from_u32_slice(&[0, 1, 1, 2, 2, 3, 3, 0], &Shape::new(&[4, 2]), &cpu())
                .unwrap();
        let loss = load_balance_loss::<CandleBackend>(&logits, &indices, 4, &cpu()).unwrap();
        let val = loss.to_vec_f32().unwrap()[0];
        // should be close to 1.0 for perfectly uniform routing with 4 experts
        assert!(val > 0.0, "loss should be positive");
        assert!(val < 2.0, "loss should not be huge for uniform routing");
    }

    #[test]
    fn test_load_balance_loss_collapsed_routing_is_high() {
        let mut collapsed_logits_vec = vec![0.0f32; 16];
        for chunk in collapsed_logits_vec.chunks_mut(4) {
            chunk[0] = 10.0; // Expert 0 is dominant
        }
        let collapsed_logits =
            CandleTensor::from_slice(&collapsed_logits_vec, &Shape::new(&[4, 4]), &cpu()).unwrap();

        let indices =
            CandleTensor::from_u32_slice(&[0, 0, 0, 0, 0, 0, 0, 0], &Shape::new(&[4, 2]), &cpu())
                .unwrap();

        // 2. Uniform Routing Setup: Router has completely flat layout
        let uniform_logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let uniform_indices =
            CandleTensor::from_u32_slice(&[0, 1, 1, 2, 2, 3, 3, 0], &Shape::new(&[4, 2]), &cpu())
                .unwrap();

        let collapsed_loss =
            load_balance_loss::<CandleBackend>(&collapsed_logits, &indices, 4, &cpu()).unwrap();
        let uniform_loss =
            load_balance_loss::<CandleBackend>(&uniform_logits, &uniform_indices, 4, &cpu())
                .unwrap();

        let collapsed_val = collapsed_loss.to_vec_f32().unwrap()[0];
        let uniform_val = uniform_loss.to_vec_f32().unwrap()[0];

        assert!(
            collapsed_val > uniform_val,
            "collapsed routing loss {} should exceed uniform {}",
            collapsed_val,
            uniform_val
        );
    }

    #[test]
    fn test_z_loss_scalar_shape() {
        let logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let loss = z_loss::<CandleBackend>(&logits, &cpu()).unwrap();
        assert!(loss.numel() == 1);
    }

    #[test]
    fn test_z_loss_zero_logits_is_positive() {
        // log(sum(exp(0))) = log(E) > 0, squared and meaned is still positive
        let logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let loss = z_loss::<CandleBackend>(&logits, &cpu()).unwrap();
        let val = loss.to_vec_f32().unwrap()[0];
        assert!(val > 0.0);
    }

    #[test]
    fn test_z_loss_large_logits_is_larger() {
        // larger logits should produce larger z-loss
        let small = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let large_data = vec![10.0f32; 16];
        let large = CandleTensor::from_slice(&large_data, &Shape::new(&[4, 4]), &cpu()).unwrap();
        let small_loss = z_loss::<CandleBackend>(&small, &cpu()).unwrap();
        let large_loss = z_loss::<CandleBackend>(&large, &cpu()).unwrap();
        let sv = small_loss.to_vec_f32().unwrap()[0];
        let lv = large_loss.to_vec_f32().unwrap()[0];
        assert!(
            lv > sv,
            "large logit z-loss {} should exceed small {}",
            lv,
            sv
        );
    }

    #[test]
    fn test_compute_aux_loss_training_returns_tensors() {
        let config = AuxLossConfig::training();
        let logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let indices =
            CandleTensor::from_u32_slice(&[0, 1, 0, 1, 2, 3, 2, 3], &Shape::new(&[4, 2]), &cpu())
                .unwrap();
        let result =
            compute_aux_loss::<CandleBackend>(&logits, &indices, 4, &config, &cpu()).unwrap();
        assert!(result.load_balance_loss.is_some());
        assert!(result.z_loss.is_some());
        assert!(result.total_aux_loss.is_some());
    }

    #[test]
    fn test_compute_aux_loss_total_is_sum_of_parts() {
        let config = AuxLossConfig::training();
        let logits = CandleTensor::zeros(&Shape::new(&[4, 4]), DType::F32, &cpu()).unwrap();
        let indices =
            CandleTensor::from_u32_slice(&[0, 1, 0, 1, 2, 3, 2, 3], &Shape::new(&[4, 2]), &cpu())
                .unwrap();
        let result =
            compute_aux_loss::<CandleBackend>(&logits, &indices, 4, &config, &cpu()).unwrap();
        let lb = result.load_balance_loss.unwrap().to_vec_f32().unwrap()[0];
        let zl = result.z_loss.unwrap().to_vec_f32().unwrap()[0];
        let total = result.total_aux_loss.unwrap().to_vec_f32().unwrap()[0];
        let expected = config.load_balance_alpha * lb + config.z_loss_beta * zl;
        assert!(
            (total - expected).abs() < 1e-5,
            "total {} != alpha*lb + beta*zl = {}",
            total,
            expected
        );
    }
}
