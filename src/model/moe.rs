use std::eprint;

use crate::core::backend::Backend;
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::CoreError;
use crate::core::error::Result;
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::core::tensor::TopKLastDimOp;
use crate::metal::runner::MetalRunner;
use crate::model::config::ModelConfig;
use crate::model::linear::Linear;
use crate::model::moe_loss::{compute_aux_loss, AuxLossConfig, MoeLossOutput};
use crate::weights::gguf::GgufDType;
use crate::weights::lazy::QuantizedWeight;

/*
The Mixture Of Expert Backend. Very derivative from what vLLM did and the LLM seriving stratergies for the Megatron Core
of NVIDIA. Needs a lot more changes and additions but this is the basic working MoE for now.
*/

//Opaque index into the expert array. Prevents mixing expert indices
//with arbitrary usizes at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertIndex(pub usize);

//Token position within a flattened [batch * seq_len] token stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenPos(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunction {
    Softmax,
    Sigmoid,
}

pub struct ExpertParallelConfig {
    pub ep_size: usize, // 1 for single GPU
    pub ep_rank: usize, // 0 for single GPU
    pub experts_per_rank: usize,
}

#[derive(Debug, Clone)]
pub struct RouterConfig {
    //Total number of routed experts E.
    pub num_experts: usize,
    //How many experts each token activates (top-k, K << E).
    pub experts_per_token: usize,
    //Score function applied after the gating linear projection.
    pub score_fn: ScoreFunction,
    //Multiplier applied to scores before softmax/sigmoid.
    //Set to 1.0 to disable. Used by some DeepSeek variants.
    pub score_scale_factor: f32,
    //If Some(threshold), speculative pre-gating is active.
    //A token whose max score from the *previous* step exceeds this
    //value has its expert set prefetched before the current forward.
    pub prefetch_threshold: Option<f32>,
}

impl RouterConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_local_experts = cfg.num_local_experts.unwrap();
        let num_local_experts_per_tok = cfg.num_experts_per_tok.unwrap();
        let prefetch_threshold = cfg.prefetch_threshold;
        let default_fn = ScoreFunction::Softmax;
        let score_scale_factor: f32 = 1.0;

        Self {
            num_experts: num_local_experts,
            experts_per_token: num_local_experts_per_tok,
            score_fn: default_fn,
            score_scale_factor: score_scale_factor,
            prefetch_threshold: prefetch_threshold,
        }
    }
}

/*
 LatentMoE compression parameters
When present, a shared down-projection maps hidden - latent before
dispatch and a shared up-projection maps latent - hidden after combine.
Experts then operate on the latent dimension, cutting all-to-all volume
by the ratio hidden/latent.
*/

#[derive(Debug, Clone)]
pub struct LatentConfig {
    //Compressed dimension < hidden_size.
    pub latent_dim: usize,
    //hidden_size / latent_dim — precomputed for clarity.
    pub compression_ratio: usize,
}

impl LatentConfig {
    pub fn new(hidden_size: usize, latent_dim: usize) -> Self {
        // your body: assert latent_dim < hidden_size, compute ratio
        assert!(latent_dim < hidden_size);
        let ratio = hidden_size / latent_dim as usize;
        Self {
            latent_dim: latent_dim,
            compression_ratio: ratio,
        }
    }
}

/*
routing data structures
Token-to-expert assignment produced by one Router forward pass.
Shapes are given in terms of T = num_tokens, K = experts_per_token.
*/

#[derive(Debug, Clone)]
pub struct RoutingOutput<B: Backend> {
    //Expert indices each token is assigned to.  shape [T, K].
    pub expert_indices: B::Tensor,
    //Routing weights (after score_fn + top-k renormalisation)
    pub routing_weights: B::Tensor,
    //Raw pre-softmax logits for aux-loss computation. shape [T, E].
    pub router_logits: B::Tensor,
}

/*
Per-token record written after each forward step and read by the
next step's speculative pre-gating pass.
Stored as plain vecs so it lives cheaply on the host between steps.
*/

#[derive(Debug, Clone)]
pub struct SpeculativeRecord {
    pub token_pos: TokenPos,
    //Which expert indices were selected
    pub expert_indices: Vec<ExpertIndex>,
    //Maximum routing score from this step (used for threshold check).
    pub max_score: f32,
}

/*
Fixed-capacity ring buffer holding one SpeculativeRecord per token
position. Written after forward, read before the next forward.
Capacity = max_tokens = batch * max_seq_len.
When a position is not yet populated, the entry is None.
*/

#[derive(Debug)]
pub struct PreGateBuffer {
    // Use a simple Vec to allow runtime capacity
    records: Vec<Option<SpeculativeRecord>>,
    capacity: usize,
}

impl PreGateBuffer {
    pub fn new(capacity: usize) -> Self {
        // Initialize the Vec with 'None' for the given capacity
        let records = vec![None; capacity];

        Self { records, capacity }
    }

    pub fn write(&mut self, pos: TokenPos, record: SpeculativeRecord) {
        // Cast TokenPos to usize for indexing
        let index = pos.0 % self.capacity;
        self.records[index] = Some(record);
    }

    pub fn read(&self, pos: TokenPos) -> Option<&SpeculativeRecord> {
        let index = pos.0 % self.capacity;
        // Check if the record exists and matches the token position
        self.records[index]
            .as_ref()
            .filter(|rec| rec.token_pos == pos)
    }

    pub fn prefetch_candidates(&self, threshold: f32) -> Vec<ExpertIndex> {
        self.records
            .iter()
            .flatten()
            .filter(|rec| rec.max_score > threshold)
            .flat_map(|rec| rec.expert_indices.iter().copied())
            .collect()
    }

    pub fn clear(&mut self) {
        self.records.fill(None);
    }
}

/*
Router:

gating linear - score function - top-k.
The gating linear maps [*, hidden] - [*, E].
Router weights are always duplicated across EP ranks (not sharded),
so Router is unaware of parallelism — it sees the full hidden vector.
*/

pub struct Router<B: Backend> {
    gate: Linear<B>,
    config: RouterConfig,
}

impl<B: Backend> Router<B> {
    //Construct with zero-initialised weights. Weights are loaded later
    //via checkpoint loading
    pub fn new(hidden_size: usize, config: RouterConfig, device: &Device) -> Result<Self> {
        let make_linear = |out: usize, inp: usize| -> Result<Linear<B>> {
            let w = B::Tensor::zeros(&Shape::new(&[out, inp]), DType::F32, device)?;
            Ok(Linear::new(w, None))
        };

        let gate_proj = make_linear(config.num_experts, hidden_size)?;

        Ok(Self {
            gate: gate_proj,
            config: config,
        })
    }

    /* Full router forward: logits - score_fn - top-k - RoutingOutput.
       `x` shape: [batch, seq_len, hidden]  or  [T, hidden] after flattening.
    */
    pub fn forward(&self, x: &B::Tensor) -> Result<RoutingOutput<B>> {
        // your body
        let batch = x.shape().dim(0)?;
        let seq_len = x.shape().dim(1)?;
        let hidden = x.shape().dim(2)?;
        let t = batch * seq_len;
        let x_flattened = x.reshape(&Shape::new(&[t, hidden]))?;

        let logits_result = self.gate.forward(&x_flattened);

        let mut logits = logits_result?;

        if self.config.score_scale_factor != 1.0 {
            logits = logits.scale(self.config.score_scale_factor as f64)?;
        }

        let scores = match self.config.score_fn {
            ScoreFunction::Sigmoid => logits.sigmoid()?,
            ScoreFunction::Softmax => logits.softmax(logits.rank() - 1)?,
        };

        let topk_out = TopKLastDimOp::topk(&scores, self.config.experts_per_token)?;
        let routing_weights = topk_out.values;
        let expert_indices = topk_out.indices;

        let final_weights = if let ScoreFunction::Softmax = self.config.score_fn {
            let sum = routing_weights.sum_keepdim(routing_weights.rank() - 1)?;
            routing_weights.broadcast_div_rows(&sum)?
        } else {
            routing_weights
        };

        Ok(RoutingOutput {
            routing_weights: final_weights,
            expert_indices: expert_indices,
            router_logits: logits,
        })
    }

    /*
    Speculative correction pass
    Given the RoutingOutput computed from the *current* hidden state
    and the predicted routing from `buffer`, compute and return a
    correction mask: a bool-like tensor of shape [T, K] that is 1.0
    where the speculative choice matches actual, 0.0 where it differs.
    The MoeLayer uses this mask to zero-out the contribution of
    mis-predicted experts and re-compute only those tokens.
     */

    pub fn speculative_correction(
        actual: &RoutingOutput<B>,
        buffer: &PreGateBuffer,
        device: &Device,
    ) -> Result<B::Tensor> {
        let t = actual.expert_indices.shape().dim(0)?;
        let k = actual.expert_indices.shape().dim(1)?;

        let actual_indices_flat: Vec<u32> = actual.expert_indices.to_vec_u32()?;
        let mut mask_data = vec![0.0_f32; t * k];

        for token_pos in 0..t {
            //Check if we made a prediction for this token
            if let Some(rec) = buffer.read(TokenPos(token_pos)) {
                for k_slot in 0..k {
                    let flat_idx = token_pos * k + k_slot;

                    //Get the actual expert chosen by the router
                    let actual_expert = ExpertIndex(actual_indices_flat[flat_idx] as usize);

                    //Get the predicted expert from the buffer
                    if let Some(&predicted_expert) = rec.expert_indices.get(k_slot) {
                        if predicted_expert == actual_expert {
                            mask_data[flat_idx] = 1.0;
                        }
                    }
                }
            }
        }

        //Wrap back into a [T, K] tensor on the target device
        B::Tensor::from_slice(&mask_data, &Shape::new(&[t, k]), device)
    }

    pub fn load_balance_loss(
        router_logits: &B::Tensor,  // [T, E]
        expert_indices: &B::Tensor, // [T, K]
        num_experts: usize,
    ) -> Result<B::Tensor> {
        let device = router_logits.device().clone();
        crate::model::moe_loss::load_balance_loss::<B>(
            router_logits,
            expert_indices,
            num_experts,
            &device,
        )
    }

    /// Build a Router from an already-loaded gate weight, skipping the
    /// zero-allocation Router::new does.
    pub fn from_weight(gate_weight: B::Tensor, config: RouterConfig) -> Self {
        Self {
            gate: Linear::new(gate_weight, None),
            config,
        }
    }
}

/* Expert  (single routed expert)
One routed expert: a two-layer SwiGLU MLP operating on dimension `d`.
In standard MoE: d = hidden_size
In LatentMoE: d = latent_dim   (experts never see full hidden)
each expert is `gate_proj`, `up_proj`, `down_proj`.
gate and up are parallel; their hadamard product feeds down.
*/

#[derive(Clone)]
pub struct Expert<B: Backend> {
    gate_proj: Linear<B>, // [intermediate, d]
    up_proj: Linear<B>,   // [intermediate, d]
    down_proj: Linear<B>, // [d, intermediate]
    //Cached expert identity — used in dispatch bookkeeping.
    pub index: ExpertIndex,
}

impl<B: Backend> Expert<B> {
    //`in_dim` is hidden_size for standard MoE, latent_dim for LatentMoE.
    pub fn new(
        index: ExpertIndex,
        in_dim: usize,
        intermediate_size: usize,
        device: &Device,
    ) -> Result<Self> {
        // your body
        let make_linear = |out: usize, inp: usize| -> Result<Linear<B>> {
            let w = B::Tensor::zeros(&Shape::new(&[out, inp]), DType::F32, device)?;
            Ok(Linear::new(w, None))
        };

        let gate_proj = make_linear(intermediate_size, in_dim)?;
        let up_proj = make_linear(intermediate_size, in_dim)?;
        let down_proj = make_linear(in_dim, intermediate_size)?;

        Ok(Self {
            gate_proj: gate_proj,
            up_proj: up_proj,
            down_proj: down_proj,
            index: index,
        })
    }

    pub fn forward(&self, x: &B::Tensor) -> Result<B::Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }

    pub fn from_weights(
        gate_w: B::Tensor,
        up_w: B::Tensor,
        down_w: B::Tensor,
        index: ExpertIndex,
    ) -> Self {
        Self {
            gate_proj: Linear::new(gate_w, None),
            up_proj: Linear::new(up_w, None),
            down_proj: Linear::new(down_w, None),
            index,
        }
    }
}

pub struct LazyExpertBank<B: Backend> {
    gate_exps: QuantizedWeight, // raw bytes, shape-equivalent to [num_experts * intermediate, hidden]
    up_exps: QuantizedWeight,   // same layout
    down_exps: QuantizedWeight, // [num_experts * hidden, intermediate]
    num_experts: usize,
    intermediate_size: usize,
    hidden_size: usize,
    device: Device,
    _marker: std::marker::PhantomData<B>,
}

// as opposed to the eager expert.

impl<B: Backend> LazyExpertBank<B> {
    pub fn new(
        gate_exps: QuantizedWeight,
        up_exps: QuantizedWeight,
        down_exps: QuantizedWeight,
        num_experts: usize,
        intermediate_size: usize,
        hidden_size: usize,
        device: Device,
    ) -> Self {
        Self {
            gate_exps,
            up_exps,
            down_exps,
            num_experts,
            intermediate_size,
            hidden_size,
            device,
            _marker: std::marker::PhantomData,
        }
    }

    /// Dequantize just expert `idx`'s three weight matrices and hand back a
    /// throwaway Expert -- everything about it (the F32 tensors, the
    /// wrapping Linears) drops as soon as the caller is done with it.
    pub fn materialize(&self, idx: ExpertIndex) -> Result<Expert<B>> {
        let i = self.intermediate_size;
        let h = self.hidden_size;

        let gate_rows = self.gate_exps.materialize_rows(idx.0 * i, (idx.0 + 1) * i);
        let up_rows = self.up_exps.materialize_rows(idx.0 * i, (idx.0 + 1) * i);
        let down_rows = self.down_exps.materialize_rows(idx.0 * h, (idx.0 + 1) * h);

        let gate_w = B::Tensor::from_slice(&gate_rows, &Shape::new(&[i, h]), &self.device)?;
        let up_w = B::Tensor::from_slice(&up_rows, &Shape::new(&[i, h]), &self.device)?;
        let down_w = B::Tensor::from_slice(&down_rows, &Shape::new(&[h, i]), &self.device)?;

        Ok(Expert::from_weights(gate_w, up_w, down_w, idx))
    }
}

/*
 Shared Expert

Processes *all* tokens regardless of routing.
Architecture identical to Expert but always receives the full token
stream

The shared expert always operates on hidden_size, not latent_dim,
because it runs before the down-projection in LatentMoE.

Did not implement dispatch-compute-combine on top of this
*/
pub struct SharedExpert<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
    pub num_shared: usize,
}

impl<B: Backend> SharedExpert<B> {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_shared: usize,
        device: &Device,
    ) -> Result<Self> {
        let make_linear = |out: usize, inp: usize| -> Result<Linear<B>> {
            let w = B::Tensor::zeros(&Shape::new(&[out, inp]), DType::F32, device)?;
            Ok(Linear::new(w, None))
        };

        let gate_proj = make_linear(intermediate_size * num_shared, hidden_size)?;
        let up_proj = make_linear(intermediate_size * num_shared, hidden_size)?;
        let down_proj = make_linear(hidden_size, intermediate_size * num_shared)?;

        Ok(Self {
            gate_proj: gate_proj,
            up_proj: up_proj,
            down_proj: down_proj,
            num_shared: num_shared,
        })
    }

    //Returns [T, hidden_size].
    //The shared expert output is *added* to the routed expert output
    //in MoeLayer::forward — not gated.
    pub fn forward(&self, x: &B::Tensor) -> Result<B::Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let fused = gate.mul(&up)?;
        self.down_proj.forward(&fused)
    }
}

//Latent Projections

pub struct LatentProjection<B: Backend> {
    //stored as [latent, hidden].
    pub down_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub config: LatentConfig,
}

impl<B: Backend> LatentProjection<B> {
    pub fn new(config: LatentConfig, device: &Device, hidden_size: usize) -> Result<Self> {
        let make_linear = |out: usize, inp: usize| -> Result<Linear<B>> {
            let w = B::Tensor::zeros(&Shape::new(&[out, inp]), DType::F32, device)?;
            Ok(Linear::new(w, None))
        };

        let down_proj = make_linear(config.latent_dim, hidden_size)?;
        let up_proj = make_linear(hidden_size, config.latent_dim)?;

        Ok(Self {
            down_proj,
            up_proj,
            config,
        })
    }

    pub fn project_down(&self, x: &B::Tensor) -> Result<B::Tensor> {
        self.down_proj.forward(x)
    }

    pub fn project_up(&self, x: &B::Tensor) -> Result<B::Tensor> {
        self.up_proj.forward(x)
    }
}

// dispatch/combine(CPU-side, single device)

/*
 Gather the tokens assigned to one expert from the full token stream.
`token_features`  shape: [T, d]
`expert_idx`      which expert we are gathering for
`routing_output`  contains expert_indices [T, K]
*/

pub fn dispatch<B: Backend>(
    token_features: &B::Tensor,
    expert_idx: ExpertIndex,
    routing_output: &RoutingOutput<B>,
) -> Result<(B::Tensor, Vec<TokenPos>, Vec<usize>)> {
    let indices_flat: Vec<u32> = routing_output.expert_indices.to_vec_u32()?;
    let t_total = routing_output.expert_indices.shape().dim(0)?;
    let k = routing_output.expert_indices.shape().dim(1)?;

    let mut selected_indices = Vec::new();
    let mut original_positions = Vec::new();
    let mut k_slots = Vec::new();

    let device = token_features.device();

    for t in 0..t_total {
        for k_slot in 0..k {
            let flat_idx = t * k + k_slot;
            let assigned_expert = indices_flat[flat_idx] as usize;
            if assigned_expert == expert_idx.0 {
                // token t is dispatched to this expert at slot k_slot
                selected_indices.push(t as usize);
                original_positions.push(TokenPos(t));
                k_slots.push(k_slot);
            }
        }
    }

    //If no tokens were assigned to this expert, handle the empty case
    if selected_indices.is_empty() {
        let hidden_dim = token_features.shape().dims().last().unwrap_or(&0);
        // Return an empty tensor with shape [0, hidden_dim]
        let empty_tensor = B::Tensor::zeros(&Shape::new(&[0, *hidden_dim]), DType::F32, device)?;
        return Ok((empty_tensor, original_positions, k_slots));
    }

    let indices_u32: Vec<u32> = selected_indices.iter().map(|&i| i as u32).collect();
    let indices_tensor =
        B::Tensor::from_u32_slice(&indices_u32, &Shape::new(&[indices_u32.len()]), device)?;
    let dispatched_features = token_features.index_select(&indices_tensor, 0)?;

    Ok((dispatched_features, original_positions, k_slots))
}

/*
Scatter expert outputs back into the full output buffer and accumulate
weighted. This is the inverse of dispatch.
`expert_out`          shape: [n, d] — output of Expert::forward
`routing_weights`     shape: [T, K] — from RoutingOutput
`positions`           which token rows these n outputs came from
`expert_k_slot`       which k-slot (0..K) this expert occupies per token
`output_accumulator`  mutable [T, d] buffer — accumulated in place
                       (represented as a Vec of row tensors for now)
*/

pub fn combine<B: Backend>(
    expert_out: &B::Tensor,
    routing_weights: &B::Tensor,
    positions: &[TokenPos],
    expert_k_slot: &[usize],
    output_accumulator: &mut Vec<Option<B::Tensor>>,
) -> Result<()> {
    let n = expert_out.shape().dim(0)?;
    let _d = expert_out.shape().dim(1)?;

    for i in 0..n {
        let t = positions[i].0;
        let k = expert_k_slot[i];
        let weight = routing_weights
            .narrow(0, t, 1)? // Row t
            .narrow(1, k, 1)?;

        //Get the specific row from the expert's output
        let expert_row = expert_out.narrow(0, i, 1)?;

        //Scale the row by the weight
        let scaled_row = expert_row.broadcast_mul(&weight)?;

        if let Some(existing_tensor) = &output_accumulator[t] {
            // If another expert already contributed to this token, add to it
            output_accumulator[t] = Some(existing_tensor.add(&scaled_row)?);
        } else {
            // First expert contributing to this token
            output_accumulator[t] = Some(scaled_row);
        }
    }

    Ok(())
}

// Moe Top layer

#[derive(Debug)]
pub struct MoeOutput<B: Backend> {
    pub hidden_states: B::Tensor,
    pub aux_loss: Option<B::Tensor>,
    // replace the placeholder with the real loss output
    pub loss_output: Option<MoeLossOutput<B>>,
}

pub struct MoeLayer<B: Backend> {
    pub router: Router<B>,
    pub experts: Vec<Expert<B>>,
    pub shared_expert: Option<SharedExpert<B>>,
    pub latent: Option<LatentProjection<B>>,
    //Lives on the layer; reset between unrelated sequences.
    pub pregate_buffer: Option<PreGateBuffer>,
    pub config: RouterConfig,
    pub hidden_size: usize,
    //The dimension experts actually compute over.
    //latent_dim if LatentMoE, else hidden_size.
    pub expert_dim: usize,
    pub aux_loss_config: AuxLossConfig,
}

impl<B: Backend> MoeLayer<B> {
    pub fn new(config: &ModelConfig, latent_dim: Option<usize>, device: &Device) -> Result<Self> {
        let router_config = RouterConfig::from_model_config(config);
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let num_experts = config.num_local_experts.unwrap_or(1);
        let num_shared = config.num_shared_experts.unwrap_or(0);
        let expert_dim = latent_dim.unwrap_or(hidden_size);

        let router = Router::new(hidden_size, router_config.clone(), device)?;

        let experts = (0..num_experts)
            .map(|i| Expert::new(ExpertIndex(i), expert_dim, intermediate_size, device))
            .collect::<Result<Vec<_>>>()?;

        let shared_expert = if num_shared > 0 {
            Some(SharedExpert::new(
                hidden_size,
                intermediate_size,
                num_shared,
                device,
            )?)
        } else {
            None
        };

        let latent = if let Some(ldim) = latent_dim {
            Some(LatentProjection::new(
                LatentConfig::new(hidden_size, ldim),
                device,
                hidden_size,
            )?)
        } else {
            None
        };

        let pregate_buffer = config
            .prefetch_threshold
            .map(|_| PreGateBuffer::new(config.max_position_embeddings));

        Ok(Self {
            router,
            experts,
            shared_expert,
            latent,
            pregate_buffer,
            config: router_config,
            hidden_size,
            expert_dim,
            aux_loss_config: AuxLossConfig::inference(),
        })
    }

    /*
     Full forward pass. See struct-level doc for the flow.

    `token_offset` — starting position in the sequence; used to index
    into pregate_buffer correctly for KV-cache generation steps.
    */

    pub fn forward(&mut self, x: &B::Tensor, token_offset: usize) -> Result<MoeOutput<B>> {
        let batch = x.shape().dim(0)?;
        let seq_len = x.shape().dim(1)?;
        let hidden_dim = x.shape().dim(2)?;
        let t = batch * seq_len;
        let routing_output = self.router.forward(x)?;

        let loss_output = compute_aux_loss::<B>(
            &routing_output.router_logits,
            &routing_output.expert_indices,
            self.experts.len(),
            &self.aux_loss_config,
            x.device(),
        )
        .ok();

        let x_flat = x.reshape(&Shape::new(&[t, hidden_dim]))?;

        let shared_out = if let Some(shared) = &self.shared_expert {
            Some(shared.forward(&x_flat)?)
        } else {
            None
        };

        let routed_input = if let Some(proj) = &self.latent {
            proj.project_down(&x_flat)?
        } else {
            x_flat.clone()
        };

        // route tokens
        // speculative correction — if we have a pregate buffer,
        // compute the correction mask and store it for logging/metrics
        // does not change routing, just identifies mispredictions
        let _correction_mask = if let Some(ref buf) = self.pregate_buffer {
            let device = x.device();
            Router::<B>::speculative_correction(&routing_output, buf, device).ok()
        } else {
            None
        };

        // dispatch to expert to combine
        let mut accumulator: Vec<Option<B::Tensor>> = vec![None; t];

        for e in 0..self.experts.len() {
            let (gathered, positions, k_slots) =
                dispatch::<B>(&routed_input, ExpertIndex(e), &routing_output)?;
            if positions.is_empty() {
                continue;
            }
            let expert_out = self.experts[e].forward(&gathered)?;
            combine::<B>(
                &expert_out,
                &routing_output.routing_weights,
                &positions,
                &k_slots,
                &mut accumulator,
            )?;
        }

        // stack accumulator rows into [T, expert_dim]
        let routed_refs: Vec<&B::Tensor> = accumulator
            .iter()
            .map(|opt| {
                opt.as_ref()
                    .expect("every token must have at least one expert")
            })
            .collect();
        let mut routed_out = B::Tensor::cat(&routed_refs, 0)?;

        // project back to hidden if latent
        if let Some(proj) = &self.latent {
            routed_out = proj.project_up(&routed_out)?;
        }

        // add shared expert output
        let mut hidden_states = if let Some(s) = shared_out {
            routed_out.add(&s)?
        } else {
            routed_out
        };

        // update pregate buffer
        self.update_pregate_buffer(&routing_output, token_offset, t)?;

        // reshape back to [batch, seq_len, hidden]
        hidden_states = hidden_states.reshape(&Shape::new(&[batch, seq_len, hidden_dim]))?;

        Ok(MoeOutput {
            hidden_states,
            aux_loss: loss_output
                .as_ref()
                .and_then(|l| l.total_aux_loss.as_ref())
                .cloned(),
            loss_output,
        })
    }

    /*
    Write routing decisions into PreGateBuffer after a forward pass.
    Call this at the *end* of forward so the next step can read it.
    Converts routing_output.routing_weights to host scalars to build
    SpeculativeRecord per token.
    */

    fn update_pregate_buffer(
        &mut self,
        routing_output: &RoutingOutput<B>,
        token_offset: usize,
        num_tokens: usize,
    ) -> Result<()> {
        let buf = match &mut self.pregate_buffer {
            Some(b) => b,
            None => return Ok(()),
        };

        let weights_flat = routing_output.routing_weights.to_vec_f32()?;
        let indices_flat = routing_output.expert_indices.to_vec_u32()?;
        let k = routing_output.routing_weights.shape().dim(1)?;

        for t in 0..num_tokens {
            let max_score = (0..k)
                .map(|ki| weights_flat[t * k + ki])
                .fold(f32::NEG_INFINITY, f32::max);

            let expert_indices: Vec<ExpertIndex> = (0..k)
                .map(|ki| ExpertIndex(indices_flat[t * k + ki] as usize))
                .collect();

            buf.write(
                TokenPos(token_offset + t),
                SpeculativeRecord {
                    token_pos: TokenPos(token_offset + t),
                    expert_indices,
                    max_score,
                },
            );
        }
        Ok(())
    }

    /*
     Reset the pre-gate buffer. Call between unrelated sequences
    /// (e.g. a new prompt when batch_size > 1 sequences share a layer).
    */

    pub fn reset_speculation(&mut self) {
        if let Some(buf) = &mut self.pregate_buffer {
            buf.clear();
        }
    }

    pub fn set_training_mode(&mut self, config: AuxLossConfig) {
        self.aux_loss_config = config;
    }

    //Returns the set of expert indices that should be prefetched
    //for the next step based on high-confidence predictions in the buffer.
    //Called before forward() in the generation loop.
    pub fn prefetch_experts(&self) -> Vec<ExpertIndex> {
        match (&self.pregate_buffer, self.config.prefetch_threshold) {
            (Some(buf), Some(threshold)) => buf.prefetch_candidates(threshold),
            _ => vec![],
        }
    }

    pub fn is_prefetched(&self, e: ExpertIndex) -> bool {
        self.prefetch_experts().contains(&e)
    }
}

pub struct LazyMoeLayer<B: Backend> {
    pub router: Router<B>,
    pub expert_bank: LazyExpertBank<B>,
    pub shared_expert: Option<SharedExpert<B>>,
    pub latent: Option<LatentProjection<B>>,
    pub pregate_buffer: Option<PreGateBuffer>,
    pub config: RouterConfig,
    pub hidden_size: usize,
    pub expert_dim: usize,
    pub aux_loss_config: AuxLossConfig,
}

impl<B: Backend> LazyMoeLayer<B> {
    /// Mirrors MoeLayer::new, except the loader hands in an already-loaded
    /// router gate weight and expert_bank directly, instead of a
    /// ModelConfig this constructs zero-initialized weights from.
    /// Qwen3-30B-A3B needs shared_expert: None, latent_dim: None -- both
    /// kept as parameters for parity with MoeLayer / future models that do
    /// use them.
    pub fn new(
        router_gate_weight: B::Tensor,
        expert_bank: LazyExpertBank<B>,
        shared_expert: Option<SharedExpert<B>>,
        latent_dim: Option<usize>,
        config: &ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let router_config = RouterConfig::from_model_config(config);
        let hidden_size = config.hidden_size;
        let expert_dim = latent_dim.unwrap_or(hidden_size);

        let router = Router::from_weight(router_gate_weight, router_config.clone());

        let latent = if let Some(ldim) = latent_dim {
            Some(LatentProjection::new(
                LatentConfig::new(hidden_size, ldim),
                device,
                hidden_size,
            )?)
        } else {
            None
        };

        let pregate_buffer = config
            .prefetch_threshold
            .map(|_| PreGateBuffer::new(config.max_position_embeddings));

        Ok(Self {
            router,
            expert_bank,
            shared_expert,
            latent,
            pregate_buffer,
            config: router_config,
            hidden_size,
            expert_dim,
            aux_loss_config: AuxLossConfig::inference(),
        })
    }

    pub fn forward(
        &mut self,
        x: &B::Tensor,
        token_offset: usize,
        mut runner: Option<&mut MetalRunner>,
    ) -> Result<MoeOutput<B>> {
        let batch = x.shape().dim(0)?;
        let seq_len = x.shape().dim(1)?;
        let hidden_dim = x.shape().dim(2)?;
        let t = batch * seq_len;
        let routing_output = self.router.forward(x)?;

        let loss_output = compute_aux_loss::<B>(
            &routing_output.router_logits,
            &routing_output.expert_indices,
            self.expert_bank.num_experts,
            &self.aux_loss_config,
            x.device(),
        )
        .ok();

        let x_flat = x.reshape(&Shape::new(&[t, hidden_dim]))?;

        let shared_out = if let Some(shared) = &self.shared_expert {
            Some(shared.forward(&x_flat)?)
        } else {
            None
        };

        let routed_input = if let Some(proj) = &self.latent {
            proj.project_down(&x_flat)?
        } else {
            x_flat.clone()
        };

        let _correction_mask = if let Some(ref buf) = self.pregate_buffer {
            let device = x.device();
            Router::<B>::speculative_correction(&routing_output, buf, device).ok()
        } else {
            None
        };

        let mut accumulator: Vec<Option<B::Tensor>> = vec![None; t];
        for e in 0..self.expert_bank.num_experts {
            let (gathered, positions, k_slots) =
                dispatch::<B>(&routed_input, ExpertIndex(e), &routing_output)?;
            if positions.is_empty() {
                continue;
            }
            let expert = self.expert_bank.materialize(ExpertIndex(e))?;

            let expert_out = if let Some(r) = runner.as_deref_mut() {
                Self::expert_forward_via_runner(&expert, &gathered, r)?
            } else {
                expert.forward(&gathered)?
            };

            combine::<B>(
                &expert_out,
                &routing_output.routing_weights,
                &positions,
                &k_slots,
                &mut accumulator,
            )?;
        }

        let routed_refs: Vec<&B::Tensor> = accumulator
            .iter()
            .map(|opt| {
                opt.as_ref()
                    .expect("every token must have at least one expert")
            })
            .collect();
        let mut routed_out = B::Tensor::cat(&routed_refs, 0)?;

        if let Some(proj) = &self.latent {
            routed_out = proj.project_up(&routed_out)?;
        }

        let mut hidden_states = if let Some(s) = shared_out {
            routed_out.add(&s)?
        } else {
            routed_out
        };

        self.update_pregate_buffer(&routing_output, token_offset, t)?;

        hidden_states = hidden_states.reshape(&Shape::new(&[batch, seq_len, hidden_dim]))?;

        Ok(MoeOutput {
            hidden_states,
            aux_loss: loss_output
                .as_ref()
                .and_then(|l| l.total_aux_loss.as_ref())
                .cloned(),
            loss_output,
        })
    }

    fn expert_forward_via_runner(
        expert: &Expert<B>,
        gathered: &B::Tensor,
        runner: &mut MetalRunner,
    ) -> Result<B::Tensor> {
        let n_tok = gathered.shape().dim(0)?;
        let intermediate = expert.gate_proj.weight().shape().dim(0)?; // [intermediate, hidden]
        let hidden = expert.down_proj.weight().shape().dim(0)?; // [hidden, intermediate]
        let device = gathered.device();

        let gate_2d = B::Tensor::uninit_pooled(
            &Shape::new(&[n_tok, intermediate]),
            gathered.dtype(),
            device,
        )?;
        runner.matmul(
            gathered
                .as_metal()
                .ok_or_else(|| CoreError::Internal("gathered not Metal".into()))?,
            expert
                .gate_proj
                .weight()
                .as_metal()
                .ok_or_else(|| CoreError::Internal("gate weight not Metal".into()))?,
            gate_2d
                .as_metal()
                .ok_or_else(|| CoreError::Internal("gate_2d not Metal".into()))?,
        )?;

        let up_2d = B::Tensor::uninit_pooled(
            &Shape::new(&[n_tok, intermediate]),
            gathered.dtype(),
            device,
        )?;
        runner.matmul(
            gathered
                .as_metal()
                .ok_or_else(|| CoreError::Internal("gathered not Metal".into()))?,
            expert
                .up_proj
                .weight()
                .as_metal()
                .ok_or_else(|| CoreError::Internal("up weight not Metal".into()))?,
            up_2d
                .as_metal()
                .ok_or_else(|| CoreError::Internal("up_2d not Metal".into()))?,
        )?;

        let swiglu_out = B::Tensor::uninit_pooled(
            &Shape::new(&[n_tok, intermediate]),
            gathered.dtype(),
            device,
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
            (n_tok * intermediate) as u32,
        )?;

        let down_2d =
            B::Tensor::uninit_pooled(&Shape::new(&[n_tok, hidden]), gathered.dtype(), device)?;
        runner.matmul(
            swiglu_out
                .as_metal()
                .ok_or_else(|| CoreError::Internal("swiglu_out not Metal".into()))?,
            expert
                .down_proj
                .weight()
                .as_metal()
                .ok_or_else(|| CoreError::Internal("down weight not Metal".into()))?,
            down_2d
                .as_metal()
                .ok_or_else(|| CoreError::Internal("down_2d not Metal".into()))?,
        )?;

        Ok(down_2d)
    }

    fn update_pregate_buffer(
        &mut self,
        routing_output: &RoutingOutput<B>,
        token_offset: usize,
        num_tokens: usize,
    ) -> Result<()> {
        let buf = match &mut self.pregate_buffer {
            Some(b) => b,
            None => return Ok(()),
        };

        let weights_flat = routing_output.routing_weights.to_vec_f32()?;
        let indices_flat = routing_output.expert_indices.to_vec_u32()?;
        let k = routing_output.routing_weights.shape().dim(1)?;

        for t in 0..num_tokens {
            let max_score = (0..k)
                .map(|ki| weights_flat[t * k + ki])
                .fold(f32::NEG_INFINITY, f32::max);

            let expert_indices: Vec<ExpertIndex> = (0..k)
                .map(|ki| ExpertIndex(indices_flat[t * k + ki] as usize))
                .collect();

            buf.write(
                TokenPos(token_offset + t),
                SpeculativeRecord {
                    token_pos: TokenPos(token_offset + t),
                    expert_indices,
                    max_score,
                },
            );
        }
        Ok(())
    }

    pub fn reset_speculation(&mut self) {
        if let Some(buf) = &mut self.pregate_buffer {
            buf.clear();
        }
    }

    /// Skeleton with an empty expert bank and a zero-initialized router
    /// gate -- built at Block::new time, before any real GGUF tensor bytes
    /// have been read. The loader replaces this wholesale via
    /// Block::set_lazy_moe once it's collected the layer's real tensors.
    pub fn placeholder(config: &ModelConfig, device: &Device) -> Result<Self> {
        let router_config = RouterConfig::from_model_config(config);
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let num_experts = config.num_local_experts.unwrap_or(1);

        let gate_weight =
            B::Tensor::zeros(&Shape::new(&[num_experts, hidden_size]), DType::F32, device)?;
        let router = Router::from_weight(gate_weight, router_config.clone());

        let empty = |shape: Vec<usize>| QuantizedWeight {
            dtype: GgufDType::F32,
            data: Vec::new(),
            shape,
            numel: 0,
        };
        let expert_bank = LazyExpertBank::new(
            empty(vec![num_experts * intermediate_size, hidden_size]),
            empty(vec![num_experts * intermediate_size, hidden_size]),
            empty(vec![num_experts * hidden_size, intermediate_size]),
            num_experts,
            intermediate_size,
            hidden_size,
            device.clone(),
        );

        let pregate_buffer = config
            .prefetch_threshold
            .map(|_| PreGateBuffer::new(config.max_position_embeddings));

        Ok(Self {
            router,
            expert_bank,
            shared_expert: None,
            latent: None,
            pregate_buffer,
            config: router_config,
            hidden_size,
            expert_dim: hidden_size,
            aux_loss_config: AuxLossConfig::inference(),
        })
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
    use crate::model::config::ModelConfig;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn make_moe_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            vocab_size: 1000,
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
    fn test_pregate_buffer_write_read() {
        let mut buf = PreGateBuffer::new(16);
        let record = SpeculativeRecord {
            token_pos: TokenPos(3),
            expert_indices: vec![ExpertIndex(0), ExpertIndex(2)],
            max_score: 0.8,
        };
        buf.write(TokenPos(3), record);
        let read = buf.read(TokenPos(3));
        assert!(read.is_some());
        assert_eq!(read.unwrap().max_score, 0.8);
    }

    #[test]
    fn test_pregate_buffer_miss() {
        let buf = PreGateBuffer::new(16);
        assert!(buf.read(TokenPos(5)).is_none());
    }

    #[test]
    fn test_pregate_buffer_wraparound() {
        let mut buf = PreGateBuffer::new(4);
        let record = SpeculativeRecord {
            token_pos: TokenPos(0),
            expert_indices: vec![ExpertIndex(1)],
            max_score: 0.5,
        };
        buf.write(TokenPos(0), record.clone());
        // position 4 wraps to slot 0 — should NOT return old record
        let read = buf.read(TokenPos(4));
        assert!(read.is_none());
    }

    #[test]
    fn test_pregate_buffer_clear() {
        let mut buf = PreGateBuffer::new(16);
        buf.write(
            TokenPos(0),
            SpeculativeRecord {
                token_pos: TokenPos(0),
                expert_indices: vec![ExpertIndex(0)],
                max_score: 0.9,
            },
        );
        buf.clear();
        assert!(buf.read(TokenPos(0)).is_none());
    }

    #[test]
    fn test_pregate_buffer_prefetch_candidates() {
        let mut buf = PreGateBuffer::new(16);
        buf.write(
            TokenPos(0),
            SpeculativeRecord {
                token_pos: TokenPos(0),
                expert_indices: vec![ExpertIndex(1), ExpertIndex(3)],
                max_score: 0.9, // above threshold
            },
        );
        buf.write(
            TokenPos(1),
            SpeculativeRecord {
                token_pos: TokenPos(1),
                expert_indices: vec![ExpertIndex(0)],
                max_score: 0.1, // below threshold
            },
        );
        let candidates = buf.prefetch_candidates(0.5);
        assert_eq!(candidates.len(), 2);
        assert!(candidates.contains(&ExpertIndex(1)));
        assert!(candidates.contains(&ExpertIndex(3)));
    }

    // --- RouterConfig ---

    #[test]
    fn test_router_config_from_model_config() {
        let config = make_moe_config();
        let rc = RouterConfig::from_model_config(&config);
        assert_eq!(rc.num_experts, 4);
        assert_eq!(rc.experts_per_token, 2);
        assert_eq!(rc.score_scale_factor, 1.0);
        assert!(rc.prefetch_threshold.is_some());
    }

    // --- Expert ---

    #[test]
    fn test_expert_forward_shape() {
        let expert = Expert::<CandleBackend>::new(ExpertIndex(0), 32, 64, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[4, 32]), DType::F32, &cpu()).unwrap();
        let out = expert.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[4, 32]));
    }

    #[test]
    fn test_expert_index_stored() {
        let expert = Expert::<CandleBackend>::new(ExpertIndex(3), 32, 64, &cpu()).unwrap();
        assert_eq!(expert.index, ExpertIndex(3));
    }

    // --- SharedExpert ---

    #[test]
    fn test_shared_expert_forward_shape() {
        let shared = SharedExpert::<CandleBackend>::new(32, 64, 2, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[8, 32]), DType::F32, &cpu()).unwrap();
        let out = shared.forward(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[8, 32]));
    }

    // --- LatentProjection ---

    #[test]
    fn test_latent_projection_down() {
        let config = LatentConfig::new(32, 16);
        let proj = LatentProjection::<CandleBackend>::new(config, &cpu(), 32).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[4, 32]), DType::F32, &cpu()).unwrap();
        let out = proj.project_down(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[4, 16]));
    }

    #[test]
    fn test_latent_projection_up() {
        let config = LatentConfig::new(32, 16);
        let proj = LatentProjection::<CandleBackend>::new(config, &cpu(), 32).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[4, 16]), DType::F32, &cpu()).unwrap();
        let out = proj.project_up(&x).unwrap();
        assert_eq!(out.shape(), &Shape::new(&[4, 32]));
    }

    #[test]
    fn test_latent_config_compression_ratio() {
        let config = LatentConfig::new(32, 8);
        assert_eq!(config.compression_ratio, 4);
    }

    // --- Router ---

    #[test]
    fn test_router_forward_output_shapes() {
        let config = make_moe_config();
        let rc = RouterConfig::from_model_config(&config);
        let router = Router::<CandleBackend>::new(32, rc, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = router.forward(&x).unwrap();
        // T = 1 * 4 = 4 tokens, K = 2 experts per token
        assert_eq!(out.routing_weights.shape(), &Shape::new(&[4, 2]));
        assert_eq!(out.expert_indices.shape(), &Shape::new(&[4, 2]));
        assert_eq!(out.router_logits.shape(), &Shape::new(&[4, 4]));
    }

    #[test]
    fn test_router_indices_in_range() {
        let config = make_moe_config();
        let rc = RouterConfig::from_model_config(&config);
        let router = Router::<CandleBackend>::new(32, rc, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = router.forward(&x).unwrap();
        let indices = out.expert_indices.to_vec_u32().unwrap();
        for idx in indices {
            assert!(idx < 4, "expert index out of range: {}", idx);
        }
    }

    // --- dispatch ---

    #[test]
    fn test_dispatch_returns_correct_tokens() {
        let config = make_moe_config();
        let rc = RouterConfig::from_model_config(&config);
        let router = Router::<CandleBackend>::new(32, rc, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let routing = router.forward(&x).unwrap();
        let token_features =
            CandleTensor::zeros(&Shape::new(&[4, 32]), DType::F32, &cpu()).unwrap();

        // at least one expert should get some tokens across all 4 tokens with k=2
        let mut total_dispatched = 0;
        for e in 0..4 {
            let (gathered, positions, k_slots) =
                dispatch::<CandleBackend>(&token_features, ExpertIndex(e), &routing).unwrap();
            assert_eq!(positions.len(), k_slots.len());
            total_dispatched += positions.len();
        }
        // with 4 tokens and k=2, total dispatches must equal 4*2=8
        assert_eq!(total_dispatched, 8);
    }

    #[test]
    fn test_dispatch_empty_expert() {
        // create routing that assigns all tokens to experts 0 and 1 only
        let indices_data: Vec<u32> = vec![0, 1, 0, 1, 0, 1, 0, 1]; // T=4, K=2
        let weights_data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let logits_data: Vec<f32> = vec![0.0; 16]; // T=4, E=4

        let routing = RoutingOutput::<CandleBackend> {
            expert_indices: CandleTensor::from_u32_slice(
                &indices_data,
                &Shape::new(&[4, 2]),
                &cpu(),
            )
            .unwrap(),
            routing_weights: CandleTensor::from_slice(&weights_data, &Shape::new(&[4, 2]), &cpu())
                .unwrap(),
            router_logits: CandleTensor::from_slice(&logits_data, &Shape::new(&[4, 4]), &cpu())
                .unwrap(),
        };

        let token_features =
            CandleTensor::zeros(&Shape::new(&[4, 32]), DType::F32, &cpu()).unwrap();

        // expert 2 and 3 should get no tokens
        let (_, positions, _) =
            dispatch::<CandleBackend>(&token_features, ExpertIndex(2), &routing).unwrap();
        assert!(positions.is_empty());
    }

    // --- MoeLayer ---

    #[test]
    fn test_moe_layer_new() {
        let config = make_moe_config();
        let layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu());
        assert!(layer.is_ok());
    }

    #[test]
    fn test_moe_layer_expert_count() {
        let config = make_moe_config();
        let layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        assert_eq!(layer.experts.len(), 4);
    }

    #[test]
    fn test_moe_layer_shared_expert_present() {
        let config = make_moe_config();
        let layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        assert!(layer.shared_expert.is_some());
    }

    #[test]
    fn test_moe_layer_pregate_buffer_present() {
        let config = make_moe_config();
        let layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        assert!(layer.pregate_buffer.is_some());
    }

    #[test]
    fn test_moe_layer_forward_shape() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert_eq!(out.hidden_states.shape(), &Shape::new(&[1, 4, 32]));
    }

    #[test]
    fn test_moe_layer_forward_single_token() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 1, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert_eq!(out.hidden_states.shape(), &Shape::new(&[1, 1, 32]));
    }

    #[test]
    fn test_moe_layer_aux_loss_none() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert!(out.aux_loss.is_none());
    }

    #[test]
    fn test_moe_layer_pregate_buffer_updated_after_forward() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        layer.forward(&x, 0).unwrap();
        // after forward, token 0 should have a record
        let buf = layer.pregate_buffer.as_ref().unwrap();
        assert!(buf.read(TokenPos(0)).is_some());
    }

    #[test]
    fn test_moe_layer_reset_speculation() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        layer.forward(&x, 0).unwrap();
        layer.reset_speculation();
        let buf = layer.pregate_buffer.as_ref().unwrap();
        assert!(buf.read(TokenPos(0)).is_none());
    }

    #[test]
    fn test_moe_layer_with_latent() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, Some(16), &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert_eq!(out.hidden_states.shape(), &Shape::new(&[1, 4, 32]));
    }

    #[test]
    fn test_moe_layer_forward_with_offset() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 1, 32]), DType::F32, &cpu()).unwrap();
        // simulate generation step at position 10
        let out = layer.forward(&x, 10).unwrap();
        assert_eq!(out.hidden_states.shape(), &Shape::new(&[1, 1, 32]));
        let buf = layer.pregate_buffer.as_ref().unwrap();
        assert!(buf.read(TokenPos(10)).is_some());
    }

    // ── Phase 4 tests ──

    #[test]
    fn test_moe_layer_has_inference_aux_config_by_default() {
        let config = make_moe_config();
        let layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        assert!(!layer.aux_loss_config.enabled);
    }

    #[test]
    fn test_moe_layer_set_training_mode() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        layer.set_training_mode(crate::model::moe_loss::AuxLossConfig::training());
        assert!(layer.aux_loss_config.enabled);
    }

    #[test]
    fn test_moe_forward_inference_aux_loss_is_none() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert!(out.aux_loss.is_none());
    }

    #[test]
    fn test_moe_forward_training_aux_loss_is_some() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        layer.set_training_mode(crate::model::moe_loss::AuxLossConfig::training());
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert!(out.aux_loss.is_some());
    }

    #[test]
    fn test_moe_forward_training_aux_loss_is_scalar() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        layer.set_training_mode(crate::model::moe_loss::AuxLossConfig::training());
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert_eq!(out.aux_loss.unwrap().numel(), 1);
    }

    #[test]
    fn test_moe_forward_training_aux_loss_is_positive() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        layer.set_training_mode(crate::model::moe_loss::AuxLossConfig::training());
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        let val = out.aux_loss.unwrap().to_vec_f32().unwrap()[0];
        assert!(val > 0.0, "aux loss should be positive, got {}", val);
    }

    #[test]
    fn test_prefetch_experts_empty_without_buffer() {
        let mut config = make_moe_config();
        config.prefetch_threshold = None;
        let layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        assert!(layer.prefetch_experts().is_empty());
    }

    #[test]
    fn test_prefetch_experts_returns_high_confidence_experts() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        if let Some(ref mut buf) = layer.pregate_buffer {
            buf.write(
                TokenPos(0),
                SpeculativeRecord {
                    token_pos: TokenPos(0),
                    expert_indices: vec![ExpertIndex(1), ExpertIndex(3)],
                    max_score: 0.9,
                },
            );
        }
        let candidates = layer.prefetch_experts();
        assert!(!candidates.is_empty());
        assert!(candidates.contains(&ExpertIndex(1)));
        assert!(candidates.contains(&ExpertIndex(3)));
    }

    #[test]
    fn test_prefetch_experts_ignores_low_confidence() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        if let Some(ref mut buf) = layer.pregate_buffer {
            buf.write(
                TokenPos(0),
                SpeculativeRecord {
                    token_pos: TokenPos(0),
                    expert_indices: vec![ExpertIndex(0)],
                    max_score: 0.1,
                },
            );
        }
        let candidates = layer.prefetch_experts();
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_is_prefetched_true_for_candidate() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        if let Some(ref mut buf) = layer.pregate_buffer {
            buf.write(
                TokenPos(0),
                SpeculativeRecord {
                    token_pos: TokenPos(0),
                    expert_indices: vec![ExpertIndex(2)],
                    max_score: 0.95,
                },
            );
        }
        assert!(layer.is_prefetched(ExpertIndex(2)));
    }

    #[test]
    fn test_is_prefetched_false_for_non_candidate() {
        let config = make_moe_config();
        let layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        assert!(!layer.is_prefetched(ExpertIndex(0)));
    }

    #[test]
    fn test_speculative_correction_wired_in_forward() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 2, 32]), DType::F32, &cpu()).unwrap();
        layer.forward(&x, 0).unwrap();
        let buf = layer.pregate_buffer.as_ref().unwrap();
        assert!(buf.read(TokenPos(0)).is_some());
        assert!(buf.read(TokenPos(1)).is_some());
        // second forward — correction mask computed internally, no panic
        layer.forward(&x, 0).unwrap();
    }

    #[test]
    fn test_router_load_balance_loss_delegates_correctly() {
        let config = make_moe_config();
        let rc = RouterConfig::from_model_config(&config);
        let router = Router::<CandleBackend>::new(32, rc, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let routing = router.forward(&x).unwrap();
        let loss = Router::<CandleBackend>::load_balance_loss(
            &routing.router_logits,
            &routing.expert_indices,
            4,
        )
        .unwrap();
        assert_eq!(loss.numel(), 1);
        let val = loss.to_vec_f32().unwrap()[0];
        assert!(val > 0.0);
    }

    #[test]
    fn test_loss_output_field_populated_in_training_mode() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        layer.set_training_mode(crate::model::moe_loss::AuxLossConfig::training());
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        assert!(out.loss_output.is_some());
        let loss_out = out.loss_output.unwrap();
        assert!(loss_out.load_balance_loss.is_some());
        assert!(loss_out.z_loss.is_some());
        assert!(loss_out.total_aux_loss.is_some());
    }

    #[test]
    fn test_loss_output_none_in_inference_mode() {
        let config = make_moe_config();
        let mut layer = MoeLayer::<CandleBackend>::new(&config, None, &cpu()).unwrap();
        let x = CandleTensor::zeros(&Shape::new(&[1, 4, 32]), DType::F32, &cpu()).unwrap();
        let out = layer.forward(&x, 0).unwrap();
        // in inference mode loss_output should be None or all fields None
        if let Some(loss_out) = out.loss_output {
            assert!(loss_out.total_aux_loss.is_none());
        }
    }
}
