use crate::core::backend::{Backend, CandleTensor};
use crate::core::device::Device;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::shape::Shape;
use crate::core::tensor::TensorOps;
use crate::model::config::ModelConfig;
use crate::model::linear::Linear;
use candle_core::shape::D::Minus1;
use candle_core::{IndexOp, Tensor};
use candle_nn;

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
    pub ep_size: usize,      // 1 for single GPU
    pub ep_rank: usize,      // 0 for single GPU
    pub experts_per_rank: usize,
}

// taken from mistral.rs
// https://github.com/EricLBuehler/mistral.rs/blob/6aec940499be1cf72c628f7ddaa8b3e59bcb4fda/mistralrs-core/src/ops.rs#L482-L504
// Define this to be generic over the Tensor type

pub struct TopKOutput<T> {
    pub values: T,
    pub indices: T,
}

pub trait TopKLastDimOp {
    // Return the generic TopKOutput
    fn topk(&self, k: usize) -> Result<TopKOutput<Self>>
    where
        Self: Sized;
}

impl TopKLastDimOp for CandleTensor {
    fn topk(&self, k: usize) -> Result<TopKOutput<Self>> {
        let sorted_indices = self.inner.arg_sort_last_dim(false)?;
        let topk_indices = sorted_indices
            .narrow(candle_core::D::Minus1, 0, k)?
            .contiguous()?;
        let values = self.inner.gather(&topk_indices, candle_core::D::Minus1)?;

        Ok(TopKOutput {
            values: CandleTensor {
                inner: values,
                shape: Shape::from(values.dims()),
                dtype: self.dtype,
                device: self.device.clone(),
            },
            indices: CandleTensor {
                inner: topk_indices,
                shape: Shape::from(topk_indices.dims()),
                dtype: DType::U32,
                device: self.device.clone(),
            },
        })
    }
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
            ScoreFunction::Sigmoid => logits.sigmoid(&x_flattened)?,
            ScoreFunction::Softmax => logits.softmax(Minus1 as usize)?, // not sure how to get -1
        };

        let topk_out = TopKLastDimOp::topk(&scores, self.config.experts_per_token)?;
        let mut routing_weights = topk_out.values;
        let expert_indices = topk_out.indices;

        //Renormalize (if using Softmax)
        let final_weights = if let ScoreFunction::Softmax = self.config.score_fn {
            let sum = routing_weights.sum_keepdim(Minus1)?;
            routing_weights.broadcast_div(&sum)?
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

        let actual_indices_flat: Vec<u32> = actual.expert_indices.to_vec1()?;
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
        router_logits: &B::Tensor,    // [T, E]
        expert_indices: &B::Tensor,   // [T, K]
        num_experts: usize,
    ) -> Result<B::Tensor>       {
        todo!()
    }     // scalar loss
}

/* Expert  (single routed expert)
One routed expert: a two-layer SwiGLU MLP operating on dimension `d`.
In standard MoE: d = hidden_size
In LatentMoE: d = latent_dim   (experts never see full hidden)
each expert is `gate_proj`, `up_proj`, `down_proj`.
gate and up are parallel; their hadamard product feeds down.
*/

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
        let x_proj_gate = self.gate_proj.forward(x)?;
        let silu_x = x_proj_gate.silu();

        let x_proj_up = self.up_proj.forward(x)?;
        let inner = x_proj_gate.matmul(&x_proj_up)?;

        let x_proj_down = self.down_proj.forward(&inner)?;

        Ok(x_proj_down)
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
        pub fn forward(&self, x: &B::Tensor) -> Result<B::Tensor> {
            let gate = self.gate_proj.forward(x)?.silu()?;
            let up = self.up_proj.forward(x)?;
            let fused = gate.mul(&up)?;
            self.down_proj.forward(&fused)
        }
}

/*Latent Projections

/// Implements the LatentMoE shared projections
///
///   forward flow:
///     x [T, hidden]
///       → down_proj  [T, latent]      ← shared across all routed experts
///       → dispatch to assigned experts
///       → each Expert operates on [n, latent]
///       → combine weighted sum        [T, latent]
///       → up_proj    [T, hidden]      ← shared across all routed experts
///
///   shared expert path bypasses both projections (runs on hidden_size).
*/
pub struct LatentProjection<B: Backend> {
    /// W↓ ∈ ℝ^{latent × hidden}  stored as [latent, hidden].
    pub down_proj: Linear<B>,
    /// W↑ ∈ ℝ^{hidden × latent}  stored as [hidden, latent].
    pub up_proj: Linear<B>,
    pub config: LatentConfig,
}

impl<B: Backend> LatentProjection<B> {
    pub fn new(config: LatentConfig, device: &Device) -> Result<Self> {
        // your body: build down [latent, hidden], up [hidden, latent]
        todo!()
    }

    /// Project hidden → latent.  [T, hidden] → [T, latent].
    pub fn project_down(&self, x: &B::Tensor) -> Result<B::Tensor> {
        // your body
        todo!()
    }

    /// Project latent → hidden.  [T, latent] → [T, hidden].
    pub fn project_up(&self, x: &B::Tensor) -> Result<B::Tensor> {
        // your body
        todo!()
    }
}

// dispatch/combine(CPU-side, single device)

/*
 Gather the tokens assigned to one expert from the full token stream.
`token_features`  shape: [T, d]
`expert_idx`      which expert we are gathering for
`routing_output`  contains expert_indices [T, K]
*/

pub fn dispatch(
    token_features: &impl TensorOps,
    expert_idx: ExpertIndex,
    routing_output: &RoutingOutput<impl Backend>,
) -> Result<(impl TensorOps, Vec<TokenPos>)> {
    // your body:
    //   walk routing_output.expert_indices [T, K],
    //   collect rows where any k matches expert_idx,
    //   narrow / index_select those rows from token_features
    todo!()
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
pub fn combine(
    expert_out: &impl TensorOps,
    routing_weights: &impl TensorOps,
    positions: &[TokenPos],
    expert_k_slot: &[usize],
    output_accumulator: &mut Vec<Option<impl TensorOps>>,
) -> Result<()> {
    // your body:
    //   for each (i, pos) in positions:
    //     weight = routing_weights[pos.0, expert_k_slot[i]]
    //     accumulator[pos.0] += weight * expert_out[i]
    todo!()
}

// Moe Top layer

#[derive(Debug)]
pub struct MoeOutput<B: Backend> {
    //The transformed hidden states. Shape matches input x.
    pub hidden_states: B::Tensor,
    //Placeholder for auxiliary loss — filled in when you wire the
    //load-balancing file. None until then.
    pub aux_loss: Option<B::Tensor>,
}

/// Full Megatron-style MoE layer (§2.1, §7.2, §7.3).
///
/// Forward pass (LatentMoE path, Megatron figure 1 + §7.3):
///
///  x [B,S,H]
///   ├─ shared_expert(x)                       → shared_out [T, H]
///   ├─ router(x)                              → RoutingOutput
///   ├─ latent.project_down(x)                 → x_latent [T, ℓ]
///   ├─ for each expert e in 0..E:
///   │     (gathered, positions) = dispatch(x_latent, e, routing)
///   │     expert_out = experts[e].forward(gathered)
///   │     combine(expert_out, routing_weights, positions, …, accumulator)
///   ├─ routed_out = stack accumulator         → [T, ℓ]
///   ├─ routed_out = latent.project_up(routed_out) → [T, H]
///   └─ output = routed_out + shared_out       → [T, H]  reshape → [B,S,H]
///
/// Speculative pre-gating sits *before* router(x):
///   1. read PreGateBuffer for current token positions
///   2. run router(x) to get actual routing
///   3. run speculative_correction → correction_mask
///   4. zero-out mis-predicted slots; schedule re-computation
///      (in this single-device implementation we simply recompute all
///       mis-predicted tokens — the structure is there for future EP)
pub struct MoeLayer<B: Backend> {
    pub router: Router<B>,
    pub experts: Vec<Expert<B>>,
    pub shared_expert: Option<SharedExpert<B>>,
    pub latent: Option<LatentProjection<B>>,
    /// Lives on the layer; reset between unrelated sequences.
    pub pregate_buffer: Option<PreGateBuffer>,
    pub config: RouterConfig,
    pub hidden_size: usize,
    /// The dimension experts actually compute over.
    /// = latent_dim if LatentMoE, else hidden_size.
    pub expert_dim: usize,
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
            Some(SharedExpert::new(hidden_size, intermediate_size, num_shared, device)?)
        } else {
            None
        };

        let latent = if let Some(ldim) = latent_dim {
            Some(LatentProjection::new(LatentConfig::new(hidden_size, ldim), device)?)
        } else {
            None
        };

        let pregate_buffer = config.prefetch_threshold.map(|_| {
            PreGateBuffer::new(config.max_position_embeddings)
        });

        Ok(Self {
            router,
            experts,
            shared_expert,
            latent,
            pregate_buffer,
            config: router_config,
            hidden_size,
            expert_dim,
        })
    }

    /// Full forward pass. See struct-level doc for the flow.
    ///
    /// `token_offset` — starting position in the sequence; used to index
    /// into pregate_buffer correctly for KV-cache generation steps.
    pub fn forward(&mut self, x: &B::Tensor, token_offset: usize) -> Result<MoeOutput<B>> {
        // your body — implement the 9-step flow described above
        todo!()
    }

    /// Write routing decisions into PreGateBuffer after a forward pass.
    /// Call this at the *end* of forward so the next step can read it.
    ///
    /// Converts routing_output.routing_weights to host scalars to build
    /// SpeculativeRecord per token.
    fn update_pregate_buffer(
        &mut self,
        routing_output: &RoutingOutput<B>,
        token_offset: usize,
        num_tokens: usize,
    ) -> Result<()> {
        // your body
        todo!()
    }

    /// Reset the pre-gate buffer. Call between unrelated sequences
    /// (e.g. a new prompt when batch_size > 1 sequences share a layer).
    pub fn reset_speculation(&mut self) {
        if let Some(buf) = &mut self.pregate_buffer {
            buf.clear();
        }
    }
}
