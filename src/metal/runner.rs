use crate::core::backend::MetalTensor;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::metal::allocator::MetalAllocator;
use crate::metal::error::MetalError;
use crate::metal::state::MetalState;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandEncoder;
use objc2_metal::{MTLCommandBuffer, MTLComputeCommandEncoder};

pub struct MetalRunner<'a> {
    state: &'a MetalState,
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    allocator: &'a MetalAllocator,
}

impl<'a> MetalRunner<'a> {
    pub fn new(state: &'a MetalState, allocator: &'a MetalAllocator) -> Result<Self> {
        let cmd_buf = state.ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;
        Ok(Self {
            state,
            cmd_buf,
            encoder,
            allocator,
        })
    }

    pub fn rms_norm(
        &self,
        input: &MetalTensor,
        weight: &MetalTensor,
        output: &MetalTensor,
        eps: f32,
    ) -> Result<()> {
        let shape = input.metal_shape();
        let hidden = *shape
            .dims()
            .last()
            .ok_or_else(|| MetalError::Internal("empty shape".into()))?;
        let num_tokens = shape.numel() / hidden;

        match input.metal_dtype() {
            DType::F32 => {
                self.state.kernels.rms_norm_f32(
                    &self.encoder,
                    self.allocator,
                    input.block(),
                    weight.block(),
                    output.block(),
                    num_tokens as u32,
                    hidden as u32,
                    eps,
                )?;
            }
            DType::F16 => {
                self.state.kernels.rms_norm_f16(
                    &self.encoder,
                    self.allocator,
                    input.block(),
                    weight.block(),
                    output.block(),
                    num_tokens as u32,
                    hidden as u32,
                    eps,
                )?;
            }
            _ => return Err(MetalError::Internal("rms_norm: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn softmax(&self, input: &MetalTensor, output: &MetalTensor) -> Result<()> {
        let shape = input.metal_shape();
        let row_size = *shape
            .dims()
            .last()
            .ok_or_else(|| MetalError::Internal("empty shape".into()))?;
        let num_rows = shape.numel() / row_size;

        match input.metal_dtype() {
            DType::F32 => {
                self.state.kernels.softmax_f32(
                    &self.encoder,
                    self.allocator,
                    input.block(),
                    output.block(),
                    num_rows as u32,
                    row_size as u32,
                )?;
            }
            DType::F16 => {
                self.state.kernels.softmax_f16(
                    &self.encoder,
                    self.allocator,
                    input.block(),
                    output.block(),
                    num_rows as u32,
                    row_size as u32,
                )?;
            }
            _ => return Err(MetalError::Internal("softmax: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn rope(
        &self,
        x: &MetalTensor,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        theta: f32,
        offset: u32,
    ) -> Result<()> {
        match x.metal_dtype() {
            DType::F32 => {
                self.state.kernels.rope_f32(
                    &self.encoder,
                    self.allocator,
                    x.block(),
                    seq_len,
                    n_heads,
                    head_dim,
                    theta,
                    offset,
                )?;
            }
            DType::F16 => {
                self.state.kernels.rope_f16(
                    &self.encoder,
                    self.allocator,
                    x.block(),
                    seq_len,
                    n_heads,
                    head_dim,
                    theta,
                    offset,
                )?;
            }
            _ => return Err(MetalError::Internal("rope: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn matmul(&self, a: &MetalTensor, b: &MetalTensor, c: &MetalTensor) -> Result<()> {
        let a_shape = a.metal_shape();
        let b_shape = b.metal_shape();
        if a_shape.rank() != 2 || b_shape.rank() != 2 {
            return Err(MetalError::Internal("matmul only supports 2D tensors".into()).into());
        }
        let m = a_shape.dims()[0];
        let k = a_shape.dims()[1];
        let n = b_shape.dims()[1];
        if b_shape.dims()[0] != k {
            return Err(MetalError::Internal("inner dimension mismatch".into()).into());
        }

        match a.metal_dtype() {
            DType::F32 => {
                self.state.kernels.matmul_f32(
                    &self.encoder,
                    self.allocator,
                    a.block(),
                    b.block(),
                    c.block(),
                    m as u32,
                    n as u32,
                    k as u32,
                )?;
            }
            DType::F16 => {
                self.state.kernels.matmul_f16(
                    &self.encoder,
                    self.allocator,
                    a.block(),
                    b.block(),
                    c.block(),
                    m as u32,
                    n as u32,
                    k as u32,
                )?;
            }
            _ => return Err(MetalError::Internal("matmul: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn broadcast_matmul(
        &self,
        a: &MetalTensor,
        b: &MetalTensor,
        c: &MetalTensor,
    ) -> Result<()> {
        let rank_a = a.metal_shape().rank();
        let rank_b = b.metal_shape().rank();
        let k = a.metal_shape().dims()[rank_a - 1];

        let n = if rank_b == 2 {
            if b.metal_shape().dims()[0] != k {
                return Err(MetalError::Internal(format!(
                    "inner dim mismatch: {} vs {}",
                    k,
                    b.metal_shape().dims()[0]
                ))
                .into());
            }
            b.metal_shape().dims()[1]
        } else {
            b.metal_shape().dims()[rank_b - 1]
        };

        let m_per = a.metal_shape().dims()[rank_a - 2];
        let batch_a: usize = a.metal_shape().dims()[..rank_a - 2].iter().product();
        let batch_b: usize = if rank_b > 2 {
            b.metal_shape().dims()[..rank_b - 2].iter().product()
        } else {
            1
        };
        let batch_out = batch_a.max(batch_b);

        let dtype_size = a.metal_dtype().size_in_bytes();
        let stride_a = m_per * k * dtype_size;
        let stride_b = k * n * dtype_size;
        let stride_c = m_per * n * dtype_size;

        for batch_idx in 0..batch_out {
            let self_b = if batch_a == 1 { 0 } else { batch_idx };
            let other_b = if batch_b == 1 { 0 } else { batch_idx };

            let mut block_a = (*a.block()).clone();
            block_a.offset_bytes += a.metal_offset() + self_b * stride_a;

            let mut block_b = (*b.block()).clone();
            block_b.offset_bytes += b.metal_offset() + other_b * stride_b;

            let mut block_c = (*c.block()).clone();
            block_c.offset_bytes += c.metal_offset() + batch_idx * stride_c;

            match a.metal_dtype() {
                DType::F32 => {
                    self.state.kernels.matmul_f32(
                        &self.encoder,
                        self.allocator,
                        &block_a,
                        &block_b,
                        &block_c,
                        m_per as u32,
                        n as u32,
                        k as u32,
                    )?;
                }
                DType::F16 => {
                    self.state.kernels.matmul_f16(
                        &self.encoder,
                        self.allocator,
                        &block_a,
                        &block_b,
                        &block_c,
                        m_per as u32,
                        n as u32,
                        k as u32,
                    )?;
                }
                _ => {
                    return Err(
                        MetalError::Internal("broadcast_matmul: unsupported dtype".into()).into(),
                    )
                }
            }
        }
        Ok(())
    }

    pub fn attention_qk(
        &self,
        q: &MetalTensor,
        k_cache: &MetalTensor,
        scores: &MetalTensor,
        n_heads: u32,
        head_dim: u32,
        seq_len: u32,
        current_pos: u32,
    ) -> Result<()> {
        self.state.kernels.attention_qk_f16(
            &self.encoder,
            self.allocator,
            q.block(),
            k_cache.block(),
            scores.block(),
            n_heads,
            head_dim,
            seq_len,
            current_pos,
        )?;
        Ok(())
    }

    pub fn attention_pv(
        &self,
        scores: &MetalTensor,
        v_cache: &MetalTensor,
        out: &MetalTensor,
        n_heads: u32,
        seq_len: u32,
        head_dim: u32,
        current_pos: u32,
    ) -> Result<()> {
        self.state.kernels.attention_pv_f16(
            &self.encoder,
            self.allocator,
            scores.block(),
            v_cache.block(),
            out.block(),
            n_heads,
            seq_len,
            head_dim,
            current_pos,
        )?;
        Ok(())
    }

    pub fn swiglu(
        &self,
        gate: &MetalTensor,
        up: &MetalTensor,
        output: &MetalTensor,
        num_elements: u32,
    ) -> Result<()> {
        self.state.kernels.swiglu_f16(
            &self.encoder,
            self.allocator,
            gate.block(),
            up.block(),
            output.block(),
            num_elements,
        )?;
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        self.encoder.endEncoding();
        self.cmd_buf.commit();
        self.cmd_buf.waitUntilCompleted();
        Ok(())
    }
}
