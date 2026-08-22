use crate::core::backend::MetalTensor;
use crate::core::dtype::DType;
use crate::core::error::{CoreError, Result};
use crate::core::tensor::TensorOps;
use crate::metal::allocator::BlockHandle;
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

    pub fn flush(&mut self) -> Result<()> {
        self.encoder.endEncoding();
        self.cmd_buf.commit();
        self.cmd_buf.waitUntilCompleted();

        let cmd_buf = self.state.ctx.command_buffer()?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Internal("failed to create compute encoder".into()))?;
        self.cmd_buf = cmd_buf;
        self.encoder = encoder;
        Ok(())
    }

    pub fn read_f32(&mut self, tensor: &MetalTensor) -> Result<Vec<f32>> {
        self.flush()?;
        tensor.to_vec_f32()
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

    pub fn dequantize_q4k(
        &self,
        data: &MetalTensor,
        out: &MetalTensor,
        num_blocks: u32,
        numel: u32,
    ) -> Result<()> {
        self.state.kernels.dequantize_q4_k_f32(
            &self.encoder,
            self.allocator,
            data.block(),
            out.block(),
            num_blocks,
            numel,
        )?;
        Ok(())
    }

    pub fn dequantize_q4k_raw(
        &self,
        data: &BlockHandle,
        out: &BlockHandle,
        num_blocks: u32,
        numel: u32,
    ) -> Result<()> {
        self.state.kernels.dequantize_q4_k_f32(
            &self.encoder,
            self.allocator,
            data,
            out,
            num_blocks,
            numel,
        )?;
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
        n_kv_heads: u32,
        head_dim: u32,
        seq_len: u32,
        current_pos: u32,
    ) -> Result<()> {
        // q.block() alone ignores q's own tensor-level offset_bytes -- fine as
        // long as every caller only ever passed a fresh, non-narrowed tensor
        // (offset always 0), which was true when this only ran for
        // single-token decode. forward_metal now passes narrow()'d views (one
        // query position out of a multi-position batch) for prefill, so the
        // offset has to be folded in here, same as broadcast_matmul already
        // does for its operands.
        let mut block_q = (*q.block()).clone();
        block_q.offset_bytes += q.metal_offset();
        let mut block_k = (*k_cache.block()).clone();
        block_k.offset_bytes += k_cache.metal_offset();
        let mut block_scores = (*scores.block()).clone();
        block_scores.offset_bytes += scores.metal_offset();

        match q.metal_dtype() {
            DType::F32 => self.state.kernels.attention_qk_f32(
                &self.encoder,
                self.allocator,
                &block_q,
                &block_k,
                &block_scores,
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                current_pos,
            )?,
            DType::F16 => self.state.kernels.attention_qk_f16(
                &self.encoder,
                self.allocator,
                &block_q,
                &block_k,
                &block_scores,
                n_heads,
                head_dim,
                seq_len,
                current_pos,
            )?,
            _ => return Err(MetalError::Internal("attention_qk: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn attention_pv(
        &self,
        scores: &MetalTensor,
        v_cache: &MetalTensor,
        out: &MetalTensor,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        head_dim: u32,
        current_pos: u32,
    ) -> Result<()> {
        // See attention_qk: `out` is now often a narrow()'d row (p) of a shared
        // [seq_len, n_heads, head_dim] buffer during prefill, so its
        // tensor-level offset has to be folded into the block before binding,
        // or every position would silently write to row 0.
        let mut block_scores = (*scores.block()).clone();
        block_scores.offset_bytes += scores.metal_offset();
        let mut block_v = (*v_cache.block()).clone();
        block_v.offset_bytes += v_cache.metal_offset();
        let mut block_out = (*out.block()).clone();
        block_out.offset_bytes += out.metal_offset();

        match scores.metal_dtype() {
            DType::F32 => self.state.kernels.attention_pv_f32(
                &self.encoder,
                self.allocator,
                &block_scores,
                &block_v,
                &block_out,
                n_heads,
                n_kv_heads,
                seq_len,
                head_dim,
                current_pos,
            )?,
            DType::F16 => self.state.kernels.attention_pv_f16(
                &self.encoder,
                self.allocator,
                &block_scores,
                &block_v,
                &block_out,
                n_heads,
                seq_len,
                head_dim,
                current_pos,
            )?,
            _ => return Err(MetalError::Internal("attention_pv: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn swiglu(
        &self,
        gate: &MetalTensor,
        up: &MetalTensor,
        output: &MetalTensor,
        num_elements: u32,
    ) -> Result<()> {
        match gate.metal_dtype() {
            DType::F32 => self.state.kernels.swiglu_f32(
                &self.encoder,
                self.allocator,
                gate.block(),
                up.block(),
                output.block(),
                num_elements,
            )?,
            DType::F16 => self.state.kernels.swiglu_f16(
                &self.encoder,
                self.allocator,
                gate.block(),
                up.block(),
                output.block(),
                num_elements,
            )?,
            _ => return Err(MetalError::Internal("swiglu: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn add(&self, a: &MetalTensor, b: &MetalTensor, output: &MetalTensor) -> Result<()> {
        let num_elements = output.metal_shape().numel() as u32;
        match a.metal_dtype() {
            DType::F32 => self.state.kernels.add_f32(
                &self.encoder,
                self.allocator,
                a.block(),
                b.block(),
                output.block(),
                num_elements,
            )?,
            DType::F16 => self.state.kernels.add_f16(
                &self.encoder,
                self.allocator,
                a.block(),
                b.block(),
                output.block(),
                num_elements,
            )?,
            _ => return Err(MetalError::Internal("add: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn cache_write(
        &self,
        src: &MetalTensor,
        cache: &MetalTensor,
        write_pos: u32,
        n_kv_heads: u32,
        head_dim: u32,
        write_len: u32,
    ) -> Result<()> {
        match src.metal_dtype() {
            DType::F32 => self.state.kernels.cache_write_f32(
                &self.encoder,
                self.allocator,
                src.block(),
                cache.block(),
                write_pos,
                n_kv_heads,
                head_dim,
                write_len,
            )?,
            DType::F16 => self.state.kernels.cache_write_f16(
                &self.encoder,
                self.allocator,
                src.block(),
                cache.block(),
                write_pos,
                n_kv_heads,
                head_dim,
                write_len,
            )?,
            _ => return Err(MetalError::Internal("cache_write: unsupported dtype".into()).into()),
        }
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        self.encoder.endEncoding();
        self.cmd_buf.commit();
        self.cmd_buf.waitUntilCompleted();
        Ok(())
    }
}
