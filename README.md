# Axiom

A high-performance large language model inference engine built in Rust with custom CUDA kernels. Axiom implements the LLaMA and DeepSeek model families with support for grouped-query attention, Mixture-of-Experts with speculative pre-gating, and LoRA adapter hot-swapping. The engine is built around three core ideas: a ForkKV disaggregated KV cache that separates shared base prefixes from per-agent residual caches, a persistent producer/consumer attention pipeline with asynchronous TMA-based tile loading targeting Ada, Hopper, and Blackwell GPUs, and a speculative decoding stack that integrates cleanly with the paged memory system. The runtime exposes an OpenAI-compatible HTTP API and is designed for multi-agent serving workloads where sequences fork from a shared context.

---

## Architecture Overview

```
axiom/
  build.rs                  — root build script, compiles .cu → PTX via nvcc
  kernel/
    src/                    — CUDA kernel source files (.cu)
  src/
    core/                   — tensor abstraction layer
    model/                  — LLaMA / DeepSeek model implementation
    inference/              — generation engine, session, sampler, batch
    kv_cache/               — ForkKV dual radix tree, KV manager
    cuda/                   — cudarc bindings, paged allocator, kernel wrappers
    tokenizer/              — from-scratch BPE tokenizer
    weights/                — GGUF loader, quantization stubs
    lora/                   — LoRA adapter config, loader, layer
    server/                 — Axum HTTP server, OpenAI-compatible routes
```

---

## Features

- **Full LLaMA / DeepSeek model family** — dense and Mixture-of-Experts layers, GQA, RoPE
- **ForkKV KV cache** — disaggregated base + residual cache with O(prefix_len) lookup
- **Custom CUDA kernels** — sm_80 / sm_89 / sm_90 / sm_100 targets
- **Persistent producer/consumer attention** — async TMA K/V loading, ping-pong double buffer
- **Speculative decoding** — draft token generation integrated with paged memory
- **MoE speculative pre-gating** — per-token expert prefetch via confidence-scored ring buffer
- **LoRA hot-swapping** — adapter registry, per-layer weight injection
- **GGUF weight loading** — F32, F16, BF16, quantization stubs for Q4/Q8
- **From-scratch BPE tokenizer** — validated byte-for-byte against LLaMA 3
- **OpenAI-compatible HTTP API** — `/v1/completions`, `/v1/chat/completions`
- **Paged block allocator** — reference-counted copy-on-write GPU memory pool

---

## CUDA Kernels

All kernels are compiled to PTX at build time via `build.rs` and loaded at runtime via cudarc.

### Compute Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `rms_norm_f16_kernel` | `rms_norm_f16.cu` | RMSNorm, f16 I/O, f32 accumulation |
| `fused_residual_rmsnorm_f16_kernel` | `fused_residual_rmsnorm_f16.cu` | Fused residual add + RMSNorm |
| `rotary_embedding_f16_kernel` | `rotary_embedding_f16.cu` | RoPE, in-place Q/K rotation, GQA-aware |
| `embedding_gather_f16_kernel` | `embedding_gather_f16.cu` | GPU-side token embedding lookup |
| `argmax_f16_kernel` | `argmax_f16.cu` | GPU-side argmax for zero-copy greedy decode |

### KV Cache Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `reshape_and_cache_f16io_kernel` | `reshape_and_cache_f16.cu` | Scatter per-token K/V into paged cache blocks |
| `copy_blocks_f16_kernel` | `copy_blocks_f16.cu` | Block copy for ForkKV copy-on-write |

### Attention Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `flash_attention_3_decode_f16io_kernel` | `flash_attention_3.cu` | FA3 decode, warp-parallel QK^T, online softmax |
| `flash_attention_3_decode_gqa_f16io_kernel` | `flash_attention_3.cu` | FA3 GQA variant, Q heads in registers, V reuse |
| `residual_attention_decode_f16io_kernel` | `residual_attention.cu` | ForkKV Algorithm 1, dual bCache/rCache, single online softmax pass |
| `flash_attention_4_decode_f16io_kernel` | `flash_attention_4.cu` | Persistent producer/consumer, TMA async loads, ping-pong double buffer |

### FlashAttention-3 Design

- Warp-parallel QK^T: 8 KV positions per round via warp shuffle reduce
- Bank-conflict-free shared memory: KV rows padded to `(head_dim + 2)`
- GQA optimization: all query heads in group pre-loaded into registers, V reused across heads
- Online softmax with correction: numerically stable, no two-pass required
- Launch: `grid(num_seqs, num_kv_heads)`, `block(256)`, dynamic smem

### ResidualAttention Design (ForkKV Algorithm 1)

Implements arXiv:2604.06370. Attends over two separate paged KV caches in a single kernel:

- **Phase 1**: iterate bCache tiles (shared base prefix, read-only)
- **Phase 2**: iterate rCache tiles (per-agent residual)
- **Critical invariant**: `row_max` and `row_sum` are never reset between phases — the online softmax state continues across both caches, producing a single correctly normalized output over the full `(base + residual)` context
- V reuse across GQA heads: one smem V read serves all heads in the group

### FlashAttention-4 Design

- **512 threads / block** = 4 warpgroups: WG 0-1 producers, WG 2-3 consumers
- **Ping-pong double buffer**: producer fills stage `(tile+1) % 2` while consumer processes stage `tile % 2`
- **sm_90+**: `cuda::pipeline` + `cuda::memcpy_async` for TMA-based K/V loads
- **sm_89**: `cp.async.cg` inline asm for Ada Lovelace
- **Persistent scheduling**: blocks atomicAdd into `d_tile_counter`, loop over `(seq, kv_head)` work items until exhausted
- Launch: `grid(sm_count * 2)`, `block(512)`, dynamic smem = `sizeof(FA4Smem)`

---

## Model Architecture

### LLaMA / Dense

```
Embedding
  └── N × Block
        ├── RMSNorm
        ├── Attention (GQA, RoPE, paged KV)
        ├── RMSNorm
        └── FeedForward (SwiGLU: gate × up → down)
  └── RMSNorm
  └── LM Head (linear projection to vocab)
```

### DeepSeek MoE

Replaces dense FeedForward with `MoeLayer` at configured intervals:

```
MoeLayer
  ├── Router (gate linear → score_fn → top-k → renormalize)
  ├── N × Expert (SwiGLU MLP, routed)
  ├── SharedExpert (SwiGLU MLP, always active)
  └── PreGateBuffer (speculative expert prefetch ring buffer)
```

**Speculative pre-gating**: after each forward pass, routing decisions and confidence scores are written to `PreGateBuffer`. Before the next step, high-confidence tokens have their expert sets prefetched. A speculative correction mask identifies mispredictions and recomputes only those tokens.

**LatentMoE**: optional shared down/up projections compress hidden → latent before dispatch, reducing all-to-all volume by `hidden/latent` ratio.

---

## KV Cache System

### Dual Radix Tree

Two radix trees keyed by token sequence:

- **BaseTree**: shared base prefixes, O(prefix_len) lookup and longest-prefix match
- **ResidualTree**: per-agent residual caches, keyed by `(token_sequence, agent_id)`

### Paged Block Allocator

- Fixed GPU memory pool divided into equal-sized blocks
- Free list: `VecDeque<BlockId>` for O(1) alloc/free
- Reference counting per block: `inc_ref` / `dec_ref`
- Copy-on-write: `is_shared()` check before any write; shared blocks duplicated via `copy_blocks_f16_kernel`

### ForkKV Fork Operation

When a session forks from a parent:

1. Parent's block table is inherited as the child's bCache block table
2. All shared blocks get `inc_ref`
3. Child gets a fresh empty rCache block table
4. Subsequent child tokens are written to rCache only
5. Attention uses `residual_attention_decode_f16io_kernel` to attend over both caches

---

## Tokenizer

From-scratch BPE implementation:

- **Pretokenization**: GPT-2 regex splits, special token detection with longest-match, byte-to-unicode mapping
- **BPE merge loop**: O(n²) merge with rank-based priority, `MergeMode::Rank` for HuggingFace tokenizers
- **Vocab**: `Vocab` struct owns all mappings, special token registry, BOS/EOS/PAD/UNK sentinels
- **Validated**: byte-for-byte match against LLaMA 3 tokenizer on all test strings

---

## Weight Loading

- **GGUF v2/v3 parser**: memory-mapped, full metadata and tensor info parsing
- **Supported dtypes**: F32, F16, BF16 (loaded directly), Q4_0, Q4_1, Q8_0, Q4_K, Q6_K (stubs, dequant planned for Arc 4)
- **LlamaTensor key mapping**: parses GGUF tensor names to typed `LlamaTensor` enum variants
- **LoRA loading**: paired `lora_a` / `lora_b` tensors keyed by module name

---

## LoRA

- **`LoraLinear<B>`**: drop-in replacement for `Linear<B>`, adds `(x @ A.T) @ B.T * scaling` residual path
- **`AdapterRegistry<B>`**: HashMap-backed registry, O(1) get/register/remove
- **`LoadedAdapter<B>`**: loaded weight pairs keyed by `"blk.{i}.{module}"` strings
- Adapters loaded from GGUF, hot-swapped without model restart

---

## Inference Engine

- **`Session<B>`**: prompt tokens, generated tokens, KV cache, offset tracking, EOS/max-token stop conditions
- **`Sampler`**: greedy, temperature, top-k, top-p (nucleus), repetition penalty
- **`Batch<B>`**: fixed-capacity session pool, active/finished/failed state tracking
- **`Generator<B>`**: wires model forward pass, sampler, and session into the generation loop
- **`Engine<B>`**: top-level API — submit, step, run_to_completion, drain_finished
- **`ForkManager`**: manages paged sessions, fork operations, copy-on-write (Arc 3)

---

## HTTP Server

OpenAI-compatible API via Axum:

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/v1/completions` | POST | Text completion |
| `/v1/chat/completions` | POST | Chat completion |

Middleware: request ID injection, request logging, CORS.

---

## Build

### Requirements

- Rust 1.75+
- CUDA Toolkit (tested on 13.0 with `CUDA_VERSION=12050` override)
- nvcc on PATH

### Environment

```bash
export CUDA_VERSION=12050
export CUDA_ROOT=/usr/local/cuda
export CUDA_ARCH=sm_80        # sm_80=A100, sm_89=4090, sm_90=H100, sm_100=5090
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build

```bash
# CPU only (Candle backend)
cargo build

# With CUDA kernels
cargo build --features cuda
```

### Test

```bash
# All tests
cargo test --features cuda

# Kernel tests only (requires GPU)
cargo test --features cuda cuda::
```

---

## Development Arcs

| Arc | Status | Description |
|-----|--------|-------------|
| Arc 1 | Complete | Core tensor layer, model, tokenizer, sampler, server, GGUF loader |
| Arc 2 | Complete | CUDA kernels, PTX compilation, cudarc bindings, paged allocator |
| Arc 3 | In progress | ForkKV engine integration, paged sessions, fork operation |
| Arc 3.5 | Planned | Speculative decoding integrated with ForkKV |
| Arc 4 | Planned | MoE pre-gating prefetch, novel routing, quantization dequant kernels |

---

## References

- [ForkKV: Disaggregated KV Cache Management](https://arxiv.org/abs/2604.06370)
- [FlashAttention-3: Fast and Accurate Attention](https://arxiv.org/abs/2407.08608)
- [NVIDIA Hopper Architecture TMA White Paper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
