# Axiom

A large language model inference engine built in Rust, targeting Apple Silicon via
Metal. Axiom loads GGUF-quantized LLaMA-family models and runs inference through a
hand-written Metal Shading Language kernel path, with a generic (backend-agnostic)
CPU fallback used for anything not yet on the fast path. This document covers the
project through the dense LLaMA (Llama 3.2 1B) milestone — Metal-accelerated prefill
and decode at 9.0 tok/s. Mixture-of-Experts support is under active development on
top of this foundation and isn't covered here.

---

## Architecture Overview

```
axiom/
  src/
    core/
      tensor.rs        — TensorOps trait; CandleTensor, MetalTensor, CudarcTensor (stub) impls
      backend.rs        — Backend trait; TopKLastDimOp trait + per-backend impls
    metal/
      allocator.rs       — MetalAllocator: bump allocator + free list, arena-style pool
      runner.rs           — MetalRunner: command buffer/encoder lifecycle, per-op dispatch
      kernels.rs           — Rust wrappers binding buffers and dispatching each .metal kernel
      kernels/*.metal       — Metal Shading Language kernel source
      state.rs, context.rs, device.rs — global Metal state, device/context setup
    model/
      model.rs           — LlamaModel<B>: forward() dispatcher, forward_metal() fast path
      block.rs            — Block<B>, FeedForwardLayer (Dense today; Moe/LazyMoe in progress)
      attention.rs          — GQA attention, RoPE application
      rope.rs, norm.rs, linear.rs — supporting layers
      config.rs            — ModelConfig
      moe.rs                — MoE scaffolding (Router, Expert, dispatch/combine) — WIP, not on this milestone's path
    weights/
      loader.rs            — GGUF -> ModelConfig + LlamaModel, tensor name -> field mapping
      gguf.rs               — memory-mapped GGUF parser (metadata + tensor info)
      quantize.rs            — dequantize(): F32/F16/BF16 + Q4_0/Q4_1/Q8_0/Q4_K/Q6_K
      lazy.rs                — QuantizedWeight: partial/lazy dequantization (materialize_rows)
    inference/
      session.rs            — prompt/generated tokens, KV cache, offset, EOS tracking
      sampler.rs             — greedy, temperature, top-k, top-p, repetition penalty, no-repeat-ngram
      generator.rs            — wires model forward + sampler + session into one step
      engine.rs               — submit/step over a batch of sessions
    tokenizer/
      tokenizer.rs          — BPE encode/decode over a loaded vocab
      loader.rs              — loads HuggingFace-format tokenizer.json (GGUF-embedded tokenizer loading is a stub)
    bin/
      smoke.rs              — CLI: load a GGUF + tokenizer, run a prompt, print tok/s
```

---

## What's implemented

- **Dense LLaMA architecture** — RMSNorm, grouped-query attention, RoPE, SwiGLU
  feed-forward, tied through a generic `forward()` and a Metal-accelerated
  `forward_metal()`.
- **GGUF loading** — memory-mapped parser, full metadata + tensor info; dequantizes
  F32/F16/BF16 directly and Q4_0/Q4_1/Q8_0/Q4_K/Q6_K via block-wise dequantization.
- **Metal fast path** — hand-written kernels for every op in the forward pass (see
  below), dispatched through a single command buffer/encoder per forward call.
- **Pooled tensor allocation** — an arena allocator for the scratch tensors created
  and discarded within one `forward_metal` call (see *Tensor Allocation* below).
- **Sampling** — greedy, temperature, top-k, top-p, repetition penalty, no-repeat-ngram.
- **BPE tokenizer** — loads a standard HuggingFace `tokenizer.json`; exercised in
  practice against both LLaMA 3's and Qwen3's tokenizer formats.
- **Backend abstraction** — `TensorOps` is implemented for `CandleTensor` (CPU,
  via the `candle` crate) and `MetalTensor` (Apple Silicon). A `CudarcTensor`
  (CUDA) implementation exists as an unimplemented placeholder for future work.

## Metal Kernels

Each kernel has a `.metal` source file plus a Rust wrapper in `kernels.rs` that
binds buffers (correctly offset into the shared pool where relevant) and dispatches
it via `MetalRunner`. Primary path is F32; some kernels (`rms_norm`, `attention_qk`,
`attention_pv`) also have F16 variants.

| Kernel | Purpose |
|---|---|
| `rms_norm` | RMSNorm over the trailing dimension, any leading rank |
| `rope` | Rotary position embedding, in-place on Q/K |
| `matmul` / `broadcast_matmul` | Tiled matmul; `broadcast_matmul` adds per-batch offset handling |
| `attention_qk` | Q·K^T against the KV cache, causal-masked by `current_pos` |
| `attention_pv` | Attention-weighted V accumulation |
| `cache_write` | Scatter new K/V rows into the persistent KV cache at a given offset |
| `swiglu` | SwiGLU activation (`silu(gate) * up`) |
| `add` | Elementwise residual addition |

All of these are dispatched within one `MetalRunner`-owned encoder per
`forward_metal` call — the whole forward pass for one call (every layer) is
encoded before anything is committed to the GPU, then run and waited on once
via `runner.finish()`.

## Tensor Allocation

Every intermediate tensor in `forward_metal` — norm outputs, projections,
attention scores, FFN activations — used to get its own dedicated `MTLBuffer`
via `newBufferWithBytes_length_options`, one Metal API call per tensor per
layer per token. That's correct but has real per-call overhead, and was one of
two things standing between the naive implementation and a reasonable
throughput number.

The fix is a bump allocator (`MetalAllocator`) backed by one large, persistent
`MTLBuffer`, reset to empty at the start of every `forward_metal` call
(`alloc.reset()`). Ephemeral per-layer scratch tensors (`zeros_pooled` /
`uninit_pooled` — the latter skips the zero-fill for tensors that are about to
be fully overwritten by a kernel anyway) get bump-allocated from this pool and
implicitly freed in bulk on the next call's reset. Two things are deliberately
**not** pooled, since they need to survive past a single call: the KV cache
persist buffers, and the returned logits tensor — both still go through
dedicated buffers.

The pool only ever needs to hold one call's worth of scratch tensors — for
this model, on the order of a few MB for decode and tens of MB for a full
prefill (most per-layer allocations scale with `seq_len`, not just fixed
per-layer constants). An oversized pool (an early version used 8GB, sized for
a since-abandoned design where the pool held far more than transient scratch)
turned out to cause real problems under memory pressure on top of dequantized
model weights, not just wasted allocation — right-sizing it fixed a
correctness issue, not just a performance one.

## Weight Loading

GGUF files are memory-mapped (`GgufFile::from_file`) and exposed as a metadata
map plus a tensor-name-to-`GgufTensorInfo` map. `weights/loader.rs` reads model
hyperparameters from metadata keys, builds a `ModelConfig`, constructs the
model skeleton (`LlamaModel::new`), then iterates every tensor in the file,
dequantizes it (`quantize::dequantize`), and writes it into the matching field
via `LlamaTensor::parse` + `set_tensor`.

`weights/lazy.rs`'s `QuantizedWeight` supports **partial** dequantization —
`materialize_rows(start, end)` dequantizes a contiguous row range out of a
larger tensor without touching the rest, and without ever fully materializing
it. Not exercised on this milestone's path (dense LLaMA weights are small
enough to dequantize eagerly at load time), but load-bearing infrastructure
for anything where eager dequantization of the full tensor isn't viable.

## Inference Engine

`Session` tracks a single generation's prompt tokens, generated tokens, KV
cache, and position offset. `Generator::step` runs one forward pass (dispatched
by `LlamaModel::forward` to either the generic or Metal path depending on
backend) and samples the next token via `Sampler`. `Engine` wraps a batch of
sessions and exposes `submit`/`step` as the top-level API; `smoke.rs` is a thin
CLI driving one session through the batch loop and printing decoded text plus
tokens/sec.

## Performance

Starting point, per-tensor dedicated Metal buffers, decode-only fast path:
**6.1–6.2 tok/s**. Two changes got this to where it is now:

1. **Pooled allocation** (above) — cut per-tensor `MTLBuffer` creation
   overhead. Encode time (CPU-side command buffer construction) dropped to
   ~130µs per decode step, effectively negligible.
2. **Prefill on the Metal fast path** — `forward_metal` originally only
   handled single-token decode (`seq_len == 1`); prefill ran through a slow
   generic CPU-loop path (a ~4.5 billion-multiply-add naive LM-head matmul,
   for this model's ~17-token prompt). Extending `forward_metal` to handle
   `seq_len > 1` required fixing an attention-output-assembly bug (was only
   ever correct for exactly one query position) and a latent buffer-offset
   bug in the attention kernel wrappers (narrowed/non-zero-offset tensors were
   silently reading from the wrong location — harmless while nothing was ever
   narrowed, real the moment prefill's multi-position batch needed it).

Result: **9.0 tok/s**, decode and prefill both running through Metal kernels,
on Llama 3.2 1B Instruct (Q4_K_M) on Apple Silicon.

## Build & Run

```bash
# Metal backend (Apple Silicon)
cargo build --release --features metal

# Run the smoke test
cargo run --release --features metal --bin smoke -- \
    <path-to-gguf> <path-to-tokenizer.json> "<prompt>" <max_new_tokens> <temperature>
```

## Status

- Dense LLaMA (this document): Metal fast path for both prefill and decode, 9.0 tok/s.
- Mixture-of-Experts: scaffolding exists (`model/moe.rs` — router, dispatch/combine,
  lazy per-expert weight materialization for models too large to eagerly
  dequantize in full), currently being extended and debugged against a real
  MoE model on the generic (non-Metal) path. Not yet on the Metal fast path.
- CUDA backend (`CudarcTensor`): unimplemented placeholder, no current work planned.
- GGUF-embedded tokenizer loading (`tokenizer::loader::load_from_gguf`): unimplemented;
  a separate `tokenizer.json` is required alongside the GGUF file.
