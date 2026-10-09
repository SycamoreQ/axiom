# Axiom × Qwen3-30B-A3B — Performance Analysis & Fixes

## Executive Summary

The slowness comes from **two major bugs** and **several architectural bottlenecks**. The bugs are fixable right now; the architectural items need incremental work.

---

## 🐛 Bug #1 (Critical): Router + Dispatch + Combine run on CPU every layer

The `LazyMoeLayer::forward()` calls `self.router.forward(x)` — which calls [`Linear::forward()`](file:///Users/kaushikmuthukumar/Documents/projects/axiom/src/model/linear.rs#L17-L24), which calls `x.broadcast_matmul()` on the **Candle backend's CPU path**, even when `x` is a Metal tensor. Then `dispatch()`, `combine()` all operate via `to_vec_u32()`, `to_vec_f32()`, `narrow()`, `broadcast_mul()`, `add()` — all going through Candle's CPU-side generic ops.

**Impact**: For every one of the 48 layers, you're:
1. **Flushing the GPU** (the `runner.finish()` or implicit sync happens when `to_vec_f32()` reads back data to do routing on CPU)
2. Running the gating linear projection **on CPU** (`[1, 2048] × [2048, 128]` — small but serialized)
3. Running softmax + top-k **on CPU**
4. Running dispatch/combine with per-token `narrow()` + `broadcast_mul()` + `add()` — **O(k × T) individual tensor ops, each a separate Candle kernel**

This is the **single biggest bottleneck**. Even though the actual expert matmuls run on Metal via `expert_forward_via_runner`, the surrounding routing and combine work **serializes every layer** and forces GPU↔CPU round-trips.

## 🐛 Bug #2 (Critical): Iterating ALL 128 experts per layer

In [`LazyMoeLayer::forward()`](file:///Users/kaushikmuthukumar/Documents/projects/axiom/src/model/moe.rs#L1127), the loop is:

```rust
for e in 0..self.expert_bank.num_experts {  // 128 experts!
    let (gathered, positions, k_slots) = dispatch(..., ExpertIndex(e), ...)?;
    if positions.is_empty() { continue; }
    // dequantize + forward expert...
}
```

Qwen3-30B-A3B has **128 routed experts** per layer, with **top-8** activation. So each layer iterates 128 experts, calling `dispatch()` for all 128 (each doing `to_vec_u32()` + scan), and 120 of those iterations do nothing. But:

- Each `dispatch()` call does `routing_output.expert_indices.to_vec_u32()` — that's a **GPU→CPU readback of the same tensor 128 times** per layer
- Even the `continue` path has non-trivial overhead at 128 × 48 = **6,144 iterations per token**

**Fix**: Pre-compute which experts are active from the routing output once, then only iterate those.

---

## 🐛 Bug #3 (Moderate): `dequantize_expert_via_runner` transposes + `.contiguous()` on CPU

In [`LazyExpertBank::dequantize_expert_via_runner()`](file:///Users/kaushikmuthukumar/Documents/projects/axiom/src/model/moe.rs#L488-L535), after the GPU dequantization kernel runs, the code does:

```rust
.transpose(0, 1)?
.contiguous()?
```

The `.contiguous()` call on a transposed Metal tensor almost certainly falls through to a CPU copy (read back all values, rewrite in transposed order, upload again), because the Metal backend doesn't have a dedicated transpose-copy kernel. This happens **3 times per expert × 8 active experts × 48 layers = 1,152 CPU transpose-copies per decode step**.

---

## Architecture Bottlenecks (Not bugs, but slow-by-design)

### 4. No fused dequant-matmul kernel
Each expert forward does: dequantize Q4_K → F32 → matmul. A fused kernel that reads quantized weights and multiplies on-the-fly would cut memory bandwidth ~4× and eliminate the intermediate F32 buffer.

### 5. Pool allocator size (4 GB)
[`smoke.rs` L56](file:///Users/kaushikmuthukumar/Documents/projects/axiom/src/bin/smoke.rs#L56) allocates 4GB for the pool. The README explicitly warns oversized pools cause "real problems under memory pressure." With a 30B model whose dequantized expert weights are transient but large, the pool + model weights may exceed your Mac's unified memory, causing kernel memory compression/swap.

### 6. Single command buffer for all 48 layers + all expert work
The `MetalRunner` encodes **everything** into a single command buffer before `finish()`. For a 30B MoE model this is a massively long encoding that can't pipeline GPU execution with CPU-side command buffer construction.

---

## Recommended Fixes (Ordered by Impact)

| Priority | Fix | Expected Impact |
|:--------:|-----|:---------------:|
| **P0** | Pre-compute active expert set from routing output, skip inactive experts | **~16× fewer dispatch iterations** |
| **P0** | Cache `to_vec_u32()` result in dispatch loop instead of re-reading each iteration | **~128× fewer GPU readbacks per layer** |
| **P1** | Output dequantized weights in transposed layout directly (swap out_rows/out_cols in `dequant_one`) to eliminate `.transpose().contiguous()` CPU round-trip | **Eliminates 1,152 CPU copies/step** |
| **P1** | Reduce pool size from 4GB to ~512MB–1GB | **Reduces memory pressure** |
| **P2** | Move router gating + softmax + top-k to Metal | Further reduces GPU↔CPU syncs |
| **P2** | Move dispatch/combine to Metal (scatter/gather kernels) | Eliminates all per-token CPU work |
| **P3** | Fused dequant-matmul kernel | ~4× bandwidth reduction for expert matmuls |

---

## Scalability Options (Beyond Code Fixes)

If after the above fixes the model is still too slow for your needs:

1. **Smaller quantization**: If using Q4_K_M, try Q4_K_S or Q3_K — trades quality for speed
2. **Mac Studio / M2 Ultra / M4 Max**: More GPU cores + more unified memory bandwidth is the single biggest hardware lever for Apple Silicon inference
3. **Expert parallelism across machines**: Rent 2–4 Mac Minis with M4 chips, shard experts across them via network. Each machine handles a subset of experts. The communication volume is small (just the routed hidden states for active experts).
4. **Cloud GPU inference**: For batch throughput, a single A100/H100 with vLLM will be dramatically faster than Apple Silicon for a 30B model. Modal, RunPod, or Lambda Labs offer pay-per-second GPU rental.
5. **Speculative decoding**: Your `draft.rs` / `speculative.rs` files exist but aren't wired in. A small draft model (e.g., Qwen3-0.5B) verifying against the 30B model can yield 2–3× speedup.
