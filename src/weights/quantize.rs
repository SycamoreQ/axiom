/*
Dequantization kernels for GGUF quantized weight formats.
Ported from llama.cpp / ggml-quants.c

Supported:
  Q4_0  — 4-bit with f16 scale per 32-element block
  Q4_1  — 4-bit with f16 scale + bias per 32-element block
  Q8_0  — 8-bit with f16 scale per 32-element block
  Q4_K  — 4-bit super-blocks of 256 elements
  Q6_K  — 6-bit super-blocks of 256 elements

References:
  https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c
*/

use crate::weights::gguf::GgufDType;

//Dequantize a block-quantized byte slice into f32 values.
pub fn dequantize(data: &[u8], dtype: GgufDType, numel: usize) -> Vec<f32> {
    match dtype {
        GgufDType::Q4_0 => dequantize_q4_0(data, numel),
        GgufDType::Q4_1 => dequantize_q4_1(data, numel),
        GgufDType::Q8_0 => dequantize_q8_0(data, numel),
        GgufDType::Q4_K => dequantize_q4_k(data, numel),
        GgufDType::Q6_K => dequantize_q6_k(data, numel),
        _ => {
            eprintln!(
                "WARNING: dequantize called for unsupported dtype {:?} — returning zeros",
                dtype
            );
            vec![0.0f32; numel]
        }
    }
}

//
// Block layout (18 bytes, 32 elements):
//   bytes [0..2]  : f16 scale `d`
//   bytes [2..18] : 16 bytes of nibbles
//     qs[j] low  nibble → element j       (value = nibble - 8) * d
//     qs[j] high nibble → element j + 16  (value = nibble - 8) * d

const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 2 + Q4_0_BLOCK_SIZE / 2; // 18 bytes

fn dequantize_q4_0(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q4_0_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();
        let qs = &block[2..18];

        for j in 0..16 {
            if out_idx >= numel {
                break;
            }
            out[out_idx] = ((qs[j] & 0x0F) as i32 - 8) as f32 * scale;
            out_idx += 1;
        }
        for j in 0..16 {
            if out_idx >= numel {
                break;
            }
            out[out_idx] = ((qs[j] >> 4) as i32 - 8) as f32 * scale;
            out_idx += 1;
        }
    }

    out
}

//
// Block layout (20 bytes, 32 elements):
//   bytes [0..2]  : f16 scale `d`
//   bytes [2..4]  : f16 minimum `m`
//   bytes [4..20] : 16 bytes of nibbles
//     element j       = (qs[j] & 0xF) * d + m
//     element j + 16  = (qs[j] >>  4) * d + m

const Q4_1_BLOCK_BYTES: usize = 4 + Q4_0_BLOCK_SIZE / 2; // 20 bytes

fn dequantize_q4_1(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q4_1_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let m = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
        let qs = &block[4..20];

        for j in 0..16 {
            if out_idx >= numel {
                break;
            }
            out[out_idx] = (qs[j] & 0x0F) as f32 * d + m;
            out_idx += 1;
        }
        for j in 0..16 {
            if out_idx >= numel {
                break;
            }
            out[out_idx] = (qs[j] >> 4) as f32 * d + m;
            out_idx += 1;
        }
    }

    out
}

//
// Block layout (34 bytes, 32 elements):
//   bytes [0..2]   : f16 scale `d`
//   bytes [2..34]  : 32 signed i8 quantized values
//     element j = qs[j] * d

const Q8_0_BLOCK_BYTES: usize = 2 + Q4_0_BLOCK_SIZE; // 34 bytes

fn dequantize_q8_0(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q8_0_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let qs = &block[2..34];

        for &q in qs {
            if out_idx >= numel {
                break;
            }
            out[out_idx] = (q as i8) as f32 * d;
            out_idx += 1;
        }
    }

    out
}

// ─── Q4_K ────────────────────────────────────────────────────────────────────
//
// Super-block of 256 elements (8 sub-blocks of 32 each).
// Block layout (144 bytes):
//   bytes [0..2]    : f16 super-block scale `d`
//   bytes [2..4]    : f16 super-block min   `dmin`
//   bytes [4..16]   : 12 bytes of packed 6-bit scales and mins (8 scale + 8 min)
//   bytes [16..144] : 128 bytes of 4-bit quantized values (256 nibbles)
//
// Scale unpacking (6 bits each, 8 scales + 8 mins packed into 12 bytes):
//   The 12 bytes encode 16 values of 6 bits each using the layout from ggml-quants.c:
//     scales[0..5]  encode scale[0..7] (low 6 bits of bytes 0..5 plus high bits of 8..11)
//     scales[0..5]  also encode min[0..7] in upper 4 bits + low 2 bits of bytes 6..11
//
// Value reconstruction per sub-block s (0..8), element j (0..32):
//   nibble index = s*32 + j  (but stored across 128 bytes as pairs of nibbles)
//   x_j = (nibble & 0xF) * scale[s] - min[s] * dmin  (low nibbles: elements 0..15)
//   x_j = (nibble >> 4)  * scale[s] - min[s] * dmin  (high nibbles: elements 16..31)

const Q4_K_BLOCK_SIZE: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 4 + 12 + Q4_K_BLOCK_SIZE / 2; // 144 bytes

fn dequantize_q4_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q4_K_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        // super‑block scale and min (both f16)
        let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();

        let sc = &block[4..16]; // 12 bytes of packed 6‑bit scales and mins
        let qs = &block[16..144]; // 128 bytes of 4‑bit values

        // Unpack 8 scales and 8 mins (each 6 bits)
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = sc[i] & 0x3F;
            mins[i] = sc[i + 4] & 0x3F;
        }
        for i in 0..4 {
            scales[4 + i] = (sc[8 + i] & 0x0F) | ((sc[i] >> 6) << 4);
            mins[4 + i] = (sc[8 + i] >> 4) | ((sc[i + 4] >> 6) << 4);
        }

        // 8 sub‑blocks of 32 elements each
        for s in 0..8 {
            let scale = scales[s] as f32 * d; // scale for this sub‑block
            let magic = mins[s] as f32 * dmin; // offset (not scaled by d)

            let qblock = &qs[s * 16..(s + 1) * 16];

            // Low nibbles → elements 0..15
            for j in 0..16 {
                if out_idx >= numel {
                    break;
                }
                let nibble = (qblock[j] & 0x0F) as f32 - 32.0;
                out[out_idx] = scale * nibble - magic;
                out_idx += 1;
            }
            // High nibbles → elements 16..31
            for j in 0..16 {
                if out_idx >= numel {
                    break;
                }
                let nibble = (qblock[j] >> 4) as f32 - 32.0;
                out[out_idx] = scale * nibble - magic;
                out_idx += 1;
            }
        }
    }

    out
}
//
// Super-block of 256 elements.
// Block layout (210 bytes):
//   bytes [0..128]   : ql — lower 4 bits of each 6-bit value (128 bytes, 2 per byte)
//   bytes [128..192] : qh — upper 2 bits of each 6-bit value (64 bytes, 4 per byte)
//   bytes [192..208] : scales — 16 int8 values, one per 16-element sub-block
//   bytes [208..210] : f16 super-block scale `d`
//
// Value reconstruction for element i:
//   lower 4 bits: from ql[i/2], lo nibble if i even, hi nibble if i odd
//   upper 2 bits: from qh[i/4], at bit position 2*(i%4)
//   6-bit value q = lower | (upper << 4), centered at 32 → q -= 32
//   scale for element i: scales[i/16] (int8)
//   final: q * scales[i/16] * d

const Q6_K_BLOCK_SIZE: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // 210 bytes

fn dequantize_q6_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q6_K_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];

        for i in 0..Q6_K_BLOCK_SIZE {
            if out_idx >= numel {
                break;
            }

            let lower = if i % 2 == 0 {
                (ql[i / 2] & 0x0F) as i32
            } else {
                ((ql[i / 2] >> 4) & 0x0F) as i32
            };

            let shift = (i % 4) * 2;
            let upper = ((qh[i / 4] >> shift) & 0x03) as i32;

            let q = (lower | (upper << 4)) - 32;
            let scale = scales[i / 16] as i8 as f32;

            out[out_idx] = q as f32 * scale * d;
            out_idx += 1;
        }
    }

    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_stub_returns_correct_length() {
        let data = vec![0u8; 100];
        let out = dequantize(&data, GgufDType::Q4_0, 64);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_dequantize_unsupported_returns_zeros() {
        let data = vec![0u8; 100];
        let out = dequantize(&data, GgufDType::IQ1_S, 32);
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_q4_0_block_size_constant() {
        assert_eq!(Q4_0_BLOCK_SIZE, 32);
        assert_eq!(Q4_0_BLOCK_BYTES, 18);
    }

    #[test]
    fn test_q8_0_block_bytes() {
        assert_eq!(Q8_0_BLOCK_BYTES, 34);
    }

    #[test]
    fn test_q4_0_zero_scale_produces_zeros() {
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x00;
        for i in 2..Q4_0_BLOCK_BYTES {
            block[i] = 0x88; // nibbles 8 and 8 → 8-8=0
        }
        let out = dequantize_q4_0(&block, 32);
        assert_eq!(out.len(), 32);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_q4_0_known_values() {
        // scale = f16(1.0) = 0x3C00, LE = [0x00, 0x3C]
        // qs[0] = 0x0F → lo nibble=15→15-8=7, hi nibble=0→0-8=-8
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[2] = 0x0F; // lo=15 → elem 0 = 7.0, hi=0 → elem 16 = -8.0
        for i in 3..Q4_0_BLOCK_BYTES {
            block[i] = 0x88; // rest = 0
        }
        let out = dequantize_q4_0(&block, 32);
        assert_eq!(out.len(), 32);
        assert!((out[0] - 7.0).abs() < 1e-3, "elem 0 = {}", out[0]);
        assert!((out[16] - -8.0).abs() < 1e-3, "elem 16 = {}", out[16]);
        for i in 1..16 {
            assert!(out[i].abs() < 1e-3, "elem {} = {}", i, out[i]);
        }
        for i in 17..32 {
            assert!(out[i].abs() < 1e-3, "elem {} = {}", i, out[i]);
        }
    }

    #[test]
    fn test_q4_0_multiple_blocks() {
        let block = vec![0u8; Q4_0_BLOCK_BYTES];
        let data = [block.clone(), block].concat();
        let out = dequantize_q4_0(&data, 64);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_q4_0_numel_truncation() {
        let block = vec![0u8; Q4_0_BLOCK_BYTES];
        let out = dequantize_q4_0(&block, 16);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_dequantize_dispatch_q4_0() {
        let block = vec![0u8; Q4_0_BLOCK_BYTES];
        let out = dequantize(&block, GgufDType::Q4_0, 32);
        assert_eq!(out.len(), 32);
    }

    // ── Q4_1 ──

    #[test]
    fn test_q4_1_block_bytes() {
        assert_eq!(Q4_1_BLOCK_BYTES, 20);
    }

    #[test]
    fn test_q4_1_zero_d_zero_m_produces_zeros() {
        let block = vec![0u8; Q4_1_BLOCK_BYTES]; // d=0, m=0
        let out = dequantize_q4_1(&block, 32);
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_q4_1_known_value() {
        // d = f16(1.0) = [0x00, 0x3C]
        // m = f16(0.0) = [0x00, 0x00]
        // qs[0] = 0xF0 → lo nibble=0 → elem 0 = 0*1+0 = 0
        //               → hi nibble=F=15 → elem 16 = 15*1+0 = 15
        let mut block = vec![0u8; Q4_1_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // d=1.0
        block[2] = 0x00;
        block[3] = 0x00; // m=0.0
        block[4] = 0xF0; // lo=0, hi=15
        let out = dequantize_q4_1(&block, 32);
        assert_eq!(out.len(), 32);
        assert!((out[0] - 0.0).abs() < 1e-3);
        assert!((out[16] - 15.0).abs() < 1e-3);
    }

    // ── Q8_0 ──

    #[test]
    fn test_q8_0_zero_scale_produces_zeros() {
        let block = vec![0u8; Q8_0_BLOCK_BYTES];
        let out = dequantize_q8_0(&block, 32);
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_q8_0_known_value() {
        // scale = f16(1.0), qs[0] = 42 (positive i8)
        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // d=1.0
        block[2] = 42u8;
        let out = dequantize_q8_0(&block, 32);
        assert!((out[0] - 42.0).abs() < 1e-3, "elem 0 = {}", out[0]);
    }

    #[test]
    fn test_q8_0_negative_i8() {
        // scale = f16(1.0), qs[0] = 0xFF = -1 as i8
        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C;
        block[2] = 0xFF; // -1 as i8
        let out = dequantize_q8_0(&block, 32);
        assert!((out[0] - (-1.0)).abs() < 1e-3);
    }

    // ── Q4_K ──

    #[test]
    fn test_q4_k_block_bytes() {
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
    }

    #[test]
    fn test_q4_k_output_length() {
        let data = vec![0u8; Q4_K_BLOCK_BYTES];
        let out = dequantize_q4_k(&data, 256);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_q4_k_zero_block_produces_zeros() {
        // d=0, dmin=0 → all outputs zero regardless of nibbles
        let block = vec![0u8; Q4_K_BLOCK_BYTES];
        let out = dequantize_q4_k(&block, 256);
        assert!(out.iter().all(|&v| v == 0.0), "expected all zeros");
    }

    #[test]
    fn test_q4_k_multiple_blocks() {
        let block = vec![0u8; Q4_K_BLOCK_BYTES];
        let data = [block.clone(), block].concat();
        let out = dequantize_q4_k(&data, 512);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_q4_k_numel_truncation() {
        let block = vec![0u8; Q4_K_BLOCK_BYTES];
        let out = dequantize_q4_k(&block, 128);
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn test_q4_k_nonzero_scale_gives_nonzero_output() {
        // Set d=1.0, dmin=0.0, scales[0]=1, all nibbles=0xF (15)
        // Expected: first 16 elems = 15 * 1 - 0 = 15.0
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        // d = f16(1.0) = [0x00, 0x3C]
        block[0] = 0x00;
        block[1] = 0x3C;
        // dmin = f16(0.0) = [0x00, 0x00]
        block[2] = 0x00;
        block[3] = 0x00;
        // scales[0] = 1 → byte 4 = 0x01 (sc[0] & 0x3F = 1)
        block[4] = 0x01;
        // qs: first 16 bytes all 0xFF → all nibbles = 0xF = 15
        for i in 16..32 {
            block[i] = 0xFF;
        }
        let out = dequantize_q4_k(&block, 32);
        // first sub-block: scale = 1*1.0 = 1.0, min = 0
        assert!(
            (out[0] - 15.0).abs() < 1e-3,
            "expected 15.0, got {}",
            out[0]
        );
    }

    // ── Q6_K ──

    #[test]
    fn test_q6_k_block_bytes_constant() {
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
    }

    #[test]
    fn test_q6_k_output_length() {
        let data = vec![0u8; Q6_K_BLOCK_BYTES];
        let out = dequantize_q6_k(&data, 256);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_q6_k_zero_scale_produces_zeros() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        block[208] = 0x00;
        block[209] = 0x00;
        let out = dequantize_q6_k(&block, 256);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_q6_k_multiple_blocks() {
        let block = vec![0u8; Q6_K_BLOCK_BYTES];
        let data = [block.clone(), block].concat();
        let out = dequantize_q6_k(&data, 512);
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn test_q6_k_numel_truncation() {
        let block = vec![0u8; Q6_K_BLOCK_BYTES];
        let out = dequantize_q6_k(&block, 128);
        assert_eq!(out.len(), 128);
    }

    #[test]
    fn test_dequantize_dispatch_q6_k() {
        let block = vec![0u8; Q6_K_BLOCK_BYTES];
        let out = dequantize(&block, GgufDType::Q6_K, 256);
        assert_eq!(out.len(), 256);
    }
}
