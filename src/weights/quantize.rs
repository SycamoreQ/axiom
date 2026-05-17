/*
Dequantization kernels for GGUF quantized weight formats.
These are stubs — full implementations come in Phase 4 (CUDA) and Phase 8 (CPU).

Supported formats to implement:
  Q4_0  — 4-bit with f16 scale per 32-element block
  Q4_1  — 4-bit with f16 scale + bias per 32-element block
  Q8_0  — 8-bit with f16 scale per 32-element block
  Q4_K  — 4-bit super-blocks (used in most modern GGUF files)
  Q6_K  — 6-bit super-blocks

References:
  https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c
  https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c
*/

use axum::http::header::CONTENT_SECURITY_POLICY_REPORT_ONLY;

use crate::weights::gguf::GgufDType;

//Dequantize a block-quantized byte slice into f32 values.
//'numel' is the total number of elements the output should contain.
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

//Block size: 32 elements
//Layout per block: [f16 scale (2 bytes)] [16 bytes of nibbles]
//Each nibble stores a 4-bit value in range [0, 15], centered at 8

const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 2 + Q4_0_BLOCK_SIZE / 2; // 18 bytes

fn dequantize_q4_0(data: &[u8], numel: usize) -> Vec<f32> {
    // block layout: 2 bytes f16 scale + 16 bytes nibbles = 18 bytes per 32 elements
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q4_0_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        // first 2 bytes: f16 scale
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        // next 16 bytes: 32 nibbles packed, two per byte
        // low nibble first, then high nibble
        for byte_idx in 0..16 {
            if out_idx + 1 >= numel + 1 {
                break;
            }
            let byte = block[2 + byte_idx];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;

            if out_idx < numel {
                out[out_idx] = lo as f32 * scale;
                out_idx += 1;
            }
            if out_idx < numel {
                out[out_idx] = hi as f32 * scale;
                out_idx += 1;
            }
        }
    }

    out
}

//Block size: 32 elements
//Layout: [f16 scale (2)] [f16 min (2)] [16 bytes nibbles]

const Q4_1_BLOCK_BYTES: usize = 4 + Q4_0_BLOCK_SIZE / 2; // 20 bytes

fn dequantize_q4_1(data: &[u8], numel: usize) -> Vec<f32> {
    // todo()
    let _ = data;
    vec![0.0f32; numel]
}

//Block size: 32 elements
//Layout: [f16 scale (2)] [32 i8 values]

const Q8_0_BLOCK_BYTES: usize = 2 + Q4_0_BLOCK_SIZE; // 34 bytes

fn dequantize_q8_0(data: &[u8], numel: usize) -> Vec<f32> {
    // todo()
    let _ = data;
    vec![0.0f32; numel]
}

//Super-block of 256 elements (8 sub-blocks of 32)
//More complex layout — see ggml-quants.c for reference

fn dequantize_q4_k(data: &[u8], numel: usize) -> Vec<f32> {
    // todo()
    let _ = data;
    vec![0.0f32; numel]
}

fn dequantize_q6_k(data: &[u8], numel: usize) -> Vec<f32> {
    // todo()
    let _ = data;
    vec![0.0f32; numel]
}

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
        // scale = f16(0.0) = 0x0000, nibbles all 0x88 (value 8, centered = 0)
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        // scale bytes = 0x0000 = f16 zero
        block[0] = 0x00;
        block[1] = 0x00;
        // nibbles: 0x88 = lo nibble 8, hi nibble 8 → both center at 0
        for i in 2..Q4_0_BLOCK_BYTES {
            block[i] = 0x88;
        }
        let out = dequantize_q4_0(&block, 32);
        assert_eq!(out.len(), 32);
        for v in &out {
            assert_eq!(*v, 0.0, "expected zero with centered nibbles");
        }
    }

    #[test]
    fn test_q4_0_known_values() {
        // construct one block with known scale and nibbles
        // scale = 1.0 in f16 = 0x3C00
        // nibble value 15 → (15 - 8) * 1.0 = 7.0
        // nibble value 0  → (0  - 8) * 1.0 = -8.0
        // pack as byte: lo=15 (0xF), hi=0 (0x0) → byte = 0x0F
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0 in little-endian = [0x00, 0x3C]
                         // first nibble byte: lo=15, hi=0
        block[2] = 0x0F;
        // rest: lo=8, hi=8 → centered at 0
        for i in 3..Q4_0_BLOCK_BYTES {
            block[i] = 0x88;
        }
        let out = dequantize_q4_0(&block, 32);
        assert_eq!(out.len(), 32);
        assert!(
            (out[0] - 7.0).abs() < 1e-3,
            "lo nibble 15 → 7.0, got {}",
            out[0]
        );
        assert!(
            (out[1] - (-8.0)).abs() < 1e-3,
            "hi nibble 0 → -8.0, got {}",
            out[1]
        );
        // rest should be 0.0
        for v in &out[2..] {
            assert!(v.abs() < 1e-3, "expected ~0.0, got {}", v);
        }
    }

    #[test]
    fn test_q4_0_multiple_blocks() {
        // two blocks, verify output length = 64
        let block = vec![0u8; Q4_0_BLOCK_BYTES];
        let data = [block.clone(), block].concat();
        let out = dequantize_q4_0(&data, 64);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_q4_0_numel_truncation() {
        // request fewer elements than blocks provide
        let block = vec![0u8; Q4_0_BLOCK_BYTES];
        let out = dequantize_q4_0(&block, 16);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_dequantize_dispatch_q4_0() {
        // verify dequantize() routes to q4_0 correctly
        let block = vec![0u8; Q4_0_BLOCK_BYTES];
        let out = dequantize(&block, GgufDType::Q4_0, 32);
        assert_eq!(out.len(), 32);
    }
}
