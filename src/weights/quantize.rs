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
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q4_0_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        // little-endian f16 scale
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        // 16 bytes = 32 nibbles
        // qs[i] low  nibble = element i
        // qs[i] high nibble = element i + 16
        let qs = &block[2..18];

        // first pass: low nibbles → elements 0..15
        for i in 0..16 {
            if out_idx >= numel {
                break;
            }
            let lo = (qs[i] & 0x0F) as i32 - 8;
            out[out_idx] = lo as f32 * scale;
            out_idx += 1;
        }

        // second pass: high nibbles → elements 16..31
        for i in 0..16 {
            if out_idx >= numel {
                break;
            }
            let hi = ((qs[i] >> 4) & 0x0F) as i32 - 8;
            out[out_idx] = hi as f32 * scale;
            out_idx += 1;
        }
    }

    out
}

const Q6_K_BLOCK_SIZE: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // 210 bytes

fn dequantize_q6_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; numel];
    let mut out_idx = 0;

    for block in data.chunks_exact(Q6_K_BLOCK_BYTES) {
        if out_idx >= numel {
            break;
        }

        // superblock scale: f16 at bytes 208-209
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d = half::f16::from_bits(d_bits).to_f32();

        // ql: bytes 0..128   — lower 4 bits of each quant
        // qh: bytes 128..192 — upper 2 bits of each quant
        // scales: bytes 192..208 — int8 scale per 16-element subblock
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];

        // reconstruct 256 quantized values
        for i in 0..Q6_K_BLOCK_SIZE {
            if out_idx >= numel {
                break;
            }

            // lower 4 bits come from ql
            // ql packs two 4-bit values per byte
            // element i uses ql[i/2], taking low nibble for even i, high for odd
            let ql_byte = ql[i / 2];
            let lower = if i % 2 == 0 {
                (ql_byte & 0x0F) as i32
            } else {
                ((ql_byte >> 4) & 0x0F) as i32
            };

            // upper 2 bits come from qh
            // qh packs four 2-bit values per byte
            // element i uses qh[i/4], taking bits [2*(i%4)] and [2*(i%4)+1]
            let qh_byte = qh[i / 4];
            let shift = (i % 4) * 2;
            let upper = ((qh_byte >> shift) & 0x03) as i32;

            // combine: 6-bit value centered at 32
            let q = (lower | (upper << 4)) - 32;

            // scale: one int8 scale per 16-element subblock
            let scale_idx = i / 16;
            let scale = scales[scale_idx] as i8 as f32;

            out[out_idx] = q as f32 * scale * d;
            out_idx += 1;
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
        // scale = f16(1.0) = 0x3C00, little-endian = [0x00, 0x3C]
        // qs[0] = 0x0F → lo nibble=15 → 15-8=7, hi nibble=0 → 0-8=-8
        // element 0 = 7.0, element 16 = -8.0
        let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[2] = 0x0F; // lo=15, hi=0
                         // rest: 0x88 → lo=8-8=0, hi=8-8=0
        for i in 3..Q4_0_BLOCK_BYTES {
            block[i] = 0x88;
        }
        let out = dequantize_q4_0(&block, 32);
        assert_eq!(out.len(), 32);
        assert!(
            (out[0] - 7.0).abs() < 1e-3,
            "element 0 should be 7.0, got {}",
            out[0]
        );
        assert!(
            (out[16] - (-8.0)).abs() < 1e-3,
            "element 16 should be -8.0, got {}",
            out[16]
        );
        for i in 1..16 {
            assert!(
                out[i].abs() < 1e-3,
                "element {} should be 0.0, got {}",
                i,
                out[i]
            );
        }
        for i in 17..32 {
            assert!(
                out[i].abs() < 1e-3,
                "element {} should be 0.0, got {}",
                i,
                out[i]
            );
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
        // d = f16(0) = 0x0000, all other bytes zero
        // q values = (0 | 0) - 32 = -32, but scale d=0 → output 0
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
