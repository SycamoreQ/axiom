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
    // todo(): Phase 8
    let _ = data;
    vec![0.0f32; numel]
}

//Block size: 32 elements
//Layout: [f16 scale (2)] [f16 min (2)] [16 bytes nibbles]

const Q4_1_BLOCK_BYTES: usize = 4 + Q4_0_BLOCK_SIZE / 2; // 20 bytes

fn dequantize_q4_1(data: &[u8], numel: usize) -> Vec<f32> {
    // todo(): Phase 8
    let _ = data;
    vec![0.0f32; numel]
}

//Block size: 32 elements
//Layout: [f16 scale (2)] [32 i8 values]

const Q8_0_BLOCK_BYTES: usize = 2 + Q4_0_BLOCK_SIZE; // 34 bytes

fn dequantize_q8_0(data: &[u8], numel: usize) -> Vec<f32> {
    // todo(): Phase 8
    let _ = data;
    vec![0.0f32; numel]
}

//Super-block of 256 elements (8 sub-blocks of 32)
//More complex layout — see ggml-quants.c for reference

fn dequantize_q4_k(data: &[u8], numel: usize) -> Vec<f32> {
    // todo(): Phase 8
    let _ = data;
    vec![0.0f32; numel]
}

fn dequantize_q6_k(data: &[u8], numel: usize) -> Vec<f32> {
    // todo(): Phase 8
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
}
