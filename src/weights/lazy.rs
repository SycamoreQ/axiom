use crate::weights::gguf::GgufDType;
use crate::weights::quantize::dequantize;

pub struct QuantizedWeight {
    pub dtype: GgufDType,
    pub data: Vec<u8>,     // raw GGUF bytes, untouched
    pub shape: Vec<usize>, // [out_features, in_features] or [vocab, hidden]
    pub numel: usize,
}

impl QuantizedWeight {
    pub fn materialize(&self) -> Vec<f32> {
        dequantize(&self.data, self.dtype, self.numel)
    }

    // Dequantize a contiguous row range without touching the rest of the tensor.
    // Only valid for row-major 2D tensors where each row starts on a block boundary
    // (true for Q4_K/Q6_K since hidden_size is always a multiple of 256).
    pub fn materialize_rows(&self, row_start: usize, row_end: usize) -> Vec<f32> {
        let row_len = self.shape[1];
        let (block_elems, block_bytes) =
            block_info(self.dtype).expect("materialize_rows: unsupported dtype for row slicing");
        assert_eq!(row_len % block_elems, 0, "row length must be block-aligned");

        let bytes_per_row = (row_len / block_elems) * block_bytes;
        let byte_start = row_start * bytes_per_row;
        let byte_end = row_end * bytes_per_row;
        let numel = (row_end - row_start) * row_len;

        dequantize(&self.data[byte_start..byte_end], self.dtype, numel)
    }
}

// (block_size_in_elements, block_size_in_bytes) for row-alignment math.
pub fn block_info(dtype: GgufDType) -> Option<(usize, usize)> {
    match dtype {
        GgufDType::Q4_0 => Some((32, 18)),
        GgufDType::Q4_1 => Some((32, 20)),
        GgufDType::Q8_0 => Some((32, 34)),
        GgufDType::Q4_K => Some((256, 144)),
        GgufDType::Q6_K => Some((256, 210)),
        _ => None,
    }
}
