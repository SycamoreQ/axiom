use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;
use tokio::io::AsyncSeekExt;

/*
GGUF is the binary format used by llama.cpp and most community model distributions.
It replaced GGML. Every quantized LLaMA model on HuggingFace is distributed as .gguf.

Format layout:
  [Header] magic(u32) version(u32) tensor_count(u64) kv_count(u64)
  [Metadata] kv_count * (key_string, value_type_u32, value)
  [Tensor info] tensor_count * (name_string, n_dims_u32, dims[u64], dtype_u32, offset_u64)
  [Padding] aligned to 32 bytes
  [Tensor data] raw weight bytes, each tensor at its recorded offset
*/

#[derive(Debug)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
}

impl GgufDType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            34 => Some(Self::TQ1_0),
            35 => Some(Self::TQ2_0),
            39 => Some(Self::MXFP4),
            _ => None,
        }
    }

    //Returns bytes per element. For block-quantized types returns
    //the bytes for one block divided by elements per block.
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            Self::F32 | Self::I32 => 4.0,
            Self::F16 | Self::BF16 | Self::I16 => 2.0,
            Self::I8 => 1.0,
            Self::Q8_0 => 1.0 + (4.0 / 32.0), // 32 i8 + 1 f16 scale per block
            Self::Q4_0 => 0.5 + (2.0 / 32.0), // 32 nibbles + 1 f16 scale
            Self::Q4_1 => 0.5 + (4.0 / 32.0), // 32 nibbles + 2 f16
            _ => 0.5,                         // conservative default
        }
    }

    /// Is this type directly loadable as f32 without dequantization?
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDType,
    pub offset: u64, // byte offset relative to data_offset in GgufFile
}

impl GgufTensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Byte size on disk (approximate for quantized types).
    pub fn byte_size(&self) -> usize {
        (self.numel() as f32 * self.dtype.bytes_per_element()).ceil() as usize
    }
}

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub data: Mmap,
    //Absolute byte offset within `data` where tensor bytes begin.
    pub data_offset: usize,
}

impl GgufFile {
    //Open and parse a GGUF file, memory-mapping the tensor data.
    pub fn from_file(path: &Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path).map_err(GgufError::Io)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(GgufError::Io)?;
        parse_gguf_mmap(mmap)
    }

    //Raw byte slice for a named tensor.
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = self.data_offset + info.offset as usize;
        let end = start + info.byte_size();
        Some(&self.data[start..end])
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid magic bytes — not a GGUF file")]
    InvalidMagic,
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),
    #[error("unknown tensor dtype {0}")]
    UnknownDType(u32),
    #[error("invalid utf-8 string: {0}")]
    InvalidString(#[from] std::string::FromUtf8Error),
    #[error("parse error: {0}")]
    Parse(String),
}

const GGUF_MAGIC: u32 = 0x46554747;

fn parse_gguf_mmap(mmap: Mmap) -> Result<GgufFile, GgufError> {
    let mut cursor = Cursor::new(mmap.as_ref());

    // Header
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = cursor.read_u32::<LittleEndian>()?;
    if version < 2 || version > 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = cursor.read_u64::<LittleEndian>()?;
    let kv_count = cursor.read_u64::<LittleEndian>()?;

    // Metadata
    let mut metadata: HashMap<String, GgufValue> = HashMap::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = read_string(&mut cursor)?;
        let val_type = cursor.read_u32::<LittleEndian>()?;
        let value = read_value(&mut cursor, val_type)?;
        metadata.insert(key, value);
    }

    // Tensor info
    let mut tensors: HashMap<String, GgufTensorInfo> =
        HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut cursor)?;
        let n_dims = cursor.read_u32::<LittleEndian>()?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(cursor.read_u64::<LittleEndian>()?);
        }
        let dtype_raw = cursor.read_u32::<LittleEndian>()?;
        let dtype = GgufDType::from_u32(dtype_raw).ok_or(GgufError::UnknownDType(dtype_raw))?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        tensors.insert(
            name.clone(),
            GgufTensorInfo {
                name,
                shape,
                dtype,
                offset,
            },
        );
    }

    // Alignment padding — default 32, overridden by metadata
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| {
            if let GgufValue::Uint32(a) = v {
                Some(*a as u64)
            } else {
                None
            }
        })
        .unwrap_or(32u64);

    let pos = cursor.position();
    let data_offset = ((pos + alignment - 1) & !(alignment - 1)) as usize;

    Ok(GgufFile {
        metadata,
        tensors,
        data: mmap,
        data_offset,
    })
}

pub fn parse_gguf(buffer: &[u8]) -> Result<GgufFile, GgufError> {
    // Build a temporary Mmap-like owned copy via a memfd or just use a cursor.
    // For test purposes we parse metadata/tensors and return a dummy file
    // (data slice points into the buffer).
    // source : rvLLM
    let mut cursor = Cursor::new(buffer);

    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = cursor.read_u32::<LittleEndian>()?;
    if version < 2 || version > 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = cursor.read_u64::<LittleEndian>()?;
    let kv_count = cursor.read_u64::<LittleEndian>()?;

    let mut metadata: HashMap<String, GgufValue> = HashMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut cursor)?;
        let val_type = cursor.read_u32::<LittleEndian>()?;
        let value = read_value(&mut cursor, val_type)?;
        metadata.insert(key, value);
    }

    let mut tensors: HashMap<String, GgufTensorInfo> = HashMap::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut cursor)?;
        let n_dims = cursor.read_u32::<LittleEndian>()?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(cursor.read_u64::<LittleEndian>()?);
        }
        let dtype_raw = cursor.read_u32::<LittleEndian>()?;
        let dtype = GgufDType::from_u32(dtype_raw).ok_or(GgufError::UnknownDType(dtype_raw))?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        tensors.insert(
            name.clone(),
            GgufTensorInfo {
                name,
                shape,
                dtype,
                offset,
            },
        );
    }

    let mut current_offset = cursor.stream_position()?;

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| {
            if let GgufValue::Uint32(a) = v {
                Some(*a as u64)
            } else {
                None
            }
        })
        .unwrap_or(32u64);

    let pos = cursor.position();
    let data_offset = ((pos + alignment - 1) & !(alignment - 1)) as usize;

    // For in-memory tests, create an anonymous mmap from a temp file
    // backed by the buffer so GgufFile compiles. In practice always use from_file.
    use std::io::Write;
    let mut tmp = tempfile::tempfile().map_err(GgufError::Io)?;
    tmp.write_all(buffer).map_err(GgufError::Io)?;
    let mmap = unsafe { Mmap::map(&tmp) }.map_err(GgufError::Io)?;

    Ok(GgufFile {
        metadata,
        tensors,
        data: mmap,
        data_offset,
    })
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, GgufError> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_value(cursor: &mut Cursor<&[u8]>, val_type: u32) -> Result<GgufValue, GgufError> {
    match val_type {
        0 => Ok(GgufValue::Uint8(cursor.read_u8()?)),
        1 => Ok(GgufValue::Int8(cursor.read_i8()?)),
        2 => Ok(GgufValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(GgufValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(GgufValue::String(read_string(cursor)?)),
        9 => {
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_value(cursor, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::Float64(cursor.read_f64::<LittleEndian>()?)),
        t => Err(GgufError::UnknownValueType(t)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gguf_string_bytes(s: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
        b
    }

    fn minimal_gguf(kv_count: u64, extra_kvs: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(extra_kvs);
        buf
    }

    // ── dtype ──

    #[test]
    fn test_dtype_from_u32_known() {
        assert_eq!(GgufDType::from_u32(0), Some(GgufDType::F32));
        assert_eq!(GgufDType::from_u32(1), Some(GgufDType::F16));
        assert_eq!(GgufDType::from_u32(30), Some(GgufDType::BF16));
        assert_eq!(GgufDType::from_u32(2), Some(GgufDType::Q4_0));
        assert_eq!(GgufDType::from_u32(8), Some(GgufDType::Q8_0));
    }

    #[test]
    fn test_dtype_from_u32_unknown() {
        assert_eq!(GgufDType::from_u32(999), None);
    }

    #[test]
    fn test_dtype_is_float() {
        assert!(GgufDType::F32.is_float());
        assert!(GgufDType::F16.is_float());
        assert!(GgufDType::BF16.is_float());
        assert!(!GgufDType::Q4_0.is_float());
    }

    // ── header ──

    #[test]
    fn test_parse_invalid_magic() {
        let mut buf = vec![0u8; 24];
        buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        assert!(matches!(parse_gguf(&buf), Err(GgufError::InvalidMagic)));
    }

    #[test]
    fn test_parse_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // version 1 — unsupported
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        assert!(matches!(
            parse_gguf(&buf),
            Err(GgufError::UnsupportedVersion(1))
        ));
    }

    #[test]
    fn test_parse_empty_gguf() {
        let buf = minimal_gguf(0, &[]);
        let gguf = parse_gguf(&buf).unwrap();
        assert!(gguf.metadata.is_empty());
        assert!(gguf.tensors.is_empty());
    }

    // ── metadata value types ──

    #[test]
    fn test_parse_metadata_uint32() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("test_key"));
        kv.extend_from_slice(&4u32.to_le_bytes()); // uint32
        kv.extend_from_slice(&42u32.to_le_bytes());
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        assert!(matches!(gguf.metadata["test_key"], GgufValue::Uint32(42)));
    }

    #[test]
    fn test_parse_metadata_float32() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("rope_theta"));
        kv.extend_from_slice(&6u32.to_le_bytes()); // float32
        kv.extend_from_slice(&500000.0f32.to_le_bytes());
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        assert!(
            matches!(gguf.metadata["rope_theta"], GgufValue::Float32(v) if (v - 500000.0).abs() < 1.0)
        );
    }

    #[test]
    fn test_parse_metadata_string() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("general.architecture"));
        kv.extend_from_slice(&8u32.to_le_bytes()); // string
        kv.extend_from_slice(&gguf_string_bytes("llama"));
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        assert!(
            matches!(&gguf.metadata["general.architecture"], GgufValue::String(s) if s == "llama")
        );
    }

    #[test]
    fn test_parse_metadata_bool() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("some_bool"));
        kv.extend_from_slice(&7u32.to_le_bytes()); // bool
        kv.push(1u8);
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        assert!(matches!(gguf.metadata["some_bool"], GgufValue::Bool(true)));
    }

    #[test]
    fn test_parse_metadata_uint64() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("big_num"));
        kv.extend_from_slice(&10u32.to_le_bytes()); // uint64
        kv.extend_from_slice(&u64::MAX.to_le_bytes());
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        assert!(matches!(
            gguf.metadata["big_num"],
            GgufValue::Uint64(u64::MAX)
        ));
    }

    #[test]
    fn test_parse_metadata_array_of_uint32() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("arr"));
        kv.extend_from_slice(&9u32.to_le_bytes()); // array
        kv.extend_from_slice(&4u32.to_le_bytes()); // elem type = uint32
        kv.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        for v in [1u32, 2, 3] {
            kv.extend_from_slice(&v.to_le_bytes());
        }
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        match &gguf.metadata["arr"] {
            GgufValue::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert!(matches!(arr[0], GgufValue::Uint32(1)));
                assert!(matches!(arr[1], GgufValue::Uint32(2)));
                assert!(matches!(arr[2], GgufValue::Uint32(3)));
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_parse_multiple_metadata_keys() {
        let mut kv = Vec::new();
        for (name, val) in [("a", 1u32), ("b", 2), ("c", 3)] {
            kv.extend_from_slice(&gguf_string_bytes(name));
            kv.extend_from_slice(&4u32.to_le_bytes());
            kv.extend_from_slice(&val.to_le_bytes());
        }
        let gguf = parse_gguf(&minimal_gguf(3, &kv)).unwrap();
        assert_eq!(gguf.metadata.len(), 3);
    }

    // ── convenience accessors ──

    #[test]
    fn test_get_u32() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("hidden_size"));
        kv.extend_from_slice(&4u32.to_le_bytes());
        kv.extend_from_slice(&4096u32.to_le_bytes());
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        assert_eq!(gguf.get_u32("hidden_size"), Some(4096));
        assert_eq!(gguf.get_u32("missing"), None);
    }

    #[test]
    fn test_get_string() {
        let mut kv = Vec::new();
        kv.extend_from_slice(&gguf_string_bytes("arch"));
        kv.extend_from_slice(&8u32.to_le_bytes());
        kv.extend_from_slice(&gguf_string_bytes("llama"));
        let gguf = parse_gguf(&minimal_gguf(1, &kv)).unwrap();
        assert_eq!(gguf.get_string("arch"), Some("llama"));
    }

    // ── tensor info ──

    #[test]
    fn test_tensor_numel() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![2, 3, 4],
            dtype: GgufDType::F32,
            offset: 0,
        };
        assert_eq!(info.numel(), 24);
    }

    #[test]
    fn test_tensor_byte_size_f32() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![4, 4],
            dtype: GgufDType::F32,
            offset: 0,
        };
        assert_eq!(info.byte_size(), 64); // 16 * 4
    }

    #[test]
    fn test_tensor_byte_size_f16() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![4, 4],
            dtype: GgufDType::F16,
            offset: 0,
        };
        assert_eq!(info.byte_size(), 32); // 16 * 2
    }

    // ── real file ──

    #[test]
    fn test_real_gguf_file() {
        let path = std::path::Path::new("testdata/tinyllama.gguf");
        if !path.exists() {
            return;
        }
        let gguf = GgufFile::from_file(path).unwrap();
        assert!(!gguf.metadata.is_empty());
        assert!(!gguf.tensors.is_empty());
        assert!(gguf.get_string("general.architecture").is_some());
    }

    #[test]
    fn test_real_gguf_tensor_data() {
        let path = std::path::Path::new("testdata/tinyllama.gguf");
        if !path.exists() {
            return;
        }
        let gguf = GgufFile::from_file(path).unwrap();
        // embedding table always present in LLaMA
        let data = gguf.get_tensor_data("token_embd.weight");
        assert!(data.is_some());
        assert!(!data.unwrap().is_empty());
    }
}
