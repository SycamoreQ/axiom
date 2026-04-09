use crate::core::error::CoreError;
use candle_core::DType as CandleDType;
use half;

/*
Types of quantizations and their implementations
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U32,
    F32,
    F16,
    BF16,
}

pub trait Element: Copy + Send + Sync + 'static {
    fn dtype() -> DType;
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U32 => 4,
        }
    }

    pub fn is_half(self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }

    /// String suitable for kernel dispatch ("float32", "float16", "bfloat16").
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::U32 => "u32",
        }
    }
}

impl Element for f32 {
    fn dtype() -> DType {
        DType::F32
    }
}
impl Element for half::f16 {
    fn dtype() -> DType {
        DType::F16
    }
}
impl Element for half::bf16 {
    fn dtype() -> DType {
        DType::BF16
    }
}

impl From<DType> for candle_core::DType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::F32 => candle_core::DType::F32,
            DType::F16 => candle_core::DType::F16,
            DType::BF16 => candle_core::DType::BF16,
            DType::U32 => candle_core::DType::U32,
        }
    }
}

impl TryFrom<candle_core::DType> for DType {
    type Error = crate::core::error::CoreError;
    fn try_from(dt: candle_core::DType) -> Result<Self, Self::Error> {
        match dt {
            candle_core::DType::F32 => Ok(DType::F32),
            candle_core::DType::F16 => Ok(DType::F16),
            candle_core::DType::BF16 => Ok(DType::BF16),
            candle_core::DType::U32 => Ok(DType::U32),
            other => Err(CoreError::Internal(format!(
                "unsupported candle dtype: {:?}",
                other
            ))),
        }
    }
}

pub fn promote(a: DType, b: DType) -> DType {
    // F32 beats everything
    // BF16 beats F16
    match (a, b) {
        (DType::F32, _) | (_, DType::F32) => DType::F32,
        (DType::BF16, _) | (_, DType::BF16) => DType::BF16,
        _ => DType::F16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
    }

    #[test]
    fn test_name() {
        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::F16.name(), "f16");
        assert_eq!(DType::BF16.name(), "bf16");
    }

    #[test]
    fn test_element_dtype() {
        assert_eq!(f32::dtype(), DType::F32);
        assert_eq!(half::f16::dtype(), DType::F16);
        assert_eq!(half::bf16::dtype(), DType::BF16);
    }

    #[test]
    fn test_candle_roundtrip() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let candle_dt: candle_core::DType = dt.into();
            let back = DType::try_from(candle_dt).unwrap();
            assert_eq!(dt, back);
        }
    }

    #[test]
    fn test_promote() {
        assert_eq!(promote(DType::F16, DType::F32), DType::F32);
        assert_eq!(promote(DType::BF16, DType::F16), DType::BF16);
        assert_eq!(promote(DType::F16, DType::F16), DType::F16);
        assert_eq!(promote(DType::BF16, DType::F32), DType::F32);
    }

    #[test]
    fn test_candle_unsupported_dtype() {
        let result = DType::try_from(candle_core::DType::U8);
        assert!(result.is_err());
    }
}
