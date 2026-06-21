use crate::core::error::CoreError;
use candle_core;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),  // ordinal — which GPU, 0-indexed
    Metal(usize), // ordinal — which GPU, 0-indexed (Apple Silicon typically has one)
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        // Use the wildcard '_' to match ANY ordinal number
        matches!(self, Device::Cuda(_))
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, Device::Metal(_))
    }

    pub fn cuda_ordinal(&self) -> Option<usize> {
        if let Self::Cuda(id) = self {
            Some(*id)
        } else {
            None
        }
    }

    pub fn metal_ordinal(&self) -> Option<usize> {
        if let Self::Metal(id) = self {
            Some(*id)
        } else {
            None
        }
    }

    pub fn cuda(ordinal: usize) -> Self {
        Self::Cuda(ordinal)
    }

    pub fn metal(ordinal: usize) -> Self {
        Self::Metal(ordinal)
    }

    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(n) => format!("cuda:{}", n),
            Device::Metal(n) => format!("metal:{}", n),
        }
    }

    // validate two tensors are on the same device before an op
    pub fn check_same(
        op: &'static str,
        lhs: &Device,
        rhs: &Device,
    ) -> crate::core::error::Result<()> {
        if lhs != rhs {
            return Err(CoreError::DeviceMismatch {
                op,
                lhs: lhs.name(),
                rhs: rhs.name(),
            });
        }
        Ok(())
    }
}

impl TryFrom<candle_core::Device> for Device {
    type Error = crate::core::error::CoreError;
    fn try_from(device: candle_core::Device) -> std::result::Result<Self, Self::Error> {
        match device {
            candle_core::Device::Cpu => Ok(Device::Cpu),
            candle_core::Device::Cuda(_d) => Ok(Device::Cuda(0)),
            // NOTE: axiom's MetalTensor/MetalBackend do not route through candle_core at all —
            // this arm only matters if a CandleTensor is ever constructed on Candle's own Metal
            // backend, which axiom does not currently do. Kept permissive rather than erroring
            // since there's no longer a structural reason to reject it.
            candle_core::Device::Metal(_) => Ok(Device::Metal(0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cpu() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cuda(0).is_cpu());
    }

    #[test]
    fn test_is_cuda() {
        assert!(Device::Cuda(0).is_cuda());
        assert!(Device::Cuda(1).is_cuda());
        assert!(!Device::Cpu.is_cuda());
        assert!(!Device::Metal(0).is_cuda());
    }

    #[test]
    fn test_is_metal() {
        assert!(Device::Metal(0).is_metal());
        assert!(Device::Metal(1).is_metal());
        assert!(!Device::Cpu.is_metal());
        assert!(!Device::Cuda(0).is_metal());
    }

    #[test]
    fn test_cuda_ordinal() {
        assert_eq!(Device::Cuda(0).cuda_ordinal(), Some(0));
        assert_eq!(Device::Cuda(2).cuda_ordinal(), Some(2));
        assert_eq!(Device::Cpu.cuda_ordinal(), None);
    }

    #[test]
    fn test_metal_ordinal() {
        assert_eq!(Device::Metal(0).metal_ordinal(), Some(0));
        assert_eq!(Device::Metal(2).metal_ordinal(), Some(2));
        assert_eq!(Device::Cpu.metal_ordinal(), None);
    }

    #[test]
    fn test_cuda_constructor() {
        assert_eq!(Device::cuda(0), Device::Cuda(0));
        assert_eq!(Device::cuda(3), Device::Cuda(3));
    }

    #[test]
    fn test_metal_constructor() {
        assert_eq!(Device::metal(0), Device::Metal(0));
        assert_eq!(Device::metal(3), Device::Metal(3));
    }

    #[test]
    fn test_name() {
        assert_eq!(Device::Cpu.name(), "cpu");
        assert_eq!(Device::Cuda(0).name(), "cuda:0");
        assert_eq!(Device::Cuda(2).name(), "cuda:2");
        assert_eq!(Device::Metal(0).name(), "metal:0");
    }

    #[test]
    fn test_check_same_ok() {
        assert!(Device::check_same("matmul", &Device::Cpu, &Device::Cpu).is_ok());
        assert!(Device::check_same("matmul", &Device::Cuda(0), &Device::Cuda(0)).is_ok());
    }

    #[test]
    fn test_check_same_different_devices() {
        assert!(Device::check_same("matmul", &Device::Cpu, &Device::Cuda(0)).is_err());
    }

    #[test]
    fn test_check_same_different_ordinals() {
        assert!(Device::check_same("matmul", &Device::Cuda(0), &Device::Cuda(1)).is_err());
    }

    #[test]
    fn test_check_same_cuda_metal_mismatch() {
        assert!(Device::check_same("matmul", &Device::Cuda(0), &Device::Metal(0)).is_err());
    }

    #[test]
    fn test_candle_cpu_conversion() {
        let candle_dev = candle_core::Device::Cpu;
        let dev = Device::try_from(candle_dev).unwrap();
        assert_eq!(dev, Device::Cpu);
    }

    #[test]
    fn test_candle_metal_conversion() {
        // axiom's MetalTensor does not route through candle_core, but the
        // conversion should not hard-error if a CandleTensor on Candle's own
        // Metal backend is ever produced (e.g. via an external loader).
        if let Ok(candle_dev) = candle_core::Device::new_metal(0) {
            let dev = Device::try_from(candle_dev).unwrap();
            assert!(dev.is_metal());
        }
    }
}
