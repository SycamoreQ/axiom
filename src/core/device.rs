use crate::core::error::{CoreError, Result};
use candle_core;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize), // ordinal — which GPU, 0-indexed
}

impl Device {
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    pub fn is_cuda(&self) -> bool {
        // Use the wildcard '_' to match ANY ordinal number
        matches!(self, Device::Cuda(_))
    }

    pub fn cuda_ordinal(&self) -> Option<usize> {
        if let Self::Cuda(id) = self {
            Some(*id)
        } else {
            None
        }
    }

    pub fn cuda(ordinal: usize) -> Self {
        Self::Cuda(ordinal)
    }

    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(n) => format!("cuda:{}", n),
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
            candle_core::Device::Cuda(d) => Ok(Device::Cuda(0)),
            candle_core::Device::Metal(_) => Err(CoreError::Internal(
                "Metal device not supported".to_string(),
            )),
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
    }

    #[test]
    fn test_cuda_ordinal() {
        assert_eq!(Device::Cuda(0).cuda_ordinal(), Some(0));
        assert_eq!(Device::Cuda(2).cuda_ordinal(), Some(2));
        assert_eq!(Device::Cpu.cuda_ordinal(), None);
    }

    #[test]
    fn test_cuda_constructor() {
        assert_eq!(Device::cuda(0), Device::Cuda(0));
        assert_eq!(Device::cuda(3), Device::Cuda(3));
    }

    #[test]
    fn test_name() {
        assert_eq!(Device::Cpu.name(), "cpu");
        assert_eq!(Device::Cuda(0).name(), "cuda:0");
        assert_eq!(Device::Cuda(2).name(), "cuda:2");
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
    fn test_candle_cpu_conversion() {
        let candle_dev = candle_core::Device::Cpu;
        let dev = Device::try_from(candle_dev).unwrap();
        assert_eq!(dev, Device::Cpu);
    }
}
