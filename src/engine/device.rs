use std::fmt;
use std::str::FromStr;

use crate::model_store::registry::ModelFormat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceTarget {
    Auto,
    Cpu,
    Metal,
    Cuda(i32),
    Vulkan(i32),
    Mlx,
}

impl Default for DeviceTarget {
    fn default() -> Self {
        Self::Auto
    }
}

impl DeviceTarget {
    pub fn validate_for_format(&self, format: &ModelFormat) -> anyhow::Result<()> {
        match (self, format) {
            (Self::Cuda(_) | Self::Vulkan(_), ModelFormat::Mlx) => {
                anyhow::bail!("cannot use {} device with MLX model format", self);
            }
            (Self::Mlx, ModelFormat::Gguf) => {
                anyhow::bail!("MLX device requires an MLX model, got GGUF");
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            (Self::Mlx, _) => {
                anyhow::bail!("MLX device is only available on Apple Silicon");
            }
            _ => Ok(()),
        }
    }

    pub fn requires_llamacpp(&self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Vulkan(_) | Self::Cpu)
    }

    pub fn main_gpu(&self) -> Option<i32> {
        match self {
            Self::Cuda(id) => Some(*id),
            Self::Vulkan(id) => Some(*id),
            _ => None,
        }
    }

    pub fn force_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }
}

impl fmt::Display for DeviceTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Metal => write!(f, "metal"),
            Self::Cuda(id) => write!(f, "cuda:{id}"),
            Self::Vulkan(id) => write!(f, "vulkan:{id}"),
            Self::Mlx => write!(f, "mlx"),
        }
    }
}

impl FromStr for DeviceTarget {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_lowercase();
        match s.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "mlx" => Ok(Self::Mlx),
            other => {
                if let Some(id) = other.strip_prefix("cuda:") {
                    let id: i32 = id
                        .parse()
                        .map_err(|_| anyhow::anyhow!("invalid CUDA device index: {id}"))?;
                    anyhow::ensure!(id >= 0, "CUDA device index must be non-negative");
                    Ok(Self::Cuda(id))
                } else if let Some(id) = other.strip_prefix("vulkan:") {
                    let id: i32 = id
                        .parse()
                        .map_err(|_| anyhow::anyhow!("invalid Vulkan device index: {id}"))?;
                    anyhow::ensure!(id >= 0, "Vulkan device index must be non-negative");
                    Ok(Self::Vulkan(id))
                } else {
                    anyhow::bail!(
                        "unknown device: {other} (expected auto, cpu, metal, mlx, cuda:N, or vulkan:N)"
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_auto() {
        assert_eq!("auto".parse::<DeviceTarget>().unwrap(), DeviceTarget::Auto);
        assert_eq!("".parse::<DeviceTarget>().unwrap(), DeviceTarget::Auto);
    }

    #[test]
    fn parse_cpu() {
        assert_eq!("cpu".parse::<DeviceTarget>().unwrap(), DeviceTarget::Cpu);
    }

    #[test]
    fn parse_metal() {
        assert_eq!("metal".parse::<DeviceTarget>().unwrap(), DeviceTarget::Metal);
    }

    #[test]
    fn parse_mlx() {
        assert_eq!("mlx".parse::<DeviceTarget>().unwrap(), DeviceTarget::Mlx);
    }

    #[test]
    fn parse_cuda() {
        assert_eq!("cuda:0".parse::<DeviceTarget>().unwrap(), DeviceTarget::Cuda(0));
        assert_eq!("cuda:1".parse::<DeviceTarget>().unwrap(), DeviceTarget::Cuda(1));
        assert_eq!("CUDA:2".parse::<DeviceTarget>().unwrap(), DeviceTarget::Cuda(2));
    }

    #[test]
    fn parse_vulkan() {
        assert_eq!("vulkan:0".parse::<DeviceTarget>().unwrap(), DeviceTarget::Vulkan(0));
        assert_eq!("vulkan:1".parse::<DeviceTarget>().unwrap(), DeviceTarget::Vulkan(1));
    }

    #[test]
    fn parse_invalid() {
        assert!("gpu:0".parse::<DeviceTarget>().is_err());
        assert!("cuda:".parse::<DeviceTarget>().is_err());
        assert!("cuda:-1".parse::<DeviceTarget>().is_err());
        assert!("cuda:abc".parse::<DeviceTarget>().is_err());
    }

    #[test]
    fn display_roundtrip() {
        for device in [
            DeviceTarget::Auto,
            DeviceTarget::Cpu,
            DeviceTarget::Metal,
            DeviceTarget::Cuda(0),
            DeviceTarget::Cuda(3),
            DeviceTarget::Vulkan(1),
            DeviceTarget::Mlx,
        ] {
            let s = device.to_string();
            let parsed: DeviceTarget = s.parse().unwrap();
            assert_eq!(parsed, device);
        }
    }

    #[test]
    fn validate_cuda_rejects_mlx_format() {
        let d = DeviceTarget::Cuda(0);
        assert!(d.validate_for_format(&ModelFormat::Mlx).is_err());
        assert!(d.validate_for_format(&ModelFormat::Gguf).is_ok());
    }

    #[test]
    fn validate_mlx_rejects_gguf() {
        let d = DeviceTarget::Mlx;
        assert!(d.validate_for_format(&ModelFormat::Gguf).is_err());
    }

    #[test]
    fn validate_auto_accepts_both() {
        let d = DeviceTarget::Auto;
        assert!(d.validate_for_format(&ModelFormat::Gguf).is_ok());
        assert!(d.validate_for_format(&ModelFormat::Mlx).is_ok());
    }

    #[test]
    fn main_gpu_extraction() {
        assert_eq!(DeviceTarget::Cuda(2).main_gpu(), Some(2));
        assert_eq!(DeviceTarget::Vulkan(1).main_gpu(), Some(1));
        assert_eq!(DeviceTarget::Auto.main_gpu(), None);
        assert_eq!(DeviceTarget::Metal.main_gpu(), None);
    }

    #[test]
    fn default_is_auto() {
        assert_eq!(DeviceTarget::default(), DeviceTarget::Auto);
    }
}
