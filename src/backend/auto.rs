//! Automatic backend selection.
//!
//! `AutoBackend` probes available GPUs at runtime and selects the best one.
//! Priority order: Intel GPU → NVIDIA GPU → CPU fallback.

use super::cpu::CpuBackend;
use super::{BackendInfo, GpuBackend, HashAlgorithm, HashOutput};
use crate::error::HashResult;

/// Automatically selects the best available backend.
///
/// Selection priority:
/// 1. Intel Xe GPU (if `intel` feature enabled and GPU present)
/// 2. NVIDIA GPU (if `nvidia` feature enabled and GPU present)
/// 3. CPU fallback (always available)
pub struct AutoBackend {
    inner: Box<dyn GpuBackend>,
}

impl AutoBackend {
    /// Create a new AutoBackend for the given algorithm.
    ///
    /// Probes available hardware and selects the best backend.
    pub fn new(algorithm: HashAlgorithm) -> HashResult<Self> {
        let inner = Self::select_backend(algorithm)?;
        Ok(Self { inner })
    }

    /// Force a specific backend by name.
    ///
    /// Valid names: "intel", "nvidia", "cpu"
    pub fn with_backend(algorithm: HashAlgorithm, backend_name: &str) -> HashResult<Self> {
        let inner: Box<dyn GpuBackend> = match backend_name {
            #[cfg(feature = "intel")]
            "intel" => Box::new(super::intel::IntelBackend::new(algorithm)?),

            #[cfg(feature = "nvidia")]
            "nvidia" => Box::new(super::nvidia::NvidiaBackend::new(algorithm)?),

            "cpu" => Box::new(CpuBackend::new(algorithm)),

            _ => {
                return Err(crate::error::HashError::Other(format!(
                    "Unknown backend: {backend_name}. Available: cpu{}{}",
                    if cfg!(feature = "intel") { ", intel" } else { "" },
                    if cfg!(feature = "nvidia") { ", nvidia" } else { "" },
                )));
            }
        };
        Ok(Self { inner })
    }

    fn select_backend(algorithm: HashAlgorithm) -> HashResult<Box<dyn GpuBackend>> {
        // Try Intel first
        #[cfg(feature = "intel")]
        {
            match super::intel::IntelBackend::new(algorithm) {
                Ok(backend) if backend.is_gpu_accelerated() => {
                    log::info!("Selected Intel GPU backend");
                    return Ok(Box::new(backend));
                }
                Ok(_) => log::debug!("Intel backend available but not GPU-accelerated"),
                Err(e) => log::debug!("Intel backend not available: {e}"),
            }
        }

        // Try NVIDIA
        #[cfg(feature = "nvidia")]
        {
            match super::nvidia::NvidiaBackend::new(algorithm) {
                Ok(backend) if backend.is_gpu_accelerated() => {
                    log::info!("Selected NVIDIA GPU backend");
                    return Ok(Box::new(backend));
                }
                Ok(_) => log::debug!("NVIDIA backend available but not GPU-accelerated"),
                Err(e) => log::debug!("NVIDIA backend not available: {e}"),
            }
        }

        // Fallback to CPU
        log::info!("Using CPU fallback backend");
        Ok(Box::new(CpuBackend::new(algorithm)))
    }

    /// Get the name of the selected backend.
    pub fn selected_backend(&self) -> &str {
        self.inner.name()
    }
}

impl GpuBackend for AutoBackend {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn algorithm(&self) -> HashAlgorithm {
        self.inner.algorithm()
    }

    fn hash(&self, data: &[u8]) -> HashResult<HashOutput> {
        self.inner.hash(data)
    }

    fn hash_batch(&self, messages: &[Vec<u8>]) -> HashResult<Vec<HashOutput>> {
        self.inner.hash_batch(messages)
    }

    fn hash_batch_fixed(
        &self,
        data: &[u8],
        message_size: usize,
    ) -> HashResult<Vec<HashOutput>> {
        self.inner.hash_batch_fixed(data, message_size)
    }

    fn is_gpu_accelerated(&self) -> bool {
        self.inner.is_gpu_accelerated()
    }

    fn info(&self) -> BackendInfo {
        self.inner.info()
    }
}
