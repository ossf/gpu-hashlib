//! # NVIDIA GPU Backend (CUDA)
//!
//! This backend provides GPU-accelerated hashing on NVIDIA GPUs using CUDA.
//!
//! ## Status
//!
//! **Barebones / Unpopulated** — the trait interface is fully wired up but
//! the CUDA kernels are stubs. The Intel SYCL backend serves as the reference
//! implementation; this backend mirrors its architecture.
//!
//! ## Prerequisites
//!
//! - NVIDIA GPU (Compute Capability ≥ 6.0 recommended)
//! - CUDA Toolkit ≥ 11.0
//! - NVIDIA drivers
//!
//! ## Building
//!
//! ```bash
//! cargo build --features nvidia
//! ```

mod ffi;

use super::{BackendInfo, GpuBackend, HashAlgorithm, HashOutput};
use crate::error::{HashError, HashResult};

/// NVIDIA CUDA GPU backend.
///
/// Currently a barebones implementation with stubbed CUDA kernels.
/// Falls back to CPU hashing for all operations until CUDA kernels
/// are implemented.
pub struct NvidiaBackend {
    algorithm: HashAlgorithm,
    handle: ffi::CudaHashContextHandle,
    output_size: usize,
    gpu_available: bool,
}

impl NvidiaBackend {
    /// Create a new NVIDIA backend for the specified algorithm.
    pub fn new(algorithm: HashAlgorithm) -> HashResult<Self> {
        let err = unsafe { ffi::cuda_hash_init() };
        check_ffi_error(err)?;

        let gpu_available = unsafe { ffi::cuda_hash_is_available() != 0 };

        let ffi_alg = to_ffi_algorithm(algorithm);
        let mut handle: ffi::CudaHashContextHandle = std::ptr::null_mut();

        let err = unsafe { ffi::cuda_hash_create_context(ffi_alg, -1, &mut handle) };
        // Allow NoDevice — we'll fall back to CPU
        if err != ffi::CudaHashError::Success && err != ffi::CudaHashError::NoDevice {
            check_ffi_error(err)?;
        }

        Ok(Self {
            algorithm,
            handle,
            output_size: algorithm.output_size(),
            gpu_available,
        })
    }

    /// Probe for NVIDIA GPU without creating a full context.
    pub fn probe() -> HashResult<BackendInfo> {
        let err = unsafe { ffi::cuda_hash_init() };
        check_ffi_error(err)?;

        let count = unsafe { ffi::cuda_hash_get_device_count() };

        // TODO: Implement device enumeration via CUDA
        let devices: Vec<String> = (0..count)
            .map(|i| format!("NVIDIA GPU {i}"))
            .collect();

        Ok(BackendInfo {
            name: "nvidia".to_string(),
            vendor: "NVIDIA (CUDA)".to_string(),
            available: count > 0,
            device_count: count as usize,
            devices,
        })
    }

    /// CPU fallback hash (used when CUDA kernels are not yet implemented).
    fn cpu_fallback_hash(&self, data: &[u8]) -> HashResult<HashOutput> {
        use ring::digest;

        let alg = match self.algorithm {
            HashAlgorithm::Sha256 => &digest::SHA256,
            HashAlgorithm::Sha384 => &digest::SHA384,
            HashAlgorithm::Sha512 => &digest::SHA512,
        };

        let result = digest::digest(alg, data);
        Ok(HashOutput::new(
            result.as_ref().to_vec(),
            self.algorithm,
            "nvidia-cpu-fallback",
        ))
    }
}

impl GpuBackend for NvidiaBackend {
    fn name(&self) -> &str {
        if self.gpu_available {
            "nvidia"
        } else {
            "nvidia-cpu-fallback"
        }
    }

    fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    fn hash(&self, data: &[u8]) -> HashResult<HashOutput> {
        if !self.gpu_available || self.handle.is_null() {
            return self.cpu_fallback_hash(data);
        }

        let mut output = vec![0u8; self.output_size];
        let mut output_len = 0usize;

        let err = unsafe {
            ffi::cuda_hash_single(
                self.handle,
                data.as_ptr(),
                data.len(),
                output.as_mut_ptr(),
                &mut output_len,
            )
        };

        match err {
            ffi::CudaHashError::Success => {
                Ok(HashOutput::new(output, self.algorithm, "nvidia"))
            }
            // Fall back to CPU on any GPU error
            _ => {
                log::warn!("NVIDIA GPU hash failed, falling back to CPU: {:?}", err);
                self.cpu_fallback_hash(data)
            }
        }
    }

    fn hash_batch(&self, messages: &[Vec<u8>]) -> HashResult<Vec<HashOutput>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        if !self.gpu_available || self.handle.is_null() {
            return messages.iter().map(|m| self.cpu_fallback_hash(m)).collect();
        }

        let num = messages.len();
        let input_ptrs: Vec<*const u8> = messages.iter().map(|m| m.as_ptr()).collect();
        let input_lens: Vec<usize> = messages.iter().map(|m| m.len()).collect();

        let mut output_buffers: Vec<Vec<u8>> =
            (0..num).map(|_| vec![0u8; self.output_size]).collect();
        let mut output_ptrs: Vec<*mut u8> =
            output_buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();

        let err = unsafe {
            ffi::cuda_hash_batch(
                self.handle,
                input_ptrs.as_ptr(),
                input_lens.as_ptr(),
                num,
                output_ptrs.as_mut_ptr(),
                self.output_size,
            )
        };

        match err {
            ffi::CudaHashError::Success => Ok(output_buffers
                .into_iter()
                .map(|b| HashOutput::new(b, self.algorithm, "nvidia"))
                .collect()),
            _ => {
                log::warn!("NVIDIA GPU batch hash failed, falling back to CPU");
                messages.iter().map(|m| self.cpu_fallback_hash(m)).collect()
            }
        }
    }

    fn is_gpu_accelerated(&self) -> bool {
        self.gpu_available
    }

    fn info(&self) -> BackendInfo {
        NvidiaBackend::probe().unwrap_or(BackendInfo {
            name: "nvidia".to_string(),
            vendor: "NVIDIA (CUDA)".to_string(),
            available: false,
            device_count: 0,
            devices: vec![],
        })
    }
}

impl Drop for NvidiaBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::cuda_hash_destroy_context(self.handle);
            }
        }
    }
}

unsafe impl Send for NvidiaBackend {}
unsafe impl Sync for NvidiaBackend {}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn to_ffi_algorithm(alg: HashAlgorithm) -> ffi::CudaHashAlgorithm {
    match alg {
        HashAlgorithm::Sha256 => ffi::CudaHashAlgorithm::Sha256,
        HashAlgorithm::Sha384 => ffi::CudaHashAlgorithm::Sha384,
        HashAlgorithm::Sha512 => ffi::CudaHashAlgorithm::Sha512,
    }
}

fn check_ffi_error(err: ffi::CudaHashError) -> HashResult<()> {
    match err {
        ffi::CudaHashError::Success => Ok(()),
        ffi::CudaHashError::NoDevice => {
            Err(HashError::NoDeviceFound(ffi::get_last_error()))
        }
        ffi::CudaHashError::InvalidInput => {
            Err(HashError::InvalidInput(ffi::get_last_error()))
        }
        ffi::CudaHashError::Unknown => Err(HashError::Backend {
            backend: "nvidia".to_string(),
            message: ffi::get_last_error(),
        }),
    }
}
