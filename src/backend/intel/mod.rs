//! # Intel Xe GPU Backend (SYCL/oneAPI)
//!
//! This backend provides GPU-accelerated hashing on Intel Xe GPUs using
//! SYCL/oneAPI. It wraps the C API from `gpu_hash.h` via FFI.
//!
//! ## Prerequisites
//!
//! - Intel oneAPI Base Toolkit
//! - Intel GPU drivers with Level Zero support
//! - `source /opt/intel/oneapi/setvars.sh` before building

mod ffi;

use super::{BackendInfo, GpuBackend, HashAlgorithm, HashOutput};
use crate::error::{HashError, HashResult};

/// Intel SYCL GPU backend.
pub struct IntelBackend {
    algorithm: HashAlgorithm,
    handle: ffi::GpuHashContextHandle,
    output_size: usize,
    gpu_available: bool,
}

impl IntelBackend {
    /// Create a new Intel backend for the specified algorithm.
    pub fn new(algorithm: HashAlgorithm) -> HashResult<Self> {
        let err = unsafe { ffi::gpu_hash_init() };
        check_ffi_error(err)?;

        let gpu_available = unsafe { ffi::gpu_hash_is_available() != 0 };

        let ffi_alg = to_ffi_algorithm(algorithm);
        let mut handle: ffi::GpuHashContextHandle = std::ptr::null_mut();

        let err = unsafe { ffi::gpu_hash_create_context(ffi_alg, -1, &mut handle) };
        check_ffi_error(err)?;

        Ok(Self {
            algorithm,
            handle,
            output_size: algorithm.output_size(),
            gpu_available,
        })
    }

    /// Probe for Intel GPU without creating a full context.
    pub fn probe() -> HashResult<BackendInfo> {
        let err = unsafe { ffi::gpu_hash_init() };
        check_ffi_error(err)?;

        let count = unsafe { ffi::gpu_hash_get_device_count() };
        let mut devices = Vec::new();

        for i in 0..count {
            let mut info = ffi::GpuDeviceInfo::default();
            let err = unsafe { ffi::gpu_hash_get_device_info(i, &mut info) };
            if check_ffi_error(err).is_ok() {
                devices.push(ffi::device_info_name(&info));
            }
        }

        Ok(BackendInfo {
            name: "intel".to_string(),
            vendor: "Intel (SYCL/oneAPI)".to_string(),
            available: !devices.is_empty(),
            device_count: devices.len(),
            devices,
        })
    }
}

impl GpuBackend for IntelBackend {
    fn name(&self) -> &str {
        "intel"
    }

    fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    fn hash(&self, data: &[u8]) -> HashResult<HashOutput> {
        let mut output = vec![0u8; self.output_size];
        let mut output_len = 0usize;

        let err = unsafe {
            ffi::gpu_hash_single(
                self.handle,
                data.as_ptr(),
                data.len(),
                output.as_mut_ptr(),
                &mut output_len,
            )
        };
        check_ffi_error(err)?;

        Ok(HashOutput::new(output, self.algorithm, "intel"))
    }

    fn hash_batch(&self, messages: &[Vec<u8>]) -> HashResult<Vec<HashOutput>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        let num = messages.len();
        let input_ptrs: Vec<*const u8> = messages.iter().map(|m| m.as_ptr()).collect();
        let input_lens: Vec<usize> = messages.iter().map(|m| m.len()).collect();

        let mut output_buffers: Vec<Vec<u8>> =
            (0..num).map(|_| vec![0u8; self.output_size]).collect();
        let mut output_ptrs: Vec<*mut u8> =
            output_buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();

        let err = unsafe {
            ffi::gpu_hash_batch(
                self.handle,
                input_ptrs.as_ptr(),
                input_lens.as_ptr(),
                num,
                output_ptrs.as_mut_ptr(),
                self.output_size,
            )
        };
        check_ffi_error(err)?;

        Ok(output_buffers
            .into_iter()
            .map(|b| HashOutput::new(b, self.algorithm, "intel"))
            .collect())
    }

    fn hash_batch_fixed(
        &self,
        data: &[u8],
        message_size: usize,
    ) -> HashResult<Vec<HashOutput>> {
        if message_size == 0 {
            return Err(HashError::InvalidInput("message_size cannot be zero".into()));
        }
        if data.len() % message_size != 0 {
            return Err(HashError::InvalidInput(format!(
                "Data length {} not divisible by message_size {message_size}",
                data.len()
            )));
        }

        let num = data.len() / message_size;
        if num == 0 {
            return Ok(Vec::new());
        }

        let mut output = vec![0u8; self.output_size * num];

        let err = unsafe {
            ffi::gpu_hash_batch_fixed(
                self.handle,
                data.as_ptr(),
                message_size,
                num,
                output.as_mut_ptr(),
            )
        };
        check_ffi_error(err)?;

        Ok(output
            .chunks(self.output_size)
            .map(|chunk| HashOutput::new(chunk.to_vec(), self.algorithm, "intel"))
            .collect())
    }

    fn is_gpu_accelerated(&self) -> bool {
        self.gpu_available
    }

    fn info(&self) -> BackendInfo {
        IntelBackend::probe().unwrap_or(BackendInfo {
            name: "intel".to_string(),
            vendor: "Intel (SYCL/oneAPI)".to_string(),
            available: false,
            device_count: 0,
            devices: vec![],
        })
    }
}

impl Drop for IntelBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::gpu_hash_destroy_context(self.handle);
            }
        }
    }
}

unsafe impl Send for IntelBackend {}
unsafe impl Sync for IntelBackend {}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn to_ffi_algorithm(alg: HashAlgorithm) -> ffi::GpuHashAlgorithm {
    match alg {
        HashAlgorithm::Sha256 => ffi::GpuHashAlgorithm::Sha256,
        HashAlgorithm::Sha384 => ffi::GpuHashAlgorithm::Sha384,
        HashAlgorithm::Sha512 => ffi::GpuHashAlgorithm::Sha512,
    }
}

fn check_ffi_error(err: ffi::GpuHashError) -> HashResult<()> {
    match err {
        ffi::GpuHashError::Success => Ok(()),
        ffi::GpuHashError::NoDevice => Err(HashError::NoDeviceFound(ffi::get_last_error())),
        ffi::GpuHashError::InvalidAlgorithm => {
            Err(HashError::InvalidAlgorithm(ffi::get_last_error()))
        }
        ffi::GpuHashError::MemoryAllocation => {
            Err(HashError::MemoryAllocation(ffi::get_last_error()))
        }
        ffi::GpuHashError::KernelExecution => {
            Err(HashError::KernelExecution(ffi::get_last_error()))
        }
        ffi::GpuHashError::InvalidInput => Err(HashError::InvalidInput(ffi::get_last_error())),
        ffi::GpuHashError::NotInitialized => Err(HashError::NotInitialized(ffi::get_last_error())),
        ffi::GpuHashError::Unknown => Err(HashError::Backend {
            backend: "intel".to_string(),
            message: ffi::get_last_error(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intel_probe() {
        // Should not panic even if no Intel GPU present
        let info = IntelBackend::probe();
        println!("Intel probe result: {:?}", info);
    }
}
