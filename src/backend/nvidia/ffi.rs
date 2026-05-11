//! FFI bindings to the NVIDIA CUDA GPU hashing library.
//!
//! Barebones — mirrors the Intel FFI structure. The actual CUDA kernels
//! in `gpu_hash_cuda.cu` are stubs that return `CUDA_HASH_ERROR_NO_DEVICE`.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaHashAlgorithm {
    Sha256 = 0,
    Sha384 = 1,
    Sha512 = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaHashError {
    Success = 0,
    NoDevice = 1,
    InvalidInput = 5,
    Unknown = 99,
}

pub type CudaHashContextHandle = *mut c_void;

#[link(name = "gpu_hash_nvidia")]
unsafe extern "C" {
    pub fn cuda_hash_init() -> CudaHashError;
    pub fn cuda_hash_cleanup();
    pub fn cuda_hash_is_available() -> c_int;
    pub fn cuda_hash_get_device_count() -> c_int;
    pub fn cuda_hash_create_context(
        algorithm: CudaHashAlgorithm,
        device_index: c_int,
        handle: *mut CudaHashContextHandle,
    ) -> CudaHashError;
    pub fn cuda_hash_destroy_context(handle: CudaHashContextHandle);
    pub fn cuda_hash_single(
        handle: CudaHashContextHandle,
        input: *const u8,
        input_len: usize,
        output: *mut u8,
        output_len: *mut usize,
    ) -> CudaHashError;
    pub fn cuda_hash_batch(
        handle: CudaHashContextHandle,
        inputs: *const *const u8,
        input_lens: *const usize,
        num_inputs: usize,
        outputs: *mut *mut u8,
        output_size: usize,
    ) -> CudaHashError;
    pub fn cuda_hash_output_size(algorithm: CudaHashAlgorithm) -> usize;
    pub fn cuda_hash_get_last_error() -> *const c_char;
}

/// Safe wrapper to get the last error message.
pub fn get_last_error() -> String {
    unsafe {
        let ptr = cuda_hash_get_last_error();
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}
