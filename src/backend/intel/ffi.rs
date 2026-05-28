//! FFI bindings to the Intel SYCL GPU hashing library.
//!
//! Low-level bindings to the C API defined in gpu_hash.h.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHashAlgorithm {
    Sha256 = 0,
    Sha384 = 1,
    Sha512 = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuHashError {
    Success = 0,
    NoDevice = 1,
    InvalidAlgorithm = 2,
    MemoryAllocation = 3,
    KernelExecution = 4,
    InvalidInput = 5,
    NotInitialized = 6,
    Unknown = 99,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    Gpu = 0,
    Cpu = 1,
    Accelerator = 2,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: [c_char; 256],
    pub vendor: [c_char; 256],
    pub driver_version: [c_char; 64],
    pub device_type: GpuDeviceType,
    pub max_compute_units: u32,
    pub global_memory_size: u64,
    pub local_memory_size: u64,
    pub max_work_group_size: usize,
    pub is_intel: c_int,
    pub is_intel_xe: c_int,
}

impl Default for GpuDeviceInfo {
    fn default() -> Self {
        Self {
            name: [0; 256],
            vendor: [0; 256],
            driver_version: [0; 64],
            device_type: GpuDeviceType::Gpu,
            max_compute_units: 0,
            global_memory_size: 0,
            local_memory_size: 0,
            max_work_group_size: 0,
            is_intel: 0,
            is_intel_xe: 0,
        }
    }
}

pub type GpuHashContextHandle = *mut c_void;

#[link(name = "gpu_hash_intel")]
unsafe extern "C" {
    pub fn gpu_hash_init() -> GpuHashError;
    pub fn gpu_hash_cleanup();
    pub fn gpu_hash_is_available() -> c_int;
    pub fn gpu_hash_get_device_count() -> c_int;
    pub fn gpu_hash_get_device_info(device_index: c_int, info: *mut GpuDeviceInfo) -> GpuHashError;
    pub fn gpu_hash_create_context(
        algorithm: GpuHashAlgorithm,
        device_index: c_int,
        handle: *mut GpuHashContextHandle,
    ) -> GpuHashError;
    pub fn gpu_hash_destroy_context(handle: GpuHashContextHandle);
    pub fn gpu_hash_single(
        handle: GpuHashContextHandle,
        input: *const u8,
        input_len: usize,
        output: *mut u8,
        output_len: *mut usize,
    ) -> GpuHashError;
    pub fn gpu_hash_batch(
        handle: GpuHashContextHandle,
        inputs: *const *const u8,
        input_lens: *const usize,
        num_inputs: usize,
        outputs: *mut *mut u8,
        output_size: usize,
    ) -> GpuHashError;
    pub fn gpu_hash_batch_fixed(
        handle: GpuHashContextHandle,
        input: *const u8,
        message_size: usize,
        num_messages: usize,
        output: *mut u8,
    ) -> GpuHashError;
    pub fn gpu_hash_output_size(algorithm: GpuHashAlgorithm) -> usize;
    pub fn gpu_hash_get_last_error() -> *const c_char;
}

/// Safe wrapper to get the last error message.
pub fn get_last_error() -> String {
    unsafe {
        let ptr = gpu_hash_get_last_error();
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}

/// Extract device name from GpuDeviceInfo.
pub fn device_info_name(info: &GpuDeviceInfo) -> String {
    unsafe {
        std::ffi::CStr::from_ptr(info.name.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}
