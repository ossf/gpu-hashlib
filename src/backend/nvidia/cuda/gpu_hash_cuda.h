/**
 * @file gpu_hash_cuda.h
 * @brief C API for CUDA-based GPU hashing on NVIDIA GPUs
 *
 * Barebones header — mirrors the Intel SYCL API structure.
 * TODO: Implement actual CUDA kernels for SHA-256/384/512.
 *
 * Compile with: nvcc -shared -Xcompiler -fPIC -o libgpu_hash_nvidia.so gpu_hash_cuda.cu
 */

#ifndef GPU_HASH_CUDA_H
#define GPU_HASH_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUDA_HASH_SHA256 = 0,
    CUDA_HASH_SHA384 = 1,
    CUDA_HASH_SHA512 = 2
} CudaHashAlgorithm;

typedef enum {
    CUDA_HASH_SUCCESS = 0,
    CUDA_HASH_ERROR_NO_DEVICE = 1,
    CUDA_HASH_ERROR_INVALID_INPUT = 5,
    CUDA_HASH_ERROR_UNKNOWN = 99
} CudaHashError;

typedef void* CudaHashContextHandle;

/**
 * Initialize the CUDA hashing library.
 * Enumerates NVIDIA GPUs.
 */
CudaHashError cuda_hash_init(void);

/**
 * Cleanup CUDA resources.
 */
void cuda_hash_cleanup(void);

/**
 * Check if CUDA GPU hashing is available.
 * @return 1 if available, 0 otherwise
 */
int cuda_hash_is_available(void);

/**
 * Get number of CUDA-capable GPUs.
 */
int cuda_hash_get_device_count(void);

/**
 * Create a hashing context for a specific algorithm and device.
 *
 * @param algorithm Hash algorithm to use
 * @param device_index GPU index (-1 for auto-select)
 * @param handle Output: opaque context handle
 */
CudaHashError cuda_hash_create_context(CudaHashAlgorithm algorithm,
                                       int device_index,
                                       CudaHashContextHandle* handle);

/**
 * Destroy a hashing context.
 */
void cuda_hash_destroy_context(CudaHashContextHandle handle);

/**
 * Hash a single message.
 */
CudaHashError cuda_hash_single(CudaHashContextHandle handle,
                               const uint8_t* input, size_t input_len,
                               uint8_t* output, size_t* output_len);

/**
 * Hash multiple messages in parallel (batch).
 */
CudaHashError cuda_hash_batch(CudaHashContextHandle handle,
                              const uint8_t** inputs, const size_t* input_lens,
                              size_t num_inputs,
                              uint8_t** outputs, size_t output_size);

/**
 * Get output size for a hash algorithm.
 */
size_t cuda_hash_output_size(CudaHashAlgorithm algorithm);

/**
 * Get the last error message.
 */
const char* cuda_hash_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_HASH_CUDA_H */
