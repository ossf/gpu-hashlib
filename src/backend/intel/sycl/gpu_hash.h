/**
 * @file gpu_hash.h
 * @brief C API for SYCL-based GPU hashing on Intel Xe GPUs
 *
 * Provides SHA-256, SHA-384, and SHA-512 hashing accelerated by
 * Intel Xe GPUs via SYCL/oneAPI. Falls back to CPU when no GPU
 * is available.
 *
 * Build:
 *   icpx -fsycl -fPIC -O3 -std=c++17 -shared gpu_hash.cpp -o libgpu_hash_intel.so
 *
 * Prerequisites:
 *   - Intel oneAPI Base Toolkit (icpx compiler)
 *   - Intel GPU with Level Zero driver support
 *   - source /opt/intel/oneapi/setvars.sh
 */

#ifndef GPU_HASH_H
#define GPU_HASH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────────────
 * Enums
 * ────────────────────────────────────────────────────────────────────────── */

typedef enum {
    GPU_HASH_SHA256 = 0,
    GPU_HASH_SHA384 = 1,
    GPU_HASH_SHA512 = 2
} GpuHashAlgorithm;

typedef enum {
    GPU_HASH_SUCCESS              = 0,
    GPU_HASH_ERROR_NO_DEVICE      = 1,
    GPU_HASH_ERROR_INVALID_ALGORITHM = 2,
    GPU_HASH_ERROR_MEMORY_ALLOCATION = 3,
    GPU_HASH_ERROR_KERNEL_EXECUTION  = 4,
    GPU_HASH_ERROR_INVALID_INPUT     = 5,
    GPU_HASH_ERROR_NOT_INITIALIZED   = 6,
    GPU_HASH_ERROR_UNKNOWN           = 99
} GpuHashError;

typedef enum {
    GPU_DEVICE_TYPE_GPU         = 0,
    GPU_DEVICE_TYPE_CPU         = 1,
    GPU_DEVICE_TYPE_ACCELERATOR = 2
} GpuDeviceType;

/* ──────────────────────────────────────────────────────────────────────────
 * Device Info
 * ────────────────────────────────────────────────────────────────────────── */

typedef struct {
    char          name[256];
    char          vendor[256];
    char          driver_version[64];
    GpuDeviceType device_type;
    uint32_t      max_compute_units;
    uint64_t      global_memory_size;
    uint64_t      local_memory_size;
    size_t        max_work_group_size;
    int           is_intel;
    int           is_intel_xe;
} GpuDeviceInfo;

/* ──────────────────────────────────────────────────────────────────────────
 * Opaque context handle
 * ────────────────────────────────────────────────────────────────────────── */

typedef void* GpuHashContextHandle;

/* ──────────────────────────────────────────────────────────────────────────
 * Lifecycle
 * ────────────────────────────────────────────────────────────────────────── */

/** Initialize the GPU hashing library. Call once at startup. */
GpuHashError gpu_hash_init(void);

/** Release global resources. */
void gpu_hash_cleanup(void);

/* ──────────────────────────────────────────────────────────────────────────
 * Device discovery
 * ────────────────────────────────────────────────────────────────────────── */

/** Returns 1 if at least one GPU is available, 0 otherwise. */
int gpu_hash_is_available(void);

/** Number of discovered GPU devices. */
int gpu_hash_get_device_count(void);

/** Populate info for device at index. */
GpuHashError gpu_hash_get_device_info(int device_index, GpuDeviceInfo* info);

/* ──────────────────────────────────────────────────────────────────────────
 * Context management
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * Create a hashing context.
 * @param algorithm   Hash algorithm
 * @param device_index  GPU index, or -1 for auto-select (first Intel GPU)
 * @param handle      Output handle
 */
GpuHashError gpu_hash_create_context(GpuHashAlgorithm algorithm,
                                     int device_index,
                                     GpuHashContextHandle* handle);

/** Destroy a context and free resources. */
void gpu_hash_destroy_context(GpuHashContextHandle handle);

/* ──────────────────────────────────────────────────────────────────────────
 * Hashing
 * ────────────────────────────────────────────────────────────────────────── */

/** Hash a single message. */
GpuHashError gpu_hash_single(GpuHashContextHandle handle,
                             const uint8_t* input, size_t input_len,
                             uint8_t* output, size_t* output_len);

/** Hash multiple variable-length messages in parallel. */
GpuHashError gpu_hash_batch(GpuHashContextHandle handle,
                            const uint8_t** inputs, const size_t* input_lens,
                            size_t num_inputs,
                            uint8_t** outputs, size_t output_size);

/** Hash multiple fixed-size messages from a contiguous buffer. */
GpuHashError gpu_hash_batch_fixed(GpuHashContextHandle handle,
                                  const uint8_t* input,
                                  size_t message_size,
                                  size_t num_messages,
                                  uint8_t* output);

/* ──────────────────────────────────────────────────────────────────────────
 * Utilities
 * ────────────────────────────────────────────────────────────────────────── */

/** Output size in bytes for a given algorithm. */
size_t gpu_hash_output_size(GpuHashAlgorithm algorithm);

/** Last error message (thread-local). */
const char* gpu_hash_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_HASH_H */
