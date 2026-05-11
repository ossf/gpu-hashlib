/* Stub library for when Intel oneAPI/SYCL is not available */
#include <stdint.h>
#include <stddef.h>

typedef enum { GPU_HASH_SHA256 = 0, GPU_HASH_SHA384 = 1, GPU_HASH_SHA512 = 2 } GpuHashAlgorithm;
typedef enum {
    GPU_HASH_SUCCESS = 0, GPU_HASH_ERROR_NO_DEVICE = 1,
    GPU_HASH_ERROR_INVALID_ALGORITHM = 2, GPU_HASH_ERROR_MEMORY_ALLOCATION = 3,
    GPU_HASH_ERROR_KERNEL_EXECUTION = 4, GPU_HASH_ERROR_INVALID_INPUT = 5,
    GPU_HASH_ERROR_NOT_INITIALIZED = 6, GPU_HASH_ERROR_UNKNOWN = 99
} GpuHashError;
typedef enum { GPU_DEVICE_TYPE_GPU = 0, GPU_DEVICE_TYPE_CPU = 1, GPU_DEVICE_TYPE_ACCELERATOR = 2 } GpuDeviceType;
typedef struct {
    char name[256]; char vendor[256]; char driver_version[64]; GpuDeviceType device_type;
    uint32_t max_compute_units; uint64_t global_memory_size; uint64_t local_memory_size;
    size_t max_work_group_size; int is_intel; int is_intel_xe;
} GpuDeviceInfo;
typedef void* GpuHashContextHandle;
static const char* last_error = "Intel SYCL not available — stub library";

GpuHashError gpu_hash_init(void) { return GPU_HASH_SUCCESS; }
void gpu_hash_cleanup(void) {}
int gpu_hash_is_available(void) { return 0; }
int gpu_hash_get_device_count(void) { return 0; }
GpuHashError gpu_hash_get_device_info(int idx, GpuDeviceInfo* info) { (void)idx; (void)info; return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_create_context(GpuHashAlgorithm alg, int idx, GpuHashContextHandle* h) { (void)alg; (void)idx; (void)h; return GPU_HASH_ERROR_NO_DEVICE; }
void gpu_hash_destroy_context(GpuHashContextHandle h) { (void)h; }
GpuHashError gpu_hash_single(GpuHashContextHandle h, const uint8_t* in, size_t len, uint8_t* out, size_t* olen) { (void)h; (void)in; (void)len; (void)out; (void)olen; return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_batch(GpuHashContextHandle h, const uint8_t** ins, const size_t* lens, size_t n, uint8_t** outs, size_t osize) { (void)h; (void)ins; (void)lens; (void)n; (void)outs; (void)osize; return GPU_HASH_ERROR_NO_DEVICE; }
GpuHashError gpu_hash_batch_fixed(GpuHashContextHandle h, const uint8_t* in, size_t msize, size_t n, uint8_t* out) { (void)h; (void)in; (void)msize; (void)n; (void)out; return GPU_HASH_ERROR_NO_DEVICE; }
size_t gpu_hash_output_size(GpuHashAlgorithm alg) { return alg == GPU_HASH_SHA256 ? 32 : alg == GPU_HASH_SHA384 ? 48 : 64; }
const char* gpu_hash_get_last_error(void) { return last_error; }
