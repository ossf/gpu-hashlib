/**
 * @file gpu_hash.cpp
 * @brief SYCL GPU-accelerated SHA-256/384/512 for Intel Xe GPUs
 *
 * Optimised implementation using:
 *   - Local memory for K constants (shared within work-groups)
 *   - Vectorized 4-byte big-endian loads
 *   - Unrolled compression rounds (#pragma unroll)
 *   - Work-group sizes: 256 (SHA-256), 128 (SHA-512/384)
 *   - Batch processing for parallel message hashing
 *   - CPU fallback for batches < 4 messages
 *
 * Build:
 *   icpx -fsycl -fPIC -O3 -std=c++17 -shared -o libgpu_hash_intel.so gpu_hash.cpp
 */

#include "gpu_hash.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef __INTEL_LLVM_COMPILER
#include <sycl/sycl.hpp>
#define HAS_SYCL 1
#else
#define HAS_SYCL 0
#endif

/* ════════════════════════════════════════════════════════════════════════════
 * SHA-256 Constants
 * ════════════════════════════════════════════════════════════════════════════ */

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t H256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* ════════════════════════════════════════════════════════════════════════════
 * SHA-512 / SHA-384 Constants
 * ════════════════════════════════════════════════════════════════════════════ */

static const uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

static const uint64_t H512_INIT[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

static const uint64_t H384_INIT[8] = {
    0xcbbb9d5dc1059ed8ULL, 0x629a292a367cd507ULL, 0x9159015a3070dd17ULL, 0x152fecd8f70e5939ULL,
    0x67332667ffc00b31ULL, 0x8eb44a8768581511ULL, 0xdb0c2e0d64f98fa7ULL, 0x47b5481dbefa4fa4ULL
};

/* ════════════════════════════════════════════════════════════════════════════
 * Inline helpers (shared between CPU and GPU)
 * ════════════════════════════════════════════════════════════════════════════ */

static inline uint32_t rotr32(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static inline uint64_t rotr64(uint64_t x, unsigned n) { return (x >> n) | (x << (64 - n)); }

static inline uint32_t ch32(uint32_t e, uint32_t f, uint32_t g) { return (e & f) ^ (~e & g); }
static inline uint32_t maj32(uint32_t a, uint32_t b, uint32_t c) { return (a & b) ^ (a & c) ^ (b & c); }
static inline uint32_t bsig0_256(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
static inline uint32_t bsig1_256(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
static inline uint32_t ssig0_256(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
static inline uint32_t ssig1_256(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

static inline uint64_t ch64(uint64_t e, uint64_t f, uint64_t g) { return (e & f) ^ (~e & g); }
static inline uint64_t maj64(uint64_t a, uint64_t b, uint64_t c) { return (a & b) ^ (a & c) ^ (b & c); }
static inline uint64_t bsig0_512(uint64_t x) { return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39); }
static inline uint64_t bsig1_512(uint64_t x) { return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41); }
static inline uint64_t ssig0_512(uint64_t x) { return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7); }
static inline uint64_t ssig1_512(uint64_t x) { return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6); }

/* Vectorized big-endian load (4 bytes) */
static inline uint32_t load_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}

static inline uint64_t load_be64(const uint8_t* p) {
    return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) |
           ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32) |
           ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
           ((uint64_t)p[6] << 8)  | (uint64_t)p[7];
}

static inline void store_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
}

static inline void store_be64(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v >> 56); p[1] = (uint8_t)(v >> 48);
    p[2] = (uint8_t)(v >> 40); p[3] = (uint8_t)(v >> 32);
    p[4] = (uint8_t)(v >> 24); p[5] = (uint8_t)(v >> 16);
    p[6] = (uint8_t)(v >> 8);  p[7] = (uint8_t)v;
}

/* ════════════════════════════════════════════════════════════════════════════
 * CPU SHA-256 (fallback / small batches)
 * ════════════════════════════════════════════════════════════════════════════ */

static void cpu_sha256(const uint8_t* input, size_t len, uint8_t* output) {
    uint32_t h[8];
    for (int i = 0; i < 8; i++) h[i] = H256_INIT[i];

    /* Process complete 64-byte blocks */
    size_t full_blocks = len / 64;
    for (size_t blk = 0; blk < full_blocks; blk++) {
        const uint8_t* block = input + blk * 64;
        uint32_t w[64];

        #pragma unroll 4
        for (int i = 0; i < 16; i++)
            w[i] = load_be32(block + i * 4);

        #pragma unroll 4
        for (int i = 16; i < 64; i++)
            w[i] = ssig1_256(w[i-2]) + w[i-7] + ssig0_256(w[i-15]) + w[i-16];

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        #pragma unroll 8
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + bsig1_256(e) + ch32(e, f, g) + K256[i] + w[i];
            uint32_t t2 = bsig0_256(a) + maj32(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    /* Final block with padding */
    uint8_t final_block[128];
    size_t remaining = len - full_blocks * 64;
    memcpy(final_block, input + full_blocks * 64, remaining);
    final_block[remaining] = 0x80;
    memset(final_block + remaining + 1, 0, 128 - remaining - 1);

    size_t final_blocks;
    if (remaining >= 56) {
        store_be64(final_block + 120, (uint64_t)len * 8);
        final_blocks = 2;
    } else {
        store_be64(final_block + 56, (uint64_t)len * 8);
        final_blocks = 1;
    }

    for (size_t blk = 0; blk < final_blocks; blk++) {
        const uint8_t* block = final_block + blk * 64;
        uint32_t w[64];

        for (int i = 0; i < 16; i++)
            w[i] = load_be32(block + i * 4);
        for (int i = 16; i < 64; i++)
            w[i] = ssig1_256(w[i-2]) + w[i-7] + ssig0_256(w[i-15]) + w[i-16];

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + bsig1_256(e) + ch32(e, f, g) + K256[i] + w[i];
            uint32_t t2 = bsig0_256(a) + maj32(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    for (int i = 0; i < 8; i++)
        store_be32(output + i * 4, h[i]);
}

/* ════════════════════════════════════════════════════════════════════════════
 * CPU SHA-512 / SHA-384 (fallback)
 * ════════════════════════════════════════════════════════════════════════════ */

static void cpu_sha512_core(const uint8_t* input, size_t len, uint8_t* output,
                            const uint64_t* init, size_t out_bytes) {
    uint64_t h[8];
    for (int i = 0; i < 8; i++) h[i] = init[i];

    size_t full_blocks = len / 128;
    for (size_t blk = 0; blk < full_blocks; blk++) {
        const uint8_t* block = input + blk * 128;
        uint64_t w[80];

        #pragma unroll 4
        for (int i = 0; i < 16; i++)
            w[i] = load_be64(block + i * 8);

        #pragma unroll 4
        for (int i = 16; i < 80; i++)
            w[i] = ssig1_512(w[i-2]) + w[i-7] + ssig0_512(w[i-15]) + w[i-16];

        uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];

        #pragma unroll 8
        for (int i = 0; i < 80; i++) {
            uint64_t t1 = hh + bsig1_512(e) + ch64(e, f, g) + K512[i] + w[i];
            uint64_t t2 = bsig0_512(a) + maj64(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    /* Final block with padding */
    uint8_t final_block[256];
    size_t remaining = len - full_blocks * 128;
    memcpy(final_block, input + full_blocks * 128, remaining);
    final_block[remaining] = 0x80;
    memset(final_block + remaining + 1, 0, 256 - remaining - 1);

    size_t final_blocks;
    if (remaining >= 112) {
        /* Length goes at byte 248..255 of second block */
        store_be64(final_block + 248, (uint64_t)len * 8);
        final_blocks = 2;
    } else {
        store_be64(final_block + 120, (uint64_t)len * 8);
        final_blocks = 1;
    }

    for (size_t blk = 0; blk < final_blocks; blk++) {
        const uint8_t* block = final_block + blk * 128;
        uint64_t w[80];

        for (int i = 0; i < 16; i++)
            w[i] = load_be64(block + i * 8);
        for (int i = 16; i < 80; i++)
            w[i] = ssig1_512(w[i-2]) + w[i-7] + ssig0_512(w[i-15]) + w[i-16];

        uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];

        for (int i = 0; i < 80; i++) {
            uint64_t t1 = hh + bsig1_512(e) + ch64(e, f, g) + K512[i] + w[i];
            uint64_t t2 = bsig0_512(a) + maj64(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    for (size_t i = 0; i < out_bytes / 8; i++)
        store_be64(output + i * 8, h[i]);
}

static void cpu_sha512(const uint8_t* input, size_t len, uint8_t* output) {
    cpu_sha512_core(input, len, output, H512_INIT, 64);
}

static void cpu_sha384(const uint8_t* input, size_t len, uint8_t* output) {
    cpu_sha512_core(input, len, output, H384_INIT, 48);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Internal State
 * ════════════════════════════════════════════════════════════════════════════ */

static thread_local std::string g_last_error;

struct GpuHashContext {
    GpuHashAlgorithm algorithm;
    int device_index;
    size_t output_size;
#if HAS_SYCL
    std::unique_ptr<sycl::queue> queue;
#endif
};

#if HAS_SYCL
static std::vector<sycl::device> g_devices;
#endif
static bool g_initialized = false;
static std::mutex g_init_mutex;

/* Batch threshold: use GPU only when >= this many messages */
static const size_t GPU_BATCH_THRESHOLD = 4;

/* ════════════════════════════════════════════════════════════════════════════
 * SYCL GPU Kernels
 * ════════════════════════════════════════════════════════════════════════════ */

#if HAS_SYCL

/* ── SHA-256 GPU kernel ────────────────────────────────────────────────── */

static void gpu_sha256_batch_kernel(sycl::queue& q,
                                    const uint8_t* d_input,
                                    const uint64_t* d_offsets,
                                    const uint64_t* d_lengths,
                                    uint8_t* d_output,
                                    size_t num_messages) {
    constexpr size_t WG_SIZE = 256;
    size_t global_size = ((num_messages + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    q.submit([&](sycl::handler& cgh) {
        /* Local memory for K constants — shared across work-group */
        sycl::local_accessor<uint32_t, 1> lk(sycl::range<1>(64), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) {
                size_t gid = item.get_global_id(0);
                size_t lid = item.get_local_id(0);

                /* Cooperatively load K constants into local memory */
                if (lid < 64) {
                    lk[lid] = K256[lid];
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (gid >= num_messages) return;

                uint64_t offset = d_offsets[gid];
                uint64_t msg_len = d_lengths[gid];
                const uint8_t* msg = d_input + offset;

                /* Initialize state */
                uint32_t h[8];
                for (int i = 0; i < 8; i++) h[i] = H256_INIT[i];

                /* Process 64-byte blocks */
                size_t full_blocks = msg_len / 64;
                for (size_t blk = 0; blk < full_blocks; blk++) {
                    const uint8_t* block_ptr = msg + blk * 64;
                    uint32_t w[64];

                    #pragma unroll 4
                    for (int i = 0; i < 16; i++) {
                        w[i] = ((uint32_t)block_ptr[i*4] << 24) |
                               ((uint32_t)block_ptr[i*4+1] << 16) |
                               ((uint32_t)block_ptr[i*4+2] << 8) |
                               (uint32_t)block_ptr[i*4+3];
                    }

                    #pragma unroll 4
                    for (int i = 16; i < 64; i++) {
                        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
                        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
                        w[i] = s1 + w[i-7] + s0 + w[i-16];
                    }

                    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
                    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

                    #pragma unroll 8
                    for (int i = 0; i < 64; i++) {
                        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
                        uint32_t c1 = (e & f) ^ (~e & g);
                        uint32_t t1 = hh + S1 + c1 + lk[i] + w[i];
                        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
                        uint32_t m1 = (a & b) ^ (a & c) ^ (b & c);
                        uint32_t t2 = S0 + m1;
                        hh = g; g = f; f = e; e = d + t1;
                        d = c; c = b; b = a; a = t1 + t2;
                    }

                    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
                    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
                }

                /* Padding in registers */
                uint8_t pad[128];
                size_t rem = msg_len - full_blocks * 64;
                for (size_t i = 0; i < rem; i++)
                    pad[i] = msg[full_blocks * 64 + i];
                pad[rem] = 0x80;
                for (size_t i = rem + 1; i < 128; i++) pad[i] = 0;

                size_t pad_blocks;
                if (rem >= 56) {
                    uint64_t bits = (uint64_t)msg_len * 8;
                    pad[120] = (uint8_t)(bits >> 56); pad[121] = (uint8_t)(bits >> 48);
                    pad[122] = (uint8_t)(bits >> 40); pad[123] = (uint8_t)(bits >> 32);
                    pad[124] = (uint8_t)(bits >> 24); pad[125] = (uint8_t)(bits >> 16);
                    pad[126] = (uint8_t)(bits >> 8);  pad[127] = (uint8_t)bits;
                    pad_blocks = 2;
                } else {
                    uint64_t bits = (uint64_t)msg_len * 8;
                    pad[56] = (uint8_t)(bits >> 56); pad[57] = (uint8_t)(bits >> 48);
                    pad[58] = (uint8_t)(bits >> 40); pad[59] = (uint8_t)(bits >> 32);
                    pad[60] = (uint8_t)(bits >> 24); pad[61] = (uint8_t)(bits >> 16);
                    pad[62] = (uint8_t)(bits >> 8);  pad[63] = (uint8_t)bits;
                    pad_blocks = 1;
                }

                for (size_t blk = 0; blk < pad_blocks; blk++) {
                    const uint8_t* bp = pad + blk * 64;
                    uint32_t w[64];
                    for (int i = 0; i < 16; i++) {
                        w[i] = ((uint32_t)bp[i*4] << 24) | ((uint32_t)bp[i*4+1] << 16) |
                               ((uint32_t)bp[i*4+2] << 8)  | (uint32_t)bp[i*4+3];
                    }
                    for (int i = 16; i < 64; i++) {
                        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
                        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
                        w[i] = s1 + w[i-7] + s0 + w[i-16];
                    }

                    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
                    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
                    for (int i = 0; i < 64; i++) {
                        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
                        uint32_t t1 = hh + S1 + ((e & f) ^ (~e & g)) + lk[i] + w[i];
                        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
                        uint32_t t2 = S0 + ((a & b) ^ (a & c) ^ (b & c));
                        hh = g; g = f; f = e; e = d + t1;
                        d = c; c = b; b = a; a = t1 + t2;
                    }
                    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
                    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
                }

                /* Write output */
                uint8_t* out = d_output + gid * 32;
                for (int i = 0; i < 8; i++) {
                    out[i*4]   = (uint8_t)(h[i] >> 24);
                    out[i*4+1] = (uint8_t)(h[i] >> 16);
                    out[i*4+2] = (uint8_t)(h[i] >> 8);
                    out[i*4+3] = (uint8_t)h[i];
                }
            }
        );
    }).wait();
}

/* ── SHA-512/384 GPU kernel ──────────────────────────────────────────── */

static void gpu_sha512_batch_kernel(sycl::queue& q,
                                    const uint8_t* d_input,
                                    const uint64_t* d_offsets,
                                    const uint64_t* d_lengths,
                                    uint8_t* d_output,
                                    size_t num_messages,
                                    const uint64_t* init_h,
                                    size_t out_bytes) {
    constexpr size_t WG_SIZE = 128;
    size_t global_size = ((num_messages + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint64_t, 1> lk(sycl::range<1>(80), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) {
                size_t gid = item.get_global_id(0);
                size_t lid = item.get_local_id(0);

                /* Cooperatively load K512 into local memory */
                for (size_t i = lid; i < 80; i += WG_SIZE) {
                    lk[i] = K512[i];
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (gid >= num_messages) return;

                uint64_t offset = d_offsets[gid];
                uint64_t msg_len = d_lengths[gid];
                const uint8_t* msg = d_input + offset;

                uint64_t h[8];
                for (int i = 0; i < 8; i++) h[i] = init_h[i];

                size_t full_blocks = msg_len / 128;
                for (size_t blk = 0; blk < full_blocks; blk++) {
                    const uint8_t* bp = msg + blk * 128;
                    uint64_t w[80];

                    #pragma unroll 4
                    for (int i = 0; i < 16; i++) {
                        w[i] = ((uint64_t)bp[i*8] << 56) | ((uint64_t)bp[i*8+1] << 48) |
                               ((uint64_t)bp[i*8+2] << 40) | ((uint64_t)bp[i*8+3] << 32) |
                               ((uint64_t)bp[i*8+4] << 24) | ((uint64_t)bp[i*8+5] << 16) |
                               ((uint64_t)bp[i*8+6] << 8)  | (uint64_t)bp[i*8+7];
                    }
                    #pragma unroll 4
                    for (int i = 16; i < 80; i++) {
                        uint64_t s0 = rotr64(w[i-15], 1) ^ rotr64(w[i-15], 8) ^ (w[i-15] >> 7);
                        uint64_t s1 = rotr64(w[i-2], 19) ^ rotr64(w[i-2], 61) ^ (w[i-2] >> 6);
                        w[i] = s1 + w[i-7] + s0 + w[i-16];
                    }

                    uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
                    uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];

                    #pragma unroll 8
                    for (int i = 0; i < 80; i++) {
                        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
                        uint64_t t1 = hh + S1 + ((e & f) ^ (~e & g)) + lk[i] + w[i];
                        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
                        uint64_t t2 = S0 + ((a & b) ^ (a & c) ^ (b & c));
                        hh = g; g = f; f = e; e = d + t1;
                        d = c; c = b; b = a; a = t1 + t2;
                    }
                    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
                    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
                }

                /* Padding */
                uint8_t pad[256];
                size_t rem = msg_len - full_blocks * 128;
                for (size_t i = 0; i < rem; i++) pad[i] = msg[full_blocks * 128 + i];
                pad[rem] = 0x80;
                for (size_t i = rem + 1; i < 256; i++) pad[i] = 0;

                size_t pad_blocks;
                if (rem >= 112) {
                    uint64_t bits = (uint64_t)msg_len * 8;
                    for (int i = 0; i < 8; i++)
                        pad[248 + i] = (uint8_t)(bits >> (56 - i * 8));
                    pad_blocks = 2;
                } else {
                    uint64_t bits = (uint64_t)msg_len * 8;
                    for (int i = 0; i < 8; i++)
                        pad[120 + i] = (uint8_t)(bits >> (56 - i * 8));
                    pad_blocks = 1;
                }

                for (size_t blk = 0; blk < pad_blocks; blk++) {
                    const uint8_t* bp = pad + blk * 128;
                    uint64_t w[80];
                    for (int i = 0; i < 16; i++) {
                        w[i] = ((uint64_t)bp[i*8] << 56) | ((uint64_t)bp[i*8+1] << 48) |
                               ((uint64_t)bp[i*8+2] << 40) | ((uint64_t)bp[i*8+3] << 32) |
                               ((uint64_t)bp[i*8+4] << 24) | ((uint64_t)bp[i*8+5] << 16) |
                               ((uint64_t)bp[i*8+6] << 8)  | (uint64_t)bp[i*8+7];
                    }
                    for (int i = 16; i < 80; i++) {
                        uint64_t s0 = rotr64(w[i-15], 1) ^ rotr64(w[i-15], 8) ^ (w[i-15] >> 7);
                        uint64_t s1 = rotr64(w[i-2], 19) ^ rotr64(w[i-2], 61) ^ (w[i-2] >> 6);
                        w[i] = s1 + w[i-7] + s0 + w[i-16];
                    }
                    uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
                    uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];
                    for (int i = 0; i < 80; i++) {
                        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
                        uint64_t t1 = hh + S1 + ((e & f) ^ (~e & g)) + lk[i] + w[i];
                        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
                        uint64_t t2 = S0 + ((a & b) ^ (a & c) ^ (b & c));
                        hh = g; g = f; f = e; e = d + t1;
                        d = c; c = b; b = a; a = t1 + t2;
                    }
                    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
                    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
                }

                uint8_t* out = d_output + gid * out_bytes;
                for (size_t i = 0; i < out_bytes / 8; i++) {
                    for (int j = 0; j < 8; j++)
                        out[i * 8 + j] = (uint8_t)(h[i] >> (56 - j * 8));
                }
            }
        );
    }).wait();
}

/* ── GPU batch dispatch ──────────────────────────────────────────────── */

static GpuHashError gpu_batch_dispatch(GpuHashContext* ctx,
                                       const uint8_t* concat_input,
                                       const uint64_t* offsets,
                                       const uint64_t* lengths,
                                       size_t total_input_bytes,
                                       size_t num_messages,
                                       uint8_t* output) {
    try {
        sycl::queue& q = *ctx->queue;
        size_t out_size = ctx->output_size;

        /* Allocate device memory */
        uint8_t*  d_input   = sycl::malloc_device<uint8_t>(total_input_bytes, q);
        uint64_t* d_offsets = sycl::malloc_device<uint64_t>(num_messages, q);
        uint64_t* d_lengths = sycl::malloc_device<uint64_t>(num_messages, q);
        uint8_t*  d_output  = sycl::malloc_device<uint8_t>(out_size * num_messages, q);

        if (!d_input || !d_offsets || !d_lengths || !d_output) {
            g_last_error = "Failed to allocate device memory";
            sycl::free(d_input, q); sycl::free(d_offsets, q);
            sycl::free(d_lengths, q); sycl::free(d_output, q);
            return GPU_HASH_ERROR_MEMORY_ALLOCATION;
        }

        /* Copy to device */
        q.memcpy(d_input, concat_input, total_input_bytes);
        q.memcpy(d_offsets, offsets, num_messages * sizeof(uint64_t));
        q.memcpy(d_lengths, lengths, num_messages * sizeof(uint64_t));
        q.wait();

        /* Launch kernel */
        switch (ctx->algorithm) {
            case GPU_HASH_SHA256:
                gpu_sha256_batch_kernel(q, d_input, d_offsets, d_lengths,
                                        d_output, num_messages);
                break;
            case GPU_HASH_SHA384:
                gpu_sha512_batch_kernel(q, d_input, d_offsets, d_lengths,
                                        d_output, num_messages,
                                        H384_INIT, 48);
                break;
            case GPU_HASH_SHA512:
                gpu_sha512_batch_kernel(q, d_input, d_offsets, d_lengths,
                                        d_output, num_messages,
                                        H512_INIT, 64);
                break;
        }

        /* Copy results back */
        q.memcpy(output, d_output, out_size * num_messages);
        q.wait();

        sycl::free(d_input, q);
        sycl::free(d_offsets, q);
        sycl::free(d_lengths, q);
        sycl::free(d_output, q);

        return GPU_HASH_SUCCESS;
    } catch (sycl::exception& e) {
        g_last_error = std::string("SYCL exception: ") + e.what();
        return GPU_HASH_ERROR_KERNEL_EXECUTION;
    } catch (std::exception& e) {
        g_last_error = std::string("Exception: ") + e.what();
        return GPU_HASH_ERROR_UNKNOWN;
    }
}

#endif /* HAS_SYCL */

/* ════════════════════════════════════════════════════════════════════════════
 * CPU batch (fallback for small batches or no GPU)
 * ════════════════════════════════════════════════════════════════════════════ */

static void cpu_hash_single_dispatch(GpuHashAlgorithm alg,
                                     const uint8_t* input, size_t len,
                                     uint8_t* output) {
    switch (alg) {
        case GPU_HASH_SHA256: cpu_sha256(input, len, output); break;
        case GPU_HASH_SHA384: cpu_sha384(input, len, output); break;
        case GPU_HASH_SHA512: cpu_sha512(input, len, output); break;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * C API Implementation
 * ════════════════════════════════════════════════════════════════════════════ */

extern "C" {

GpuHashError gpu_hash_init(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    if (g_initialized) return GPU_HASH_SUCCESS;

#if HAS_SYCL
    try {
        auto platforms = sycl::platform::get_platforms();
        for (auto& p : platforms) {
            auto devices = p.get_devices(sycl::info::device_type::gpu);
            for (auto& d : devices) {
                g_devices.push_back(d);
            }
        }
    } catch (sycl::exception& e) {
        g_last_error = std::string("SYCL init: ") + e.what();
    }
#endif

    g_initialized = true;
    return GPU_HASH_SUCCESS;
}

void gpu_hash_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
#if HAS_SYCL
    g_devices.clear();
#endif
    g_initialized = false;
}

int gpu_hash_is_available(void) {
#if HAS_SYCL
    return g_devices.empty() ? 0 : 1;
#else
    return 0;
#endif
}

int gpu_hash_get_device_count(void) {
#if HAS_SYCL
    return (int)g_devices.size();
#else
    return 0;
#endif
}

GpuHashError gpu_hash_get_device_info(int device_index, GpuDeviceInfo* info) {
    if (!info) return GPU_HASH_ERROR_INVALID_INPUT;
#if HAS_SYCL
    if (device_index < 0 || device_index >= (int)g_devices.size()) {
        g_last_error = "Device index out of range";
        return GPU_HASH_ERROR_NO_DEVICE;
    }

    auto& dev = g_devices[device_index];
    memset(info, 0, sizeof(*info));

    auto name = dev.get_info<sycl::info::device::name>();
    auto vendor = dev.get_info<sycl::info::device::vendor>();
    auto driver = dev.get_info<sycl::info::device::driver_version>();

    strncpy(info->name, name.c_str(), sizeof(info->name) - 1);
    strncpy(info->vendor, vendor.c_str(), sizeof(info->vendor) - 1);
    strncpy(info->driver_version, driver.c_str(), sizeof(info->driver_version) - 1);

    info->device_type = GPU_DEVICE_TYPE_GPU;
    info->max_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    info->global_memory_size = dev.get_info<sycl::info::device::global_mem_size>();
    info->local_memory_size = dev.get_info<sycl::info::device::local_mem_size>();
    info->max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();

    info->is_intel = (vendor.find("Intel") != std::string::npos) ? 1 : 0;
    /* Xe detection: check for known Xe device name substrings */
    info->is_intel_xe = (name.find("Arc") != std::string::npos ||
                         name.find("Xe") != std::string::npos ||
                         name.find("Iris") != std::string::npos ||
                         name.find("UHD") != std::string::npos) ? 1 : 0;

    return GPU_HASH_SUCCESS;
#else
    g_last_error = "SYCL not available";
    return GPU_HASH_ERROR_NO_DEVICE;
#endif
}

GpuHashError gpu_hash_create_context(GpuHashAlgorithm algorithm,
                                     int device_index,
                                     GpuHashContextHandle* handle) {
    if (!handle) return GPU_HASH_ERROR_INVALID_INPUT;
    if (!g_initialized) {
        g_last_error = "Library not initialized — call gpu_hash_init() first";
        return GPU_HASH_ERROR_NOT_INITIALIZED;
    }

    auto* ctx = new GpuHashContext();
    ctx->algorithm = algorithm;
    ctx->device_index = device_index;

    switch (algorithm) {
        case GPU_HASH_SHA256: ctx->output_size = 32; break;
        case GPU_HASH_SHA384: ctx->output_size = 48; break;
        case GPU_HASH_SHA512: ctx->output_size = 64; break;
        default:
            delete ctx;
            g_last_error = "Invalid algorithm";
            return GPU_HASH_ERROR_INVALID_ALGORITHM;
    }

#if HAS_SYCL
    if (!g_devices.empty()) {
        try {
            int idx = device_index;
            if (idx < 0) {
                /* Auto-select: prefer first Intel GPU */
                idx = 0;
                for (int i = 0; i < (int)g_devices.size(); i++) {
                    auto v = g_devices[i].get_info<sycl::info::device::vendor>();
                    if (v.find("Intel") != std::string::npos) { idx = i; break; }
                }
            }
            if (idx >= (int)g_devices.size()) idx = 0;
            ctx->device_index = idx;
            ctx->queue = std::make_unique<sycl::queue>(
                g_devices[idx],
                sycl::property::queue::in_order{}
            );
        } catch (sycl::exception& e) {
            g_last_error = std::string("Queue creation failed: ") + e.what();
            ctx->queue = nullptr;
        }
    }
#endif

    *handle = (GpuHashContextHandle)ctx;
    return GPU_HASH_SUCCESS;
}

void gpu_hash_destroy_context(GpuHashContextHandle handle) {
    if (!handle) return;
    auto* ctx = (GpuHashContext*)handle;
    delete ctx;
}

GpuHashError gpu_hash_single(GpuHashContextHandle handle,
                             const uint8_t* input, size_t input_len,
                             uint8_t* output, size_t* output_len) {
    if (!handle || !output) return GPU_HASH_ERROR_INVALID_INPUT;

    auto* ctx = (GpuHashContext*)handle;

    /* Single message: always use CPU (faster than GPU launch overhead) */
    cpu_hash_single_dispatch(ctx->algorithm, input, input_len, output);
    if (output_len) *output_len = ctx->output_size;
    return GPU_HASH_SUCCESS;
}

GpuHashError gpu_hash_batch(GpuHashContextHandle handle,
                            const uint8_t** inputs, const size_t* input_lens,
                            size_t num_inputs,
                            uint8_t** outputs, size_t output_size) {
    if (!handle || !inputs || !input_lens || !outputs)
        return GPU_HASH_ERROR_INVALID_INPUT;
    if (num_inputs == 0) return GPU_HASH_SUCCESS;

    auto* ctx = (GpuHashContext*)handle;

#if HAS_SYCL
    /* Use GPU for batches >= threshold when GPU is available */
    if (ctx->queue && num_inputs >= GPU_BATCH_THRESHOLD) {
        /* Concatenate inputs into contiguous buffer */
        size_t total = 0;
        for (size_t i = 0; i < num_inputs; i++) total += input_lens[i];

        std::vector<uint8_t> concat(total);
        std::vector<uint64_t> offsets(num_inputs);
        std::vector<uint64_t> lengths(num_inputs);
        size_t pos = 0;
        for (size_t i = 0; i < num_inputs; i++) {
            offsets[i] = pos;
            lengths[i] = input_lens[i];
            memcpy(concat.data() + pos, inputs[i], input_lens[i]);
            pos += input_lens[i];
        }

        std::vector<uint8_t> flat_output(ctx->output_size * num_inputs);

        GpuHashError err = gpu_batch_dispatch(
            ctx, concat.data(), offsets.data(), lengths.data(),
            total, num_inputs, flat_output.data());

        if (err == GPU_HASH_SUCCESS) {
            for (size_t i = 0; i < num_inputs; i++) {
                memcpy(outputs[i], flat_output.data() + i * ctx->output_size,
                       ctx->output_size);
            }
            return GPU_HASH_SUCCESS;
        }
        /* Fall through to CPU on GPU error */
    }
#endif

    /* CPU fallback */
    for (size_t i = 0; i < num_inputs; i++) {
        cpu_hash_single_dispatch(ctx->algorithm, inputs[i], input_lens[i], outputs[i]);
    }
    return GPU_HASH_SUCCESS;
}

GpuHashError gpu_hash_batch_fixed(GpuHashContextHandle handle,
                                  const uint8_t* input,
                                  size_t message_size,
                                  size_t num_messages,
                                  uint8_t* output) {
    if (!handle || !input || !output) return GPU_HASH_ERROR_INVALID_INPUT;
    if (num_messages == 0) return GPU_HASH_SUCCESS;

    auto* ctx = (GpuHashContext*)handle;

#if HAS_SYCL
    if (ctx->queue && num_messages >= GPU_BATCH_THRESHOLD) {
        std::vector<uint64_t> offsets(num_messages);
        std::vector<uint64_t> lengths(num_messages);
        for (size_t i = 0; i < num_messages; i++) {
            offsets[i] = i * message_size;
            lengths[i] = message_size;
        }

        GpuHashError err = gpu_batch_dispatch(
            ctx, input, offsets.data(), lengths.data(),
            message_size * num_messages, num_messages, output);

        if (err == GPU_HASH_SUCCESS) return GPU_HASH_SUCCESS;
    }
#endif

    /* CPU fallback */
    for (size_t i = 0; i < num_messages; i++) {
        cpu_hash_single_dispatch(ctx->algorithm,
                                 input + i * message_size, message_size,
                                 output + i * ctx->output_size);
    }
    return GPU_HASH_SUCCESS;
}

size_t gpu_hash_output_size(GpuHashAlgorithm algorithm) {
    switch (algorithm) {
        case GPU_HASH_SHA256: return 32;
        case GPU_HASH_SHA384: return 48;
        case GPU_HASH_SHA512: return 64;
        default: return 0;
    }
}

const char* gpu_hash_get_last_error(void) {
    return g_last_error.c_str();
}

} /* extern "C" */
