#include <stdint.h>
#include <stdio.h>
#include "utils/utils.h"
#include "utils/channel.hpp"
#include "common.h"

// Sampled CTAs: up to 4 linear ids packed two per 64-bit argument; 0xffffffff = unused.
__device__ __forceinline__ bool wp_sampled(uint32_t *cta, uint64_t s01, uint64_t s23) {
    uint32_t id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    *cta = id;
    return id == (uint32_t)s01 || id == (uint32_t)(s01 >> 32) || id == (uint32_t)s23 ||
           id == (uint32_t)(s23 >> 32);
}

__device__ __forceinline__ void wp_fill(wp_rec_t &r, uint32_t cta, uint32_t idx, uint8_t kind,
                                        uint32_t active, uint32_t predon) {
    uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    r.cta = cta;
    r.warp = tid / 32;
    r.kind = kind;
    r.aux = 0;
    r.idx = idx;
    r.active = active;
    r.predon = predon;
    r.pad = 0;
    r.a = r.b = r.c = 0;
    r.clk = clock64();
}

extern "C" __device__ __noinline__ void wp_basic(int pred, uint32_t idx, uint64_t s01, uint64_t s23,
                                                 uint64_t pch) {
    uint32_t cta;
    if (!wp_sampled(&cta, s01, s23)) return;
    uint32_t active = __activemask();
    uint32_t predon = __ballot_sync(active, pred);
    if (get_laneid() != __ffs(active) - 1) return;
    wp_rec_t r;
    wp_fill(r, cta, idx, WP_BASIC, active, predon);
    ((ChannelDev *)pch)->push(&r, sizeof(r));
}

// a/b/c are per-thread values; the first guard-true lane's values are kept.
__device__ __forceinline__ void wp_vals_impl(int pred, uint32_t idx, uint32_t kind, uint32_t aux,
                                             uint64_t a, uint64_t b, uint64_t c, uint64_t s01,
                                             uint64_t s23, uint64_t pch) {
    uint32_t cta;
    if (!wp_sampled(&cta, s01, s23)) return;
    uint32_t active = __activemask();
    uint32_t predon = __ballot_sync(active, pred);
    int src = predon ? __ffs(predon) - 1 : __ffs(active) - 1;
    a = __shfl_sync(active, a, src);
    b = __shfl_sync(active, b, src);
    c = __shfl_sync(active, c, src);
    if (get_laneid() != __ffs(active) - 1) return;
    wp_rec_t r;
    wp_fill(r, cta, idx, (uint8_t)kind, active, predon);
    r.aux = (uint8_t)aux;
    r.a = a;
    r.b = b;
    r.c = c;
    ((ChannelDev *)pch)->push(&r, sizeof(r));
}

// mbarrier ops and try_wait: a = 64-bit address, b/c = 32-bit register values.
extern "C" __device__ __noinline__ void wp_vals(int pred, uint32_t idx, uint32_t kind, uint32_t aux,
                                                uint64_t a, uint32_t b, uint32_t c, uint64_t s01,
                                                uint64_t s23, uint64_t pch) {
    wp_vals_impl(pred, idx, kind, aux, a, b, c, s01, s23, pch);
}

// tcgen05 ops: three 32-bit uniform register values.
extern "C" __device__ __noinline__ void wp_ureg(int pred, uint32_t idx, uint32_t a, uint32_t b, uint32_t c,
                                                uint64_t s01, uint64_t s23, uint64_t pch) {
    wp_vals_impl(pred, idx, WP_UREG, 0, a, b, c, s01, s23, pch);
}

extern "C" __device__ __noinline__ void wp_mem(int pred, uint32_t idx, uint64_t addr, uint64_t s01,
                                               uint64_t s23, uint64_t pch) {
    uint32_t cta;
    if (!wp_sampled(&cta, s01, s23)) return;
    uint32_t active = __activemask();
    uint32_t predon = __ballot_sync(active, pred);
    wp_mem_t m;
    for (int i = 0; i < 32; i++) m.addrs[i] = __shfl_sync(active, pred ? addr : 0, i);
    if (get_laneid() != __ffs(active) - 1) return;
    wp_fill(m.r, cta, idx, WP_MEM, active, predon);
    ((ChannelDev *)pch)->push(&m, sizeof(m));
}

// all != 0: record TMA copies of every CTA (address-only), not just the sampled ones.
extern "C" __device__ __noinline__ void wp_tma(int pred, uint32_t idx, uint64_t handle, uint32_t size,
                                               uint32_t all, uint64_t s01, uint64_t s23, uint64_t pch) {
    uint32_t cta;
    if (!wp_sampled(&cta, s01, s23) && !all) return;
    uint32_t active = __activemask();
    uint32_t predon = __ballot_sync(active, pred);
    int src = predon ? __ffs(predon) - 1 : __ffs(active) - 1;
    if (get_laneid() != src) return;
    wp_tma_t t;
    wp_fill(t.r, cta, idx, WP_TMA, active, predon);
    t.handle_size = size;
    t.pad2 = 0;
    t.handle_ptr = handle;
    memset(t.handle, 0, sizeof(t.handle));
    if (pred) memcpy(t.handle, (uint8_t *)handle, size > 128 ? 128 : size);
    ((ChannelDev *)pch)->push(&t, sizeof(t));
}
