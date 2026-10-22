// Instruction fetch: straight-line blocks of N independent FMULs (N = 64 ..
// 16384, 8 accumulators) between two clock reads. ifetch.py patches the
// stall count of every FMUL and times one warp (and 4 warps on one SMSP):
// cycles per instruction vs code size shows the instruction buffer refill
// cost and the L0 / L1 instruction-cache capacities.
#include <cstdint>

#define CLK(t) asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory")
#define FM(x) asm volatile("mul.f32 %0, %0, %1;" : "+r"(x) : "r"(a))

#define IFETCH(N)                                                                        \
  extern "C" __global__ void f_##N(uint32_t *out, const uint32_t *in, int k) {           \
    __shared__ uint32_t sink_[1024];                                                     \
    uint32_t x0 = in[threadIdx.x & 255], x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, \
             x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7, a = in[(threadIdx.x + 32) & 255];       \
    sink_[threadIdx.x & 1023] = x0 ^ a;                                                  \
    __syncthreads();                                                                     \
    if (sink_[(threadIdx.x + 1) & 1023] == 0x9e3779b9u) out[4000] = 1;                     \
    uint32_t t0, t1;                                                                     \
    CLK(t0);                                                                             \
    _Pragma("unroll") for (int i = 0; i < N / 8; i++) {                                  \
      FM(x0); FM(x1); FM(x2); FM(x3); FM(x4); FM(x5); FM(x6); FM(x7);                    \
    }                                                                                    \
    CLK(t1);                                                                             \
    out[threadIdx.x & 4095] = x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7;                     \
    if ((threadIdx.x & 31) == 0) { out[4096 + threadIdx.x / 32] = t1 - t0; out[6144 + 2 * (threadIdx.x / 32)] = t0; out[6145 + 2 * (threadIdx.x / 32)] = t1; }                 \
  }

IFETCH(64)
IFETCH(128)
IFETCH(256)
IFETCH(512)
IFETCH(1024)
IFETCH(2048)
IFETCH(4096)
IFETCH(8192)
IFETCH(16384)
