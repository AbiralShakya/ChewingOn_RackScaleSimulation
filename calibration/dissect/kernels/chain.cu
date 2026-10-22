// Dependent chains of one SASS opcode, bracketed by clock reads.
// The host tool patches the stall count of every chain instruction and
// checks the result: the smallest stall that still gives the right answer
// is the opcode's true pipeline latency (no hardware interlock exists for
// fixed-latency instructions on Volta and later).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#define N 64
#define CLK(t) asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory")

#define CHAIN32(NAME, BODY)                                                     \
  extern "C" __global__ void NAME(uint32_t *out, const uint32_t *in, int k) {   \
    uint32_t x = in[threadIdx.x], a = in[32 + threadIdx.x],                    \
             b = in[64 + threadIdx.x];                                          \
    uint32_t t0, t1;                                                            \
    if ((x ^ a ^ b) == 0x9e3779b9u) out[40] = 1;                                \
    CLK(t0);                                                                    \
    _Pragma("unroll") for (int i = 0; i < N; i++) { BODY; }                     \
    CLK(t1);                                                                    \
    out[threadIdx.x] = x;                                                       \
    if (threadIdx.x == 0) out[32] = t1 - t0;                                    \
  }

CHAIN32(c_ffma_a, asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_ffma_c, asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_ffma_b, asm volatile("fma.rn.f32 %0, %1, %0, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_fadd, asm volatile("add.f32 %0, %0, %1;" : "+r"(x) : "r"(a)))
CHAIN32(c_fmul, asm volatile("mul.f32 %0, %0, %1;" : "+r"(x) : "r"(a)))
CHAIN32(c_hfma2, asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_hfma2_bf16, asm volatile("fma.rn.bf16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_imad, asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_imad_hi, asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_lop3, asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_iadd3, asm volatile("add.u32 %0, %0, %1;\n\txor.b32 %0, %0, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_shf, asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_prmt, asm volatile("prmt.b32 %0, %0, %1, 0x3254;\n\txor.b32 %0, %0, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_imnmx, asm volatile("min.s32 %0, %0, %1;\n\txor.b32 %0, %0, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_fmnmx, asm volatile("max.f32 %0, %0, %1;\n\tadd.f32 %0, %0, %2;" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_f2fp_bf16, asm volatile("cvt.rn.bf16x2.f32 %0, %0, %1;" : "+r"(x) : "r"(a)))
CHAIN32(c_i2fp, asm volatile("cvt.rn.f32.u32 %0, %0;\n\txor.b32 %0, %0, %1;" : "+r"(x) : "r"(a)))
CHAIN32(c_popc, asm volatile("popc.b32 %0, %0;\n\tadd.u32 %0, %0, %1;" : "+r"(x) : "r"(a)))
CHAIN32(c_mufu_ex2, asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+r"(x)))
CHAIN32(c_mufu_rsq, asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+r"(x)))
CHAIN32(c_mufu_tanh, asm volatile("tanh.approx.f32 %0, %0;" : "+r"(x)))
CHAIN32(c_sel, asm volatile("{.reg .pred p; setp.lt.u32 p, %0, %1; selp.b32 %0, %1, %2, p; xor.b32 %0, %0, %1;}" : "+r"(x) : "r"(a), "r"(b)))
CHAIN32(c_shfl, asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;\n\tadd.u32 %0, %0, %1;" : "+r"(x) : "r"(a)))

// f32x2 FMA (FFMA2) on a 64-bit register pair.
extern "C" __global__ void c_ffma2(uint32_t *out, const uint32_t *in, int k) {
  uint64_t x = ((uint64_t)in[threadIdx.x] << 32) | in[96 + threadIdx.x];
  uint64_t a = ((uint64_t)in[32 + threadIdx.x] << 32) | in[32 + threadIdx.x];
  uint64_t b = ((uint64_t)in[64 + threadIdx.x] << 32) | in[64 + threadIdx.x];
  uint32_t t0, t1;
  if (((uint32_t)x ^ (uint32_t)(x >> 32) ^ (uint32_t)a ^ (uint32_t)b) == 0x9e3779b9u) out[40] = 1;
  __syncwarp();
  CLK(t0);
#pragma unroll
  for (int i = 0; i < N; i++) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(x) : "l"(a), "l"(b));
  CLK(t1);
  out[threadIdx.x] = (uint32_t)x ^ (uint32_t)(x >> 32);
  if (threadIdx.x == 0) out[32] = t1 - t0;
}

extern "C" __global__ void c_dfma(uint32_t *out, const uint32_t *in, int k) {
  double x = in[threadIdx.x] * 1e-3, a = 0.999 + in[32 + threadIdx.x] * 1e-9, b = in[64 + threadIdx.x] * 1e-6;
  uint32_t t0, t1;
  if (x + a + b == 0.123) out[40] = 1;
  __syncwarp();
  CLK(t0);
#pragma unroll
  for (int i = 0; i < N; i++) asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(x) : "d"(a), "d"(b));
  CLK(t1);
  uint64_t u = __double_as_longlong(x);
  out[threadIdx.x] = (uint32_t)u ^ (uint32_t)(u >> 32);
  if (threadIdx.x == 0) out[32] = t1 - t0;
}

// HMMA m16n8k16 bf16 -> f32 with the accumulator as the dependence.
extern "C" __global__ void c_hmma_bf16(uint32_t *out, const uint32_t *in, int k) {
  uint32_t a0 = in[threadIdx.x], a1 = in[32 + threadIdx.x], a2 = in[64 + threadIdx.x], a3 = in[96 + threadIdx.x];
  uint32_t b0 = in[128 + threadIdx.x], b1 = in[160 + threadIdx.x];
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  uint32_t t0, t1;
  if ((a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1) == 0x9e3779b9u) out[40] = 1;
  __syncwarp();
  CLK(t0);
#pragma unroll
  for (int i = 0; i < N; i++)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  CLK(t1);
  out[threadIdx.x] = __float_as_uint(c0) ^ __float_as_uint(c1) ^ __float_as_uint(c2) ^ __float_as_uint(c3);
  if (threadIdx.x == 0) out[32] = t1 - t0;
}
