// 256 instructions of one opcode between two clock reads, 8 independent
// accumulators. The host tool rewrites their registers (so nothing in the
// block depends on anything else, or a chosen bank pattern is formed) and
// their control bits, and measures cycles per instruction for one warp.
#include <cstdint>

#define CLK(t) asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory")

#define STREAM(NAME, BODY)                                                         \
  extern "C" __global__ void __launch_bounds__(32) NAME(uint32_t *out, const uint32_t *in, int k) { \
    uint32_t x0 = in[threadIdx.x], x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4,  \
             x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;                                  \
    uint32_t a = in[32 + threadIdx.x], b = in[64 + threadIdx.x];                     \
    uint32_t t0, t1;                                                                \
    /* The clock window starts after the loads: their values go to shared */     \
    /* memory before a barrier (ptxas may not sink a store past the barrier). */      \
    __shared__ uint32_t sink_[1024];                                                \
    sink_[threadIdx.x & 1023] = x0 ^ a ^ b;                                         \
    __syncthreads();                                                                \
    /* Read the store back: otherwise ptxas deletes it and the wait with it. */      \
    if (sink_[(threadIdx.x + 1) & 1023] == 0x9e3779b9u) out[4000] = 1;             \
    CLK(t0);                                                                        \
    _Pragma("unroll") for (int i = 0; i < 32; i++) {                                \
      BODY(x0); BODY(x1); BODY(x2); BODY(x3); BODY(x4); BODY(x5); BODY(x6); BODY(x7); \
    }                                                                               \
    CLK(t1);                                                                        \
    out[threadIdx.x] = x0 ^ x1 ^ x2 ^ x3 ^ x4 ^ x5 ^ x6 ^ x7;                       \
    if (threadIdx.x == 0) out[32] = t1 - t0;                                        \
  }

#define FFMA(x) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b))
#define FADD(x) asm volatile("add.f32 %0, %0, %1;" : "+r"(x) : "r"(a))
#define FMUL(x) asm volatile("mul.f32 %0, %0, %1;" : "+r"(x) : "r"(a))
#define IMAD(x) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b))
#define LOP3(x) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x) : "r"(a), "r"(b))
#define SHF(x) asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b))
#define HFMA2(x) asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b))
#define HFMA2B(x) asm volatile("fma.rn.bf16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(a), "r"(b))
#define F2FP(x) asm volatile("cvt.rn.bf16x2.f32 %0, %0, %1;" : "+r"(x) : "r"(a))
#define EX2(x) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+r"(x))
#define RSQ(x) asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+r"(x))
#define POPC(x) asm volatile("popc.b32 %0, %0;" : "+r"(x))
#define I2FP(x) asm volatile("cvt.rn.f32.u32 %0, %0;" : "+r"(x))
#define F2I(x) asm volatile("cvt.rzi.u32.f32 %0, %0;" : "+r"(x))
#define IMNMX(x) asm volatile("min.u32 %0, %0, %1;" : "+r"(x) : "r"(a))
#define FMNMX(x) asm volatile("max.f32 %0, %0, %1;" : "+r"(x) : "r"(a))
#define MOV(x) asm volatile("mov.b32 %0, %1;" : "=r"(x) : "r"(a))
#define SHFL(x) asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(x))

STREAM(s_ffma, FFMA)
STREAM(s_fadd, FADD)
STREAM(s_fmul, FMUL)
STREAM(s_imad, IMAD)
STREAM(s_lop3, LOP3)
STREAM(s_shf, SHF)
STREAM(s_hfma2, HFMA2)
STREAM(s_hfma2b, HFMA2B)
STREAM(s_f2fp, F2FP)
STREAM(s_ex2, EX2)
STREAM(s_rsq, RSQ)
STREAM(s_popc, POPC)
STREAM(s_i2fp, I2FP)
STREAM(s_f2i, F2I)
STREAM(s_imnmx, IMNMX)
STREAM(s_fmnmx, FMNMX)
STREAM(s_shfl, SHFL)
