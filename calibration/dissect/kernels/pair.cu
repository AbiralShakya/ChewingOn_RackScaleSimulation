// Dispatch occupancy and pipe initiation interval per opcode.
// p_<X>: 128 x [X, FMUL] with independent registers; a_<X>: 256 x X.
// The host (pair_sweep.py) patches the stall count of X (and sets the
// filler's to 1) and times the block for 1..16 warps per SM sub-partition.
#include <cstdint>

#define CLK(t) asm volatile("mov.u32 %0, %%clock;" : "=r"(t)::"memory")
#define W(t) ((threadIdx.x / 32) * 4 + 0)

#define KERNEL(NAME, DECL, X, FILL, NREP, FIN, WAITIN)                                            \
  extern "C" __global__ void NAME(uint32_t *out, const uint32_t *in, int k) {             \
    __shared__ __align__(16) uint32_t sm[4096];                                           \
    sm[threadIdx.x & 4095] = in[threadIdx.x & 255];                                       \
    __syncthreads();                                                                      \
    DECL();                                                                               \
    float f0 = __uint_as_float(in[threadIdx.x & 255]), f1 = f0 + 1, f2 = f0 + 2, f3 = f0 + 3; \
    float g = __uint_as_float(in[(threadIdx.x + 7) & 255]);                               \
    uint32_t t0, t1;                                                                      \
    /* Loaded operands go to shared memory before a barrier, so the clock */           \
    /* window starts after the loads (ptxas cannot sink a store past it). */              \
    WAITIN();                                                                             \
    sm[(threadIdx.x + 3072) & 4095] = __float_as_uint(f0 + g);                            \
    __syncthreads();                                                                      \
    if (sm[(threadIdx.x + 3073) & 4095] == 0x9e3779b9u) out[4000] = 1;                    \
    CLK(t0);                                                                              \
    _Pragma("unroll") for (int i = 0; i < NREP; i++) {                                    \
      X(0); FILL(f0); X(1); FILL(f1); X(2); FILL(f2); X(3); FILL(f3);                     \
      X(4); FILL(f0); X(5); FILL(f1); X(6); FILL(f2); X(7); FILL(f3);                     \
    }                                                                                     \
    CLK(t1);                                                                              \
    uint32_t r = __float_as_uint(f0 + f1 + f2 + f3);                                      \
    FIN();                                                                                \
    out[threadIdx.x] = r;                                                                 \
    if ((threadIdx.x & 31) == 0) { out[4096 + threadIdx.x / 32] = t1 - t0; out[6144 + 2 * (threadIdx.x / 32)] = t0; out[6145 + 2 * (threadIdx.x / 32)] = t1; }                  \
  }

#define NOFILL(f)
#define FILL_FMUL(f) asm volatile("mul.f32 %0, %0, %1;" : "+f"(f) : "f"(g))

// --- X definitions over 8 independent accumulators -----------------------
#define D_U() uint32_t u[8]; _Pragma("unroll") for (int j = 0; j < 8; j++) u[j] = in[(threadIdx.x + j) & 255];
#define W_D_U() { uint32_t z = 0; _Pragma("unroll") for (int j = 0; j < 8; j++) z ^= u[j]; sm[(threadIdx.x + 2048) & 4095] = z; }
#define F_U() _Pragma("unroll") for (int j = 0; j < 8; j++) r ^= u[j];
#define X_FFMA(j) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+r"(u[j]) : "r"(u[(j + 1) & 7]), "r"(u[(j + 2) & 7]))
#define X_IMAD(j) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(u[j]) : "r"(u[(j + 1) & 7]), "r"(u[(j + 2) & 7]))
#define X_HFMA2(j) asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(u[j]) : "r"(u[(j + 1) & 7]), "r"(u[(j + 2) & 7]))
#define X_EX2(j) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+r"(u[j]))
#define X_RSQ(j) asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+r"(u[j]))
#define X_POPC(j) asm volatile("popc.b32 %0, %0;" : "+r"(u[j]))
#define X_I2FP(j) asm volatile("cvt.rn.f32.u32 %0, %0;" : "+r"(u[j]))
#define X_F2FP(j) asm volatile("cvt.rn.bf16x2.f32 %0, %0, %1;" : "+r"(u[j]) : "r"(u[(j + 3) & 7]))
#define X_SHFL(j) asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(u[j]))
#define X_LDS(j) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(u[j]) : "r"((uint32_t)__cvta_generic_to_shared(sm) + 4 * ((threadIdx.x + 64 * j) & 4095)))
#define X_LDS128(j) asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(u[j]), "=r"(u[(j + 1) & 7]), "=r"(u[(j + 2) & 7]), "=r"(u[(j + 3) & 7]) : "r"((uint32_t)__cvta_generic_to_shared(sm) + 16 * ((threadIdx.x + 32 * j) & 1023)))
#define X_LDG(j) asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(u[j]) : "l"(in + ((threadIdx.x + 32 * j) & 255)))
#define X_STS(j) asm volatile("st.shared.u32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(sm) + 4 * ((threadIdx.x + 64 * j) & 4095)), "r"(u[j]) : "memory")

#define D_D() double d[8]; _Pragma("unroll") for (int j = 0; j < 8; j++) d[j] = in[(threadIdx.x + j) & 255] * 1e-3;
#define W_D_D() { double z = 0; _Pragma("unroll") for (int j = 0; j < 8; j++) z += d[j]; sm[(threadIdx.x + 2048) & 4095] = (uint32_t)__double_as_longlong(z); }
#define F_D() _Pragma("unroll") for (int j = 0; j < 8; j++) r ^= (uint32_t)__double_as_longlong(d[j]);
#define X_DFMA(j) asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(d[j]) : "d"(d[(j + 1) & 7]), "d"(d[(j + 2) & 7]))

#define D_H() float c[8][4] = {}; uint32_t a0 = in[threadIdx.x & 255], a1 = a0 ^ 1, a2 = a0 ^ 2, a3 = a0 ^ 3, b0 = a0 ^ 4, b1 = a0 ^ 5;
#define W_D_H() { sm[(threadIdx.x + 2048) & 4095] = a0 ^ b1; }
#define F_H() _Pragma("unroll") for (int j = 0; j < 8; j++) r ^= __float_as_uint(c[j][0] + c[j][1] + c[j][2] + c[j][3]);
#define X_HMMA(j) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
    : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define X_HMMA_F16(j) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" \
    : "+r"(*(uint32_t *)&c[j][0]), "+r"(*(uint32_t *)&c[j][1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define D_L() uint32_t v[8][4]; uint32_t sb = (uint32_t)__cvta_generic_to_shared(sm) + 16 * (threadIdx.x & 31);
#define W_D_L() {}
#define F_L() _Pragma("unroll") for (int j = 0; j < 8; j++) r ^= v[j][0] ^ v[j][1] ^ v[j][2] ^ v[j][3];
#define X_LDSM(j) asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" \
    : "=r"(v[j][0]), "=r"(v[j][1]), "=r"(v[j][2]), "=r"(v[j][3]) : "r"(sb + 512 * j))

#define BOTH(N, D, X, F)                      \
  KERNEL(p_##N, D, X, FILL_FMUL, 16, F, W_##D)      \
  KERNEL(a_##N, D, X, NOFILL, 32, F, W_##D)

BOTH(ffma, D_U, X_FFMA, F_U)
BOTH(imad, D_U, X_IMAD, F_U)
BOTH(hfma2, D_U, X_HFMA2, F_U)
BOTH(ex2, D_U, X_EX2, F_U)
BOTH(rsq, D_U, X_RSQ, F_U)
BOTH(popc, D_U, X_POPC, F_U)
BOTH(i2fp, D_U, X_I2FP, F_U)
BOTH(f2fp, D_U, X_F2FP, F_U)
BOTH(shfl, D_U, X_SHFL, F_U)
BOTH(lds, D_U, X_LDS, F_U)
BOTH(lds128, D_U, X_LDS128, F_U)
BOTH(ldg, D_U, X_LDG, F_U)
BOTH(sts, D_U, X_STS, F_U)
BOTH(dfma, D_D, X_DFMA, F_D)
BOTH(hmma, D_H, X_HMMA, F_H)
BOTH(hmma_f16, D_H, X_HMMA_F16, F_H)
BOTH(ldsm, D_L, X_LDSM, F_L)
