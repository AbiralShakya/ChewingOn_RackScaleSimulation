// tcgen05 helpers shared by bench_tc.cu and bench_tc05x.cu: shared-memory and
// instruction descriptors, UMMA issue for every kind and CTA group, commit,
// mbarrier wait, cluster helpers.
#pragma once
#include "common.cuh"

// ---------------------------------------------------------------------------
// tcgen05 (sm_100+)
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define HAS_TCGEN05 1
#endif

__host__ __device__ constexpr uint64_t smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo, uint32_t swz) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3FFF) << 32) | (1ull << 46) | ((uint64_t)(swz & 7) << 61);
}

enum Kind { KIND_F16 = 0, KIND_TF32 = 1, KIND_F8F6F4 = 2, KIND_MXF8 = 3, KIND_NVFP4 = 4, KIND_I8 = 5, KIND_MXF4 = 6, KIND_NVFP4_K96 = 7, KIND_MXF4_K96 = 8 };

// Instruction descriptor (PTX ISA, tcgen05 "Instruction descriptor"):
// [4,6) D fmt, [7,10) A fmt, [10,13) B fmt, [15]/[16] A/B transpose (0 = K-major),
// [17,23) N>>3, [24,29) M>>4. Block-scaled kinds add [4,6) SFB id, [23] scale
// format, [29,31) SFA id.
__host__ __device__ constexpr uint32_t make_idesc(int kind, int M, int N) {
  uint32_t d = 0;
  switch (kind) {
    case KIND_F16: d = (1u << 4) | (1u << 7) | (1u << 10); break;    // F32 <- BF16 x BF16
    case KIND_TF32: d = (1u << 4) | (2u << 7) | (2u << 10); break;   // F32 <- TF32 x TF32
    case KIND_F8F6F4: d = (1u << 4); break;                          // F32 <- E4M3 x E4M3
    case KIND_I8: d = (2u << 4) | (1u << 7) | (1u << 10); break;     // S32 <- S8 x S8
    case KIND_MXF8: d = (1u << 23); break;                           // E4M3 x E4M3, UE8M0 scales
    case KIND_NVFP4: d = (1u << 7) | (1u << 10); break;              // E2M1 x E2M1, UE4M3 scales
    case KIND_MXF4: d = (1u << 7) | (1u << 10) | (1u << 23); break;  // E2M1 x E2M1, UE8M0 scales
    // sm_103 "ultra" FP4: bit 31 (k_size) selects K = 96 instead of 64.
    case KIND_NVFP4_K96: d = (1u << 7) | (1u << 10) | (1u << 31); break;
    case KIND_MXF4_K96: d = (1u << 7) | (1u << 10) | (1u << 23) | (1u << 31); break;
  }
  return d | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

__device__ __forceinline__ void mbar_init(uint32_t mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  uint32_t done;
  do {
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                 : "=r"(done)
                 : "r"(mbar), "r"(phase)
                 : "memory");
  } while (!done);
}

#define UMMA_PLAIN(CG, KS)                                                                                   \
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::" CG ".kind::" KS " [%0], %1, %2, %3, p;}" ::"r"( \
                   tmem_d),                                                                                    \
               "l"(ad), "l"(bd), "r"(idesc), "r"(accum))
#define UMMA_SCALED(CG, KS)                                                                                    \
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::" CG ".kind::" KS                   \
               " [%0], %1, %2, %3, [%5], [%6], p;}" ::"r"(tmem_d),                                             \
               "l"(ad), "l"(bd), "r"(idesc), "r"(accum), "r"(tsfa), "r"(tsfb))

// A operand read from TMEM instead of shared memory ("ts" form).
#define UMMA_TS_PLAIN(KS)                                                                                     \
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::" KS " [%0], [%1], %2, %3, p;}" ::"r"( \
                   tmem_d),                                                                                   \
               "r"(ta), "l"(bd), "r"(idesc), "r"(accum))
#define UMMA_TS_SCALED(KS)                                                                                    \
  asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::" KS                      \
               " [%0], [%1], %2, %3, [%5], [%6], p;}" ::"r"(tmem_d),                                          \
               "r"(ta), "l"(bd), "r"(idesc), "r"(accum), "r"(tsfa), "r"(tsfb))

// kind::i8 exists on sm_100a but ptxas rejects it for sm_103a (B300 has no
// INT8 UMMA), so it is not built here.
template <int KIND, int CG>
__device__ __forceinline__ void umma(uint32_t tmem_d, uint64_t ad, uint64_t bd, uint32_t idesc, uint32_t accum,
                                     uint32_t tsfa, uint32_t tsfb) {
#ifdef HAS_TCGEN05
  if constexpr (CG == 3) {  // CG 3 = one SM, A from TMEM
    uint32_t ta = tmem_d + 256;
    if constexpr (KIND == KIND_F16) UMMA_TS_PLAIN("f16");
    else if constexpr (KIND == KIND_F8F6F4) UMMA_TS_PLAIN("f8f6f4");
    else if constexpr (KIND == KIND_NVFP4) UMMA_TS_SCALED("mxf4nvf4.block_scale.scale_vec::4X");
  } else if constexpr (CG == 1) {
    if constexpr (KIND == KIND_F16) UMMA_PLAIN("1", "f16");
    else if constexpr (KIND == KIND_TF32) UMMA_PLAIN("1", "tf32");
    else if constexpr (KIND == KIND_F8F6F4) UMMA_PLAIN("1", "f8f6f4");
    else if constexpr (KIND == KIND_MXF8) UMMA_SCALED("1", "mxf8f6f4.block_scale.scale_vec::1X");
    else if constexpr (KIND == KIND_NVFP4) UMMA_SCALED("1", "mxf4nvf4.block_scale.scale_vec::4X");
    else if constexpr (KIND == KIND_MXF4) UMMA_SCALED("1", "mxf4.block_scale.scale_vec::2X");
    else if constexpr (KIND == KIND_NVFP4_K96) UMMA_SCALED("1", "mxf4nvf4.block_scale.block16");
    else if constexpr (KIND == KIND_MXF4_K96) UMMA_SCALED("1", "mxf4nvf4.block_scale.block32");
  } else {
    if constexpr (KIND == KIND_F16) UMMA_PLAIN("2", "f16");
    else if constexpr (KIND == KIND_TF32) UMMA_PLAIN("2", "tf32");
    else if constexpr (KIND == KIND_F8F6F4) UMMA_PLAIN("2", "f8f6f4");
    else if constexpr (KIND == KIND_MXF8) UMMA_SCALED("2", "mxf8f6f4.block_scale.scale_vec::1X");
    else if constexpr (KIND == KIND_NVFP4) UMMA_SCALED("2", "mxf4nvf4.block_scale.scale_vec::4X");
    else if constexpr (KIND == KIND_NVFP4_K96) UMMA_SCALED("2", "mxf4nvf4.block_scale.block16");
  }
#endif
}

template <int CG>
__device__ __forceinline__ void umma_commit(uint32_t mbar) {
#ifdef HAS_TCGEN05
  if constexpr (CG != 2)
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(mbar) : "memory");
  else  // signal the mbarrier at the same offset in both CTAs of the pair
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(mbar),
                 "h"((uint16_t)0x3)
                 : "memory");
#endif
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

