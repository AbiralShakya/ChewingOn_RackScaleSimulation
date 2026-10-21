// Tensor cores: warp-level mma.sync (HMMA/QMMA/IMMA) and, on sm_100+, the
// asynchronous tcgen05 path (UMMA issued by one thread, accumulator in TMEM).
//
// Each case runs in its own process (`bench_tc <case>`; `bench_tc --list`
// prints the cases) so that a descriptor the hardware rejects cannot take the
// other measurements down with it.
//
// tcgen05 operands live in shared memory with no swizzle (K-major, 8x16 B core
// matrices); values are arbitrary, so only timing is meaningful here.
#include <cstring>

#include "common.cuh"

// ---------------------------------------------------------------------------
// mma.sync
// ---------------------------------------------------------------------------
struct MmaBF16 {  // m16n8k16 bf16 -> f32
  static constexpr const char* name = "mma_sync_m16n8k16_bf16_f32";
  static constexpr int M = 16, N = 8, K = 16, NA = 4, NB = 2, NC = 4;
  __device__ static void mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};
struct MmaF16F16 {  // m16n8k16 f16 -> f16 accum
  static constexpr const char* name = "mma_sync_m16n8k16_f16_f16";
  static constexpr int M = 16, N = 8, K = 16, NA = 4, NB = 2, NC = 2;
  __device__ static void mma(float* c, const uint32_t* a, const uint32_t* b) {
    uint32_t* ci = reinterpret_cast<uint32_t*>(c);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(ci[0]), "+r"(ci[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};
struct MmaE4M3 {  // m16n8k32 e4m3 -> f32
  static constexpr const char* name = "mma_sync_m16n8k32_e4m3_f32";
  static constexpr int M = 16, N = 8, K = 32, NA = 4, NB = 2, NC = 4;
  __device__ static void mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};
struct MmaTF32 {  // m16n8k8 tf32 -> f32
  static constexpr const char* name = "mma_sync_m16n8k8_tf32_f32";
  static constexpr int M = 16, N = 8, K = 8, NA = 4, NB = 2, NC = 4;
  __device__ static void mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};
struct MmaS8 {  // m16n8k32 s8 -> s32
  static constexpr const char* name = "mma_sync_m16n8k32_s8_s32";
  static constexpr int M = 16, N = 8, K = 32, NA = 4, NB = 2, NC = 4;
  __device__ static void mma(float* c, const uint32_t* a, const uint32_t* b) {
    uint32_t* ci = reinterpret_cast<uint32_t*>(c);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(ci[0]), "+r"(ci[1]), "+r"(ci[2]), "+r"(ci[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};

template <class Op, int ILP>
__global__ void mma_sync_kernel(int iters, uint64_t* out, float* sink) {
  extern __shared__ uint8_t pad[];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  uint32_t a[Op::NA], b[Op::NB];
  float c[ILP][4];
#pragma unroll
  for (int i = 0; i < Op::NA; ++i) a[i] = 0x3c003c00u ^ (threadIdx.x * (i + 1));
#pragma unroll
  for (int i = 0; i < Op::NB; ++i) b[i] = 0x3c003c00u ^ (threadIdx.x * (i + 3));
#pragma unroll
  for (int k = 0; k < ILP; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[k][i] = 0.f;
  __syncthreads();
  uint64_t t0 = cta_timer_start(&tm);
  a[0] += (uint32_t)(t0 >> 63);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < 16; ++u)
#pragma unroll
      for (int k = 0; k < ILP; ++k) Op::mma(c[k], a, b);
  }
  float acc = 0;
#pragma unroll
  for (int k = 0; k < ILP; ++k) acc += c[k][0] + c[k][1] + c[k][2] + c[k][3];
  if (acc == 1234.5f) sink[0] = acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) out[blockIdx.x] = span;
  if (iters < 0) pad[0] = 0;
}

template <class Op>
void run_mma_sync(Json& j, const DeviceInfo& d, size_t one_cta_smem) {
  uint64_t* out;
  float* sink;
  CK(cudaMalloc(&out, 8 * d.sm_count));
  CK(cudaMalloc(&sink, 64));
  auto span_of = [&](auto kern, int blocks, int threads, int iters) {
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)one_cta_smem));
    std::vector<double> best;
    for (int r = 0; r < 3; ++r) {
      kern<<<cal_grid(blocks), threads, one_cta_smem>>>(iters, out, sink);
      CK(cudaDeviceSynchronize());
      std::vector<uint64_t> h(blocks);
      CK(cudaMemcpy(h.data(), out, 8 * blocks, cudaMemcpyDeviceToHost));
      best.push_back(median(std::vector<double>(h.begin(), h.end())));
    }
    return vmin(best);
  };
  // latency: one warp, one dependent chain (two lengths, slope)
  double l1 = span_of(mma_sync_kernel<Op, 1>, 1, 32, 8);
  double l2 = span_of(mma_sync_kernel<Op, 1>, 1, 32, 40);
  double lat = (l2 - l1) / (32.0 * 16);
  // throughput: 16 warps per SM, 4 independent accumulators each
  const int iters = 64, threads = 512, ilp = 4;
  double span = span_of(mma_sync_kernel<Op, 4>, d.sm_count, threads, iters);
  double mmas = (threads / 32.0) * iters * 16 * ilp;
  double flop_per_mma = 2.0 * Op::M * Op::N * Op::K;
  j.key(Op::name);
  j.obj();
  j.kv("latency_cycles", lat);
  j.kv("warp_mma_per_clk_per_sm", mmas / span);
  j.kv("flop_per_clk_per_sm", mmas * flop_per_mma / span);
  j.end_obj();
  fprintf(stderr, "%-30s lat %5.1f cyc  %7.1f FLOP/clk/SM\n", Op::name, lat, mmas * flop_per_mma / span);
  cudaFree(out);
  cudaFree(sink);
}

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

// CG = 1: one SM, M <= 128.  CG = 2: a CTA pair on two SMs (cluster of 2)
// computes M = 256; each CTA holds half of A and half of B in its own smem and
// half of D in its own TMEM, and the even CTA issues for both.
// mode 0: throughput (iters back-to-back MMAs, one commit at the end)
// mode 1: latency (each MMA committed and waited for before the next)
template <int KIND, int CG>
__global__ void tc05_kernel(int M, int N, int iters, int mode, uint64_t* out) {
#ifdef HAS_TCGEN05
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint32_t tmem_slot;
  __shared__ __align__(8) uint64_t mbar_storage;
  uint8_t* A = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  uint8_t* B = A + 16 * 1024;  // B spans up to 32 x 384 B = 12 KB
  uint32_t mbar = smem_u32(&mbar_storage);
  const int warp = threadIdx.x / 32;
  constexpr int G = CG == 2 ? 2 : 1;  // CTAs per MMA; CG 3 is the one-SM A-from-TMEM variant
  const bool leader = G == 1 || cluster_ctarank() == 0;
  if (warp == 0) {
    if constexpr (G == 1) {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "r"(512));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    } else {
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "r"(512));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
  }
  if (threadIdx.x == 0) mbar_init(mbar, 1);
  for (int i = threadIdx.x; i < 28 * 1024 / 4; i += blockDim.x) ((uint32_t*)A)[i] = 0x38383838u ^ i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // make generic-proxy writes visible to UMMA
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  if constexpr (G == 1) __syncthreads();
  else cluster_sync();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint32_t tmem = tmem_slot;
  if (threadIdx.x == 0) {
    uint64_t t0 = clk64();
    if (leader) {
      // K-major, no swizzle: core matrix (8 rows x 16 B) at (mi, ki) -> mi*384 + ki*128
      // (room for the three 16 B K-slices of a K=96 FP4 row).
      uint64_t ad = smem_desc(smem_u32(A), 128, 384, 0);
      uint64_t bd = smem_desc(smem_u32(B), 128, 384, 0);
      uint32_t idesc = make_idesc(KIND, M, N);
      uint32_t tsfa = tmem + 256, tsfb = tmem + 384;  // scale factors (block-scaled kinds only)
      uint32_t phase = 0;
      if (mode == 0) {
        for (int i = 0; i < iters; ++i) umma<KIND, CG>(tmem, ad, bd, idesc, i > 0, tsfa, tsfb);
        umma_commit<CG>(mbar);
        mbar_wait(mbar, 0);
      } else {
        for (int i = 0; i < iters; ++i) {
          umma<KIND, CG>(tmem, ad, bd, idesc, i > 0, tsfa, tsfb);
          umma_commit<CG>(mbar);
          mbar_wait(mbar, phase);
          phase ^= 1;
        }
      }
    } else {
      // The peer only waits for every commit to land on its own mbarrier.
      uint32_t phase = 0;
      for (int i = 0; i < (mode == 0 ? 1 : iters); ++i, phase ^= 1) mbar_wait(mbar, phase);
    }
    uint64_t t1 = clk64();
    out[2 * blockIdx.x] = t1 - t0;
    out[2 * blockIdx.x + 1] = smid();
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  if constexpr (G == 1) __syncthreads();
  else cluster_sync();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (warp == 0) {
    if constexpr (G == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(512));
    else asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(512));
  }
#endif
}

// TMEM -> registers: tcgen05.ld 32x32b.x32 (32 lanes x 32 columns per warp).
__global__ void tmem_ld_kernel(int iters, uint64_t* out, float* sink) {
#ifdef HAS_TCGEN05
  extern __shared__ uint8_t pad[];
  __shared__ uint32_t tmem_slot;
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  const int warp = threadIdx.x / 32;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_slot)), "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  // Warp w may only touch lanes 32*(w%4) .. +31.
  uint32_t taddr = tmem_slot + ((uint32_t)(32 * (warp % 4)) << 16) + 32 * (warp / 4 % 4);
  float acc = 0;
  uint64_t t0 = cta_timer_start(&tm);
  taddr += (uint32_t)(t0 >> 63);
  for (int it = 0; it < iters; ++it) {
    uint32_t r[32];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),
          "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]),
          "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
          "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int k = 0; k < 32; ++k) acc += __uint_as_float(r[k]);
    taddr ^= (uint32_t)(acc == 1e30f);  // serialise iterations on the loaded data
  }
  if (acc == 1234.5f) sink[0] = acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) out[blockIdx.x] = span;
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_slot), "r"(512));
  if (iters < 0) pad[0] = 0;
#endif
}

struct Tc05Case {
  const char* name;
  int kind, M, N, K;
  int cg = 1;
};

static const Tc05Case TC05_CASES[] = {
    {"tcgen05_f16_bf16_m128n64k16", KIND_F16, 128, 64, 16},
    {"tcgen05_f16_bf16_m128n128k16", KIND_F16, 128, 128, 16},
    {"tcgen05_f16_bf16_m128n256k16", KIND_F16, 128, 256, 16},
    {"tcgen05_f16_bf16_m64n256k16", KIND_F16, 64, 256, 16},
    {"tcgen05_tf32_m128n256k8", KIND_TF32, 128, 256, 8},
    {"tcgen05_f8f6f4_e4m3_m128n256k32", KIND_F8F6F4, 128, 256, 32},
    {"tcgen05_mxf8_e4m3_m128n256k32", KIND_MXF8, 128, 256, 32},
    {"tcgen05_nvfp4_m128n256k64", KIND_NVFP4, 128, 256, 64},
    {"tcgen05_mxf4_m128n256k64", KIND_MXF4, 128, 256, 64},
    {"tcgen05_nvfp4_k96_m128n256k96", KIND_NVFP4_K96, 128, 256, 96},
    {"tcgen05_mxf4_k96_m128n256k96", KIND_MXF4_K96, 128, 256, 96},
    // 2-SM UMMA (cta_group::2): M = 256 across a CTA pair
    {"tcgen05x2_f16_bf16_m256n256k16", KIND_F16, 256, 256, 16, 2},
    {"tcgen05x2_f8f6f4_e4m3_m256n256k32", KIND_F8F6F4, 256, 256, 32, 2},
    {"tcgen05x2_mxf8_e4m3_m256n256k32", KIND_MXF8, 256, 256, 32, 2},
    {"tcgen05x2_nvfp4_m256n256k64", KIND_NVFP4, 256, 256, 64, 2},
    {"tcgen05x2_nvfp4_k96_m256n256k96", KIND_NVFP4_K96, 256, 256, 96, 2},
    // A operand from TMEM ("ts"): halves the smem operand traffic per MMA
    {"tcgen05ts_f16_bf16_m128n256k16", KIND_F16, 128, 256, 16, 3},
    {"tcgen05ts_f8f6f4_e4m3_m128n256k32", KIND_F8F6F4, 128, 256, 32, 3},
    {"tcgen05ts_nvfp4_m128n256k64", KIND_NVFP4, 128, 256, 64, 3},
};

template <int KIND, int CG>
void run_tc05(Json& j, const DeviceInfo& d, const Tc05Case& c, size_t one_cta_smem) {
  uint64_t* out;
  CK(cudaMalloc(&out, 16 * d.sm_count));
  constexpr int G = CG == 2 ? 2 : 1;  // CTAs per MMA
  auto kern = tc05_kernel<KIND, CG>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)one_cta_smem));
  if (CG == 2) CK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  auto run = [&](int blocks, int iters, int mode) {
    std::vector<double> best;
    for (int r = 0; r < 3; ++r) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(blocks);
      cfg.blockDim = dim3(128);
      cfg.dynamicSmemBytes = one_cta_smem;
      cudaLaunchAttribute attr[1];
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = G;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      cfg.attrs = attr;
      cfg.numAttrs = 1;
      CK(cudaLaunchKernelEx(&cfg, kern, c.M, c.N, iters, mode, out));
      CK(cudaDeviceSynchronize());
      std::vector<uint64_t> h(2 * blocks);
      CK(cudaMemcpy(h.data(), out, 16 * blocks, cudaMemcpyDeviceToHost));
      std::vector<double> v;
      for (int b = 0; b < blocks; b += G) v.push_back((double)h[2 * b]);  // issuing CTAs
      best.push_back(median(v));
    }
    return vmin(best);
  };
  // FLOP per SM: a 2-SM MMA's work is split over two SMs.
  double flop = 2.0 * c.M * c.N * c.K / G;
  // latency: slope of (MMA + commit + mbarrier wait) round trips
  double lat = (run(G, 64, 1) - run(G, 16, 1)) / 48.0;
  // single-SM (or single-pair) and all-SM throughput: slope over MMA count
  double s1 = (run(G, 4096, 0) - run(G, 1024, 0)) / 3072.0;
  int all = d.sm_count / G * G;
  double sa = (run(all, 4096, 0) - run(all, 1024, 0)) / 3072.0;
  j.key(c.name);
  j.obj();
  j.kvi("M", c.M);
  j.kvi("N", c.N);
  j.kvi("K", c.K);
  j.kvi("cta_group", G);
  j.kvb("a_from_tmem", CG == 3);
  j.kv("roundtrip_latency_cycles", lat);
  j.kv("cycles_per_mma_1sm", s1);
  j.kv("cycles_per_mma_allsm", sa);
  j.kv("flop_per_clk_per_sm_1sm", flop / s1);
  j.kv("flop_per_clk_per_sm_allsm", flop / sa);
  j.end_obj();
  fprintf(stderr, "%-34s rt-lat %6.1f cyc  %6.1f cyc/MMA (1 SM) %6.1f (all)  %8.0f FLOP/clk/SM\n", c.name, lat, s1, sa, flop / sa);
  cudaFree(out);
}

void run_tmem(Json& j, const DeviceInfo& d, size_t one_cta_smem) {
  uint64_t* out;
  float* sink;
  CK(cudaMalloc(&out, 8 * d.sm_count));
  CK(cudaMalloc(&sink, 64));
  CK(cudaFuncSetAttribute(tmem_ld_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)one_cta_smem));
  auto span = [&](int threads, int iters) {
    std::vector<double> v;
    for (int r = 0; r < 3; ++r) {
      tmem_ld_kernel<<<1, threads, one_cta_smem>>>(iters, out, sink);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost));
      v.push_back((double)c);
    }
    return vmin(v);
  };
  double lat = (span(32, 256) - span(32, 64)) / 192.0;
  double t4 = (span(128, 256) - span(128, 64)) / 192.0;    // 4 warps, one per lane quadrant
  double t16 = (span(512, 256) - span(512, 64)) / 192.0;   // 16 warps
  j.key("tmem_ld_32x32b_x32");
  j.obj();
  j.kv("latency_cycles_ld_plus_wait", lat);
  j.kv("bytes_per_clk_4warps", 4 * 4096.0 / t4);
  j.kv("bytes_per_clk_16warps", 16 * 4096.0 / t16);
  j.end_obj();
  fprintf(stderr, "tcgen05.ld 32x32b.x32: lat %.1f cyc, %.0f B/clk (4 warps), %.0f B/clk (16 warps)\n", lat, 4 * 4096.0 / t4,
          16 * 4096.0 / t16);
  cudaFree(out);
  cudaFree(sink);
}

int main(int argc, char** argv) {
  DeviceInfo d = device_info();
  std::vector<std::string> cases = {"mma_sync"};
  bool tc05 = d.p.major >= 10 && d.p.major < 12;
  if (tc05) {
    cases.push_back("tmem");
    for (auto& c : TC05_CASES) cases.push_back(c.name);
  }
  if (argc > 1 && !strcmp(argv[1], "--list")) {
    for (auto& c : cases) printf("%s\n", c.c_str());
    return 0;
  }
  std::string want = argc > 1 ? argv[1] : "mma_sync";
  size_t one_cta_smem = d.p.sharedMemPerMultiprocessor / 2 + 4096;
  if (one_cta_smem > d.p.sharedMemPerBlockOptin) one_cta_smem = d.p.sharedMemPerBlockOptin;
  one_cta_smem = std::max<size_t>(one_cta_smem, 33 * 1024);
  Json j;
  j.obj();
  emit_header(j, ("tc:" + want).c_str());
  j.kv("sm_clock_mhz_measured", measure_sm_clock_mhz());
  j.key("tc");
  j.obj();
  if (want == "mma_sync") {
    run_mma_sync<MmaBF16>(j, d, one_cta_smem);
    run_mma_sync<MmaF16F16>(j, d, one_cta_smem);
    run_mma_sync<MmaTF32>(j, d, one_cta_smem);
    // e4m3 mma.sync is not timed: on sm_103a ptxas lowers it to conversions
    // plus HMMA.F16 and leaves too few MMAs in the loop for a clean number.
    run_mma_sync<MmaS8>(j, d, one_cta_smem);
  } else if (want == "tmem") {
    run_tmem(j, d, one_cta_smem);
  } else {
    bool found = false;
    for (auto& c : TC05_CASES) {
      if (want != c.name) continue;
      found = true;
      switch (c.kind * 4 + c.cg) {
        case KIND_F16 * 4 + 1: run_tc05<KIND_F16, 1>(j, d, c, one_cta_smem); break;
        case KIND_TF32 * 4 + 1: run_tc05<KIND_TF32, 1>(j, d, c, one_cta_smem); break;
        case KIND_F8F6F4 * 4 + 1: run_tc05<KIND_F8F6F4, 1>(j, d, c, one_cta_smem); break;
        case KIND_MXF8 * 4 + 1: run_tc05<KIND_MXF8, 1>(j, d, c, one_cta_smem); break;
        case KIND_NVFP4 * 4 + 1: run_tc05<KIND_NVFP4, 1>(j, d, c, one_cta_smem); break;
        case KIND_MXF4 * 4 + 1: run_tc05<KIND_MXF4, 1>(j, d, c, one_cta_smem); break;
        case KIND_NVFP4_K96 * 4 + 1: run_tc05<KIND_NVFP4_K96, 1>(j, d, c, one_cta_smem); break;
        case KIND_MXF4_K96 * 4 + 1: run_tc05<KIND_MXF4_K96, 1>(j, d, c, one_cta_smem); break;
        case KIND_NVFP4_K96 * 4 + 2: run_tc05<KIND_NVFP4_K96, 2>(j, d, c, one_cta_smem); break;
        case KIND_F16 * 4 + 2: run_tc05<KIND_F16, 2>(j, d, c, one_cta_smem); break;
        case KIND_F8F6F4 * 4 + 2: run_tc05<KIND_F8F6F4, 2>(j, d, c, one_cta_smem); break;
        case KIND_MXF8 * 4 + 2: run_tc05<KIND_MXF8, 2>(j, d, c, one_cta_smem); break;
        case KIND_NVFP4 * 4 + 2: run_tc05<KIND_NVFP4, 2>(j, d, c, one_cta_smem); break;
        case KIND_F16 * 4 + 3: run_tc05<KIND_F16, 3>(j, d, c, one_cta_smem); break;
        case KIND_F8F6F4 * 4 + 3: run_tc05<KIND_F8F6F4, 3>(j, d, c, one_cta_smem); break;
        case KIND_NVFP4 * 4 + 3: run_tc05<KIND_NVFP4, 3>(j, d, c, one_cta_smem); break;
      }
    }
    if (!found) {
      fprintf(stderr, "unknown case %s\n", want.c_str());
      return 2;
    }
  }
  j.end_obj();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
