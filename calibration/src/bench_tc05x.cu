// tcgen05 in depth (sm_100/sm_103): the parts of the Blackwell tensor-core
// path a cycle-level model needs beyond peak MMA rates.
//
//   tmem_ld / tmem_st   every tcgen05.ld/st shape and repeat count: latency of
//                       ld+wait::ld (st+wait::st) for one warp, and bytes/clk
//                       per SM with 4 and 16 warps
//   umma_pipeline       k back-to-back MMAs then one commit: time from the
//                       first issue to the mbarrier arrival vs k (pipeline
//                       fill latency and per-MMA time), and when each issue
//                       retires on the issuing thread (issue queue depth)
//   tcgen05_cp          smem -> TMEM copies: latency (cp + commit + wait) and
//                       throughput
//   tmem_alloc          tcgen05.alloc / dealloc latency vs column count
//   overlap             MMA stream on one TMEM region while 4 warps read
//                       another with tcgen05.ld (the FlashAttention-4 pattern)
#include <cstring>

#include "common.cuh"
#include "tc05.cuh"
#include "tmem_ops.cuh"

#define FENCE_SYNC()                                                   \
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");      \
  __syncthreads();                                                     \
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory")

__device__ __forceinline__ uint32_t tmem_alloc_512(uint32_t* slot) {
#ifdef HAS_TCGEN05
  if (threadIdx.x / 32 == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(slot)), "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
#endif
  FENCE_SYNC();
  return *slot;
}

__device__ __forceinline__ void tmem_free_512(uint32_t t) {
  FENCE_SYNC();
#ifdef HAS_TCGEN05
  if (threadIdx.x / 32 == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(t), "r"(512));
#endif
}

// ---------------------------------------------------------------------------
// TMEM load/store, one kernel per form. mode 0: ld + wait::ld per iteration;
// mode 1: st + wait::st; mode 2: 4 lds then one wait (forms with <= 32 regs).
// ---------------------------------------------------------------------------
#define TMEM_KERNEL(SHAPE, PER, N)                                                          \
  __global__ void tmem_k_##SHAPE##_x##N(int mode, int iters, uint64_t* out, uint32_t* sink) { \
    __shared__ uint32_t slot;                                                               \
    __shared__ CtaTimer tm;                                                                 \
    cta_timer_init(&tm);                                                                    \
    uint32_t t = tmem_alloc_512(&slot);                                                     \
    const int warp = threadIdx.x / 32;                                                      \
    uint32_t taddr = t + ((uint32_t)(32 * (warp % 4)) << 16);                               \
    constexpr int K = PER * N;                                                              \
    uint32_t r[K];                                                                          \
    _Pragma("unroll") for (int i = 0; i < K; i++) r[i] = threadIdx.x * 7 + i;               \
    uint32_t acc = 0;                                                                       \
    uint64_t t0 = cta_timer_start(&tm);                                                     \
    taddr += (uint32_t)(t0 >> 63);                                                          \
    if (mode == 0) {                                                                        \
      for (int it = 0; it < iters; ++it) {                                                  \
        tmem_ld_##SHAPE##_x##N(taddr, r);                                                   \
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");                       \
        acc += r[0] + r[K - 1];                                                             \
        taddr ^= (uint32_t)(acc == 0xdeadbeefu);                                            \
      }                                                                                     \
    } else if (mode == 1) {                                                                 \
      for (int it = 0; it < iters; ++it) {                                                  \
        r[0] += acc;                                                                        \
        tmem_st_##SHAPE##_x##N(taddr, r);                                                   \
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");                       \
        acc += it;                                                                          \
      }                                                                                     \
    } else if (K <= 32) {                                                                   \
      uint32_t q[3 * K + 1];                                                                \
      for (int it = 0; it < iters; it += 4) {                                               \
        tmem_ld_##SHAPE##_x##N(taddr, r);                                                   \
        tmem_ld_##SHAPE##_x##N(taddr, q);                                                   \
        tmem_ld_##SHAPE##_x##N(taddr, q + K);                                               \
        tmem_ld_##SHAPE##_x##N(taddr, q + 2 * K);                                           \
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");                       \
        acc += r[0] + q[0] + q[K] + q[2 * K];                                               \
        taddr ^= (uint32_t)(acc == 0xdeadbeefu);                                            \
      }                                                                                     \
    }                                                                                       \
    __syncthreads();                                                                        \
    uint64_t span = cta_timer_stop(&tm);                                                    \
    if (acc == 0x12345678u) sink[0] = acc;                                                  \
    if (threadIdx.x == 0) out[0] = span;                                                    \
    tmem_free_512(t);                                                                       \
  }
TMEM_FORMS(TMEM_KERNEL)

typedef void (*TmemKernel)(int, int, uint64_t*, uint32_t*);
struct TmemForm {
  const char* shape;
  int per, n;
  TmemKernel k;
};
#define TMEM_ENTRY(SHAPE, PER, N) {#SHAPE, PER, N, tmem_k_##SHAPE##_x##N},
static const TmemForm TMEM_TABLE[] = {TMEM_FORMS(TMEM_ENTRY)};

// ---------------------------------------------------------------------------
// UMMA pipeline: thread 0 issues k MMAs (BF16 / FP8 / NVFP4, M = 128) back to
// back, reading %clock64 after each issue, then commits and waits.
// out[0..k) issue-return times, out[k] arrival time, all relative to the
// clock read just before the first issue.
// ---------------------------------------------------------------------------
template <int KIND>
__global__ void umma_pipe_kernel(int N, int k, int reps, uint64_t* out) {
#ifdef HAS_TCGEN05
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint32_t slot;
  __shared__ __align__(8) uint64_t mbar_storage;
  __shared__ uint64_t ts[129];
  uint8_t* A = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  uint8_t* B = A + 16 * 1024;
  uint32_t mbar = smem_u32(&mbar_storage);
  for (int i = threadIdx.x; i < 28 * 1024 / 4; i += blockDim.x) ((uint32_t*)A)[i] = 0x38383838u ^ i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (threadIdx.x == 0) mbar_init(mbar, 1);
  uint32_t t = tmem_alloc_512(&slot);
  if (threadIdx.x == 0) {
    uint64_t ad = smem_desc(smem_u32(A), 128, 384, 0);
    uint64_t bd = smem_desc(smem_u32(B), 128, 384, 0);
    uint32_t idesc = make_idesc(KIND, 128, N);
    uint32_t phase = 0;
    for (int rep = 0; rep < reps; ++rep) {
      uint64_t t0 = clk64();
      for (int i = 0; i < k; ++i) {
        umma<KIND, 1>(t, ad, bd, idesc, i > 0, t + 256, t + 384);
        ts[i] = clk64();
      }
      umma_commit<1>(mbar);
      uint64_t tc = clk64();
      mbar_wait(mbar, phase);
      uint64_t t1 = clk64();
      phase ^= 1;
      if (rep == reps - 1) {
        for (int i = 0; i < k; ++i) out[i] = ts[i] - t0;
        out[k] = t1 - t0;
        out[k + 1] = tc - t0;
      }
    }
  }
  tmem_free_512(t);
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.cp smem -> TMEM. mode 0: latency (cp, commit, wait) per iteration;
// mode 1: iters copies then one commit.
// ---------------------------------------------------------------------------
template <int SHAPE>  // 0 = 128x256b, 1 = 128x128b, 2 = 4x256b
__global__ void tc_cp_kernel(int mode, int iters, uint64_t* out) {
#ifdef HAS_TCGEN05
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint32_t slot;
  __shared__ __align__(8) uint64_t mbar_storage;
  uint8_t* S = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  uint32_t mbar = smem_u32(&mbar_storage);
  for (int i = threadIdx.x; i < 16 * 1024 / 4; i += blockDim.x) ((uint32_t*)S)[i] = i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (threadIdx.x == 0) mbar_init(mbar, 1);
  uint32_t t = tmem_alloc_512(&slot);
  if (threadIdx.x == 0) {
    uint64_t sd = smem_desc(smem_u32(S), 128, 256, 0);
    uint32_t phase = 0;
    uint64_t t0 = clk64();
    for (int i = 0; i < iters; ++i) {
      uint32_t dst = t + 8 * (i & 31);
      if constexpr (SHAPE == 0) asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" ::"r"(dst), "l"(sd));
      else if constexpr (SHAPE == 1) asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" ::"r"(dst), "l"(sd));
      else asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;" ::"r"(dst), "l"(sd));
      if (mode == 0) {
        umma_commit<1>(mbar);
        mbar_wait(mbar, phase);
        phase ^= 1;
      }
    }
    if (mode == 1) {
      umma_commit<1>(mbar);
      mbar_wait(mbar, 0);
    }
    out[0] = clk64() - t0;
  }
  tmem_free_512(t);
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.alloc / dealloc latency (warp 0), ncols columns, reps times.
// ---------------------------------------------------------------------------
__global__ void alloc_kernel(int ncols, int reps, uint64_t* out) {
#ifdef HAS_TCGEN05
  __shared__ uint32_t slot;
  if (threadIdx.x >= 32) return;
  uint64_t ta = 0, td = 0;
  for (int r = 0; r < reps; ++r) {
    uint64_t t0 = clk64();
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&slot)), "r"(ncols));
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncwarp();
    uint32_t t = *(volatile uint32_t*)&slot;
    uint64_t t1 = clk64();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(t), "r"(ncols));
    uint64_t t2 = clk64();
    if (r > 0) {
      ta += t1 - t0;
      td += t2 - t1;
    }
  }
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  if (threadIdx.x == 0) {
    out[0] = ta / (reps - 1);
    out[1] = td / (reps - 1);
  }
#endif
}

// ---------------------------------------------------------------------------
// Overlap: thread 0 streams `mmas` BF16 128x256x16 MMAs into columns 0..255;
// warps 4..7 (if ld) loop tcgen05.ld 32x32b.x32 + wait over columns 256..287.
// out[0] = MMA span (issue of first to commit arrival), out[1] = ld count
// summed over the 4 warps, out[2] = ld span.
// ---------------------------------------------------------------------------
__global__ void overlap_kernel(int mmas, int ld, int ld_iters, uint64_t* out) {
#ifdef HAS_TCGEN05
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint32_t slot;
  __shared__ __align__(8) uint64_t mbar_storage;
  __shared__ unsigned long long ld_t0, ld_t1;
  uint8_t* A = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  uint8_t* B = A + 16 * 1024;
  uint32_t mbar = smem_u32(&mbar_storage);
  for (int i = threadIdx.x; i < 28 * 1024 / 4; i += blockDim.x) ((uint32_t*)A)[i] = 0x38383838u ^ i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (threadIdx.x == 0) {
    mbar_init(mbar, 1);
    ld_t0 = ~0ull;
    ld_t1 = 0;
  }
  uint32_t t = tmem_alloc_512(&slot);
  const int warp = threadIdx.x / 32;
  if (threadIdx.x == 0 && mmas > 0) {
    uint64_t ad = smem_desc(smem_u32(A), 128, 384, 0);
    uint64_t bd = smem_desc(smem_u32(B), 128, 384, 0);
    uint32_t idesc = make_idesc(KIND_F16, 128, 256);
    uint64_t t0 = clk64();
    for (int i = 0; i < mmas; ++i) umma<KIND_F16, 1>(t, ad, bd, idesc, i > 0, 0, 0);
    umma_commit<1>(mbar);
    mbar_wait(mbar, 0);
    out[0] = clk64() - t0;
  }
  if (warp >= 4 && ld) {
    uint32_t taddr = t + ((uint32_t)(32 * (warp % 4)) << 16) + 256;
    uint32_t r[32], acc = 0;
    uint64_t a = clk64();
    for (int it = 0; it < ld_iters; ++it) {
      tmem_ld_32x32b_x32(taddr, r);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      acc += r[0] + r[31];
      taddr ^= (uint32_t)(acc == 0xdeadbeefu);
    }
    uint64_t b = clk64();
    if ((threadIdx.x & 31) == 0) {
      atomicMin(&ld_t0, (unsigned long long)a);
      atomicMax(&ld_t1, (unsigned long long)b);
    }
    if (acc == 7) out[3] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) out[2] = ld ? ld_t1 - ld_t0 : 0;
  tmem_free_512(t);
#endif
}

// ---------------------------------------------------------------------------
// Shared-memory contention: thread 0 streams `mmas` BF16 UMMAs (M=128, N=n,
// A and B from smem: 128 B/clk of operand reads at N=128) while warps 4..7
// either idle (mode 0), store 16 B per thread to smem (mode 1) or load it
// (mode 2) in a loop over a separate 32 KB region. out[0] = MMA span,
// out[1] = smem bytes moved by the other warps, out[2] = their span.
// ---------------------------------------------------------------------------
__global__ void smem_contention_kernel(int mmas, int n, int mode, int iters, uint64_t* out) {
#ifdef HAS_TCGEN05
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint32_t slot;
  __shared__ __align__(8) uint64_t mbar_storage;
  __shared__ unsigned long long o_t0, o_t1;
  uint8_t* A = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  uint8_t* B = A + 16 * 1024;
  uint4* X = (uint4*)(A + 32 * 1024);  // 32 KB scratch for the other warps
  uint32_t mbar = smem_u32(&mbar_storage);
  for (int i = threadIdx.x; i < 64 * 1024 / 4; i += blockDim.x) ((uint32_t*)A)[i] = 0x38383838u ^ i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (threadIdx.x == 0) {
    mbar_init(mbar, 1);
    o_t0 = ~0ull;
    o_t1 = 0;
  }
  uint32_t t = tmem_alloc_512(&slot);
  const int warp = threadIdx.x / 32;
  if (threadIdx.x == 0) {
    uint64_t ad = smem_desc(smem_u32(A), 128, 384, 0);
    uint64_t bd = smem_desc(smem_u32(B), 128, 384, 0);
    uint32_t idesc = make_idesc(KIND_F16, 128, n);
    uint64_t t0 = clk64();
    for (int i = 0; i < mmas; ++i) umma<KIND_F16, 1>(t, ad, bd, idesc, i > 0, 0, 0);
    umma_commit<1>(mbar);
    mbar_wait(mbar, 0);
    out[0] = clk64() - t0;
  }
  if (warp >= 4 && mode) {
    uint4 v = make_uint4(threadIdx.x, 1, 2, 3), acc = make_uint4(0, 0, 0, 0);
    int idx = threadIdx.x - 128;
    uint64_t a = clk64();
    for (int it = 0; it < iters; ++it) {
      int k = (idx + it * 128) & 2047;
      if (mode == 1) {
        X[k] = v;
        v.x += 1;
      } else {
        uint4 r = X[k];
        acc.x ^= r.x;
        acc.y ^= r.w;
      }
    }
    __syncwarp();
    uint64_t b = clk64();
    if ((threadIdx.x & 31) == 0) {
      atomicMin(&o_t0, (unsigned long long)a);
      atomicMax(&o_t1, (unsigned long long)b);
    }
    if (acc.x == 12345) out[5] = acc.y;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    out[1] = mode ? 128ull * 16 * iters : 0;
    out[2] = mode ? o_t1 - o_t0 : 0;
  }
  tmem_free_512(t);
#endif
}

// ---------------------------------------------------------------------------
static double run_span(TmemKernel k, int threads, int mode, int iters, uint64_t* out, uint32_t* sink) {
  std::vector<double> v;
  for (int r = 0; r < 3; ++r) {
    k<<<1, threads>>>(mode, iters, out, sink);
    CK(cudaDeviceSynchronize());
    uint64_t c;
    CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost));
    v.push_back((double)c);
  }
  return vmin(v);
}

void run_tmem(Json& j) {
  uint64_t* out;
  uint32_t* sink;
  CK(cudaMalloc(&out, 64));
  CK(cudaMalloc(&sink, 64));
  for (const char* what : {"tmem_ld", "tmem_st"}) {
    bool st = what[5] == 's';
    j.key(what);
    j.obj();
    for (auto& f : TMEM_TABLE) {
      int bytes = f.per * f.n * 4 * 32;  // per warp per instruction
      int mode = st ? 1 : 0;
      double lat = (run_span(f.k, 32, mode, 256, out, sink) - run_span(f.k, 32, mode, 64, out, sink)) / 192.0;
      double t4 = (run_span(f.k, 128, mode, 256, out, sink) - run_span(f.k, 128, mode, 64, out, sink)) / 192.0;
      double t16 = (run_span(f.k, 512, mode, 256, out, sink) - run_span(f.k, 512, mode, 64, out, sink)) / 192.0;
      char name[64];
      snprintf(name, sizeof name, "%s.x%d", f.shape, f.n);
      j.key(name);
      j.obj();
      j.kvi("bytes_per_warp", bytes);
      j.kv("latency_cycles", lat);
      j.kv("bytes_per_clk_4warps", 4 * bytes / t4);
      j.kv("bytes_per_clk_16warps", 16 * bytes / t16);
      if (!st && f.per * f.n <= 32) {
        double p4 = (run_span(f.k, 128, 2, 256, out, sink) - run_span(f.k, 128, 2, 64, out, sink)) / 192.0;
        j.kv("bytes_per_clk_4warps_4inflight", 4 * bytes / p4);
      }
      j.end_obj();
      fprintf(stderr, "%s %-14s %5d B/warp  lat %6.1f  %6.0f B/clk (4w) %6.0f (16w)\n", what, name, bytes, lat,
              4 * bytes / t4, 16 * bytes / t16);
    }
    j.end_obj();
  }
  cudaFree(out);
  cudaFree(sink);
}

template <int KIND>
void run_pipe(Json& j, const char* name, int N, size_t smem) {
  uint64_t* out;
  CK(cudaMalloc(&out, 8 * 160));
  CK(cudaFuncSetAttribute(umma_pipe_kernel<KIND>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  j.key(name);
  j.obj();
  j.kvi("N", N);
  j.key("k");
  j.obj();
  for (int k : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}) {
    std::vector<uint64_t> best;
    for (int r = 0; r < 3; ++r) {
      umma_pipe_kernel<KIND><<<1, 128, smem>>>(N, k, 4, out);
      CK(cudaDeviceSynchronize());
      std::vector<uint64_t> h(k + 2);
      CK(cudaMemcpy(h.data(), out, 8 * (k + 2), cudaMemcpyDeviceToHost));
      if (best.empty() || h[k] < best[k]) best = h;
    }
    char key[16];
    snprintf(key, sizeof key, "%d", k);
    j.key(key);
    j.obj();
    j.kvi("arrival", (long long)best[k]);
    j.kvi("commit_issued", (long long)best[k + 1]);
    j.key("issue_returns");
    j.arr();
    for (int i = 0; i < k; ++i) j.integer((long long)best[i]);
    j.end_arr();
    j.end_obj();
    fprintf(stderr, "%s k=%3d arrival %6llu  last issue returns %6llu\n", name, k, (unsigned long long)best[k],
            (unsigned long long)best[k - 1]);
  }
  j.end_obj();
  j.end_obj();
  cudaFree(out);
}

template <int S>
void run_cp(Json& j, const char* name, size_t smem) {
  uint64_t* out;
  CK(cudaMalloc(&out, 64));
  CK(cudaFuncSetAttribute(tc_cp_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  auto t = [&](int mode, int iters) {
    std::vector<double> v;
    for (int r = 0; r < 3; ++r) {
      tc_cp_kernel<S><<<1, 128, smem>>>(mode, iters, out);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost));
      v.push_back((double)c);
    }
    return vmin(v);
  };
  double lat = (t(0, 64) - t(0, 16)) / 48.0;
  double per = (t(1, 256) - t(1, 64)) / 192.0;
  int bytes = S == 0 ? 128 * 32 : S == 1 ? 128 * 16 : 4 * 32;
  j.key(name);
  j.obj();
  j.kvi("bytes", bytes);
  j.kv("roundtrip_latency_cycles", lat);
  j.kv("cycles_per_cp", per);
  j.kv("bytes_per_clk", bytes / per);
  j.end_obj();
  fprintf(stderr, "%s: round trip %.1f cyc, %.1f cyc/cp, %.1f B/clk\n", name, lat, per, bytes / per);
  cudaFree(out);
}

void run_alloc(Json& j) {
  uint64_t* out;
  CK(cudaMalloc(&out, 64));
  j.key("tmem_alloc");
  j.obj();
  for (int n : {32, 64, 128, 256, 512}) {
    std::vector<double> a, d;
    for (int r = 0; r < 5; ++r) {
      alloc_kernel<<<1, 32>>>(n, 9, out);
      CK(cudaDeviceSynchronize());
      uint64_t h[2];
      CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost));
      a.push_back((double)h[0]);
      d.push_back((double)h[1]);
    }
    char key[16];
    snprintf(key, sizeof key, "%d", n);
    j.key(key);
    j.obj();
    j.kv("alloc_cycles", median(a));
    j.kv("dealloc_cycles", median(d));
    j.end_obj();
    fprintf(stderr, "tcgen05.alloc %3d cols: %.0f cyc, dealloc %.0f cyc\n", n, median(a), median(d));
  }
  j.end_obj();
  cudaFree(out);
}

void run_overlap(Json& j, size_t smem) {
  uint64_t* out;
  CK(cudaMalloc(&out, 64));
  CK(cudaFuncSetAttribute(overlap_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  auto go = [&](int mmas, int ld, int iters) {
    std::vector<uint64_t> best;
    for (int r = 0; r < 3; ++r) {
      overlap_kernel<<<1, 256, smem>>>(mmas, ld, iters, out);
      CK(cudaDeviceSynchronize());
      std::vector<uint64_t> h(3);
      CK(cudaMemcpy(h.data(), out, 24, cudaMemcpyDeviceToHost));
      if (best.empty() || h[0] + h[2] < best[0] + best[2]) best = h;
    }
    return best;
  };
  auto mma_alone = go(256, 0, 0);
  auto ld_alone = go(0, 1, 512);
  auto both = go(256, 1, 512);
  double ld_bytes = 4.0 * 512 * 4096;
  j.key("overlap_umma_bf16_n256_with_tmem_ld");
  j.obj();
  j.kv("mma_cycles_alone", (double)mma_alone[0]);
  j.kv("mma_cycles_with_ld", (double)both[0]);
  j.kv("ld_bytes_per_clk_alone", ld_bytes / ld_alone[2]);
  j.kv("ld_bytes_per_clk_with_mma", ld_bytes / both[2]);
  j.end_obj();
  fprintf(stderr, "overlap: 256 MMAs %llu cyc alone, %llu with TMEM loads; ld %.0f B/clk alone, %.0f with MMA\n",
          (unsigned long long)mma_alone[0], (unsigned long long)both[0], ld_bytes / ld_alone[2], ld_bytes / both[2]);
  cudaFree(out);
}

void run_smem_contention(Json& j, size_t smem) {
  uint64_t* out;
  CK(cudaMalloc(&out, 64));
  smem = std::max<size_t>(smem, 100 * 1024);
  CK(cudaFuncSetAttribute(smem_contention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  j.key("umma_smem_contention");
  j.obj();
  for (int n : {128, 256}) {
    for (int mode : {0, 1, 2}) {
      std::vector<uint64_t> best;
      for (int r = 0; r < 3; ++r) {
        smem_contention_kernel<<<1, 256, smem>>>(256, n, mode, 8192, out);
        CK(cudaDeviceSynchronize());
        std::vector<uint64_t> h(3);
        CK(cudaMemcpy(h.data(), out, 24, cudaMemcpyDeviceToHost));
        if (best.empty() || h[0] < best[0]) best = h;
      }
      const char* names[] = {"alone", "with_sts128", "with_lds128"};
      char key[48];
      snprintf(key, sizeof key, "n%d_%s", n, names[mode]);
      j.key(key);
      j.obj();
      j.kv("cycles_per_mma", best[0] / 256.0);
      j.kv("other_bytes_per_clk", best[2] ? (double)best[1] / best[2] : 0.0);
      j.end_obj();
      fprintf(stderr, "UMMA N=%d %-12s %6.1f cyc/MMA   other warps %6.1f B/clk\n", n, names[mode], best[0] / 256.0,
              best[2] ? (double)best[1] / best[2] : 0.0);
    }
  }
  j.end_obj();
  cudaFree(out);
}

int main(int argc, char** argv) {
  DeviceInfo d = device_info();
  std::string want = argc > 1 ? argv[1] : "all";
  size_t smem = 64 * 1024;
  Json j;
  j.obj();
  emit_header(j, "tc05x");
  j.key("tc05x");
  j.obj();
  if (want == "all" || want == "tmem") run_tmem(j);
  if (want == "all" || want == "pipe") {
    run_pipe<KIND_F16>(j, "umma_pipeline_bf16_m128n64k16", 64, smem);
    run_pipe<KIND_F16>(j, "umma_pipeline_bf16_m128n128k16", 128, smem);
    run_pipe<KIND_F16>(j, "umma_pipeline_bf16_m128n256k16", 256, smem);
    run_pipe<KIND_F8F6F4>(j, "umma_pipeline_fp8_m128n256k32", 256, smem);
    run_pipe<KIND_NVFP4>(j, "umma_pipeline_nvfp4_m128n256k64", 256, smem);
  }
  if (want == "all" || want == "cp") {
    run_cp<0>(j, "tcgen05_cp_128x256b", smem);
    run_cp<1>(j, "tcgen05_cp_128x128b", smem);
    run_cp<2>(j, "tcgen05_cp_4x256b", smem);
  }
  if (want == "all" || want == "alloc") run_alloc(j);
  if (want == "all" || want == "overlap") run_overlap(j, smem);
  if (want == "all" || want == "smem") run_smem_contention(j, smem);
  j.end_obj();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  (void)d;
  return 0;
}
