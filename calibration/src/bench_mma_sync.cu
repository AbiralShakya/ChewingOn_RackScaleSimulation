// Warp-level tensor core (HMMA/IMMA/DMMA via mma.sync) and ldmatrix, sm_80+.
//
// This is the Ampere tensor-core path: one warp issues a synchronous MMA whose
// fragments live in registers. (sm_90 adds wgmma, sm_100 tcgen05; those are in
// bench_tc.cu. mma.sync still runs on both, so this bench is useful there as a
// legacy-path reference.)
//
// Latency: one warp, a chain of MMAs dependent through the accumulator
// (D = A*B + D); slope between two chain lengths, as in bench_alu.
//
// Throughput: one CTA per SM, W warps, ILP independent accumulators per warp,
// CTA span from CtaTimer. Reported as dense FLOPs (or int ops) per clock per SM
// for the best (W, ILP) and for every point of the sweep, so the simulator can
// see how many warps per SMSP it takes to saturate the pipe.
//
// ldmatrix: latency by address chasing through shared memory, throughput as
// bytes/clk/SM with every warp loading x4 fragments from conflict-free rows.
#include "common.cuh"

constexpr int UNROLL = 32;

// Each op: A, B fragment registers (b32, or f64 for DMMA), C/D accumulator
// registers. step() is D = A*B + D with D aliased to C.
#define MMA_OP(NAME, SASS, SHAPE_M, SHAPE_N, SHAPE_K, AT, NA, NB, CT, NC, ASM_BODY)   \
  struct NAME {                                                                    \
    static constexpr const char* name = #NAME;                                     \
    static constexpr const char* sass = SASS;                                      \
    static constexpr int M = SHAPE_M, N = SHAPE_N, K = SHAPE_K;                    \
    static constexpr int na = NA, nb = NB, nc = NC;                                \
    using A_t = AT;                                                                \
    using C_t = CT;                                                                \
    __device__ __forceinline__ static void step(C_t* c, const A_t* a, const A_t* b) { \
      ASM_BODY;                                                                    \
    }                                                                              \
  };

// clang-format off
MMA_OP(F16_F32_m16n8k16, "HMMA.16816.F32", 16, 8, 16, uint32_t, 4, 2, float, 4,
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])))
MMA_OP(F16_F16_m16n8k16, "HMMA.16816.F16", 16, 8, 16, uint32_t, 4, 2, uint32_t, 2,
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])))
MMA_OP(BF16_F32_m16n8k16, "HMMA.16816.F32.BF16", 16, 8, 16, uint32_t, 4, 2, float, 4,
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])))
MMA_OP(F16_F32_m16n8k8, "HMMA.1688.F32", 16, 8, 8, uint32_t, 2, 1, float, 4,
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(b[0])))
MMA_OP(TF32_F32_m16n8k8, "HMMA.1688.F32.TF32", 16, 8, 8, uint32_t, 4, 2, float, 4,
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])))
MMA_OP(S8_S32_m16n8k32, "IMMA.16832.S8.S8", 16, 8, 32, uint32_t, 4, 2, int32_t, 4,
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])))
MMA_OP(S4_S32_m16n8k64, "IMMA.16864.S4.S4", 16, 8, 64, uint32_t, 4, 2, int32_t, 4,
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])))
MMA_OP(F64_F64_m8n8k4, "DMMA.884", 8, 8, 4, double, 1, 1, double, 2,
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
               "{%0,%1}, {%2}, {%3}, {%0,%1};"
               : "+d"(c[0]), "+d"(c[1])
               : "d"(a[0]), "d"(b[0])))
// 2:4 structured sparsity: A is the compressed 16x16 half of a 16x32 tile, the
// metadata register selects the nonzero pair positions. Counted as dense-
// equivalent FLOPs (2*M*N*K with K=32), which is how NVIDIA quotes it.
MMA_OP(SP_F16_F32_m16n8k32, "HMMA.SP.16832.F32", 16, 8, 32, uint32_t, 4, 4, float, 4,
  asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                 "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(0x44444444u)))
// clang-format on

template <class Op>
__device__ __forceinline__ void load_frags(const uint32_t* io, typename Op::A_t* a,
                                           typename Op::A_t* b) {
  using A = typename Op::A_t;
#pragma unroll
  for (int i = 0; i < Op::na; ++i) a[i] = (A)io[(i + threadIdx.x) & 7];
#pragma unroll
  for (int i = 0; i < Op::nb; ++i) b[i] = (A)io[(i + 3 + threadIdx.x) & 7];
  if constexpr (std::is_same_v<A, double>) {
#pragma unroll
    for (int i = 0; i < Op::na; ++i) a[i] = 1e-3 * (double)(io[i & 7] & 7);
#pragma unroll
    for (int i = 0; i < Op::nb; ++i) b[i] = 1e-3 * (double)(io[(i + 1) & 7] & 7);
  }
}

template <class Op>
__global__ void mma_lat_kernel(const uint32_t* io, int iters, uint64_t* cyc, float* sink) {
  using C = typename Op::C_t;
  typename Op::A_t a[Op::na], b[Op::nb];
  load_frags<Op>(io, a, b);
  C c[Op::nc];
#pragma unroll
  for (int i = 0; i < Op::nc; ++i) c[i] = (C)0;
  __syncwarp();
  uint64_t t0 = clk64();
  c[0] = c[0] + (C)(t0 >> 63);  // opaque zero
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) Op::step(c, a, b);
  }
  // Consume the accumulator before reading the clock, so t1 waits for the last MMA.
  C acc = c[0];
#pragma unroll
  for (int i = 1; i < Op::nc; ++i) acc = acc + c[i];
  __shared__ float fence_word;
  asm volatile("st.shared.f32 [%0], %1;" ::"r"(smem_u32(&fence_word)), "f"((float)acc) : "memory");
  uint64_t t1 = clk64();
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  if (acc == (C)12345) sink[threadIdx.x] = (float)acc;
}

template <class Op, int ILP>
__global__ void mma_thr_kernel(const uint32_t* io, int iters, uint64_t* cyc, float* sink) {
  using C = typename Op::C_t;
  extern __shared__ uint8_t pad[];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  typename Op::A_t a[Op::na], b[Op::nb];
  load_frags<Op>(io, a, b);
  C c[ILP][Op::nc];
#pragma unroll
  for (int j = 0; j < ILP; ++j)
#pragma unroll
    for (int i = 0; i < Op::nc; ++i) c[j][i] = (C)0;
  __syncthreads();
  uint64_t t0 = cta_timer_start(&tm);
#pragma unroll
  for (int j = 0; j < ILP; ++j) c[j][0] = c[j][0] + (C)(t0 >> 63);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
#pragma unroll
      for (int j = 0; j < ILP; ++j) Op::step(c[j], a, b);
    }
  }
  C acc = (C)0;
#pragma unroll
  for (int j = 0; j < ILP; ++j)
#pragma unroll
    for (int i = 0; i < Op::nc; ++i) acc = acc + c[j][i];
  if (acc == (C)12345) sink[threadIdx.x] = (float)acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) {
    cyc[2 * blockIdx.x] = span;
    cyc[2 * blockIdx.x + 1] = smid();
  }
  if (iters < 0) pad[threadIdx.x] = 0;
}

struct Ctx {
  DeviceInfo d;
  size_t one_cta_smem;
  uint32_t* io;
  uint64_t* cyc;
  float* sink;
};

template <class Op, int ILP>
double thr_point(Ctx& x, int warps, int iters) {
  auto k = mma_thr_kernel<Op, ILP>;
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)x.one_cta_smem));
  double best = 0;
  // CAL_THR_REPS=1 under ncu, which profiles every launch.
  static const int reps = getenv("CAL_THR_REPS") ? atoi(getenv("CAL_THR_REPS")) : 3;
  for (int r = 0; r < reps; ++r) {
    k<<<x.d.sm_count, 32 * warps, x.one_cta_smem>>>(x.io, iters, x.cyc, x.sink);
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> h(2 * x.d.sm_count);
    CK(cudaMemcpy(h.data(), x.cyc, h.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> rates;
    double flops = 2.0 * Op::M * Op::N * Op::K * warps * ILP * UNROLL * iters;
    for (int b = 0; b < x.d.sm_count; ++b) rates.push_back(flops / (double)h[2 * b]);
    best = std::max(best, median(rates));
  }
  return best;
}

template <class Op>
void run_mma(Json& j, Ctx& x) {
  auto lat = [&](int iters) {
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) {
      mma_lat_kernel<Op><<<1, 32>>>(x.io, iters, x.cyc, x.sink);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, x.cyc, 8, cudaMemcpyDeviceToHost));
      v.push_back((double)c);
    }
    return vmin(v);
  };
  const int n1 = 8, n2 = 40;
  double c1 = lat(n1), c2 = lat(n2);
  double lat_per = (c2 - c1) / ((double)(n2 - n1) * UNROLL);

  const int iters = std::is_same_v<typename Op::A_t, double> ? 4 : 16;
  j.key(Op::name);
  j.obj();
  j.kvs("sass_expected", Op::sass);
  j.kvs("lat_kernel", std::string("mma_lat_kernel<") + Op::name + ">");
  j.key("shape");
  j.arr();
  j.integer(Op::M); j.integer(Op::N); j.integer(Op::K);
  j.end_arr();
  j.kvi("flops_per_mma", 2LL * Op::M * Op::N * Op::K);
  j.kv("latency_cycles", lat_per);
  j.kv("lat_raw_cycles_n1", c1);
  j.kv("lat_raw_cycles_n2", c2);
  double best = 0;
  int best_w = 0, best_ilp = 0;
  j.key("sweep");
  j.arr();
  for (int warps : {1, 2, 4, 8, 16, 32}) {
    for (int ilp : {1, 2, 4}) {
      double r = ilp == 1 ? thr_point<Op, 1>(x, warps, iters)
               : ilp == 2 ? thr_point<Op, 2>(x, warps, iters)
                          : thr_point<Op, 4>(x, warps, iters);
      j.obj();
      j.kvi("warps", warps);
      j.kvi("ilp", ilp);
      j.kv("flops_per_clk_per_sm", r);
      j.end_obj();
      if (r > best) { best = r; best_w = warps; best_ilp = ilp; }
    }
  }
  j.end_arr();
  j.kv("throughput_flops_per_clk_per_sm", best);
  j.kv("throughput_mma_per_clk_per_smsp", best / (2.0 * Op::M * Op::N * Op::K) / 4.0);
  // Cycles between back-to-back independent MMAs on one SMSP at saturation.
  j.kv("issue_interval_cycles_per_smsp", (2.0 * Op::M * Op::N * Op::K) * 4.0 / best);
  j.kvi("best_warps", best_w);
  j.kvi("best_ilp", best_ilp);
  j.end_obj();
  fprintf(stderr, "%-22s lat %6.2f cyc  thr %8.1f flop/clk/SM (w=%d ilp=%d)\n", Op::name,
          lat_per, best, best_w, best_ilp);
}

// ---------------------------------------------------------------------------
// ldmatrix
// ---------------------------------------------------------------------------
// Shared memory holds 8x8 b16 matrices; each row is 16 B. With every word of a
// row holding the byte offset of the next row, lane 0's first register after an
// x1 load is that offset, so a dependent chain through it measures latency.
constexpr int LDM_ROWS = 1024;  // 16 KB

__global__ void ldm_lat_kernel(const uint32_t* io, int iters, uint64_t* cyc, uint32_t* sink) {
  __shared__ __align__(128) uint32_t sm[LDM_ROWS * 4];
  for (int i = threadIdx.x; i < LDM_ROWS * 4; i += blockDim.x) {
    uint32_t row = i / 4;
    sm[i] = ((row + 8) % LDM_ROWS) * 16;  // next group of 8 rows
  }
  __syncthreads();
  uint32_t base = smem_u32(sm);
  uint32_t mask = io[0];  // ~0 at run time: opaque to ptxas
  uint32_t off = (threadIdx.x & 7) * 16;
  uint32_t r = 0;
  __syncwarp();
  uint64_t t0 = clk64();
  off += (uint32_t)(t0 >> 63);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
      asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r) : "r"(base + off));
      // All lanes take lane 0's value (row 0's next-row offset) plus their own row.
      uint32_t nxt = __shfl_sync(0xffffffffu, r, 0) & mask;
      off = nxt + (threadIdx.x & 7) * 16;
    }
  }
  uint64_t t1 = clk64() + (off == 0xdeadbeefu);
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  if (off == 0xdeadbeefu) sink[0] = r;
}

// Same chain without the ldmatrix (shuffle + and + add only), to subtract.
__global__ void ldm_lat_overhead_kernel(const uint32_t* io, int iters, uint64_t* cyc, uint32_t* sink) {
  uint32_t mask = io[0];
  uint32_t off = (threadIdx.x & 7) * 16;
  __syncwarp();
  uint64_t t0 = clk64();
  off += (uint32_t)(t0 >> 63);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
      uint32_t nxt = __shfl_sync(0xffffffffu, off + 16, 0) & mask;
      off = nxt + (threadIdx.x & 7) * 16;
    }
  }
  uint64_t t1 = clk64() + (off == 0xdeadbeefu);
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  if (off == 0xdeadbeefu) sink[0] = off;
}

template <int X>
__global__ void ldm_thr_kernel(int iters, uint64_t* cyc, uint32_t* sink) {
  extern __shared__ __align__(128) uint8_t dyn[];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  uint32_t base = smem_u32(dyn);
  // Each warp reads its own 8*X rows; rows are consecutive 16 B lines, so the
  // eight addresses of one 8x8 matrix cover one 128 B span: conflict-free.
  uint32_t warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  uint32_t off = ((warp * 8 * X + lane % (8 * X)) * 16) % (32 * 1024);
  uint32_t acc = 0;
  __syncthreads();
  uint64_t t0 = cta_timer_start(&tm);
  off += (uint32_t)(t0 >> 63);
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
      uint32_t r[4];
      if constexpr (X == 1)
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r[0]) : "r"(base + off));
      if constexpr (X == 2)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r[0]), "=r"(r[1]) : "r"(base + off));
      if constexpr (X == 4)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(base + off));
#pragma unroll
      for (int q = 0; q < X; ++q) acc ^= r[q];
    }
  }
  if (acc == 0x5a5a5a5au) sink[threadIdx.x] = acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) {
    cyc[2 * blockIdx.x] = span;
    cyc[2 * blockIdx.x + 1] = smid();
  }
}

template <int X>
void run_ldm_thr(Json& j, Ctx& x) {
  size_t smem = std::max(x.one_cta_smem, (size_t)32 * 1024);
  CK(cudaFuncSetAttribute(ldm_thr_kernel<X>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  const int iters = 32;
  double best = 0;
  int best_w = 0;
  j.key((std::string("ldmatrix_x") + std::to_string(X)).c_str());
  j.obj();
  j.key("sweep");
  j.arr();
  for (int warps : {4, 8, 16, 32}) {
    double pt = 0;
    for (int r = 0; r < 3; ++r) {
      ldm_thr_kernel<X><<<x.d.sm_count, 32 * warps, smem>>>(iters, x.cyc, (uint32_t*)x.sink);
      CK(cudaDeviceSynchronize());
      std::vector<uint64_t> h(2 * x.d.sm_count);
      CK(cudaMemcpy(h.data(), x.cyc, h.size() * 8, cudaMemcpyDeviceToHost));
      std::vector<double> rates;
      double bytes = 128.0 * X * warps * UNROLL * iters;  // 8 rows * 16 B per matrix
      for (int b = 0; b < x.d.sm_count; ++b) rates.push_back(bytes / (double)h[2 * b]);
      pt = std::max(pt, median(rates));
    }
    j.obj();
    j.kvi("warps", warps);
    j.kv("bytes_per_clk_per_sm", pt);
    j.end_obj();
    if (pt > best) { best = pt; best_w = warps; }
  }
  j.end_arr();
  j.kv("throughput_bytes_per_clk_per_sm", best);
  j.kv("throughput_inst_per_clk_per_sm", best / (128.0 * X));
  j.kvi("best_warps", best_w);
  j.end_obj();
  fprintf(stderr, "ldmatrix.x%d            thr %8.1f B/clk/SM (w=%d)\n", X, best, best_w);
}

int main() {
  Ctx x;
  x.d = device_info();
  x.one_cta_smem = x.d.p.sharedMemPerMultiprocessor / 2 + 4096;
  if (x.one_cta_smem > x.d.p.sharedMemPerBlockOptin) x.one_cta_smem = x.d.p.sharedMemPerBlockOptin;
  CK(cudaMalloc(&x.io, 64 * sizeof(uint32_t)));
  CK(cudaMalloc(&x.cyc, 2 * sizeof(uint64_t) * x.d.sm_count));
  CK(cudaMalloc(&x.sink, 1024 * sizeof(float)));
  // Small fp16 values (0x3c00 = 1.0, 0x2000 ~ 0.0078) keep the f16 accumulator finite.
  uint32_t h[64];
  for (int i = 0; i < 64; ++i) h[i] = 0x20002000u;
  CK(cudaMemcpy(x.io, h, sizeof h, cudaMemcpyHostToDevice));

  Json j;
  j.obj();
  emit_header(j, "mma_sync");
  j.kv("sm_clock_mhz_measured", measure_sm_clock_mhz());
  j.key("mma");
  j.obj();
  run_mma<F16_F32_m16n8k16>(j, x);
  run_mma<F16_F16_m16n8k16>(j, x);
  run_mma<BF16_F32_m16n8k16>(j, x);
  run_mma<F16_F32_m16n8k8>(j, x);
  run_mma<TF32_F32_m16n8k8>(j, x);
  run_mma<S8_S32_m16n8k32>(j, x);
  run_mma<S4_S32_m16n8k64>(j, x);
  run_mma<F64_F64_m8n8k4>(j, x);
  run_mma<SP_F16_F32_m16n8k32>(j, x);
  j.end_obj();

  // ldmatrix latency (mask = ~0 so the chase follows the stored offsets).
  uint32_t ones = ~0u;
  CK(cudaMemcpy(x.io, &ones, 4, cudaMemcpyHostToDevice));
  auto lat = [&](auto kern, int iters) {
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) {
      kern<<<1, 32>>>(x.io, iters, x.cyc, (uint32_t*)x.sink);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, x.cyc, 8, cudaMemcpyDeviceToHost));
      v.push_back((double)c);
    }
    return vmin(v);
  };
  double l1 = lat(ldm_lat_kernel, 8), l2 = lat(ldm_lat_kernel, 40);
  double o1 = lat(ldm_lat_overhead_kernel, 8), o2 = lat(ldm_lat_overhead_kernel, 40);
  double per = (l2 - l1) / (32.0 * UNROLL), ovh = (o2 - o1) / (32.0 * UNROLL);
  j.key("ldmatrix");
  j.obj();
  j.kv("latency_cycles_x1_with_chain", per);
  j.kv("chain_overhead_cycles", ovh);
  j.kv("latency_cycles_x1", per - ovh);
  run_ldm_thr<1>(j, x);
  run_ldm_thr<2>(j, x);
  run_ldm_thr<4>(j, x);
  j.end_obj();
  fprintf(stderr, "ldmatrix.x1 latency %.2f cyc (chain %.2f - overhead %.2f)\n", per - ovh, per, ovh);

  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
