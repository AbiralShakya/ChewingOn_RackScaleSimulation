// Pipe co-issue under warp-MMA load, and warp collectives. sm_80+.
//
// FlashAttention-2 on Ampere interleaves HMMA (QK^T, PV) with MUFU.EX2 and
// FFMA/FMUL (online softmax) and LDSM (fragments) in the same warps. Whether
// those streams overlap or serialize decides the softmax bubble. Selective
// scan (Mamba) leans on FFMA/EX2 recurrences and warp scans.
//
// Mix: one CTA per SM, 32 warps (8 per SMSP). Each iteration a warp issues NH
// HMMA.16816.F32 (4 independent accumulators) and NO ops of another kind
// (8 independent chains, or LDSM.x4), interleaved. We report cycles per
// warp-iteration per SMSP for every (NH, NO), and an overlap factor
//
//   overlap = (t(NH,0) + t(0,NO) - t(NH,NO)) / min(t(NH,0), t(0,NO))
//
// 1 = the smaller stream is fully hidden, 0 = the two streams serialize.
//
// Collectives: latency of a 5-step shfl.up inclusive scan, a 5-step
// shfl.bfly all-reduce (max, as in softmax), and redux.sync.add.u32 (sm_80).
#include "common.cuh"

constexpr int WARPS = 32;
constexpr int UNROLL = 4;

enum Kind { K_EX2 = 0, K_FFMA = 1, K_FMUL = 2, K_LDSM = 3 };
static const char* kind_name[] = {"ex2", "ffma", "fmul", "ldsm_x4"};
static const char* kind_sass[] = {"MUFU.EX2", "FFMA", "FMUL", "LDSM.16.M88.4"};

__device__ __forceinline__ void hmma(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int KIND>
__device__ __forceinline__ void other(float& x, float m, uint32_t sbase, uint32_t& lacc) {
  if constexpr (KIND == K_EX2) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x));
  if constexpr (KIND == K_FFMA) asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(x) : "f"(m));
  if constexpr (KIND == K_FMUL) asm volatile("mul.f32 %0, %0, %1;" : "+f"(x) : "f"(m));
  if constexpr (KIND == K_LDSM) {
    uint32_t r0, r1, r2, r3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sbase));
    lacc ^= r0 ^ r1 ^ r2 ^ r3;
  }
}

template <int KIND, int NH, int NO>
__global__ void mix_kernel(const uint32_t* io, int iters, uint64_t* cyc, float* sink) {
  extern __shared__ __align__(128) uint8_t dyn[];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  uint32_t a[4], b[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = io[(i + threadIdx.x) & 7];
#pragma unroll
  for (int i = 0; i < 2; ++i) b[i] = io[(i + 3 + threadIdx.x) & 7];
  float c[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[j][i] = 0.f;
  float x[8];
  float m = __uint_as_float(io[8]);
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = __uint_as_float(io[9]) + (float)(j * threadIdx.x) * 1e-6f;
  uint32_t lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  // Conflict-free x4: 32 consecutive 16 B rows per warp.
  uint32_t sbase = smem_u32(dyn) + ((warp * 32 + lane) * 16) % (64 * 1024);
  uint32_t lacc = 0;
  __syncthreads();
  uint64_t t0 = cta_timer_start(&tm);
  sbase += (uint32_t)(t0 >> 63);
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      constexpr int N = NH > NO ? NH : NO;
#pragma unroll
      for (int k = 0; k < N; ++k) {
        if (k < NH) hmma(c[k & 3], a, b);
        if (k < NO) other<KIND>(x[k & 7], m, sbase, lacc);
      }
    }
  }
  float acc = (float)lacc;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc += c[j][i];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc += x[j];
  if (acc == 1234.5f) sink[threadIdx.x] = acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) {
    cyc[2 * blockIdx.x] = span;
    cyc[2 * blockIdx.x + 1] = smid();
  }
}

struct Ctx {
  DeviceInfo d;
  size_t smem;
  uint32_t* io;
  uint64_t* cyc;
  float* sink;
};

// Cycles per warp-iteration per SMSP (8 warps share each SMSP).
template <int KIND, int NH, int NO>
double mix_point(Ctx& x) {
  auto k = mix_kernel<KIND, NH, NO>;
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)x.smem));
  const int iters = 32;
  double best = 1e30;
  for (int r = 0; r < 3; ++r) {
    k<<<cal_grid(x.d.sm_count), 32 * WARPS, x.smem>>>(x.io, iters, x.cyc, x.sink);
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> h(2 * x.d.sm_count);
    CK(cudaMemcpy(h.data(), x.cyc, h.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> v;
    for (int b = 0; b < x.d.sm_count; ++b) v.push_back((double)h[2 * b] / (iters * UNROLL * (WARPS / 4)));
    best = std::min(best, median(v));
  }
  return best;
}

template <int KIND, int NH>
void mix_row(Json& j, Ctx& x, double t_h, const double* t_o) {
  // t_o[i] = other-only time for NO = 4 << i.
  double t[4] = {mix_point<KIND, NH, 4>(x), mix_point<KIND, NH, 8>(x), mix_point<KIND, NH, 16>(x),
                 mix_point<KIND, NH, 32>(x)};
  for (int i = 0; i < 4; ++i) {
    j.obj();
    j.kvi("hmma", NH);
    j.kvi("other", 4 << i);
    j.kv("cycles", t[i]);
    j.kv("cycles_hmma_alone", t_h);
    j.kv("cycles_other_alone", t_o[i]);
    double lo = std::min(t_h, t_o[i]);
    j.kv("overlap", lo > 0 ? (t_h + t_o[i] - t[i]) / lo : NAN);
    j.end_obj();
  }
}

template <int KIND>
void run_kind(Json& j, Ctx& x, double t_h4, double t_h8) {
  double t_o[4] = {mix_point<KIND, 0, 4>(x), mix_point<KIND, 0, 8>(x), mix_point<KIND, 0, 16>(x),
                   mix_point<KIND, 0, 32>(x)};
  j.key(kind_name[KIND]);
  j.obj();
  j.kvs("sass", kind_sass[KIND]);
  j.key("points");
  j.arr();
  mix_row<KIND, 4>(j, x, t_h4, t_o);
  mix_row<KIND, 8>(j, x, t_h8, t_o);
  j.end_arr();
  j.end_obj();
  fprintf(stderr, "%-8s alone: 4->%.1f 8->%.1f 16->%.1f 32->%.1f cyc/warp-iter/SMSP\n", kind_name[KIND],
          t_o[0], t_o[1], t_o[2], t_o[3]);
}

// ---------------------------------------------------------------------------
// Warp collectives
// ---------------------------------------------------------------------------
template <int OP>  // 0: shfl.up inclusive scan, 1: shfl.bfly max all-reduce, 2: redux.sync.add
__global__ void coll_lat_kernel(const uint32_t* io, int iters, uint64_t* cyc, uint32_t* sink) {
  uint32_t v = io[0] + threadIdx.x;
  float f = __uint_as_float(io[1]) + threadIdx.x;
  __shared__ uint32_t fence_word;
  __syncwarp();
  uint64_t t0 = clk64();
  v += (uint32_t)(t0 >> 63);
  for (int i = 0; i < iters; ++i) {
    if constexpr (OP == 0) {
#pragma unroll
      for (int s = 1; s < 32; s <<= 1) {
        uint32_t n = __shfl_up_sync(0xffffffffu, v, s);
        if ((threadIdx.x & 31) >= s) v += n;
      }
    }
    if constexpr (OP == 1) {
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) f = fmaxf(f, __shfl_xor_sync(0xffffffffu, f, s));
      f += 1.0f;  // keep the next round dependent and not idempotent
    }
    if constexpr (OP == 2) {
      uint32_t r;
      asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v));
      v = r + threadIdx.x;
    }
  }
  uint32_t out = v + __float_as_uint(f);
  asm volatile("st.shared.u32 [%0], %1;" ::"r"(smem_u32(&fence_word)), "r"(out) : "memory");
  uint64_t t1 = clk64();
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  if (out == 0xdeadbeefu) sink[threadIdx.x] = out;
}

template <int OP>
double coll_lat(Ctx& x) {
  auto run = [&](int iters) {
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) {
      coll_lat_kernel<OP><<<1, 32>>>(x.io, iters, x.cyc, (uint32_t*)x.sink);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, x.cyc, 8, cudaMemcpyDeviceToHost));
      v.push_back((double)c);
    }
    return vmin(v);
  };
  return (run(256) - run(64)) / 192.0;
}

int main() {
  Ctx x;
  x.d = device_info();
  x.smem = std::max<size_t>(x.d.p.sharedMemPerMultiprocessor / 2 + 4096, 64 * 1024);
  if (x.smem > x.d.p.sharedMemPerBlockOptin) x.smem = x.d.p.sharedMemPerBlockOptin;
  CK(cudaMalloc(&x.io, 64 * sizeof(uint32_t)));
  CK(cudaMalloc(&x.cyc, 2 * sizeof(uint64_t) * x.d.sm_count));
  CK(cudaMalloc(&x.sink, 1024 * sizeof(float)));
  uint32_t h[64];
  for (int i = 0; i < 64; ++i) h[i] = 0x20002000u;  // small fp16 pairs
  float m = 0.999f, x0 = 0.5f;
  memcpy(&h[8], &m, 4);
  memcpy(&h[9], &x0, 4);
  h[0] = 1;
  CK(cudaMemcpy(x.io, h, sizeof h, cudaMemcpyHostToDevice));

  Json j;
  j.obj();
  emit_header(j, "overlap");
  j.kv("sm_clock_mhz_measured", measure_sm_clock_mhz());
  j.kvi("warps_per_sm", WARPS);
  double t_h4 = mix_point<K_FFMA, 4, 0>(x), t_h8 = mix_point<K_FFMA, 8, 0>(x);
  j.kv("hmma4_alone_cycles", t_h4);
  j.kv("hmma8_alone_cycles", t_h8);
  fprintf(stderr, "HMMA alone: 4->%.1f 8->%.1f cyc/warp-iter/SMSP\n", t_h4, t_h8);
  j.key("mix");
  j.obj();
  run_kind<K_EX2>(j, x, t_h4, t_h8);
  run_kind<K_FFMA>(j, x, t_h4, t_h8);
  run_kind<K_FMUL>(j, x, t_h4, t_h8);
  run_kind<K_LDSM>(j, x, t_h4, t_h8);
  j.end_obj();

  double scan = coll_lat<0>(x), bfly = coll_lat<1>(x), redux = coll_lat<2>(x);
  j.key("collectives");
  j.obj();
  j.kv("shfl_up_inclusive_scan32_cycles", scan);
  j.kv("shfl_xor_max_allreduce32_cycles", bfly);
  j.kv("redux_sync_add_u32_cycles", redux);
  j.end_obj();
  fprintf(stderr, "warp scan %.1f  bfly-max allreduce %.1f  redux.add %.1f cyc\n", scan, bfly, redux);
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
