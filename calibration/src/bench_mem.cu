// Memory hierarchy: latency (pointer chase) and bandwidth for shared memory,
// distributed shared memory (cluster), L1, L2 and HBM.
//
// Latency is a single thread chasing a random cyclic permutation of 128 B
// lines, so every load depends on the one before and no prefetcher can help.
// The footprint sweep exposes each level as a plateau. Footprints that fit in
// L2 are warmed first; larger ones are chased cold, each line visited once,
// after an L2 flush, so the result is a DRAM access including any TLB miss.
//
// Bandwidth is timed with events for whole-GPU numbers, and with the CTA timer
// for per-SM numbers.
#include <cooperative_groups.h>

#include <map>
#include <numeric>
#include <random>

#include "common.cuh"

namespace cg = cooperative_groups;

constexpr int LINE_WORDS = 32;  // 128 B lines of uint32
constexpr int CHASE_HOPS = 4096;

// ---------------------------------------------------------------------------
// Pointer chase over global memory
// ---------------------------------------------------------------------------
// 0: ld.global.ca  -> LDG.E.STRONG.SM
// 1: ld.global.cg  -> LDG.E.STRONG.GPU (L2 only)
// 2: ld.global.L1::no_allocate -> plain (weak) LDG that does not fill L1
template <int MODE>
__device__ __forceinline__ uint32_t ld_chase(const uint32_t* p) {
  uint32_t v;
  if (MODE == 0)
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  else if (MODE == 1)
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  else
    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <int MODE>
__global__ void chase_kernel(const uint32_t* arr, uint32_t start, int warm_hops, int hops,
                             uint64_t* out, uint32_t* sink) {
  uint32_t j = start;
  for (int i = 0; i < warm_hops; ++i) j = ld_chase<MODE>(arr + j);
  uint64_t t0 = clk64();
  for (int i = 0; i < hops; ++i) j = ld_chase<MODE>(arr + j);
  // The store needs j, so it issues only after the last load returns.
  sink[0] = j;
  uint64_t t1 = clk64();
  out[0] = t1 - t0;
}

// Per-hop latency, recorded one load at a time, to see the near/far L2 split.
__global__ void chase_hist_kernel(const uint32_t* arr, uint32_t start, int warm_hops, int hops,
                                  uint32_t* lat_out, uint32_t* sink) {
  uint32_t j = start;
  for (int i = 0; i < warm_hops; ++i) j = ld_chase<1>(arr + j);
  for (int i = 0; i < hops; ++i) {
    uint32_t t0 = clk32();
    j = ld_chase<1>(arr + j);
    // A clock read does not wait on the scoreboard, so a store of j is not
    // enough (ptxas also drops it). A branch on j is: the clock read below
    // cannot issue before the branch resolves, which needs the loaded value.
    if (j == 0xffffffffu) break;
    uint32_t t1 = clk32();
    lat_out[i] = t1 - t0;
  }
  sink[0] = j;
}

__global__ void read_all_kernel(const uint4* p, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) sink[1] = acc;
}

__global__ void scatter_kernel(uint32_t* arr, const uint32_t* pos, const uint32_t* val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) arr[pos[i]] = val[i];
}

struct ChaseBuf {
  uint32_t* d = nullptr;
  size_t bytes = 0;
};

// Writes a random cycle through `lines` lines (chosen out of footprint/128) and
// returns the start index.
uint32_t build_cycle(ChaseBuf& b, size_t footprint, size_t lines, std::mt19937_64& rng) {
  size_t total_lines = footprint / (LINE_WORDS * 4);
  std::vector<uint32_t> sel;
  if (lines >= total_lines) {
    sel.resize(total_lines);
    std::iota(sel.begin(), sel.end(), 0u);
  } else {
    // Sample distinct lines spread over the footprint.
    std::uniform_int_distribution<uint64_t> u(0, total_lines - 1);
    std::vector<uint32_t> s;
    s.reserve(lines * 2);
    while (s.size() < lines * 2) s.push_back((uint32_t)u(rng));
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    std::shuffle(s.begin(), s.end(), rng);
    s.resize(std::min(lines, s.size()));
    sel = s;
  }
  std::shuffle(sel.begin(), sel.end(), rng);
  size_t n = sel.size();
  std::vector<uint32_t> pos(n), val(n);
  for (size_t i = 0; i < n; ++i) {
    pos[i] = sel[i] * LINE_WORDS;
    val[i] = sel[(i + 1) % n] * LINE_WORDS;
  }
  uint32_t *dp, *dv;
  CK(cudaMalloc(&dp, n * 4));
  CK(cudaMalloc(&dv, n * 4));
  CK(cudaMemcpy(dp, pos.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, val.data(), n * 4, cudaMemcpyHostToDevice));
  scatter_kernel<<<(unsigned)((n + 255) / 256), 256>>>(b.d, dp, dv, (int)n);
  CK(cudaDeviceSynchronize());
  cudaFree(dp);
  cudaFree(dv);
  return pos[0];
}

void flush_l2(uint4* flush, size_t bytes, uint32_t* sink) {
  read_all_kernel<<<1024, 512>>>(flush, bytes / 16, sink);
  CK(cudaDeviceSynchronize());
}

void run_latency_sweep(Json& j, const DeviceInfo& d) {
  const size_t max_fp = 4ull << 30;
  ChaseBuf b;
  b.bytes = max_fp;
  CK(cudaMalloc(&b.d, b.bytes));
  size_t flush_bytes = std::max<size_t>(8ull * d.l2_bytes, 256ull << 20);
  uint4* flush;
  CK(cudaMalloc(&flush, flush_bytes));
  CK(cudaMemset(flush, 1, flush_bytes));
  uint32_t* sink;
  uint64_t* out;
  CK(cudaMalloc(&sink, 64));
  CK(cudaMalloc(&out, 64));
  std::mt19937_64 rng(1234);

  const size_t warm_limit = (size_t)d.l2_bytes;  // fully-permuted, warmed sweeps up to L2 size
  j.key("latency_sweep");
  j.arr();
  for (size_t fp = 8 << 10; fp <= max_fp; fp *= 2) {
    bool warm = fp <= 2 * warm_limit;
    size_t lines = warm ? fp / 128 : CHASE_HOPS;
    uint32_t start = build_cycle(b, fp, lines, rng);
    // Warmed footprints: one parallel read from all SMs (each line is touched by
    // one SM, on either die), then the chase from SM 0. The first chase pass
    // pays the far-partition cost for lines homed on the other die; repeats hit
    // the copy the near partition now holds. Order: cg passes, then ca passes.
    auto chase = [&](int mode, int warm_hops, int hops) {
      if (mode == 0) chase_kernel<0><<<1, 1>>>(b.d, start, warm_hops, hops, out, sink);
      else if (mode == 1) chase_kernel<1><<<1, 1>>>(b.d, start, warm_hops, hops, out, sink);
      else chase_kernel<2><<<1, 1>>>(b.d, start, warm_hops, hops, out, sink);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost));
      return (double)c / hops;
    };
    double cg_first, cg_repeat, ca_repeat, weak_repeat;
    if (warm) {
      read_all_kernel<<<1024, 512>>>((const uint4*)b.d, fp / 16, sink);
      CK(cudaDeviceSynchronize());
      cg_first = chase(1, 0, CHASE_HOPS);
      cg_repeat = std::min(chase(1, 0, CHASE_HOPS), chase(1, 0, CHASE_HOPS));
      weak_repeat = std::min(chase(2, 0, CHASE_HOPS), chase(2, 0, CHASE_HOPS));
      // L1 can only be warmed from the chasing SM itself.
      int warm_hops = fp <= (1 << 20) ? (int)(fp / 128) : 0;
      ca_repeat = std::min(chase(0, warm_hops, CHASE_HOPS), chase(0, warm_hops, CHASE_HOPS));
    } else {
      std::vector<double> v0, v1, v2;
      for (int r = 0; r < 3; ++r) {
        flush_l2(flush, flush_bytes, sink);
        start = build_cycle(b, fp, lines, rng);  // fresh lines: never touched before
        flush_l2(flush, flush_bytes, sink);
        v1.push_back(chase(1, 0, (int)lines - 1));
        flush_l2(flush, flush_bytes, sink);
        start = build_cycle(b, fp, lines, rng);
        flush_l2(flush, flush_bytes, sink);
        v0.push_back(chase(0, 0, (int)lines - 1));
        flush_l2(flush, flush_bytes, sink);
        start = build_cycle(b, fp, lines, rng);
        flush_l2(flush, flush_bytes, sink);
        v2.push_back(chase(2, 0, (int)lines - 1));
      }
      weak_repeat = median(v2);
      cg_first = cg_repeat = median(v1);
      ca_repeat = median(v0);
    }
    double cyc[2] = {ca_repeat, cg_repeat};
    j.obj();
    j.kvi("footprint_bytes", (long long)fp);
    j.kvb("warmed", warm);
    j.kv("ld_ca_cycles", ca_repeat);
    j.kv("ld_cg_cycles", cg_repeat);
    j.kv("ld_cg_first_pass_cycles", cg_first);
    j.kv("ld_weak_noalloc_cycles", weak_repeat);
    j.end_obj();
    fprintf(stderr, "chase %10zu B  ca %7.1f  cg %7.1f  cg-first %7.1f  weak %7.1f cyc%s\n", fp, cyc[0], cyc[1], cg_first,
            weak_repeat, warm ? "" : " (cold)");
  }
  j.end_arr();

  // Per-hop histogram at a footprint that sits in L2 but far exceeds L1.
  {
    size_t fp = std::min<size_t>(d.l2_bytes / 4, 16ull << 20);
    uint32_t start = build_cycle(b, fp, fp / 128, rng);
    read_all_kernel<<<1024, 512>>>((const uint4*)b.d, fp / 16, sink);
    const int hops = 4096;
    uint32_t* lat;
    CK(cudaMalloc(&lat, hops * 4));
    chase_hist_kernel<<<1, 1>>>(b.d, start, 0, hops, lat, sink);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> h(hops);
    CK(cudaMemcpy(h.data(), lat, hops * 4, cudaMemcpyDeviceToHost));
    cudaFree(lat);
    std::vector<double> hv(h.begin(), h.end());
    std::sort(hv.begin(), hv.end());
    j.key("l2_hit_latency_distribution");
    j.obj();
    j.kvi("footprint_bytes", (long long)fp);
    j.kvs("note", "per-load cycles incl. a compare-and-branch and the clock read; spread reflects near/far L2 partitions");
    j.key("percentiles");
    j.obj();
    for (int pc : {1, 5, 10, 25, 50, 75, 90, 95, 99}) {
      char k[8];
      snprintf(k, sizeof k, "p%d", pc);
      j.kv(k, hv[(size_t)(pc / 100.0 * (hops - 1))]);
    }
    j.end_obj();
    // Split at the gap between the two modes.
    double thr = 0.5 * (hv[hops / 10] + hv[hops - 1 - hops / 100]);
    double sn = 0, sf = 0;
    int nn = 0, nf = 0;
    for (double x : hv) (x < thr ? (sn += x, ++nn) : (sf += x, ++nf));
    j.kv("split_threshold_cycles", thr);
    j.kv("near_mean_cycles", nn ? sn / nn : NAN);
    j.kv("far_mean_cycles", nf ? sf / nf : NAN);
    j.kv("far_fraction", (double)nf / hops);
    // Coarse histogram, 8-cycle bins
    j.key("hist_8cyc");
    j.obj();
    std::map<int, int> bins;
    for (double x : hv) bins[(int)x / 8 * 8]++;
    for (auto& kv : bins) {
      if (kv.second < 8) continue;
      j.kvi(std::to_string(kv.first).c_str(), kv.second);
    }
    j.end_obj();
    j.end_obj();
    fprintf(stderr, "L2 hit per-hop p10 %.0f p50 %.0f p90 %.0f\n", hv[hops / 10], hv[hops / 2], hv[hops * 9 / 10]);
  }
  cudaFree(b.d);
  cudaFree(flush);
  cudaFree(sink);
  cudaFree(out);
}

// ---------------------------------------------------------------------------
// Shared memory: latency, bandwidth, bank conflicts; DSMEM latency.
// ---------------------------------------------------------------------------
__global__ void smem_chase_kernel(int hops, uint64_t* out, uint32_t* sink) {
  __shared__ uint32_t s[1024];
  for (int i = 0; i < 1024; ++i) s[i] = (i * 97 + 31) % 1024;  // single-thread init
  uint32_t j = 0;
  uint32_t base = smem_u32(s);
  uint64_t t0 = clk64();
  for (int i = 0; i < hops; ++i) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(j) : "r"(base + j * 4) : "memory");
  sink[0] = j;
  uint64_t t1 = clk64();
  out[0] = t1 - t0;
}

// Each warp reads one 32-bit word per lane at `stride` words apart.
__global__ void smem_bw_kernel(int iters, int stride, uint64_t* out, uint32_t* sink) {
  extern __shared__ uint32_t s[];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  int n = 32 * 1024;  // 128 KB
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = i;
  __syncthreads();
  uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  (void)lane;
  // lane*stride + warp*7 + 15*128 < n for every stride used, so the 16 loads
  // are immediate offsets from one base.
  uint32_t base = smem_u32(s) + 4 * (lane * stride + warp * 7);
  uint32_t acc = 0;
  uint64_t t0 = cta_timer_start(&tm);
  uint32_t z = (uint32_t)(t0 >> 63);  // opaque zero
  for (int i = 0; i < iters; ++i) {
    // ptxas would otherwise hoist the loop-invariant loads out of the loop.
    uint32_t b = base + i * z;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      uint32_t v;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(b + k * 512) : "memory");
      acc += v;
    }
  }
  if (acc == 0x12345) sink[0] = acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) out[blockIdx.x] = span;
}

__global__ void smem_bw128_kernel(int iters, uint64_t* out, uint32_t* sink) {
  extern __shared__ uint4 s4[];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  const int n = 8 * 1024;  // 128 KB of uint4
  for (int i = threadIdx.x; i < n; i += blockDim.x) s4[i] = make_uint4(i, i, i, i);
  __syncthreads();
  uint32_t base = smem_u32(s4) + 16 * threadIdx.x;
  uint32_t acc = 0;
  uint64_t t0 = cta_timer_start(&tm);
  uint32_t z = (uint32_t)(t0 >> 63);  // opaque zero
  for (int i = 0; i < iters; ++i) {
    uint32_t bi = base + i * z;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      uint32_t a, b, c, e;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a), "=r"(b), "=r"(c), "=r"(e)
                   : "r"(bi + ((k * 1024 * 16) % (n * 16)))
                   : "memory");
      acc ^= a + b + c + e;
    }
  }
  if (acc == 0x12345) sink[0] = acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) out[blockIdx.x] = span;
}

__global__ void __cluster_dims__(2, 1, 1) dsmem_chase_kernel(int hops, uint64_t* out, uint32_t* sink) {
  __shared__ uint32_t s[1024];
  cg::cluster_group cl = cg::this_cluster();
  if (threadIdx.x == 0)
    for (int i = 0; i < 1024; ++i) s[i] = (i * 97 + 31) % 1024;
  cl.sync();
  if (cl.block_rank() == 0 && threadIdx.x == 0) {
    uint32_t* remote = cl.map_shared_rank(s, 1);
    uint32_t j = 0;
    uint64_t t0 = clk64();
    for (int i = 0; i < hops; ++i) j = *(volatile uint32_t*)(remote + j);
    sink[0] = j;
    uint64_t t1 = clk64();
    out[0] = t1 - t0;
    // same loop against local smem through a generic pointer, for the delta
    uint32_t* local = cl.map_shared_rank(s, 0);
    j = 0;
    t0 = clk64();
    for (int i = 0; i < hops; ++i) j = *(volatile uint32_t*)(local + j);
    sink[1] = j;
    t1 = clk64();
    out[1] = t1 - t0;
  }
  cl.sync();
}

void run_smem(Json& j, const DeviceInfo& d) {
  uint64_t* out;
  uint32_t* sink;
  CK(cudaMalloc(&out, 8 * d.sm_count));
  CK(cudaMalloc(&sink, 64));
  auto one = [&](auto launch, int hops) {
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) {
      launch();
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost));
      v.push_back((double)c / hops);
    }
    return vmin(v);
  };
  double l1 = one([&] { smem_chase_kernel<<<1, 1>>>(512, out, sink); }, 512);
  double l2 = one([&] { smem_chase_kernel<<<1, 1>>>(4096, out, sink); }, 4096);
  (void)l1;
  j.key("smem");
  j.obj();
  j.kv("ld_shared_latency_cycles", l2);

  size_t smem = 128 * 1024 + 64;
  CK(cudaFuncSetAttribute(smem_bw_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CK(cudaFuncSetAttribute(smem_bw128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  auto per_sm_span = [&](auto launch) {
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> h(d.sm_count);
    CK(cudaMemcpy(h.data(), out, 8 * d.sm_count, cudaMemcpyDeviceToHost));
    std::vector<double> v(h.begin(), h.end());
    return median(v);
  };
  const int iters = 256, threads = 1024;
  j.key("bank_conflict_sweep");
  j.arr();
  for (int stride : {1, 2, 4, 8, 16, 32, 33}) {
    double span = per_sm_span([&] { smem_bw_kernel<<<d.sm_count, threads, smem>>>(iters, stride, out, sink); });
    double warp_loads = (threads / 32.0) * iters * 16;
    j.obj();
    j.kvi("stride_words", stride);
    j.kv("cycles_per_warp_load_per_sm", span / warp_loads);
    j.kv("bytes_per_clk_per_sm", warp_loads * 128 / span);
    j.end_obj();
    fprintf(stderr, "smem stride %2d: %.2f cyc/warp-load (SM-wide)\n", stride, span / warp_loads);
  }
  j.end_arr();
  double span = per_sm_span([&] { smem_bw128_kernel<<<d.sm_count, threads, smem>>>(iters, out, sink); });
  double bytes = (double)threads * iters * 8 * 16;
  j.kv("ld_shared_v4_bytes_per_clk_per_sm", bytes / span);
  fprintf(stderr, "smem v4 bw %.1f B/clk/SM\n", bytes / span);

  // DSMEM
  int hops = 2048;
  std::vector<double> rem, loc;
  for (int r = 0; r < 5; ++r) {
    dsmem_chase_kernel<<<2, 32>>>(hops, out, sink);
    CK(cudaDeviceSynchronize());
    uint64_t c[2];
    CK(cudaMemcpy(c, out, 16, cudaMemcpyDeviceToHost));
    rem.push_back((double)c[0] / hops);
    loc.push_back((double)c[1] / hops);
  }
  j.kv("dsmem_remote_latency_cycles", vmin(rem));
  j.kv("dsmem_local_generic_latency_cycles", vmin(loc));
  fprintf(stderr, "smem lat %.1f  dsmem remote %.1f local-generic %.1f\n", l2, vmin(rem), vmin(loc));
  j.end_obj();
  cudaFree(out);
  cudaFree(sink);
}

// ---------------------------------------------------------------------------
// Bandwidth: HBM read/write/copy, L2 read, L1 read, per-SM scaling.
// ---------------------------------------------------------------------------
__global__ void bw_read_kernel(const uint4* __restrict__ p, size_t n, int reps, uint32_t* sink) {
  uint32_t acc = 0;
  for (int r = 0; r < reps; ++r)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
      uint4 v;
      asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
      acc ^= v.x + v.y + v.z + v.w;
    }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__global__ void bw_write_kernel(uint4* __restrict__ p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(p + i), "r"((uint32_t)i) : "memory");
}

__global__ void bw_copy_kernel(const uint4* __restrict__ a, uint4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}

// One CTA per SM reading a private 32 KB window repeatedly through L1.
__global__ void l1_bw_kernel(const uint4* p, int reps, uint64_t* out, uint32_t* sink) {
  extern __shared__ uint8_t pad[];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  const int n = 2048;  // 32 KB of uint4
  const uint4* w = p + (size_t)blockIdx.x * n;
  uint32_t acc = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc ^= w[i].x;  // warm L1
  __syncthreads();
  uint64_t t0 = cta_timer_start(&tm);
  uint32_t z = (uint32_t)(t0 >> 63);  // opaque zero
  for (int r = 0; r < reps; ++r)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      uint4 v;
      asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(w + i + r * z));
      acc ^= v.x + v.y + v.z + v.w;
    }
  if (acc == 0x9e3779b9u) sink[0] = acc;
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) out[blockIdx.x] = span;
  if (reps < 0) pad[0] = 0;
}

void run_bandwidth(Json& j, const DeviceInfo& d, double mhz) {
  size_t bytes = 8ull << 30;
  uint4 *a, *b;
  uint32_t* sink;
  uint64_t* out;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&b, bytes));
  CK(cudaMalloc(&sink, 64));
  CK(cudaMalloc(&out, 8 * d.sm_count));
  CK(cudaMemset(a, 1, bytes));
  size_t n = bytes / 16;
  int grid = d.sm_count * 4, blk = 1024;

  j.key("bandwidth");
  j.obj();
  double ms = time_ms([&] { bw_read_kernel<<<grid, blk>>>(a, n, 1, sink); });
  j.kv("hbm_read_GBps", bytes / ms / 1e6);
  double hbm_read = bytes / ms / 1e6;
  ms = time_ms([&] { bw_write_kernel<<<grid, blk>>>(b, n); });
  j.kv("hbm_write_GBps", bytes / ms / 1e6);
  double hbm_write = bytes / ms / 1e6;
  ms = time_ms([&] { bw_copy_kernel<<<grid, blk>>>(a, b, n); });
  j.kv("hbm_copy_GBps_rd_plus_wr", 2.0 * bytes / ms / 1e6);
  double hbm_copy = 2.0 * bytes / ms / 1e6;
  {
    double peak = 2.0 * d.mem_clock_khz * 1e3 * (d.bus_width / 8) / 1e9;
    j.kv("hbm_peak_GBps_from_attrs", peak);
    j.kvi("mem_bus_width_bits", d.bus_width);
    j.kvi("mem_clock_khz", d.mem_clock_khz);
  }
  fprintf(stderr, "HBM read %.0f write %.0f copy %.0f GB/s\n", hbm_read, hbm_write, hbm_copy);

  // L2: footprint well inside L2, many passes
  for (double frac : {0.125, 0.25, 0.5}) {
    size_t l2b = (size_t)(d.l2_bytes * frac) / 16 * 16;
    int reps = 32;
    bw_read_kernel<<<grid, blk>>>(a, l2b / 16, 2, sink);
    ms = time_ms([&] { bw_read_kernel<<<grid, blk>>>(a, l2b / 16, reps, sink); });
    double gbps = (double)l2b * reps / ms / 1e6;
    char k[64];
    snprintf(k, sizeof k, "l2_read_GBps_fp%.3f", frac);
    j.kv(k, gbps);
    snprintf(k, sizeof k, "l2_read_bytes_per_clk_per_sm_fp%.3f", frac);
    j.kv(k, gbps * 1e9 / (mhz * 1e6) / d.sm_count);
    fprintf(stderr, "L2 read (%.0f%% of L2) %.0f GB/s = %.1f B/clk/SM\n", frac * 100, gbps, gbps * 1e3 / mhz / d.sm_count);
  }

  // L1
  {
    size_t smem = d.p.sharedMemPerMultiprocessor / 2 + 4096;
    if (smem > d.p.sharedMemPerBlockOptin) smem = d.p.sharedMemPerBlockOptin;
    // Leave L1 room: request a smaller carve-out than max for this kernel.
    smem = 0;
    int reps = 64;
    l1_bw_kernel<<<d.sm_count, 1024, smem>>>(a, reps, out, sink);
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> h(d.sm_count);
    CK(cudaMemcpy(h.data(), out, 8 * d.sm_count, cudaMemcpyDeviceToHost));
    std::vector<double> v(h.begin(), h.end());
    double bpc = 32768.0 * reps / median(v);
    j.kv("l1_read_bytes_per_clk_per_sm", bpc);
    fprintf(stderr, "L1 read %.1f B/clk/SM\n", bpc);
  }

  // HBM read bandwidth as a function of SMs in use (one CTA per SM).
  {
    size_t smem = d.p.sharedMemPerMultiprocessor / 2 + 4096;
    if (smem > d.p.sharedMemPerBlockOptin) smem = d.p.sharedMemPerBlockOptin;
    CK(cudaFuncSetAttribute(bw_read_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    j.key("hbm_read_vs_sms");
    j.arr();
    std::vector<int> counts = {1, 2, 4, 8, 16, 32, 64, 96, 128};
    counts.push_back(d.sm_count);
    for (int c : counts) {
      if (c > d.sm_count) continue;
      size_t nb = std::min<size_t>(bytes, (size_t)c * (64ull << 20));
      ms = time_ms([&] { bw_read_kernel<<<c, 1024, smem>>>(a, nb / 16, 1, sink); }, 3);
      double gbps = nb / ms / 1e6;
      j.obj();
      j.kvi("sms", c);
      j.kv("GBps", gbps);
      j.kv("GBps_per_sm", gbps / c);
      j.end_obj();
      fprintf(stderr, "HBM read with %3d SMs: %.0f GB/s (%.1f per SM)\n", c, gbps, gbps / c);
    }
    j.end_arr();
  }
  j.end_obj();
  cudaFree(a);
  cudaFree(b);
  cudaFree(sink);
  cudaFree(out);
}

int main(int argc, char** argv) {
  DeviceInfo d = device_info();
  double mhz = measure_sm_clock_mhz();
  Json j;
  j.obj();
  emit_header(j, "mem");
  j.kv("sm_clock_mhz_measured", mhz);
  j.kvi("l2_bytes", d.l2_bytes);
  j.kvi("smem_per_sm_bytes", (long long)d.p.sharedMemPerMultiprocessor);
  j.kvi("smem_per_block_optin_bytes", (long long)d.p.sharedMemPerBlockOptin);
  j.kvi("regs_per_sm", d.p.regsPerMultiprocessor);
  j.kvi("max_threads_per_sm", d.p.maxThreadsPerMultiProcessor);
  run_smem(j, d);
  run_bandwidth(j, d, mhz);
  run_latency_sweep(j, d);
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
