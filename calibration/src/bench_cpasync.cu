// cp.async (LDGSTS), sm_80+: the Ampere global->shared copy that bypasses the
// register file. Hopper/Blackwell keep it next to TMA, so this runs there too.
//
// Latency: one thread chases a random permutation of 128 B lines. Each step
// issues cp.async for the next line into shared memory, waits on the group, and
// reads the next index out of shared memory. The same chase with a plain
// ld.global.cg gives the register-path baseline; the difference is what the
// async path costs (or saves) per hop. Done for an L2-resident and a
// DRAM-sized buffer.
//
// Bandwidth: a grid of CTAs streams a buffer into a ring of shared-memory
// stages (commit_group / wait_group<STAGES-1>), sweeping stages and copy
// width/cache operator, against an ld.global.v4 -> st.shared baseline. Run
// over an L2-resident and a DRAM-sized buffer.
#include "common.cuh"

#include <numeric>
#include <random>

__device__ __forceinline__ void cp_async_cg16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_ca16(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_ca8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_ca4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

constexpr int LINE = 128;  // bytes; chase entries are one uint32 per line

// ---------------------------------------------------------------------------
// Latency
// ---------------------------------------------------------------------------
// next[i] (at word i*LINE/4) holds the line index of the following hop.
// 0: ld.global.cg (LDG.E.STRONG.GPU on sm_80), 1: cp.async.cg 16,
// 2: cp.async.ca 16, 3: cp.async.ca 4, 4: ld.global.ca (plain LDG.E; the
// buffer is far larger than L1, so random hops miss L1)
template <int MODE>
__global__ void chase_kernel(const uint32_t* buf, int steps, uint64_t* out) {
  __shared__ __align__(16) uint32_t st[4];
  uint32_t dst = smem_u32(st);
  uint32_t idx = 0;
  // Warm the clock path and TLB for the first line.
  uint64_t t0 = clk64();
  idx += (uint32_t)(t0 >> 63);
  for (int s = 0; s < steps; ++s) {
    const uint32_t* p = buf + (size_t)idx * (LINE / 4);
    if constexpr (MODE == 0) {
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(p) : "memory");
    } else if constexpr (MODE == 4) {
      asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(idx) : "l"(p) : "memory");
    } else {
      if constexpr (MODE == 1) cp_async_cg16(dst, p);
      if constexpr (MODE == 2) cp_async_ca16(dst, p);
      if constexpr (MODE == 3) cp_async_ca4(dst, p);
      cp_commit();
      cp_wait<0>();
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(idx) : "r"(dst) : "memory");
    }
  }
  uint64_t t1 = clk64() + (idx == 0xffffffffu);
  out[0] = t1 - t0;
  out[1] = idx;
}

// Shared-memory load latency alone, to separate it out of the cp.async hop.
__global__ void lds_chase_kernel(int steps, uint64_t* out) {
  __shared__ uint32_t st[256];
  for (int i = 0; i < 256; ++i) st[i] = (i * 37 + 11) & 255;
  __syncwarp();
  uint32_t idx = 0;
  uint64_t t0 = clk64();
  idx += (uint32_t)(t0 >> 63);
  for (int s = 0; s < steps; ++s)
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(idx) : "r"(smem_u32(st) + 4 * idx) : "memory");
  uint64_t t1 = clk64() + (idx == 0xffffffffu);
  out[0] = t1 - t0;
  out[1] = idx;
}

struct ChaseBuf {
  uint32_t* d = nullptr;
  size_t lines = 0;
};

ChaseBuf make_chase(size_t bytes, uint64_t seed) {
  ChaseBuf b;
  b.lines = bytes / LINE;
  std::vector<uint32_t> perm(b.lines);
  std::iota(perm.begin(), perm.end(), 0u);
  std::mt19937_64 rng(seed);
  std::shuffle(perm.begin() + 1, perm.end(), rng);  // single cycle through all lines
  std::vector<uint32_t> h(b.lines * (LINE / 4), 0);
  for (size_t i = 0; i < b.lines; ++i) h[(size_t)perm[i] * (LINE / 4)] = perm[(i + 1) % b.lines];
  CK(cudaMalloc(&b.d, bytes));
  CK(cudaMemcpy(b.d, h.data(), bytes, cudaMemcpyHostToDevice));
  return b;
}

// Evicts L2 by streaming writes over a buffer several times its size.
__global__ void flush_kernel(uint4* p, size_t n16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_uint4(i, 0, 0, 0);
}
uint4* g_flush = nullptr;
size_t g_flush_n16 = 0;

template <int MODE>
double chase_cycles(const ChaseBuf& b, uint64_t* out, int steps, bool warm) {
  if (warm) chase_kernel<MODE><<<1, 1>>>(b.d, (int)std::min<size_t>(b.lines, 1u << 20), out);
  std::vector<double> v;
  for (int r = 0; r < 3; ++r) {
    // DRAM case: every repetition starts from a cold L2, otherwise reps 2 and 3
    // re-walk the 4096 lines rep 1 just brought into L2.
    if (!warm) flush_kernel<<<1024, 256>>>(g_flush, g_flush_n16);
    chase_kernel<MODE><<<1, 1>>>(b.d, steps, out);
    CK(cudaDeviceSynchronize());
    uint64_t h[2];
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost));
    v.push_back((double)h[0] / steps);
  }
  return median(v);
}

void run_latency(Json& j, const DeviceInfo& d, uint64_t* out) {
  const int steps = 4096;
  // L2-resident: 1/4 of L2 so it fits even in one A100 partition's half.
  size_t l2_bytes = std::max<size_t>(d.l2_bytes / 4, 1 << 20);
  size_t dram_bytes = std::max<size_t>((size_t)d.l2_bytes * 16, 256ull << 20);
  ChaseBuf l2 = make_chase(l2_bytes, 1), dram = make_chase(dram_bytes, 2);
  g_flush_n16 = (size_t)d.l2_bytes * 4 / 16;
  CK(cudaMalloc(&g_flush, g_flush_n16 * 16));

  std::vector<double> v;
  for (int r = 0; r < 3; ++r) {
    lds_chase_kernel<<<1, 1>>>(steps, out);
    CK(cudaDeviceSynchronize());
    uint64_t h[2];
    CK(cudaMemcpy(h, out, 16, cudaMemcpyDeviceToHost));
    v.push_back((double)h[0] / steps);
  }
  double lds = median(v);

  j.key("latency");
  j.obj();
  j.kv("lds_hop_cycles", lds);
  const char* names[] = {"ld_global_cg", "cp_async_cg16", "cp_async_ca16", "cp_async_ca4", "ld_global_ca"};
  for (int level = 0; level < 2; ++level) {
    const ChaseBuf& b = level == 0 ? l2 : dram;
    // L2 case: pre-warm so every hop is an L2 hit (cg skips L1).
    // DRAM case: buffer is 16x L2, random order, so hops miss L2.
    bool warm = level == 0;
    double c[5] = {chase_cycles<0>(b, out, steps, warm), chase_cycles<1>(b, out, steps, warm),
                   chase_cycles<2>(b, out, steps, warm), chase_cycles<3>(b, out, steps, warm),
                   chase_cycles<4>(b, out, steps, warm)};
    j.key(level == 0 ? "l2" : "dram");
    j.obj();
    j.kvi("buffer_bytes", (long long)(level == 0 ? l2_bytes : dram_bytes));
    for (int m = 0; m < 5; ++m) {
      j.key(names[m]);
      j.obj();
      j.kv("hop_cycles", c[m]);
      // cp.async hop = copy + commit/wait + LDS; report the copy part alone.
      if (m >= 1 && m <= 3) j.kv("copy_cycles_excl_lds", c[m] - lds);
      j.end_obj();
    }
    j.end_obj();
    fprintf(stderr, "%-4s chase  ld.ca %6.1f  ld.cg %6.1f  cp.cg16 %6.1f  cp.ca16 %6.1f  cp.ca4 %6.1f  (lds %.1f)\n",
            level == 0 ? "L2" : "DRAM", c[4], c[0], c[1], c[2], c[3], lds);
  }
  j.end_obj();
  cudaFree(l2.d);
  cudaFree(dram.d);
  cudaFree(g_flush);
}

// ---------------------------------------------------------------------------
// Bandwidth
// ---------------------------------------------------------------------------
// MODE 0: ld.global.v4 -> st.shared (register path)
// MODE 1: cp.async.cg 16 B   MODE 2: cp.async.ca 16 B   MODE 3: cp.async.ca 8 B
// MODE 4: cp.async.ca 4 B
template <int MODE, int STAGES>
__global__ void stream_kernel(const uint4* __restrict__ src, size_t n16, int passes, uint32_t* sink) {
  extern __shared__ __align__(16) uint4 ring[];  // STAGES * blockDim.x * 16 B
  const int T = blockDim.x;
  constexpr int ELEM = MODE == 3 ? 8 : MODE == 4 ? 4 : 16;
  constexpr int SUB = 16 / ELEM;  // copies per 16 B chunk
  uint32_t acc = 0;
  size_t tiles = n16 / T;
  for (int p = 0; p < passes; ++p) {
    size_t first = blockIdx.x, step = gridDim.x;
    size_t ntile = first < tiles ? (tiles - first + step - 1) / step : 0;
    auto issue = [&](size_t t, int stage) {
      const uint4* g = src + (first + t * step) * T + threadIdx.x;
      uint4* s = ring + stage * T + threadIdx.x;
      if constexpr (MODE == 0) {
        uint4 v;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(g));
        *s = v;
      } else {
#pragma unroll
        for (int q = 0; q < SUB; ++q) {
          uint32_t dst = smem_u32(s) + q * ELEM;
          const char* gp = (const char*)g + q * ELEM;
          if constexpr (MODE == 1) cp_async_cg16(dst, gp);
          if constexpr (MODE == 2) cp_async_ca16(dst, gp);
          if constexpr (MODE == 3) cp_async_ca8(dst, gp);
          if constexpr (MODE == 4) cp_async_ca4(dst, gp);
        }
      }
    };
    // Prologue: fill STAGES-1 stages.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if ((size_t)s < ntile) issue(s, s);
      if constexpr (MODE != 0) cp_commit();
    }
    for (size_t t = 0; t < ntile; ++t) {
      size_t nxt = t + STAGES - 1;
      if (nxt < ntile) issue(nxt, (int)(nxt % STAGES));
      if constexpr (MODE != 0) {
        cp_commit();
        cp_wait<STAGES - 1>();
      }
      __syncthreads();
      // Consume one word per thread of the stage that just landed (a real
      // consumer would run MMAs here), then release it.
      acc += ring[(t % STAGES) * T + ((threadIdx.x * 5) % T)].x;
      __syncthreads();
    }
    if constexpr (MODE != 0) cp_wait<0>();
  }
  if (acc == 0x12345678u) sink[blockIdx.x] = acc;
}

template <int MODE, int STAGES>
double stream_gbs(const uint4* buf, size_t bytes, int ctas_per_sm, int threads, const DeviceInfo& d,
                  uint32_t* sink, int passes) {
  auto k = stream_kernel<MODE, STAGES>;
  size_t smem = (size_t)STAGES * threads * 16;
  if (smem > (size_t)d.p.sharedMemPerBlockOptin) return NAN;
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, threads, smem));
  if (occ < ctas_per_sm) return NAN;
  int grid = cal_grid(d.sm_count * ctas_per_sm);
  size_t n16 = bytes / 16;
  double ms = time_ms([&] { k<<<grid, threads, smem>>>(buf, n16, passes, sink); }, 5, 1);
  return (double)(n16 / threads) * threads * 16 * passes / (ms * 1e-3) / 1e9;
}

template <int MODE>
void sweep_mode(Json& j, const char* name, const uint4* buf, size_t bytes, int passes,
                const DeviceInfo& d, uint32_t* sink) {
  j.key(name);
  j.obj();
  j.key("sweep");
  j.arr();
  double best = 0;
  for (int threads : {128, 256}) {
    for (int cps : {1, 2, 4}) {
      double r[4] = {stream_gbs<MODE, 1>(buf, bytes, cps, threads, d, sink, passes),
                     stream_gbs<MODE, 2>(buf, bytes, cps, threads, d, sink, passes),
                     stream_gbs<MODE, 4>(buf, bytes, cps, threads, d, sink, passes),
                     stream_gbs<MODE, 8>(buf, bytes, cps, threads, d, sink, passes)};
      int st[4] = {1, 2, 4, 8};
      for (int s = 0; s < 4; ++s) {
        if (!std::isfinite(r[s])) continue;
        j.obj();
        j.kvi("threads", threads);
        j.kvi("ctas_per_sm", cps);
        j.kvi("stages", st[s]);
        j.kvi("smem_bytes_in_flight_per_sm", (long long)cps * threads * 16 * std::max(1, st[s] - 1));
        j.kv("gbps", r[s]);
        j.end_obj();
        best = std::max(best, r[s]);
      }
    }
  }
  j.end_arr();
  j.kv("best_gbps", best);
  j.end_obj();
  fprintf(stderr, "  %-16s best %8.1f GB/s\n", name, best);
}

void run_bandwidth(Json& j, const DeviceInfo& d, uint32_t* sink) {
  size_t l2b = (size_t)d.l2_bytes / 2;  // comfortably L2-resident
  const size_t alloc = 2ull << 30;
  size_t drb = cal_bytes(alloc);  // trace mode shortens the streams, not the buffer
  uint4* buf;
  CK(cudaMalloc(&buf, alloc));
  CK(cudaMemset(buf, 1, alloc));
  j.key("bandwidth");
  j.obj();
  for (int level = 0; level < 2; ++level) {
    size_t bytes = cal_bytes(level == 0 ? l2b : drb);
    int passes = level == 0 && cal_bytes(l2b) == l2b ? 16 : 1;  // one pass in trace mode
    fprintf(stderr, "%s stream (%zu MB)\n", level == 0 ? "L2" : "DRAM", bytes >> 20);
    j.key(level == 0 ? "l2" : "dram");
    j.obj();
    j.kvi("buffer_bytes", (long long)bytes);
    sweep_mode<0>(j, "ld_global_v4_st_shared", buf, bytes, passes, d, sink);
    sweep_mode<1>(j, "cp_async_cg16", buf, bytes, passes, d, sink);
    sweep_mode<2>(j, "cp_async_ca16", buf, bytes, passes, d, sink);
    sweep_mode<3>(j, "cp_async_ca8", buf, bytes, passes, d, sink);
    sweep_mode<4>(j, "cp_async_ca4", buf, bytes, passes, d, sink);
    j.end_obj();
  }
  j.end_obj();
  cudaFree(buf);
}

int main() {
  DeviceInfo d = device_info();
  uint64_t* out;
  uint32_t* sink;
  CK(cudaMalloc(&out, 64));
  CK(cudaMalloc(&sink, 1 << 20));
  Json j;
  j.obj();
  emit_header(j, "cpasync");
  j.kv("sm_clock_mhz_measured", measure_sm_clock_mhz());
  j.kvi("l2_bytes", d.l2_bytes);
  j.kvi("mem_clock_khz", d.mem_clock_khz);
  j.kvi("bus_width_bits", d.bus_width);
  // DDR: two transfers per clock.
  j.kv("dram_peak_gbps_spec", 2.0 * d.mem_clock_khz * 1e3 * d.bus_width / 8 / 1e9);
  run_latency(j, d, out);
  run_bandwidth(j, d, sink);
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
