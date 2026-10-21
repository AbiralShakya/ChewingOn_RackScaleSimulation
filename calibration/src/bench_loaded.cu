// Loaded latency: how memory latency grows with the bandwidth other SMs draw.
//
// One kernel, one CTA per SM. CTA 0 is the probe: a single thread chases a
// random cycle of 128 B lines and records cycles per hop. Every other CTA
// streams 16 B loads over its own region until the probe raises a flag, and
// reports the bytes it moved. Sweeping the number of loading CTAs (and their
// thread count) traces latency against achieved bandwidth, the curve a
// queueing memory model has to reproduce.
//
// Two targets: "hbm" (probe chases never-touched lines over 4 GB after an L2
// flush, load streams an 8 GB buffer) and "l2" (probe chases a warmed 16 MB
// footprint, load re-reads a 48 MB L2-resident region).
#include <numeric>
#include <random>

#include "common.cuh"

constexpr int LINE_WORDS = 32;

__global__ void loaded_kernel(const uint32_t* chase, uint32_t start, int hops, const uint4* load, size_t load_per_cta,
                              int load_threads, volatile uint32_t* done, uint64_t* probe_out, unsigned long long* bytes_out,
                              uint64_t* ns_out) {
  extern __shared__ uint8_t pad[];
  if (blockIdx.x == 0) {
    if (threadIdx.x != 0) return;
    // Let the loaders ramp up before timing.
    uint64_t w0 = gtimer_ns();
    while (gtimer_ns() - w0 < 20000) {
    }
    uint32_t j = start;
    uint64_t t0 = clk64();
    for (int i = 0; i < hops; ++i) asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(j) : "l"(chase + j) : "memory");
    probe_out[1] = j;
    uint64_t t1 = clk64();
    probe_out[0] = t1 - t0;
    __threadfence();
    *done = 1;
    return;
  }
  if (threadIdx.x >= load_threads) return;
  const uint4* base = load + (size_t)(blockIdx.x - 1) * load_per_cta;
  size_t n = load_per_cta;
  uint32_t acc = 0;
  unsigned long long bytes = 0;
  uint64_t t0 = gtimer_ns();
  while (!*done) {
    // UNR independent 16 B loads in flight per thread: one CTA per SM needs the
    // memory-level parallelism to reach full HBM bandwidth.
    constexpr int UNR = 4;
    for (size_t i = threadIdx.x; i + (UNR - 1) * load_threads < n; i += UNR * load_threads) {
      uint4 v[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u)
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w)
                     : "l"(base + i + u * load_threads));
#pragma unroll
      for (int u = 0; u < UNR; ++u) acc ^= v[u].x + v[u].w;
      bytes += 16 * UNR;
      if ((i & 0xFFFF) == threadIdx.x && *done) break;
    }
  }
  uint64_t t1 = gtimer_ns();
  if (acc == 0x9e3779b9u) pad[0] = 1;
  atomicAdd(bytes_out, bytes);
  if (threadIdx.x == 0) ns_out[blockIdx.x] = t1 - t0;
}

__global__ void scatter_kernel(uint32_t* arr, const uint32_t* pos, const uint32_t* val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) arr[pos[i]] = val[i];
}

__global__ void read_kernel(const uint4* p, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc ^= p[i].x;
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

// Random cycle over `lines` lines chosen from `footprint` bytes (all of them if lines covers it).
uint32_t build_cycle(uint32_t* d, size_t footprint, size_t lines, std::mt19937_64& rng) {
  size_t total = footprint / 128;
  std::vector<uint32_t> sel;
  if (lines >= total) {
    sel.resize(total);
    std::iota(sel.begin(), sel.end(), 0u);
  } else {
    std::uniform_int_distribution<uint64_t> u(0, total - 1);
    while (sel.size() < lines * 2) sel.push_back((uint32_t)u(rng));
    std::sort(sel.begin(), sel.end());
    sel.erase(std::unique(sel.begin(), sel.end()), sel.end());
    std::shuffle(sel.begin(), sel.end(), rng);
    sel.resize(std::min(lines, sel.size()));
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
  scatter_kernel<<<(unsigned)((n + 255) / 256), 256>>>(d, dp, dv, (int)n);
  CK(cudaDeviceSynchronize());
  cudaFree(dp);
  cudaFree(dv);
  return pos[0];
}

int main() {
  DeviceInfo d = device_info();
  double mhz = measure_sm_clock_mhz();
  Json j;
  j.obj();
  emit_header(j, "loaded");
  j.kv("sm_clock_mhz_measured", mhz);

  const size_t chase_bytes = 4ull << 30, load_bytes = 8ull << 30;
  uint32_t* chase;
  uint4* load;
  uint32_t *done, *sink;
  uint64_t *probe, *ns;
  unsigned long long* bytes;
  CK(cudaMalloc(&chase, chase_bytes));
  CK(cudaMalloc(&load, load_bytes));
  CK(cudaMemset(load, 1, load_bytes));
  CK(cudaMalloc(&done, 4));
  CK(cudaMalloc(&sink, 64));
  CK(cudaMalloc(&probe, 16));
  CK(cudaMalloc(&ns, 8 * d.sm_count));
  CK(cudaMalloc(&bytes, 8));
  size_t smem = d.p.sharedMemPerMultiprocessor / 2 + 4096;
  if (smem > d.p.sharedMemPerBlockOptin) smem = d.p.sharedMemPerBlockOptin;
  CK(cudaFuncSetAttribute(loaded_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  std::mt19937_64 rng(77);

  for (int target = 0; target < 2; ++target) {
    const char* name = target == 0 ? "hbm" : "l2";
    j.key(name);
    j.arr();
    std::vector<std::pair<int, int>> pts = {{0, 0}};
    for (int c : {8, 16, 32, 64, 96, 128, 147})
      for (int t : {256, 1024})
        if (c < d.sm_count) pts.push_back({c, t});
    for (auto [ctas, threads] : pts) {
      int hops;
      uint32_t start;
      size_t per_cta;
      if (target == 0) {
        hops = 4096;
        start = build_cycle(chase, chase_bytes, hops + 1, rng);
        per_cta = ctas ? load_bytes / 16 / ctas : 0;
        read_kernel<<<1024, 512>>>(load, (512ull << 20) / 16, sink);  // flush L2 with unrelated data
      } else {
        hops = 4096;
        size_t fp = 16ull << 20;
        start = build_cycle(chase, fp, fp / 128, rng);
        per_cta = ctas ? (48ull << 20) / 16 / ctas : 0;
        read_kernel<<<1024, 512>>>((const uint4*)chase, fp / 16, sink);
        read_kernel<<<1024, 512>>>(load, (48ull << 20) / 16, sink);
      }
      CK(cudaMemset(done, 0, 4));
      CK(cudaMemset(bytes, 0, 8));
      CK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      loaded_kernel<<<1 + ctas, 1024, smem>>>(chase, start, hops, load, per_cta, threads, done, probe, bytes, ns);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      CK(cudaGetLastError());
      uint64_t pc;
      unsigned long long by;
      CK(cudaMemcpy(&pc, probe, 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&by, bytes, 8, cudaMemcpyDeviceToHost));
      std::vector<uint64_t> h(1 + ctas);
      CK(cudaMemcpy(h.data(), ns, 8 * (1 + ctas), cudaMemcpyDeviceToHost));
      double secs = 0;
      for (int i = 1; i <= ctas; ++i) secs = std::max(secs, h[i] * 1e-9);
      double gbps = ctas ? by / secs / 1e9 : 0;
      double lat = (double)pc / hops;
      j.obj();
      j.kvi("load_ctas", ctas);
      j.kvi("threads_per_cta", threads);
      j.kv("load_GBps", gbps);
      j.kv("probe_latency_cycles", lat);
      j.kv("probe_latency_ns", lat / mhz * 1e3);
      j.end_obj();
      fprintf(stderr, "%-3s %3d CTAs x %4d thr: %7.0f GB/s  probe %6.1f cyc (%5.0f ns)\n", name, ctas, threads, gbps, lat,
              lat / mhz * 1e3);
    }
    j.end_arr();
  }
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
