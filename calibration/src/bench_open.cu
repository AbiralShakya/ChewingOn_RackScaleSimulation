// The open microbenchmarks of docs/CALIBRATION.md that need new kernels:
//
//   1  launch / tail: every CTA of an FFMA and an EX2 throughput kernel stamps
//      %smid, %clock64 and %globaltimer at entry and exit, 10 runs each.
//      Separates start skew (first to last CTA entry), tail (median to last
//      exit) and per-SM clock (clock64 over globaltimer for each CTA), which
//      ncu's sm__cycles_elapsed averages cannot.
//   2  per-SM clock under tensor load: the same stamps on an HMMA-heavy kernel
//      (mma.sync BF16, the closest stand-in for FA4 a standalone binary has),
//      so per-SM MHz under tensor load reads as clock spread or start skew.
//   4  DRAM latency distribution: per-hop latency of a cold random chase over
//      a 1 GiB footprint, idle and with every other SM streaming HBM. A bulk
//      copy finishes with its slowest line, so the tail matters, not the mean.
//   8  MIO arbitration: conflict-free ld.shared from warps on 1, 2 and 4 SMSPs
//      with a per-warp end stamp; the end-time spread is the arbiter's fairness.
//   9  shared FP64: DFMA throughput with one warp on each of 1, 2 and 4 SMSPs;
//      a single per-SM unit gives the same lanes/clk in every case.
//
// One JSON object on stdout, like every bench here.
#include "common.cuh"

#include <cuda_bf16.h>

constexpr int RUNS = 10;

struct Stamp {
  uint32_t sm;
  uint32_t pad;
  uint64_t c0, c1, t0, t1;
};

__device__ __forceinline__ void stamp_in(Stamp* s) {
  if (threadIdx.x == 0) {
    s[blockIdx.x].sm = smid();
    s[blockIdx.x].c0 = clk64();
    s[blockIdx.x].t0 = gtimer_ns();
  }
}

__device__ __forceinline__ void stamp_out(Stamp* s) {
  __syncthreads();
  if (threadIdx.x == 0) {
    s[blockIdx.x].c1 = clk64();
    s[blockIdx.x].t1 = gtimer_ns();
  }
}

// 1: FFMA and EX2 throughput, the shape of bench_alu's throughput kernels
// (1024 threads, 8 independent chains per thread).
template <int KIND>
__global__ void __launch_bounds__(1024) thr_stamped(float* io, int iters, Stamp* st) {
  stamp_in(st);
  float x[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) x[c] = io[0] + threadIdx.x * 1e-6f + c;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 16; ++u) {
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        if (KIND == 0)
          asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x[c]) : "f"(0.999f), "f"(1e-3f));
        else
          asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x[c]));
      }
    }
  }
  float s = 0;
#pragma unroll
  for (int c = 0; c < 8; ++c) s += x[c];
  if (s == 12345.f) io[1] = s;
  stamp_out(st);
}

// 2: back-to-back mma.sync m16n8k16 BF16 from every warp (tensor-pipe load).
__global__ void __launch_bounds__(512) hmma_stamped(float* io, int iters, Stamp* st) {
  stamp_in(st);
  uint32_t a[4], b[2];
  float d[4][4] = {};
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = 0x3f803f80u ^ (threadIdx.x & 7);
  b[0] = b[1] = 0x3f803f80u;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 16; ++u) {
#pragma unroll
      for (int c = 0; c < 4; ++c)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
  }
  float s = 0;
#pragma unroll
  for (int c = 0; c < 4; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  if (s == 12345.f) io[1] = s;
  stamp_out(st);
}

static void stamps_json(Json& j, const char* name, const std::vector<std::vector<Stamp>>& runs, int sms) {
  j.key(name);
  j.obj();
  std::vector<double> skew, tail, span, mhz_med, mhz_min, mhz_max;
  for (auto& r : runs) {
    uint64_t tmin = ~0ull, tmax = 0, emin = ~0ull, emax = 0;
    std::vector<double> ends, mhz;
    for (auto& s : r) {
      tmin = std::min(tmin, s.t0), tmax = std::max(tmax, s.t0);
      emin = std::min(emin, s.t1), emax = std::max(emax, s.t1);
      ends.push_back((double)s.t1);
      if (s.t1 > s.t0) mhz.push_back(1e3 * (double)(s.c1 - s.c0) / (double)(s.t1 - s.t0));
    }
    skew.push_back((double)(tmax - tmin));
    tail.push_back((double)emax - median(ends));
    span.push_back((double)(emax - tmin));
    mhz_med.push_back(median(mhz));
    mhz_min.push_back(vmin(mhz));
    mhz_max.push_back(*std::max_element(mhz.begin(), mhz.end()));
  }
  j.kvi("ctas", (long long)runs[0].size());
  j.kv("start_skew_ns_median", median(skew));
  j.kv("start_skew_ns_max", *std::max_element(skew.begin(), skew.end()));
  j.kv("tail_ns_median", median(tail));
  j.kv("span_ns_median", median(span));
  j.kv("span_ns_cv", [&] {
    double m = 0, v = 0;
    for (double x : span) m += x;
    m /= span.size();
    for (double x : span) v += (x - m) * (x - m);
    return std::sqrt(v / span.size()) / m;
  }());
  j.kv("sm_mhz_median", median(mhz_med));
  j.kv("sm_mhz_min", vmin(mhz_min));
  j.kv("sm_mhz_max", *std::max_element(mhz_max.begin(), mhz_max.end()));
  // Run 0 in full: per CTA, SM, entry and exit relative to the first entry, MHz.
  j.key("run0");
  j.arr();
  uint64_t t0 = ~0ull;
  for (auto& s : runs[0]) t0 = std::min(t0, s.t0);
  for (auto& s : runs[0]) {
    j.arr();
    j.integer(s.sm);
    j.integer((long long)(s.t0 - t0));
    j.integer((long long)(s.t1 - t0));
    j.num(s.t1 > s.t0 ? 1e3 * (double)(s.c1 - s.c0) / (double)(s.t1 - s.t0) : NAN);
    j.end_arr();
  }
  j.end_arr();
  j.end_obj();
  (void)sms;
}

template <class K>
static std::vector<std::vector<Stamp>> stamped_runs(K kernel, int grid, int block, int iters, float* io) {
  Stamp* d;
  CK(cudaMalloc(&d, grid * sizeof(Stamp)));
  std::vector<std::vector<Stamp>> out;
  kernel<<<grid, block>>>(io, iters, d);  // warm up and settle the clock
  CK(cudaDeviceSynchronize());
  for (int r = 0; r < RUNS; ++r) {
    kernel<<<grid, block>>>(io, iters, d);
    CK(cudaDeviceSynchronize());
    std::vector<Stamp> h(grid);
    CK(cudaMemcpy(h.data(), d, grid * sizeof(Stamp), cudaMemcpyDeviceToHost));
    out.push_back(h);
  }
  cudaFree(d);
  return out;
}

// ---------------------------------------------------------------------------
// 4: DRAM latency histogram

__global__ void chase_hops(const uint64_t* __restrict__ p, int hops, uint32_t* lat, uint64_t start) {
  uint64_t i = start;
  for (int h = 0; h < hops; ++h) {
    uint64_t c0 = clk64();
    uint64_t n = p[i];
    // Branch on the loaded value so the clock read waits for the load.
    if (n == ~0ull) lat[hops] = 1;
    uint64_t c1 = clk64();
    lat[h] = (uint32_t)(c1 - c0);
    i = n;
  }
  lat[hops + 1] = (uint32_t)i;
}

__global__ void hbm_stream(const float4* __restrict__ src, size_t n, volatile int* stop, float* sink) {
  float4 acc = make_float4(0, 0, 0, 0);
  while (!*stop) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
      float4 v = __ldcs(src + i);
      acc.x += v.x;
    }
  }
  if (acc.x == 12345.f) sink[0] = acc.x;
}

__global__ void flush_l2(float4* buf, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    buf[i].x += 1.0f;
}

static void dram_hist(Json& j, int sms) {
  const size_t lines = (1ull << 30) / 128;  // 1 GiB of 128 B lines
  const int hops = 4096;
  std::vector<uint64_t> h(lines * 16);
  // One random cycle over the lines, the chase pointer in each line's first word.
  std::vector<uint64_t> perm(lines);
  for (size_t i = 0; i < lines; ++i) perm[i] = i;
  uint64_t s = 88172645463325252ull;
  for (size_t i = lines - 1; i > 0; --i) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    std::swap(perm[i], perm[s % (i + 1)]);
  }
  for (size_t i = 0; i < lines; ++i) h[perm[i] * 16] = perm[(i + 1) % lines] * 16;
  uint64_t *d, *dummy;
  uint32_t* dl;
  CK(cudaMalloc(&d, h.size() * 8));
  CK(cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dl, (hops + 2) * 4));
  size_t flush_n = (512ull << 20) / 16;
  CK(cudaMalloc(&dummy, flush_n * 16));
  int* stop;
  CK(cudaMallocManaged(&stop, sizeof(int)));
  float* sink;
  CK(cudaMalloc(&sink, 4));
  cudaStream_t bg;
  CK(cudaStreamCreateWithFlags(&bg, cudaStreamNonBlocking));
  j.key("dram_latency");
  j.obj();
  for (int loaded = 0; loaded < 2; ++loaded) {
    std::vector<double> all;
    for (int rep = 0; rep < 4; ++rep) {
      flush_l2<<<sms * 4, 256>>>((float4*)dummy, flush_n);
      CK(cudaDeviceSynchronize());
      if (loaded) {
        *stop = 0;
        hbm_stream<<<sms - 1, 1024, 0, bg>>>((const float4*)dummy, flush_n, stop, sink);
      }
      chase_hops<<<1, 1>>>(d, hops, dl, perm[rep * 9973 % lines] * 16);
      CK(cudaStreamSynchronize(0));
      if (loaded) {
        *stop = 1;
        CK(cudaStreamSynchronize(bg));
      }
      std::vector<uint32_t> l(hops);
      CK(cudaMemcpy(l.data(), dl, hops * 4, cudaMemcpyDeviceToHost));
      for (int i = 16; i < hops; ++i) all.push_back(l[i]);  // skip the TLB-cold start
    }
    std::sort(all.begin(), all.end());
    auto pct = [&](double q) { return all[std::min(all.size() - 1, (size_t)(q * all.size()))]; };
    j.key(loaded ? "loaded" : "idle");
    j.obj();
    j.kvi("hops", (long long)all.size());
    j.kv("p10", pct(0.10));
    j.kv("p50", pct(0.50));
    j.kv("p90", pct(0.90));
    j.kv("p99", pct(0.99));
    j.kv("p999", pct(0.999));
    j.kv("max", all.back());
    double m = 0;
    for (double x : all) m += x;
    j.kv("mean", m / all.size());
    // 32-cycle histogram up to 4096 cycles, then one overflow bin.
    j.key("hist32");
    j.arr();
    std::vector<long long> bins(129, 0);
    for (double x : all) bins[std::min(128, (int)(x / 32))]++;
    for (auto b : bins) j.integer(b);
    j.end_arr();
    j.end_obj();
  }
  j.end_obj();
  cudaFree(d), cudaFree(dl), cudaFree(dummy), cudaFree(stop), cudaFree(sink);
  cudaStreamDestroy(bg);
}

// ---------------------------------------------------------------------------
// 8 and 9: per-SMSP work. Warp w runs on SMSP w % 4. `mask` selects which
// SMSPs' warps work; the rest exit at once. One CTA, 4 * per warps.

__global__ void smem_mio(int iters, int mask, uint64_t* out) {
  __shared__ float buf[32 * 64];
  int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < 32 * 64; i += blockDim.x) buf[i] = i;
  __syncthreads();
  if (!((mask >> (w % 4)) & 1)) return;
  float acc = 0;
  uint32_t base = smem_u32(buf) + lane * 4 + (uint32_t)(clk64() >> 63);
  uint64_t t0 = clk64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int u = 0; u < 16; ++u) {
      float v;
      asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(base + ((u * 32) & 2047) * 4));
      acc += v;
    }
  }
  uint64_t t1 = clk64();
  if (lane == 0) {
    out[2 * w] = t0;
    out[2 * w + 1] = t1;
  }
  if (acc == 12345.f) out[0] = 0;
}

__global__ void dfma_smsp(int iters, int mask, uint64_t* out, double* io) {
  int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (!((mask >> (w % 4)) & 1)) return;
  double x[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) x[c] = io[0] + lane + c;
  uint64_t t0 = clk64();
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < 8; ++c) asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(x[c]) : "d"(0.999), "d"(1e-3));
  }
  uint64_t t1 = clk64();
  if (lane == 0) {
    out[2 * w] = t0;
    out[2 * w + 1] = t1;
  }
  double s = 0;
#pragma unroll
  for (int c = 0; c < 8; ++c) s += x[c];
  if (s == 12345.0) io[1] = s;
}

template <class F>
static void per_smsp(Json& j, const char* name, F launch, int per, double ops_per_warp, const char* unit) {
  uint64_t* d;
  int warps = 4 * per;
  CK(cudaMalloc(&d, 2 * warps * 8));
  j.key(name);
  j.obj();
  j.kvs("unit", unit);
  for (int mask : {0x1, 0x3, 0xF}) {
    CK(cudaMemset(d, 0, 2 * warps * 8));
    launch(mask, d);  // warm
    CK(cudaDeviceSynchronize());
    launch(mask, d);
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> h(2 * warps);
    CK(cudaMemcpy(h.data(), d, h.size() * 8, cudaMemcpyDeviceToHost));
    uint64_t t0 = ~0ull, t1 = 0;
    std::vector<double> ends;
    int active = 0;
    for (int w = 0; w < warps; ++w) {
      if (!((mask >> (w % 4)) & 1)) continue;
      ++active;
      t0 = std::min(t0, h[2 * w]);
      t1 = std::max(t1, h[2 * w + 1]);
      ends.push_back((double)h[2 * w + 1]);
    }
    std::sort(ends.begin(), ends.end());
    int smsps = __builtin_popcount(mask);
    char key[16];
    snprintf(key, sizeof key, "smsps_%d", smsps);
    j.key(key);
    j.obj();
    j.kvi("warps", active);
    j.kv("cycles", (double)(t1 - t0));
    j.kv("per_clk_per_sm", active * ops_per_warp / (double)(t1 - t0));
    // Fairness: first to last warp end, as a fraction of the span.
    j.kv("end_spread", (ends.back() - ends.front()) / (double)(t1 - t0));
    j.end_obj();
  }
  j.end_obj();
  cudaFree(d);
}

int main() {
  Json j;
  j.obj();
  emit_header(j, "open");
  DeviceInfo dev = device_info();
  int sms = dev.sm_count;
  float* io;
  CK(cudaMalloc(&io, 64));
  CK(cudaMemset(io, 0, 64));

  // 1 and 2: one 1024-thread (512 for HMMA) CTA per SM, as the ncu references launch them.
  stamps_json(j, "ffma_stamps", stamped_runs(thr_stamped<0>, sms, 1024, 4000, io), sms);
  stamps_json(j, "ex2_stamps", stamped_runs(thr_stamped<1>, sms, 1024, 1000, io), sms);
  stamps_json(j, "hmma_stamps", stamped_runs(hmma_stamped, sms, 512, 4000, io), sms);
  fprintf(stderr, "stamps done\n");

  dram_hist(j, sms);
  fprintf(stderr, "dram histogram done\n");

  const int iters = 20000, per = 4;
  per_smsp(j, "mio_smem", [&](int mask, uint64_t* d) { smem_mio<<<1, 128 * per>>>(iters, mask, d); }, per,
           iters * 16.0 * 32, "ld.shared lanes/clk/SM");
  double* dio;
  CK(cudaMalloc(&dio, 16));
  CK(cudaMemset(dio, 0, 16));
  per_smsp(j, "dfma_smsp", [&](int mask, uint64_t* d) { dfma_smsp<<<1, 128>>>(iters, mask, d, dio); }, 1,
           iters * 8.0 * 32, "DFMA lanes/clk/SM");
  fprintf(stderr, "per-SMSP done\n");
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
