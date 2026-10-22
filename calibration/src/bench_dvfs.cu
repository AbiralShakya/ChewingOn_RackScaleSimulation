// SM clock under load at microsecond resolution (sm_100+ for the UMMA loads).
//
// Every CTA runs a load (idle / FFMA / tcgen05 BF16 / tcgen05 NVFP4 / NVFP4
// K=96) for `ms` milliseconds of %globaltimer; lane 0 of the last warp of
// CTAs 0..7 samples (%globaltimer, %clock64) every ~20000 cycles (~10 us). The host
// turns consecutive samples into an SM clock frequency trace:
//   * steady-state clock per load (median over the last half),
//   * time from load start until the clock settles within 2% of steady state,
//   * the trace itself (100 us bins), for docs/BLACKWELL.md,
//   * %globaltimer granularity (smallest non-zero step seen).
#include <cstring>

#include "common.cuh"
#include "tc05.cuh"

constexpr int MAX_SAMPLES = 1 << 16;

template <int MODE>
__global__ void dvfs_kernel(uint64_t duration_ns, uint64_t* samples, int* nsamp, float* sink) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint32_t slot;
  __shared__ __align__(8) uint64_t mbar_storage;
  __shared__ volatile int stop;
  const int warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  const bool sampler = warp == nwarps - 1 && blockIdx.x < 8;
  if (threadIdx.x == 0) stop = 0;
#ifdef HAS_TCGEN05
  uint8_t* A = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  uint8_t* B = A + 16 * 1024;
  uint32_t mbar = smem_u32(&mbar_storage);
  if (MODE >= 2) {
    for (int i = threadIdx.x; i < 28 * 1024 / 4; i += blockDim.x) ((uint32_t*)A)[i] = 0x38383838u ^ i;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    if (warp == 0) {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&slot)), "r"(512));
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (threadIdx.x == 0) mbar_init(mbar, 1);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
#endif
  __syncthreads();
  uint64_t t0 = gtimer_ns();
  if (sampler) {
    if ((threadIdx.x & 31) == 0) {
      int n = 0;
      uint64_t* out = samples + (size_t)blockIdx.x * MAX_SAMPLES * 2;
      uint64_t gt = t0;
      while (gt - t0 < duration_ns && n < MAX_SAMPLES) {
        uint64_t c = clk64();
        gt = gtimer_ns();
        out[2 * n] = gt;
        out[2 * n + 1] = c;
        ++n;
        uint64_t c1 = c;
        while (clk64() - c1 < 20000) {
        }
      }
      nsamp[blockIdx.x] = n;
    }
    __syncwarp();
    if ((threadIdx.x & 31) == 0) stop = 1;
  } else if (MODE == 1) {
    float x = threadIdx.x, y = x + 1, z = x + 2, w = x + 3, a = 1.0001f, b = 1e-7f;
    while (!stop && gtimer_ns() - t0 < duration_ns) {
#pragma unroll 16
      for (int i = 0; i < 256; ++i) {
        x = fmaf(x, a, b); y = fmaf(y, a, b); z = fmaf(z, a, b); w = fmaf(w, a, b);
      }
    }
    if (x + y + z + w == 1.234f) sink[0] = x;
  } else if (MODE >= 2) {
#ifdef HAS_TCGEN05
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (threadIdx.x == 0) {
      uint32_t t = slot, phase = 0;
      uint64_t ad = smem_desc(smem_u32(A), 128, 384, 0), bd = smem_desc(smem_u32(B), 128, 384, 0);
      constexpr int KIND = MODE == 2 ? KIND_F16 : MODE == 3 ? KIND_NVFP4 : KIND_NVFP4_K96;
      uint32_t idesc = make_idesc(KIND, 128, 256);
      while (gtimer_ns() - t0 < duration_ns) {
        for (int i = 0; i < 64; ++i) umma<KIND, 1>(t, ad, bd, idesc, 1, t + 256, t + 384);
        umma_commit<1>(mbar);
        mbar_wait(mbar, phase);
        phase ^= 1;
      }
    }
#endif
  }
#ifdef HAS_TCGEN05
  if (MODE >= 2) {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(slot), "r"(512));
  }
#endif
}

static const char* MODE_NAMES[] = {"idle", "ffma", "umma_bf16", "umma_nvfp4", "umma_nvfp4_k96"};

template <int MODE>
void run(Json& j, const DeviceInfo& d, double ms, uint64_t* samples, int* nsamp, float* sink) {
  size_t smem = 64 * 1024;
  CK(cudaFuncSetAttribute(dvfs_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CK(cudaMemset(nsamp, 0, 8 * sizeof(int)));
  dvfs_kernel<MODE><<<d.sm_count, 256, smem>>>((uint64_t)(ms * 1e6), samples, nsamp, sink);
  CK(cudaDeviceSynchronize());
  std::vector<int> n(8);
  CK(cudaMemcpy(n.data(), nsamp, 8 * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<uint64_t> h((size_t)8 * MAX_SAMPLES * 2);
  CK(cudaMemcpy(h.data(), samples, h.size() * 8, cudaMemcpyDeviceToHost));
  // Frequency between samples ~20 us apart, per sampling SM; trace of SM 0.
  uint64_t gt_step = ~0ull;
  std::vector<std::pair<double, double>> trace;  // (us since start, MHz) for CTA 0
  std::vector<double> late_all;
  std::vector<double> per_sm_late;
  for (int s = 0; s < 8; ++s) {
    const uint64_t* o = h.data() + (size_t)s * MAX_SAMPLES * 2;
    if (n[s] < 4) continue;
    uint64_t g0 = o[0];
    std::vector<double> late;
    int a = 0;
    for (int i = 1; i < n[s]; ++i) {
      uint64_t dg = o[2 * i] - o[2 * i - 2];
      if (dg > 0 && dg < gt_step) gt_step = dg;
      if (o[2 * i] - o[2 * a] >= 100000 || i == n[s] - 1) {
        double mhz = (double)(o[2 * i + 1] - o[2 * a + 1]) / (double)(o[2 * i] - o[2 * a]) * 1e3;
        double t_us = (o[2 * a] - g0) / 1e3;
        if (s == 0) trace.push_back({t_us, mhz});
        if (t_us > ms * 1e3 / 2) late.push_back(mhz);
        a = i;
      }
    }
    if (!late.empty()) {
      per_sm_late.push_back(median(late));
      late_all.insert(late_all.end(), late.begin(), late.end());
    }
  }
  double steady = late_all.empty() ? 0 : median(late_all);
  double settle_us = -1;
  for (size_t i = 0; i < trace.size(); ++i) {
    bool ok = true;
    for (size_t k = i; k < trace.size(); ++k)
      if (fabs(trace[k].second - steady) > 0.02 * steady) { ok = false; break; }
    if (ok) { settle_us = trace[i].first; break; }
  }
  j.key(MODE_NAMES[MODE]);
  j.obj();
  j.kv("duration_ms", ms);
  j.kv("steady_mhz", steady);
  j.kv("settle_us", settle_us);
  j.kv("min_mhz", trace.empty() ? 0 : std::min_element(trace.begin(), trace.end(), [](auto& x, auto& y) { return x.second < y.second; })->second);
  j.kv("globaltimer_step_ns", (double)gt_step);
  j.key("per_sm_steady_mhz");
  j.arr();
  for (double x : per_sm_late) j.num(x);
  j.end_arr();
  j.key("trace_us_mhz");
  j.arr();
  for (auto& p : trace) {
    j.arr();
    j.num(p.first);
    j.num(p.second);
    j.end_arr();
  }
  j.end_arr();
  j.end_obj();
  fprintf(stderr, "%-16s steady %7.1f MHz  settle %8.1f us  min %7.1f  (%zu points; globaltimer step %llu ns)\n",
          MODE_NAMES[MODE], steady, settle_us,
          trace.empty() ? 0 : std::min_element(trace.begin(), trace.end(), [](auto& x, auto& y) { return x.second < y.second; })->second,
          trace.size(), (unsigned long long)gt_step);
}

int main(int argc, char** argv) {
  DeviceInfo d = device_info();
  double ms = argc > 1 ? atof(argv[1]) : 1000.0;
  uint64_t* samples;
  int* nsamp;
  float* sink;
  CK(cudaMalloc(&samples, (size_t)8 * MAX_SAMPLES * 2 * 8));
  CK(cudaMalloc(&nsamp, 8 * sizeof(int)));
  CK(cudaMalloc(&sink, 64));
  Json j;
  j.obj();
  emit_header(j, "dvfs");
  j.key("dvfs");
  j.obj();
  run<0>(j, d, ms / 4, samples, nsamp, sink);
  run<1>(j, d, ms, samples, nsamp, sink);
  if (d.p.major >= 10 && d.p.major < 12) {
    run<2>(j, d, ms, samples, nsamp, sink);
    run<3>(j, d, ms, samples, nsamp, sink);
    if (d.p.major == 10 && d.p.minor == 3) run<4>(j, d, ms, samples, nsamp, sink);
  }
  j.end_obj();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
