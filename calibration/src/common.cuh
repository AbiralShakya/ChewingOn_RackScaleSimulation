// Shared helpers for the calibration microbenchmarks.
//
// Every bench binary prints exactly one JSON object on stdout. Diagnostics go to
// stderr. The merge script (scripts/merge.py) folds the per-bench objects into
// one calibration file the simulator loads.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_),      \
              __FILE__, __LINE__, cudaGetErrorString(e_));                       \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

__device__ __forceinline__ uint64_t clk64() {
  uint64_t c;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
  return c;
}

__device__ __forceinline__ uint32_t clk32() {
  uint32_t c;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(c)::"memory");
  return c;
}

__device__ __forceinline__ uint64_t gtimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)::"memory");
  return t;
}

__device__ __forceinline__ uint32_t smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// CTA-wide elapsed cycles for throughput kernels. Reading the clock in one warp
// is wrong under a greedy scheduler: the other warps can run thousands of
// cycles before that warp issues its clock read. Every warp records its own
// start after the opening barrier; the span is max(end) - min(start).
struct CtaTimer {
  uint64_t tmin, tmax;
};

__device__ __forceinline__ void cta_timer_init(CtaTimer* t) {
  if (threadIdx.x == 0) {
    t->tmin = ~0ull;
    t->tmax = 0;
  }
}

// Call right after a __syncthreads(); returns this warp's start clock.
__device__ __forceinline__ uint64_t cta_timer_start(CtaTimer* t) {
  uint64_t c = clk64();
  if ((threadIdx.x & 31) == 0) atomicMin((unsigned long long*)&t->tmin, (unsigned long long)c);
  return c;
}

// Call after a __syncthreads() that every warp reaches only once its work has
// retired; returns the CTA span (valid in all threads after the next barrier).
__device__ __forceinline__ uint64_t cta_timer_stop(CtaTimer* t) {
  uint64_t c = clk64();
  if ((threadIdx.x & 31) == 0) atomicMax((unsigned long long*)&t->tmax, (unsigned long long)c);
  __syncthreads();
  return t->tmax - t->tmin;
}

// ---------------------------------------------------------------------------
// Minimal JSON writer: enough for nested objects/arrays of numbers and strings.
// ---------------------------------------------------------------------------
struct Json {
  std::string s;
  std::vector<bool> first{true};

  void sep() {
    if (!first.back()) s += ",";
    first.back() = false;
  }
  void key(const char* k) {
    sep();
    s += "\"";
    s += k;
    s += "\":";
    // the value that follows must not emit another separator
    first.back() = true;
    pending_value = true;
  }
  void pre_value() {
    if (pending_value) {
      pending_value = false;
      first.back() = false;
    } else {
      sep();
    }
  }
  void obj() { pre_value(); s += "{"; first.push_back(true); }
  void end_obj() { s += "}"; first.pop_back(); }
  void arr() { pre_value(); s += "["; first.push_back(true); }
  void end_arr() { s += "]"; first.pop_back(); }
  void num(double v) {
    pre_value();
    char b[64];
    if (std::isfinite(v)) snprintf(b, sizeof b, "%.6g", v);
    else snprintf(b, sizeof b, "null");
    s += b;
  }
  void integer(long long v) { pre_value(); s += std::to_string(v); }
  void str(const std::string& v) {
    pre_value();
    s += "\"";
    for (char c : v) {
      if (c == '"' || c == '\\') s += '\\';
      s += c;
    }
    s += "\"";
  }
  void boolean(bool v) { pre_value(); s += v ? "true" : "false"; }

  void kv(const char* k, double v) { key(k); num(v); }
  void kvi(const char* k, long long v) { key(k); integer(v); }
  void kvs(const char* k, const std::string& v) { key(k); str(v); }
  void kvb(const char* k, bool v) { key(k); boolean(v); }

  bool pending_value = false;
};

struct DeviceInfo {
  cudaDeviceProp p;
  int dev = 0;
  int clock_khz = 0;
  int mem_clock_khz = 0;
  int bus_width = 0;
  int sm_count = 0;
  int l2_bytes = 0;
};

inline DeviceInfo device_info() {
  DeviceInfo d;
  CK(cudaGetDevice(&d.dev));
  CK(cudaGetDeviceProperties(&d.p, d.dev));
  CK(cudaDeviceGetAttribute(&d.clock_khz, cudaDevAttrClockRate, d.dev));
  CK(cudaDeviceGetAttribute(&d.mem_clock_khz, cudaDevAttrMemoryClockRate, d.dev));
  CK(cudaDeviceGetAttribute(&d.bus_width, cudaDevAttrGlobalMemoryBusWidth, d.dev));
  d.sm_count = d.p.multiProcessorCount;
  d.l2_bytes = d.p.l2CacheSize;
  return d;
}

inline void emit_header(Json& j, const char* bench) {
  DeviceInfo d = device_info();
  j.kvs("bench", bench);
  j.key("device");
  j.obj();
  j.kvs("name", d.p.name);
  j.kvi("cc_major", d.p.major);
  j.kvi("cc_minor", d.p.minor);
  j.kvi("sm_count", d.sm_count);
  j.kvi("clock_khz_reported", d.clock_khz);
  j.end_obj();
}

// Median of a vector (copies).
inline double median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline double vmin(const std::vector<double>& v) {
  return v.empty() ? NAN : *std::min_element(v.begin(), v.end());
}

// Time a launch functor with events, returning milliseconds per call (median of reps).
template <class F>
inline double time_ms(F launch, int reps = 5, int warmup = 1) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int i = 0; i < warmup; ++i) launch();
  CK(cudaDeviceSynchronize());
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    CK(cudaEventRecord(a));
    launch();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  CK(cudaGetLastError());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return median(t);
}

// Measures the SM clock actually achieved while a busy kernel runs, by
// comparing %clock64 against %globaltimer over a ~20 ms spin on every SM.
__global__ void clock_probe_kernel(uint64_t* out, uint64_t spin_ns) {
  uint64_t t0 = gtimer_ns();
  uint64_t c0 = clk64();
  uint64_t t1;
  float x = threadIdx.x;
  do {
#pragma unroll 16
    for (int i = 0; i < 64; ++i) x = fmaf(x, 0.999f, 0.5f);
    t1 = gtimer_ns();
  } while (t1 - t0 < spin_ns);
  uint64_t c1 = clk64();
  if (threadIdx.x == 0) {
    out[2 * blockIdx.x] = c1 - c0;
    out[2 * blockIdx.x + 1] = t1 - t0;
  }
  if (x == 12345.f) out[0] = 0;  // keep x live
}

// Returns achieved SM clock in MHz (median across SMs).
inline double measure_sm_clock_mhz() {
  DeviceInfo d = device_info();
  uint64_t* dout;
  CK(cudaMalloc(&dout, 2 * sizeof(uint64_t) * d.sm_count));
  clock_probe_kernel<<<d.sm_count, 256>>>(dout, 5'000'000);  // warm the clock up
  clock_probe_kernel<<<d.sm_count, 256>>>(dout, 20'000'000);
  CK(cudaDeviceSynchronize());
  std::vector<uint64_t> h(2 * d.sm_count);
  CK(cudaMemcpy(h.data(), dout, h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  cudaFree(dout);
  std::vector<double> mhz;
  for (int i = 0; i < d.sm_count; ++i) mhz.push_back(1e3 * (double)h[2 * i] / (double)h[2 * i + 1]);
  return median(mhz);
}
