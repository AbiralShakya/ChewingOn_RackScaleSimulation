// Launch overheads, CTA barrier cost and atomics.
#include <chrono>

#include "common.cuh"

__global__ void empty_kernel() {}

__global__ void bar_kernel(int iters, uint64_t* out) {
  __syncthreads();
  uint64_t t0 = clk64();
  for (int i = 0; i < iters; ++i) __syncthreads();
  uint64_t t1 = clk64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
}

__global__ void atomic_global_kernel(uint32_t* p, int iters, int spread) {
  uint32_t* a = p + (spread ? (blockIdx.x * blockDim.x + threadIdx.x) * 32 : 0);
  for (int i = 0; i < iters; ++i) atomicAdd(a, 1u);
}

__global__ void atomic_shared_kernel(int iters, int spread, uint64_t* out, uint32_t* sink) {
  __shared__ uint32_t s[1024];
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  s[threadIdx.x] = 0;
  __syncthreads();
  uint32_t* a = spread ? &s[threadIdx.x] : &s[0];
  cta_timer_start(&tm);
  for (int i = 0; i < iters; ++i) atomicAdd(a, 1u);
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = span;
    sink[0] = s[0];
  }
}

int main() {
  DeviceInfo d = device_info();
  double mhz = measure_sm_clock_mhz();
  Json j;
  j.obj();
  emit_header(j, "misc");
  j.kv("sm_clock_mhz_measured", mhz);
  uint64_t* out;
  uint32_t *buf, *sink;
  CK(cudaMalloc(&out, 8 * d.sm_count));
  CK(cudaMalloc(&sink, 64));
  size_t nthreads = (size_t)d.sm_count * 4 * 256;
  CK(cudaMalloc(&buf, nthreads * 32 * 4));
  CK(cudaMemset(buf, 0, nthreads * 32 * 4));

  // Launch: synchronous round trip, back-to-back throughput, and CUDA graph replay.
  j.key("launch");
  j.obj();
  {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < 100; ++i) empty_kernel<<<1, 32>>>();
    CK(cudaDeviceSynchronize());
    const int n = 2000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
      empty_kernel<<<1, 32>>>();
      CK(cudaDeviceSynchronize());
    }
    auto t1 = std::chrono::steady_clock::now();
    double sync_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
    CK(cudaEventRecord(a));
    for (int i = 0; i < n; ++i) empty_kernel<<<d.sm_count, 128>>>();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms;
    CK(cudaEventElapsedTime(&ms, a, b));
    double b2b_us = 1e3 * ms / n;
    // graph of 100 kernels
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 100; ++i) empty_kernel<<<d.sm_count, 128, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    double graph_us = 1e3 * ms / 2000;
    j.kv("sync_roundtrip_us", sync_us);
    j.kv("back_to_back_us_per_kernel", b2b_us);
    j.kv("graph_us_per_kernel", graph_us);
    fprintf(stderr, "launch: sync %.2f us, back-to-back %.2f us, graph %.2f us per kernel\n", sync_us, b2b_us, graph_us);
  }
  j.end_obj();

  // __syncthreads cost vs CTA size
  j.key("syncthreads_cycles");
  j.obj();
  for (int t : {32, 128, 256, 512, 1024}) {
    std::vector<double> v;
    for (int r = 0; r < 3; ++r) {
      bar_kernel<<<1, t>>>(1024, out);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost));
      v.push_back(c / 1024.0);
    }
    j.kv(std::to_string(t).c_str(), vmin(v));
    fprintf(stderr, "syncthreads %4d threads: %.1f cyc\n", t, vmin(v));
  }
  j.end_obj();

  // Atomics
  j.key("atomics");
  j.obj();
  {
    int iters = 64;
    int grid = d.sm_count * 4, blk = 256;
    double ms = time_ms([&] { atomic_global_kernel<<<grid, blk>>>(buf, iters, 1); });
    j.kv("global_distinct_Gops", (double)grid * blk * iters / ms / 1e6);
    double ms2 = time_ms([&] { atomic_global_kernel<<<grid, blk>>>(buf, iters, 0); });
    j.kv("global_same_address_Gops", (double)grid * blk * iters / ms2 / 1e6);
    fprintf(stderr, "global atomicAdd: distinct %.1f Gop/s, same address %.2f Gop/s\n", (double)grid * blk * iters / ms / 1e6,
            (double)grid * blk * iters / ms2 / 1e6);
    for (int spread : {1, 0}) {
      atomic_shared_kernel<<<d.sm_count, 1024>>>(256, spread, out, sink);
      CK(cudaDeviceSynchronize());
      std::vector<uint64_t> h(d.sm_count);
      CK(cudaMemcpy(h.data(), out, 8 * d.sm_count, cudaMemcpyDeviceToHost));
      double ops_per_clk = 1024.0 * 256 / median(std::vector<double>(h.begin(), h.end()));
      j.kv(spread ? "shared_distinct_ops_per_clk_per_sm" : "shared_same_address_ops_per_clk_per_sm", ops_per_clk);
      fprintf(stderr, "shared atomicAdd %s: %.2f ops/clk/SM\n", spread ? "distinct" : "same-addr", ops_per_clk);
    }
  }
  j.end_obj();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
