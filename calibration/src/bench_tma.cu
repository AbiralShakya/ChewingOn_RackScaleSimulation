// Bulk asynchronous copy (TMA engine, cp.async.bulk) global -> shared.
//
// Latency: one thread issues one copy and waits on its mbarrier, for sizes from
// 128 B to 64 KB, from an L2-resident source and from a cold (HBM) source.
// Bandwidth: one CTA per SM keeps a ring of STAGES copies in flight, streaming a
// private region of a large buffer; per-SM bytes/clk from the CTA's own clock,
// whole-GPU GB/s from events.
#include "common.cuh"

__device__ __forceinline__ void mbar_init(uint32_t mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar), "r"(count));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  uint32_t done;
  do {
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                 : "=r"(done)
                 : "r"(mbar), "r"(phase)
                 : "memory");
  } while (!done);
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(mbar)
               : "memory");
}

__global__ void tma_lat_kernel(const uint8_t* src, size_t stride, uint32_t bytes, int reps, uint64_t* out) {
  extern __shared__ __align__(128) uint8_t buf[];
  __shared__ __align__(8) uint64_t mb;
  uint32_t mbar = smem_u32(&mb);
  if (threadIdx.x != 0) return;
  mbar_init(mbar, 1);
  uint64_t total = 0;
  uint32_t phase = 0;
  for (int r = 0; r < reps; ++r) {
    uint64_t t0 = clk64();
    mbar_expect_tx(mbar, bytes);
    bulk_g2s(smem_u32(buf), src + r * stride, bytes, mbar);
    mbar_wait(mbar, phase);
    uint64_t t1 = clk64();
    phase ^= 1;
    total += t1 - t0;
  }
  out[0] = total / reps;
}

template <int STAGES>
__global__ void tma_bw_kernel(const uint8_t* src, size_t per_cta_bytes, uint32_t chunk, uint64_t* out) {
  extern __shared__ __align__(128) uint8_t buf[];
  __shared__ __align__(8) uint64_t mb[STAGES];
  if (threadIdx.x != 0) return;
  for (int s = 0; s < STAGES; ++s) mbar_init(smem_u32(&mb[s]), 1);
  const uint8_t* base = src + (size_t)blockIdx.x * per_cta_bytes;
  int n = (int)(per_cta_bytes / chunk);
  uint64_t t0 = clk64();
  for (int i = 0; i < n + STAGES; ++i) {
    int s = i % STAGES;
    if (i >= STAGES) mbar_wait(smem_u32(&mb[s]), ((i / STAGES) - 1) & 1);  // stage free
    if (i < n) {
      mbar_expect_tx(smem_u32(&mb[s]), chunk);
      bulk_g2s(smem_u32(buf + s * chunk), base + (size_t)i * chunk, chunk, smem_u32(&mb[s]));
    }
  }
  uint64_t t1 = clk64();
  out[blockIdx.x] = t1 - t0;
}

__global__ void touch_kernel(const uint4* p, size_t n, uint32_t* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc ^= p[i].x;
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

int main() {
  DeviceInfo d = device_info();
  double mhz = measure_sm_clock_mhz();
  Json j;
  j.obj();
  emit_header(j, "tma");
  j.kv("sm_clock_mhz_measured", mhz);
  if (d.p.major < 9) {
    j.kvs("skipped", "cp.async.bulk needs sm_90+");
    j.end_obj();
    printf("%s\n", j.s.c_str());
    return 0;
  }
  size_t big = 8ull << 30;
  uint8_t* src;
  uint64_t* out;
  uint32_t* sink;
  CK(cudaMalloc(&src, big));
  CK(cudaMalloc(&out, 8 * d.sm_count));
  CK(cudaMalloc(&sink, 64));
  CK(cudaMemset(src, 3, big));
  // Opt-in limit covers static + dynamic smem; leave room for the mbarriers.
  const int max_smem = (int)d.p.sharedMemPerBlockOptin - 1024;
  CK(cudaFuncSetAttribute(tma_lat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));

  j.key("bulk_g2s_latency");
  j.arr();
  for (uint32_t bytes = 128; bytes <= 65536; bytes *= 2) {
    // L2-resident: the same 64 KB window, touched first.
    touch_kernel<<<64, 256>>>((const uint4*)src, 65536 / 16, sink);
    tma_lat_kernel<<<1, 32, 65536>>>(src, 0, bytes, 1, out);  // warm
    tma_lat_kernel<<<1, 32, 65536>>>(src, 0, bytes, 32, out);
    CK(cudaDeviceSynchronize());
    uint64_t warm;
    CK(cudaMemcpy(&warm, out, 8, cudaMemcpyDeviceToHost));
    // Cold: a fresh 1 MB-strided region each repetition, far beyond L2.
    tma_lat_kernel<<<1, 32, 65536>>>(src + (1ull << 30), 1 << 20, bytes, 32, out);
    CK(cudaDeviceSynchronize());
    uint64_t cold;
    CK(cudaMemcpy(&cold, out, 8, cudaMemcpyDeviceToHost));
    j.obj();
    j.kvi("bytes", bytes);
    j.kvi("l2_cycles", (long long)warm);
    j.kvi("hbm_cycles", (long long)cold);
    j.end_obj();
    fprintf(stderr, "bulk copy %6u B: L2 %5llu cyc  HBM %5llu cyc\n", bytes, (unsigned long long)warm, (unsigned long long)cold);
  }
  j.end_arr();

  j.key("bulk_g2s_bandwidth");
  j.arr();
  for (uint32_t chunk : {4096u, 16384u, 32768u}) {
    const int STAGES = 4;
    size_t smem = (size_t)STAGES * chunk;
    if ((int)smem > max_smem) continue;
    CK(cudaFuncSetAttribute(tma_bw_kernel<STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
    // Force one CTA per SM with the full opt-in smem.
    size_t per_cta = big / d.sm_count / chunk * chunk;
    tma_bw_kernel<STAGES><<<d.sm_count, 32, max_smem>>>(src, per_cta, chunk, out);
    double ms = time_ms([&] { tma_bw_kernel<STAGES><<<d.sm_count, 32, max_smem>>>(src, per_cta, chunk, out); }, 3);
    std::vector<uint64_t> h(d.sm_count);
    CK(cudaMemcpy(h.data(), out, 8 * d.sm_count, cudaMemcpyDeviceToHost));
    double med = median(std::vector<double>(h.begin(), h.end()));
    // single SM, same pipeline, 256 MB
    size_t one = 256ull << 20;
    tma_bw_kernel<STAGES><<<1, 32, max_smem>>>(src, one, chunk, out);
    CK(cudaDeviceSynchronize());
    uint64_t c1;
    CK(cudaMemcpy(&c1, out, 8, cudaMemcpyDeviceToHost));
    j.obj();
    j.kvi("chunk_bytes", chunk);
    j.kvi("stages", STAGES);
    j.kv("allsm_GBps", per_cta * d.sm_count / ms / 1e6);
    j.kv("allsm_bytes_per_clk_per_sm", per_cta / med);
    j.kv("one_sm_bytes_per_clk", one / (double)c1);
    j.kv("one_sm_GBps", one / (double)c1 * mhz / 1e3);
    j.end_obj();
    fprintf(stderr, "bulk copy bw chunk %5u x%d: all-SM %.0f GB/s (%.1f B/clk/SM), one SM %.1f B/clk\n", chunk, STAGES,
            per_cta * d.sm_count / ms / 1e6, per_cta / med, one / (double)c1);
  }
  j.end_arr();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
