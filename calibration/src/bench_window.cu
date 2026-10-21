// Fixed kernel window: what an (almost) empty kernel costs inside ncu's
// sm__cycles_elapsed window, by launch shape. CTA rasterisation, the
// shared-memory carveout, cluster launch and an FA4-style TMEM/mbarrier
// prologue all land here; the simulator's `launch.kernel_window_cycles`
// comes from these kernels (via ncu_reference.py), not from timing.
//
// Each config is its own template instance so ncu can pick it by name, and
// is launched LAUNCHES times (ncu_reference.py profiles launch 2).
#include <vector>

#include "common.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define HAS_TCGEN05 1
#endif

constexpr int LAUNCHES = 5;

// TAG only names the config; PROLOGUE = 1 adds FA4's setup: TMEM alloc of
// 512 columns, an mbarrier init, a CTA (or cluster) barrier, TMEM dealloc.
template <int TAG, int PROLOGUE, int CLUSTER>
__global__ void window_kernel(uint32_t* sink) {
  extern __shared__ __align__(1024) uint8_t smem[];
  if constexpr (PROLOGUE) {
#ifdef HAS_TCGEN05
    __shared__ uint32_t tmem_slot;
    __shared__ __align__(8) uint64_t mbar;
    const int warp = threadIdx.x / 32;
    if (warp == 0) {
      if constexpr (CLUSTER == 2) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(smem_u32(&tmem_slot)));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
      } else {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(smem_u32(&tmem_slot)));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
      }
    }
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&mbar)));
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    if constexpr (CLUSTER == 2) {
      asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
      __syncthreads();
    }
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    if (warp == 0) {
      if constexpr (CLUSTER == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(tmem_slot));
      } else {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(tmem_slot));
      }
    }
#endif
  }
  // Keep the kernel (and its smem request) from being optimised away.
  if (sink != nullptr && threadIdx.x == 0 && blockIdx.x == 0) sink[0] = smem[0];
}

struct Config {
  const char* name;
  int threads;
  int smem;  // dynamic bytes; -1 = the per-block opt-in maximum
  int cluster;
  bool prologue;
  void (*kern)(uint32_t*);
};

template <int TAG, int P, int C>
void (*K())(uint32_t*) {
  return window_kernel<TAG, P, C>;
}

int main() {
  DeviceInfo d = device_info();
  double mhz = measure_sm_clock_mhz();
  int optin = d.p.sharedMemPerBlockOptin;
  bool tc05 = d.p.major >= 10;
  std::vector<Config> cfgs = {
      {"t128_smem0", 128, 0, 1, false, K<0, 0, 1>()},
      {"t512_smem0", 512, 0, 1, false, K<1, 0, 1>()},
      {"t512_smem_max", 512, -1, 1, false, K<2, 0, 1>()},
  };
  // Thread block clusters are sm_90+.
  if (d.p.major >= 9) cfgs.push_back({"t512_smem_max_cluster2", 512, -1, 2, false, K<3, 0, 2>()});
  if (tc05) {
    cfgs.push_back({"t512_smem_max_cluster2_fa4_prologue", 512, -1, 2, true, K<4, 1, 2>()});
    cfgs.push_back({"t512_smem_max_fa4_prologue", 512, -1, 1, true, K<5, 1, 1>()});
  }
  uint32_t* sink;
  CK(cudaMalloc(&sink, 64));

  Json j;
  j.obj();
  emit_header(j, "window");
  j.kv("sm_clock_mhz_measured", mhz);
  j.key("window");
  j.obj();
  for (auto& c : cfgs) {
    int smem = c.smem < 0 ? optin - 1024 : c.smem;  // static smem of the prologue
    CK(cudaFuncSetAttribute(c.kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (c.cluster > 1) CK(cudaFuncSetAttribute(c.kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    int grid = cal_grid(d.sm_count / c.cluster * c.cluster);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(c.threads);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = c.cluster;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = c.cluster > 1 ? 1 : 0;  // no cluster attribute before sm_90
    auto launch = [&] { CK(cudaLaunchKernelEx(&cfg, c.kern, (uint32_t*)nullptr)); };
    for (int i = 0; i < LAUNCHES; ++i) launch();
    CK(cudaDeviceSynchronize());
    // Event time of a back-to-back stream of them (host-visible, not the ncu window).
    double ms = time_ms([&] {
      for (int i = 0; i < 100; ++i) launch();
    });
    j.key(c.name);
    j.obj();
    j.kvi("grid", grid);
    j.kvi("threads", c.threads);
    j.kvi("smem_dynamic", smem);
    j.kvi("cluster", c.cluster);
    j.kvb("fa4_prologue", c.prologue);
    j.kv("back_to_back_us", ms * 10.0);
    j.end_obj();
    fprintf(stderr, "%-38s back-to-back %6.2f us/kernel\n", c.name, ms * 10.0);
  }
  j.end_obj();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  cudaFree(sink);
  return 0;
}
