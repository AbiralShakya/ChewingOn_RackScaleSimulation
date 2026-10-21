// TMA with tensor maps (cp.async.bulk.tensor.2d): tile-load latency and
// streaming bandwidth for the box shapes GEMM/attention kernels use, with and
// without the 128 B swizzle, and cluster multicast (one CTA loads a tile into
// the shared memory of every CTA in its cluster).
//
// The global tensor is bf16 [rows x 8192]; a box is box_rows x 64 elements
// (128 B inner extent, the widest the 128 B swizzle allows).
#include <cuda.h>

#include "common.cuh"

constexpr int COLS = 8192;  // bf16 elements per row (16 KB)
constexpr int BOX_COLS = 64;  // 128 B

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
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, int x, int y, uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(tm), "r"(x), "r"(y), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void tma_load_2d_mcast(uint32_t dst, const CUtensorMap* tm, int x, int y, uint32_t mbar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(
          dst),
      "l"(tm), "r"(x), "r"(y), "r"(mbar), "h"(mask)
      : "memory");
}

// Latency: one thread loads one tile and waits; `reps` tiles at distinct rows.
__global__ void tmat_lat_kernel(const __grid_constant__ CUtensorMap tm, int box_rows, int row0, int row_step, int reps,
                                uint64_t* out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t mb;
  uint8_t* buf = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  if (threadIdx.x != 0) return;
  uint32_t mbar = smem_u32(&mb);
  mbar_init(mbar, 1);
  uint32_t bytes = box_rows * 128;
  uint64_t total = 0;
  for (int r = 0; r < reps; ++r) {
    uint64_t t0 = clk64();
    mbar_expect_tx(mbar, bytes);
    tma_load_2d(smem_u32(buf), &tm, 0, row0 + r * row_step, mbar);
    mbar_wait(mbar, r & 1);
    total += clk64() - t0;
  }
  out[0] = total / reps;
}

// Streaming: each cluster walks its own band of rows with a STAGES-deep ring.
// mcast = 0: every CTA loads its own tiles. mcast = 1: CTA rank 0 multicasts
// each tile to all CTAs in the cluster; every CTA expects the bytes.
template <int STAGES>
__global__ void tmat_bw_kernel(const __grid_constant__ CUtensorMap tm, int box_rows, int tiles, int csize, int mcast,
                               int band_rows, uint64_t* out) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ __align__(8) uint64_t mb[STAGES];
  uint8_t* buf = (uint8_t*)(((uintptr_t)smem + 1023) & ~(uintptr_t)1023);
  uint32_t rank = csize > 1 ? cluster_ctarank() : 0;
  uint32_t bytes = box_rows * 128;
  if (threadIdx.x == 0)
    for (int s = 0; s < STAGES; ++s) mbar_init(smem_u32(&mb[s]), 1);
  if (csize > 1) cluster_sync();
  else __syncthreads();
  if (threadIdx.x == 0) {
    // Band of rows for this CTA (no multicast) or this cluster (multicast).
    int unit = mcast ? blockIdx.x / csize : blockIdx.x;
    // Tiles walk down the band, then across columns, so every load stays in bounds.
    int row_base = unit * band_rows;
    int rt = band_rows / box_rows;
    bool issuer = !mcast || rank == 0;
    uint64_t t0 = clk64();
    for (int i = 0; i < tiles + STAGES; ++i) {
      int s = i % STAGES;
      if (i >= STAGES) mbar_wait(smem_u32(&mb[s]), ((i / STAGES) - 1) & 1);
      if (i < tiles) {
        mbar_expect_tx(smem_u32(&mb[s]), bytes);
        if (issuer) {
          int y = row_base + (i % rt) * box_rows;
          int x = ((i / rt) % (COLS / BOX_COLS)) * BOX_COLS;
          if (mcast) tma_load_2d_mcast(smem_u32(buf + s * bytes), &tm, x, y, smem_u32(&mb[s]), (uint16_t)((1u << csize) - 1));
          else tma_load_2d(smem_u32(buf + s * bytes), &tm, x, y, smem_u32(&mb[s]));
        }
      }
    }
    out[blockIdx.x] = clk64() - t0;
  }
  if (csize > 1) cluster_sync();
}

static CUtensorMap make_map(void* base, size_t rows, int box_rows, CUtensorMapSwizzle swz) {
  CUtensorMap m;
  cuuint64_t dims[2] = {COLS, rows};
  cuuint64_t strides[1] = {COLS * 2};
  cuuint32_t box[2] = {BOX_COLS, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)r);
    exit(1);
  }
  return m;
}

template <class K>
static void launch_cluster(K kern, int grid, int csize, size_t smem, CUtensorMap tm, int box_rows, int tiles, int mcast,
                           int band_rows, uint64_t* out) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(32);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = csize;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  CK(cudaLaunchKernelEx(&cfg, kern, tm, box_rows, tiles, csize, mcast, band_rows, out));
}

int main() {
  DeviceInfo d = device_info();
  double mhz = measure_sm_clock_mhz();
  Json j;
  j.obj();
  emit_header(j, "tmat");
  j.kv("sm_clock_mhz_measured", mhz);
  if (d.p.major < 9) {
    j.kvs("skipped", "tensor-map TMA needs sm_90+");
    j.end_obj();
    printf("%s\n", j.s.c_str());
    return 0;
  }
  const size_t rows = 262144;  // 4 GB of bf16
  void* base;
  uint64_t* out;
  uint32_t* sink;
  CK(cudaMalloc(&base, rows * COLS * 2));
  CK(cudaMemset(base, 0, rows * COLS * 2));
  CK(cudaMalloc(&out, 8 * d.sm_count));
  CK(cudaMalloc(&sink, 64));
  const int max_smem = (int)d.p.sharedMemPerBlockOptin - 2048;
  CK(cudaFuncSetAttribute(tmat_lat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));

  struct Swz {
    const char* name;
    CUtensorMapSwizzle v;
  } swzs[] = {{"none", CU_TENSOR_MAP_SWIZZLE_NONE}, {"128B", CU_TENSOR_MAP_SWIZZLE_128B}};

  j.key("tile_latency");
  j.arr();
  for (auto& sw : swzs)
    for (int box_rows : {8, 64, 128, 256}) {
      CUtensorMap tm = make_map(base, rows, box_rows, sw.v);
      // L2-warm: the same tile repeatedly (after one warm-up load).
      tmat_lat_kernel<<<1, 32, 64 * 1024>>>(tm, box_rows, 0, 0, 1, out);
      tmat_lat_kernel<<<1, 32, 64 * 1024>>>(tm, box_rows, 0, 0, 16, out);
      CK(cudaDeviceSynchronize());
      uint64_t warm;
      CK(cudaMemcpy(&warm, out, 8, cudaMemcpyDeviceToHost));
      // Cold: distinct rows 4096 apart (64 MB) each repetition.
      tmat_lat_kernel<<<1, 32, 64 * 1024>>>(tm, box_rows, 8192, 4096, 32, out);
      CK(cudaDeviceSynchronize());
      uint64_t cold;
      CK(cudaMemcpy(&cold, out, 8, cudaMemcpyDeviceToHost));
      j.obj();
      j.kvs("swizzle", sw.name);
      j.kvi("box_rows", box_rows);
      j.kvi("bytes", box_rows * 128);
      j.kvi("l2_cycles", (long long)warm);
      j.kvi("hbm_cycles", (long long)cold);
      j.end_obj();
      fprintf(stderr, "tile %-4s %3d x 128 B: L2 %5llu  HBM %5llu cyc\n", sw.name, box_rows, (unsigned long long)warm,
              (unsigned long long)cold);
    }
  j.end_arr();

  j.key("stream");
  j.arr();
  constexpr int STAGES = 4;
  CK(cudaFuncSetAttribute(tmat_bw_kernel<STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
  CK(cudaFuncSetAttribute(tmat_bw_kernel<STAGES>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  for (auto& sw : swzs)
    for (int box_rows : {64, 128, 256})
      for (int csize : {1, 2, 4})
        for (int mcast : {0, 1}) {
          if (csize == 1 && mcast) continue;
          CUtensorMap tm = make_map(base, rows, box_rows, sw.v);
          int grid = d.sm_count / csize * csize;
          int units = mcast ? grid / csize : grid;
          int band_rows = (int)(rows / units) / box_rows * box_rows;
          int tiles = std::min(band_rows / box_rows * (COLS / BOX_COLS), 2048);
          size_t smem = (size_t)STAGES * box_rows * 128 + 1024;
          // One CTA per SM: request most of the SM's shared memory.
          smem = std::max<size_t>(smem, d.p.sharedMemPerMultiprocessor / 2 + 4096);
          smem = std::min<size_t>(smem, max_smem);
          launch_cluster(tmat_bw_kernel<STAGES>, grid, csize, smem, tm, box_rows, tiles, mcast, band_rows, out);
          double ms = time_ms([&] { launch_cluster(tmat_bw_kernel<STAGES>, grid, csize, smem, tm, box_rows, tiles, mcast, band_rows, out); }, 3);
          std::vector<uint64_t> h(grid);
          CK(cudaMemcpy(h.data(), out, 8 * grid, cudaMemcpyDeviceToHost));
          double med = median(std::vector<double>(h.begin(), h.end()));
          double fetched = (double)units * tiles * box_rows * 128;    // bytes read from memory
          double delivered = (double)grid * tiles * box_rows * 128;   // bytes landed in smem
          j.obj();
          j.kvs("swizzle", sw.name);
          j.kvi("box_rows", box_rows);
          j.kvi("cluster", csize);
          j.kvb("multicast", mcast);
          j.kv("fetched_GBps", fetched / ms / 1e6);
          j.kv("delivered_GBps", delivered / ms / 1e6);
          j.kv("delivered_bytes_per_clk_per_sm", (double)tiles * box_rows * 128 / med);
          j.end_obj();
          fprintf(stderr, "stream %-4s %3d rows cluster %d %s: fetched %6.0f GB/s, delivered %6.0f GB/s (%.1f B/clk/SM)\n", sw.name,
                  box_rows, csize, mcast ? "mcast" : "     ", fetched / ms / 1e6, delivered / ms / 1e6,
                  (double)tiles * box_rows * 128 / med);
        }
  j.end_arr();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
