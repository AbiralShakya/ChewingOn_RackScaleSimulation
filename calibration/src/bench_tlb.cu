// TLB reach and miss cost: one thread chases a random cyclic permutation of
// N lines placed one per `stride` bytes (N * 128 B stays L2-resident, so a
// rising per-hop latency is address translation, not data). Sweeps N for
// strides 64 KB, 2 MB and 32 MB over a 16 GB allocation.
#include <cstring>

#include "common.cuh"

__global__ void tlb_chase(const uint64_t* __restrict__ base, uint64_t start, int hops, uint64_t* out) {
  uint64_t p = start;
  // Warm: one full lap (fills L2 and the TLBs as far as they reach).
  for (int i = 0; i < hops; ++i) p = __ldcg((const unsigned long long*)base + p);
  uint64_t t0 = clk64();
  for (int i = 0; i < hops; ++i) p = __ldcg((const unsigned long long*)base + p);
  // Branch on the loaded value before the clock (CS2R does not wait on loads).
  if (p == 0xffffffffffffull) out[2] = p;
  out[0] = clk64() - t0;
  out[1] = p;
}

int main(int argc, char** argv) {
  DeviceInfo d = device_info();
  size_t bytes = (size_t)16 << 30;
  uint64_t* buf;
  CK(cudaMalloc(&buf, bytes));
  uint64_t* out;
  CK(cudaMalloc(&out, 64));
  Json j;
  j.obj();
  emit_header(j, "tlb");
  j.key("tlb");
  j.obj();
  srand(12345);
  for (size_t stride : {(size_t)64 << 10, (size_t)2 << 20, (size_t)32 << 20}) {
    char key[32];
    snprintf(key, sizeof key, "stride_%zuKB", stride >> 10);
    j.key(key);
    j.obj();
    for (int n : {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
      if ((size_t)n * stride > bytes) break;
      // Random cyclic order over the n lines; line k sits at k*stride + jitter
      // (a random 128 B line inside the page so the L2 sets spread).
      std::vector<int> perm(n);
      for (int i = 0; i < n; ++i) perm[i] = i;
      for (int i = n - 1; i > 0; --i) std::swap(perm[i], perm[rand() % (i + 1)]);
      std::vector<uint64_t> where(n);
      size_t lines_per_page = std::max<size_t>(1, stride / 128);
      for (int k = 0; k < n; ++k) where[k] = ((size_t)k * stride + (rand() % lines_per_page) * 128) / 8;
      for (int i = 0; i < n; ++i) {
        uint64_t from = where[perm[i]], to = where[perm[(i + 1) % n]];
        CK(cudaMemcpy(buf + from, &to, 8, cudaMemcpyHostToDevice));
      }
      int hops = std::max(4 * n, 2048);
      std::vector<double> v;
      for (int r = 0; r < 3; ++r) {
        tlb_chase<<<1, 1>>>(buf, where[perm[0]], hops, out);
        CK(cudaDeviceSynchronize());
        uint64_t c;
        CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost));
        v.push_back((double)c / hops);
      }
      char nk[16];
      snprintf(nk, sizeof nk, "%d", n);
      j.kv(nk, vmin(v));
      fprintf(stderr, "stride %8zu KB  pages %5d  span %9.1f MB  %6.1f cyc/hop\n", stride >> 10, n,
              (double)n * stride / 1048576.0, vmin(v));
    }
    j.end_obj();
  }
  j.end_obj();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  (void)d;
  return 0;
}
