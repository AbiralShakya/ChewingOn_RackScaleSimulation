// Per-warp instruction timelines. Each kernel is straight-line code (fully
// unrolled, no data-dependent control flow) with clock64 probes (CS2R
// SR_CLOCKLO) between phases. Every warp writes its probe values to
// out[global_warp * 64 + p]. The simulator runs the same SASS and reports
// the issue cycle of each probe instruction for every warp; the two
// timelines are compared probe by probe (timeline.py).
#include <cstdint>

#define NP 64
#define PROBE(i) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[i])::"memory")
#define FINISH                                                                         \
  {                                                                                    \
    uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;                         \
    if ((threadIdx.x & 31) == 0) {                                                     \
      _Pragma("unroll") for (int i = 0; i < NPROBE; i++) out[w * NP + i] = (uint32_t)t[i]; \
      out[w * NP + NP - 1] = __smid_();                                                \
    }                                                                                  \
  }

__device__ __forceinline__ uint32_t __smid_() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

// Dependent FP32/INT chains with a growing amount of ILP per phase.
#define NPROBE 17
extern "C" __global__ void t_alu(uint32_t *out, const float *in, int k) {
  uint64_t t[NPROBE];
  float x0 = in[threadIdx.x & 255], x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
  float a = in[256 + (threadIdx.x & 31)], b = in[288 + (threadIdx.x & 31)];
  uint32_t u = __float_as_uint(x0), v = __float_as_uint(a);
  {  // loads done before probe 0: a store of their values, then a barrier
    __shared__ uint32_t sink_[1024];
    sink_[threadIdx.x & 1023] = __float_as_uint(x0 + a + b);
    __syncthreads();
    if (sink_[(threadIdx.x + 1) & 1023] == 0x9e3779b9u) out[NP * 4096 + 1] = 1;  // keeps the store
  }
  PROBE(0);
#pragma unroll
  for (int p = 0; p < 4; p++) {
#pragma unroll
    for (int i = 0; i < 16; i++) x0 = fmaf(x0, a, b);  // 1 chain
    PROBE(1 + 4 * p);
#pragma unroll
    for (int i = 0; i < 16; i++) { x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); }  // 2 chains
    PROBE(2 + 4 * p);
#pragma unroll
    for (int i = 0; i < 16; i++) { x0 = fmaf(x0, a, b); x1 = fmaf(x1, b, a); x2 = fmaf(x2, a, b); x3 = fmaf(x3, b, a); }
    PROBE(3 + 4 * p);
#pragma unroll
    for (int i = 0; i < 16; i++) { u = u * v + 0x9e3779b9u; v = (v << 3) ^ (u >> 7); }  // IMAD/SHF/LOP3
    PROBE(4 + 4 * p);
  }
  if (x0 + x1 + x2 + x3 == 1.234f || u == v) out[NP * 4096] = 1;
  FINISH
}
#undef NPROBE

// Softmax-like: scale, max via shuffles, exp2, sum via shuffles, rescale.
#define NPROBE 13
extern "C" __global__ void t_softmax(uint32_t *out, const float *in, int k) {
  uint64_t t[NPROBE];
  float s[16];
#pragma unroll
  for (int i = 0; i < 16; i++) s[i] = in[(threadIdx.x * 16 + i) & 4095];
  {
    float z = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) z += s[i];
    __shared__ float sink_[1024];  // loads done before probe 0: store, then barrier
    sink_[threadIdx.x & 1023] = z;
    __syncthreads();
    if (sink_[(threadIdx.x + 1) & 1023] == 1.2345f) out[NP * 4096 + 1] = 1;  // keeps the store
  }
  PROBE(0);
#pragma unroll
  for (int r = 0; r < 3; r++) {
    float m = s[0];
#pragma unroll
    for (int i = 1; i < 16; i++) m = fmaxf(m, s[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    PROBE(1 + 4 * r);
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) { s[i] = exp2f(s[i] * 1.4426950f - m); sum += s[i]; }
    PROBE(2 + 4 * r);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    PROBE(3 + 4 * r);
    float inv = __frcp_rn(sum);
#pragma unroll
    for (int i = 0; i < 16; i++) s[i] = s[i] * inv + 0.25f * (float)r;
    PROBE(4 + 4 * r);
  }
  float acc = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) acc += s[i];
  if (acc == 1.234f) out[NP * 4096] = 1;
  FINISH
}
#undef NPROBE

// Shared memory: stores, barrier, dependent loads, conflicting loads.
#define NPROBE 11
extern "C" __global__ void t_smem(uint32_t *out, const uint32_t *in, int k) {
  __shared__ uint32_t sm[8192];
  uint64_t t[NPROBE];
  uint32_t x = in[threadIdx.x & 1023];
  sm[(threadIdx.x + 4096) & 8191] = x;  // load done before probe 0: store, then barrier
  __syncthreads();
  PROBE(0);
#pragma unroll
  for (int i = 0; i < 8; i++) sm[(threadIdx.x + i * blockDim.x) & 8191] = (x + i) & 8191;
  PROBE(1);
  __syncthreads();
  PROBE(2);
  uint32_t p = threadIdx.x & 8191;
#pragma unroll
  for (int i = 0; i < 16; i++) p = sm[p] & 8191;  // dependent chain, conflict-free start
  PROBE(3);
  uint32_t acc = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) acc += sm[(threadIdx.x * 33 + i * 7) & 8191];  // independent
  PROBE(4);
#pragma unroll
  for (int i = 0; i < 16; i++) acc += sm[((threadIdx.x & 31) * 32 + i) & 8191];  // 32-way conflicts
  PROBE(5);
#pragma unroll
  for (int i = 0; i < 8; i++) acc += sm[((threadIdx.x & 31) * 2 + i * 64) & 8191];  // 2-way
  PROBE(6);
  __syncthreads();
  PROBE(7);
  uint4 v = reinterpret_cast<uint4 *>(sm)[(threadIdx.x * 3) & 2047];  // LDS.128
  acc += v.x + v.y + v.z + v.w;
  PROBE(8);
#pragma unroll
  for (int i = 0; i < 8; i++) acc += __shfl_sync(0xffffffff, acc, (threadIdx.x + i) & 31);
  PROBE(9);
  __syncthreads();
  PROBE(10);
  if (acc + p == 0x12345) out[NP * 4096] = 1;
  FINISH
}
#undef NPROBE

// Global memory: an L2-resident buffer (in[] is 16 KB), dependent chase,
// then independent loads, then stores.
#define NPROBE 7
extern "C" __global__ void t_global(uint32_t *out, const uint32_t *in, int k) {
  uint64_t t[NPROBE];
  uint32_t p = (threadIdx.x * 97) & 4095;
  PROBE(0);
#pragma unroll
  for (int i = 0; i < 8; i++) p = __ldcg(in + p) & 4095;  // L2 chase (cg: bypass L1)
  PROBE(1);
#pragma unroll
  for (int i = 0; i < 8; i++) p = __ldca(in + p) & 4095;  // L1 chase (warm after first)
  PROBE(2);
  uint32_t acc = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) acc += __ldcg(in + ((p + i * 128 + threadIdx.x) & 4095));
  PROBE(3);
#pragma unroll
  for (int i = 0; i < 8; i++) out[NP * 4096 + 64 + ((threadIdx.x + i * 1024) & 65535)] = acc + i;
  PROBE(4);
#pragma unroll
  for (int i = 0; i < 8; i++) acc ^= __ldcg(in + ((acc + i * 512) & 4095));
  PROBE(5);
  float f = __uint_as_float(acc | 0x3f000000);
#pragma unroll
  for (int i = 0; i < 32; i++) f = fmaf(f, 0.999f, 0.001f);
  PROBE(6);
  if (f == 1.234f) out[NP * 4096] = 1;
  FINISH
}
#undef NPROBE

// Tensor cores (mma.sync bf16) with ldmatrix operands from shared memory.
#define NPROBE 9
extern "C" __global__ void t_hmma(uint32_t *out, const uint32_t *in, int k) {
  __shared__ __align__(128) uint32_t sm[4096];
  uint64_t t[NPROBE];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) sm[i] = in[i & 1023] | 0x3c003c00u;
  __syncthreads();
  float c[4][4] = {};
  PROBE(0);
  uint32_t a[4], b[4][2];
  uint32_t base = (uint32_t)__cvta_generic_to_shared(sm) + ((threadIdx.x & 31) * 16) + ((threadIdx.x / 32) & 7) * 512;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(base));
#pragma unroll
  for (int j = 0; j < 4; j++)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[j][0]), "=r"(b[j][1]) : "r"(base + 2048 + j * 256));
  PROBE(1);
#pragma unroll
  for (int r = 0; r < 3; r++) {
#pragma unroll
    for (int i = 0; i < 8; i++)  // dependent on the same accumulator
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+f"(c[0][0]), "+f"(c[0][1]), "+f"(c[0][2]), "+f"(c[0][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0][0]), "r"(b[0][1]));
    PROBE(2 + 2 * r);
#pragma unroll
    for (int i = 0; i < 4; i++)  // four independent accumulators
#pragma unroll
      for (int j = 0; j < 4; j++)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[j][0]), "r"(b[j][1]));
    PROBE(3 + 2 * r);
  }
  PROBE(8);
  float acc = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) acc += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (acc == 1.234f) out[NP * 4096] = 1;
  FINISH
}
#undef NPROBE
