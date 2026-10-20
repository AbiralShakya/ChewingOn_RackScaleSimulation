// Per-instruction latency and issue throughput for the SM's ALU, FMA, MUFU,
// FP64, conversion and shuffle pipes.
//
// Latency: one warp runs a chain of dependent ops (x = op(x, a, b)); the cycle
// count is taken at two chain lengths and the slope is the latency, which
// removes the fixed cost of reading the clock and the loop prologue.
//
// Throughput: one 1024-thread CTA per SM (forced by a large dynamic smem
// request), 8 independent chains per thread, cycles measured per CTA with
// %clock64. Reported as thread-ops per cycle per SM.
//
// ptxas may fuse or fold PTX ops (e.g. two dependent adds into one IADD3).
// scripts/sass_check.py counts the target SASS opcode inside each timed kernel
// and merge.py rescales from "per PTX op" to "per SASS instruction".
#include "common.cuh"

constexpr int UNROLL = 64;
constexpr int ILP = 8;

#define DEF_OP(NAME, SASS, TYPE, CONS, ASM)                                     \
  struct NAME {                                                                  \
    using T = TYPE;                                                              \
    static constexpr const char* name = #NAME;                                   \
    static constexpr const char* sass = SASS;                                    \
    static constexpr const char* ptx = ASM;                                      \
    __device__ __forceinline__ static void step(T& x, T a, T b) {               \
      asm volatile(ASM : "+" CONS(x) : CONS(a), CONS(b));                         \
    }                                                                            \
  };

// clang-format off
DEF_OP(OpFADD,   "FADD",   float,    "f", "add.f32 %0, %0, %1;")
DEF_OP(OpFMUL,   "FMUL",   float,    "f", "mul.f32 %0, %0, %1;")
DEF_OP(OpFFMA,   "FFMA",   float,    "f", "fma.rn.f32 %0, %0, %1, %2;")
DEF_OP(OpFMNMX,  "FMNMX,FMNMX3", float,    "f", "max.f32 %0, %0, %1;")
#if CAL_SM >= 100  // f32x2 packed FP32 is sm_100+
DEF_OP(OpFFMA2,  "FFMA2",  uint64_t, "l", "fma.rn.f32x2 %0, %0, %1, %2;")
DEF_OP(OpFADD2,  "FADD2",  uint64_t, "l", "add.rn.f32x2 %0, %0, %1;")
#endif
DEF_OP(OpHFMA2,  "HFMA2",  uint32_t, "r", "fma.rn.f16x2 %0, %0, %1, %2;")
DEF_OP(OpHFMA2BF,"HFMA2",  uint32_t, "r", "fma.rn.bf16x2 %0, %0, %1, %2;")
DEF_OP(OpHADD2,  "HADD2,HFMA2", uint32_t, "r", "add.rn.f16x2 %0, %0, %1;")
DEF_OP(OpIADD,   "IADD3",  uint32_t, "r", "add.u32 %0, %0, %1;")
DEF_OP(OpIMAD,   "IMAD",   uint32_t, "r", "mad.lo.u32 %0, %0, %1, %2;")
DEF_OP(OpLOP3,   "LOP3",   uint32_t, "r", "lop3.b32 %0, %0, %1, %2, 0x6A;")
DEF_OP(OpSHF,    "SHF",    uint32_t, "r", "shf.l.wrap.b32 %0, %0, %0, %1;")
DEF_OP(OpPRMT,   "PRMT",   uint32_t, "r", "prmt.b32 %0, %0, %1, 0x4321;")
DEF_OP(OpPOPC,   "POPC",   uint32_t, "r", "popc.b32 %0, %0;")
DEF_OP(OpFLO,    "FLO",    uint32_t, "r", "bfind.u32 %0, %0;")
DEF_OP(OpIMNMX,  "IMNMX,VIMNMX",  uint32_t, "r", "max.u32 %0, %0, %1;")
DEF_OP(OpEX2,    "MUFU.EX2",  float, "f", "ex2.approx.ftz.f32 %0, %0;")
DEF_OP(OpRSQ,    "MUFU.RSQ",  float, "f", "rsqrt.approx.ftz.f32 %0, %0;")
// rcp(rcp(x)) folds to x, so RCP is timed as FADD+RCP; merge.py subtracts FADD.
DEF_OP(OpRCP,    "MUFU.RCP",  float, "f", "{.reg .f32 t; add.f32 t, %0, %1; rcp.approx.ftz.f32 %0, t;}")
DEF_OP(OpSQRT,   "MUFU.SQRT", float, "f", "sqrt.approx.ftz.f32 %0, %0;")
DEF_OP(OpLG2,    "MUFU.LG2",  float, "f", "lg2.approx.ftz.f32 %0, %0;")
DEF_OP(OpTANH,   "MUFU.TANH", float, "f", "tanh.approx.f32 %0, %0;")
DEF_OP(OpSIN,    "MUFU.SIN",  float, "f", "sin.approx.ftz.f32 %0, %0;")
DEF_OP(OpDFMA,   "DFMA",   double,   "d", "fma.rn.f64 %0, %0, %1, %2;")
DEF_OP(OpDADD,   "DADD",   double,   "d", "add.f64 %0, %0, %1;")
DEF_OP(OpF2FBF,  "F2FP",   uint32_t, "r", "{.reg .f32 t; mov.b32 t, %0; cvt.rn.bf16x2.f32 %0, t, t;}")
DEF_OP(OpF2I,    "F2I",    uint32_t, "r", "{.reg .f32 t; mov.b32 t, %0; cvt.rzi.s32.f32 %0, t;}")
DEF_OP(OpI2F,    "I2F",    uint32_t, "r", "{.reg .f32 t; cvt.rn.f32.s32 t, %0; mov.b32 %0, t;}")
// A fixed-lane shuffle makes the value warp-uniform and a repeated butterfly is
// the identity; ptxas folds both. A butterfly over a perturbed value (IADD
// first) survives, and merge.py subtracts the IADD3.
DEF_OP(OpSHFL,   "SHFL",   uint32_t, "r", "{.reg .b32 t; add.u32 t, %0, %1; shfl.sync.bfly.b32 %0, t, %2, 0x1f, 0xffffffff;}")
// clang-format on

template <class Op>
__global__ void lat_kernel(typename Op::T* io, int iters, uint64_t* cyc) {
  using T = typename Op::T;
  // Lane-dependent start value: with a warp-uniform chain ptxas may move the
  // work to the uniform datapath or prove a shuffle is the identity.
  T x = io[0] + (T)threadIdx.x, a = io[1], b = io[2];
  __syncwarp();
  uint64_t t0 = clk64();
  x = x + (T)(t0 >> 63);  // opaque zero: chain cannot start before t0
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) Op::step(x, a, b);
  }
  uint64_t t1 = clk64();
  if (threadIdx.x == 0) {
    io[3] = x;
    cyc[0] = t1 - t0;
  }
}

template <class Op>
__global__ void thr_kernel(typename Op::T* io, int iters, uint64_t* cyc) {
  using T = typename Op::T;
  extern __shared__ uint8_t pad[];  // only used to force one CTA per SM
  __shared__ CtaTimer tm;
  cta_timer_init(&tm);
  T x[ILP];
  T a = io[1], b = io[2];
#pragma unroll
  for (int j = 0; j < ILP; ++j) x[j] = io[0] + (T)(j * threadIdx.x);
  __syncthreads();
  uint64_t t0 = cta_timer_start(&tm);
#pragma unroll
  for (int j = 0; j < ILP; ++j) x[j] = x[j] + (T)(t0 >> 63);  // opaque zero
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int k = 0; k < UNROLL; ++k) {
#pragma unroll
      for (int j = 0; j < ILP; ++j) Op::step(x[j], a, b);
    }
  }
  // The predicated store sits before the barrier, so every chain has retired
  // before any warp passes it and reads t1.
  T acc = x[0];
#pragma unroll
  for (int j = 1; j < ILP; ++j) acc = acc + x[j];
  if (acc == (T)0x5a5a) io[4] = acc;  // keep live, practically never taken
  __syncthreads();
  uint64_t span = cta_timer_stop(&tm);
  if (threadIdx.x == 0) {
    cyc[2 * blockIdx.x] = span;
    cyc[2 * blockIdx.x + 1] = smid();
  }
  if (iters < 0) pad[threadIdx.x] = 0;
}

template <class Op>
void run_op(Json& j, const DeviceInfo& d, size_t one_cta_smem) {
  using T = typename Op::T;
  T* io;
  uint64_t* cyc;
  CK(cudaMalloc(&io, 8 * sizeof(T)));
  CK(cudaMalloc(&cyc, 2 * sizeof(uint64_t) * d.sm_count));
  // Values chosen to keep chains finite where the op allows it; timing does not
  // depend on values for these pipes, but finite values avoid surprises.
  T h[8];
  memset(h, 0, sizeof h);
  if constexpr (std::is_same_v<T, float>) { h[0] = 1.0001f; h[1] = 0.9999f; h[2] = 1e-7f; }
  else if constexpr (std::is_same_v<T, double>) { h[0] = 1.0001; h[1] = 0.9999; h[2] = 1e-9; }
  else if constexpr (std::is_same_v<T, uint64_t>) {
    float f[2] = {1.0001f, 0.9999f};
    memcpy(&h[0], f, 8); memcpy(&h[1], f, 8); float g[2] = {1e-7f, 1e-7f}; memcpy(&h[2], g, 8);
  } else { h[0] = 0x3c003c01u; h[1] = 0x3bff3bffu; h[2] = 7u; }
  CK(cudaMemcpy(io, h, sizeof h, cudaMemcpyHostToDevice));

  auto lat = [&](int iters) {
    std::vector<double> v;
    for (int r = 0; r < 5; ++r) {
      lat_kernel<Op><<<1, 32>>>(io, iters, cyc);
      CK(cudaDeviceSynchronize());
      uint64_t c;
      CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
      v.push_back((double)c);
    }
    return vmin(v);
  };
  const int n1 = 8, n2 = 40;
  double c1 = lat(n1), c2 = lat(n2);
  double lat_per_op = (c2 - c1) / ((double)(n2 - n1) * UNROLL);

  const int titers = std::is_same_v<T, double> ? 2 : 16;
  CK(cudaFuncSetAttribute(thr_kernel<Op>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)one_cta_smem));
  std::vector<double> per_sm;
  for (int r = 0; r < 3; ++r) {
    thr_kernel<Op><<<d.sm_count, 1024, one_cta_smem>>>(io, titers, cyc);
    CK(cudaDeviceSynchronize());
    std::vector<uint64_t> hc(2 * d.sm_count);
    CK(cudaMemcpy(hc.data(), cyc, hc.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> rates;
    for (int b = 0; b < d.sm_count; ++b)
      rates.push_back(1024.0 * ILP * UNROLL * titers / (double)hc[2 * b]);
    per_sm.push_back(median(rates));
  }
  double thr = *std::max_element(per_sm.begin(), per_sm.end());

  j.key(Op::name);
  j.obj();
  j.kvs("sass_expected", Op::sass);
  j.kvs("ptx", Op::ptx);
  j.kvs("lat_kernel", std::string("lat_kernel<") + Op::name + ">");
  j.kvs("thr_kernel", std::string("thr_kernel<") + Op::name + ">");
  j.kvi("ptx_ops_per_iter_lat", UNROLL);
  j.kvi("ptx_ops_per_iter_thr", UNROLL * ILP);
  j.kv("latency_cycles_per_ptx_op", lat_per_op);
  j.kv("throughput_ptx_ops_per_clk_per_sm", thr);
  j.kv("throughput_warp_inst_per_clk_per_smsp", thr / 32.0 / 4.0);
  j.kv("lat_raw_cycles_n1", c1);
  j.kv("lat_raw_cycles_n2", c2);
  j.end_obj();
  fprintf(stderr, "%-10s lat %6.2f cyc  thr %7.2f ops/clk/SM\n", Op::name, lat_per_op, thr);
  cudaFree(io);
  cudaFree(cyc);
}

int main() {
  DeviceInfo d = device_info();
  // More than half the SM's shared memory: guarantees one CTA per SM.
  size_t one_cta_smem = d.p.sharedMemPerMultiprocessor / 2 + 4096;
  if (one_cta_smem > d.p.sharedMemPerBlockOptin) one_cta_smem = d.p.sharedMemPerBlockOptin;
  Json j;
  j.obj();
  emit_header(j, "alu");
  j.kv("sm_clock_mhz_measured", measure_sm_clock_mhz());
  j.key("ops");
  j.obj();
  run_op<OpFADD>(j, d, one_cta_smem);
  run_op<OpFMUL>(j, d, one_cta_smem);
  run_op<OpFFMA>(j, d, one_cta_smem);
  run_op<OpFMNMX>(j, d, one_cta_smem);
#if CAL_SM >= 100
  run_op<OpFFMA2>(j, d, one_cta_smem);
  run_op<OpFADD2>(j, d, one_cta_smem);
#endif
  run_op<OpHFMA2>(j, d, one_cta_smem);
  run_op<OpHFMA2BF>(j, d, one_cta_smem);
  run_op<OpHADD2>(j, d, one_cta_smem);
  run_op<OpIADD>(j, d, one_cta_smem);
  run_op<OpIMAD>(j, d, one_cta_smem);
  run_op<OpLOP3>(j, d, one_cta_smem);
  run_op<OpSHF>(j, d, one_cta_smem);
  run_op<OpPRMT>(j, d, one_cta_smem);
  run_op<OpPOPC>(j, d, one_cta_smem);
  run_op<OpFLO>(j, d, one_cta_smem);
  run_op<OpIMNMX>(j, d, one_cta_smem);
  run_op<OpEX2>(j, d, one_cta_smem);
  run_op<OpRSQ>(j, d, one_cta_smem);
  run_op<OpRCP>(j, d, one_cta_smem);
  run_op<OpSQRT>(j, d, one_cta_smem);
  run_op<OpLG2>(j, d, one_cta_smem);
  run_op<OpTANH>(j, d, one_cta_smem);
  run_op<OpSIN>(j, d, one_cta_smem);
  run_op<OpDFMA>(j, d, one_cta_smem);
  run_op<OpDADD>(j, d, one_cta_smem);
  run_op<OpF2FBF>(j, d, one_cta_smem);
  run_op<OpF2I>(j, d, one_cta_smem);
  run_op<OpI2F>(j, d, one_cta_smem);
  run_op<OpSHFL>(j, d, one_cta_smem);
  j.end_obj();
  j.end_obj();
  printf("%s\n", j.s.c_str());
  return 0;
}
