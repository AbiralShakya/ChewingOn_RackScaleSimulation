// L2 partition map: hit latency as a function of (SM, address).
//
// A100's 40 MB L2 is two partitions joined by a crossbar link; an SM reaching a
// line homed in the far partition pays extra hops. (H100 and B200/B300 keep a
// split L2 too, across the two dies on Blackwell.) bench_mem reports the
// aggregate near/far histogram; this bench reports the full matrix so the
// simulator can place SMs and address slices on the right side:
//
//   lat[sm][line] for LINES distinct 128 B lines spread over a 64 MB window,
//   every line L2-resident (warmed with ld.global.cg, so L1 is bypassed).
//
// Each SM's latencies split into two clusters (near/far). Lines near for one
// group of SMs are far for the other; the bench emits per-SM near/far means,
// a 0/1 side label per SM and per line (by 2-means on the latency profile),
// and the address bits that best predict the line side.
#include "common.cuh"

constexpr int LINES = 2048;
constexpr int REPS = 3;
constexpr int CHAIN = 8;

__global__ void l2part_kernel(const uint32_t* buf, const uint32_t* offs, uint32_t* lat,
                              int* claimed, uint32_t* sink, uint32_t z) {
  // One warp per CTA; the first CTA to land on each SM records for that SM.
  __shared__ int mine;
  __shared__ uint32_t fence_word;
  uint32_t sm = smid();
  if (threadIdx.x == 0) mine = atomicCAS(&claimed[sm], 0, 1) == 0;
  __syncthreads();
  if (!mine || threadIdx.x != 0) return;
  uint32_t acc = 0;
  // Warm: bring every line into L2 (and resolve TLB) before timing.
  for (int i = 0; i < LINES; ++i) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(buf + offs[i]) : "memory");
    acc += v;
  }
  // Timed: a dependent chain of CHAIN loads on the same line (the buffer is
  // zero, so each load's value is the next offset). Same method as the
  // cpasync/mem chases; an independent load bracketed by clock reads gave
  // pattern-dependent artifacts on sm_80.
  for (int i = 0; i < LINES; ++i) {
    const uint32_t* p = buf + offs[i] + (acc & z);  // z == 0 at run time: depends on warm-up
    uint64_t best = ~0ull;
    for (int r = 0; r < REPS; ++r) {
      uint32_t v = 0;
      uint64_t t0 = clk64();
      v += (uint32_t)(t0 & z);
#pragma unroll
      for (int c = 0; c < CHAIN; ++c)
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p + v) : "memory");
      // The shared store needs v and is ordered before the clock read, so the
      // in-order warp cannot read t1 until the last load has returned.
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(smem_u32(&fence_word)), "r"(v) : "memory");
      uint64_t t1 = clk64();
      acc += v;
      best = min(best, t1 - t0);
    }
    lat[(size_t)sm * LINES + i] = (uint32_t)(best / CHAIN);
  }
  if (acc == 0x12345678u) sink[sm] = acc;
}

int main() {
  DeviceInfo d = device_info();
  const size_t window = 64ull << 20;  // larger than any partition interleave period
  uint32_t* buf;
  CK(cudaMalloc(&buf, window));
  CK(cudaMemset(buf, 0, window));
  // Lines: the first 1024 are consecutive (fine interleave), the rest are
  // spread with a coprime stride over the window (coarse bits).
  std::vector<uint32_t> offs(LINES);
  for (int i = 0; i < LINES / 2; ++i) offs[i] = i * 32;  // 32 words = 128 B
  size_t nlines = window / 128;
  for (int i = LINES / 2; i < LINES; ++i)
    offs[i] = (uint32_t)(((size_t)(i - LINES / 2) * 40961ull + 7) % nlines * 32);
  uint32_t *doffs, *dlat, *sink;
  int* claimed;
  CK(cudaMalloc(&doffs, LINES * 4));
  CK(cudaMemcpy(doffs, offs.data(), LINES * 4, cudaMemcpyHostToDevice));
  size_t nsm_slots = 256;  // %smid can exceed sm_count on some parts
  CK(cudaMalloc(&dlat, nsm_slots * LINES * 4));
  CK(cudaMemset(dlat, 0, nsm_slots * LINES * 4));
  CK(cudaMalloc(&claimed, nsm_slots * 4));
  CK(cudaMemset(claimed, 0, nsm_slots * 4));
  CK(cudaMalloc(&sink, nsm_slots * 4));
  // Launch many one-warp CTAs so every SM gets at least one; extras exit.
  l2part_kernel<<<d.sm_count * 8, 32>>>(buf, doffs, dlat, claimed, sink, 0u);
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> lat(nsm_slots * LINES);
  CK(cudaMemcpy(lat.data(), dlat, lat.size() * 4, cudaMemcpyDeviceToHost));

  // SMs that recorded anything.
  std::vector<int> sms;
  for (size_t s = 0; s < nsm_slots; ++s)
    if (lat[s * LINES] != 0) sms.push_back((int)s);

  // Global 2-means threshold on all samples.
  std::vector<double> all;
  for (int s : sms)
    for (int i = 0; i < LINES; ++i) all.push_back(lat[(size_t)s * LINES + i]);
  double lo = vmin(all), hi = *std::max_element(all.begin(), all.end());
  double c0 = lo, c1 = median(all) + (hi - median(all)) * 0.25;
  for (int it = 0; it < 50; ++it) {
    double s0 = 0, s1 = 0;
    long n0 = 0, n1 = 0;
    for (double v : all) {
      if (std::fabs(v - c0) <= std::fabs(v - c1)) { s0 += v; ++n0; } else { s1 += v; ++n1; }
    }
    if (n0) c0 = s0 / n0;
    if (n1) c1 = s1 / n1;
  }
  double thr = 0.5 * (c0 + c1);

  // Per-line "far for SM 0's side" bits: label SMs by agreement with sms[0].
  auto far = [&](int s, int i) { return lat[(size_t)s * LINES + i] > thr; };
  std::vector<int> side(nsm_slots, -1);
  for (int s : sms) {
    int agree = 0;
    for (int i = 0; i < LINES; ++i) agree += far(s, i) == far(sms[0], i);
    side[s] = agree * 2 >= LINES ? 0 : 1;
  }
  // Line home side: the side of SMs for which it is near (majority vote).
  std::vector<int> home(LINES);
  for (int i = 0; i < LINES; ++i) {
    int v0 = 0, v1 = 0;
    for (int s : sms) {
      if (!far(s, i)) (side[s] == 0 ? v0 : v1)++;
    }
    home[i] = v0 >= v1 ? 0 : 1;
  }
  // Which single address bit (byte address) predicts home best.
  int best_bit = -1;
  double best_acc = 0;
  for (int b = 7; b < 26; ++b) {
    int ok = 0;
    for (int i = 0; i < LINES; ++i) ok += (int)(((offs[i] * 4ull) >> b) & 1) == home[i];
    double a = std::max(ok, LINES - ok) / (double)LINES;
    if (a > best_acc) { best_acc = a; best_bit = b; }
  }

  Json j;
  j.obj();
  emit_header(j, "l2part");
  j.kv("sm_clock_mhz_measured", measure_sm_clock_mhz());
  j.kvi("lines", LINES);
  j.kvi("window_bytes", (long long)window);
  j.kv("cluster_near_cycles", c0);
  j.kv("cluster_far_cycles", c1);
  j.kv("threshold_cycles", thr);
  j.kv("far_minus_near_cycles", c1 - c0);
  j.kvi("best_single_address_bit", best_bit);
  j.kv("best_single_bit_accuracy", best_acc);
  j.key("sm");
  j.arr();
  int n_side0 = 0;
  for (int s : sms) {
    double sn = 0, sf = 0;
    int nn = 0, nf = 0;
    for (int i = 0; i < LINES; ++i) {
      double v = lat[(size_t)s * LINES + i];
      if (v > thr) { sf += v; ++nf; } else { sn += v; ++nn; }
    }
    j.obj();
    j.kvi("smid", s);
    j.kvi("side", side[s]);
    j.kv("near_mean", nn ? sn / nn : NAN);
    j.kv("far_mean", nf ? sf / nf : NAN);
    j.kv("far_fraction", (double)nf / LINES);
    j.end_obj();
    n_side0 += side[s] == 0;
  }
  j.end_arr();
  j.kvi("sms_side0", n_side0);
  j.kvi("sms_side1", (long long)sms.size() - n_side0);
  // Line home bitstring, line order as generated (first half consecutive).
  std::string hs;
  for (int i = 0; i < LINES; ++i) hs += home[i] ? '1' : '0';
  j.kvs("line_home_bits", hs);
  j.end_obj();
  printf("%s\n", j.s.c_str());
  fprintf(stderr, "L2 near %.1f far %.1f cyc (thr %.1f); SMs side0 %d side1 %zu; best bit %d (%.3f)\n",
          c0, c1, thr, n_side0, sms.size() - n_side0, best_bit, best_acc);
  return 0;
}
