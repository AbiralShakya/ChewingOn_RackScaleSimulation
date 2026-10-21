# Hardware calibration

These microbenchmarks measure a real GPU so the simulator's SM, memory and
tensor-core models load measured latencies and throughputs instead of guesses.
A second step records ground-truth kernels with Nsight Compute: their SASS,
their launch configuration and the cycles ncu saw. The simulator is then
checked by simulating that exact SASS and comparing.

```bash
CUDA_VISIBLE_DEVICES=3 ./run.sh      # build, run all benches, merge, ncu reference set
NCU=0 ./run.sh                       # skip Nsight Compute
```

Output goes to `results/<gpu>-sm<cc>/`:

| file | what it is |
|---|---|
| `calibration.json` | `model`: what the simulator loads (SASS opcode → latency/issue rate, memory levels, tensor-core shapes, TMA, launch). `raw`: every bench's full output |
| `raw/*.json`, `raw/*.log` | one JSON object per bench (or tensor-core case), plus stderr |
| `model.power` | clock, board power and utilization per CUTLASS FP4 GEMM size, power cap, max clock (from `scripts/cutlass_fp4.sh`) |
| `ncu_reference.json` | 12 reference kernels: launch config, ncu `sm__cycles_elapsed`, instruction count, DRAM/L2 bytes, warp-stall breakdown |
| `sass/*.sass` | the SASS of each reference kernel, for the simulator's SASS front end |

The only requirements are CUDA 12.8+ (13.x for sm_103), Python 3 with no extra
packages, and `ncu` if you want the reference set. On Della, `scripts/sync_della.sh`
copies this directory to della-tridao and runs a command there on GPU 3.

## Benches

| bench | measures | method |
|---|---|---|
| `bench_alu` | latency and issue rate of 29 ALU, FMA, MUFU, FP64, conversion and shuffle instructions | dependent chain at two lengths (slope removes the fixed overhead); throughput uses one 1024-thread CTA per SM with 8 chains per thread |
| `bench_mem` | pointer-chase latency over an 8 KB to 4 GB footprint sweep (L1, L2, DRAM), per-hop L2 latency distribution, smem latency, bank conflicts, DSMEM latency, HBM/L2/L1/smem bandwidth, HBM bandwidth vs SM count | random cyclic permutation of 128 B lines; cold footprints are chased after an L2 flush, each line once |
| `bench_tc` | `mma.sync` (HMMA/IMMA), `tcgen05.mma` (UMMA) BF16/TF32/FP8/MXFP8/NVFP4/MXFP4 at several shapes, 1-SM and 2-SM (`cta_group::2`), A from smem or TMEM, and the TMEM load rate | UMMA is issued by one thread with the accumulator in TMEM; round trip is MMA + commit + mbarrier wait |
| `bench_tma` | `cp.async.bulk` global→smem latency (128 B to 64 KB, from L2 and from HBM) and streaming bandwidth (1 SM and all SMs) | 4-stage mbarrier ring |
| `bench_misc` | kernel launch (sync, back-to-back, graph), `__syncthreads` vs CTA size, global and shared atomics | |
| `bench_mma_sync` (sm_80+) | `mma.sync` HMMA f16/bf16/tf32 (incl. f16 accumulate and m16n8k8), IMMA s8/s4, DMMA f64, 2:4 sparse `mma.sp`; `ldmatrix` x1/x2/x4 latency and bandwidth | accumulator-dependent chain for latency; throughput swept over warps (1 to 32) × independent accumulators (1, 2, 4) so the simulator sees how many warps per SMSP saturate the pipe |
| `bench_cpasync` (sm_80+) | `cp.async` (LDGSTS) .cg/.ca, 4/8/16 B: per-hop latency from L2 and DRAM, and staged global→smem streaming bandwidth vs the `ld.global.v4`→`st.shared` path | random line chase (L2 flushed before each DRAM rep); 1 to 8 stage commit/wait_group ring, swept over CTA size and CTAs per SM |
| `bench_l2part` (sm_80+) | L2 hit latency per (SM, address) over 2048 lines: near/far split, which SMs share a side, which address bits pick the side | one thread per SM, dependent 8-load chain per line, 2-means on the latencies |
| `bench_overlap` (sm_80+) | how well MUFU.EX2, FFMA, FMUL and LDSM.x4 co-issue with HMMA.16816 in the same warps (the FlashAttention-2 softmax mix), plus warp collectives: shfl.up scan, shfl.xor max all-reduce, `redux.sync` | 32 warps per SM; cycles per warp-iteration per SMSP for (HMMA, other) mixes vs each stream alone; overlap = (t_h + t_o − t_mix) / min(t_h, t_o) |

### Keeping the measurements honest

ptxas optimizes the PTX a microbenchmark writes. These cases came up while building
the suite, and each one is now guarded:

* **Fusion.** Two dependent `add.u32` become one `IADD3`, and two `max.f32` become one
  `FMNMX3`. `scripts/sass_check.py` counts the target opcode statically in each kernel's
  SASS and PTX, and rescales per-PTX-op numbers to per-SASS-instruction numbers.
* **Folding.** `rcp(rcp(x))`, repeated butterflies, and shuffles of a warp-uniform value
  were all removed outright. Chains now start lane-dependent; RCP, SHFL and SIN are
  timed together with a partner op, and `merge.py` subtracts the partner's latency.
* **Hoisting.** Loop-invariant `ld.shared` and `ld.global.ca` loads were moved out of the
  loop. Each iteration's address now carries an opaque zero derived from the clock.
* **Clock reads.** `CS2R SR_CLOCKLO` does not wait on the scoreboard, so a clock read
  after a load does not mean the load has finished. The per-hop L2 timer branches on
  the loaded value before reading the clock.
* **Scheduler skew.** Under greedy scheduling, warp 0 can read its start clock thousands
  of cycles after the other warps have started. This inflated FP32 throughput to 140/clk/SM,
  above the 128 lanes the SM has. Throughput kernels therefore use `CtaTimer`, which takes
  max(end) − min(start) across all warps.

## B300 (sm_103, della-tridao GPU 3), first results

148 SMs, 2032 MHz max SM clock, 2030 MHz measured under load, 126.5 MB L2,
228 KB smem per SM. Driver 610.57, CUDA 13.3.

**Pipes** (latency in cycles; rate in thread-ops/clk/SM):

| SASS | lat | rate | | SASS | lat | rate |
|---|---|---|---|---|---|---|
| FFMA / FADD / FMUL | 4 | 124–125 | | MUFU.EX2 | 14 | 32 |
| FFMA2 / FADD2 (f32x2) | 4 | 58 / 64 | | MUFU.RSQ/SQRT/LG2/TANH | 18 | 16 |
| HFMA2 (f16x2, bf16x2) | 4 | 64 | | MUFU.RCP / MUFU.SIN | 20 | 16 |
| IADD3, IMAD, LOP3, SHF, PRMT | 4 | 64 | | POPC, FLO, F2I | 18 | 16 |
| FMNMX3, VIMNMX3 | 4 | 64 | | DFMA / DADD | 64 | 1.9 / 1.6 |
| F2FP (bf16x2 pack), I2FP | 4 | 64 | | SHFL.BFLY | 27 | 32 |

Two of these differ from Hopper. MUFU.EX2 runs at twice the rate of the other MUFU
functions, which matters for softmax. FP64 is about 1.2 TFLOP/s for the whole chip.

**Tensor cores** (one CTA per SM, issued back to back):

| instruction | shape | cycles/MMA | FLOP/clk/SM | chip at 2.03 GHz |
|---|---|---|---|---|
| `tcgen05.mma kind::f16` (BF16→F32) | 128×256×16 | 128 | 8192 | 2.46 PFLOP/s |
| `tcgen05.mma kind::f16` | 128×128×16 | 64 | 8192 | |
| `tcgen05.mma kind::f16` | 128×64×16 | 48 | 5461 (N=64 can't fill the array) | |
| `tcgen05.mma kind::f16` | 64×256×16 | 128 | 4096 (M=64 runs at half rate) | |
| `tcgen05.mma kind::tf32` | 128×256×8 | 128 | 4096 | 1.23 PFLOP/s |
| `tcgen05.mma kind::f8f6f4` (E4M3) | 128×256×32 | 128 | 16384 | 4.92 PFLOP/s |
| `tcgen05.mma kind::mxf8f6f4.block_scale` | 128×256×32 | 128 | 16384 | 4.92 PFLOP/s |
| `tcgen05.mma kind::mxf4nvf4.block_scale.scale_vec::4X` (NVFP4) | 128×256×64 | 128 | 32768 | 9.85 PFLOP/s |
| `tcgen05.mma.cta_group::2 kind::mxf4nvf4.block_scale.block16`, K=96 | 256×256×96 | 128 | 49152 | 14.8 PFLOP/s |
| `mma.sync` m16n8k16 BF16 (HMMA) | | lat 20 | 2037 | 1/4 of UMMA |
| `mma.sync` m16n8k8 TF32 | | lat 21 | 1018 | |
| `mma.sync` m16n8k32 S8 (IMMA) | | lat 74 | 504 | |

The round trip for one UMMA (issue, commit, mbarrier wait) is about 428 cycles
at N ≥ 128. A `tcgen05.ld 32x32b.x32` plus `wait::ld` takes 182 cycles.
`tcgen05.mma kind::i8` is rejected by ptxas for sm_103a, so B300 has no INT8 UMMA.
The SASS opcodes are UTCHMMA (f16/tf32), UTCQMMA (f8f6f4/mxf8) and
UTCOMMA.BLOCK16 (NVFP4).

**B300's 1.5× FP4 needs K=96 UMMA on an SM pair.** The standard block-scaled FP4 MMA
(`mxf4nvf4`/`mxf4`, K=64) runs at 32768 FLOP/clk/SM, the B200 rate, whether it is issued
on one SM, on a pair, or with A in TMEM. sm_103 adds a K=96 form, which CUTLASS calls
"ultra" FP4. It is selected by bit 31 (`k_size`) of the block-scaled instruction
descriptor and spelled `.block_scale.block16` / `.block32` in PTX. Only that form, issued
on an SM pair, reaches 1.5×:

| FP4 UMMA | cycles/MMA | FLOP/clk/SM | chip at 2.03 GHz |
|---|---|---|---|
| K=64, 1 SM (128×256) | 128 | 32768 | 9.85 PF |
| K=64, SM pair (256×256) | 128 | 32768 | 9.85 PF |
| K=64, 1 SM, A from TMEM | 128 | 32768 | 9.85 PF |
| K=96, 1 SM (128×256) | 144 | 43691 | 13.1 PF |
| **K=96, SM pair (256×256)** | **128** | **49152** | **14.8 PF** |

BF16 and FP8 are unchanged across the pair and TMEM-A variants (8192 and 16384 FLOP/clk/SM).
This is a simulator rule: FP4 throughput depends on K and on `cta_group`, not just on the dtype.

**FP4 GEMMs hit the 1100 W power cap, so the clock falls.** CUTLASS example 89 (ultra FP4,
which passes verification) was run about 3 s per size with clock and power sampled every
100 ms. The table uses only steady-state samples, meaning those above half the cap:

| GEMM | TFLOP/s | SM MHz median (min) | board W median (max) | power-capped samples |
|---|---|---|---|---|
| 4096³ | 4493 | 2032 (1987) | 1012 (1034) | 44% |
| 8192³ | 6445 | 1717 (1447) | 1064 (1098) | 98% |
| 16384³ | 4418 | 1455 (1042) | 1077 (1116) | 80% |

The cap is 1100 W, idle draw is 261 W, and the maximum clock is 2032 MHz. At 1455 MHz the K=96 pair
peak is 10.6 PF, so the 16k GEMM runs at about 42% of its power-limited peak. The 14.8 PF peak is
only reachable for bursts. A rack-scale model has to carry a power-dependent clock. These numbers
are in `calibration.json` under `model.power`, and `scripts/cutlass_fp4.sh` regenerates them
(raw trace in `raw/cutlass_fp4.clk.csv`).

**Memory:**

| level | latency (cycles) | bandwidth |
|---|---|---|
| smem | 29 (chase, incl. address op) | 128 B/clk/SM, conflicts scale linearly (stride 32 = 32×) |
| L1 hit | 39 | 79 B/clk/SM (`ld.global.ca.v4`) |
| L2 hit, near partition | 298 (repeat access) | 59 B/clk/SM, 17.6 TB/s chip |
| L2 hit, first touch of a far-homed line | ~706 (24% of lines at a 16 MB footprint) | |
| HBM | 852 (~420 ns, incl. TLB) | 6.82 TB/s read, 6.32 write, 6.45 copy (of 7.67 from attributes) |
| DSMEM (cluster peer) | 193 | |

The chase uses `ld.global.cg` (`LDG.E.STRONG.GPU`). On B300 a weak `LDG.E.NA` measures
the same 299 cycles at every footprint, so unlike A100 there is no strong-load penalty
(`ld_weak_noalloc_cycles` in the sweep).

The far-partition first-touch effect is visible in the footprint sweep. The first
pass over a warmed footprint costs about 390 cycles per hop on average, and repeat
passes cost about 298 because the near partition now holds a copy. L1 capacity
falls between 128 and 256 KB, and L2 capacity sits near 128 MB, where the sweep
steps from 298 to 546 to 850 cycles.

A single SM pulls 66 GB/s from HBM with 1024 threads of 16 B loads. Bandwidth then
scales at about 36 GB/s per SM and reaches 4.8 TB/s at 148 SMs with one CTA per SM.
That falls short of the 6.8 TB/s that four CTAs per SM reach, because one CTA per SM
can't keep enough requests in flight.

**TMA** (`cp.async.bulk`): latency is 367 cycles from L2 and about 880 from HBM for
copies up to 4 KB, and 869 / 1773 cycles for 64 KB. Streaming with 16–32 KB chunks
in a 4-stage ring reaches **7.2 TB/s** across the chip, more than `ld.global` gets. One
SM alone streams 85 B/clk.

**Launch:** a synchronous round trip takes 9.1 µs, back-to-back launches take 2.6 µs
per kernel, and CUDA graph replay takes 0.55 µs per kernel. `__syncthreads` costs 15
cycles at 32 threads and 77 at 1024.

## A100 80GB PCIe (sm_80, Della `gpu80`), first results

108 SMs, 1410 MHz max, 1395 MHz measured, 40 MB L2, 1935 GB/s HBM2e spec. Run with
`JOBID=<slurm job> scripts/run_della_slurm.sh` (Della compute nodes lack `cuobjdump`
and `cu++filt`; the script puts login-node copies on `PATH`).

**ALU:** FFMA/FADD/FMUL take 4 cycles at 64/clk/SM, HFMA2 (f16x2) 128/clk/SM, bf16x2
HFMA2 64/clk/SM, DFMA 6 cycles at 32/clk/SM, MUFU and POPC/FLO 16/clk/SM at 17 cycles.
SHFL takes 31 cycles.

**Tensor cores** (`mma.sync`; latency is dependent through the accumulator):

| op | latency | FLOP/clk/SM | % of peak |
|---|---|---|---|
| HMMA m16n8k16 f16/bf16 → f32 | 24 | 2026 | 99 |
| HMMA m16n8k8 f16 → f32 | 17 | 2003 | 98 |
| HMMA m16n8k8 tf32 | 24 | 1013 | 99 |
| IMMA m16n8k32 s8 / m16n8k64 s4 | 24 | 4046 / 8091 | 99 |
| DMMA m8n8k4 f64 | 18 | 126 | 98 |
| HMMA.SP m16n8k32 2:4 (dense-equivalent) | 24 | 4045 | 99 |

Every op reaches peak only at 32 warps × 4 accumulators per SM. `scripts/ncu_compare.py`
recomputes the same rates from `sm__inst_executed_pipe_tensor_op_*` and
`sm__cycles_active`; they agree within 1.5–3 % (ncu also counts the kernel's prologue
and tail), with tensor pipe utilization of 95–98 %. `ldmatrix` takes 25 cycles and moves
125–127 B/clk/SM, the full shared-memory bandwidth.

**Memory:** L1 hit 39 cycles, L2 hit 211 (plain LDG) or 283 (`ld.cg` = `LDG.E.STRONG.GPU`),
DRAM 560–565, smem 29. The L2 map shows two sides about 150 cycles apart
(near ~224, far ~373 per hop in the chain). SMs fall 62/46 into the sides in a pattern
that repeats every 14 SM ids, and no single address bit predicts a line's side (best 74 %),
so the slice hash spans several bits. HBM reads reach 1.70 TB/s (88 % of spec); one SM
pulls 31 GB/s, and bandwidth saturates around 96 SMs.

**cp.async:** from L2, `cp.async.cg` 16 B streams **4.49 TB/s** (30 B/clk/SM), against
2.5 TB/s for `ld.global.v4` + `st.shared`; the 4 B and 8 B forms lose a lot of that (2.2 and
3.3 TB/s). From DRAM, all paths come within 10 % of each other (1.64 to 1.79 TB/s).
Per hop, a cp.async plus wait_group plus LDS costs 243 cycles from L2, cheaper than an
`ld.cg` hop, and about 597 from DRAM.

**Launch:** sync round trip 6.2 µs, back-to-back 1.9 µs, graph 0.94 µs per kernel.
`__syncthreads` takes 24 cycles at 32 threads and 86 at 1024.

**HMMA co-issue** (`bench_overlap`, measured on the SXM4 part, cycles per warp-iteration
per SMSP; HMMA alone: 4 → 33.6, 8 → 65.6):

| mixed with HMMA ×4 | other alone | mixed | overlap |
|---|---|---|---|
| 16 MUFU.EX2 | 128.7 | 132 | 0.89 |
| 16 LDSM.x4 | 256.9 | 258 | 0.95 |
| 16 FFMA | 32.7 | 48 | 0.57 |
| 32 FFMA | 64.7 | 80 | 0.54 |

EX2 and ldmatrix hide behind tensor work almost completely. FFMA and FMUL do not: each
HMMA takes roughly 4 cycles away from the FP32 pipe (80 ≈ 64 + 4 × 4), consistent with
HMMA's register-file reads competing with the FMA datapath's. That cost is what an
FA2-style softmax pays when its rescale FMULs and FFMAs run between MMAs. Warp collectives:
a 32-lane shfl.up inclusive scan takes 171 cycles, a shfl.xor max all-reduce 152, and
`redux.sync.add.u32` 48.

### A100 SXM4 80GB (Della `della-l07g4`)

Della's `a100&gpu80` constraint can also land on SXM4 nodes; that run is in
`results/a100-sxm4-80gb-sm80/`. Per-SM compute and latencies match the PCIe card
(HMMA 2026 FLOP/clk/SM; L2 284, DRAM 555 cycles; L2 sides 224 / 372). HBM is faster:
1.79 TB/s read out of a 2.04 TB/s spec, and cp.async.cg reaches 1.88 TB/s from DRAM. The
clock probe measured 1275 MHz under full load (500 W limit) against 1395 MHz on the PCIe
card, so per-second rates here run below the per-clock ones.

### A100 MIG 3g.40gb slice (Della `intel&gpu40`)

A 3g.40gb slice of an A100 80GB PCIe has 42 SMs, 20 MB of L2 and half the HBM. Compute
per SM is unchanged (HMMA 2026 FLOP/clk/SM, same ALU rates). The memory system is not
just a scaled-down copy: the slice sits on **one L2 partition**. The near/far split
almost disappears (210 vs 230 cycles per hop, against 224 vs 373 on the full GPU), and
the L2 hit latency drops from 284 to 202 cycles (`ld.cg`). DRAM latency is 486 cycles
against 560. Per SM, L2 read bandwidth rises from 18.7 to 34.4 B/clk, and HBM reads reach
858 GB/s, half of the full GPU. cp.async.cg streams 1.87 TB/s from L2 and 0.90 TB/s from
DRAM. Nsight Compute can't profile on the shared MIG node ("driver resource
unavailable"), so the slice has no `ncu_reference.json`.

## Not covered yet

* TMEM stores and `tcgen05.cp`, TMA tensor maps (2-D/5-D, multicast), sparse UMMA
* The clock/power curve for BF16 and FP8 GEMMs (only FP4 is traced so far)
* `mma.sync` with FP8 (ptxas lowers it on sm_103a; not timed)
* NVLink/NVSwitch: these benches cover a single GPU; fabric calibration is a separate step
* Hopper (H100/H200) runs
* A100 NVLink (NVLink 3, NVSwitch) fabric calibration
