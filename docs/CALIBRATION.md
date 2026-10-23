# Calibration

Every preset number carries a provenance tag: `spec` (datasheet), `ubench` (published
microbenchmarks), `est` (estimate) or `calibrated` (measured with `calibration/`).

## Loading measured data

The `calibration/` microbenchmark suite writes
`calibration/results/<gpu>/calibration.json` with schema `rackscale-calibration/v1`. To use it,
point a GPU group at the file:

```yaml
gpus:
  - preset: b300
    count: 8
    calibration: calibration/results/b300-sxm6-ac-sm103/calibration.json
```

`rsim_hw::calibration::apply` maps the fields as follows. Missing fields keep the preset value.

| calibration.json | GpuSpec |
|---|---|
| `model.device.sm_count`, `model.sm_clock_mhz_measured` | `sm_count`, `sm_clock_mhz` |
| `memory.latency_cycles.{l1_hit,l2_hit_repeat,smem}` | `l1.hit_latency`, `l2.hit_latency`, `latency.lds` |
| `memory.latency_cycles.dram − l2_hit_first_touch` | `hbm.latency_ns` |
| `memory.bandwidth.hbm_peak_GBps_from_attrs`, `hbm_read_GBps / peak` | `hbm.bw_gbps`, `hbm.efficiency` |
| `memory.bandwidth.l2_read_bytes_per_clk_per_sm × SMs` | `l2_bytes_per_cycle` |
| `memory.{l2_bytes,smem_per_sm_bytes}` | `l2.size_bytes`, `smem_per_sm_bytes` |
| `sass.<OPCODE>.{latency_cycles, issue_interval_cycles_per_smsp}` | `opcode_timing[OPCODE]`, which overrides class latency/issue interval for ALU/FMA/XU/SHFL ops |
| `tensor.mma_sync_m16n8k16_bf16_f32.latency_cycles` | `tensor.hmma_latency` |
| `tensor.tcgen05_*.flop_per_clk_per_sm_allsm` | `tensor.flops_per_cycle_per_sm[bf16/fp16/tf32/fp8/fp6/fp4]` |
| `launch.back_to_back_us_per_kernel` | `launch_overhead_ns` |
| `syncthreads_cycles.256` | `latency.bar` |

| `memory.l2_partitions.{near,far}_mean_cycles, far_fraction` | `l2.hit_latency` (near), `d2d.extra_latency` (far − near), `d2d.far_fraction` |
| `tma.bulk_g2s_latency` (linear fit, L2-resident) | `tma.bytes_per_cycle` (1/slope), `tma.issue_latency` (intercept − near L2) |
| `tensor.tcgen05_*`: median of `roundtrip_latency_cycles − cycles_per_mma_1sm` | `tensor.tcgen05_latency` (commit visibility after the MMA leaves the engine) |
| `tensor.tcgen05x2_nvfp4_k96_m256n256k96.flop_per_clk_per_sm_allsm` | `tensor.flops_per_cycle_per_sm.fp4_k96` (Blackwell Ultra K=96 FP4 UMMA; FP4 GEMMs use it when present, `fp4_k64` forces the old form) |
| `power.{power_cap_w, idle_w, max_sm_mhz, runs[]}` | `power`: power-capped clock model (below) |
| `power.tcgen05_sm_mhz` | `power.tcgen05_mhz`: the SM clock whenever tcgen05 MMAs run (B300: 1905 MHz) |
| `issue.{rf_banks, reuse_cache, reuse_any_slot, yield_min_stall, tcgen05_queue_depth, tmem, scheduler}` | `issue`: register banks, operand reuse, yield, UMMA issue queue, TMEM port timing, warp scheduler. Written by `calibration/dissect/fit_issue.py` from the dissect measurements; see docs/BLACKWELL.md |

Opcode lookup tries the full opcode first (`MUFU.EX2`) and then the base (`FFMA`).

## Power-capped clocks

Real GEMMs on B300 hit the 1100 W board cap and the SM clock drops (measured with CUTLASS FP4: 2032 MHz at 4096³, 1717 at 8192³, 1455 at 16384³). The model takes `P = idle + D·(f/f_max)³`. Each measured run fixes the `D` its kernel needs at `f_max`. A kernel's sustained clock is the largest `f` with `P ≤ cap`, with `D` interpolated in log(FLOPs) between the runs of the same dtype. A kernel that keeps the tensor cores busy less than 70% of the time draws proportionally less, so its clock drops that fraction of the way (`rsim_gpu::dvfs_clock_mhz`). The kernel is then simulated again at that clock (`rsim_gpu::dvfs_stretch`): HBM and L2 keep their speed in nanoseconds, so only the SM-bound part slows down. In a rack run the factor is computed once per distinct kernel and shared across ranks.

BF16 has no power trace. Its runs are derived: `scripts/fit_sustained_clock.py` divides each cuBLAS BF16 GEMM's Nsight Compute cycles by its back-to-back CUDA-graph time (both from `research/ssm_vs_attention`), bins the resulting clocks by size and writes them to `results/b300-sxm6-ac-sm103/raw/sustained_clock_bf16.json` as runs at the cap (`"derived": true`), which `merge.py` adds to `model.power`:

| GEMM FLOPs | GEMMs | sustained clock |
|---|---|---|
| 1.7·10¹⁰ | 7 | 2032 MHz |
| 5.6·10¹⁰ | 4 | 1932 MHz |
| 1.8·10¹¹ | 6 | 1591 MHz |
| 5.9·10¹¹ | 4 | 1301 MHz |
| 1.6·10¹² | 5 | 1304 MHz |
| 5.0·10¹² | 1 | 1235 MHz |

Shapes of the same size differ by up to 270 MHz (1248 to 1516 MHz at 2–6·10¹¹ FLOPs), which a size-only model cannot capture; that is most of the remaining large-GEMM error below. FP8 and the B200/GB200/H100 presets have no power model, so they run at the full clock.

Synthetic BF16 GEMMs against the 54 measured cuBLAS GEMMs (`python scripts/check_gemms.py --gpu b300`, CUDA-graph timing):

| class | n | median \|error\| | mean error | before the sustained-clock model and epilogue overlap |
|---|---|---|---|---|
| decode (< 10¹⁰ FLOP) | 27 | 3.9% | −0.8% | 4.7%, −2.8% |
| mid | 11 | 19.1% | −4.7% | 17.7%, −4.2% |
| large (≥ 10¹¹ FLOP) | 16 | 11.6% | +5.6% | 19.1%, −19.5% |

## Validation against ncu

`rsim validate calibration/results/<gpu>/ncu_reference.json --gpu <preset> [--calibration <calibration.json>]` replays each reference kernel and compares simulated SM cycles with ncu's `sm__cycles_elapsed.avg`. Every non-window reference now has an exact recording: an NVBit warp path (`path_file`, `.rpath.gz`) on B300, and an Accel-Sim trace (`trace_file`, `.traceg`) on both A100s. The A100 traces run at a reduced size (`trace_env`), and their `ncu` entry is re-measured at that size; the full-size numbers stay in `ncu_full_size`. A kernel with neither is walked from its static SASS. Loop trips are then solved from `smsp__inst_executed.sum`, an instruction count rather than a timing. When that count stops at the first guarded EXIT, the walk takes the exit.

Each kernel is simulated at the SM clock its ncu run actually ran at (`sm__cycles_elapsed.avg / gpu__time_duration.sum`, for kernels longer than 100 us). `ncu --clock-control none` leaves the clock to the driver, and memory-bound cycle counts only compare at the same clock.

Validation charges the measured **kernel window**, the fixed cost of an almost empty kernel inside ncu's window, by launch shape (`calibration/src/bench_window.cu`, `model.launch.kernel_window`). It charges the measured cost minus the model's own time for that empty kernel. The `window_*` rows check this. They aren't independent: the six plain shapes test that the subtraction is consistent, and the two FA4-prologue shapes are the points `tensor.tmem_alloc_latency` is fitted to (1430 cycles for `cta_group::1`, 1990 for `cta_group::2`). The FlashAttention study's 16 recorded kernels are the out-of-sample check, and all are within ±6% (research/flash_attention).

Results (simulated vs measured cycles, signed; `before` is main at 736466d):

| kernel | B300 (path) | before | A100 SXM4 (trace) | before | A100 PCIe (trace) | before |
|---|---|---|---|---|---|---|
| hbm_stream_read | +0.3% | +0.3% | −10.2% | −4.3% | −13.6% | −7.8% |
| hbm_copy | +0.4% | −5.1% | −28.3% | −23.5% | −26.8% | −21.8% |
| cp_async_cg16_dram_stream | n/a | | −7.6% | −3.1% | −8.3% | −3.7% |
| tma_bulk_stream | **−17.7%** | −6.2% | n/a | | n/a | |
| ffma_throughput | −6.5% | −6.5% | +0.7% | +0.8% | +0.9% | +0.9% |
| ex2_throughput | +0.1% | +0.1% | +0.1% | +0.1% | +0.2% | +0.2% |
| dfma_latency_chain | +0.2% | +3.2% | −3.6% | −2.3% | +0.2% | +1.3% |
| smem_conflict_free | −0.8% | +6.8% | −1.1% | +6.4% | −3.4% | +3.9% |
| smem_32way_conflict | −0.3% | −0.3% | −0.0% | −0.0% | −0.1% | −0.1% |
| l2_pointer_chase | +1.8% | −3.7% | +9.2% | +16.3% | +4.5% | +11.8% |
| hmma_bf16_throughput | −0.1% | −0.1% | +0.5% | +0.5% | +0.3% | +0.2% |
| hmma_f16_f32_m16n8k16_w32_ilp4 | n/a | | +0.5% | +0.6% | +1.0% | +1.1% |
| umma_bf16_m128n256k16 | −0.1% | −0.1% | n/a | | n/a | |
| umma_nvfp4_m128n256k64 | −0.0% | −0.0% | n/a | | n/a | |
| window_* (plain shapes) | +0.2% | +0.2% | ±0.5% | ±0.5% | ±0.5% | ±0.6% |
| window_* (FA4 prologue, fit points) | +0.3%, +0.5% | 0.0%, −0.1% | n/a | | n/a | |

FlashAttention on B300 (research/flash_attention recordings; ncu `sm__cycles_elapsed.max`, with `.avg` in brackets):

| kernel | now | before |
|---|---|---|
| FA2 s4096 | +3.6% (+3.7%) | +1.6% |
| FA4 s4096 | +0.2% (+1.5%) | +0.4% |
| FA4 ring step c1024 kv8 / kv64 | −6.0% / −8.5% | −5.4% / −7.3% |
| FA4 ring step c2048 kv8 / kv64 | +5.1% / +1.3% | +6.1% / +2.1% |
| FA4 ring step c4096 kv8 / kv64 | +2.9% / +0.3% | +3.1% / +0.3% |

`hbm_stream_read`, `hbm_copy`, `ex2`, `dfma` and the `umma`/`hmma` throughput kernels are the same programs the calibration measured, so on B300 they are consistency checks, not held-out tests. The held-out checks are the A100 boards, the FlashAttention kernels and `tma_bulk_stream`.

### Splitting each error into steady state and launch/tail

ncu's `smsp__cycles_active.avg` counts the cycles an SMSP has a warp resident. Comparing it with the simulator's per-SMSP active cycles (`RSIM_VALIDATE_STATS=1`, scheduler cycles minus idle) separates the per-SMSP steady state from launch, drain and SM-to-SM skew:

| B300 kernel | steady state | total | reading |
|---|---|---|---|
| ffma_throughput | +0.2% | −6.5% | the FFMA stream is exact; hardware has 12.4K inactive cycles per SMSP where `ex2_throughput` (same launch shape) has 6.1K |
| hbm_copy (main) | −0.4% | −5.1% | per-SMSP rate right; total was short by the copy's lower HBM bandwidth |
| hbm_stream_read | +2.5% | +0.3% | per-SMSP rate 2.5% slow, hardware has 2.5% SM-to-SM tail; the measured bandwidth folds both together |
| tma_bulk_stream | −16.1% | −17.7% | per-copy round trip from HBM too short |
| smem_conflict_free (main) | +7.2% | +6.8% | steady state, fixed by fair MIO arbitration |

### What each fix was

- **FP64 is one unit per SM (dfma +3.2% → +0.2%).** B300's measured DFMA chain is 64 cycles per op with one warp, but full occupancy gives 66.4 cycles per warp instruction per SMSP. A per-SMSP pipe with a 66-cycle interval would make the one-warp chain 66 cycles, so the limit must be a unit the four SMSPs share: one warp instruction every 16.6 cycles per SM. `calibration::apply` marks an opcode `sm_shared` whenever its measured per-SMSP interval exceeds its latency (only DFMA and DADD on B300; nothing on A100).
- **The SMSPs take turns at shared resources (smem_conflict_free +6.8% → −0.8%).** The SM stepped its SMSPs in a fixed order each cycle, so SMSP 0 always got the MIO slot first. In the 32-warp shared-memory stream the last SMSP's warps sat in `mio_throttle` while the first SMSP's reached the barrier early: barrier stalls 6.7 per issue against ncu's 0.37. The order now rotates every cycle; the stall profile matches ncu (mio_throttle 12.0 vs 11.6, barrier 0.1 vs 0.37).
- **Writes cost more HBM time than reads (hbm_copy −5.1% → +0.4%).** The calibration measured read 6818, write 6296 and copy 6446 GB/s but only the read number was used. A written byte now costs `rd/wr` = 1.083 read bytes, and mixed traffic pays a read/write turnaround of 1.6% at a 1:1 mix (from the copy number), scaled by the last window's mix.
- **The L1 trip was counted twice in every L2 hit (B300 l2_pointer_chase −3.7% → +1.8%; A100 +16%/+12% → +9%/+4.5%).** The pointer chases that give `l2_hit_repeat` time a load from issue to use, through L1. The simulator adds `l1.hit_latency` to every global load on top of `l2.hit_latency`, so a simulated L2 hit took 338 cycles on B300 against 299 measured. `l2.hit_latency` is now the measured value minus the L1 trip. HBM latency is the cold chase minus the warm L2 chase, so a simulated cold miss is exactly the measured 850 cycles (it was 805: subtracting the first-touch L2 latency dropped the ~85 cycles every first touch pays).
- **TMA copies follow their own measured curve.** Each line of a bulk copy drew its own near/far L2 partition, and a copy finished with its slowest line, so almost every copy over 512 B paid the 278-cycle far penalty: a 1 KB L2-resident copy took 609 cycles against 315 measured. The far penalty is part of the TMA calibration (intercept and slope fitted to whole copies), so bulk copies no longer add it. The cold (HBM) curve is steeper than the L2 one; the difference is now an in-order fill rate for lines of one copy that miss L2 (`tma.hbm_fill_bytes_per_cycle`, 189 B/clk). `mem.rs` has a test that replays the chases and TMA copies against the measured numbers.

### Open misses and why

- **tma_bulk_stream −17.7% (was −6.2%).** The −6.2% was two errors cancelling: the far-partition penalty on every copy made each 4 KB copy ~100 cycles too slow in isolation, while the stream itself was too fast. With the penalty gone the isolated cold 4 KB copy is 933 cycles against 1029 measured, and the stream's round trip is 1057 against 1260 measured (hardware: 1162 with one SM streaming, so ~100 cycles come from load and ~130 from four copies in flight). The microbenchmarks cannot separate per-copy latency, overlap and TLB cost: the cold-latency bench strides 1 MB per copy. Microbenchmarks 3 and 4 below.
- **ffma_throughput −6.5%.** The steady state is +0.2%. The rest is launch/tail: hardware's SMSPs are inactive for 12.4K of the 82.1K cycles, against 6.1K for `ex2_throughput` with the same launch shape and binary. Not settleable from ncu averages (microbenchmark 1).
- **FA2 +3.6% (was +1.6%).** The fixed SMSP order was hiding an error. An earlier reading blamed fixed-latency `wait` stalls (3.0 per issue against 2.16): that was the stall accounting, which folded each issue cycle into the following stall; counted as Nsight counts it (the issue cycle is `selected`), `wait` matches. What is left is register-bank dispatch: the replay loses 0.88 cycles per issue to bank conflicts against ncu's 0.20 `dispatch_stall`, mostly HMMA's operand reads blocking the LDSM, MUFU and FMUL around it. The bank model was fitted only on FFMA/FMUL streams. Taking HMMA out of it overshoots to −8.0% (dispatch 0.09, math throttle 1.38 vs 1.24), so neither extreme is right and the data cannot place it in between. Microbenchmark 10.
- **FA4 ring steps (c1024 −6.0/−8.5%, c2048 kv8 +5.1%).** Confounded by the recording, not just the model: the persistent kernel gives CTAs different tile counts (c1024: 256 tiles over 148 CTAs, so 108 CTAs do 2 and 40 do 1), but only CTAs 0–3 were recorded and every other CTA replays them. The replay therefore issues 15–16% more instructions than ncu counted (c4096 and s4096: 2.4%), loading L2 and HBM 15% harder than hardware. Microbenchmark 6.
- **FA4 s4096 (+0.2% against the slowest SM, +1.5% against the average).** ncu reports `sm__cycles_elapsed` from 494.8K to 510.2K across SMs. That spread does not track when each SM finishes its work. FA2 s4096's 2048 CTAs leave a wave-quantisation tail: in the simulator SMs finish between 2.14M and 2.46M cycles, yet hardware's per-SM elapsed is 2.375M–2.378M. The FA4 spread is therefore per-SM clocks or counting windows, not work. The simulator's own per-SM finish times for FA4 (496.2K / 504.6K / 511.2K) land close to ncu's min/avg/max, but that is not evidence. FA4's total is known to ±1.5% until microbenchmark 2. The per-role timelines' −5% end error (docs/BLACKWELL.md §10.1) cannot be read as a rate error either; see there.
- **Attention kernels get no power-cap clock drop.** `dvfs_clock_mhz` reads the dtype from the kernel name ("bf16", "fp8", ...) to pick the measured GEMM power run. FA4's CuTe name and cuDNN's `sm100_fmha_*` names carry no dtype, so they keep only the fixed tcgen05 clock (1905 MHz) and never the power-cap drop the GEMMs get. Inferring the dtype from the MMA instructions would fix the lookup, but the GEMM runs are not a measurement of attention's power: FA4 keeps the tensor pipe busy about half the time with heavy MUFU and FMA work beside it. Microbenchmark 11.
- **A100 streams (−8% to −28%) got 5% faster.** These traces run at reduced size and are latency-bound per warp. Removing the double-counted L1 trip took 39 cycles off every L2 access; the streams were already too fast because the model has no per-SM limit on the SM-to-L2 path under load (round trip ~680 cycles in the model against ~940 on hardware).

### Microbenchmarks that would settle the rest

None of these were run: they need the GPUs.

1. **FFMA launch/tail.** `thr_kernel<FFMA>` and `<EX2>` with every CTA stamping `%globaltimer`, `clock64` and `%smid` at entry and exit, 10 runs each, plus ncu with `--clock-control base` five times (`sm__cycles_elapsed.{min,max}`, `smsp__cycles_active.{min,max}`). Separates start skew, tail and replay-pass variance.
2. **Per-SM clocks under FA4.** The same stamps on all 148 SMs during FA4 s4096: per-SM MHz says whether `sm__cycles_elapsed` spread is clock (compare with min) or skew (compare with max).
3. **TMA from HBM.** Cold latency for 128 B–64 KB with a sequential source as well as the 1 MB stride (TLB cost); one SM streaming 4 KB and 16 KB chunks with 1, 2, 4 and 8 stages; all 148 SMs with the same stages.
4. **DRAM latency distribution.** Histogram of a cold chase over ≥1 GB, idle and under the loaded-probe loads (the probe records only the mean). A copy finishes with its slowest line, so the tail matters, not the mean.
5. **Near/far partitions for TMA vs LDG.** TMA L2-resident copies of 1–64 KB from SMs on both dies with the buffer homed on each die: does a bulk copy ever pay the far partition?
6. **Re-record the FA4 ring steps** with one CTA of each tile count (e.g. CTAs 0, 1, 146, 147 for c1024), so the replay does each CTA's real work.
7. **FA2 per-PC stalls.** ncu source-level stall sampling (`smsp__pcsamp_*`) on FA2 s4096, to compare per-instruction `wait` with the simulator's per-PC stats.
8. **MIO arbitration.** `smem_conflict_free` with warps on 1, 2 and 4 SMSPs and a per-warp end stamp, to measure the arbiter's fairness directly.
9. **Shared FP64.** DFMA throughput with warps on 1, 2 and 4 SMSPs: a shared unit gives the same ~1.93 lanes/clk per SM in every case.
10. **HMMA operand reads vs the register banks.** `calibration/src/bench_overlap.cu` (run so far only on A100) on B300: an HMMA.16816 stream interleaved with FFMA, FMUL and LDSM at fixed ratios, with the FFMA operands chosen to hit and to miss HMMA's banks. Gives how many cycles an HMMA's 10-register read blocks a neighbour's dispatch.
11. **Clock under attention.** `nvidia-smi --query-gpu=clocks.sm,power.draw` sampled at 1 ms while FA4 s4096 and `sm100_fmha` run back to back for ~10 s, without ncu's clock control.

`--only a,b` restricts the run to some labels, and `--max-error N` fails when any |error| exceeds N%. CI runs the fast kernels this way. `RSIM_VALIDATE_STATS=1` dumps each kernel's simulated stats (stalls, cache hits, instruction mix) to stderr.

Model fixes behind this table:

- Kernel window, from measured `window_*` points, with the model's own empty-kernel time subtracted along the path every warp really takes.
- `tcgen05.alloc` latency (`UTCATOMSWS.FIND_AND_SET`).
- HBM latency rising with load. It uses the measured probe when the calibration has one, and otherwise the per-SM read sweep. Points above 80% utilisation are left to the bandwidth server, which queues there already.
- Trace replay:
  - warps retire when their stream ends rather than at an EXIT, because a recorded EXIT can have no lanes active;
  - only a CTA's last warp waits for its outstanding copies;
  - `DEPBAR.LE`'s count is read from the trace's immediate field (cp.async went from +216% to −3%).
- Earlier fixes:
  - `cp.async.wait_group` waits on LDGDEPBAR commit groups;
  - the fractional `mma.sync` issue interval;
  - the SMEM port rate from `pipes.l1_bytes_per_cycle`.

`ncu_reference.json` holds reference kernels with their SASS and measured cycle counts. These
form the end-to-end regression set: `rsim sass <file>` should land within tolerance of `ncu`.
