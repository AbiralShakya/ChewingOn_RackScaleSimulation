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

Opcode lookup tries the full opcode first (`MUFU.EX2`) and then the base (`FFMA`).

## Power-capped clocks

Real GEMMs on B300 hit the 1100 W board cap and the SM clock drops (measured with CUTLASS FP4: 2032 MHz at 4096³, 1717 at 8192³, 1455 at 16384³). The model takes `P = idle + D·(f/f_max)³`. Each measured run fixes the `D` its kernel needs at `f_max`. A kernel's sustained clock is the largest `f` with `P ≤ cap`, with `D` interpolated in log(FLOPs) between the runs of the same dtype. The tensor-bound share of the kernel's time (tensor utilisation / 0.7, capped at 1) is stretched by `f_max / f`. Memory time is left alone. Only dtypes with measured runs are throttled (FP4 today).

## Validation against ncu

`rsim validate calibration/results/<gpu>/ncu_reference.json --gpu <preset> [--calibration <calibration.json>]` replays each reference kernel and compares simulated SM cycles with ncu's `sm__cycles_elapsed.avg`. Every non-window reference now has an exact recording: an NVBit warp path (`path_file`, `.rpath.gz`) on B300, and an Accel-Sim trace (`trace_file`, `.traceg`) on both A100s. The A100 traces run at a reduced size (`trace_env`), and their `ncu` entry is re-measured at that size; the full-size numbers stay in `ncu_full_size`. A kernel with neither is walked from its static SASS. Loop trips are then solved from `smsp__inst_executed.sum`, an instruction count rather than a timing. When that count stops at the first guarded EXIT, the walk takes the exit.

Each kernel is simulated at the SM clock its ncu run actually ran at (`sm__cycles_elapsed.avg / gpu__time_duration.sum`, for kernels longer than 100 us). `ncu --clock-control none` leaves the clock to the driver, and memory-bound cycle counts only compare at the same clock.

Validation charges the measured **kernel window**, the fixed cost of an almost empty kernel inside ncu's window, by launch shape (`calibration/src/bench_window.cu`, `model.launch.kernel_window`). It charges the measured cost minus the model's own time for that empty kernel. The `window_*` rows check this. They aren't independent: the six plain shapes test that the subtraction is consistent, and the two FA4-prologue shapes are the points `tensor.tmem_alloc_latency` is fitted to (1430 cycles for `cta_group::1`, 1990 for `cta_group::2`). The FlashAttention study's 16 recorded kernels are the out-of-sample check, and all are within ±6% (research/flash_attention).

Results (simulated vs measured cycles; saved as `validation.json` next to each reference):

| kernel | B300 (path) | A100 SXM4 (trace) | A100 PCIe (trace) |
|---|---|---|---|
| hbm_stream_read | +0.3% | −4.3% | −7.8% |
| hbm_copy | −5.1% | −23.5% | −21.8% |
| cp_async_cg16_dram_stream | n/a | −3.1% | −3.7% |
| tma_bulk_stream | −6.2% | n/a | n/a |
| ffma_throughput | −7.0% | +0.8% | +0.9% |
| ex2_throughput | +0.1% | +0.1% | +0.2% |
| dfma_latency_chain | +3.2% | −2.3% | +1.3% |
| smem_conflict_free | +10.0% | +6.4% | +3.9% |
| smem_32way_conflict | −0.3% | −0.0% | −0.1% |
| l2_pointer_chase | −3.7% | +16.3% | +11.8% |
| hmma_bf16_throughput | −0.1% | +0.5% | +0.2% |
| hmma_f16_f32_m16n8k16_w32_ilp4 | n/a | +0.6% | +1.1% |
| umma_bf16_m128n256k16 | −0.1% | n/a | n/a |
| umma_nvfp4_m128n256k64 | −0.0% | n/a | n/a |
| window_* (plain shapes) | +0.2% | ±0.5% | ±0.6% |
| window_* (FA4 prologue, fit points) | 0.0%, −0.1% | n/a | n/a |

B300: 11 of 12 kernels within 7%, and smem_conflict_free at +10.0%. A100: 9 of 11 within 8% on both boards.

`--only a,b` restricts the run to some labels, and `--max-error N` fails when any |error| exceeds N%. CI runs the fast kernels this way. `RSIM_VALIDATE_STATS=1` dumps each kernel's simulated stats (stalls, cache hits, instruction mix) to stderr.

What the remaining misses need:

- **hbm_copy on A100** (about −23%): at the traced size (2 CTAs, 1 MB) each warp keeps one 512 B load in flight. The model's round trip is about 680 cycles; the hardware's is about 940. The model has no per-SM limit on the SM-to-L2 path, and stores share it with loads. The read-only stream is within 8%.
- **l2_pointer_chase on A100** (+12–16%): the chase runs slow even with the near/far L2 split scaled to the measured mean.
- **smem_conflict_free on B300** (+10%) and **ffma_throughput on B300** (−7%): open.
- **tma_bulk_stream on B300** (−6%): 4 KB chunks, 4 stages. HBM latency under load now comes from the measured loaded-latency probe (`raw/loaded`: 417 ns idle, 471 ns at 5.4 TB/s), which took this from −10.6% to −6.2%. The rest is probably in the TMA engine.

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
