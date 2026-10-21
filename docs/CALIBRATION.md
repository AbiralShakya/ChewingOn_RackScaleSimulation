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

`rsim validate calibration/results/<gpu>/ncu_reference.json --gpu <preset> [--calibration <calibration.json>]` replays each reference kernel's SASS. It compares simulated SM cycles with `sm__cycles_elapsed.avg`. Loop trip counts are solved from `smsp__inst_executed.sum`, an instruction count rather than a timing, so the timing is not fitted. For streaming kernels (more than 64 MB of DRAM traffic), the main-loop trip count is pinned to the measured DRAM bytes instead. Both are workload facts, not timings.

Each kernel is simulated at the SM clock its ncu run actually ran at (`sm__cycles_elapsed.avg / gpu__time_duration.sum`, for kernels longer than 100 us). `ncu --clock-control none` leaves the clock to the driver, and it can sit far from the calibration run's clock: the A100 PCIe ran at 1065 MHz under ncu but 1395 MHz in the microbenchmarks. Memory-bound cycle counts only compare at the same clock.

Results (simulated vs measured cycles; saved as `validation.json` next to each reference). B300 uses the `b300` preset; the A100s use `a100_sxm` / `a100_pcie` plus their own `calibration.json`:

| kernel | B300 | A100 SXM4 | A100 PCIe |
|---|---|---|---|
| hbm_stream_read | +0.7% | +0.2% | −0.4% |
| hbm_copy | −5.0% | −6.3% | −7.0% |
| cp_async_cg16_dram_stream | n/a | −3.4% | −11.0% |
| tma_bulk_stream | +6.9% | n/a | n/a |
| ffma_throughput | −18.2% | −0.8% | −1.0% |
| ex2_throughput | −7.9% | −0.1% | −0.1% |
| dfma_latency_chain | −2.2% | −42.5% | −39.7% |
| smem_conflict_free | +5.2% | +11.0% | +10.7% |
| l2_pointer_chase | +9.6% | +25.9% | +19.9% |
| hmma_bf16_throughput | +5.2% | +11.2% | +11.2% |
| hmma_f16_f32_m16n8k16_w32_ilp4 | n/a | +10.2% | +10.0% |
| smem_32way_conflict | −96.5% | −96.4% | −96.4% |
| umma_bf16_m128n256k16 | −99.5% | n/a | n/a |
| umma_nvfp4_m128n256k64 | −99.5% | n/a | n/a |

`--only a,b` restricts the run to some labels and `--max-error N` fails when any |error| exceeds N%; CI runs the fast kernels this way.

B300: 8 of 12 within 10%. A100: 8 of 11 within about 11% on both boards. What the remaining misses need:

- **smem_32way_conflict** (−96% everywhere): bank conflicts need addresses, and bare SASS has none. A `.traceg` or `.rpath` recording fixes it; `rsim validate` replays `trace_file` / `path_file` entries exactly.
- **umma_*** (−99.5%): the kernel picks its mode (latency vs throughput loop) and its issuing thread from runtime arguments, so the static walker takes the wrong path. Needs a `.rpath` recording of the throughput launch.
- **dfma_latency_chain on A100** (−40%): a 7k-cycle kernel, so fixed launch and drain costs dominate. Not yet modelled at this granularity.
- **l2_pointer_chase on A100** (+20-26%): the A100 near/far L2 split is applied, but the chase still runs slow. Open.
- **ffma_throughput on B300** (−18%): the simulated 67k cycles match ncu's `smsp__cycles_active.avg` (70k). The gap is about 12k cycles in which the SMs sit idle inside the kernel window. It is probably launch and shared-memory carveout cost (the kernel asks for 118 KB of dynamic smem), which the model does not have yet. The A100 version is within 1%.

Two model fixes came out of the A100 data. `DEPBAR.LE SB0, N` (`cp.async.wait_group N`) now waits on LDGDEPBAR commit groups. CTAs now drain their async copies before retiring. Together these took cp_async_cg16_dram_stream from −76% to −3%. The FlashAttention-thread merge (HMMA rate from `sync_flops_per_cycle_per_sm`) took B300 hmma from −76% to +5%.

`ncu_reference.json` holds reference kernels with their SASS and measured cycle counts. These
form the end-to-end regression set: `rsim sass <file>` should land within tolerance of `ncu`.
