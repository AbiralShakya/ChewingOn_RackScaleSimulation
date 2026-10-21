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

`rsim validate calibration/results/<gpu>/ncu_reference.json --gpu <preset>` replays each reference kernel's SASS. It compares simulated SM cycles with `sm__cycles_elapsed.avg`. Loop trip counts are solved from `smsp__inst_executed.sum`, an instruction count rather than a timing, so the timing is not fitted. The B300 results are in `calibration/results/b300-sxm6-ac-sm103/validation.json`:

| kernel | ncu cycles | sim | error |
|---|---|---|---|
| dfma_latency_chain | 40875 | 39394 | −3.6% |
| smem_conflict_free | 142278 | 149043 | +4.8% |
| hbm_copy | 5377992 | 5116606 | −4.9% |
| ex2_throughput | 271264 | 248396 | −8.4% |
| l2_pointer_chase | 3458464 | 3788536 | +9.5% |
| ffma_throughput | 82103 | 64585 | −21.3% |
| hbm_stream_read | 2562049 | 1146474 | −55% |
| hmma_bf16_throughput | 139972 | 33802 | −76% |
| smem_32way_conflict | 4214901 | 149043 | −96% (needs address annotations, so no bank conflicts are modelled from bare SASS) |
| umma_* | 534974 | 2067 | −99.6% (the issuer loop is not reconstructed yet) |
| tma_bulk_stream | 4569374 | 19395951 | +325% |

The misses are open work. The walker has no register values, so address-dependent effects (bank conflicts, the TMA stream's addresses) need annotations or an NVBit trace.

`ncu_reference.json` holds reference kernels with their SASS and measured cycle counts. These
form the end-to-end regression set: `rsim sass <file>` should land within tolerance of `ncu`.
