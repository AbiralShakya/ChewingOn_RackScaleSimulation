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

Opcode lookup tries the full opcode first (`MUFU.EX2`) and then the base (`FFMA`).

## Still to calibrate

- tcgen05 round-trip latency and commit latency are measured (`tensor.tcgen05_*.roundtrip_latency_cycles`)
  but not yet wired into `async_mma_overhead`
- near and far L2 partition latency (`memory.l2_partitions`)
- the TMA latency/size curve (`tma.bulk_g2s_latency`)
- HBM bandwidth vs. active SMs (`memory.bandwidth.hbm_read_vs_sms`), which matters for
  partial-occupancy kernels

`ncu_reference.json` holds reference kernels with their SASS and measured cycle counts. These
form the end-to-end regression set: `rsim sass <file>` should land within tolerance of `ncu`.
