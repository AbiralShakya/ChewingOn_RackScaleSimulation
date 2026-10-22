# 9. How far to trust the numbers

A simulator is only as useful as the evidence that it is right where you are using it. This
chapter collects that evidence: what was measured on hardware, how closely the simulator
reproduces each class of kernel, and which results rest entirely on presets. Use it to decide
how many significant figures a number in this guide deserves.

## 9.1 The method

The rule for this project, from [RESEARCH.md](../../RESEARCH.md): **every study runs the real
kernel on the single GPUs we have (B300 sm_103 and A100 sm_80), under Nsight Compute, and the
simulator must reproduce it within ±10% with the same dominant stall reason before the
simulator alone is used to extrapolate** to HGX, NVL72, H100 or mixed systems.

Calibration and validation are kept separate. `calibration.json` holds microbenchmark
results: latencies, issue rates, bandwidths, tensor-core rates, power runs. Validation then
runs *different* kernels (Nsight reference kernels, FlashAttention, cuBLAS GEMMs, Mamba
kernels) with nothing fitted per kernel. Loop trip counts come from instruction counts
(`smsp__inst_executed.sum`), which are workload facts, not timings.

## 9.2 Microbenchmark kernels against Nsight Compute

`rsim validate` replays each reference kernel's SASS at the clock its ncu run actually used.
Simulated minus measured SM cycles, over measured (committed `validation.json`; docs in
[CALIBRATION.md](../CALIBRATION.md)):

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
| smem_32way_conflict | −96.5% | −96.4% | −96.4% |
| umma_bf16 / umma_nvfp4 | −99.5% | n/a | n/a |

Since that table was written, the simulator gained the measured **kernel window** (chapter
3.6) for kernels timed alone. Rerunning the B300 set on the current tree gives ffma −9.5%,
ex2 −5.2%, dfma +14.9%, smem_conflict_free +10.1%: the short kernels moved toward the
measurement, and the 41k-cycle dfma chain now overshoots.

What the misses mean:

- **smem_32way_conflict** and **umma_\*** are not timing errors. Bank conflicts need lane
  addresses, which bare SASS does not have, and the UMMA benchmark picks its loop and issuing
  thread from runtime arguments, so the static walker follows the wrong path. Both are fixed by
  replaying an NVBit `.rpath` recording, which is how the FlashAttention kernels below run.
- **A100 dfma** is a 7k-cycle kernel where fixed launch and drain costs dominate.
- **A100 l2_pointer_chase** (+20 to 26%) is open: the near/far L2 split is modelled, but the
  chase still runs slow.

## 9.3 Real kernels

| kernels | GPU | cases | result | source |
|---|---|---|---|---|
| FlashAttention-2 and -4, recorded paths, 4k shape | A100 PCIe, B300 | 4 | **−6.0% to +0.2%** cycles, instruction counts exact | [`research/flash_attention`](../../research/flash_attention/README.md) |
| FA4 / FA2 ring-attention step kernels, out of sample | B300, A100 SXM4 | 12 | FA4 −3.7% to +5.4%, FA2 −4.6% to −5.9% | same |
| cuBLAS BF16 GEMMs, decode and prefill shapes | B300 | 54 | decode shapes median \|error\| 3.9%; large GEMMs median 12%, mean +6% (was 15–35% fast before the sustained-clock model) | [`calibration/scripts/check_gemms.py`](../../calibration/scripts/check_gemms.py) |
| cuBLAS BF16 GEMMs | A100 PCIe | 48 | median \|error\| 33%; large GEMMs 30–90% slow | same |
| FA decode/prefill attention | B300 / A100 | 30 / 22 | median 10% / 20% | same |
| Mamba-2 state update (real Triton SASS) | B300 / A100 | 6 / 6 | median 10% / 20% | same |
| SSD chunked-scan prefill | B300 / A100 | 6 / 5 | median 24% / 68% | same |
| per-GPU shapes of the rack sweeps | B300 | 33 | median 11% | same |

Read this as:

- **Recorded kernels on either GPU are the most trustworthy numbers** in the guide: all 16
  within ±6%, with the A100 recordings consistently about 5% fast.
- **Synthetic Blackwell kernels** are good to about 4% (median) at decode shapes and 12% on
  large GEMMs, once the kernel runs at its sustained, power-capped clock. Large GEMMs used to
  be 15 to 35% optimistic: BF16 ran at the full 2032 MHz, while the B300 holds about 1.3 GHz
  on a large GEMM run back to back. That was partly hidden by a second error, a kernel that
  was 10 to 30% slow in cycles because it never overlapped one tile's epilogue with the next
  tile's mainloop. Both are fixed. The remaining spread is mostly shape-to-shape variation in
  the sustained clock (1250 to 1520 MHz at the same FLOP count), which the size-only power
  model cannot see.
- **Synthetic Ampere kernels are the weak spot.** The multistage `mma.sync` GEMM reaches about
  55% of peak where cuBLAS reaches 87%, and the SSD scan runs 60 to 70% slow. No rack result in
this guide uses the A100 model; the A100 crossovers in chapter 6 come from hardware.

## 9.4 Fast mode against exact co-simulation

Fast mode simulates each distinct `(kernel, reserved SMs)` pair once on a cold GPU and reuses
the result. The fabric, collectives and stream dependencies are still simulated exactly. What
it gives up is interference between concurrent kernels on one GPU and warm caches between
kernels.

| system | exact | fast | error | wall time (exact → fast) |
|---|---|---|---|---|
| HGX B200, DeepSeek-V3 EP8, batch 32 | 2110 µs | 2121 µs | +0.5% | |
| GB200 NVL72, DeepSeek-V3 EP72, batch 32 | 1712 µs | 1729 µs | +1.0% | 6.5 min → 12 s |
| GB200 NVL72, Transformer 70B TP8 vs 72-GPU run, batch 64, 16k | 14.30 ms (72 GPUs) | 14.30 ms (one TP group) | 0% | |

## 9.5 What rests on presets

Nothing in this list has been measured, and every result that depends on it is model
behaviour:

| what | used in | what would calibrate it |
|---|---|---|
| H100 / H200 (all numbers) | chapters 2, 4, 8 | the calibration suite on an H100 (`calibration/run.sh`) |
| B200 / GB200 GPU (clock, HBM, power) | chapters 1, 5, 6 | the suite on a B200, including the power runs |
| NVLink link bandwidth and latency, NVSwitch latency, NVLS efficiency | chapters 4, 5, 6 | `nccl-tests` all-reduce / all-gather / all-to-all sweeps on an HGX board and an NVL72 |
| InfiniBand NIC and switch latency, rail routing (no PXN) | chapters 4, 5, 8 | `nccl-tests` across two nodes; `ib_write_bw` / `ib_write_lat` |
| BF16 power draw on B300 (derived from ncu cycles over CUDA-graph time, not a power trace), FP8 power draw | chapters 1, 6 | the `cutlass_power` runs with BF16 and FP8 GEMMs |
| a GB300 board (vs the HGX B300 we measured) | chapters 5, 6 | the suite on a GB300 tray |

The network is the largest gap. All of chapter 4 and every multi-GPU number is a prediction
from link rates and a chunked cut-through model, with no multi-GPU measurement behind it yet.

## 9.6 Reading a result

When you see a number in this guide, or produce one with the simulator, ask three questions:

1. **Which GPU model produced it?** Calibrated B300, calibrated A100, or a preset.
2. **Which kernel model?** Recorded `.rpath` (±6%), real SASS, or synthetic (±10% at decode
   shapes on Blackwell; worse on Ampere and for large GEMMs).
3. **Does it cross a fabric?** Then the network part is a preset, and the result is a model
   prediction until the fabric is calibrated.

A comparison between two configurations simulated the same way is usually more reliable than
either absolute number, because most model errors move both sides together. That is why the
studies in this repository report speedups and crossovers, and why the one-number what-ifs of
chapter 2 (EX2 rate, tensor-core rate) are the strongest results in the guide: the same
recorded kernel, the same calibration, one number changed.

## Reproduce

| number | command |
|---|---|
| microbenchmark validation | `rsim validate calibration/results/<gpu>/ncu_reference.json --gpu <preset> --calibration calibration/results/<gpu>/calibration.json` |
| a quick CI subset | `rsim validate ... --only hbm_stream_read,ex2_throughput --max-error 10` |
| FA validation | `python research/flash_attention/run_study.py validate` (recordings in `research/flash_attention/rpath/`) |
| GEMM / attention / Mamba validation | `python research/ssm_vs_attention/validate.py --gpu b300` |
| fast vs exact | `python examples/deepseek_v3_nvl72.py --layers 4 --exact` |

Back to the [contents](README.md).
