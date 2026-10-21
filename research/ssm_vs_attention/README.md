# Mamba-2 vs attention: where each wins, at cycle level

Which sequence mixer is cheaper per layer, Mamba-2 (SSD) or attention, on Ampere and Blackwell, for decode and
prefill, and what that means for a GB200 NVL72 rack. Single-GPU kernels are measured on real hardware (B300 SXM6,
sm_103; A100 80GB PCIe, sm_80) and explained from their SASS and Nsight Compute counters; everything beyond one GPU
is the simulator only.

| | B300 | A100 |
|---|---|---|
| decode crossover (context where an attention layer costs more than a Mamba-2 layer, batch 32-256) | ~3.4k tokens | ~1.1k tokens |
| prefill crossover (prompt length where FlashAttention costs more than the SSD scan) | ~10.9k tokens | ~4.2k tokens |
| `selective_state_update` state bandwidth | 2.0 TB/s (26% of HBM) | 1.0 TB/s (55%) |

The write-up with figures is `results/report.html` (published as an Artifact); the numbers behind it are in
`results/summary.json`, `results/validation_*.json` and `results/sweep_*.json`.

## Layout

| path | what |
|---|---|
| `hw/setup_env.sh` | builds torch + state-spaces/mamba + causal-conv1d + FlashAttention (FA4 on sm_100+, FA2 on sm_80) |
| `hw/bench_kernels.py` | times every case (CUDA graphs, cold L2 via rotating input copies), one NVTX range per case |
| `hw/run_ncu.sh` | Nsight Compute over those ranges (cycles, instructions, DRAM bytes, stall reasons) + Triton SASS |
| `hw/run_b300.sh`, `hw/run_b300_ncu.sh`, `hw/run_b300_tp8.sh`, `hw/run_a100.sbatch` | how the runs were launched (B300: della-tridao GPU 3 under the shared `flock`; A100: Della Slurm) |
| `data/<gpu>/` | measured timings, Nsight Compute CSV, Triton SASS + metadata |
| `configs/*_calibrated.yaml` | one-GPU systems that load the measured calibration |
| `validate.py` | simulates every measured case and reports the error |
| `sweep.py` | GB200 NVL72 decode/prefill sweeps for the 8B and 70B model families |
| `analyze.py`, `build_report.py`, `svgchart.py` | summary JSON, figures and the HTML report |

## Reproduce

```bash
# simulator side (no GPU needed)
maturin develop --release
python research/ssm_vs_attention/validate.py --gpu b300      # and --gpu a100
python research/ssm_vs_attention/sweep.py --family 70b --full-rack
python research/ssm_vs_attention/analyze.py
python research/ssm_vs_attention/build_report.py
```

## Simulator changes made for this study

- New kernels (`crates/rsim-gpu/src/synth_ssm.rs`): Mamba-2 `selective_state_update` (runs the real Triton SASS for
  sm_103 and sm_80), the five SSD kernels with their autotuned tiles, `causal_conv1d` update/forward, and
  FlashAttention-2 for Ampere.
- Lowering for Mamba-2 and hybrid block stacks (`python/rackscale/lower.py`, `models.py`), 8B and 70B families.
- `a100_pcie` preset; a separate measured `mma.sync` rate (`TensorSpec.mma_sync_flops_per_cycle_per_sm`, 1/4 of
  `tcgen05` on Blackwell); split-K and narrow tiles for decode GEMMs; CUTLASS-style multistage Ampere GEMM;
  FlashAttention-4's two-Q-tile ping-pong; independent per-element ALU work in synthetic kernels.
- MFU no longer counts an LM head that prefill and training never lower.
