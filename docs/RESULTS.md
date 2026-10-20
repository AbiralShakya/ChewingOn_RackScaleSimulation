# First results

These are the results of the first demo runs. They are reproducible with the commands shown.
All numbers use uncalibrated presets unless marked otherwise, so read them as *model
behaviour*, not as hardware claims.

## DeepSeek-V3 decode on GB200 NVL72 (EP72)

This run covers four layers: the 3 dense layers and the first MoE layer. Each GPU holds 64
sequences at 4k context, uses BF16, and runs MLA decode attention.

```bash
python examples/deepseek_v3_nvl72.py --layers 4      # fast mode, ~12 s
```

| | makespan | MFU | GPU idle |
|---|---|---|---|
| single stream | 1899 us | 10.2% | 5.2% |
| two-micro-batch overlap | 2456 us | 7.9% | 0.9% |

What the model says:

- **Decode is weight-bandwidth bound, even on GB200.** The LM head (129280×7168) and the
  dense-layer GEMMs stream the weights at 7.1 TB/s. That is close to the 8 TB/s HBM peak times
  the 0.9 streaming efficiency. With M=64 against a 128-row UMMA tile, half the tensor work is
  padding (`tc%` ≈ 35%).
- **MLA attention is compute-heavy.** The 128 query heads share one 576-wide latent, so the
  kernel runs at about 980 TFLOPS, not at HBM speed. It is the largest single kernel.
- **Two-micro-batch overlap loses at this size.** Each half-batch reads the full weights again,
  so the weight-bound GEMMs cost almost the same at half the batch. The all-to-all it hides is
  only about 10 us per layer on NVLink5 (509-722 GB/s busbw). Overlap pays off only when
  dispatch/combine is a large part of the step: for example over IB, at larger EP, or with
  FP8 weights.
- **Bubble anatomy.** About 60% of scheduler cycles are `async_wait`: warps parked on
  mbarriers while TMA/UMMA work is in flight. That is the expected signature of a
  warp-specialised kernel. The bubbles that matter are the ~11% `tma_wait` (the tensor core
  waiting on loads) and the 5% of time the GPU sits idle between kernels (launch gaps).

**Fast vs exact.** HGX B200 ×8 with EP8 gives 2110 us exact and 2121 us fast (+0.5%). NVL72
gives 1712 us exact and 1729 us fast (+1.0%) at batch 32. The exact NVL72 run takes 6.5
minutes; fast mode takes 12 seconds.

## Per-SM timeline

```bash
rackscale decode deepseek-v3 --system hgx_b200_8 --ep 8 --batch 32 --context 4096 --layers 4 \
    --per-sm --trace out/deepseek_v3_ep8_b200_per_sm.json
```

Open the trace in [ui.perfetto.dev](https://ui.perfetto.dev). Each GPU is a process with:

- stream tracks (kernels and collectives)
- one track per SM, with a span per CTA (`cta` id and cycle count)
- counter tracks for issue utilisation, tensor utilisation, DRAM GB/s and active SMs

Wave quantisation, tail effects and SM reservation for comm kernels are all directly visible.

## Async RL on a mixed cluster

The cluster has 8 H100 rollout GPUs and 8 B200 trainer GPUs on 400G IB
(`configs/mixed_rl_h100_b200.yaml`).

```bash
python examples/async_rl_mixed.py
```

Llama-3-8B (2 layers) does 4 decode steps, then a training step, then the weight push. The
step takes 25.9 ms, and about 80% of it is the trainer→sampler weight transfer (872 MB per
pair at 42 GB/s effective over one 50 GB/s NIC). Compute is about 5 ms per side. In this setup,
weight sync dominates. Bucketed or broadcast weight sync, NVLink-local samplers, or FP8 weight
transfer are the levers to model next.

## B300 with measured calibration

`configs/hgx_b300_8_calibrated.yaml` loads `calibration/results/b300-sxm6-ac-sm103/calibration.json`.
The measured values replace the presets: 2003 MHz clock, 89% HBM read efficiency, per-opcode
ALU/XU latencies, and the tcgen05 FLOP/clk. Llama-3-70B TP8 decode (2 layers, b32) runs 2% slower
than with the uncalibrated preset, mostly because of the lower measured clock and HBM efficiency.
