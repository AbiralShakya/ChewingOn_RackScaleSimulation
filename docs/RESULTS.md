# First results

These are the results of the demo runs, refreshed 2026-10-21 on the `integration` branch (warp paths, cp.async fix, calibration, split-K GEMMs). They are reproducible with the commands shown.
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
| single stream | 1416 us | 13.7% | 6.9% |
| two-micro-batch overlap | 2354 us | 8.3% | 0.9% |

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

## GB200 vs GB300 NVL72: FP4 prefill under the power cap

Llama-3-405B prefill, TP8 × 9 replicas, 8k tokens, first 2 layers, fast mode (`python examples/gb300_fp4.py`).
GB300 uses the measured B300 calibration, including the K=96 FP4 UMMA (1.5× the K=64 rate) and power-capped clocks.

| system | dtype | step | GEMM time / GPU |
|---|---|---|---|
| GB200 NVL72 | bf16 | 11825 us | 8844 us |
| GB200 NVL72 | fp4 | 5370 us | 2388 us |
| GB300 NVL72 | bf16 | 9477 us | 6668 us |
| GB300 NVL72 | fp4 | 5541 us | 2420 us |

GB300's FP4 GEMMs are no faster than GB200's (1% slower here), not 1.5×. Large FP4 GEMMs pin the 1100 W cap, so the clock falls to about 1.6-1.9 GHz, and the GEMMs left at that point are no longer the bottleneck for the step. The step is then set by attention, norms and TP all-reduces. GB300's BF16 advantage (−19%) comes mainly from the measured memory system and the higher clock on kernels that are not power-bound. GB200 has no measured power data yet, so it runs unthrottled. That favours GB200, and GB200 power runs would make this comparison fair.

## DeepSeek-V3 decode on GB200 vs GB300 NVL72: BF16 vs FP4 weights

EP72, 4k context, first 4 layers, fast mode (`python examples/gb300_deepseek_fp4.py`). Batch is per GPU.

| system | batch | dtype | step | GEMM time / GPU | tok/s/GPU |
|---|---|---|---|---|---|
| GB200 NVL72 | 64 | bf16 | 1416 us | 997 us | 45.2k |
| GB200 NVL72 | 64 | fp4 | 776 us | 357 us | 82.4k |
| GB300 NVL72 | 64 | bf16 | 1401 us | 1002 us | 45.7k |
| GB300 NVL72 | 64 | fp4 | 746 us | 347 us | 85.8k |
| GB200 NVL72 | 256 | bf16 | 2542 us | 1347 us | 100.7k |
| GB200 NVL72 | 256 | fp4 | 1675 us | 480 us | 152.9k |
| GB300 NVL72 | 256 | bf16 | 2437 us | 1215 us | 105.0k |
| GB300 NVL72 | 256 | fp4 | 1551 us | 429 us | 165.0k |

- **FP4 nearly doubles decode throughput on both racks** (1.8–1.9× at batch 64, 1.5–1.6× at batch 256). Decode GEMMs stream weights, so the gain comes from moving a quarter of the bytes, not from the tensor core.
- **GB300 is only 1–7% faster than GB200 for decode.** The two have the same HBM bandwidth, and decode does not use GB300's extra FP4 FLOPs. The power cap does not bite here either. Decode GEMMs keep the tensor cores mostly idle, so they stay well under the 1100 W board limit that throttles FP4 prefill.
- **At batch 256, FP4 GEMMs are no longer the step.** GEMMs are under a third of the 1.55–1.68 ms step. MLA attention, the all-to-alls and the norms set the pace, so the next decode lever is attention (an FP8 KV cache, or FlashMLA-style kernels), not lower-precision weights.

The step also got faster since the first results. GEMMs with fewer output tiles than SMs now use split-K (from the Mamba study), so the M=64 decode GEMMs spread over all SMs instead of a few.

## FlashAttention on recorded warp paths: one-number what-ifs and ring attention

These come from the FlashAttention study (`research/flash_attention/`, write-up in `writeup.html`). The kernels are real FA2/FA4 binaries recorded with NVBit (`.rpath`) and validated against ncu within ±4%. The recordings are too large for git, so the numbers are from the committed `results/study.json` and `results/scale_rows.json`. Rerun them with `run_study.py whatif|scale` once the traces are on disk.

**One hardware number at a time** (B300, forward, BF16, d=128, 16k tokens; speedup vs. the calibrated B300):

| kernel | change | speedup |
|---|---|---|
| FA4 | MUFU.EX2 at half rate (B200-like exp unit) | 0.79× |
| FA4 | MUFU.EX2 at 2× | 1.00× |
| FA4 | tensor core 1.5× / 2× | 1.21× / 1.21× |
| FA4 | tensor 2× + EX2 2× | 1.51× |
| FA4 | tensor 2× + EX2 2× + FFMA2 2× | 1.56× |
| FA4 | HBM bandwidth halved | 1.00× |
| FA2 | EX2 at 2× / HBM halved | 1.02× / 1.00× |
| FA2 on A100 | no HMMA↔FP32-pipe coupling | 1.01× |

On B300, FA4 is balanced between the tensor core and the exp unit. A faster tensor core alone stalls on EX2 at about 1.2×. B300's doubled EX2 rate is what keeps FA4 from being exp-bound, which is why FA4 emulates exp on B200.

**Context-parallel ring attention, 72 GPUs** (Llama-3-70B heads: 64 query heads, 8 or 64 KV heads; each step is one query chunk against one KV chunk while the next chunk moves to the neighbour):

| system | kernel | chunk | KV heads | context | step | ring efficiency | TFLOPS/GPU |
|---|---|---|---|---|---|---|---|
| GB300 NVL72 | FA4 | 1024 | 8 | 74k | 1.89 ms | 94% | 1311 |
| GB300 NVL72 | FA4 | 1024 | 64 | 74k | 3.20 ms | 96% | 773 |
| GB300 NVL72 | FA4 | 2048 | 8 | 147k | 6.04 ms | 98% | 1637 |
| GB300 NVL72 | FA4 | 2048 | 64 | 147k | 8.44 ms | 99% | 1173 |
| 9×8 B300 over IB | FA4 | 1024 | 8 | 74k | 7.17 ms | 42% | 345 |
| 9×8 B300 over IB | FA4 | 2048 | 64 | 147k | 112 ms | 9% | 88 |
| 9×8 A100 over IB | FA2 | 2048 | 8 | 147k | 52.3 ms | 100% | 189 |
| 9×8 A100 over IB | FA2 | 2048 | 64 | 147k | 225 ms | 23% | 44 |

Ring efficiency is the share of the step spent in attention kernels. On NVL72 the KV transfer hides behind the kernel at every shape. Over InfiniBand, FA4 on B300 is fast enough that the ring becomes transfer-bound, even with GQA (8 KV heads) at 1k chunks. The slower FA2 on A100 still hides the transfer when KV is small. A faster kernel needs a faster fabric to stay compute-bound, and NVL72 provides that.

## One-number what-ifs on GB300 (synthetic kernels and rack workloads)

`python examples/gb300_whatifs.py`. Each variant changes one number of the calibrated B300: every MUFU function at 2× issue rate, shared memory at 256 B/clk per SM instead of 128 (`pipes.l1_bytes_per_cycle`, which now drives the SMEM port), or no 1100 W board cap. The kernels here are the simulator's synthetic ones, not the recorded FA binaries above.

| workload | mufu 2× | smem 2× | no power cap | all three |
|---|---|---|---|---|
| attention 8k causal, 64q/8kv heads (754 us) | 1.15× | 1.00× | 1.00× | 1.15× |
| GEMM bf16 8192³ (557 us) | 1.00× | 1.00× | 1.00× | 1.00× |
| GEMM fp4 16384³ (1731 us) | 1.00× | 1.00× | 1.24× | 1.24× |
| Llama-3-405B fp4 prefill 8k, GB300 NVL72 (5.54 ms) | 1.01× | 1.00× | 1.06× | 1.07× |
| DeepSeek-V3 fp4 decode b256, GB300 NVL72 (1.66 ms) | 1.01× | 1.00× | 1.00× | 1.01× |

- **The power cap costs large FP4 GEMMs about 24%, but the rack step only 6%.** Even the 405B prefill spends most of its step outside the GEMMs. Decode never reaches the cap.
- **The exp unit is the next limit for attention**, as in the recorded FA4 what-ifs. Removing the power cap does nothing there: only FP4 has measured power runs, so BF16 kernels are never throttled. That favours BF16 until BF16 power data exists.
- **2× SMEM bandwidth does nothing on these kernels.** Their operands go TMA → SMEM → UMMA, and the model does not charge UMMA descriptor reads to the SMEM port. Only LDS/LDSM traffic uses it. Read this as "not modelled", not as "SMEM is never the bottleneck". The recorded FA kernels with LDSM-heavy softmax paths are where to test this once the traces are back.
