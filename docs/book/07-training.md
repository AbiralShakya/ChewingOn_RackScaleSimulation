# 7. Training

Training is prefill run forwards and then backwards, plus an optimizer step and gradient
communication. Everything from chapters 1 to 5 applies; this chapter adds the FLOP and memory
accounting that is specific to training and runs a training step through the simulator.

## 7.1 FLOPs: 6 · P · T

For a dense model with *P* parameters processing *T* tokens:

- the forward pass does **2·P·T** FLOP (one multiply-add per parameter per token);
- the backward pass does **4·P·T**: one product for the gradient with respect to the input
  (dgrad) and one for the gradient with respect to the weight (wgrad), each the size of the
  forward.

So a step costs about **6·P·T** FLOP, plus attention: `6 · L · H · d · S · T` for causal
attention over sequences of length *S* (the forward does 4·*H*·*d*·*S* per token per layer,
halved by the causal mask, and the backward twice that). For Llama-3-70B at *S* = 4096 the
attention term is about 4% of the matmul term; at 32k it is about 30%.

**Model FLOPs utilisation** (MFU) is 6·P·T plus the attention term, divided by step time and
by the GPUs' peak. It counts only useful work, so recomputation (activation checkpointing)
lowers MFU even though it keeps the GPU busy. The simulator reports MFU against the preset or
calibrated BF16 peak.

**Worked example.** Llama-3-70B, 4M tokens per step (1024 sequences of 4096), on 72 GB200s at
2.5 PFLOP/s and 40% MFU:

```
6 · 70.6e9 · 4.19e6 ≈ 1.78e18 FLOP  (+4% attention ≈ 1.85e18)
1.85e18 / (72 · 2.5e15 · 0.40) ≈ 26 s per step
```

## 7.2 Memory: 16 bytes per parameter, before activations

With BF16 weights and mixed-precision Adam, each parameter costs

| | bytes |
|---|---|
| BF16 weight | 2 |
| BF16 gradient | 2 |
| FP32 master weight | 4 |
| Adam first moment (FP32) | 4 |
| Adam second moment (FP32) | 4 |
| **total** | **16** |

Llama-3-70B therefore needs 70.6 × 10<sup>9</sup> × 16 ≈ **1.13 TB** of model state, six B200s'
worth of HBM before a single activation. This is why training shards the state:

| scheme | per GPU, over *N* DP ranks | extra communication |
|---|---|---|
| plain DP | 16*P* | all-reduce of gradients |
| ZeRO-1 (optimizer sharded) | 4*P* + 12*P*/*N* | same bytes, reduce-scatter + all-gather |
| ZeRO-2 / FSDP grad sharding | 2*P* + 14*P*/*N* | same |
| ZeRO-3 / FSDP full sharding | 16*P*/*N* | + an all-gather of each layer's weights in forward and backward |

At *N* = 72, ZeRO-3 puts 15.7 GB of model state on each GPU. Activations (chapter 5.3: about
34·*T*·*D* bytes per layer without recomputation) then decide the micro-batch size, and
selective or full recomputation trades 33% more forward FLOPs for most of that memory.

## 7.3 A simulated training step

rackscale lowers a training micro-batch as: the forward layers, then for each layer in reverse
order its backward (every GEMM and attention kernel twice, for dgrad and wgrad), with that
layer's gradient reduce-scatter recorded on a second stream as soon as its backward is done.
This is FSDP/ZeRO-2 style overlap.

```bash
rackscale train llama3-8b --system hgx_b200_8 --dp 8 --layers 2 --fast
rackscale train llama3-8b --system h100_2node_ib --dp 16 --layers 2 --fast
```

Llama-3-8B, 2 of its 32 layers, one sequence of 4096 tokens per GPU, DP8 on HGX B200:

| | |
|---|---|
| step (forward + backward, 2 layers) | **10.72 ms** |
| MFU | 47.8% |
| per-GPU kernel time | 10.14 ms |
| gradient reduce-scatter per layer | 436 MB, ring, 759 GB/s busbw, about 500 µs |
| exposed communication | 0.50 ms (4.7% of the step) |

The biggest kernels are the MLP GEMMs (4096 × 28672 × 4096 at 1.23 PFLOP/s and 4096 × 4096
× 14336 at 1.29), about 55% of the B200's preset peak; attention at 4k runs at 0.75 PFLOP/s.

The gradient overlap works as designed for every layer but one. Layer 1's reduce-scatter
runs while layer 0's backward computes and is fully hidden. **The last bucket (layer 0, the
first layer of the model) has no backward left to hide behind**, so its 500 µs is exposed.
With all 32 layers the same exposed tail is 1/32 of the gradient traffic, about 0.3% of a
full step (about 0.5 ms of 160 ms), which is why per-layer bucketing works.

The same job as DP16 on two HGX H100 nodes over 400G InfiniBand:

| | HGX B200, DP8 | 2 × HGX H100 over IB, DP16 |
|---|---|---|
| step (2 layers) | 10.72 ms | 19.49 ms |
| MFU (against each GPU's own peak) | 47.8% | 59.9% |
| reduce-scatter per layer | 436 MB at 759 GB/s busbw, 0.50 ms | 436 MB at 338 GB/s busbw, 1.21 ms |
| exposed communication | 0.50 ms (4.7%) | 1.21 ms (6.2%) |

Two details worth noticing. The H100 run has the higher MFU because its peak is 2.3× lower
while its simulated GEMMs reach a larger fraction of it (both GPUs are presets here). And the
reduce-scatter over InfiniBand is only 2.2× slower than on NVLink5, not 18×: the rail-style
multi-ring of chapter 4.3 spreads the inter-node traffic over all 8 NICs per node, so the
NVLink4 side (450 GB/s) is the limit, not one 50 GB/s NIC. Data parallelism across nodes is
cheap as long as each GPU has its own NIC and the backward pass is long enough to hide the
buckets.

## 7.4 Where training time goes at scale

Combining chapters 4 and 5, a training step on *N* GPUs spends time in

```
T_step ≈ T_compute / (MFU-like efficiency)
       + exposed TP all-reduces          (≈ 17% at TP8 on HGX B200 if not overlapped, ch. 5.2)
       + pipeline bubble                  ((p−1)/(m+p−1) of the pipelined part, ch. 5.3)
       + exposed DP gradient sync         (zero if it fits under the backward pass)
       + optimizer step and launch gaps
```

The levers are therefore: overlap TP communication (sequence parallelism, collective matmul),
raise micro-batch count or interleave to shrink bubbles, keep gradient sync under the backward
pass by bucketing, and keep FP8 or FP4 GEMMs from hitting the power cap, which on B300 costs the
largest GEMMs up to 24% (chapter 1.6).

## Exercises

1. How many GPU-hours does pre-training Llama-3-70B on 15T tokens take on B300s at 40% MFU
   (BF16 peak 2.46 PFLOP/s, ignoring attention)?
2. With ZeRO-3 over 8 GPUs, how many bytes does each GPU all-gather per step for a 70B model,
   counting forward and backward?
3. Why does activation checkpointing reduce MFU but sometimes increase throughput?

<details><summary>Answers</summary>

1. 6 × 70.6 × 10<sup>9</sup> × 15 × 10<sup>12</sup> = 6.4 × 10<sup>24</sup> FLOP. Divided by
   2.46 × 10<sup>15</sup> × 0.40 = 9.8 × 10<sup>14</sup> FLOP/s per GPU: 6.5 × 10<sup>9</sup>
   GPU-seconds, about **1.8 million B300-hours**.
2. Each layer's BF16 weights are gathered twice (forward and backward): 2 × 2*P* × 7/8 ≈
   **247 GB** per GPU per step, plus a 7/8 × 2*P* reduce-scatter of gradients.
3. Recomputation adds a second forward (about 2·P·T FLOP not counted as useful), so MFU drops.
   But the freed memory allows a larger micro-batch, fewer pipeline bubbles or less TP, and
   that can more than pay for the extra compute.

</details>

## Reproduce

| number | command |
|---|---|
| training step, DP8 on HGX B200 | `rackscale train llama3-8b --system hgx_b200_8 --dp 8 --layers 2 --fast` |
| training step over InfiniBand | `rackscale train llama3-8b --system h100_2node_ib --dp 16 --layers 2 --fast` |
| pipeline-parallel training | `python docs/book/scripts/pipeline_bubbles.py` |
| see what was lowered | add `--dump-workload wl.json` |

Next: [8. Asynchronous RL](08-async-rl.md).
