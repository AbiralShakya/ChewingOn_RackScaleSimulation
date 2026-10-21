# 8. Asynchronous RL across mixed hardware

Reinforcement learning for language models alternates two very different workloads.
**Rollout** generates completions: it is decode (chapter 6), weight-bandwidth bound, and
latency sensitive. **Training** updates the policy on those completions: it is prefill-shaped
forward and backward (chapter 7), compute bound. Between them the new weights must reach every
rollout GPU. Asynchronous RL runs the two on separate GPUs at the same time, with the samplers
one policy version behind, and that makes the weight transfer a first-class part of the step.
Systems such as [ThunderSync](https://seilk.github.io/thundersync/) and Modal's
[mixed-precision, mixed-accelerator samplers](https://modal.com/blog/toward-rl-on-mixed-precision-mixed-accelerator-samplers)
attack exactly that transfer.

**Everything in this chapter is simulated on presets** (H100, B200, InfiniBand).

## 8.1 The three costs

For a policy with *P* parameters in *b* bytes each, rollout batch *B* per sampler, *n* decode
steps per policy version, and *R* samplers fed by *R*<sub>t</sub> trainers:

```
T_rollout ≈ n · T_decode(B)                        (chapter 6)
T_train   ≈ 6 · P · T_train_tokens / (N_t · π_eff) (chapter 7)
T_sync    ≈ P · b · (R / R_t) / W_link             (each trainer pushes to R/R_t samplers)
```

The step of an asynchronous pipeline is `max(T_rollout, T_train) + exposed T_sync`. The sync
term is the one that scales badly: it grows with the model, and it crosses whatever link
joins the two pools, which in a mixed cluster is InfiniBand at 50 GB/s per GPU, not NVLink.

## 8.2 A simulated step

[`configs/mixed_rl_h100_b200.yaml`](../../configs/mixed_rl_h100_b200.yaml): one node of 8 H100
samplers and one node of 8 B200 trainers on rail-optimised 400G InfiniBand. Llama-3-8B
truncated to 2 layers (to keep the run short), 64 sequences per sampler at 2k context, 4
decode steps, then one training micro-batch of 2048 tokens per trainer, then each trainer
pushes the 2-layer weights (872 MB) to its sampler:

```bash
python examples/async_rl_mixed.py
```

| | per GPU |
|---|---|
| step | **25.9 ms** |
| H100 sampler: kernels (4 decode steps) | 5.2 ms |
| B200 trainer: kernels (forward + backward) | 4.8 ms |
| trainer reduce-scatters (2 × 436 MB, 339 GB/s busbw) | overlapped with the backward, as in chapter 7 |
| weight push, trainer *i* → sampler *i* | 872 MB at **42 GB/s**, about 21 ms |
| GPU idle | **80.6%** |

**About 80% of the step is the weight push.** Compute is about 5 ms on each side, and neither
pool is the bottleneck. The push is a single point-to-point stream per pair over one 50 GB/s
NIC, reaching 42 GB/s. The samplers' decode kernels look like chapter 6: the H100 LM head and
MLP GEMMs stream at about 3.0 TB/s (90% of the 3.35 TB/s preset), and decode attention at
batch 64 and 2k context runs at 3.0 TB/s.

## 8.3 Scaling the weight push to the full model

The 2-layer weights are 872 MB. The full Llama-3-8B in BF16 is 16 GB. At the simulated 42 GB/s
per pair:

| policy | bytes | push time at 42 GB/s | at 400 GB/s (NVLink-class) | FP8 push at 42 GB/s |
|---|---|---|---|---|
| 8B, BF16 | 16 GB | **380 ms** | 40 ms | 190 ms |
| 70B, BF16 | 141 GB | **3.4 s** | 350 ms | 1.7 s |

A 70B policy's rollout of a few hundred decode steps at a few ms each takes about a second, so
an unoverlapped BF16 push over InfiniBand would take longer than the rollout it feeds. The
levers, each of which the simulator can express as a workload or a system change:

1. **Overlap the push with rollout.** Stream layers as the trainer finishes them and swap at
   a step boundary; then only the last layer's transfer is exposed.
2. **Send fewer bytes.** FP8 or FP4 weights for the samplers (Modal's mixed-precision
   samplers) halve or quarter the push, and also speed up decode (chapter 6.5).
3. **Broadcast, don't unicast.** One trainer sending to many samplers is a broadcast; a tree
   or pipelined broadcast uses every link once instead of the trainer's NIC *R*/*R*<sub>t</sub>
   times.
4. **Put the samplers inside the NVLink domain.** On an NVL72, trainers and samplers share
   900 GB/s links and the push drops 18×.

## 8.4 Heterogeneous samplers

Decode is weight-bandwidth bound, so a sampler's throughput tracks its HBM bandwidth, not
its FLOPs. From the presets, an H100 streams 3.35 TB/s and a B200 7.7 TB/s, so a B200 sampler
is about 2.3× an H100 sampler at the same batch; for training, the B200's 2.25 PFLOP/s against
989 TFLOP/s is also about 2.3×. Between these two generations the ratio is the same for both
jobs, so the case for putting older GPUs on rollout rests on cost per TB/s and availability,
and on running the samplers at lower precision. What the simulator adds is the check that
the weight path between the two pools does not become the bottleneck.

## Exercises

1. With 8 samplers and only 2 trainers, how long does the 16 GB push of a Llama-3-8B take at
   42 GB/s per trainer NIC with unicast? With a pipelined broadcast?
2. A rollout of 256 decode steps at 3 ms each overlaps a training step of 600 ms. Which is the
   bottleneck, and how much of a 380 ms push can hide?
3. Why does the simulated step put about 5 ms of compute on each side but 25 ms overall?

<details><summary>Answers</summary>

1. Each trainer serves 4 samplers: 4 × 16 GB / 42 GB/s ≈ **1.5 s**. A pipelined broadcast
   sends each byte once per link, about 16 GB / 42 GB/s ≈ **0.38 s** plus pipeline fill.
2. Rollout is 768 ms, longer than training (600 ms), so samplers set the pace. The push can
   start when training ends and must finish by the end of rollout: 168 ms of the 380 ms hides,
   212 ms is exposed unless the push is streamed by layer during training.
3. The weight transfer (872 MB per pair at about 42 GB/s ≈ 21 ms) runs after both sides
   finish and is not overlapped in this workload, so it adds directly to the step.

</details>

## Reproduce

| number | command |
|---|---|
| the simulated step | `python examples/async_rl_mixed.py` |
| change the cluster | edit `configs/mixed_rl_h100_b200.yaml` (link speeds, NICs, GPU presets) |
| change the schedule | `lower.async_rl(...)` in `python/rackscale/lower.py`: decode steps, batch, trainers |

Next: [9. How far to trust the numbers](09-validation.md).
