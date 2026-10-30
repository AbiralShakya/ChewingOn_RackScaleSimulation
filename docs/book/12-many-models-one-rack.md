# 12. Many models, one rack

A GB300 NVL72 is one NVLink domain: 72 GPUs, 20.7 TB of HBM, and every GPU reaching every
other at 900 GB/s each way through 18 switch chips. The largest models need most of it
(DeepSeek-V3 decodes on 64 GPUs with expert parallelism), but most jobs need a slice: a
Llama-3-8B replica fits on one GPU, and a 70B decode server wants eight. So a real rack runs
several jobs at once: a fleet of small models beside a large one, a prefill pool feeding a
decode pool, a trainer next to the samplers that generate its rollouts. This chapter asks
what those jobs cost each other, first with formulas and then with the simulator.

Everything here is in [`rackscale.colocate`](../../python/rackscale/colocate.py), callable from
Python and from the `simulate_colocated` agent tool, and reproducible with
[`research/colocation/run.py`](../../research/colocation/run.py). **All numbers are simulated
on the GB300 NVL72 preset; none was measured on more than one GPU.**

## 12.1 Three ways to share a rack

| | what the jobs share | first-order cost to a job |
|---|---|---|
| **partition** by GPU | NVSwitch chips (non-blocking), their SHARP reduction units | none |
| **flows** between partitions (KV caches, weights) | the receiving GPU's NVLink ports, the SMs that run the copy, HBM write bandwidth | *R*/*W* on link-bound collectives, *d* · *s*/148 on SM-bound kernels, *R*/*β* on HBM-bound kernels |
| **share GPUs** (MPS, green contexts) | SMs, L2, HBM, the launch queue | up to the other job's whole time |

*R* is the flow's rate into one GPU, *W* = 900 GB/s and *β* = 6.82 TB/s as in chapter 11,
*s* the SMs a copy kernel occupies and *d* the fraction of time a copy is running. The first
row is zero because an NVL72 GPU has its own link to each of the 18 switch chips and each
switch chip is non-blocking: two jobs on different GPUs never queue on the same link. The
only shared component inside the switch is the SHARP unit that reduces NVLS all-reduces.

## 12.2 How the simulator puts several models on one rack

A **tenant** is one model with its own phase (`decode`, `prefill` or `train`), parallel plan
and GPUs:

```python
from rackscale import colocate as co
from rackscale.scaling import Plan

tenants = [
    co.Tenant("ds", "deepseek-v3", "decode", Plan(ep=64), gpus=range(64), batch=64, context=4096),
    co.Tenant("8b", "llama3-8b", "decode", Plan(tp=8), gpus=range(64, 72), batch=64, context=8192),
]
print(co.summary(co.run(tenants, "gb300", layers=4)))
```

Each tenant is lowered on its own (chapter 11's lowering, any plan), then its ranks are placed
on its GPUs: collective groups, send/recv peers, events and kernel names are rewritten to rack
GPU ids and tagged with the tenant. The result is one workload with an explicit program per
GPU, so every tenant runs on the same simulated NVLink links and switch chips at the same
time. Each tenant loops `steps` times so the tenants overlap, and its step time is read from
its **own streams** (`stream_finish_us` in the report), never from another job sharing its
GPU. The same tenant simulated alone on the same GPUs gives the baseline; the ratio is the
interference.

Two extras complete the picture:

* **Flows** are traffic that belongs to no model's program: `Flow("kv", src=range(8),
  dst=range(8, 16), bytes=4 << 20, every_ns=...)` sends from each source GPU to the
  destination at the same position, one transfer issued on a clock (the `host` op), so the
  offered rate is `bytes / every_ns`. A transfer still running when the next is due delays it,
  and the report gives the rate actually delivered.
* **Shared GPUs.** Give two tenants overlapping `gpus` and each shared GPU runs both programs as
  separate streams. Fast mode replays each kernel's solo timing on the SMs left free, so
  sharing becomes time-slicing; exact mode (`fast=False`) runs both tenants' CTAs on the same
  SMs cycle by cycle, competing for issue slots, L2 and HBM.

A job's timing does not depend on unrelated jobs elsewhere in the rack: the network model
seeds each transfer's starting NVLink plane from the communicator's own GPUs, and a test
(`disjoint_jobs_do_not_perturb_each_other`) pins it. Before that fix an unrelated TP4 group on
other GPUs moved a TP4 all-reduce by 2%, which would have read as interference.

## 12.3 A partitioned rack: zero, and why

DeepSeek-V3 decodes with EP64 on GPUs 0-63 (batch 64 per GPU, 4k context) while Llama-3-8B
decodes with TP8 on GPUs 64-71 (batch 64, 8k context), four layers each, three steps:

| tenant | plan | GPUs | step together | alone | slowdown |
|---|---|---|---|---|---|
| DeepSeek-V3 decode | EP64 | 64 | 1520.4 µs | 1520.4 µs | +0.00% |
| Llama-3-8B decode | TP8 | 8 | 535.7 µs | 535.7 µs | +0.00% |

Nothing is shared, so nothing interferes: the EP64 all-to-alls of DeepSeek-V3 use the links of
GPUs 0-63, the 8B's NVLS all-reduces use those of GPUs 64-71 and the SHARP units, which the
all-to-all does not touch. The same holds for a busier rack:

| tenant | plan | GPUs | step together | alone | slowdown |
|---|---|---|---|---|---|
| Llama-3-70B training, 4k tokens | TP8 × DP2, FSDP | 16 | 6433.3 µs | 6433.3 µs | +0.00% |
| Qwen3-30B-A3B decode, batch 128 | TP4 | 4 | 397.3 µs | 397.3 µs | +0.00% |
| 4 × Llama-3.2-1B decode, batch 64 | 1 GPU each | 4 | 478.1 µs | 478.1 µs | +0.00% |

(two layers per model; 22 GPUs busy, 50 idle)

Here the trainer's TP8 all-reduces and the MoE server's TP4 all-reduces both use NVLS, so
both reduce inside the same 18 switch chips. They still did not queue: each chip's SHARP unit
is modelled at the chip's port bandwidth, 72 × 50 GB/s, while an 8-GPU NVLS all-reduce feeds
each chip at 8 × *W*/18 = 400 GB/s.

What would couple partitions on real hardware, and is not in this model: the rack's power
and cooling budget (each GPU here has its own power-capped clock, chapter 1, but no shared rack
limit), the host CPUs and their launch queues, and, past one rack, the shared InfiniBand
fabric.

## 12.4 Disaggregated prefill and decode: the KV stream

Disaggregated serving runs prefill on one pool of GPUs and decode on another, and ships each
request's KV cache between them. The cost to decode is set by the rate *R* at which KV arrives
per decode GPU:

```
R = S · k / (tp_decode · t_prefill) · (prefill groups per decode group)
```

For Llama-3-70B, *k* = 2 × 80 layers × 8 KV heads × 128 × 2 bytes = 320 KiB per token. An
8k-token prompt is 2.7 GB of KV, 335 MB per GPU of a TP8 decode group. The closed form puts a
TP8 prefill of 8k tokens at 86 ms, so a 1:1 pair of pools gives *R* ≈ 3.9 GB/s per decode GPU,
and a 4:1 ratio about 16 GB/s. Against *W* = 900 GB/s and *β* = 6.82 TB/s that is small, and
the formulas of 12.1 predict well under 1% at 4 GB/s.

The simulation streams 4 MiB chunks (one layer's KV of an 8k prompt, per GPU) from GPUs 0-7 to
a TP8 decode group on GPUs 8-15 over NCCL send/recv on 4 SMs, issued on a clock, while the
decode group runs Llama-3-70B at batch 64 and 8k context:

| KV rate per decode GPU | decode step | slowdown | of which all-reduces | of which kernels |
|---|---|---|---|---|
| 0 (alone) | 365.2 µs | | 735 µs per 8 steps | 2080 µs per 8 steps |
| 2 GB/s | 365.3 µs | +0.02% | +0.12% | −0.02% |
| 10 GB/s | 366.4 µs | +0.32% | +1.3% | −0.03% |
| 25 GB/s | 367.6 µs | +0.65% | +2.8% | −0.05% |
| 50 GB/s | 372.0 µs | +1.87% | +7.7% | −0.12% |
| 100 GB/s | 376.9 µs | +3.19% | +13.4% | −0.26% |

Every rate was delivered in full (4 MiB send/recv on 4 SMs tops out near 145 GB/s in this
network model).

All of the cost is in the decode group's TP all-reduces, and the formula of 12.1 predicts it:
the all-reduces are 26% of this decode step, the KV stream takes *R*/*W* of each decode GPU's
NVLink ports (11% at 100 GB/s), so the step should slow by 0.26 × 11% = 2.9%. The simulator
says all-reduces +13.4% (1.2× *R*/*W*, since a chunk in flight delays the all-reduce's small
messages by more than its share of bandwidth) and the step +3.2%. The SMs the copies take cost
nothing measurable: decode GEMMs are HBM bound, so 144 SMs read the weights about as fast as
148 (the kernels even came out 0.26% faster on 144). HBM contention from the KV writes
(*R*/*β* = 1.5% of bandwidth at 100 GB/s) is not in fast mode.

So a disaggregated pool pays about **0.3% of decode time per 10 GB/s per GPU**: the 2-16 GB/s
of real 70B pools cost under 0.6%. What sets the price is the decode plan's communication
share; a decode step with no TP collectives would barely notice the stream.

## 12.5 Sharing GPUs: where fast mode and cycle level disagree

Two models on the same four GPUs: Llama-3-8B TP4 at batch 32 and Llama-3.2-1B TP4 at batch 64,
both decoding with 4k context, one layer each.

| tenant | mode | step together | alone | slowdown |
|---|---|---|---|---|
| Llama-3-8B, batch 32 | fast | 210.4 µs | 145.6 µs | +44.5% |
| Llama-3.2-1B, batch 64 | fast | 165.7 µs | 124.6 µs | +33.0% |
| Llama-3-8B, batch 32 | **exact** | 173.4 µs | 138.3 µs | **+25.4%** |
| Llama-3.2-1B, batch 64 | **exact** | 144.6 µs | 120.3 µs | **+20.2%** |

Fast mode replays each kernel's solo timing on the SMs left free, and a kernel that would get
fewer SMs than it can use waits for the running ones to drain, so the two tenants mostly take
turns: the 8B pays 44%. Exact mode runs both tenants' CTAs on the same SMs at once: when one
kernel's warps wait on HBM, the other's issue, and what remains is real contention for issue
slots, L2 and HBM bandwidth. The 8B pays 25%, and the 1B 20% instead of 33%. Fast mode
overstates the interference by about 1.6-1.75×. (Exact mode is also 5% faster alone, because it
carries L2 contents from one kernel to the next where fast mode times each kernel on a cold
GPU.)

Neither number is in a closed form. Two HBM-bound kernels sharing bandwidth would each slow by
the other's bytes over their own; two latency-bound ones could overlap almost for free. Decode
at these batch sizes sits between the two, which is why a cycle-level co-run is the only way
to get the number.

## 12.6 What to take away

* On an NVL72, **partitioning by GPU is free**: a job on its own GPUs runs as if the rack were
  empty, whatever its neighbours do, as long as nothing reduces through the same SHARP units
  at the same time.
* **Flows cost what they occupy.** A KV stream costs decode about 0.3% of step time per
  10 GB/s per GPU, mostly in the all-reduces that share the decode GPUs' NVLink ports. The
  2-16 GB/s that real 70B pools produce costs under 0.6%.
* **Sharing a GPU is never free and fast mode overstates it.** Use exact mode for any study
  where tenants share SMs; the gap between the two modes is the part of interference that only
  a cycle-level model sees.

## Exercises

1. A 1:1 pair of Llama-3-70B pools serves 32k-token prompts, with prefill at 4× the 8k time.
   What KV rate arrives per decode GPU, and what does the sweep in 12.4 predict for decode?
2. Which single change to the partitioned scenario of 12.3 would make the two tenants slow
   each other down, and through which component?
3. In 12.5, why does fast mode report a larger slowdown than exact mode for the 8B?

<details><summary>Answers</summary>

1. 32k tokens × 320 KiB / 8 = 1.34 GB per decode GPU per prompt, over 4 × 86 ms ≈ 0.34 s:
   **≈ 3.9 GB/s**, the same as at 8k, because prompt length scales the KV and the prefill
   time together (prefill attention grows faster than linearly, so a little less). The sweep
   gives about 0.1% (the formula: 0.26 × 3.9 / 900 = 0.11%).
2. Give the DeepSeek tenant NVLS all-reduces (for example a TP8 attention layout instead of
   EP): both tenants' reductions then queue on the same 18 SHARP units. Overlapping GPUs or a
   flow into the 8B's GPUs would also do it.
3. Fast mode replays each kernel's solo timing on the SMs left free and makes a kernel that
   wants more SMs wait for the others to drain, so two tenants' kernels mostly take turns.
   Exact mode runs both tenants' CTAs side by side: one's memory stalls are hidden under the
   other's issue, and only real contention for issue slots, L2 and HBM remains.

</details>

## Reproduce

| what | command |
|---|---|
| any rack | `co.run([co.Tenant(...), ...], "gb300", layers=..., flows=[co.Flow(...)])` |
| the scenarios in this chapter | `python research/colocation/run.py` (about 5 minutes on 4 cores) |
| one scenario | `python research/colocation/run.py --only disagg` |
| from an agent | MCP tool `simulate_colocated` |

Next: back to the [contents](README.md).
