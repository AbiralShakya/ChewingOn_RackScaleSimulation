# How to simulate your rack

*A first-principles guide to GPU performance, from one SASS instruction to a 72-GPU NVLink
domain, with every number tied to a measurement or a simulator command.*

This guide is modelled on DeepMind's [How to Scale Your Model](https://jax-ml.github.io/scaling-book/),
which explains TPU systems with rooflines and worked arithmetic. We do the same for NVIDIA
GPUs and the racks built from them, using the hardware we have measured: a **B300 SXM6**
(Blackwell Ultra, sm_103) and **A100 80GB** PCIe and SXM4 boards (Ampere, sm_80), both on
Princeton's Della cluster. Everything larger than one GPU (HGX 8-GPU boards, GB200 and GB300
NVL72, InfiniBand clusters) is simulated with rackscale, and we say so each time.

Each chapter has worked numbers, a short list of exercises with answers, and the commands that
reproduce its numbers. The scripts live in [`scripts/`](scripts/).

## Chapters

| | chapter | what you will be able to do afterwards |
|---|---|---|
| 1 | [Rooflines, and why the roof moves](01-rooflines.md) | predict whether a kernel is compute or memory bound, and by how much; account for the power-capped clock |
| 2 | [Inside one GPU](02-inside-the-gpu.md) | reason about SMs, warp schedulers, pipes, MUFU, the memory hierarchy, and three generations of tensor core (`mma.sync`, `wgmma`, `tcgen05`) |
| 3 | [From a kernel to SASS to cycles](03-kernels-sass-cycles.md) | read SASS control bits, see how a warp-specialised kernel spends its cycles, and know how the simulator turns instructions into time |
| 4 | [Networks and collectives](04-networks-collectives.md) | cost an all-reduce, all-gather or all-to-all on NVLink, NVSwitch (with in-switch reduction) and InfiniBand |
| 5 | [Parallelism: DP, TP, PP, EP, CP](05-parallelism.md) | choose a sharding by comparing communication with compute, and size pipeline bubbles |
| 6 | [Inference: prefill, decode, MoE, and state-space models](06-inference.md) | work out the batch at which decode stops being weight-bandwidth bound, and where Mamba beats attention |
| 7 | [Training](07-training.md) | estimate step time, MFU and the cost of gradient communication |
| 8 | [Asynchronous RL across mixed hardware](08-async-rl.md) | find the bottleneck when rollout and training run on different GPUs |
| 9 | [How far to trust the numbers](09-validation.md) | know which results are measured, which are calibrated simulation, and which rest on presets |
| 10 | [Bring your own workload](10-bring-your-own-workload.md) | capture any engine or kernel library on one GPU and replay it on a rack |
| 11 | [Every parallelism, in closed form, checked cycle by cycle](11-every-parallelism.md) | write the step time of any DP / ZeRO / FSDP / HSDP / TP / SP / CP / PP / EP plan as formulas, and know how far and why the simulator departs from them |
| 12 | [Many models, one rack](12-many-models-one-rack.md) | say what jobs sharing an NVL72 cost each other: partitioned GPUs, KV streams between prefill and decode pools, and models sharing a GPU |
| 13 | [Tokens or bytes: H-Net on one Blackwell GPU](13-tokens-or-bytes.md) | count FLOPs and bytes moved per byte of text for a byte-level hierarchical model, and see why equal FLOPs do not mean equal decode cost |
| 14 | [Why cycle level, and how far to trust a simulated rack](14-why-cycle-level.md) | say which effects only a cycle-level model sees, what in it is real SASS, and how per-kernel error carries to a node and a rack |
| 15 | [Parallelism inside one GPU](15-parallelism-inside-one-gpu.md) | climb the ladder inside one chip (lanes, warps, warp roles, pipeline stages, tiles, clusters, waves, split-K, SMs taken by NCCL) with a closed form and a simulator run at each rung |
| 16 | [Stragglers, flaps and tail latency](16-stragglers-flaps-tail-latency.md) | say what jitter, one slow GPU, a dead or degraded link, a flap and ECMP hash collisions cost a synchronous step, from closed forms checked against the simulator |
| 17 | [Power, cooling and the clock](17-power-cooling-and-the-clock.md) | follow the watts from one kernel to a rack's power trace, size the energy storage it needs, trade the board limit against work per joule, and say what the coolant temperature, 800 V feeds and CDUs do to the clock |
| 18 | [Beyond language models: vision, diffusion, VLMs and recommenders](18-beyond-language-models.md) | say what bounds a ViT encoder, a diffusion transformer, a video DiT sharded with Ulysses, a VLM's prefill and a DLRM-style recommender, and which rung of the ladder each one stresses |

## The ladder

The book is one argument made at every scale: split the work across more workers until a
shared resource saturates, then pay for agreement between them. The same three questions
(what is split, what saturates, what it costs to agree) recur at each rung, from one lane of
a warp to a building full of racks. This is where each rung is treated.

| rung | workers | what saturates | cost to agree | chapters |
|---|---|---|---|---|
| lanes and warps | 32 lanes, up to 16 warps per SMSP | issue slot, registers | 4–27 cycles | [2](02-inside-the-gpu.md), [15.2](15-parallelism-inside-one-gpu.md) |
| warp roles and CTAs on one SM | producer, MMA, softmax, epilogue | the slowest role, 228 KB of shared memory | 21–77 cycles (`bar.sync`), mbarriers | [3](03-kernels-sass-cycles.md), [15.3–15.4](15-parallelism-inside-one-gpu.md) |
| SM pairs (clusters) | 2 SMs | L2 → SM bandwidth | 193 cycles (DSMEM) | [15.5](15-parallelism-inside-one-gpu.md) |
| one GPU | 148 SMs | SM count, HBM, the power cap | 320–706 cycles (L2), 0.55–1.7 µs (launch) | [1](01-rooflines.md), [15.6–15.9](15-parallelism-inside-one-gpu.md), [17.1](17-power-cooling-and-the-clock.md) |
| GPUs in a node | 8 GPUs on NVLink | NVLink bandwidth, SMs for NCCL | µs-scale collectives | [4](04-networks-collectives.md), [5](05-parallelism.md), [11](11-every-parallelism.md) |
| NVL72 rack | 72 GPUs on NVSwitch | NVLink, in-switch reduction, rack power | µs-scale collectives | [6](06-inference.md), [7](07-training.md), [11](11-every-parallelism.md), [12](12-many-models-one-rack.md), [17.2–17.4](17-power-cooling-and-the-clock.md) |
| nodes and racks | many NVLink domains on InfiniBand / RoCE | NIC bandwidth, oversubscribed spines | tens of µs, plus tail latency | [4](04-networks-collectives.md), [8](08-async-rl.md), [11](11-every-parallelism.md), [16](16-stragglers-flaps-tail-latency.md) |
| every rung above one GPU | ranks in lockstep | the slowest rank, the slowest link | the maximum, not the mean | [16](16-stragglers-flaps-tail-latency.md) |
| the facility | racks, power shelves, CDUs | megawatts, coolant temperature and flow | the power swing of a synchronous step | [17](17-power-cooling-and-the-clock.md) |

Chapters 1 to 17 climb the ladder mostly with language models. Chapter 18 climbs it again with
vision encoders, diffusion transformers, VLMs and recommenders, which load different rungs:
attention on one GPU, all-to-all on the rack, HBM gathers on every GPU.

Still being written: sender-side congestion control (DCQCN, Ultra Ethernet) for lossy fabrics
(chapter 16.5 says what is missing).

## Seeing it

Chapters 3, 4 and 6 include figures from `rsim viz`, the simulator's rack-to-SASS viewer. It
goes from the rack fabric to every GPU's streams, from there to the 148 SMs of one GPU, then
to a kernel's stall anatomy, the timeline of each warp, and the SASS listing ranked by stall.
The interactive pages are in [`docs/viz/`](../viz/), and [`docs/VIZ.md`](../VIZ.md) explains
each view. To see any simulation this way, add `rsim viz run ...` or call `rs.viz()` from
Python.

Chapters 6 and 11 to 13 have a companion page, the [NVL72 Scaling Lab](../viz/scaling-lab.html).
Pick a model, a plan and a batch, and it recomputes every closed form of chapter 11 in the
browser, next to the replay and simulator numbers from those chapters: collectives, the
roofline, pipeline schedules, jobs sharing a rack, and tokens against bytes.

## The three kinds of number in this guide

Every number is one of these, and the text says which:

- **Measured.** Taken on the B300 or an A100 with microbenchmarks, CUDA events or Nsight
  Compute. These come from [`calibration/results/`](../../calibration/results/) and the study
  directories under [`research/`](../../research/).
- **Simulated, calibrated.** rackscale running with a GPU's measured `calibration.json`.
  Single-GPU kernels in this class have been checked against Nsight Compute (chapter 9 lists
  how closely).
- **Simulated, preset.** rackscale running on built-in values for hardware we have not
  measured: H100, B200, GB200, every NVLink and InfiniBand fabric, and all multi-GPU systems.
  Read these as what the model predicts, not as hardware claims.

## Hardware at a glance

Measured values unless marked *(spec)* or *(preset)*.

| | A100 80GB PCIe | B300 SXM6 | H100 SXM *(preset)* | GB200 *(preset)* |
|---|---|---|---|---|
| SMs | 108 | 148 | 132 | 148 |
| SM clock | 1395 MHz | 2032 MHz, falling to 1455 MHz under the 1100 W cap | 1830 MHz | 2062 MHz |
| dense BF16 tensor | 2029 FLOP/clk/SM → **306 TFLOP/s** | 8192 FLOP/clk/SM → **2.46 PFLOP/s** | 989 TFLOP/s | 2.50 PFLOP/s |
| FP8 / FP4 tensor | none (INT8 `mma.sync`: 4057 FLOP/clk/SM) | 16384 / 32768 (K=64), 43691 (K=96, 1-SM), 49152 (K=96, 2-SM) FLOP/clk/SM | FP8 1979 TFLOP/s | FP8 5.0, FP4 10 PFLOP/s |
| FP32 (FFMA) | 64 lanes/SM → 19 TFLOP/s | 128 lanes/SM → 77 TFLOP/s | | |
| HBM read | 1.70 TB/s (spec 1.94) | 6.82 TB/s (spec 7.67), 7.19 TB/s with TMA | 3.35 TB/s *(spec)* | 8.0 TB/s *(spec)* |
| L2 | 40 MB, two halves 270 / 422 cycles | 126 MB, near 320 / far 706 cycles | 50 MB | 126 MB |
| HBM latency | 560 cycles (402 ns) | 850 cycles (418 ns) | | |
| NVLink | 12 × 25 GB/s/dir on SXM4 *(spec)* | 18 × 50 GB/s/dir *(preset)* | 18 × 25 GB/s/dir | 18 × 50 GB/s/dir |
| kernel launch, back-to-back / CUDA graph | 1.93 / 0.94 µs | 1.71 / 0.55 µs | | |

Sources: `calibration/results/a100-80gb-pcie-sm80/calibration.json` and
`calibration/results/b300-sxm6-ac-sm103/calibration.json`. Peaks are SM count × measured
clock × measured FLOP per clock per SM, so they are what the silicon sustains in a
microbenchmark, not the datasheet.

## Setup

```bash
git clone https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation rackscale && cd rackscale
cargo build --release -p rsim-cli            # the rsim binary, used by every chapter
python -m venv .venv && . .venv/bin/activate
pip install maturin pyyaml && maturin develop --release   # the rackscale Python package
```

No GPU is needed. A laptop runs every command in this guide. The slowest (a full NVL72
decode step in exact mode) takes a few minutes; the rest take seconds to a minute.

## Notation

We follow the scaling book where it has a convention.

| symbol | meaning |
|---|---|
| *B* | batch (sequences) |
| *T* | tokens in a step (*B* × new tokens per sequence) |
| *D* | model width (hidden size); *F* the FFN width |
| *L* | layers; *S* context length |
| *H*, *H*<sub>kv</sub>, *d* | query heads, KV heads, head dimension |
| *P* | parameters |
| *N* | GPUs in a group |
| FLOP | one multiply or one add; a multiply-add is 2 FLOP |
| AI | arithmetic intensity, FLOP per byte moved to or from HBM |
| busbw | NCCL bus bandwidth (chapter 4) |

Times are in µs unless stated. GB and TB are 10<sup>9</sup> and 10<sup>12</sup> bytes.
