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
