# 14. Why cycle level, and how far to trust a simulated rack

Chapters 11 to 13 wrote every plan as formulas, and the simulator then disagreed with those
formulas by up to 2×. This chapter collects the disagreements and asks three questions: which
of them a cheaper model would have caught, what in the simulator is real SASS and what is a
stand-in, and how far a cycle-level result for a 72-GPU rack can be trusted when only one GPU
of it was ever measured. It ends with what it would take to make these numbers trustworthy to
the standard of a research lab, and with what the tool is for beyond performance engineering.

## 14.1 Three models of the same step

| model | what it knows | time per answer | where |
|---|---|---|---|
| **closed form** (`scaling.estimate`) | FLOPs, bytes, link bandwidths and latencies, perfect overlap | microseconds | chapter 11 |
| **roofline replay** (`scaling.replay`) | the exact lowered program: every kernel, collective, stream and dependency, each at its roofline | milliseconds to seconds | chapter 11 |
| **cycle level** (`rsim`) | each kernel's warps, pipes, caches and memory queues at the measured, power-capped clock; NCCL's algorithms as transfers or packets on modelled links and switches | seconds to minutes | every chapter |

The closed form is for ranking plans and reading off scaling laws; it should be the first
thing anyone computes. The replay adds the schedule, so it sees bubbles and dependencies but
still gives every kernel and collective its ideal time. The cycle-level model is the only one
of the three that can be wrong in the same ways a GPU is.

## 14.2 What only the cycle-level model saw

Every row below changed a number by more than the noise of the model, and none is in the
closed form. The sizes are from this book's simulations (GB300 presets, B300 calibration).

| effect | size | chapter |
|---|---|---|
| **The power-capped clock.** Training GEMMs run at about 1455 MHz, not 2032 | 2620 µs of the 4025 µs gap between the 1-GPU training step and its roofline replay (65%) | 11.6 |
| **Kernel efficiency.** A 4096 × 28672 × 4096 GEMM reaches 81% of peak at the preset clock, causal FlashAttention 46%; batch-64 decode GEMMs read their weights at 24-62% of HBM bandwidth (47% across a step) | the other third of the training gap; most of the decode gap | 11.6, 13 |
| **Collective efficiency.** NCCL's algorithms, protocols and launch costs on the modelled fabric | 32 MiB NVLS all-reduce 2.2× its closed form; ring reduce-scatter and all-gather about 75% of the link; a 1 MiB decode all-reduce 27% | 11.6 |
| **Communication takes SMs.** A send/recv on 2 SMs is a rate-limited copy | a 32 MiB pipeline hop at 155 GB/s, not 900 | 11.6 |
| **The schedule.** Which transfer leaves first decides what is exposed | the same pipeline: 1F1B +729 µs of exposed time, ZB-H1 −954 µs | 11.6 |
| **Two jobs on one GPU.** Co-resident CTAs share SMs, L2 and HBM cycle by cycle | +25% / +20% when co-simulated, against +44% / +33% when kernels are time-sliced (fast mode) | 12.5 |
| **Traffic coupling.** A KV stream delays the small messages of the decode pool's all-reduces | all-reduces slow by 1.2× the stream's share of the link | 12.4 |
| **Bytes vs tokens.** H-Net matches a BPE Transformer in FLOPs per byte | but decodes 2.2× slower per byte at batch 64; counting bytes moved explains 1.3× of it, and the rest is its many small kernels | 13 |

A careful person with a calculator could have estimated three of these: the clock (chapter 1
measures it), the SM cost of a copy, and the part of H-Net's decode cost that comes from
re-reading weights. The other five, and the rest of H-Net's gap, need a model of the machine.
Several change a decision this book makes: TP8 with sequence parallelism comes out 22% slower
than TP8 without it (the formulas got the direction, not the size), fast mode would have
overstated the cost of sharing a GPU by 1.6-1.75×, and FLOPs parity would have hidden that a
byte-level model needs a larger batch than a token model to be as cheap per byte.

## 14.3 What is real SASS today, and what is a stand-in

The simulator models a GPU down to the warp scheduler, the issue slots, the register banks,
the MIO and tensor pipes, L1, the two-die L2 and HBM. What it executes depends on the kernel:

- **Recorded kernels run their real SASS.** FlashAttention-4 and -2 and the calibration
  microbenchmarks are replayed from NVBit recordings of every warp's instruction stream on the
  B300, including `tcgen05` / UMMA, tensor memory and TMA. FA4 at 4096 tokens is within +0.2%
  of the slowest SM's measured cycles (+1.5% against the average SM), and its per-warp-role
  timelines match hardware to a 1.6% shape error ([BLACKWELL.md §10.1](../BLACKWELL.md)).
  FA2 is +3.6%, from register-bank dispatch that the model cannot yet place, and FA4's
  ring-attention step kernels are −8.5% to +5.1%. Most microbenchmarks are within ±2%, but
  several are the programs the calibration was fitted on, so they check consistency, not
  prediction; the held-out B300 checks are the FlashAttention kernels and a TMA bulk stream,
  which is −17.7%. An FFMA throughput stream is −6.5%, from launch and tail time.
- **Model steps run synthetic kernels.** Every Llama, DeepSeek or H-Net step in chapters 11 to
  13 is built from synthetic GEMM, attention and Mamba kernels that the same pipe model times,
  not from cuBLAS's or cuDNN's binaries. The one exception is the Mamba-2 decode state update,
  which replays the real SASS Triton compiled for the B300. Against measurement on the B300
  (chapter 9.3): 54 cuBLAS GEMMs have a median |error| of 3.9% at decode shapes (mean −0.8%),
  19% at mid sizes and 12% for large GEMMs (mean +5.6%); attention for decode and prefill 10%
  median; the Mamba-2 state update 10%; the SSD chunked scan 24%.
- **The fabric is preset and checked against published numbers.** We have measured no
  multi-GPU system. The NVLink and NVSwitch model is checked against published NCCL,
  TensorRT-LLM and DeepEP measurements: the measured holdout points have a median |error| of
  7.8% in packet mode and 9.1% in transfer mode ([NETWORK.md](../NETWORK.md)).

So "cycle level" is literal for one GPU's recorded kernels, and an approximation everywhere
else. The model-level results inherit the synthetic kernels' errors.

## 14.4 Why bounds and not exact

Blackwell's microarchitecture is not documented below the ISA. Every unit in the model was
inferred from microbenchmarks ([BLACKWELL.md](../BLACKWELL.md)), and three things keep the
error from going to zero:

1. **Units that microbenchmarks cannot separate.** Per-copy TMA latency, overlap and TLB cost
   look the same from the outside; so do per-SM clock spread and start skew. The 11
   experiments that would separate them are listed in [CALIBRATION.md](../CALIBRATION.md)
   and need the GPU.
2. **The clock depends on the work.** Under the power cap a GEMM's sustained clock depends on
   its shape, not only its size: GEMMs of the same FLOP count run anywhere from 1248 to
   1516 MHz. A size-only power model cannot see that, and it is most of the remaining
   large-GEMM error.
3. **Errors cancel.** Fixing one term has twice uncovered a second error the first was hiding:
   the TMA stream went from −6.2% to −17.7% when a double-counted far-partition penalty was
   removed, and FA2 went from +1.6% to +3.6% when the SM sub-partitions started taking turns
   as the hardware does. This is why the book reports every kernel's signed error, not a pass
   count: a table of small errors can be two large errors pointing in opposite directions.

## 14.5 How per-kernel error carries to a node and a rack

Write a step's time as the sum of the kernels on its critical path, *T* = Σ *t*ₖ, and each
kernel's relative error as a bias shared by its class plus its own scatter,
*e*ₖ = *b*₍class₎ + *ε*ₖ. The step's relative error is then

&nbsp;&nbsp;&nbsp;&nbsp;*e*_T = Σ *w*ₖ *e*ₖ, with *w*ₖ = *t*ₖ / *T*,

and three rules follow:

1. **A class's bias passes through in proportion to its share of the step.** A training step
   that is 70% large GEMMs inherits 0.7 × +5.6% ≈ +4% from them, at one GPU or at 72.
2. **Scatter averages over distinct kernels, not over launches.** A training step launches
   thousands of kernels but only a handful of distinct shapes, and every launch of a shape has
   the same error. With five shapes carrying the step, a 12% median scatter (σ ≈ 17%) still
   leaves about σ/√5 ≈ 8% on the step. Measuring the few shapes that dominate a step is worth
   more than any amount of averaging.
3. **Scale does not multiply kernel error, but it adds two terms.** Each GPU of an NVL72 runs
   the same kernels as one GPU, so the per-GPU compute error is unchanged. What grows is:
   - **communication's share of the step**, and the fabric is the least measured part of the
     model (8-9% median |error| against published numbers, none of our own);
   - **stragglers**, which the simulator does not have: its GPUs are identical and
     deterministic. Every synchronising collective waits for the slowest GPU. If per-GPU step
     time scatters with standard deviation *σ*, the expected maximum of *n* GPUs is about
     1.42 *σ* above the mean for *n* = 8 and 2.39 *σ* for *n* = 72. At *σ* = 1%, a step that
     synchronises once loses about 1.4% on a node and 2.4% on a rack that the simulator will
     not show. Power capping makes *σ* a property of each GPU's silicon and cooling, which is
     why it has to be measured.

So a simulated rack inherits one GPU's kernel bias unchanged, carries the fabric's error in
proportion to how much time it spends communicating, and is optimistic by the straggler term.

## 14.6 What research-lab rigor would take

Four steps, in order of what they buy:

1. **Replay the real kernels of a model step.** Record or capture cuBLAS, CUTLASS and
   DeepGEMM kernels per shape (chapter 10's capture path does this for any engine), so a
   model step replays real SASS instead of synthetic kernels. This removes rule 2's shape
   scatter, the largest term today.
2. **Run the 11 microbenchmarks** in [CALIBRATION.md](../CALIBRATION.md), which settle the
   open single-GPU misses (TMA from HBM, per-SM clocks under FA4, register-bank dispatch under
   HMMA, the clock under attention).
3. **Check the node and the rack against measurements we never fit to.** Without our own
   8-GPU node, that means public numbers (nccl-tests results, MLPerf, NVIDIA's DGX Cloud
   recipes, TorchTitan, TRT-LLM and InferenceX tables) at matching configs, with the
   simulator's predictions committed before the comparison
   ([calibration/nodeday](../../calibration/nodeday/README.md)). That checks the fabric and
   the composition of kernels, collectives and streams into a step; a node of our own, if
   one turns up, runs the same scripts.
4. **Report distributions, not single numbers:** per-SM and per-GPU spread and run-to-run
   variance, so the straggler term of 14.5 is measured, not assumed.

Steps 1 to 3 need GPU time; nothing in this chapter used any.

## 14.7 What it is for, beyond performance engineering

- **Explaining published results.** A paper's speed-up can be traced to the stall, bubble or
  packet behind it, and the explanation checked by changing one thing.
- **Predicting hardware nobody has.** A chip's units, clocks and links are inputs: change the
  HBM bandwidth, the NVLink domain or the tensor-core width and rerun.
- **Designing models for the machine.** Chapter 13 compares tokens and bytes on Blackwell before
  anything is trained; the same works for Mamba against attention or MoE shapes.
- **A sandbox for agents.** Every result in this book is one tool call (`scaling_math`,
  `scaling_ladder`, `check_scaling_math`, `simulate_colocated`, `simulate`), and a call takes
  seconds rather than hours on a rack. That makes systems work an environment an agent can
  explore and be graded in.
- **Learning.** Every number decomposes into the effect behind it, which is how this book
  teaches.

## Exercises

1. A training step is 60% large GEMMs (mean error +5.6%) and 40% attention with a mean error of
   −3%. What error do you expect on the step?
2. Per-GPU step time scatters by 2% (one standard deviation). How much longer is a step that
   ends in one all-reduce over 72 GPUs than the simulator's?
3. Why does the power-capped clock stretch training but barely touch decode (chapter 11.5:
   +2620 µs against +0)?
4. A decode step of a 24-layer model launches 266 kernels, and each is simulated. Why does that
   not make the step's error small?

Answers:

1. 0.6 × 5.6% + 0.4 × (−3%) = **+2.2%**, before scatter.
2. About 2.39 × 2% ≈ **4.8%** longer, which the simulator does not show.
3. The cap bites when the tensor cores draw full power for long enough. Training GEMMs run for
   milliseconds at full tensor throughput; decode kernels are memory bound and short, so the
   GPU stays under its power limit.
4. Errors average over distinct shapes, not launches: the 266 launches are 8 distinct kernels,
   most repeated once per layer, so their errors add coherently.

## Reproduce

| what | where |
|---|---|
| the gap decomposition | `python research/parallelism/check_math.py` (chapter 11) |
| shared GPUs, fast and exact | `python research/colocation/run.py --only shared` (chapter 12) |
| tokens vs bytes | `python research/bytes/run.py` (chapter 13) |
| kernel errors against measurement | `rsim validate calibration/results/b300-sxm6-ac-sm103/ncu_reference.json --gpu b300`; `python calibration/scripts/check_gemms.py --gpu b300` |
| the network against published numbers | [NETWORK.md](../NETWORK.md) |

Next: back to the [contents](README.md).
