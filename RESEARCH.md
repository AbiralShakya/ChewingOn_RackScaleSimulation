# Research agenda

rackscale exists to answer one kind of question: **where do the cycles go, and what does an idea
change about that?** This document turns that into a research program in the style of the
[Dao AI Lab](https://dao-lab.ai/publications/): find the real hardware bottleneck, co-design the
algorithm and kernel around it, and ship both as open source with a careful IO and roofline
analysis.

The simulator's role in that loop is the instrument. For each study we simulate the *baseline*
and the *researched idea* side by side at cycle level, check both against real kernels on the
single GPUs we have, and then use the simulator alone to carry the comparison to scales and
hardware we do not have: HGX B200/B300, GB200 NVL72, HGX H100, mixed clusters.

Contents

- [Method: validate on one GPU, extrapolate with the simulator](#method-validate-on-one-gpu-extrapolate-with-the-simulator)
- [Priorities at a glance](#priorities-at-a-glance)
- [Studies](#studies)
- [Future directions](#future-directions)
- [Simulator work these studies need](#simulator-work-these-studies-need)
- [Publication map](#publication-map)

---

## Method: validate on one GPU, extrapolate with the simulator

Every study follows the same four steps. A result that skips step 2 is labelled *model
behaviour*, the same convention [docs/RESULTS.md](docs/RESULTS.md) already uses.

1. **Run the real thing on one GPU.** We have two calibration targets on Della: a **B300 SXM6**
   (sm_103, Blackwell Ultra) and **A100** (sm_80, both the 80 GB part and a 40 GB MIG slice).
   Run the lab's published kernel (FlashAttention, Mamba, causal-conv1d, QuACK, SonicMoE, ...)
   and its baseline under Nsight Compute. Record time, achieved FLOP/s and bytes, the top stall
   reasons, and the SASS (`cuobjdump -sass`). Shapes are chosen so a single GPU covers the
   per-GPU slice of the large-scale configuration we care about (for example, the
   per-GPU expert GEMMs of an EP72 decode step, not a toy shape).
2. **Reproduce it in the simulator on the same GPU.** Feed the simulator the same kernel, as
   real SASS through `rsim sass` where possible, otherwise as a synthetic kernel with matched
   tiling. Use the calibrated preset (`configs/hgx_b300_8_calibrated.yaml`, and the A100 preset
   once it lands). The acceptance band is **±10% on kernel time** and the **same dominant
   stall reason**; outside that band the study first fixes the model (and the fix goes in the
   changelog with its provenance), it does not tune a fudge factor.
3. **Extrapolate with the simulator only.** With the per-kernel model validated, run the
   end-to-end workload at scale: 8-GPU HGX, NVL72, H100 comparisons, multi-node IB, mixed
   accelerators. Sweep the knobs the paper's argument depends on (batch, context, EP/TP degree,
   NVLink bandwidth, SM clock).
4. **Ship three artifacts.** A simulator study (script in `examples/research/`, expected output
   checked into `docs/research/`), the validated single-GPU kernel numbers with the `ncu`
   reports, and an installable reproduction. Where the study produces a better kernel or
   schedule, that kernel is tested on the B300 before any claim about it is made.

Two consequences worth stating:

- **Cross-generation claims go through both GPUs.** A100 and B300 bracket three generations
  of tensor core (HMMA → HGMMA → tcgen05/TMEM). A model that is right on both ends is much more
  credible on H100 and B200, which we can only simulate.
- **Fabric claims are model behaviour until the fabric is calibrated.** The calibration suite
  is single-GPU today. NVLink5/NVSwitch numbers come from the presets and from the
  collective bus-bandwidth tests, so rack-level conclusions are stated with that caveat until
  a multi-GPU calibration exists (see [simulator work](#simulator-work-these-studies-need)).

---

## Priorities at a glance

Ranked by (how strong the bottleneck story is) × (how much of it the simulator can already do)
× (how directly it builds on something we have measured).

| # | study | lab work it builds on | status | sim readiness |
|---|---|---|---|---|
| 1 | [FlashAttention generations on Blackwell Ultra](#1-flashattention-generations-on-blackwell-ultra) | FA1–FA4, Flash-Decoding | **running** (own thread) | high |
| 2 | [Mamba vs attention at rack scale](#2-mamba-hybrids-and-attention-at-rack-scale) | Mamba 1/2/3, causal-conv1d, Marconi, M1, Mamba-in-the-Llama | **running** (own thread) | medium: needs scan/conv kernels |
| 3 | [Low precision: advertised vs delivered](#3-low-precision-advertised-vs-delivered) | FA3 FP8, QuACK, fast-hadamard-transform, BitDelta | next | high |
| 4 | [MoE decode and training at rack scale](#4-moe-decode-and-training-at-rack-scale) | SonicMoE, Opportunistic Expert Activation | next | high |
| 5 | [Decode-efficient attention variants](#5-decode-efficient-attention-variants) | Hardware-Efficient Attention for Fast Decoding (GTA/GLA), Flash-Decoding | next | high |
| 6 | [Async RL: rollout, training, weight sync](#6-async-rl-rollout-training-and-weight-sync) | Beat the Long Tail, Speculative Speculative Decoding | next | medium |
| 7 | [Speculative decoding and sampling kernels](#7-speculative-decoding-and-sampling-kernels) | Medusa, Speculative Speculative Decoding, SonicSampler | later | medium |
| 8 | [Communication-overlapping architectures](#8-communication-overlapping-architectures) | Ladder-residual | later | high |
| 9 | [Orthogonal optimizers at scale](#9-orthogonal-optimizers-at-scale) | Gram Newton-Schulz, Dion3 | later | medium |
| 10 | [Test-time compute and parallel reasoning](#10-test-time-compute-and-parallel-reasoning) | Gambit, M1, Thinking Slow Fast, Squeeze Evolve | later | low: needs a serving-level model |
| 11 | [Sub-quadratic attention: log-linear](#11-sub-quadratic-attention-log-linear) | Log-Linear Attention, Hyena, H3 | later | low: needs chunked-scan kernels |
| 12 | [Diffusion language models](#12-diffusion-language-models) | Introspective Diffusion LMs (I-DLM) | later | medium |
| F1 | [Pipeline bubbles](#f1-pipeline-bubbles-at-cycle-level) | Ladder-residual, the async work above | future | low: needs PP lowering |
| F2 | [System 1 / System 2 action decoding for robots](#f2-system-1--system-2-action-decoding-for-robot-vlas) | Pigey (Orchestration Gap), the Mamba line | future | low: needs edge presets |

---

## Studies

Each study lists the **baseline**, the **idea**, the **hardware and fabric effects** that decide
it, **what we measure**, the **single-GPU validation**, the **at-scale simulation**, and what
the simulator **can do today vs what is missing**.

### 1. FlashAttention generations on Blackwell Ultra

*Running in its own thread ("FlashAttention generations on B300").*

- **Baseline:** naive attention (materialised S and P), then FA2.
- **Idea:** the FlashAttention line. FA1 tiles to avoid HBM traffic for S/P; FA2 fixes work
  partitioning; FA3 adds warp specialisation, TMA, WGMMA ping-pong and FP8 on Hopper; FA4
  ([MLSys 2026](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute),
  CuTe-DSL) co-designs the pipeline for Blackwell's "asymmetric scaling": tensor-core throughput
  grew much faster than the exp unit and shared memory bandwidth. It uses `tcgen05` MMAs with
  accumulators in TMEM, optional 2-CTA (`cta_group::2`) MMAs, software emulation of part of
  the `exp2` work on the FMA pipe, and a lazy rescaling threshold for the online softmax.
- **What decides it:** MUFU.EX2 throughput vs tensor throughput, TMEM ↔ register traffic,
  shared memory bandwidth for the second GEMM, register pressure in the softmax warpgroups.
- **Our measured hook:** on B300 we measured **MUFU.EX2 at 32/clk/SM, twice the other MUFU
  functions**. FA4's own tuning table agrees: in `flash_fwd_sm100.py` the sm_103 configurations
  set `ex2_emu_freq = 0` (no emulated exp), while the sm_100 (B200) configurations emulate one
  exp in every 8 to 32. So the exp bottleneck that motivated part of FA4 is largely removed in
  silicon on Blackwell Ultra. The open question is what becomes the bottleneck instead.
- **Measure:** per-generation forward and backward time, achieved FLOP/s, stall breakdown per
  warp role (load, MMA, softmax, correction), for head dims 64/128/192/256, causal and not, BF16
  and FP8, at 1k–128k context.
- **Validate (single GPU):** FA2 on A100; FA2, FA3-style and FA4 on B300, with `ncu`.
- **Extrapolate:** B200 vs B300 (emulation on vs off), H100 (FA3) vs B300 (FA4); the
  sensitivity of FA4 to the EX2 rate and to TMEM latency, by changing one number at a time.
  Context-parallel attention (ring) on NVL72 for long prefill.
- **Today:** the synthetic Blackwell attention kernel already pipelines `tcgen05` with explicit
  `MUFU.EX2`. **Missing:** an FMA-polynomial exp path in the synthetic kernel, TMEM stores
  (`tcgen05.st`) and `tcgen05.cp`, the measured tcgen05 round-trip latency (in the calibration
  file but not wired), and an FA4 SASS regression against `ncu`.

### 2. Mamba, hybrids and attention at rack scale

*Running in its own thread ("Mamba vs attention at rack scale").*

- **Baseline:** full attention transformer (GQA and MLA) at matched quality.
- **Idea:** selective SSMs ([Mamba](https://github.com/state-spaces/mamba)), the SSD
  formulation that turns the scan into chunked matmuls (Mamba-2), Mamba-3's state-space
  refinements (ICLR 2026), and hybrids (Mamba-in-the-Llama, M1, HybriDNA, the Nemotron-H / Jamba
  style mixes). Inference-side: Marconi's prefix caching for hybrids, where SSM states cannot be
  reused at arbitrary prefixes the way KV blocks can.
- **What decides it:** decode is a memory-bound state update (state size × batch) vs a KV-cache
  read that grows with context; prefill is the SSD chunked matmul vs FlashAttention;
  causal-conv1d is a small, latency-sensitive kernel; at rack scale, TP of SSM heads and the
  absence of a growing KV cache change how much batch fits per GPU.
- **Measure:** tokens/s/GPU and latency per token vs batch and context (1k–1M), crossover
  context where hybrids win, HBM bytes per token, and the per-layer bubble share. NVL72 vs HGX
  H100.
- **Validate:** `mamba_ssm` (selective scan, SSD chunked kernels) and `causal-conv1d` on A100
  and B300, against FA2/FA4 at the same shapes.
- **Extrapolate:** full models (dense 8B/70B-class vs Mamba-2 and hybrid at matched params) on
  NVL72 and HGX H100; prefix-cache hit rates à la Marconi changing effective prefill cost.
- **Today:** attention, GEMM and norm kernels; decode/prefill lowering. **Missing:** SSM
  layer types in `models.py`/`lower.py`, synthetic kernels for the selective scan, SSD chunked
  scan and causal-conv1d (or SASS capture of the real ones), and a state-cache model.

### 3. Low precision: advertised vs delivered

- **Baseline:** BF16 GEMMs and attention.
- **Idea:** FP8 (FA3's FP8 path with incoherent processing via Hadamard rotations, using
  [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)), MXFP4 and
  NVFP4 block-scaled GEMMs ([QuACK](https://github.com/Dao-AILab/quack) and CUTLASS), and
  low-bit deltas (BitDelta) for serving many fine-tunes.
- **Our measured hook:** **NVFP4 on our B300 measures 32768 FLOP/clk/SM, the B200 rate (2×
  FP8), not the 1.5× higher Blackwell Ultra figure.** Every block-scaled FP4 variant we tried
  (NVFP4 and MXFP4, 1-SM and 2-SM, A from SMEM or TMEM) gives the same number. A CUTLASS NVFP4
  GEMM cross-check is in progress to separate "our microbenchmark misses the fast path" from
  "this part does not have it".
- **What decides it:** whether a quantised kernel is still tensor-bound once scale-factor
  loads, dequant/requant epilogues, Hadamard rotations and the softmax are counted; for decode,
  bytes per weight matter more than FLOPs.
- **Measure:** delivered vs peak FLOP/s per precision per kernel; end-to-end decode and
  training step time at BF16/FP8/FP4; the fraction of step time where precision changes the
  bound (compute ↔ memory).
- **Validate:** CUTLASS and QuACK GEMMs at BF16/FP8/NVFP4/MXFP4 on B300; FP8 FA3-style kernels;
  INT8/INT4 IMMA on A100 as the older-generation reference (we measured s8 4046 and s4 8091
  FLOP/clk/SM).
- **Extrapolate:** DeepSeek-V3 and Llama-3-405B decode on NVL72 with FP8 and FP4 weights;
  training step with FP8 GEMMs; what a true 1.5× FP4 would buy end to end (a one-number change).
- **Today:** dtype-aware synthetic GEMMs (`bf16`, `fp8`, `fp4`) with per-precision tensor
  rates from calibration. **Missing:** scale-factor traffic and block-scale epilogues in the
  synthetic GEMM, a Hadamard kernel, quantised KV cache in attention.

### 4. MoE decode and training at rack scale

- **Baseline:** DeepSeek-V3-style EP with grouped GEMMs and DeepEP-style dispatch/combine;
  ScatterMoE-style training kernels.
- **Idea:** [SonicMoE](https://github.com/Dao-AILab/sonic-moe) (ICLR 2026): IO- and tile-aware
  MoE kernels whose activation memory stays constant as experts get finer, with 1.87–4.04×
  reported speedups over existing MoE training kernels, now on B200/B300 through QuACK grouped
  GEMMs. Opportunistic Expert Activation: batch-aware routing that reduces how many distinct
  experts a decode batch touches, without retraining.
- **Our simulated hook:** in our DeepSeek-V3 NVL72 runs, **decode is weight-bandwidth bound**
  (7.1 TB/s streamed out of 8 TB/s peak) and **two-micro-batch overlap loses** at 64
  sequences/GPU because each half re-reads the weights while the all-to-all it hides is only
  ~10 µs per layer on NVLink5. Fewer distinct experts per batch attacks exactly the
  weight-read term.
- **What decides it:** expert weight bytes read per step, tile padding at small M per expert
  (wave quantisation on 148 SMs), all-to-all bytes and latency on NVSwitch vs IB, SMs reserved
  for communication, expert placement relative to the switch topology.
- **Measure:** step time and its breakdown (expert GEMM, attention, all-to-all, idle) vs batch,
  EP degree (8/32/72), routing policy (top-k vs opportunistic), and expert granularity; training
  step time and activation memory for SonicMoE vs baseline.
- **Validate:** SonicMoE and a grouped-GEMM baseline on B300 at per-GPU EP72 shapes; the same
  grouped GEMMs on A100 (which SonicMoE does not target, so it is a baseline-only point).
- **Extrapolate:** EP72 on NVL72 vs EP8 × 9 nodes over IB; overlap schedules and expert
  placement co-designed with the NVSwitch fabric; Qwen3-235B and Mixtral alongside DeepSeek-V3.
- **Today:** grouped GEMM, all-to-all, EP lowering, two-micro-batch overlap, fast mode.
  **Missing:** a routing model with a real token→expert distribution (we assume balance), the
  opportunistic routing policy, and SonicMoE's fused kernel structure.

### 5. Decode-efficient attention variants

- **Baseline:** GQA (Llama-3) and MLA (DeepSeek-V3) decode attention.
- **Idea:** Hardware-Efficient Attention for Fast Decoding (COLM 2025): attention variants that
  raise arithmetic intensity in decode, Grouped-Tied Attention (shares K and V state) and
  Grouped Latent Attention (an MLA-like latent that shards cleanly under TP). Flash-Decoding
  (split-KV) for long-context batch-1.
- **What decides it:** decode attention is a KV read; FLOPs per byte of KV decides whether it
  is HBM-bound or tensor-bound. Our MLA decode kernel already runs compute-heavy (~980
  TFLOPS simulated) because 128 query heads share one 576-wide latent. TP sharding of a single
  latent is what GLA addresses.
- **Measure:** decode attention time, bytes/token and arithmetic intensity vs heads, latent
  width, TP degree, batch, context; the end-to-end decode step share.
- **Validate:** FlashMLA / FA decode kernels and GTA/GLA reference kernels on B300; GQA decode
  on A100.
- **Extrapolate:** TP8 vs TP over NVL72 for MLA vs GLA; long-context (128k–1M) decode on NVL72.
- **Today:** attention kernel with arbitrary `heads_q`/`heads_kv`/`d`, MLA lowering.
  **Missing:** GTA/GLA lowering, split-KV (Flash-Decoding) reduction kernel.

### 6. Async RL: rollout, training and weight sync

- **Baseline:** synchronous on-policy RL (generate, then train, then sync).
- **Idea:** asynchronous and off-policy-tolerant pipelines
  ([ThunderSync](https://seilk.github.io/thundersync/), Modal's mixed-precision /
  mixed-accelerator samplers); Beat the Long Tail (MLSys 2026): distribution-aware speculative
  decoding to cut the long-tail rollouts that stall each RL step; Speculative Speculative
  Decoding (ICLR 2026) for the decode itself.
- **Our simulated hook:** in our 8×H100 rollout + 8×B200 trainer run over 400G IB, **~80% of
  the 25.9 ms step is the trainer→sampler weight push**; compute is ~5 ms per side.
- **What decides it:** rollout length distribution (the tail sets the step), weight sync bytes
  and topology (NVLink-local vs IB, broadcast vs bucketed, FP8 transfer), staleness allowed,
  sampler and trainer precision mismatch.
- **Measure:** step time and GPU idle per role, bubble timeline across roles, tokens/s of
  rollout, policy lag in steps, sensitivity to the length tail.
- **Validate:** decode and train-step kernels on B300 and A100 (A100 as the "older sampler"
  point in a mixed fleet).
- **Extrapolate:** NVL72 split into sampler and trainer partitions vs separate racks over IB;
  long-tail rollouts with and without speculative decoding; FP8 weight sync.
- **Today:** `lower.async_rl`, mixed-GPU systems, send/recv and broadcast, bubble reports.
  **Missing:** a rollout length distribution (all sequences finish together today), staleness
  accounting, speculative decoding in the decode lowering.

### 7. Speculative decoding and sampling kernels

- **Baseline:** one-token-per-step decode with a separate sampling kernel.
- **Idea:** Medusa (multiple heads), Speculative Speculative Decoding (overlapping drafting and
  verification), SonicSampler (COLM 2026: tile-aware kernels that unify sampling and
  speculative verification).
- **What decides it:** decode is weight-bound, so verifying k tokens costs little more than 1;
  sampling over a 128k–150k vocabulary is a memory-bound top-k/top-p pass after the LM head
  GEMM; draft and target compete for the same SMs.
- **Measure:** accepted tokens per verify step vs cost per step, LM-head + sampling time, step
  time vs speculation length, per-GPU and on NVL72 at TP/EP.
- **Validate:** LM head + sampling kernels and a draft/verify loop on B300.
- **Today:** GEMM and elementwise kernels. **Missing:** a sampling/top-k kernel, a
  speculative decode lowering (draft model, k-token verify, acceptance model).

### 8. Communication-overlapping architectures

- **Baseline:** standard residual stream with TP all-reduce on the critical path.
- **Idea:** Ladder-residual (ICML 2025): reroute residuals so a layer's all-reduce overlaps
  with the next layer's compute, an architecture change made for parallelism.
- **What decides it:** all-reduce latency vs per-layer compute, which shrinks with TP degree and
  batch; NVLS in-switch reduction on NVL72 vs ring over IB.
- **Measure:** decode and prefill step time vs TP degree (2–72) and fabric, exposed comm time.
- **Validate:** per-layer kernels on B300; the overlap itself is a scheduling question the
  simulator answers once kernels are validated.
- **Today:** TP lowering, NVLS all-reduce, streams/events for overlap. **Missing:** a
  ladder-residual lowering option (a small change in `lower.py`).

### 9. Orthogonal optimizers at scale

- **Baseline:** AdamW (elementwise, communication = gradient reduce-scatter).
- **Idea:** Muon-family optimizers with
  [Gram Newton-Schulz](https://github.com/Dao-AILab/gram-newton-schulz) (iterate on the small
  symmetric Gram matrix, symmetric CuTe-DSL GEMMs, up to 2× faster) and Dion3 (ArXiv 2026,
  full-stack orthogonal updates; implementation in [microsoft/dion](https://github.com/microsoft/dion)).
- **What decides it:** Newton-Schulz needs whole matrices, which conflicts with FSDP/TP
  sharding; extra all-gathers and GEMMs per step vs Adam's elementwise cost.
- **Measure:** optimizer-step time and comm bytes vs model size and sharding on NVL72; share
  of the training step.
- **Validate:** Gram-NS vs standard NS GEMM chains on B300 and A100.
- **Today:** GEMMs, all-gather/reduce-scatter, training lowering. **Missing:** an optimizer
  phase in `train_step` and symmetric-GEMM support.

### 10. Test-time compute and parallel reasoning

- **Baseline:** best-of-N / self-consistency with independent traces.
- **Idea:** Gambit / Thought-Level Beam Search (COLM 2026,
  [code](https://github.com/Dao-AILab/gambit-parallel-reasoning)): a score-guided tournament
  over reasoning prefixes that branches through vLLM prefix caching; M1 and Thinking Slow, Fast
  (Mamba reasoners for cheaper test-time compute); Squeeze Evolve (multi-model orchestration).
- **What decides it:** KV sharing across branches, batch occupancy as traces are pruned and
  branched, Mamba's constant-size state vs growing KV.
- **Measure:** accuracy-matched tokens/s and GPU utilisation over time.
- **Missing:** a serving-level model (continuous batching, prefix cache, request arrival), which
  is also what an end-to-end vLLM/SGLang replay needs. Until then, studies use captured traces.

### 11. Sub-quadratic attention: log-linear

- **Idea:** Log-Linear Attention (ICLR 2026): hierarchical (Fenwick-tree) states give
  log-linear compute and logarithmic memory, between linear attention and softmax attention.
  Older lineage: H3, Hyena, long convolutions.
- **Measure:** the context length where each wins on B300 and across a rack.
- **Missing:** chunked-scan kernels (shared with study 2).

### 12. Diffusion language models

- **Idea:** I-DLM ([code](https://github.com/Introspective-Diffusion/I-DLM)): introspective
  strided decoding, AR-quality output, reported ~5,900 vs ~1,600 tok/s against SDAR at
  concurrency 32 on one H100.
- **What decides it:** diffusion decode turns weight-bound one-token steps into compute-heavier
  multi-token passes, so it moves the roofline point of decode. That is a pure simulator
  question once kernels are validated.
- **Missing:** a diffusion decode lowering (block of tokens per step, acceptance rate).

---

## Future directions

These follow this project. They reuse the same method and need the simulator work listed.

### F1. Pipeline bubbles at cycle level

Pipeline parallelism (GPipe, 1F1B, interleaved 1F1B, zero-bubble schedules, DeepSeek's
DualPipe) trades memory for a bubble whose analytic size, (p−1)/(m+p−1) for the simple
schedules, ignores everything that makes real bubbles bigger: uneven stage times, send/recv
latency on the fabric, kernel launch gaps, and the warmup and cooldown of each stage. A
cycle-level rack simulator can show the bubble as it actually is, per SM, and test schedules
and architectures (Ladder-residual style changes, async RL overlaps) that hide it.

- **Measure:** bubble fraction by cause (dependency, comm, launch, imbalance) per schedule,
  vs micro-batches and stages, on NVL72 and multi-rack IB.
- **Missing:** PP in `lower.Parallel` and `train_step` (stage assignment, schedule generator,
  activation send/recv), and per-cause bubble attribution in the report.

### F2. System 1 / System 2 action decoding for robot VLAs

Recent robot foundation models split control into a slow, large "System 2" vision-language
model that reasons and plans, and a fast, small "System 1" policy that turns the plan into
actions at control rate:

- **Helix** (Figure): a ~7B VLM at roughly 7–9 Hz feeding an ~80M visuomotor policy at 200 Hz.
- **GR00T N1** (NVIDIA): an Eagle VLM System 2 with a diffusion-transformer, flow-matching
  System 1 action head that emits action chunks.
- **π0 / π0.5** (Physical Intelligence): a ~3B PaliGemma backbone with an action expert that
  produces 50-step action chunks by flow matching. The lab's own Pigey work
  ([code](https://github.com/lianegalanti/Pigey)) drives π0.5-DROID with a closed-loop VLM
  orchestrator, which adds a third, slower loop on top.

(Model sizes and rates are as publicly described by each team; we will pin them to papers
before using them as inputs.)

The efficiency questions are Dao-Lab-shaped: what is the minimum-latency kernel schedule for
the System 1 loop when it shares one GPU with System 2, how should the two be co-scheduled
(SM partitioning, MPS/green contexts, preemption), what does each loop's batch-1 decode spend
on weight reads vs compute, and does an SSM backbone or low precision (FP4 on Blackwell)
change the feasible control rate? The target hardware is **Jetson Thor-class Blackwell**
(an edge Blackwell GPU with LPDDR5X rather than HBM), where decode is even more
bandwidth-bound than in the datacenter.

- **Validate:** π0 / GR00T-style inference kernels at batch 1 on B300 and A100, with the
  memory bandwidth effect isolated by the calibrated HBM numbers.
- **Missing:** an edge Blackwell preset (SM count, clocks, LPDDR5X bandwidth and latency),
  multi-rate periodic workloads with deadlines (System 1 at 50–200 Hz, System 2 at 1–10 Hz),
  SM partitioning between concurrent workloads, and deadline-miss reporting.

---

## Simulator work these studies need

Grouped by how many studies each item unblocks.

| item | unblocks | size |
|---|---|---|
| SSM / chunked-scan and causal-conv1d kernel types; SSM layers in `lower.py` | 2, 10, 11, F2 | medium |
| A100 (sm_80) preset from the measured calibration | every study's step 2 on A100 | small |
| Wire tcgen05 round-trip latency, L2 near/far partitions, TMA size curve | 1, 3, 4 | small |
| FMA-polynomial exp path and TMEM stores in the synthetic attention kernel | 1 | small |
| Scale-factor traffic and block-scale epilogues in synthetic GEMMs | 3, 4 | small |
| Routing model with a token→expert distribution; opportunistic routing | 4 | medium |
| Rollout length distribution and staleness in `async_rl` | 6 | small |
| Speculative decode lowering and a sampling kernel | 6, 7 | medium |
| `rsim sass` regression of the lab's real kernels against `ncu` | all | medium |
| Serving-level model (continuous batching, prefix cache, arrivals) | 10, engine replay | large |
| Pipeline parallelism and per-cause bubble attribution | F1 | medium |
| Edge Blackwell preset, multi-rate deadline workloads, SM partitioning | F2 | medium |
| Multi-GPU NVLink/NVSwitch calibration | every rack-level claim | needs hardware access |

---

## Publication map

How each relevant publication from the lab's list
([dao-lab.ai/publications](https://dao-lab.ai/publications/), sourced from the lab site's
bibliography) maps onto the studies. Older work that is not about hardware efficiency
(kernel methods, structured-matrix theory, datasets) is omitted.

| publication | venue | open-source code | study |
|---|---|---|---|
| FlashAttention | NeurIPS 2022 | [flash-attention](https://github.com/Dao-AILab/flash-attention) | 1 |
| FlashAttention-2 | ICLR 2024 | flash-attention | 1 |
| FlashAttention-3 | NeurIPS 2024 | flash-attention (`hopper/`) | 1, 3 |
| FlashAttention-4 | MLSys 2026 | flash-attention (`flash_attn/cute`, `pip install flash-attn-4`) | 1 |
| Flash-Decoding | PyTorch blog 2023 | flash-attention | 1, 5 |
| Hardware-Efficient Attention for Fast Decoding | COLM 2025 | | 5 |
| Mamba | COLM 2024 | [mamba](https://github.com/state-spaces/mamba), [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | 2 |
| Transformers are SSMs (Mamba-2, SSD) | ICML 2024 | mamba | 2 |
| Mamba-3 | ICLR 2026 | | 2 |
| The Mamba in the Llama; M1; Thinking Slow, Fast; HybriDNA; Empirical Study of Mamba LMs | 2024–25 | | 2, 10 |
| Marconi: Prefix Caching for Hybrid LLMs | MLSys 2025 | [marconi](https://github.com/ruipeterpan/marconi) | 2, 10 |
| Log-Linear Attention | ICLR 2026 | | 11 |
| H3, Hyena, Simple Long Convolutions | ICLR/ICML 2023 | | 11 |
| SonicMoE | ICLR 2026 | [sonic-moe](https://github.com/Dao-AILab/sonic-moe) | 4 |
| Opportunistic Expert Activation | 2025 | | 4 |
| QuACK (CuTe-DSL kernels: GEMM, norms, softmax, CE) | software | [quack](https://github.com/Dao-AILab/quack) | 3, 4, 9 |
| fast-hadamard-transform | software | [fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) | 3 |
| BitDelta | NeurIPS 2024 | | 3 |
| Medusa | ICML 2024 | | 7 |
| Speculative Speculative Decoding | ICLR 2026 | | 6, 7 |
| SonicSampler | COLM 2026 | | 7 |
| Beat the Long Tail (speculative decoding for RL) | MLSys 2026 | | 6 |
| Ladder-residual | ICML 2025 | | 8, F1 |
| Gram Newton-Schulz | 2026 | [gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz) | 9 |
| Dion3 | arXiv 2026 | [microsoft/dion](https://github.com/microsoft/dion) | 9 |
| Thought-Level Beam Search (Gambit) | COLM 2026 | [gambit-parallel-reasoning](https://github.com/Dao-AILab/gambit-parallel-reasoning) | 10 |
| Squeeze Evolve | COLM 2026 | [squeeze-evolve](https://github.com/squeeze-evolve/squeeze-evolve) | 10 |
| Introspective Diffusion LMs | COLM 2026 | [I-DLM](https://github.com/Introspective-Diffusion/I-DLM) | 12 |
| Deja Vu (contextual sparsity) | ICML 2023 | | 4 (related: activation sparsity in decode) |
| Decentralized Training in Heterogeneous Environments; Fine-tuning over Slow Networks | NeurIPS 2022 | | 6 (heterogeneous fabrics) |
| Addressing the Orchestration Gap in Generalist Robots (Pigey) | arXiv 2026 | [Pigey](https://github.com/lianegalanti/Pigey) | F2 |
| Long-context State-Space Video World Models | ICCV 2025 | | 2, F2 |

A note on sources: the descriptions of SonicSampler, Opportunistic Expert Activation,
Speculative Speculative Decoding, Beat the Long Tail, Mamba-3, Log-Linear Attention, Dion3 and
the GTA/GLA paper above are short summaries to orient the reader. Each study pins its claims
to the paper itself before it starts.
