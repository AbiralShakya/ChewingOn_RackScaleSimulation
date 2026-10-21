# 5. Parallelism: DP, TP, PP, EP, CP

A model is split across GPUs along one of five axes. Each axis trades one kind of
communication for one kind of memory or compute saving, and the right choice falls out of
comparing a communication time from chapter 4 with a compute time from chapter 1. This chapter
does that comparison for each axis, with Llama-3-70B as the running example (80 layers,
*D* = 8192, *F* = 28672, 64 query and 8 KV heads of dimension 128, 70.6 B parameters).

**Multi-GPU results here are simulated on presets** (HGX B200, GB300 NVL72, InfiniBand
clusters). Per-GPU kernels on B300 and A100 come from the calibrated models.

| axis | what is split | what moves, per layer | when it pays |
|---|---|---|---|
| data (DP) | the batch | gradients, once per step | always, if the model fits |
| tensor (TP) | each weight matrix | activations, 2 all-reduces per layer | inside an NVLink domain |
| pipeline (PP) | the layers | activations between neighbouring stages | across slow links; costs bubbles |
| expert (EP) | the experts of an MoE layer | tokens, 2 all-to-alls per MoE layer | MoE models, inside an NVLink domain |
| context (CP) | the sequence | K and V, around a ring | long context, when attention dominates |

## 5.1 Data parallelism

Every GPU holds the whole model and processes its own micro-batches. After the backward pass,
gradients are averaged: an all-reduce of 2*P* bytes in BF16. ZeRO and FSDP replace it with a
reduce-scatter (each GPU keeps 1/*N* of the averaged gradients and updates 1/*N* of the
optimizer state) and an all-gather of parameters before they are next used, which moves the
same total bytes but shards the memory.

The gradient communication happens once per step, and it can overlap with the backward pass:
layer *l*'s gradients are ready while layer *l*−1 is still being differentiated. rackscale's
training lowering does exactly that, with one reduce-scatter per layer on a separate stream
(chapter 7). DP is communication bound only if

```
2P · 2(N−1)/N / busbw  >  T_backward
```

which for a 70B model over 50 GB/s InfiniBand needs a backward pass longer than about 5.5 s
(exercise 4 of chapter 4). With a few hundred thousand tokens per step it usually is, which is
why DP across nodes works and the inner axes stay inside a node.

## 5.2 Tensor parallelism

Megatron-style TP splits each attention block by heads and each MLP by its hidden width *F*.
The first matrix of each block is split by columns, the second by rows, so each block needs
**one all-reduce of the activations** (*T* × *D* × 2 bytes) at its end, two per layer.

Compare the time of a layer's matmuls on one of *N* GPUs with the time of its two
all-reduces. Per layer, the projections do

```
W_layer = 2·T·D·(3F + 2D + 2D_kv)            (MLP gate+up+down, Q and O, K and V)
        = 2·T·D·104448                        for Llama-3-70B
```

FLOPs, so on *N* GPUs at peak *π* the compute time is `W_layer / (N·π)`. The two all-reduces
take `2 · 2TD · 2(N−1)/N / busbw`. The ratio does not depend on *T*:

```
compute / communication  =  104448 · busbw / (4 · π · (N−1))
                         =  26112 · busbw / (π · (N−1))
```

For a B200 (*π* = 2.25 PFLOP/s, preset) on its own NVLink:

| busbw | *N* = 2 | 4 | 8 | 16 | 72 |
|---|---|---|---|---|---|
| ring, 755 GB/s | 8.8 | 2.9 | 1.25 | 0.58 | 0.12 |
| NVLS, 1295 GB/s | 15 | 5.0 | 2.1 | 1.0 | 0.21 |
| InfiniBand, 42 GB/s | 0.49 | | | | |

A ratio above 1 means compute dominates. **TP8 on an HGX B200 spends between a third and half
as long communicating as computing**, unless the all-reduces overlap with compute. TP beyond
8 to 16 GPUs is communication bound even on NVLink, and TP over InfiniBand never pays.

Since π grows faster than busbw from one generation to the next, the critical TP degree
shrinks: on H100 (989 TFLOP/s, 450 GB/s, ring 338 GB/s busbw) the ring ratio at *N* = 8 is
26112 × 338 / (989 × 7) ≈ 1.27, about the same as on B200, because NVLink5 doubled bandwidth
as compute grew 2.3×.

**The simulator's version.** Llama-3-70B prefill, 4096 tokens, the first 2 layers, on HGX B200:

```bash
for tp in 1 2 4 8; do rackscale prefill llama3-70b --system hgx_b200_8 --tp $tp --seqlen 4096 --layers 2 --fast; done
```

| TP | step | speedup | MFU | per GPU: GEMMs | all-reduces (4 × 64 MB, NVLS) | GEMM / all-reduce |
|---|---|---|---|---|---|---|
| 1 | 11403 µs | 1.00× | 56.8% | 10371 µs | 0 | |
| 2 | 6349 µs | 1.80× | 51.0% | 5327 µs | 366 µs | 14.6 |
| 4 | 3606 µs | 3.16× | 44.9% | 2814 µs | 366 µs | 7.7 |
| 8 | 2217 µs | 5.14× | 36.5% | 1539 µs | 366 µs | 4.2 |

Three things show up that the one-line ratio hides:

- **The all-reduce time does not shrink with *N*.** With NVLS each GPU sends its 64 MB to the
  switch and reads 64 MB back, about 91 µs per all-reduce at 1285 GB/s busbw, whatever *N* is.
  Compute shrinks as 1/*N*, so the communication share grows from 6% to 17% of the step.
- **The GEMMs get less efficient as they get thinner.** At TP8 the K/V projection is
  4096 × 1280 × 8192 per GPU and runs at 860 TFLOP/s against 1.2 to 1.3 PFLOP/s for the big
  ones. The measured ratio (4.2 at TP8) is twice the peak-*π* estimate (2.1) because the GEMMs
  run at about half of peak, which is the right *π* to use.
- **Nothing overlaps.** This lowering runs each all-reduce on the compute stream after its
  GEMM, as plain Megatron TP does. Sequence parallelism (reduce-scatter + all-gather around the
  norms) and collective-matmul overlap exist to hide exactly this 17%.

## 5.3 Pipeline parallelism and its bubbles

PP gives each GPU a contiguous slice of the layers. A micro-batch's activations flow forward
through the stages, and its gradients flow back. Between stages only one activation tensor
moves (*T* × *D* × 2 bytes: 67 MB for 4096 tokens of Llama-3-70B), and only between
neighbours, so PP tolerates slow links. The price is the **bubble**: at the start of a step
the later stages have nothing to do until the first micro-batch reaches them, and at the end
the earlier stages wait for the last gradients.

With *p* stages, *m* micro-batches, and per-stage forward and backward times *t*<sub>F</sub>
and *t*<sub>B</sub>, a GPipe or 1F1B schedule takes

```
T_step = (m + p − 1) · (t_F + t_B)
bubble = (p − 1) / (m + p − 1)
```

The fix is more micro-batches: *m* = 4*p* brings the bubble under 20%. But GPipe runs all
forwards first, so stage 0 holds the activations of all *m* micro-batches at once. **1F1B**
(PipeDream-Flush) starts backwards as soon as possible and alternates one forward with one
backward, so a stage never holds more than *p* micro-batches. It has the same bubble as GPipe;
what it saves is memory.

**Worked numbers.** Activations of one transformer layer without recomputation take about
34 · *T* · *D* bytes with FlashAttention (Korthikanti et al., 2022: 34*sbh*, the 5*as*²*b*
score term gone). For Llama-3-70B at 4096 tokens: 34 × 4096 × 8192 ≈ 1.14 GB per layer, so
11.4 GB per micro-batch for a 10-layer stage. With *p* = 8 and *m* = 32, GPipe would hold
32 × 11.4 = 365 GB on stage 0, twice a B200's 180 GB. 1F1B holds at most 8 × 11.4 = 91 GB.

**Simulated.** Llama-3-70B training, PP = 8 on HGX B200 (10 layers per stage), 4096 tokens per
micro-batch. Each stage's forward and backward are simulated at cycle level once (56.9 ms and
111.9 ms), then the schedules run with every activation and gradient transfer simulated on
the NVLink fabric:

```bash
python docs/book/scripts/pipeline_bubbles.py
```

| *m* | GPipe step | 1F1B step | ideal (*m* · 168.8 ms) | simulated bubble | (*p*−1)/(*m*+*p*−1) |
|---|---|---|---|---|---|
| 1 | 1352 ms | 1352 ms | 169 ms | 87.5% | 87.5% |
| 4 | 1858 | 1858 | 675 | 63.7% | 63.6% |
| 8 | 2533 | 2534 | 1350 | 46.7% | 46.7% |
| 16 | 3883 | 3886 | 2701 | 30.5% | 30.4% |
| 32 | 6584 | 6589 | 5401 | 18.0% | 17.9% |

The simulated bubble matches the formula to 0.1 point. The 67 MB hops take about 90 µs each
on NVLink5 and hide behind 57 to 112 ms of compute per stage. The formula is exact here
because the stage times are uniform; the simulator earns its keep when they are not (an
embedding or LM head on the first and last stage, MoE layers with uneven routing, or
stages split across InfiniBand).

**Interleaving** (virtual stages, Narayanan et al. 2021) gives each GPU *v* non-contiguous
slices and cuts the bubble to about (*p*−1)/(*v*·*m*), at *v*× the point-to-point traffic. Zero
bubble schedules (Qi et al. 2023) split the backward into input-gradient and weight-gradient
halves and fill the bubble with the latter. Both are schedules over the same ops, and
[`pipeline_bubbles.py`](scripts/pipeline_bubbles.py) is a 100-line template for writing them.

## 5.4 Expert parallelism

An MoE layer replaces the MLP with *E* experts, of which each token uses *k*. EP puts *E*/*N*
experts on each GPU and keeps attention data parallel (the DeepSeek-V3 inference layout). Each
MoE layer then needs two all-to-alls: **dispatch** (each token's activation to the GPUs holding
its *k* experts) and **combine** (the results back). Each GPU sends `T·k·D·b` bytes per
all-to-all, with *b* bytes per element.

For DeepSeek-V3 (*D* = 7168, *k* = 8 of 256 routed experts) at 64 tokens per GPU on NVL72,
that is 64 × 8 × 7168 × 2 B ≈ 7.3 MB per GPU, about 10 µs at the 764 GB/s busbw of chapter 4.
The simulated DeepSeek-V3 decode step on NVL72 (chapter 6) shows exactly that: about 10 µs of
all-to-all per MoE layer at 509 to 722 GB/s.

The compute side is the grouped GEMM over each GPU's experts. With balanced routing, each GPU
holds 3 or 4 of the 256 experts (the lowering uses 3) and each expert sees about
64 × 8 / 3 ≈ 170 tokens, below the B300's critical batch of 361 (chapter 1): **EP decode GEMMs are weight-bandwidth bound**, just like dense decode,
and spreading experts over more GPUs makes each expert's batch smaller still.

Over InfiniBand the same dispatch takes about 175 µs, longer than the expert compute, which is
why DeepSeek overlaps one micro-batch's all-to-all with the other's compute. Chapter 6
simulates that overlap on NVL72, where it does not pay.

## 5.5 Context parallelism: ring attention

For long sequences attention dominates (its FLOPs grow as *S*²). CP gives each of *N* GPUs
*S*/*N* tokens. Each GPU computes attention of its queries against its own K/V chunk while it
passes that chunk to its neighbour and receives the next one, so after *N* steps every query
has seen every key. Each step moves `2 · (S/N) · H_kv · d · 2` bytes (K and V) and computes a
chunk-against-chunk attention of `4 · (S/N)² · H · d` FLOPs. The ratio grows with the chunk
size, so **CP is compute bound when chunks are long enough**:

```
compute / transfer = 4·c²·H·d/π  ÷  4·c·H_kv·d/W  =  c · (H / H_kv) · W / π
```

with *c* = *S*/*N* tokens per chunk. GQA (*H*/*H*<sub>kv</sub> = 8) helps by 8×.

The FlashAttention study ran this with the real FA4 kernel (B300) and FA2 kernel (A100),
recorded with NVBit and validated within 2% of Nsight Compute at chunk sizes of 2048 and up,
and the K/V transfers simulated on the fabric (Llama-3-70B attention, 64 query heads):

| system | kernel | chunk | KV heads | context | step | ring efficiency | TFLOP/s per GPU |
|---|---|---|---|---|---|---|---|
| GB300 NVL72 | FA4 | 1024 | 8 | 74k | 1.89 ms | 94% | 1311 |
| GB300 NVL72 | FA4 | 2048 | 64 | 147k | 8.44 ms | 99% | 1173 |
| 9 × 8 B300 over IB | FA4 | 1024 | 8 | 74k | 7.17 ms | 42% | 345 |
| 9 × 8 B300 over IB | FA4 | 2048 | 64 | 147k | 112 ms | **9%** | 88 |
| 9 × 8 A100 over IB | FA2 | 2048 | 8 | 147k | 52.3 ms | 100% | 189 |
| 9 × 8 A100 over IB | FA2 | 2048 | 64 | 147k | 225 ms | 23% | 44 |

Ring efficiency is the share of the step spent in the attention kernel. On NVL72 the transfer
hides behind the kernel at every shape. Over InfiniBand, FA4 on B300 is fast enough that the
ring becomes transfer bound even with GQA at 1k chunks; the slower FA2 on A100 still hides its
transfer when K/V is small. **A faster kernel needs a faster fabric to stay compute bound.**
(The step kernels themselves are validated against Nsight Compute within −3.7 to +5.4% for
FA4 and −4.6 to −5.9% for FA2, chapter 3.5. The fabrics are presets.)

## 5.6 Putting the axes together

A rule of thumb that falls out of the ratios above, for current NVIDIA systems:

1. Use **TP or EP inside the NVLink domain** (8 GPUs on HGX, up to 72 on NVL72). Keep TP at
   8 or below for dense layers; EP can span the whole domain because its all-to-all bytes per
   GPU do not grow with *N*.
2. Use **PP across NVLink domains** when the model does not fit in one, with *m* ≥ 4*p*
   micro-batches.
3. Use **DP (with ZeRO/FSDP sharding) outermost**, overlapped with the backward pass.
4. Add **CP** when the sequence is long enough that attention dominates, and keep the ring
   inside the NVLink domain if the kernel is fast.

The scaling book reaches a similar ordering for TPUs from the same kind of arithmetic. What
differs on GPUs is the size of the fast domain (8 or 72 GPUs rather than a 2D or 3D torus) and
the 18× drop to InfiniBand outside it.

## Exercises

1. With the ratio formula of 5.2, at what TP degree does Llama-3-8B (*D* = 4096, *F* = 14336,
   *D*<sub>kv</sub> = 1024) become communication bound on HGX B200 with NVLS?
2. A PP = 4 pipeline has stage times of 10 ms, 10 ms, 10 ms and 14 ms (the last holds the LM
   head). With *m* = 16, what is the step time, and what is the bubble relative to the ideal
   of 16 × 44 ms / 4?
3. How large must the CP chunk be for FA4 on B300 (1.5 PFLOP/s sustained, chapter 3) to hide
   its K/V transfer over 50 GB/s InfiniBand with 8 KV heads and 64 query heads?
4. Why does EP's all-to-all not get more expensive per GPU as *N* grows, while TP's all-reduce
   latency does?

<details><summary>Answers</summary>

1. Per layer `2TD(3F + 2D + 2D_kv) = 2TD · 53248`, so the ratio is 13312 · busbw / (π(*N*−1)).
   With 1295 GB/s and 2.25 PFLOP/s it equals 1 at *N* − 1 ≈ 7.7: about **TP8**, half the
   70B model's limit, because a narrower model does less compute per byte of activation.
2. The slowest stage sets the pace: (16 + 3) × 14 ms ≈ **266 ms** if every stage waits for the
   slowest (a simple bound; the simulator gives the exact number). The ideal is 176 ms, so the
   bubble is about 34%, against 16% for a balanced pipeline.
3. Compute/transfer = *c* · 8 · 50 GB/s / 1.5 PFLOP/s ≥ 1 gives *c* ≥ **3750 tokens** per
   chunk. The study's 2048-token chunks are transfer bound over IB, as the table shows.
4. Each GPU sends `T·k·D·b` bytes no matter how many peers it splits them over, and a direct
   all-to-all has no chain of dependent steps. A ring all-reduce has 2(*N*−1) dependent steps.

</details>

## Reproduce

| number | command |
|---|---|
| TP sweep | `rackscale prefill llama3-70b --system hgx_b200_8 --tp N --seqlen 4096 --layers 2 --fast` |
| pipeline bubbles | `python docs/book/scripts/pipeline_bubbles.py` |
| EP all-to-all in a real step | `python examples/deepseek_v3_nvl72.py --layers 4` |
| ring attention | `research/flash_attention/results/scale_rows.json`; rerun with `python research/flash_attention/run_study.py scale` (recordings in `research/flash_attention/rpath/`) |

Next: [6. Inference](06-inference.md).
