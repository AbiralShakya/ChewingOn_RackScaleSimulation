# 11. Every parallelism, in closed form, checked cycle by cycle

Chapter 5 compared one communication time with one compute time per axis. This chapter does
the whole job the way *How to Scale Your Model* does it for TPUs: one formula per axis for
compute, memory and communication, a rule for which of them overlap, and the critical
quantities that fall out (the batch at which decode stops being weight-bound, the tokens per
GPU at which data parallelism becomes free, the TP degree at which all-reduces cost as much as
matmuls). Then it checks every formula against the simulator, case by case, with signed
errors, and says where and why reality departs.

Everything here is in [`rackscale.scaling`](../../python/rackscale/scaling.py), callable from
Python, from the `scaling_math` and `check_scaling_math` agent tools, and reproducible with
[`research/parallelism/check_math.py`](../../research/parallelism/check_math.py).

**The rack is a GB300 NVL72** (72 calibrated B300 dies on an NVLink 5 switch fabric preset).
All multi-GPU numbers are simulated; no number in this chapter was measured on more than one
GPU.

## 11.1 The numbers that matter

Per GPU, from the same presets the simulator runs:

| symbol | meaning | GB300 |
|---|---|---|
| *C* | dense BF16 tensor FLOP/s at the 2032 MHz preset clock | 2.46 × 10<sup>15</sup> |
| *β* | achievable HBM bandwidth | 6.82 × 10<sup>12</sup> B/s |
| *W* | NVLink bandwidth, one direction, all 18 links | 9.0 × 10<sup>11</sup> B/s |
| *α* | fixed cost of one collective (launch, handshake) | 5 µs (assumed) |
| | HBM | 288 GB |

Three ratios follow, and most of this chapter is about them:

* *C* / *β* = **361 FLOP per byte**: a kernel below this arithmetic intensity is HBM bound.
  For BF16 weights (2 bytes each, 2 FLOP per weight per token) a matmul needs **361 tokens**
  to be compute bound. That is the decode critical batch.
* *C* / *W* = **2738 FLOP per byte of NVLink**: a GPU must do 2738 FLOP for each byte it sends
  to hide the send. That sets the data-parallel critical batch.
* *W* / *β* ≈ 0.13: NVLink is an eighth of HBM. Anything that crosses the fabric once per
  weight use is 8× costlier than reading it locally.

## 11.2 The collectives

For a buffer of *B* bytes per rank over *n* ranks (nccl-tests convention: the input of an
all-reduce or reduce-scatter, the output of an all-gather):

| collective | ring / direct | NVLS (switch reduction) |
|---|---|---|
| all-reduce | *α* + 2 (*n*−1)/*n* · *B*/*W* | *α* + *B*/*W* |
| reduce-scatter, all-gather | *α* + (*n*−1)/*n* · *B*/*W* | same |
| all-to-all | *α* + (*n*−1)/*n* · *B*/*W* | |
| point to point | *α* + *B*/*W* | |

NVLS halves the all-reduce at large *n*: each GPU writes its buffer into the switch once and
reads the reduced result once, on the two directions of its links at the same time. An
all-gather or reduce-scatter cannot gain that way, which matters for sequence parallelism
below. Point to point gets the full *W* on NVL72 because every GPU has one link to every switch
chip, so any pair can use all 18.

## 11.3 One formula per axis

*P* is the parameters held by one GPU before data-parallel sharding (*P*<sub>total</sub> /
(*tp* · *pp*), experts further / *ep*), *T* the tokens one GPU's layers see per micro-batch,
*m* micro-batches per step, *L* layers (per stage under PP), *D* the width, *b* bytes per
weight.

**Compute and memory (every plan).** A training step does 6 *N*<sub>active</sub> tokens FLOPs
(plus 12 *L H d S*/2 per token for causal attention), split evenly over the GPUs; inference
does a third of that. Weights are read once per pass: 3*m* *P b* per training step, *P b*
per decode step, plus the KV cache, *B K* · bytes per token, in decode. The roofline term
is max(FLOPs / *C*, bytes / *β*).

**Tensor parallelism (Megatron).** Two all-reduces of the activations (*T* × *D* × 2 bytes) per
layer in the forward pass and two in the backward, on the compute stream:

```
T_TP = 4 L m · AR(T D 2)            (training)       2 L · AR(T D 2)   (inference)
```

With NVLS, AR = *α* + 2*TD*/*W*. The layer's matmuls take 2 *T P*<sub>layer</sub> / (*n C*), so
the ratio of matmul to all-reduce time is *P*<sub>layer</sub> *W* / (2 *n D C*), independent of
*T*: **TP pays until *n* ≈ *P*<sub>layer</sub> *W* / (2 *D C*)**, which is 19 for Llama-3-70B
on GB300 and 9.5 for Llama-3-8B.

**Sequence parallelism** replaces each all-reduce with a reduce-scatter and an all-gather of the
same *B*, and runs norms and residuals on *T*/*n* tokens. With a ring all-reduce that is the same
traffic; with NVLS it is twice the time, because NVLS made the all-reduce cost *B*/*W* and the
pair costs 2 (*n*−1)/*n* · *B*/*W*. SP buys activation memory, not speed, on an NVLS fabric.

**Data parallelism and its sharded forms.** BF16 gradients are 2*P* bytes. Over *n* = *dp* ranks:

| stage | what moves per step | when | memory per GPU (weights / grads / Adam) |
|---|---|---|---|
| DDP (`zero=0`) | AR(2*P*) | during backward | 2*P* / 2*P* / 12*P* |
| ZeRO-1, -2 | RS(2*P*), then AG(2*P*) after the optimizer | RS in backward, AG exposed | 2*P* / 2*P* or 2*P*/*n* / 12*P*/*n* |
| FSDP (`zero=3`) | AG(2*P*) before forward, AG(2*P*) before backward, RS(2*P*) | all overlappable but the first layer's gather | 2*P*/*n* + 2 layers / 2*P*/*n* / 12*P*/*n* |
| HSDP (`shard=s`) | as FSDP inside groups of *s*, then AR(2*P*/*s*) across the *n*/*s* groups | | as FSDP with *s* for *n* |

FSDP moves 1.5× DDP's bytes (three passes of 2*P* instead of an all-reduce's two), and ZeRO-1/2
the same as DDP. The backward pass does 4 *P T* FLOPs in 4 *P T*/*C*; it hides 2*P* bytes when
4*PT*/*C* ≥ 2*P*/*W*:

```
DP traffic is free when tokens per GPU per step  T ≥ C / 2W  (ZeRO-1/2, or DDP on NVLS)  = 1369
                                                  T ≥ C / W   (FSDP, DDP on a ring)        = 2738
```

Across InfiniBand (*W* = 50 GB/s) the same bounds are 25k and 49k tokens per GPU, which is why
the data-parallel axis goes outermost.

**Pipeline parallelism.** With per-micro-batch stage time τ (compute plus the TP, EP and CP
communication inside the stage):

| schedule | step | bubble |
|---|---|---|
| GPipe, 1F1B | (*m* + *p* − 1) τ | (*p*−1)/(*m*+*p*−1) |
| interleaved, *v* chunks | (*m* + (*p*−1)/*v*) τ | (*p*−1)/(*v m* + *p*−1) |
| ZB-H1 | *m* τ + (*p*−1)(*t*<sub>F</sub> + *t*<sub>B</sub> − *t*<sub>W</sub>) | ≈ a third of 1F1B's when *t*<sub>F</sub> = *t*<sub>B</sub> = *t*<sub>W</sub> |

plus 2(*p*−1) point-to-point hops of *T D* 2 / *tp* bytes on the critical path at fill and drain.

**Expert parallelism.** Dispatch and combine per MoE layer per pass, each A2A(*T k D* 2):
4 *L*<sub>moe</sub> *m* all-to-alls per training step, 2 *L*<sub>moe</sub> per decode step.
The bytes per GPU do not grow with *ep*.

**Context parallelism.** Ring attention moves a K/V chunk of *c* · 2 *H*<sub>kv</sub> *d* · 2
bytes (*c* = *S*/*cp*) per step while computing *c* queries against *c*/2 keys (zig-zag):

```
exposed per step = max(0, 4 c H_kv d / W  −  2 c² H d / C)   ⇒   hidden when  c ≥ 2 (H_kv / H) · C / W
```

For Llama-3-70B (*H*<sub>kv</sub>/*H* = 1/8) that is *c* ≥ 684 tokens per GPU on NVLink and
12.3k over InfiniBand. The backward pass sends K/V and returns dK/dV against 2.5× the compute,
so it hides at about the same *c*.

Ulysses instead does two all-to-alls per attention layer per pass (Q, K, V in; output back),
(*c* (*H* + 2*H*<sub>kv</sub>) *d* + *c H d*) · 2 bytes, on the critical path, and needs *cp* to
divide the heads.

**Inference, request level.** A plan of *w* GPUs serves `replicas = 72 / w` independent copies.
Per-user speed is 1 / step; rack throughput is replicas × tokens per step / step. Nothing
crosses replicas, so request-level parallelism is the one axis that is free by construction.

## 11.4 Worked: Llama-3-70B and 405B on one NVL72

`python -c "import rackscale.scaling as sc; print(sc.explain(sc.estimate('llama3-70b', sc.hardware('gb300'), 'train', sc.Plan(tp=8, dp=9, zero=3), micro_batch=1, seqlen=8192)))"`

Training, 8192-token sequences (131072 for the CP rows), one sequence per micro-batch per
DP replica. "Exposed" is the share of the step in terms nothing hides (TP all-reduces, the
pipeline bubble, the post-optimizer gather, Adam). Estimates, not simulations:

| model | plan (72 GPUs or fewer) | step ms | bubble / exposed | MFU | tokens/s/GPU | memory GB | largest non-matmul terms |
|---|---|---|---|---|---|---|---|
| llama3-70b | dp72 fsdp | 1479 | 0% | 100% | 5,540 | 202 | FSDP parameter all-gathers (fwd + bwd) 309.2 ms (overlapped); FSDP gradient reduce-scatter 154.6 ms (overlapped) |
| llama3-70b | tp8 dp9 zero1 | 255 | 28% | 72% | 4,014 | 70 | TP activations 49.3 ms (exposed); ZeRO gradient reduce-scatter 17.4 ms (overlapped) |
| llama3-70b | tp8 dp9 fsdp | 238 | 23% | 77% | 4,304 | 39 | TP activations 49.3 ms (exposed); FSDP parameter all-gathers (fwd + bwd) 34.9 ms (overlapped) |
| llama3-70b | tp8 dp9 sp fsdp | 275 | 33% | 67% | 3,719 | 39 | TP activations 86.7 ms (exposed); FSDP parameter all-gathers (fwd + bwd) 34.9 ms (overlapped) |
| llama3-70b | tp4 dp18 fsdp | 422 | 13% | 87% | 4,851 | 62 | FSDP parameter all-gathers (fwd + bwd) 74.0 ms (overlapped); TP activations 49.3 ms (exposed) |
| llama3-70b | tp8 pp9 1f1b | 1142 | 36% | 64% | 3,587 | 41 | pipeline bubble 206.8 ms (exposed); TP activations 194.2 ms (exposed) |
| llama3-70b | tp8 pp9 zb | 1004 | 27% | 73% | 4,080 | 41 | TP activations 194.2 ms (exposed); pipeline bubble 68.9 ms (exposed) |
| llama3-70b | tp8 pp9 interleavedv2 | 1038 | 29% | 71% | 3,944 | 41 | TP activations 194.2 ms (exposed); pipeline bubble 103.4 ms (exposed) |
| llama3-70b | tp8 cp9 ring S=128k | 809 | 15% | 85% | 2,249 | 182 | TP activations 86.4 ms (exposed); Adam update 38.8 ms (exposed) |
| llama3-70b | tp8 cp9 ulysses S=128k | 822 | 17% | 83% | 2,216 | 182 | TP activations 86.4 ms (exposed); Adam update 38.8 ms (exposed) |
| llama3-405b | tp8 dp9 fsdp | 1222 | 15% | 85% | 838 | 164 | FSDP parameter all-gathers (fwd + bwd) 200.4 ms (overlapped); TP activations 152.8 ms (exposed) |
| llama3-405b | tp8 pp9 1f1b | 5875 | 29% | 71% | 697 | 165 | pipeline bubble 1063.5 ms (exposed); TP activations 611.4 ms (exposed) |
| llama3-405b | tp8 pp9 zb | 5166 | 19% | 81% | 793 | 165 | TP activations 611.4 ms (exposed); pipeline bubble 354.5 ms (exposed) |

What the arithmetic says, before any simulation:

- **Pure FSDP over 72 GPUs wins on paper** (100% MFU, 5540 tokens/s/GPU) because 8192 tokens per
  GPU is 3× the FSDP critical batch of 2738: every gather and scatter hides behind compute. It
  needs 202 GB per GPU, almost all activations, and it is the plan most exposed to collective
  efficiency, which 11.5 shows is 30% worse than the closed form.
- **TP8 costs 23% of the step in exposed all-reduces** (49 ms of 238), and sequence parallelism
  makes it worse on NVLS (87 ms), as 11.3 predicts. TP4 x FSDP18 exposes 13%.
- **Pipelines lose 27-36% of the step** at *m* = 4*p*; ZB-H1 recovers two thirds of the 1F1B
  bubble, interleaving (*v* = 2) half.
- **Ring and Ulysses CP are within 2% of each other** at 128k: ring transfers hide (14.5k tokens
  per GPU against the 684 needed) and Ulysses' all-to-alls cost 1.5% of the step.

Decode, 8192-token context (4096 for DeepSeek-V3), BF16 weights and KV cache:

| model | plan | batch per replica | step µs | tokens/s per user | tokens/s per GPU | rack tokens/s | dominant | critical batch |
|---|---|---|---|---|---|---|---|---|
| llama3-70b | 1 GPU | 8 | 23,846 | 42 | 335 | 24,155 | weight reads | 361 |
| llama3-70b | tp2 | 64 | 23,934 | 42 | 1,337 | 96,267 | KV cache | 361 |
| llama3-70b | tp4 | 64 | 12,460 | 80 | 1,284 | 92,456 | KV cache | 361 |
| llama3-70b | tp8 | 64 | 6,723 | 149 | 1,190 | 85,674 | KV cache | 361 |
| llama3-70b | tp8 | 256 | 16,732 | 60 | 1,913 | 137,703 | KV cache | 361 |
| llama3-8b | 1 GPU | 64 | 12,435 | 80 | 5,147 | 370,573 | KV cache | 361 |
| llama3-8b | 1 GPU | 512 | 82,989 | 12 | 6,169 | 444,201 | KV cache | 361 | (no fit)
| deepseek-v3 | ep72 | 64 per GPU | 11,900 | 84 | 5,378 | 387,225 | weight reads | 361 |
| deepseek-v3 | ep72 | 256 per GPU | 22,804 | 44 | 11,226 | 808,262 | KV cache | 361 |

At batch 64, Llama-3-70B is KV-cache bound at every TP degree (its 64 x 8192 x 320 KB of cache
outweighs the 141 GB of weights once split), so TP buys per-user speed almost linearly (42 to
149 tokens/s) at a 12% loss of tokens per GPU. The decode critical batch of 361 is per GPU for a
dense model; an MoE expert sees *B k* / *E* of it, which is why DeepSeek-V3 at 64 sequences per
GPU stays weight bound and quadruples its batch for 2× the throughput.

## 11.5 The check: formulas vs replay vs cycles

The closed forms leave things out on purpose. To know what they leave out, each case below is
computed three ways on the same model cut to two layers (four for DeepSeek-V3, so it has MoE
layers):

1. **textbook**: the formulas above;
2. **roofline replay**: the *exact program the simulator runs* (every kernel, collective, event
   and stream) on an ideal machine: each kernel at max(FLOPs/*C*, bytes/*β*), each collective at
   its closed form, streams overlapping perfectly;
3. **cycle level**: the simulator, fast mode (every distinct kernel at cycle level, every
   collective on the NVLink fabric with the NCCL algorithm and protocol model).

Errors are signed, (sim − *x*)/*x*. The gap between replay and the simulator is split into
**kernel efficiency** (the kernels' cycle-level time at the preset clock minus their roofline),
**power clock** (the stretch from the power-capped clock, chapter 1), **collectives** (simulated
minus closed-form time inside collectives, hidden or not) and **overlap + idle** (the rest:
communication that did not hide, launch gaps, dependency waits; negative when slow collectives
stayed hidden behind compute).

Llama-3-8B training, 1 sequence of 4096 tokens per micro-batch; decode at 8192 context
(DeepSeek-V3 at 4096). Every number below is simulated on GB300 presets; none is measured on
more than one GPU.

| case | GPUs | textbook µs | replay µs | sim µs | sim vs replay | sim vs textbook | kernel efficiency µs | power clock µs | collectives µs | overlap + idle µs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 GPU | 1 | 4685 | 5116 | 9141 | +78.7% | +95.1% | +1378 | +2620 | +0 | +28 |
| DDP | 8 | 4871 | 5606 | 9980 | +78.0% | +104.9% | +1415 | +2615 | +675 | -331 |
| ZeRO-2 | 8 | 4685 | 5546 | 9702 | +74.9% | +107.1% | +1372 | +2619 | +277 | -112 |
| ZeRO-2 + optimizer | 8 | 8399 | 6559 | 11009 | +67.8% | +31.1% | +1414 | +2615 | +543 | -122 |
| FSDP | 8 | 7243 | 5975 | 10377 | +73.7% | +43.3% | +1488 | +2605 | +859 | -550 |
| HSDP 4x2 | 8 | 6401 | 5980 | 10423 | +74.3% | +62.8% | +1448 | +2609 | +926 | -540 |
| TP8 | 8 | 924 | 1174 | 2056 | +75.1% | +122.6% | +355 | +101 | +397 | +29 |
| TP8 + SP | 8 | 1188 | 1266 | 2493 | +96.9% | +109.9% | +405 | +101 | +691 | +29 |
| TP2 x DP4 | 8 | 2681 | 3196 | 5672 | +77.5% | +111.6% | +763 | +1116 | +606 | -8 |
| TP2 x FSDP4 (HSDP 2) | 8 | 3099 | 3387 | 6251 | +84.5% | +101.7% | +769 | +1109 | +1336 | -350 |
| PP2 GPipe m8 | 2 | 21164 | 23105 | 41490 | +79.6% | +96.0% | +5470 | +10384 | +2805 | -274 |
| PP2 1F1B m8 | 2 | 21164 | 23346 | 42676 | +82.8% | +101.6% | +5456 | +10339 | +2805 | +729 |
| PP2 ZB-H1 m8 | 2 | 19602 | 22377 | 40024 | +78.9% | +104.2% | +5456 | +10340 | +2805 | -954 |
| TP2 x PP2 x DP2 1F1B m4 | 8 | 6745 | 7735 | 14325 | +85.2% | +112.4% | +1545 | +2236 | +1911 | +899 |
| prefill TP4 | 4 | 1155 | 1415 | 2426 | +71.5% | +110.1% | +303 | +386 | +311 | +12 |
| prefill TP4 + SP | 4 | 1155 | 1436 | 2443 | +70.1% | +111.5% | +333 | +386 | +276 | +12 |
| prefill CP8 ring | 8 | 6470 | 7727 | 13690 | +77.2% | +111.6% | +4060 | +1877 | +4916 | -4889 |
| prefill CP8 Ulysses | 8 | 6816 | 7231 | 12749 | +76.3% | +87.0% | +3383 | +1874 | +250 | +11 |
| decode 1 GPU b64 | 1 | 1066 | 936 | 1087 | +16.1% | +2.0% | +137 | +0 | +0 | +13 |
| decode TP8 b64 (70B) | 8 | 243 | 225 | 359 | +59.9% | +47.7% | +60 | +0 | +61 | +14 |
| decode MoE TP4 (Qwen3-30B-A3B) | 4 | 237 | 234 | 302 | +29.1% | +27.5% | +23 | +0 | +29 | +16 |
| decode EP8 (DeepSeek-V3) | 8 | 1740 | 1510 | 1671 | +10.6% | -4.0% | +100 | +1 | +30 | +30 |

Selected collectives (all of them are in [`results/check.md`](../../research/parallelism/results/check.md)):

| case | kind | algo | size | ranks | n | closed form µs | sim µs | error |
|---|---|---|---|---|---|---|---|---|
| DDP | all_reduce | Nvls | 416.0 MiB | 8 | 2 | 489.7 | 827.3 | +68.9% |
| FSDP | all_gather | Ring | 416.0 MiB | 8 | 4 | 429.1 | 570.8 | +33.0% |
| HSDP 4x2 | all_reduce | Nvls | 104.0 MiB | 2 | 8 | 126.2 | 276.1 | +118.8% |
| TP8 | all_reduce | Nvls | 32.0 MiB | 8 | 8 | 42.3 | 91.9 | +117.4% |
| TP8 + SP | all_gather | Ring | 32.0 MiB | 8 | 8 | 37.6 | 80.0 | +112.7% |
| PP2 1F1B m8 | send_recv | Direct | 32.0 MiB | 2 | 16 | 40.3 | 215.6 | +435.3% |
| prefill CP8 Ulysses | all_to_all | Direct | 96.0 MiB | 8 | 2 | 102.9 | 170.2 | +65.4% |
| decode TP8 b64 (70B) | all_reduce | Ring | 1.0 MiB | 8 | 4 | 6.2 | 21.4 | +247.3% |
| decode EP8 (DeepSeek-V3) | all_to_all | Direct | 7.0 MiB | 8 | 1 | 12.1 | 28.6 | +135.4% |

How to read it:

- **The roofline replay is optimistic by 68-97% for training and 11-60% for decode.** No case
  comes in under it, which is the sanity check: the replay is a lower bound on the program the
  simulator runs.
- **The textbook is closer to the replay than either is to the simulator** (-21% to +28%). Where it
  is above the replay, it double-counts what the program overlaps (the ZeRO gather and the
  optimizer); where it is below, it leaves out norms, residuals and the exact causal FLOPs.

## 11.6 Where reality departs, and why cycle level is needed

Five effects make up the gap, and none is in a closed form:

1. **The roof moves (power-capped clock): about two thirds of the training gap.** The B300's
   measured BF16 clock falls from 2032 MHz to about 1455 MHz under its power cap (chapter 1).
   Training-sized GEMMs run long enough to be throttled; decode kernels do not. In the 1-GPU
   case the stretch is 2620 µs of a 4025 µs gap. A roofline at the datasheet clock is 1.4×
   optimistic before any kernel is inefficient, and the power each GPU draws depends on what
   it runs, so the clock is a property of the *schedule*, not the chip.
2. **Kernels at 46-81% of peak: the other third.** At the preset clock the 4096 × 28672 × 4096
   GEMM reaches 1988 TFLOP/s (81%) and causal FlashAttention at 4096 tokens 1131 (46%):
   prologues, epilogues, the last partial wave, softmax on the MUFU pipe. Thin GEMMs under TP
   lose more (TP8 adds 355 µs on 836).
3. **Collectives at 25-75% of the link.** NVLS all-reduces of 32 MiB reach 42% of *B*/*W*
   (92 µs against 42), ring reduce-scatters and gathers about 75%, small decode all-reduces
   (1 MiB) 29%, where launch, fences and protocol steps dominate. Two-rank NVLS all-reduces are
   the worst case (+143% to +206%): NCCL's tuner picks NVLS for them, and the switch round trip
   costs more than the 1-hop ring would. Sequence parallelism turns a 92 µs NVLS all-reduce into
   two 80 µs ring collectives: TP8 + SP is 21% slower than TP8 here, which the formulas predicted
   in direction and the simulator in size.
4. **Communication takes SMs.** Point-to-point transfers get 2 SMs each in this lowering (the
   NCCL p2p default), and an SM channel is a rate-limited copy engine, so a 32 MiB pipeline hop
   moves at 155 GB/s, not 900 (+435%). In a pipeline it sits on the fill and drain path; in ring
   attention it stayed hidden (the -4889 µs of overlap), because each step's attention took
   longer. Every collective also takes 20 SMs from the GEMMs it overlaps with, which a roofline
   with "perfect overlap" cannot see.
5. **Overlap is a schedule, not a max().** The overlap column is near zero or negative when
   slow collectives stayed hidden (FSDP: -550 µs, they were 859 µs slower and still mostly
   hidden), positive when a dependency exposed them (1F1B's +729 µs: the activation gradient
   leaves after the weight-gradient GEMMs, so the upstream stage waits; ZB-H1 sends it first and
   gets -954 µs).

So: the formulas rank the plans and give their scaling laws; the replay says what a perfect
implementation of the exact program would do; the cycle-level simulator says what the clock,
the kernels, NCCL and the schedule actually leave. On these cases the rankings agree, and the
absolute step differs by up to 2×.

## Reproduce

| what | command |
|---|---|
| one estimate | `rackscale.scaling.estimate(model, hardware("gb300"), phase, Plan(...))` |
| one check | `rackscale.scaling.check(model, "gb300", phase, Plan(...), layers=2, ...)` |
| the table in 11.5 | `python research/parallelism/check_math.py` (about 40 minutes on 4 cores) |
| from an agent | MCP tools `scaling_math`, `check_scaling_math` |
