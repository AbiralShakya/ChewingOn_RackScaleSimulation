# 13. Tokens or bytes: H-Net on one Blackwell GPU

Every model in this book so far reads BPE tokens. A tokenizer is a fixed compression step:
Llama-3's turns English web text into one token per 4.45 bytes, and the model never sees the
bytes. **H-Net** (Hwang, Wang and Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence
Modeling*, 2025; Cartesia's "From tokens to bytes") drops the tokenizer: it reads raw UTF-8
bytes and learns where to cut them into chunks. The paper matches a BPE Transformer's quality
at the same FLOPs per byte. This chapter asks the systems question: at the same FLOPs per
byte, does a byte-level model cost the same per byte on a GPU?

Everything is in [`rackscale.hnet`](../../python/rackscale/hnet.py) and reproducible with
[`research/bytes/run.py`](../../research/bytes/run.py). **All numbers are simulated on the
calibrated B300 (one GPU, fast mode); no H-Net was run on hardware.**

## 13.1 The model

One H-Net stage wraps an inner network:

```
bytes ─► encoder (Mamba-2, width D_s, every position)
      ─► router: p_t = (1 − cos(W_q x_t, W_k x_{t−1})) / 2, a boundary where p_t ≥ 0.5
      ─► downsample: keep the boundary positions (about 1 in R_s)
      ─► inner network (the next stage, or the main Transformer)
      ─► dechunk: EMA-smooth and repeat each chunk over its bytes, add a residual
      ─► decoder (Mamba-2, width D_s) ─► next-byte logits (256)
```

So the main network sees bytes / ∏ *R*ₛ positions. The two published XL shapes, with the two
baselines we compare them with:

| model | layers | widths | bytes per main-network position | parameters |
|---|---|---|---|---|
| H-Net 1-stage XL | M4 + T24 + M4 | 1024 / 2048 | 4.7 | 1.26 B |
| H-Net 2-stage XL | M4 + T1M4 + T27 + M4T1 + M4 | 1024 / 1536 / 2048 | 3.0 × 2.3 = 6.9 | 1.59 B |
| BPE Transformer XL | T24 | 2048 | 4.45 (tokenizer) | 1.20 B |
| byte Transformer XL | T24 | 2048 | 1 | 1.20 B |

M is a Mamba-2 layer, T a Transformer block. The paper does not give head counts or FFN
widths; we use 128-wide heads with full multi-head attention and an 8/3 SwiGLU FFN, and say so
wherever it matters.

## 13.2 Work per byte, in closed form

**FLOPs.** Each stage runs on its share of the positions:

&nbsp;&nbsp;&nbsp;&nbsp;FLOPs per byte ≈ Σₛ (2 *N*ₛ + attention) / ∏ᵢ₍<ₛ₎ *R*ᵢ + (2 *N*_main + attention) / ∏ *R*,

| model | GFLOP per byte (8 KiB) | paper (Table 2) | error |
|---|---|---|---|
| H-Net 1-stage XL | 0.672 | 0.72 | −6.7% |
| H-Net 2-stage XL | 0.670 | 0.69 | −2.8% |
| BPE Transformer XL | 0.696 | 0.69 | +0.8% |
| byte Transformer XL | 3.197 | | |

The three models the paper compares do the same work per byte, within the 7% our shape
assumptions leave. A Transformer that reads raw bytes does 4.6× more.

**Bytes moved in decode.** Decode is memory bound (chapter 6), so what sets its cost is the
bytes each step reads from HBM. Per step, for a batch of *B* sequences with *S* bytes of
context:

| | what one step reads | bytes per sequence per step |
|---|---|---|
| BPE | all weights *W*, plus *B* · (*S*/4.45) KV positions | 4.45 |
| H-Net | the outer weights, every sequence's Mamba state (SSM and conv state, read and written, BF16) and any outer attention KV, plus the main weights *W*_main and the KV of the *B*/*R* sequences whose router fired, over *S*/*R* positions each | 1 |

At *B* = 64 and 8 KiB of context, with 192 KiB of KV per position
(`hnet.decode_traffic(model, 64, 8192)`):

| | GB per step | GB per byte | µs per byte at *β* = 6.82 TB/s |
|---|---|---|---|
| BPE | 2.39 weights + 0.53 LM head + 23.16 KV = 26.08 | 5.86 | 860 |
| H-Net 1-stage | 2.39 main weights + 0.12 outer weights + 4.67 main KV + 0.55 Mamba state = 7.73 | 7.73 | 1134 |
| H-Net 2-stage | 2.69 + 0.49 + 2.44 main KV + 0.72 outer attention KV + 0.83 Mamba state = 7.16 | 7.16 | 1050 |

H-Net reads about the same KV per byte as BPE (the chunks are about as long as tokens), but it
reads the main network's weights on every byte step where BPE reads them once per 4.45 bytes.
In closed form that makes H-Net **1.32×** (1-stage) and **1.22×** (2-stage) BPE's decode cost
per byte at batch 64.

## 13.3 Prefill

Eight prompts of 8 KiB on one B300:

| model | ms per step | MB/s of text | vs BPE |
|---|---|---|---|
| H-Net 1-stage | 48.0 | 1.37 | 1.43× the time |
| H-Net 2-stage | 54.7 | 1.20 | 1.63× |
| BPE | 33.5 | 1.96 | |
| byte Transformer | 175.7 | 0.37 | 5.25× |

Where the time goes, in ms per step, from the simulator's per-op trace:

| model | outer stages | main attention | main matmuls | main norms and elementwise | launch gaps |
|---|---|---|---|---|---|
| H-Net 1-stage | 18.1 | 2.4 | 24.0 | 3.2 | 0.2 |
| H-Net 2-stage | 31.2 | 1.5 | 19.3 | 2.4 | 0.3 |
| BPE | | 2.8 | 27.1 | 3.4 | 0.1 |
| byte Transformer | | 39.6 | 118.9 | 17.1 | 0.1 |

Prefill is compute bound, so FLOPs should predict it, and for the main networks they do: H-Net
1-stage's main network takes 29.7 ms against BPE's 33.3 ms for the whole model, since it sees
1743 positions per prompt to BPE's 1840 tokens. Both run at about half the datasheet peak (49%
and 55%). The difference is the outer stages: 19% of H-Net 1-stage's FLOPs but 38% of its time
(18.1 ms), and in the 2-stage model 39% of the FLOPs and 57% of the time. They run at width
1024 over all 65,536 bytes, where matmuls have a short inner dimension and the Mamba-2 chunked
scan (7.2 ms of the 18.1) runs far from the tensor cores' peak: about 19% of peak for the outer
stages as a whole. These are also the least certain numbers here: the synthetic SSD scan's
median error against measured Mamba kernels on the B300 is 24% (chapter 9.3).

## 13.4 Decode

Batch 64, 8 KiB of context, 32 byte steps:

| model | µs per byte | KB/s of text per GPU | vs BPE | closed form (13.2) |
|---|---|---|---|---|
| H-Net 1-stage, routers sampled | 2413 | 27 | 2.12× | |
| H-Net 1-stage, expected batch | 2468 | 26 | 2.17× | 1134 µs, 1.32× |
| H-Net 2-stage, routers sampled | 2703 | 24 | 2.38× | |
| H-Net 2-stage, expected batch | 2676 | 24 | 2.35× | 1050 µs, 1.22× |
| BPE (a 5062 µs step per 4.45 bytes) | 1137 | 56 | | 860 µs |
| byte Transformer | 16584 | 4 | 14.6× | 15470 µs |

Where a decoded byte's time goes (µs per byte, H-Net at its expected batch):

| model | outer stages | main attention | main matmuls | main norms and elementwise | LM head | launch gaps | kernels per byte |
|---|---|---|---|---|---|---|---|
| H-Net 1-stage | 367 | 1061 | 744 | 111 | 4 | 182 | 328 |
| H-Net 2-stage | 834 | 631 | 836 | 124 | 4 | 247 | 446 |
| BPE | | 892 | 167 | 27 | 18 | 33 | 60 |
| byte Transformer | | 15569 | 744 | 119 | 4 | 147 | 266 |

Counting bytes (13.2) predicted 1.32× for H-Net 1-stage; the simulator says 2.17×. The
anatomy splits its extra 1331 µs per byte:

- **Main-network weights, +576 µs (43%).** The same 2.39 GB of weights, read once per byte
  step instead of once per 4.45 bytes. The closed form has this, but at full bandwidth: a
  decode GEMM, at batch 14 or 64 alike, reads its weights at 24-62% of HBM bandwidth (47%
  across the step), because each kernel pays a few microseconds of launch, fill and drain
  that an 8-44 MB read cannot hide.
- **Outer stages, +367 µs (28%).** Eight Mamba-2 layers and a router over all 64 sequences
  every byte, as 63 small kernels: 3.7× the time their weight and state bytes take at full
  bandwidth.
- **Attention, +169 µs (13%).** H-Net reads 10% *less* KV per byte than BPE, but at a main
  batch of 14 the decode attention kernel streams the KV cache at 64% of HBM bandwidth,
  against 86% at batch 64.
- **Launches, +149 µs (11%), and elementwise ops, +84 µs (6%).** 328 kernels per byte
  against 60; each CUDA-graph launch costs 0.55 µs before the kernel even starts.

Only the first bullet is in the closed form, and only at the wrong rate. The byte Transformer
is the opposite case: 94% of its time is one kernel, attention over 8192 bytes of KV, which
runs near the roofline, so the simulator is only 7% above the closed form.

## 13.5 Batch

µs per byte of every sequence, with the text each GPU produces in brackets (H-Net at its
expected batch, 4 byte steps):

| model | batch 16 | batch 64 | batch 256 |
|---|---|---|---|
| H-Net 1-stage | 1597 (10 KB/s) | 2467 (26 KB/s) | 5354 (48 KB/s) |
| H-Net 2-stage | 2016 (8 KB/s) | 2676 (24 KB/s) | 5320 (48 KB/s) |
| BPE | 500 (32 KB/s) | 1137 (56 KB/s) | 3807 (67 KB/s) |
| H-Net 1-stage ÷ BPE, simulated | 3.19× | 2.17× | 1.41× |
| the same, closed form (13.2) | 1.95× | 1.32× | 1.09× |
| H-Net 2-stage ÷ BPE, simulated | 4.03× | 2.35× | 1.40× |
| the same, closed form | 2.13× | 1.22× | 0.89× |

The re-read weights amortise over the batch, so the gap closes as the batch grows, from 3.2×
at batch 16 to 1.4× at 256. The closed form gets that trend and misses the level at every
batch. At batch 256 it even says the 2-stage model should be *cheaper* than BPE: its 6.9-byte
chunks cut the main network's KV per byte to under half of BPE's. The simulator says 1.40×,
because the outer stages are now 43% of the 2-stage model's time (21% of the 1-stage's), and
most of that is the Mamba-2 state update. That kernel replays the real SASS Triton compiles
for the B300 (median error 10% against measurement, chapter 9.3), and it moves its state at
about a third of HBM bandwidth: 116 µs per layer for 268 MB at batch 256, where the roofline
is 39 µs.

## 13.6 Dynamic chunking at serving time

In decode, which sequences run the main network changes every step: a sequence's router fires
when its next byte starts a new chunk. With independent routers the main network's batch is
binomial, 13.6 on average for H-Net 1-stage at *B* = 64 but anywhere from 6 to 18 in our
32-step draw. Does that variability cost anything beyond the average? Three runs separate it:

| model | routers sampled | same draw, spread evenly | expected batch |
|---|---|---|---|
| H-Net 1-stage (main batch 13.3 sampled, 13.6 expected) | 2413 µs | 2466 µs | 2468 µs |
| H-Net 2-stage (8.8 sampled, 9.3 expected) | 2703 µs | 2597 µs | 2676 µs |

"Spread evenly" keeps the draw's total main-network work and removes its step-to-step swings,
so sampled against evened is the cost of variability alone: **−2.1%** for the 1-stage model
and **+4.1%** for the 2-stage one. The sign differs because kernel time is a step function of
batch, not a line, so which batch sizes a draw lands on matters as well as their average; the
anatomy puts almost the whole difference in the main network's attention kernel (the decode
GEMMs are weight bound and cost the same at any of these batches). Variability is worth a few
percent either way, and only a model that times each batch size can say which way.

The analytic shortcut, running the expected batch every step, is within 2.3% of the sampled
run in both models. Against the evened run it is +0.1% for the 1-stage model, whose draw
averaged close to its expectation (13.3 against 13.6), and +3.1% for the 2-stage model, whose
draw averaged 5% low (8.8 against 9.3): what is left is the draw's mean, not its swings. So the
expected batch is a good model of an H-Net server, **if** it is kept exact. Our first version
rounded it up to whole sequences (14 and 10), which added 2.8% and 7.8% of main-network work
and showed up as a 2-7% "variability" effect that was not there.

## 13.7 What it means

- **Equal FLOPs per byte is not equal cost per byte.** On one B300, H-Net does 3% fewer FLOPs
  per byte than the BPE Transformer and costs 1.43× as much per byte in prefill and 2.2× in
  decode at batch 64.
- **Counting bytes gets the trend, not the size.** The decode closed form says 1.32× at batch
  64; the simulator says 2.17×. The difference is what each kernel costs at small batch:
  weights read at half the bandwidth, attention at 64% of it with 14 sequences, 5.5 times the
  kernel launches per byte, and a state update at a third of the bandwidth. A plan made from
  FLOPs, or even from bytes, would underestimate what H-Net needs to serve.
- **Batch is the lever, and it costs latency.** At batch 256 the gap is down to 1.4×, at
  5.4 ms per byte for each user (about 187 bytes, or 42 BPE tokens' worth of text, per
  second), against BPE's 3.8 ms.
- **The next bottleneck is the outer stages.** A state update at the HBM roofline would save
  about 615 µs per byte at batch 256 and take H-Net 1-stage from 1.41× to about 1.24× BPE.
- **The result depends on attention's shape.** We gave every main network full multi-head
  attention, so KV reads dominate decode and the models' similar KV per byte keeps them close.
  With grouped-query attention (4 KV heads, as most served models use) KV reads fall for both,
  weights become the larger share, and the closed form at batch 64 goes from 1.32× to 2.1×
  for the 1-stage model (1.22× to 2.6× for the 2-stage).

Two serving ideas follow, neither simulated here: fuse each outer stage into one persistent
kernel, which removes most of its 63 launches per byte and their fill and drain, and let
sequences whose router fired wait a step so the main network runs on a larger batch, which
trades a byte of latency for fewer weight reads.

## Exercises

1. At batch 64, how many bytes of main-network weights does each generated byte cost with BPE,
   and with H-Net?
2. Why do H-Net 1-stage and BPE read about the same KV cache per byte?
3. The byte Transformer costs 5.25× BPE per byte in prefill but 14.6× in decode. Why so
   different?
4. Sampling the routers made the 2-stage model 4.1% slower than the same draw spread evenly,
   and the 1-stage model 2.1% faster. How can a variable batch be cheaper than a steady one?

Answers:

1. BPE reads 2.39 GB per step of 64 × 4.45 bytes: **8.4 MB per byte**. H-Net reads the same
   2.39 GB per step of 64 bytes: **37 MB per byte**, 4.45× more.
2. A sequence reaches the main network once every *R* bytes and then reads *S*/*R* positions:
   *S*/*R*² positions per byte, against *S*/4.45² for BPE. With *R* = 4.7 that is **0.90×**
   BPE's KV per byte.
3. Prefill is compute bound, and the byte model does 4.6× the FLOPs per byte (attention over
   sequences 4.45× longer makes it a little worse). Decode is memory bound: every byte reads
   8192 positions of KV, where a BPE token reads 1840 and yields 4.45 bytes, so **19.8×** the
   KV per byte and 4.45× the weights. That is 18× in closed form; it simulates at 14.6×
   because BPE's smaller decode kernels sit further from the roofline than the byte model's
   one big attention kernel.
4. Kernel time is a step function of batch. The decode attention kernel at some batch sizes
   costs little more than at a smaller one, so a draw that lands on cheap sizes more often comes
   out ahead, and one that lands on expensive sizes behind. The anatomy shows it: the
   difference is almost entirely in the main network's attention.

## Reproduce

| what | command |
|---|---|
| all of this chapter | `python research/bytes/run.py` (about 40 minutes on 4 cores) |
| one study | `python research/bytes/run.py --only decode` (or `flops`, `prefill`, `batch`, `closed`) |
| the closed form of 13.2 | `hnet.decode_traffic(hnet.HNET_1STAGE_XL, 64, 8192)`, bytes per step by part |
| one model in Python | `rackscale.hnet.prefill(hnet.HNET_1STAGE_XL, Parallel(), 8, 8192)`, `hnet.decode_steps(..., counts=hnet.step_counts(...))` |

Next: [14. Why cycle level, and how far to trust a simulated rack](14-why-cycle-level.md).
