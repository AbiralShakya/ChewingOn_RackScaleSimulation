# 18. Beyond language models: vision, diffusion, VLMs and recommenders

Every chapter so far climbed the ladder with language models. Other models climb the same
ladder but put their weight on different rungs. An image encoder is a prefill with no cache. A
diffusion model runs the same network a hundred times per image. A video model's sequence is so
long that attention becomes most of the work. A recommender does almost no arithmetic at all,
and is bound by HBM gathers and an all-to-all.

This chapter lowers each of them to the same kernels and collectives, gives each a closed form,
and checks the two against each other. The simulator gained:

- an `embedding_bag` kernel (random-row gathers with sum pooling, and their fused backward);
- `rackscale.vision` (ViT, DiT and VLM prefill);
- `rackscale.recsys` (DLRM-style steps).

| family | the work per sample | what bounds it | the rung it loads |
|---|---|---|---|
| ViT encoder | one forward over a few hundred patch tokens | short GEMMs, short attention, unfused elementwise | one GPU |
| image DiT | 50 steps × 2 forwards (guidance) | the same, a hundred times | one GPU |
| video DiT | 50 × 2 forwards over 75,600 tokens | attention, quadratic in tokens | many GPUs (Ulysses all-to-all) |
| VLM prefill | a ViT, then the LLM over image and text tokens | the LLM's prefill of the image tokens | one GPU or a TP group |
| DLRM-style recommender | gathers from terabyte tables, two small MLPs | HBM bandwidth and the all-to-all | every GPU's HBM, the scale-up network |

## 18.1 A ViT is a prefill with no cache

A ViT cuts a 224 × 224 image into 16 × 16 = 256 patches of 14 × 14 pixels, adds a class token,
and runs a transformer encoder over the 257 tokens. Attention is not causal, so every query
sees every key and attention costs twice as much per token as a causal prefill of the same
length. A forward costs

```
FLOPs = 2 · P · N  +  4 · L · h · N²  (+ the patch embedding, 2 · 256 · h · 588)
```

with *P* the block parameters, *N* the tokens, *L* the blocks and *h* the width (heads × head
dim). The attention term over the matmul term is 4*LhN*² / (2 · 12*Lh*²*N*) = ***N* / 6*h***:
attention is a small fraction of the FLOPs until the sequence is several times wider than the
model. At *N* = 257 and *h* = 1024 that is 4%.

```python
from rackscale import vision, models
r = rs.simulate("hgx_b300_8", vision.vit_forward(models.VIT_L_14, batch=64), fast=True)
```

| model | GPUs | params | GFLOP per image | batch of 64 | images/s | % of peak | time in attention / GEMM / other | attention share of FLOPs |
|---|---|---|---|---|---|---|---|---|
| ViT-L/14 | 1 B300 | 0.30 B | 162 | 11.2 ms | 5695 | 37% | 22 / 58 / 20% | 4% |
| ViT-H/14 | 1 B300 | 0.63 B | 335 | 22.5 ms | 2848 | 39% | 16 / 64 / 20% | 3% |
| ViT-22B | 8 B300, TP8 | 21.7 B | 11 256 | 123.3 ms | 519 | 30% | 3 / 72 / 24% | 1% |

The FLOP counts match the literature, which quotes ViT-L/14 at 81 and ViT-H/14 at 167 GFLOPs
counting multiply-adds once. The time doesn't follow the FLOPs:

- **Attention is 4% of the FLOPs and 22% of the time.** A 257-key sequence is two or three K/V
  blocks per CTA, so each attention CTA spends most of its life in prologue and epilogue. It
  runs at about 580 TFLOP/s against 1.6–1.9 PFLOP/s for the GEMMs (chapter 15's waves and
  pipeline fill, at a scale where they never amortise).
- **A fifth of the time does no matrix work.** GELU, residual adds, LayerNorm and the position
  add each read and write the whole activation once. Here they are separate kernels. Fusing
  them into the GEMM epilogues, which inference compilers do, is worth up to 20%.
- **ViT-22B on eight GPUs.** It is 72 times larger than ViT-L, and TP8 adds two all-reduces
  per block. Its GEMMs are big enough to run well, but 24% of the GPU time is still
  elementwise and norm kernels.

## 18.2 A diffusion transformer runs the network a hundred times

A DiT denoises a latent (a 256 × 256 image becomes 32 × 32 × 4 through the VAE) cut into
2 × 2 patches: 256 tokens. Each block adds adaLN, a linear map from the timestep and class
embedding to six modulation vectors. Classifier-free guidance runs every step twice, with and
without the condition, which is a batch of two. An image costs

```
FLOPs per image = steps · 2 · (2 · P · N + 4 · L · h · N²)
```

DiT-XL/2 (28 blocks, *h* = 1152), 50 steps, batch 8 (16 sequences per step), one B300:

```python
vision.dit_sample(models.DIT_XL_2, batch=8, latent=32, steps=50)
```

| image | latent tokens | GFLOP per forward | TFLOP per image | 8 images | images/s | % of peak | attention: time / FLOPs |
|---|---|---|---|---|---|---|---|
| 256 px | 256 | 237 | 23.7 | 250 ms | 32.0 | 31% | 7% / 4% |
| 512 px | 1024 | 1049 | 104.9 | 944 ms | 8.5 | 36% | 18% / 13% |

The DiT paper quotes 118.6 and 524.6 Gflops per forward for XL/2 at 256 and 512 px, counting
multiply-adds once: exactly half of these. The model is the size of a ViT-H, so its
efficiency is a ViT's, and it is a hundred forwards per image. That makes the step count the
largest knob there is. Distilled samplers that need 4 or 8 steps instead of 50, and guidance
distillation that removes the second pass, cut the cost by the same factor and touch no
hardware.

The *N* / 6*h* rule says when attention takes over. At 512 px it is 1024 / 6912 = 15% of the
FLOPs, and attention is the slower kernel, so it is 18% of the time.

## 18.3 Video: attention takes over, and Ulysses spreads it

A video model's latent has a time axis. Take the shape of Wan2.1-14B (40 blocks, *h* = 5120,
40 heads, cross-attention to 512 text tokens) and 81 frames of 720p video. The VAE compresses
it 4× in time and 8× in space, to 21 × 90 × 160 latents, which is 21 × 45 × 80 =
**75,600 tokens** in 2 × 2 patches. Now *N* / 6*h* = 2.5, and self-attention is
**72% of the 13.0 PFLOP** of one denoising step.

One sequence this long doesn't fit one GPU's time budget, and data parallelism doesn't help a
single video. Ulysses (sequence parallelism by all-to-all) shards the tokens over *cp* ranks.
Each rank projects its tokens to Q, K and V. An all-to-all turns the sequence shards into head
shards, so each rank holds every token for 40 / *cp* heads. Attention runs locally, and a
second all-to-all turns the output back. Per block and rank:

```
all-to-all bytes   = 4 · (N / cp) · h · 2              (Q, K, V out; O back; BF16)
attention FLOPs    = 4 · N² · h / cp
FLOPs per byte     = N / 2 = 37,800
```

That is ten times the ratio at which a B300 is as fast at the all-to-all as at the attention:
2.46 PFLOP/s over roughly 630 GB/s of NVLink all-to-all bandwidth is 3,900 FLOP/byte. So
Ulysses should cost a few percent at any *cp*, and the simulator agrees:

```python
vision.dit_sample(models.VIDEO_DIT_14B, latent=(90, 160), frames=21, steps=1, text_tokens=512,
                  latent_channels=16, learn_sigma=False, cp=8)
```

| system | GPUs (Ulysses) | one step | 50 steps | % of peak | self-attention share of the step | all-to-all share |
|---|---|---|---|---|---|---|
| HGX B300 | 8 | 1.25 s | 62 s | 53% | 67% | 3% |
| GB300 NVL72 | 40 | 0.27 s | 13 s | 49% | 62% | 5% |

Five times the GPUs give 4.6 times the speed. Attention runs at the efficiency of a long
FlashAttention kernel, so it takes a little less of the time (62–67%) than of the FLOPs (72%),
and the all-to-alls cost 3–5%, as the ratio said. A 5-second 720p clip takes about a minute
on one HGX B300 and 13 seconds on 40 GPUs of a rack, before the VAE decode, which is not
modelled. This is the workload for which a 72-GPU NVLink domain buys latency rather than
throughput.

Ulysses stops at the head count: *cp* must divide 40 heads, so 40 GPUs is the limit for this
model. Beyond it, ring attention (chapter 11's context parallelism) or the two combined
(USP) shard further, at the cost of more steps on the critical path.

## 18.4 VLMs: image tokens are paid for in the language model

A LLaVA-style VLM encodes each image with a ViT (ViT-L/14 at 336 px gives 24 × 24 = 576
patch tokens), maps the tokens into the language model with a two-layer projector, and
prefills the LLM over the image and text tokens together. "AnyRes" models cut a large image
into several tiles plus a thumbnail, so five images' worth of tokens per picture is common.

The closed form says where the cost lands. The ViT costs 2 × 0.30 B × 576 = 0.35 TFLOP per
image, plus 0.03 for attention. The LLM costs 2 × 7.0 B = 14 GFLOP for *every* token, so the
576 image tokens cost Llama-3-8B 8.0 TFLOP, about 21 times the encoder. Eight requests with
256 text tokens each, on one B300:

```python
vision.vlm_prefill(models.LLAMA3_8B, models.VIT_L_14, batch=8, images=1, image=336, text=256)
```

| images per request | image tokens | prefill | ViT alone | LLM alone | ViT share of the time | ViT share of the FLOPs |
|---|---|---|---|---|---|---|
| 1 | 576 | 79.0 ms | 3.5 ms | 75.4 ms | 4% | 3% |
| 5 (AnyRes tiles) | 2880 | 320.8 ms | 16.2 ms | 303.8 ms | 5% | 4% |

The encoder is a rounding error. What an image costs is its token count in the language
model, both in prefill and afterwards, because those tokens sit in the KV cache for the whole
decode (chapter 6). That is why VLMs pool or resample image tokens (2 × 2 pooling cuts 576 to
144), and why splitting the encoder onto its own GPUs, the way chapter 12 splits prefill from
decode, saves little on a model like this one.

## 18.5 Recommenders: gathers and an all-to-all

A DLRM-style recommender has dense features and categorical ones. Each categorical feature
indexes an embedding table, and a multi-hot feature looks up *L* rows of one table and sums
them (an embedding bag). A bottom MLP embeds the dense features, a pairwise dot-product
interaction mixes the *T* + 1 vectors, and a top MLP scores the result. The MLPs have tens
of millions of parameters. The tables are terabytes.

So the tables are model parallel. With table-wise sharding (TorchRec's default), each of *N*
ranks owns *T* / *N* whole tables, looks them up for the global batch, and an all-to-all sends
every sample's pooled rows to the rank that owns the sample, where the MLPs run data parallel.
Training sends the gradients back the same way, and the backward updates every gathered row in
place, with the optimizer fused in (FBGEMM's `backward_..._exact` kernels). Per rank with *b*
samples per rank, *D*-wide rows of *e* bytes:

```
HBM bytes of the gather     = b · T · L · D · e
bytes the all-to-all sends   = b · T · D · e · (N − G) / N     (G ranks share a faster fabric)
```

The gather moves *L* times more bytes through HBM than the all-to-all sends over the network,
so which one is slower depends on *L* × network bandwidth / HBM bandwidth.

The `embedding_bag` kernel gives each bag a warp, keeps four row loads in flight per warp and
spreads the rows over the whole table. On one B300, 16,384 bags of 32 rows × 128 fp32 from a
50M-row table stream 268 MB from HBM in 44.1 µs (6.3 TB/s, with 0% L2 hits). The same lookups
into a 20,000-row hot set hit L2 93% of the time and take 20.1 µs. The fused backward reads and
writes every row and takes 67.6 µs: 1.5 times, not 2, because the dirty rows still in L2 when
the kernel ends are written back later, during whatever runs next.

```bash
rsim kernel --gpu b300 --dtype fp32 embedding_bag 16384 32 128 50000000      # add 1 for the backward
```

The model is 256 tables of 40M rows × 128 fp16, which is 2.6 TB of embeddings, too much for
eight B300s. Each sample does 32 lookups per table, and each GPU takes 4096 samples:

```python
from rackscale import recsys
rs.simulate("gb300_nvl72", recsys.step(recsys.DLRM_2TB, 64, 4096), fast=True)
```

| system | GPUs | tables per GPU | step | gather | update | each all-to-all | MLPs and the rest | samples/s |
|---|---|---|---|---|---|---|---|---|
| GB300 NVL72 | 16 | 164 GB | 6.27 ms | 1.30 ms | 2.58 ms | 0.44 ms | 1.47 ms | 10.5 M |
| GB300 NVL72 | 32 | 82 GB | 6.40 | 1.30 | 2.57 | 0.46 | 1.49 | 20.5 M |
| GB300 NVL72 | 64 | 41 GB | 6.47 | 1.30 | 2.58 | 0.42 | 1.52 | 40.5 M |
| 8 × HGX B300, 800G InfiniBand | 64 | 41 GB | 10.42 | 1.30 | 2.58 | **2.43** | 1.67 | 25.2 M |

The closed forms per GPU: the gather moves 4096 × 256 × 32 × 256 B = 8.59 GB, which is
1.30 ms at the 6.6 TB/s the long gather reaches. The all-to-all sends 264 MB. Inside the
NVL72 that is 0.42 ms at the 630 GB/s bus bandwidth NCCL's all-to-all reaches there in the
simulator (chapter 4). Across eight HGX nodes, 7/8 of it, 235 MB, leaves through each GPU's
100 GB/s NIC: 2.35 ms against the simulated 2.43.

- **The arithmetic doesn't matter.** Three quarters of the step is HBM traffic (the gather
  and the row update, 60%) and the two all-to-alls (13%). The MLPs, the interaction and the
  dense all-reduce are the rest.
- **Weak scaling is flat inside the domain.** With a fixed batch per GPU, the gather bytes
  and the all-to-all bytes per GPU don't depend on *N*, so 16, 32 and 64 GPUs take the same
  6.3–6.5 ms and the throughput grows with the GPU count. What *N* buys is memory: 164 GB of
  tables per GPU at 16, 41 GB at 64.
- **The scale-up domain decides the step.** The all-to-all is 0.32 of the gather inside an
  NVL72 and 1.8 of it across InfiniBand. The same 64 B300s run 61% slower over the NICs,
  because almost every byte of the all-to-all has to leave the node. This is the workload where
  a 72-GPU NVLink domain matters most, and the reason recommender systems were early users of
  large NVLink domains and of hot-row caches, which shrink the gather.

## 18.6 What is modelled, and what is not

| | modelled | not modelled |
|---|---|---|
| ViT and DiT | patch embedding, non-causal attention, GELU MLPs, adaLN, guidance, cross-attention, Ulysses | fused epilogues, the VAE, the text encoder, image preprocessing |
| VLM | ViT, projector and causal LLM prefill over image and text tokens | token pooling and resamplers (pass fewer tokens), encoder disaggregation |
| recommenders | table-wise sharding, random-row gathers, all-to-all both ways, fused row updates, dense all-reduce | row- or column-wise sharding, hot-row caches (`hot_rows` stands in for a perfect one), index deduplication, the input pipeline, HBM page locality |
| long kernels | the 75,600-token attention and the large gathers are extrapolated from simulated smaller instances (`rackscale.extrap`, chapter 14) | |

## 18.7 What to take away

| model | closed form | simulated | knob |
|---|---|---|---|
| ViT-L/14, 257 tokens | attention = *N*/6*h* = 4% of FLOPs | 22% of time; 20% in elementwise and norms | fuse epilogues; longer sequences per kernel |
| DiT-XL/2 | steps × 2 × forward = 23.7 TFLOP per 256 px image | 32 images/s per B300 | fewer steps, guidance distillation |
| video DiT, 75,600 tokens | attention 72% of 13.0 PFLOP per step; Ulysses at *N*/2 FLOP per byte | 1.25 s per step on 8 B300s, 0.27 s on 40 GB300s; all-to-all 3–5% | *cp* up to the head count, then ring |
| VLM, 576 image tokens | the LLM pays 25 × the encoder | ViT 4–5% of prefill time | fewer image tokens |
| DLRM, 2.6 TB | gather *L* × the all-to-all bytes | 6.5 ms on 64 NVL72 GPUs, 10.4 ms over IB | a larger NVLink domain, caches of hot rows |

## Exercises

1. At what sequence length does attention take half the FLOPs of a non-causal encoder with
   *h* = 4096? Of a causal prefill?

   *Answer:* half when attention equals the matmuls, *N* / 6*h* = 1, so *N* = 24,576 tokens. A
   causal query sees half the keys on average, so the ratio is *N* / 12*h* and the length doubles
   to 49,152.

2. A text-to-image service runs DiT-XL/2 at 512 px with 50 steps and guidance. A distilled
   sampler needs 4 steps and no guidance. How many more images per second does one B300 make?

   *Answer:* 50 × 2 / 4 = 25 times as many, about 210 images/s, since nothing else changes.

3. The model of 18.5 does 8 lookups per table instead of 32. Which part of the step grows in
   importance, inside an NVL72 and across InfiniBand?

   *Answer:* the gather shrinks fourfold and the all-to-all doesn't. Their ratio is
   (*N* − *G*)/*N* × BW_HBM / (*L* × BW_net): at *L* = 8 it goes from 0.32 to 1.3 inside the
   NVL72, so even there the network takes over, and from 1.8 to 7.2 over InfiniBand
   (`recsys.balance`).

## Reproduce

| what | command |
|---|---|
| every table (about 15 minutes on 4 cores) | `python docs/book/scripts/beyond_llm.py` |
| one experiment | `python docs/book/scripts/beyond_llm.py vit` (or `dit`, `video`, `vlm`, `recsys`) |
| an embedding gather | `rsim kernel --gpu b300 --dtype fp32 embedding_bag BAGS POOLING DIM TABLE_ROWS [1]`, or `{"type": "embedding_bag", ...}` in a workload |
| the lowerings | `rackscale.vision.vit_forward`, `dit_sample`, `vlm_prefill`; `rackscale.recsys.step` |
| the closed forms | `vision.encoder_flops`, `vision.dit_flops`, `vision.vlm_flops`, `recsys.balance` |

Back to [the ladder](README.md#the-ladder).
