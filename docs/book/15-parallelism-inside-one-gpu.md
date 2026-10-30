# 15. Parallelism inside one GPU

Chapters 5 and 11 start at one GPU and go up: data, tensor, pipeline, expert and context
parallelism split a model across many GPUs. Those GPUs hold a second ladder, though, and
every rung of it is a parallelism choice with the same shape as the ones across GPUs. Some
work is split across more workers. Each worker gets a smaller share of a resource it has to
share. And the workers pay more to agree with each other. This chapter climbs that inner
ladder on the calibrated B300, from one lane of one warp up to the whole chip and the
communication kernels that share it. It changes one choice at a time in the simulator's
Blackwell GEMM, and writes down the closed form each time.

The rule of thumb at the end is the same as at rack scale. **Parallelise until a shared
resource saturates, then spend the next level's parallelism on hiding latency instead of
adding throughput.**

## 15.1 The ladder, and what it costs to agree

Each level below has more workers than the one above it, and a more expensive way for them
to synchronise. Latencies are measured on the B300 (`calibration/results/b300-sxm6-ac-sm103/`)
and given in SM cycles at 2032 MHz.

| level | workers | shared resource | how they agree | cost to agree |
|---|---|---|---|---|
| lane | 32 per warp | the warp's instruction stream | nothing: they are one instruction | 0 |
| warp | up to 16 per SMSP | the SMSP's issue slot and register file | register dependency (`FFMA`) | 4 cycles |
| | | | `SHFL` across lanes | 27 cycles |
| SM sub-partition × 4 | 4 per SM | shared memory, L1, the tensor core | `bar.sync` of 128 / 1024 threads | 21 / 77 cycles |
| | | | shared memory round trip, mbarrier | 29 cycles |
| SM pair (cluster) | 2 SMs | each other's shared memory (DSMEM), one 2-SM UMMA | remote shared-memory access | 193 cycles |
| GPU | 148 SMs | L2 (126 MB, two dies), HBM | a flag in L2 | 320 (near) to 706 (far die) cycles |
| | | | a same-address global atomic | 1.5 G/s for the whole chip |
| | | | kernel boundary: launch / CUDA graph | 1.71 / 0.55 µs = 3500 / 1100 cycles |

The scale is set by four steps of roughly 5–10× each. A register dependency costs 4 cycles.
Talking within an SM costs tens, talking across the chip costs hundreds, and a kernel
boundary costs thousands. Outside the GPU, the next rungs (chapter 4) continue the same
progression: a small NVLink collective costs several microseconds, and an InfiniBand one more.

The third column is just as important. Every level adds workers that compete for something,
and the competition, not the synchronisation, is usually what stops scaling. Warps compete
for the issue slot, CTAs for the tensor core and shared memory, SMs for L2 and HBM, and the
compute kernel competes with NCCL for SMs.

## 15.2 Lanes and warps: Little's law at the bottom

A pipe with latency *λ* and issue rate *r* needs *λ · r* independent operations in flight to
stay busy. On the B300, `FFMA` has a 4-cycle dependent latency and issues once per cycle per
SMSP, so each SMSP needs **4 independent FMAs in flight**. They can come from one warp with
four independent accumulators (instruction-level parallelism, ILP) or from four warps with
one each (thread-level parallelism). The two are interchangeable for the pipe. They are not
interchangeable for the register file. Four warps hold four times the registers of one warp
with ILP 4, which is why a register-heavy kernel tends to rely on ILP and a light one on
occupancy.

The same law sets every other rung, with bigger numbers:

| pipe | latency λ | rate | in flight needed | where it comes from |
|---|---|---|---|---|
| FFMA, per SMSP | 4 cycles | 1 / cycle | 4 instructions | ILP or 4 warps |
| `mma.sync` HMMA, per SM | 20 cycles | 0.5 / cycle | 10 MMAs | 2–3 warps per SMSP, each with independent tiles |
| `tcgen05.mma`, per SM | 128 cycles of tensor work per m128n256k16 | back to back | ≥ 2 queued (queue depth 7) | one MMA warp and a deep shared-memory ring |
| HBM, whole chip | 418 ns | 6.82 TB/s | 2.85 MB = 19 KB / SM | TMA stages (chapter 2.5) |
| L2 → SM, per SM | ~300–700 cycles | 59 B / cycle | 18–41 KB / SM | TMA stages |

On Ampere, latency was hidden by warps: many of them, each issuing `mma.sync`, and the
scheduler picked whichever was ready. On Blackwell one thread issues an MMA that runs for
about 128 cycles from shared and tensor memory. So the parallelism moved out of the warp
count and into the pipeline depth, which is the subject of 15.4.

## 15.3 Warp specialisation: parallelism by role

Once one warp can keep the tensor core busy, the other warps of a CTA get different jobs
(chapter 3.2): a TMA producer warp, an MMA issuer, and epilogue warps that drain the
accumulator. FlashAttention-4 adds two softmax warpgroups and a correction warpgroup. That
is pipeline parallelism inside one SM. The stages are roles, the micro-batches are k-blocks
of the tile, and the "bubble" is any role waiting on another. Chapter 3 measured the result:
FA4's tensor core is busy 91% of the kernel, and its softmax warps wait about 40% of their
cycles for *S*. Like a pipeline across GPUs, the slowest role sets the rate. Making a fast
role faster gains nothing. That is why doubling the tensor core alone stops paying at 1.21×
once the exponential unit becomes the slowest stage (chapter 2.4).

## 15.4 Pipeline depth: bytes in flight per SM

The GEMM's TMA producer fills a ring of shared-memory stages, and each stage is one
*A* tile and one *B* tile of a k-block. The ring has to hold enough bytes to cover the L2 or
HBM round trip (15.2). It also has to fit in 228 KB of shared memory, along with any other
CTA that wants to live on the same SM. One stage of a 128 × 256 BF16 tile with k-block 64 is
16 KB of *A* plus 32 KB of *B*, so 48 KB.

```bash
rsim kernel --gpu b300 gemm 4096 4096 4096 --stages S
```

| stages | CTAs per SM | ring per SM | µs at 2032 MHz | % of BF16 peak | µs power-capped |
|---|---|---|---|---|---|
| 1 | 2 | 96 KB | 110.5 | 50.5 | 127.4 |
| **2** (default) | **2** | 192 KB | **74.1** | **75.3** | **91.6** |
| 3 | 1 | 144 KB | 83.0 | 67.2 | 101.0 |
| 4 | 1 | 192 KB | 82.4 | 67.7 | 101.0 |

Two effects show up here. With **one stage**, a CTA's producer can't load k-block *i* + 1
until the MMA has released k-block *i*, so the round trip is fully exposed. Only the second
CTA on the SM hides any of it, and the kernel loses a third of its speed. **Three or four
stages** cover the latency, but at 144 KB or more of shared memory only one CTA fits on an
SM. Then nobody overlaps the epilogue, the 128 × 256 fp32 accumulator read out of TMEM and
written back as BF16, with the next tile's mainloop. The tensor core idles for it, which
costs 11%. Two CTAs with two stages each get both latency hiding and epilogue overlap. That's
why the simulator's default does this, and why CUTLASS's persistent `sm100` kernels
double-buffer the TMEM accumulator to get the same overlap from one CTA.

So occupancy is not a goal in itself. It is a consequence of the resources each CTA asks
for, and it buys one specific thing: one CTA's latency or epilogue can hide behind another
CTA's work.

## 15.5 Tile shape: the L2 → SM budget

A CTA computing a *t*<sub>m</sub> × *t*<sub>n</sub> output tile reads
(*t*<sub>m</sub> + *t*<sub>n</sub>) × *k*<sub>b</sub> elements per k-block and does
2 *t*<sub>m</sub> *t*<sub>n</sub> *k*<sub>b</sub> FLOPs on them. Its arithmetic intensity
against the L2, not HBM, is

```
AI_L2 = 2 · t_m · t_n / (bytes · (t_m + t_n / c))          FLOP per byte from L2
```

where *c* is the cluster size: with TMA multicast, the two CTAs of a cluster share one copy
of *B*. To keep the tensor core at its 8192 BF16 FLOP per cycle, each SM must pull
8192 / AI<sub>L2</sub> bytes per cycle out of the L2, and the L2 can deliver **59 B per
cycle per SM**. That gives a ceiling on the fraction of peak, before waves:

```
ceiling = min(1, 59 · AI_L2 / 8192)
```

```bash
rsim kernel --gpu b300 gemm 4096 4096 4096 --tile-n N --cluster C
```

| tile | cluster | AI<sub>L2</sub> (FLOP/B) | bytes/cycle needed | L2 ceiling | × wave efficiency (15.6) | simulated |
|---|---|---|---|---|---|---|
| 128 × 64 | 2 | 51 | 160 | 37% | 37% | 29.3% |
| 128 × 128 | 2 | 85 | 96 | 62% | 61% | 52.1% |
| 128 × 256 | 1 | 85 | 96 | 62% | 54% | 57.0% |
| 128 × 256 | 2 | 128 | 64 | 93% | 80% | **75.3%** |

The ordering and most of the size come from this one formula. Halving the tile width cuts
throughput nearly in half, because the GEMM becomes an L2-bandwidth problem long before HBM
notices: all four runs read the same 84–101 MB from DRAM. Multicast is worth 32% on the
same tile, because it removes half of the *B* traffic, the operand that dominates when
*t*<sub>n</sub> > *t*<sub>m</sub>. The remaining 5–20% between the ceiling and the
simulation is the prologue, the epilogue, and L2 latency the ring doesn't fully cover. The
one row where the simulation beats the ceiling × wave product is cluster 1. We have not
isolated why; a candidate is that unpaired CTAs pack the last wave more evenly than pairs.

This is the inside-the-GPU version of choosing tensor parallelism degree in chapter 11. A
bigger tile per worker means more reuse per byte moved and fewer workers. Clusters are the
first rung where the hardware lets workers share a fetched byte, which is the same trick
NVLS multicast plays across GPUs (chapter 4).

## 15.6 CTAs to SMs: waves

A GEMM has a fixed number of output tiles, and the GPU has 148 SMs. When the tiles don't
divide evenly, the last wave leaves SMs idle. Two co-resident CTAs share one SM's tensor
core, so the unit that matters is **tiles per SM**, not tiles per CTA slot:

```
t ≈ ⌈tiles / SMs⌉ · t_tile          (t_tile = one tile's time with the SM to itself)
wave efficiency = tiles / (⌈tiles / SMs⌉ · SMs)
```

```bash
rsim kernel --gpu b300 gemm M 4096 4096 --tile-n 256 --stages 2
```

| M | tiles | ⌈tiles / 148⌉ | µs at 2032 MHz | % of peak |
|---|---|---|---|---|
| 2048 | 256 | 2 | 38.4 | 72.6 |
| 2304 | 288 | 2 | 39.4 | 79.7 |
| **2432** | **304** | **3** | **65.8** | **50.3** |
| 2560 | 320 | 3 | 66.6 | 52.3 |
| 4608 | 576 | 4 | 75.0 | 83.7 |
| 4736 | 592 | 4 | 77.0 | 83.8 |

Adding 128 rows (5.6% more work) to M = 2304 costs 67% more time. The formula predicts 1.5×:
two tiles per SM become three. The extra comes from the eight SMs that run three tiles while
the other 140 sit idle: each of those third tiles runs without a partner CTA, so its
prologue and epilogue are exposed instead of hidden. Past four tiles per SM the steps flatten, because one tile more or less is a
smaller fraction of the work. Stream-K and persistent tile schedulers exist to sand down
these steps by splitting the leftover tiles' K loops across idle SMs.

## 15.7 When the tiles can't fill the GPU: split-K

At decode, a GEMM has as few as 16 rows, so a 16 × 8192 × 8192 weight multiply makes 64
output tiles for 148 SMs. The work is reading 134 MB of weights, which takes 19.7 µs at
6.82 TB/s. With only 64 SMs streaming, each would have to pull 52 B per cycle to reach full
HBM speed, and by Little's law that needs 70–100 KB of weights in flight per SM against a
loaded round trip of 1300–2000 cycles. The 4-stage ring holds 64 KB of weights, so each SM
gets 29 B per cycle. The fix is to split the K loop across several CTAs that each write fp32
partials, which are then summed. That's data parallelism over the reduction dimension, with
a reduction at the end, and it's the same structure as gradient all-reduce.

```bash
rsim kernel --gpu b300 gemm 16 8192 8192 --split-k S
```

| split | CTAs | µs | HBM bandwidth | DRAM traffic |
|---|---|---|---|---|
| 1 | 64 | 35.9 | 3.74 TB/s (55%) | 140 MB |
| 2 | 128 | 27.1 | 4.95 TB/s | 153 MB |
| **4** | 256 | **25.3** | **5.31 TB/s (78%)** | 153 MB |
| 8 | 512 | 26.8 | 5.01 TB/s | 170 MB |
| 16 | 1024 | 29.3 | 4.58 TB/s | 203 MB |

The tradeoff is the familiar one. More splits put more SMs and more bytes in flight on the
weight stream, and they also add partial sums to write and read back, which is the
"communication" of this rung. The best split sits where the two cross, here at about 2 CTAs
per SM. None of these runs ran into the power cap: a decode GEMM is memory bound and stays
at the full clock.

## 15.8 Sharing the GPU with communication

NCCL collectives run as kernels on SMs. When a collective overlaps with a GEMM, as in tensor
parallelism with async all-reduce or FSDP's prefetching gathers, those SMs are taken from
the GEMM. Chapter 11 found that each collective in its lowering takes 20 SMs.

```bash
rsim kernel --gpu b300 gemm 4096 4096 4096 --reserve-sms R
```

| SMs reserved | SMs left | ⌈512 / SMs⌉ | µs at 2032 MHz | vs 0 |
|---|---|---|---|---|
| 0 | 148 | 4 | 74.1 | |
| 8 | 140 | 4 | 73.0 | −1.5% |
| 16 | 132 | 4 | 73.1 | −1.3% |
| **24** | **124** | **5** | **99.3** | **+34%** |
| 32 | 116 | 5 | 100.0 | +35% |

The cost is not proportional to the SMs taken. It is a step function, given by the same
wave formula as 15.6. Taking 16 SMs is free here (1% faster, in fact, likely from less L2
contention), and taking 24 costs a third of the kernel. So whether
overlapping a collective with a GEMM pays depends on the GEMM's tile count modulo the SMs
left. A closed form that charges "20/148 of the GEMM" is wrong in both directions. This is
one of the effects in chapter 14.2 that only a model of the machine sees, and it is why NCCL
exposes `NCCL_MAX_CTAS` and why communication-heavy kernels such as DeepEP size their SM
budget explicitly.

## 15.9 The clock couples every rung

The power-capped column of 15.4 is worth a second look. The 1-stage kernel is 1.49× slower
than the 2-stage one at 2032 MHz, but only 1.39× slower power-capped. Its tensor core is idle
more often, so the board draws less power and the clock stays higher (1757 MHz against
1650). Every rung of this chapter makes the same trade: a more efficient kernel does more
work per cycle and gets fewer cycles per second. That's the first place where a physical
limit reaches into the parallelism ladder, and chapter 1.6 is where it is measured. At rack
scale, the same budget is shared by 72 GPUs, the power shelves and the coolant loop. Above
the rack, the facility's 800 V distribution and its CDUs take over. Those rungs are the
subject of the chapters that follow this one.

## 15.10 The whole ladder

| rung | what is split | shared resource that saturates | knob | closed form | chapter |
|---|---|---|---|---|---|
| lane / warp | instructions | issue slot, register file | ILP vs warps | Little's law, λ · r in flight | 2.2, 15.2 |
| warp roles | stages of a tile | the slowest role | warp specialisation | pipeline: slowest stage sets rate | 3.2, 15.3 |
| CTA pipeline | k-blocks | shared memory (228 KB) | stages, CTAs per SM | bytes in flight ≥ BW · λ | 2.5, 15.4 |
| tile | output elements | L2 → SM bandwidth (59 B/clk/SM) | tile shape, cluster multicast | 59 · AI<sub>L2</sub> / 8192 | 15.5 |
| SMs | tiles | SM count | tile count, stream-K | ⌈tiles / SMs⌉ | 15.6 |
| reduction | K | HBM, partial sums | split-K | max(stream, partials) | 15.7 |
| GPU sharing | SMs | SMs | `NCCL_MAX_CTAS`, SM budget | ⌈tiles / (SMs − reserved)⌉ | 15.8, 11.6 |
| board power | clock | 1100 W | none (it's a cap) | P ∝ f³ (chapter 1.6) | 1.6, 15.9 |
| GPUs in a node / rack | layers, heads, tokens, experts | NVLink, NVSwitch | DP / TP / PP / EP / CP | chapter 11.3 | 4, 5, 11 |
| jobs in a rack | GPUs | GPUs, links | partitioning | chapter 12 | 12 |
| nodes | replicas, stages | InfiniBand / RoCE | HSDP, PP across nodes | chapter 11.3 | 4, 11 |

## Exercises

1. A kernel's loop has a dependent chain of `FFMA`s and no ILP. How many warps per SMSP keep
   the FMA pipe busy? How many registers per thread can each of them use at that occupancy
   (64 K registers per SM)?

   *Answer:* 4 warps per SMSP, 16 per SM, 512 threads: up to 128 registers each.

2. What fraction of BF16 peak can a 128 × 128 tile reach with cluster 1, by the formula of
   15.5?

   *Answer:* AI = 2 · 128 · 128 / (2 · 256) = 64 FLOP/B, so it needs 128 B/clk against 59,
   which gives 46%.

3. A GEMM has 300 tiles of 128 × 256. Is it worth padding it to 296 tiles, or splitting it
   into 444?

   *Answer:* 300 tiles is ⌈300/148⌉ = 3 per SM. Making it 296 tiles (2 per SM) saves a third
   of the time if the work allows it. 444 tiles is 3 per SM again, at 2/3 of the tile size,
   so it takes about 2/3 of the time but loses reuse (15.5).

4. How many SMs can NCCL take from the 4096³ GEMM before it costs anything? And from an
   8192³ GEMM with 128 × 256 tiles?

   *Answer:* 512 tiles: up to 20 (⌈512/128⌉ = 4, ⌈512/127⌉ = 5). 2048 tiles: ⌈2048/148⌉ = 14,
   and 14 per SM holds only down to 147 SMs (2048/146 > 14). One SM is free; two cost a
   fifteenth tile per SM, about 7%. A big GEMM is not automatically a safe place to hide a
   collective.

## Reproduce

| what | command |
|---|---|
| every table in this chapter (about 20 s on 4 cores) | `python docs/book/scripts/intra_gpu.py` |
| one experiment | `python docs/book/scripts/intra_gpu.py waves` (or `depth`, `tile`, `cluster`, `splitk`, `reserve`) |
| one GEMM with explicit tiling | `rsim kernel --gpu b300 gemm M N K --tile-n 256 --stages 2 --cluster 2 --split-k 1 --reserve-sms 0` |
| see the warps of any of them | `rsim viz kernel --gpu b300 gemm M N K` |

The tiling options apply to the Blackwell lowering (`b300`, `gb300`, `b200`, `gb200`). They are the
simulator's synthetic GEMM, not cuBLAS, which chooses its own tiles. Chapter 1.4 compares the
default choices against measured cuBLAS GEMMs.

Back to [the ladder](README.md#the-ladder).
