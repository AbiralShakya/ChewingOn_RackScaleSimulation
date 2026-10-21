# 2. Inside one GPU

Chapter 1 treated a GPU as two numbers, *π* and *β*. This chapter opens it up. We go from the
chip to the streaming multiprocessor (SM), into the four sub-partitions that issue
instructions, through the pipes that execute them, down the memory hierarchy, and finally
through three generations of tensor core. Every latency and rate here was measured on our
B300 and A100 by the [calibration suite](../../calibration/README.md), unless marked *preset*.

## 2.1 The chip

| | A100 (GA100) | B300 (Blackwell Ultra) |
|---|---|---|
| SMs | 108 | 148, on two dies |
| SM sub-partitions (SMSPs) per SM | 4 | 4 |
| max resident warps per SM | 64 | 64 |
| registers per SM | 64 K × 32-bit (256 KB) | 64 K × 32-bit (256 KB) |
| shared memory per SM | 164 KB | 228 KB |
| tensor memory (TMEM) per SM | none | 256 KB |
| L2 | 40 MB | 126 MB |
| HBM | 80 GB, 1.70 TB/s measured | 288 GB *(spec)*, 6.82 TB/s measured |

A kernel launches a grid of thread blocks (CTAs). Each CTA lands on one SM and stays there.
An SM holds as many CTAs as its registers, shared memory and warp slots allow, so
**occupancy** (resident warps per SM) is set by whichever of those runs out first. Blackwell
also groups CTAs into **clusters** of up to 16 that can read each other's shared memory
(DSMEM, 193 cycles per remote access, measured) and, for `tcgen05`, run one MMA across two
SMs.

## 2.2 The SM sub-partition and the warp scheduler

Each SM is four SMSPs. Each SMSP has its own warp scheduler, register file slice and
execution pipes, and **issues at most one warp instruction per clock**. A warp is 32 threads
that execute the same instruction. So a B300 SM issues at most 4 warp instructions per
clock, and the whole chip 148 × 4 × 2.032 GHz ≈ 1.2 × 10<sup>12</sup> per second.

Every clock, each scheduler looks at its resident warps and picks one that is ready. A warp
is not ready if any of these is true (this is also the order in which the simulator checks
them, and each maps to an Nsight Compute stall reason):

1. it must still wait the fixed stall count the compiler encoded for its previous
   instruction (`wait` in Nsight);
2. an operand comes from an instruction that has not finished: a variable-latency load
   (`long_scoreboard`) or a shared-memory / MUFU result (`short_scoreboard`);
3. the pipe it needs is busy (`math_pipe_throttle`);
4. the memory instruction queue is full (`mio_throttle`, `lg_throttle`);
5. it is parked at a barrier, an mbarrier, or a wait for asynchronous copies or MMAs
   (`barrier`, plus the async waits that Nsight files under `long_scoreboard`).

If no warp is ready, the slot is lost. **Latency hiding** is just having enough ready warps
that some warp can issue every clock while others wait.

## 2.3 The pipes

Measured issue rates per SM per clock, for one instruction type at a time:

| instruction | A100 | B300 | notes |
|---|---|---|---|
| `FFMA` (FP32 multiply-add) | 64 lanes, 4-cycle latency | **128 lanes**, 4-cycle latency | B300 issues FFMA every cycle per SMSP; A100 every other cycle |
| `FFMA2` (two FP32 FMAs per lane) | n/a | 58 thread-instructions (117 FMAs) | packed FP32, new on Blackwell |
| `HFMA2` (FP16 × 2) | | 64 lanes | |
| `IADD3`, `LOP3`, `IMAD` | | 64 lanes, 4 cycles | |
| `MUFU.EX2` (2<sup>x</sup>) | 16 per clock, 17 cycles | **32 per clock**, 14 cycles | the softmax exp |
| other `MUFU` (`RSQ`, `RCP`, `LG2`, `SIN`, `TANH`) | 16 per clock, 17 cycles | 16 per clock, 18 to 20 cycles | |
| `DFMA` (FP64) | 31 per clock, 6 cycles | **1.9 per clock, 64 cycles** | B300 FP64 is about 1.2 TFLOP/s for the whole chip |
| `SHFL.BFLY` | | 32 per clock, 27 cycles | |

From these, the non-tensor peaks: FP32 is 148 × 2.032 GHz × 128 × 2 = **77 TFLOP/s** on
B300 and 108 × 1.395 GHz × 64 × 2 = **19 TFLOP/s** on A100. The BF16 tensor core is 32× the
FP32 rate on B300 and 16× on A100. Anything a kernel does outside the tensor core is
expensive, and it got relatively more expensive with each generation.

## 2.4 Worked example: the exponential budget of attention

Attention forward does, for each query *q*, each key *k* and each head:

- `S = q·k`: 2*d* FLOP on the tensor core,
- `P = exp(S − max)`: one exponential on MUFU,
- `O += P·v`: 2*d* FLOP on the tensor core.

So with head dimension *d* = 128 there are 4*d* = **512 tensor FLOP per exponential**. To keep
the tensor core saturated, the exp unit must deliver

```
exps needed per clock per SM = (tensor FLOP/clk/SM) / 512
```

| GPU | tensor FLOP/clk/SM (BF16) | exps needed | `MUFU.EX2` per clk/SM | exp unit busy |
|---|---|---|---|---|
| A100 | 2029 | 4 | 16 | 25% |
| H100 *(preset)* | 4096 | 8 | 16 *(assumed)* | 50% |
| B200 *(half the B300 EX2 rate)* | 8192 | 16 | 16 | **100%** |
| B300 | 8192 | 16 | **32** | 50% |

At B200's rate the exp unit is exactly as fast as the tensor core needs, with zero slack for
the max, scale, sum and convert instructions around it. This is why FlashAttention-4 computes
part of its exponentials with a polynomial on the FMA pipe on B200. B300 doubled `MUFU.EX2`
(and only EX2: the other MUFU functions stayed at 16), and FA4's tuning table turns the
emulation off on sm_103 (`ex2_emu_freq = 0`).

The simulator reproduces this from the recorded FA4 kernel: halving B300's EX2 rate slows
FA4 to **0.79×**, and doubling it again gains nothing (1.00×). A faster tensor core alone
stops paying at 1.21× because the exp unit becomes the limit again; doubling both gives 1.51×
([chapter 3](03-kernels-sass-cycles.md), [`research/flash_attention/`](../../research/flash_attention/)).

## 2.5 The memory hierarchy

Measured with pointer chases (one dependent load at a time):

| level | A100 | B300 |
|---|---|---|
| shared memory | 29 cycles | 29 cycles |
| L1 hit | 39 cycles | 39 cycles |
| L2 hit, near partition | 270 cycles | 320 cycles |
| L2 hit, far partition | 422 cycles (47% of lines) | 706 cycles (25% of lines) |
| HBM | 560 cycles = 402 ns | 850 cycles = 418 ns |
| DSMEM (another SM in the cluster) | n/a | 193 cycles |

Both chips split L2. On A100 an SM sees two halves about 150 cycles apart, and which half is
"near" repeats every 14 SM ids. On B300 the far partition is on the other die. A kernel whose
working set fits in L2 still pays the far latency on a quarter of its lines, which is why the
simulator models L2 per die with a die-to-die hop.

Bandwidths per SM per clock: shared memory serves **128 B** (32 banks × 4 B; measured 127),
L1 about 80 B, and L2 **59 B on B300** (17.8 TB/s for the chip) and 19 B on A100 (2.8 TB/s).

**Bank conflicts** are exact: an access where *n* threads of a warp hit the same bank takes
*n* times as long (measured 1.0, 2.0, 4.0 ... 32.0×), and padding the stride to 33 words
brings it back to 1.0×.

### Little's law: why you need bytes in flight

To sustain bandwidth *β* against latency *λ*, a chip needs `β · λ` bytes in flight. For B300
HBM that is 6.82 TB/s × 418 ns ≈ **2.85 MB**, or about **19 KB per SM**.

The calibration measured what a plain load loop achieves as more SMs join:

| SMs | 1 | 16 | 64 | 128 | 148 |
|---|---|---|---|---|---|
| B300 read GB/s | 72 | 589 | 2287 | 4319 | 4830 |

The loop holds about 15 KB per SM in flight, so it tops out at 4.8 TB/s, 71% of the
achievable 6.82. A TMA pipeline with four 16 KB stages per SM keeps 64 KB in flight and
reaches **7.19 TB/s**. This is the main reason modern kernels use TMA and deep pipelines: not
for peak bandwidth per request, but to have enough requests outstanding.

### TMA

The Tensor Memory Accelerator copies a tile from global to shared memory on one
instruction, signals an mbarrier when the bytes have arrived, and frees the issuing warp
immediately. Measured B300 latency for one bulk copy:

| bytes | 128 | 1 KB | 4 KB | 16 KB | 64 KB |
|---|---|---|---|---|---|
| from L2 | 311 cycles | 315 | 386 | 482 | 875 |
| from HBM | 855 cycles | 945 | 1029 | 1293 | 1770 |

A 64 KB copy costs only 2.8× a 128 B copy from L2. The fixed cost dominates small copies, so
kernels move big boxes.

## 2.6 Three generations of tensor core

### Ampere `mma.sync` (HMMA)

A warp-wide instruction. All 32 threads supply fragments of *A*, *B* and the accumulator *C*
from **registers**, and get *D* back in registers. `mma.sync.m16n8k16` with BF16 inputs is
16 × 8 × 16 × 2 = 4096 FLOP.

Measured on A100: 0.495 warp-MMAs per clock per SM (one per SMSP every 8 cycles),
**2029 FLOP/clk/SM**, 24-cycle latency. Every operand passes through the register file, and
the scheduler issues one instruction per MMA. To feed it, a kernel loads tiles into shared
memory with `cp.async`, then into registers with `ldmatrix` (`LDSM`), then issues HMMA.

Blackwell still runs `mma.sync`, at **2037 FLOP/clk/SM**, the same per-SM rate as A100 and a
quarter of its own `tcgen05`. A kernel written for Ampere (FlashAttention-2 for example) runs
on B300 on this legacy path. That is why FA2 tops out near 0.43 PFLOP/s on B300 while FA4
reaches 1.6 to 1.9 (chapter 3).

### Hopper `wgmma` (HGMMA)

A **warpgroup** (4 warps, 128 threads) issues one asynchronous MMA of shape m64nNk16, with *B*
(and optionally *A*) read directly from shared memory through a descriptor and the
accumulator in registers. The warps keep running; `wgmma.wait_group` (SASS
`WARPGROUP.DEPBAR`) waits for the result. The H100 preset runs it at 4096 FLOP/clk/SM.
*We have not measured a Hopper GPU, so the timings here are preset.*

### Blackwell `tcgen05.mma` (UMMA)

One **thread** issues the MMA. *A* and *B* come from shared memory (or *A* from tensor memory)
through descriptors, the accumulator lives in **TMEM**, a 256 KB per-SM memory that is not
the register file, and completion is signalled by `tcgen05.commit` arriving on an mbarrier.
The issuing warp does nothing else. Results come back to registers with `tcgen05.ld` only when
the epilogue (or the softmax, in attention) needs them.

Measured on B300, one CTA per SM issuing back to back:

| instruction shape | cycles per MMA | FLOP/clk/SM | share of peak |
|---|---|---|---|
| BF16 m128 n256 k16 | 128 | 8192 | 100% |
| BF16 m128 n128 k16 | 64 | 8192 | 100% |
| BF16 m128 n64 k16 | **48** | 5461 | 67% |
| BF16 m64 n256 k16 | 128 | 4096 | 50% |
| FP8 m128 n256 k32 | 128 | 16384 | 100% |
| NVFP4 m128 n256 k64 | 128 | 32768 | 100% |
| NVFP4 m128 n256 **k96** (sm_103) | **144** | 43691 | 89% of 49152 |
| NVFP4 m256 n256 k96, 2-SM (`cta_group::2`) | 128 | **49152** | 100% |

Round-trip latency (issue to commit visible) is 426 to 435 cycles for every shape.

**Worked example: why m128n64 runs at two-thirds speed.** A UMMA reads its operands from
shared memory, and shared memory serves 128 B per clock per SM (section 2.5). Count the bytes:

| shape | *A* bytes | *B* bytes | cycles at 128 B/clk | cycles at full tensor rate | slower one |
|---|---|---|---|---|---|
| BF16 m128n256k16 | 4 KB | 8 KB | 96 | 128 | tensor (128) |
| BF16 m128n128k16 | 4 KB | 4 KB | 64 | 64 | tie (64) |
| BF16 m128n64k16 | 4 KB | 2 KB | **48** | 32 | **shared memory (48)** |
| NVFP4 m128n256k96 | 6 KB | 12 KB | **144** | 128 | **shared memory (144)** |
| NVFP4 m256n256k96, 2-SM, per SM | 6 KB | 6 KB | 96 | 128 | tensor (128) |

Every measured shortfall matches the shared-memory port to the cycle. Narrow tiles starve the
tensor core because *A* has to be re-read for fewer output columns. The sm_103 FP4 K=96
instruction is only 1.33× the K=64 rate on one SM, but the 2-SM form splits *B* between the
pair, halves the per-SM operand traffic and reaches the full **1.5×**. This is our reading of
the measurements rather than a documented fact, but it predicts all five rows exactly.

The m64n256 row is different: it is not starved (10 KB in 128 cycles), it simply uses half of
the 128-row datapath.

**The capacity of TMEM.** TMEM is 128 lanes × 512 columns × 4 B = 256 KB. One m128n256 FP32
accumulator is 128 KB, exactly half, so a GEMM can double-buffer its accumulator: the MMA warp
fills one half while the epilogue drains the other.

### Why the interface changed

Look at what each generation asks of the warp scheduler for the same work. For a 128 × 256 ×
16 BF16 product (1 MFLOP):

| | instructions issued | operand path |
|---|---|---|
| Ampere `mma.sync` m16n8k16 | 256 HMMA, plus the `LDSM` that feed them | shared → registers → tensor core |
| Hopper `wgmma` m64n256k16 | 2 HGMMA (one per warpgroup, 4 warps each) | shared → tensor core, accumulator in registers |
| Blackwell `tcgen05` m128n256k16 | 1 UMMA, from one thread | shared (or TMEM) → tensor core → TMEM |

With 8192 FLOP/clk/SM, issuing the Ampere way would need two HMMA per clock per SM plus their
loads, half the SM's issue bandwidth, and a register file large enough to hold the operands of
four warps. Moving operands out of the register file and the issue out of every warp is what
made the 4× jump possible. The cost is that kernels must be rewritten as asynchronous,
warp-specialised pipelines (chapter 3).

## Exercises

1. A softmax-like kernel does 3 FFMA, 1 `MUFU.EX2` and 1 `F2FP` (BF16 pack) per element. On
   B300, which pipe limits it, and at what elements per clock per SM?
2. A decode attention kernel on B300 reads the KV cache with plain loads, 8 warps per SM, each
   warp keeping one 512 B load outstanding. What bandwidth will it reach?
3. Why does a 64-row (M=64) decode GEMM on Blackwell waste half the tensor core, and what can
   a kernel do about it?
4. FA2 on B300 uses `mma.sync`. What is the most TFLOP/s it could reach?

<details><summary>Answers</summary>

1. FFMA: 128 per clock, so 3 FFMA per element allows 43 elements per clock. EX2: 32 per clock.
   F2FP: 64 per clock. EX2 limits it at **32 elements per clock per SM**, or 9.6 × 10<sup>12</sup>
   per second for the chip.
2. 8 × 512 B = 4 KB in flight per SM, 600 KB for the chip. 600 KB / 418 ns ≈ **1.4 TB/s**,
   about a fifth of HBM. It needs more warps, wider loads, or TMA.
3. `tcgen05` has a 128-row datapath; an M=64 MMA uses half of it (the m64n256 row above). A
   kernel can split K across CTAs (split-K, which the simulator uses for decode GEMMs), swap
   operands so the large dimension is M, or batch more tokens.
4. 148 × 2.032 GHz × 2037 = **613 TFLOP/s**. The measured best is about 0.44 PFLOP/s (72%).

</details>

## Reproduce

| number | where |
|---|---|
| all pipe rates and latencies | `calibration/results/<gpu>/calibration.json`, `model.sass.<OPCODE>` |
| memory latencies, L2 partitions, bank conflicts, HBM vs SMs | `model.memory` |
| TMA curves | `model.tma` |
| tensor-core table | `model.tensor` |
| exp what-ifs on FA4 | `research/flash_attention/results/study.json` (`whatif`) |
| exp what-ifs on the synthetic attention kernel | `python examples/gb300_whatifs.py` |
| how the suite measures each number | [`calibration/README.md`](../../calibration/README.md) |

Next: [3. From a kernel to SASS to cycles](03-kernels-sass-cycles.md).
