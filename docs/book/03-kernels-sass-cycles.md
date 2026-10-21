# 3. From a kernel to SASS to cycles

A CUDA, Triton or CuTe kernel is compiled to PTX, a virtual ISA, and then by `ptxas` to SASS,
the machine code the SM actually runs. SASS is where performance is decided: which
instructions exist, in what order, and how long each one makes the next one wait. This chapter
reads SASS, explains how a warp-specialised Blackwell kernel is built out of it, and shows how
rackscale turns a SASS program into cycles on every SM of a rack.

## 3.1 Reading SASS

Dump any binary with `cuobjdump -sass` or `nvdisasm`. Here is the whole of a vector add for
sm_90, from [`examples/sass/vecadd_sm90.sass`](../../examples/sass/vecadd_sm90.sass):

```
/*0000*/   LDC R1, c[0x0][0x28] ;                 /* 0x00000a00ff017b82 */
                                                  /* 0x000fe40000000800 */
/*0010*/   S2R R6, SR_TID.X ;                     /* 0x0000000000067919 */
                                                  /* 0x000e280000002100 */
...
/*00a0*/   LDG.E R4, desc[UR4][R4.64] ;           /* 0x0000000404047981 */
                                                  /* 0x000ea8000c1e1900 */
/*00b0*/   LDG.E R9, desc[UR4][R2.64] ;           /* 0x0000000402097981 */
                                                  /* 0x000ea4000c1e1900 */
/*00c0*/   FADD R9, R4, R9 ;                      /* 0x0000000904097221 */
                                                  /* 0x004fca0000000000 */
/*00d0*/   STG.E desc[UR4][R2.64], R9 ;           /* 0x0000000902007986 */
```

Each instruction is 128 bits, printed as two 64-bit words. Since Volta, the top bits of the
second word are **control bits** that `ptxas` writes for the scheduler. There is no hardware
dependency check for fixed-latency results; the compiler tells the hardware how long to wait.

Shift the second word right by 41 and read the fields:

| bits | field | meaning |
|---|---|---|
| 0–3 | stall | cycles before this warp may issue its next instruction |
| 4 | yield | hint that the scheduler may switch warps |
| 5–7 | write barrier | scoreboard (SB0–5, 7 = none) set when a variable-latency result is written |
| 8–10 | read barrier | scoreboard released when the source registers have been read |
| 11–16 | wait mask | scoreboards that must clear before this instruction issues |
| 17–20 | reuse | operand reuse cache flags |

Worked through for the four instructions that matter:

| instruction | control word | stall | yield | write SB | wait on |
|---|---|---|---|---|---|
| `LDC R1, ...` | `0x000fe40000000800` | 2 | yes | none | nothing |
| `LDG.E R4, ...` | `0x000ea8000c1e1900` | 4 | yes | **SB2** | nothing |
| `LDG.E R9, ...` | `0x000ea4000c1e1900` | 2 | yes | **SB2** | nothing |
| `FADD R9, R4, R9` | `0x004fca0000000000` | 5 | no | none | **SB2** |

Both loads set scoreboard 2, and the add waits on it, so the add cannot issue until both loads
have returned. That wait is what Nsight reports as `long_scoreboard`. The add's stall count
of 5 covers the 4-cycle FP32 latency, so the store that reads R9 next cannot issue too early;
fixed-latency results never use a scoreboard.

The simulator decodes the same fields:

```bash
rsim sass examples/sass/vecadd_sm90.sass --inspect
```

```
  00a0  LDG.E                    Mem S04 Y W2 R- wait000000
  00b0  LDG.E                    Mem S02 Y W2 R- wait000000
  00c0  FADD                     Fp32 S05   W- R- wait000100
```

and runs it on every SM:

```bash
rsim sass examples/sass/vecadd_sm90.sass --gpu h100_sxm --grid 1024 --block 256
```

```
  time        1.49 us   (2718 cycles, 1024 CTAs, 8 CTA/SM)
  dram        2.1 MB read, 1.0 MB written; L2 hit 0.0%
  scheduler-cycle anatomy:
    drain                 43.06%
    wait                  24.89%
    long_scoreboard       12.45%
    ...
```

A 3 MB kernel is dominated by its tail (`drain`: SMs that have finished waiting for the last
CTAs) and by fixed latencies. That is typical of small kernels and a reason to fuse them.

## 3.2 The anatomy of a modern kernel

The kernels that matter on Hopper and Blackwell are **warp specialised**: different warps of
one CTA run different code and hand work to each other. A Blackwell GEMM (CUTLASS's
`sm100` kernels, and rackscale's synthetic GEMM) looks like this:

```
 warp 0 (TMA)        warp 1 (MMA)                 warps 4-7 (epilogue)
 ------------        ------------                 --------------------
 wait EMPTY[s]       wait FULL[s]
 UTMALDG A,B → s     UTCHMMA (×K/16) → TMEM acc
   arrives FULL[s]   UTCBAR → EMPTY[s]            wait ACC
 s = s+1 mod stages  ...                          LDTM acc → regs
                     UTCBAR → ACC                 F2FP (pack BF16), STG
```

- **mbarriers** in shared memory carry the hand-offs. A TMA copy adds its byte count to the
  barrier's transaction count (`expect_tx`), and the barrier's phase completes when all bytes
  have landed. `tcgen05.commit` (SASS `UTCBAR`) arrives on a barrier when the MMAs queued
  before it retire.
- **Stages.** The shared-memory ring holds several (*A*, *B*) tiles, so TMA runs ahead of the
  MMAs and HBM latency is hidden (Little's law, chapter 2.5).
- **One thread issues the MMA.** The MMA warp issues about one instruction per 128 cycles. Its
  warps are almost always waiting, and that is correct.

So the right way to read a Blackwell kernel's stall profile is not "warps are stalled, bad".
In the simulated 8192³ BF16 GEMM on B300, 55% of scheduler cycles are `async_wait` (warps
parked on mbarriers by design), 20% `tensor_wait`, 15% `tma_wait`. The bubbles that cost time
are the `tma_wait` (the tensor core waiting on loads) and the GPU-level idle between kernels.

```bash
rsim kernel --gpu b300 gemm 8192 8192 8192
```

## 3.3 FlashAttention-2 vs FlashAttention-4 on the same B300

The [FlashAttention study](../../research/flash_attention/README.md) ran Dao-AILab's FA2 (CUDA,
`mma.sync`) and FA4 (CuTe DSL, `tcgen05`) on the B300. Measured forward throughput, BF16,
*d* = 128, 16 heads, non-causal, 16k tokens per batch:

| sequence | FA2 | FA4 | cuDNN (SDPA) | FA4 / FA2 |
|---|---|---|---|---|
| 1k (batch 16) | 408 TFLOP/s | 1257 | 1395 | 3.1× |
| 8k (batch 2) | 444 | 1801 | 1829 | 4.1× |
| 16k (batch 1) | 440 | **1888** | 1857 | 4.3× |

(One call per CUDA event pair, so host overhead is included. FA2 at 4k measured 227 TFLOP/s
in this sweep, against 430 at its neighbours; we treat it as an outlier of that run.)

Chapter 2 already predicts the FA2 ceiling: `mma.sync` on B300 is 2037 FLOP/clk/SM, 613
TFLOP/s for the chip, and FA2 reaches 72% of it. FA4 reaches 76% of the `tcgen05` peak. The
4.3× gap is not an FA2 design flaw; it is a 4× different tensor core.

**FA4's roles.** One CTA pair (a 2-SM cluster) splits into a TMA load warp, one MMA warp
issuing 2-CTA `tcgen05.mma`, two softmax warpgroups that read *S* from TMEM and write *P* back,
a correction warpgroup that rescales *O*, and an epilogue warp. From the replay: the tensor
core is busy **91%** of the kernel in the leader CTA. The softmax warpgroups spend about 40%
of their cycles waiting for *S*; the correction warpgroup is idle almost all the time because
FA4 rescales lazily (only when the running max grows enough to matter).

**The clock drops.** Nsight counted 510 k SM cycles in FA4's 290 µs, an effective 1.76 GHz
against the 2.03 GHz the chip holds on lighter kernels. FA4 keeps the tensor core busy enough
to hit the power limit. All kernel comparisons in this study are therefore in cycles.

## 3.4 How the simulator runs SASS

rackscale has three ways to get a kernel's instructions:

| source | what it is | when to use it |
|---|---|---|
| **synthetic** (`gemm`, `attention`, `row_norm`, ...) | SASS programs written by the simulator that mirror CUTLASS / FlashAttention structure | predicting models that have not been compiled |
| **`sass`** | a `cuobjdump` listing plus a branch profile; addresses from an address generator | simple kernels with static control flow |
| **`path`** (`.rpath`) | every instruction each warp of a few CTAs executed on real hardware, recorded with NVBit | warp-specialised kernels like FA3/FA4 |

Synthetic kernels are written as SASS text and go through the same parser as real listings,
so every path through the model is the same.

For each SM, the model keeps four SMSPs with their own warp lists and a GTO (greedy, then
oldest) scheduler. Each cycle a scheduler issues the first warp that passes the checks of
chapter 2.2 in order: control-bit stall and yield, the scoreboard wait mask, pipe availability
(issue interval and lane count per pipe, per opcode from `calibration.json`), memory queue
credits, then barrier, mbarrier, `cp.async` group and warpgroup-MMA dependencies. The first
failed check names the stall reason.

**Event skipping.** When no warp on an SM can issue, the SM jumps straight to the next cycle at
which something becomes ready. A warp waiting 850 cycles on HBM costs one event, not 850
steps, but every lost issue slot is still counted and attributed. This is what makes
cycle-level simulation of 148 SMs × 72 GPUs tractable.

**Who to blame for a bubble.** On a cycle when no warp issues, the stall is attributed to the
*critical warp*: the one that will become ready first. So the scheduler-cycle anatomy adds up
to 100% of the SM's issue slots.

**Below the SM**, loads go through a sectored L1, a per-die L2 with an FCFS bandwidth server
per slice and a die-to-die hop for far lines, then HBM. TMA copies reserve their bytes on an
mbarrier and stream through the same L2 and HBM. `tcgen05.mma` goes into an SM-wide queue
whose rate comes from `TensorSpec.flops_per_cycle_per_sm[dtype]` and the instruction shape;
`UTCBAR` arrives on its mbarrier `tcgen05_latency` cycles after the MMAs before it retire.

## 3.5 Recording what really ran: `.rpath`

A static listing cannot say which path each warp of FA4 takes, how many times it spins on an
mbarrier, or which TMA box it loads. So the FlashAttention study added an NVBit tool,
[`nvbit_warp_path`](../../research/flash_attention/nvbit_warp_path/), that records for a few
CTAs every instruction each warp executed, with the operands a timing model needs: lane
addresses, the mbarrier behind each `SYNCS` op, expect-tx byte counts, the result and parity
of every `try_wait`, TMA source boxes for every CTA, UMMA instruction descriptors and the
registers that name barriers. It also saves the 128-bit encodings, so the simulator uses the
compiler's real control bits.

The converter then **removes the spin loops** (a failed `try_wait`, the sleep and the retries),
turning every successful wait into "wait for phase *n* of barrier *b*". The simulator re-times
the pipeline itself instead of replaying the spins the hardware happened to do.

Validated against Nsight Compute, with each GPU's own calibration file and no per-kernel
parameter (shape 4 × 16 × 4096, *d* = 128):

| GPU | kernel | ncu cycles | sim cycles | error |
|---|---|---|---|---|
| A100 PCIe | FA2 | 3.42 M | 3.45 M | +0.7% |
| A100 PCIe | FA4 (Sm80 path) | 3.39 M | 3.41 M | +0.6% |
| B300 | FA2 (`mma.sync`) | 2.38 M | 2.47 M | +3.7% |
| B300 | FA4 (2-CTA `tcgen05`) | 510 k | 506 k | −0.9% |

The stall mix matches too. For the HMMA kernels both hardware and simulator put about half the
stalled warp-cycles on fixed-latency dependencies (`wait`) and a third on the math pipe. For
FA4, Nsight files a warp blocked in `try_wait` as `long_scoreboard` and the simulator as an
mbarrier wait; together with named barriers that bucket dominates on both sides.

## 3.6 Fixed costs: launch and the kernel window

A kernel also costs time before its first instruction and after its last. Measured on B300:

| | cost |
|---|---|
| back-to-back launches, per kernel | 1.71 µs |
| CUDA graph, per kernel | 0.55 µs |
| kernel window, 128 threads, no shared memory | 7093 cycles (3.5 µs) |
| kernel window, 512 threads, max shared memory, 2-CTA cluster, TMEM prologue (FA4-like) | 9110 cycles (4.5 µs) |

The **kernel window** is the in-kernel fixed cost: CTA launch, shared-memory carve-out, TMEM
allocation, descriptor prefetch, pipeline fill and drain. For a 290 µs FA4 call it is noise;
for the 32 µs ring-attention step of chapter 5 it is 14%. The simulator adds the measured
window to kernels timed alone (`rsim kernel`, `rsim validate`, `rsim path`).

## Exercises

1. Decode `0x000e640000000800`, the control word of `LDC R7, c[0x0][0x0]` in the vector add.
   Which scoreboard does it set, and which later instruction waits on it?
2. FA4 reaches 1.89 PFLOP/s at 16k tokens on B300, measured at an effective 1.76 GHz. What
   fraction of the `tcgen05` peak *at that clock* is that?
3. A decode step on B300 launches 600 small kernels. What do launches cost per step without
   and with CUDA graphs?
4. Why is "warps stalled 90% of the time" not a problem for the MMA warp of a Blackwell GEMM?

<details><summary>Answers</summary>

1. `0x000e640000000800 >> 41` has stall 2, yield, write barrier **SB1**, no wait. The `IMAD`
   at `0x0040` waits on mask `000011` (SB0 and SB1), so it waits for this `LDC` and for the
   `S2R` before it.
2. Peak at 1.76 GHz is 148 × 1.76 × 10<sup>9</sup> × 8192 = 2.13 PFLOP/s, so 1.89 / 2.13 ≈
   **89%**, against 76% of the 2.03 GHz peak.
3. 600 × 1.71 µs ≈ **1.0 ms** back to back, 600 × 0.55 µs ≈ **0.33 ms** with a graph. For a
   decode step of 1 to 3 ms that is the difference between a large and a small overhead.
4. It issues one UMMA per 128 cycles and nothing else. What matters is whether the tensor core
   is busy; that shows up as low `tma_wait` and high tensor utilisation, not as a low stall
   fraction for the issuing warp.

</details>

## Reproduce

| number | command |
|---|---|
| control bits | `rsim sass examples/sass/vecadd_sm90.sass --inspect` |
| vector add on H100 | `rsim sass examples/sass/vecadd_sm90.sass --gpu h100_sxm --grid 1024 --block 256` |
| GEMM stall anatomy | `rsim kernel --gpu b300 gemm 8192 8192 8192` |
| FA timing, validation, roles | `research/flash_attention/results/study.json`; write-up `research/flash_attention/writeup.html` |
| FA replay (needs the `.rpath` recordings, not in git) | `python research/flash_attention/run_study.py all` |
| launch and kernel window | `calibration.json`, `model.launch` |
| validation of microbenchmark kernels | `rsim validate calibration/results/b300-sxm6-ac-sm103/ncu_reference.json --gpu b300` |

Next: [4. Networks and collectives](04-networks-collectives.md).
