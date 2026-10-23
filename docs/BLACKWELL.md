# Dissecting Blackwell Ultra (B300, sm_103) for cycle-level simulation

This document records what we measured on one B300 SXM6 (della-tridao GPU 3,
driver 610.57, CUDA 13.3) to move `rsim` from "kernel totals within 7%" toward
per-instruction accuracy, and what changed in the simulator because of it.
Everything here is reproducible with `calibration/dissect/` (instruction-level
experiments on patched cubins) and `calibration/src/bench_tc05x.cu`,
`bench_tlb.cu`, `bench_dvfs.cu` (new benches in the suite).

The short version:

* **A kernel-total error hides much larger per-warp errors.** The simulator
  matched ncu totals on straight-line microkernels while its per-warp
  timelines between clock64 probes were off by 30 to 50 percent per segment.
  FA4 is the kernel that matters, so the headline check is FA4's per-role
  timeline, measured inside the real kernel (section 10.1). There, *where*
  each role's events happen matches hardware to **1.6%** of its run time,
  and the sim runs about 3% fast overall.
* **What changed in the simulator:**
  * register-file banks (2 banks; register index mod 2);
  * operand reuse, served from any slot;
  * the yield bit;
  * the tcgen05 issue queue depth (7 MMAs);
  * TMEM ld/st timing by shape;
  * the tcgen05 clock offset (1905 MHz whenever UMMA runs);
  * loose round-robin warp scheduling.

  All of it sits behind `model.issue` and `model.power.tcgen05_sm_mhz` in
  calibration.json, which `calibration/dissect/fit_issue.py` writes from the
  measurements.
* **Effect on the existing checks:**
  * B300 reference set: `smem_conflict_free` goes from +10.0% to +6.8%.
    (Later: −0.8%, once the SMSPs take turns at the MIO; signed errors for
    every kernel are in docs/CALIBRATION.md.)
  * FlashAttention on B300: mean |error| over 8 kernels goes from 3.9% to 3.3%.
  * Per-warp microkernel timelines: segment error goes from 30% to 24%.
* **Measured and found not to need a model:**
  * TLB reach is flat to 16 GB.
  * ALU latency is 4 cycles whichever operand slot carries the dependence.
  * Stall counts are honoured exactly: 1 to 15 with yield, 1 to 11 without.

## 1. Method

### 1.1 Patching cubins

`calibration/dissect/cubin.py` reads and writes single instructions of a CUDA
ELF (128-bit Volta+ encoding). An experiment compiles a small kernel with
nvcc, finds the instructions of interest in its `cuobjdump -sass` listing, and
rewrites fields in place: the control bits, register operands, the reuse
flags, and the register allocation (`EIATTR_REGCOUNT` in `.nv.info` and the top
byte of the text section's `sh_info`). The patched image is loaded through the
driver API from Python (`cudrv.py`, ctypes, no other dependencies). This is
the approach of maxas, turingas and CuAssembler, without needing an assembler
for sm_103: every field we touch was verified by patching it and
re-disassembling (`check_fields.py`).

Field positions confirmed on sm_103 (bit i of the little-endian 128-bit word):

| field | bits | notes |
|---|---|---|
| Rd / Ra / Rb / Rc | [16:24) / [24:32) / [32:40) / [64:72) | Rb is a register only in the register form |
| stall | [105:109) | |
| yield | 109 | 1 = *no* yield (ptxas prints `Y` when it is 0) |
| write barrier | [110:113) | 7 = none |
| read barrier | [113:116) | 7 = none |
| wait mask | [116:122) | SB0..SB5 |
| reuse | [122:126) | slots a, b, c, d (verified by flipping each bit and reading `.reuse` back) |

### 1.2 Three ways to read a latency or a cost

1. **Correctness under a patched stall.** Fixed-latency instructions have no
   hardware interlock. Setting the stall count of every instruction of a
   dependent chain to *s* and checking the result against the unpatched
   kernel gives the smallest *s* whose answer is still right. That is the
   true pipeline latency, independent of what ptxas assumes
   (`stall_sweep.py`, kernels in `kernels/chain.cu`). The chains have to be
   non-degenerate: `ex2(ex2(x))` saturates and `x^a^b` repeats with period
   two, so those chains report nothing.
2. **Timing between clock reads.** One or more warps run a straight-line
   block between two `%clock` reads (`issue_sweep.py`, `pair_sweep.py`,
   `rfport.py`, `ifetch.py`). Loads before the window must complete before the
   first clock read. The kernels store their loaded values to shared memory,
   barrier, **and read the store back**: without the read-back ptxas deletes
   the store and the load latency (about 300 cycles) lands inside the window.
   That cost us a false finding, a "+1 cycle per instruction" that was really
   300 cycles spread over 256 instructions (`plusone.py` is the bisection that
   found it).
3. **Random designs and regression.** Register-bank behaviour came from 400
   blocks of 256 instructions with random register assignments, regressed on
   candidate conflict features (`regfile.py`, `regfile_fit.py`). The best bank
   function out of 15 candidates was register index mod 2 (R² 0.89 to 0.97).

GPU 3 is shared with other users' jobs (a vLLM server was resident during
this work). They can slow a whole block down about 2× for milliseconds at a
time, so timings are taken as the minimum over several rounds spread over the
run.

### 1.3 Mining the compiler's latency table

ptxas encodes every fixed-latency dependence as stall counts, so the tightest
producer→consumer stall distance over a large corpus is ptxas's latency for
that opcode. `mine_latency.py` over the 5,508 sm_100a functions in
`libcublasLt.so` (CUDA 13.3, 6M instructions):

| producer | ptxas latency | | producer | ptxas latency |
|---|---|---|---|---|
| FFMA, FADD, FMUL, FFMA2, HFMA2 | 4 | | HADD2.F32, I2FP.F32.S32 | 5 |
| IMAD, IADD3, LOP3, LEA, SHF, PRMT, SEL, MOV, VIMNMX, FMNMX | 4 | | IMAD.WIDE | 3 to 4 |
| ISETP, FSETP (to a predicate consumer) | 4 (13 to a branch) | | DFMA, DMUL (sm_100a) | 7 to 8 |
| UIADD3, ULEA, UISETP | 4 | | R2UR | 13 |

DFMA is fixed latency on sm_100 (B200's FP64 is full rate) but scoreboarded
on sm_103: B300's FP64 runs at 1/64 rate.

## 2. Control codes

ptxas never emits a stall above 11 or a stall of 0 with the yield bit set
(over 6M instructions). The hardware does something specific with those
combinations, so they are not free encodings. Independent FMULs, one warp,
cycles per instruction:

| stall field | yield bit 1 (no yield) | yield bit 0 (yield) |
|---|---|---|
| 0 | ~33 (pipeline drain) | ~35 |
| 1 | 1 | 2 |
| 2 .. 11 | s | s |
| 12, 13 | behave as 1 | s |
| 14, 15 | ~33 | s |

* A stall count is honoured exactly from 1 to 15 (with yield for 12 to 15),
  for every pipe measured (FMA, ALU, XU, tensor), down to 1 cycle per
  instruction for one warp (`ifetch.py`: 1.01 cycles per FMUL at stall 1).
* Yielding costs a cycle only when the warp would otherwise issue on the
  next cycle.
* Back-to-back MUFU at stall 1 without yield is pathological: a 64-op chain
  took ~25 ms instead of microseconds (repeatable). ptxas never schedules it.
* Dependent HMMA (mma.sync m16n8k16) needs 15 cycles between issues; ptxas
  uses 20 by following `HMMA … S11` with a predicated-off `@!UPT UIADD3 URZ`
  carrying `S09`, a dummy uniform instruction whose only job is to stall.

**Simulator:** `issue.yield_min_stall = 2`. Control codes were already
honoured; the stall-12..15/yield and stall-0 rules need nothing because ptxas
never emits them.

## 3. Fixed-latency pipelines

From correctness under patched stalls (`stall_sweep.json`):

| op (chain) | true latency | ptxas |
|---|---|---|
| FFMA (dependence in slot a, b or c), FFMA2, FADD, FMUL | 4 | 4 |
| HFMA2 (f16, bf16) | 4 | 4 |
| IMAD, I2FP, VIMNMX, PRMT, SEL, LOP3 (same pipe) | 4 | 4 |
| FADD↔FMNMX, IMAD↔LOP3 (FMA pipe ↔ ALU pipe) | 5 | 5 |
| HMMA.16816.F32.BF16, accumulator | 15 | 20 |

Forwarding across the FMA and ALU pipes costs one extra cycle; ptxas knows
(stall 5 in those chains), so a simulator that honours stall counts gets it
for free.

## 4. Issue rates

One warp issues an independent FMUL every cycle at stall 1 (1.01 cycles per
instruction over 1024 instructions, `ifetch.py`). There is no extra per-warp
issue interval: an apparent "+1 cycle per instruction" in early sweeps turned
out to be load latency inside the timed window (section 1.2). Pipe rates per
SMSP, from bench_alu throughput (many warps):

| pipe / op | cycles per warp-instruction per SMSP |
|---|---|
| FFMA, FMUL, FADD (operands not bank-conflicted) | 1 |
| IMAD, LOP3, SHF, HFMA2, I2FP, F2FP | 2 |
| MUFU.EX2 | 4 (twice the other MUFU functions) |
| MUFU.RSQ / TANH / RCP, POPC, F2I | 8 |
| SHFL | 4 |
| HMMA m16n8k16 BF16 | 8 (2037 FLOP/clk/SM) |
| DFMA | 64 (1.2 TFLOP/s chip) |

## 5. Register file and operand reuse

* **Two banks**, register index mod 2. Each bank delivers one 32-bit-per-lane
  operand per cycle. Bank behaviour does not depend on the register range
  (R20 vs R140) or on the allocation size.
* Reads **queue per bank across instructions and warps**: two instructions
  that each read three registers from opposite banks overlap (1.5 cycles each,
  not 3).
* **A register named twice in one instruction is read once.**
* The **reuse cache** holds one operand per slot. An operand read in the same
  slot by the previous instruction of that warp, with that slot's reuse flag
  set, is not read from the register file.

Measured on clean FMUL streams (`bankcheck.py`, stall 1, load-free window):

| sources | 1 warp | 8 warps per SMSP |
|---|---|---|
| a, b in different banks | 1.06 | 1.07 |
| a, b in the same bank | 2.05 | 2.05 |
| same bank, a reuse-cached | 1.08 | 1.08 |
| a == b (one register read once) | 1.06 | 1.07 |

FA4-style code shows that the reuse cache is not tied to the slot. FFMA pairs
that swap the reused registers between slots a and c on every instruction
(`FFMA R5, R3.reuse, R5, R2.reuse` / `FFMA R6, R2.reuse, R6, R3.reuse`) issue
one per cycle on hardware, where a per-slot cache would cost a second cycle
on every instruction.

Without reuse, an FFMA whose three sources land 2+1 in the banks can never
issue faster than every 2 cycles. ptxas marks reuse aggressively for that
reason. bench_alu's FFMA throughput (124 FLOP/clk/SM) is reached only because
its two constant operands are reuse-cached.

**Simulator:** `issue.rf_banks = 2`, `issue.reuse_cache = true`,
`issue.reuse_any_slot = true`. Each SMSP has per-bank read queues, reads are
deduplicated, each warp has a reuse cache, and a new warp state `dispatch`
(Nsight's `dispatch_stall`) covers the time a bank is busy.

## 6. Instruction fetch

Straight-line FMUL blocks, one warp, warm caches, stall 1:

| code size | cycles per instruction |
|---|---|
| ≤ 16 KB (1024 instructions) | 1.0 |
| 32 KB | 1.5 |
| 64 KB | 2.7 |
| 128 KB | 3.0 |
| 256 KB | 5.1 |

The first (cold) launch costs ~2.5 cycles per instruction even for tiny
kernels. With 4 warps per SMSP the larger sizes are partly hidden. Loop
kernels such as FA4 fit easily, so the simulator does not model fetch; a
straight-line kernel above ~16 KB would need it.

## 7. Blackwell-specific units

### 7.1 tcgen05.mma (UMMA)

One thread issues k back-to-back MMAs (M=128, BF16 K=16 / FP8 K=32 /
NVFP4 K=64), then commits and waits (`bench_tc05x pipe`):

| N | cycles per MMA (steady) | completion of k MMAs after the first issue |
|---|---|---|
| 256 | 128 | 294 + 128 k |
| 128 | 64 | ~275 + 64 k |
| 64 | 52 (issue-bound) | ~190 + 52 k |

* The issue loop of a single thread costs ~48 cycles per UMMA (descriptor
  setup, R2UR, the UTC*MMA itself). That makes N=64 issue-bound: 52 cycles per
  MMA instead of the 48 the tensor core needs.
* **Shape model.** Every one of the 19 measured shapes (`calibration.json`
  `tensor.tcgen05*`: SS and TS operands, 1- and 2-SM, BF16 to NVFP4, K=96)
  is the larger of two limits. The first is math at the per-SM rate, with
  each SM's slice of M padded to 128 rows, so M64N256K16 takes 128 cycles,
  not 64. The second is operand reads: A (unless it is in TMEM) plus this
  SM's half of B (the pair multicasts B) at 128 B/clk of smem. That makes
  M128N64K16 48 cycles, not 32, and 1-SM K=96 NVFP4 144, not 128, while
  2-SM K=96 reaches the full 1.5x. **Simulator:** `TensorSpec::umma_cycles`.
  It uses `tensor.umma_smem_bytes_per_clk = 128` and
  `umma_lds_interference = 0.107`; the second constant is fitted to the LDS
  contention point in section 10.1. The shape comes from the opcode for synthetic
  kernels and from each MMA's instruction descriptor for recordings
  (`mma` in the `.rpath` header). Older recordings infer it from the FLOPs.
* **The issuing thread blocks once 7 MMAs are in flight**, whatever their size
  (N=256: 7; N=128: 6). The simulator used to allow a 4096-cycle backlog, which
  is ~32 N=256 MMAs. **Simulator:** `issue.tcgen05_queue_depth = 7`.
* **Concurrent TMEM loads don't interfere.** 256 MMAs took 33,030 cycles with
  and without four warps streaming `tcgen05.ld` from another TMEM region (and
  the loads lost 3% bandwidth).

### 7.2 TMEM loads and stores

Latency of `tcgen05.ld` + `wait::ld` for one warp, and per-SM bandwidth with
16 warps (`bench_tc05x tmem`):

| shape (SASS) | B per warp per repeat | latency | per-warp rate | SM bandwidth |
|---|---|---|---|---|
| 32x32b (`LDTM.xN`) | 128 | 41 + B/256 | 256 B/clk | ~940 B/clk (x64) |
| 16x64b (`LDTM.16dp64bit`) | 128 | 41 + B/128 | 128 B/clk | ~500 B/clk |
| 16x128b (`LDTM.16dp128bit`) | 256 | 41 + B/128 | 128 B/clk | ~512 B/clk |
| 16x256b (`LDTM.16dp256bit`) | 512 | 41 + B/64 | 64 B/clk | 256 B/clk |

Stores (`tcgen05.st` + `wait::st`) follow the same pattern with a base of
~14.5 cycles and 265 B/clk per warp for 32x32b (16x64b and 16x128b at half,
16x256b at a quarter). SM bandwidth is four times the per-warp rate: each
warp reaches only its own lane quadrant, so each sub-partition has its own
TMEM port. FA4 uses `LDTM.x16`, `LDTM.STAT.x32` and `STTM.x16`, the fast
shape.

**Simulator:** `issue.tmem` (`ld_base 41`, `st_base 14.5`, `ld_rate_32x32b 256`,
`st_rate_32x32b 265`). Each SMSP has a TMEM port server, and the shape is
parsed from the SASS opcode.

### 7.3 tcgen05.cp, alloc

* `tcgen05.cp` costs 193 cycles per instruction from one thread whatever the
  shape (128x256b, 128x128b, 4x256b), and 322 cycles including commit and
  wait. It is issue-bound.
* `tcgen05.alloc`: 243 cycles warm, and `dealloc` 201, for 32 to 512 columns.
  The 1430-cycle `tmem_alloc_latency` fitted earlier to FA4's prologue window
  therefore includes more than the allocation.

## 8. Memory system

* **TLB: nothing to model.** A dependent chase with one L2-resident line per
  page stays at 300 to 311 cycles per hop from 0.5 MB to 16 GB spans (strides
  64 KB, 2 MB, 32 MB, up to 8192 pages). Past 512 MB at 2 MB stride it costs
  ~10 cycles more.
* **Static SASS has no addresses.** Replaying a kernel from its listing alone
  makes every global load a cold DRAM miss. The memory timelines (`t_global`,
  `t_smem`) therefore need NVBit address recordings
  (`calibration/dissect/record_paths.py`), and the issue-model comparisons
  leave them out. The loaded-latency and L2 near/far results are in
  docs/CALIBRATION.md.

## 9. Clocks

`bench_dvfs` keeps every SM busy for 2 s. Eight SMs sample (%globaltimer,
%clock64) every ~10 µs, giving the SM clock in 100 µs bins. %globaltimer
advances in ~1 µs steps.

| load (all 148 SMs) | SM clock, first 100 µs | over 700 ms |
|---|---|---|
| idle (samplers only) | 2032 MHz | 2032 MHz |
| FFMA | 2032 MHz | 2032 MHz |
| tcgen05 BF16 (1 SM, N=256) | 1905 MHz | 1905 MHz |
| tcgen05 NVFP4 K=64 | 1905 MHz | 1905 MHz |
| tcgen05 NVFP4 K=96 | 1905 MHz | 1905 MHz |

Any tcgen05 UMMA load pins the clock at **1905 MHz**, 6.3% below the 2032 MHz
maximum. The clock is already there in the first 10 µs sample and holds for
the whole run, with no ramp and no dependence on dtype or power. This is a
fixed tensor-core clock offset, separate from the 1100 W power cap that
throttles large FP4 GEMMs further (docs/CALIBRATION.md, "Power-capped
clocks").

**Simulator:** `power.tcgen05_sm_mhz = 1905`. Any kernel that issues a tcgen05
MMA runs at most at that clock when cycles are converted to time; the power
cap can lower it further.

## 10. Validation

### 10.1 FA4 per-role timelines (the headline)

`calibration/dissect/fa4/` measures timelines inside the real FA4 kernel
without editing its source:

* `fa4_probe.py` wraps every pipeline method FA4 uses (consumer_wait,
  producer_acquire, producer_commit, consumer_release, for every pipeline
  class). After each outermost call, each warp stores (%clock64, tag).
* Records go to a per-warp ring indexed by `clock64 >> 4`, so no counter,
  atomic or loop-carried state is needed.
* The probed kernel runs ~1.6% slower (CTA 0 spans ~518K cycles; ncu
  measures 510K unprobed).

`fa4_compare.py` replays the NVBit recording of the same configuration
(b=4, h=16, s=4096, d=128, BF16, non-causal) in rsim. For each warp it
compares progress through the synchronisation events (probe stamps on
hardware; SYNCS/UTCBAR issues in the replay) at q = 5%..100% of the warp's
events. Two errors are reported, both as a fraction of the warp's hardware
run time:

* **Timeline error:** mean |t_sim(q) − t_hw(q)|.
* **Shape error:** the same with both curves normalised to their own end, so
  a uniform rate difference (such as the probes' cost) drops out.

Results cover CTAs 0 to 3 (116 warps), with the original model (main) and the
current one:

| role | end vs probed hw, main | now | timeline error, main | now | shape error, main | now |
|---|---|---|---|---|---|---|
| MMA | −4.0% | −4.8% | 1.9% | 1.8% | 0.5% | 0.8% |
| load (TMA) | −4.4% | −5.1% | 1.0% | 1.4% | 1.6% | 1.5% |
| softmax, warpgroup 0 | −6.6% | −7.4% | 3.3% | 3.7% | 1.0% | 0.9% |
| softmax, warpgroup 1 | −4.5% | −5.2% | 1.8% | 2.3% | 0.7% | 0.7% |
| correction | −4.5% | −5.2% | 1.7% | 2.1% | 0.9% | 0.8% |
| all warps | | | 3.4% | 3.8% | 1.6% | 1.6% |

What this says:

* **FA4's structure is right.** Where each role spends its time and when its
  barriers fire matches hardware to ~1% of the run in every role.
* **The end-time gap is which SMs are compared, not rate.** The SMs do not
  finish together, on hardware or in the simulator. ncu's per-SM
  `sm__cycles_elapsed` runs from 494.8K to 510.2K (avg 503.5K); the
  simulator's per-SM finish times are 496.2K / 504.6K / 511.2K, within 0.3%
  at the minimum, the average and the maximum. But they order the SMs
  differently. Hardware CTAs 0–7 all span 517–519K probed, so they are among
  the slowest SMs. In the simulator CTAs 0–3 finish first: identical work
  takes 490.7K on SM 2 and 504.6K on SM 147. The lag builds at each tile's
  first barrier wait, behind the HBM and L2 queues, which serve requests in
  the order they are booked; CTAs launch one cycle apart and stay in that
  order, so the low SMs are always first in the queue (with HBM and L2
  bandwidth ×10 the spread drops from 14K to 4–6K cycles). Comparing
  hardware's slow CTAs with the simulator's fast ones gives the −5% above. The
  earlier reading, "the replay runs ~3% ahead after the probes' cost", rested
  on the probe cost being 1.6%, which was itself worked out by assuming CTA 0
  is the slowest SM. The probe cost and hardware's SM order are not
  measured, so only the shape error says anything about the per-role
  timelines.
* **FA4 is not issue-bound.** The issue-stage model barely moves it. In the
  replay its MMA warp sits in `tensor_throttle` (the 7-deep UMMA queue) about a
  third of the time, so FA4 is tensor-bound there.
* **A candidate for the 3%: shared-memory contention.** An N=128 BF16 UMMA
  reads 128 B/clk of smem operands. With other warps streaming LDS.128 at
  60 B/clk, it slows from 65.2 to 68.4 cycles (+5%); STS traffic and N=256 are
  not affected (`bench_tc05x smem`). FA4's QK GEMM is N=128 with both operands
  in smem, so concurrent smem reads (the TMA store of O, correction-warp
  loads) can take a few percent. **Now modelled, and it does not explain the
  3%.** FA4 issues `cta_group::2` MMAs: each SM reads its own 128 rows of Q
  and half of K (the pair multicasts B), 6 KB per 64-cycle MMA, so the port
  sits at 75% and the measured LDS interference never makes the MMA
  smem-bound. All seven B300 FA4 kernels simulate to the same cycle with and
  without the operand model. It does not need to: the "3%" was the SM
  ordering above.

An earlier version of this table showed per-role errors falling from 4.1% to
2.8%. That came from a "minimum 2 cycles between a warp's instructions" rule,
which section 1.2 shows was a measurement artifact; it is not in the final
model.

The epilogue warp still ends ~37% early in the replay. It never gates the
kernel, but its per-tile waits release too early. Replaying its waits with
try_wait.parity semantics made that worse (−48%), so the cause is elsewhere,
probably the TMA store's `wait_group.read`.

### 10.2 Kernel totals

B300 reference set (`rsim validate`, ncu `sm__cycles_elapsed`), main vs final:

| kernel | main | final |
|---|---|---|
| ffma_throughput | −7.0% | −6.5% |
| ex2_throughput | +0.1% | +0.1% |
| dfma_latency_chain | +3.2% | +3.2% |
| **smem_conflict_free** | **+10.0%** | **+6.8%** |
| smem_32way_conflict | −0.3% | −0.3% |
| hbm_stream_read | +0.3% | +0.3% |
| hbm_copy | −5.1% | −5.1% |
| l2_pointer_chase | −3.7% | −3.7% |
| hmma_bf16_throughput | −0.1% | −0.1% |
| umma_bf16 / umma_nvfp4 | −0.1% / −0.0% | −0.1% / −0.0% |
| tma_bulk_stream | −6.2% | −6.2% |
| window_* (8 shapes) | ≤ 0.2% | ≤ 0.2% |

(Superseded: docs/CALIBRATION.md has the current signed error of every kernel.) smem_conflict_free
improves because round-robin scheduling spreads the 32 warps' LDS stream the
way the hardware does.

FlashAttention on B300 (research/flash_attention recordings):

| kernel | main | final |
|---|---|---|
| FA2 s4096 | −4.9% | +1.6% |
| FA4 s4096 | +0.2% | +0.4% |
| FA4 ring step c1024 kv8 / kv64 | −6.6% / −10.2% | −5.4% / −7.3% |
| FA4 ring step c2048 kv8 / kv64 | +4.9% / +1.9% | +6.1% / +2.1% |
| FA4 ring step c4096 kv8 / kv64 | +2.7% / +0.1% | +3.1% / +0.3% |
| mean \|error\| (8 kernels) | 3.9% | 3.3% |

### 10.3 Per-instruction and per-warp checks

* **Patched microbenchmarks replayed in rsim** (`harness.py`). The stall
  sweep covers 216 configurations of dependent chains at patched stalls: 175
  are within 5% (median 0.0%). The misses are DFMA (FP64 issue on sm_103),
  SHFL latency (the model has 27 cycles where the chains imply ~23) and
  IMAD.HI (two passes). The independent-issue sweep covers 187
  configurations: 178 are within 5% (median 0.1%).
* **Per-warp clock64 timelines** (`kernels/timeline.cu`, 1 to 32 warps,
  compute kernels t_alu / t_softmax / t_hmma, 15 configurations). Mean
  per-segment error goes from 30% to 24% and mean span error from 4.0% to
  3.2%. The best change is round-robin scheduling: under the old greedy
  policy each warp's own segments ran back to back instead of interleaving.
  Single-warp timelines are exact to within 1% on t_alu and t_hmma.
* **Memory timelines need real addresses.** With NVBit recordings and a warm
  L2, t_smem spans are within −0.2% to +11.5%, against −50% from static SASS.
  t_global is still +26% to +49%: a small hot L2 working set costs ~380
  cycles per dependent hop on hardware and ~600 in the model, which charges
  the far-partition first-touch latency too often. This is the next memory
  item.

## 11. Prior art

* Z. Jia, M. Maggioni, B. Staiger, D. Scarpazza, *Dissecting the NVIDIA Volta
  GPU Architecture via Microbenchmarking* (2018); Z. Jia, M. Maggioni, J. Smith,
  D. Scarpazza, *Dissecting the NVidia Turing T4 GPU via Microbenchmarking*
  (2019). Control-code layout, register banks and the reuse cache on Volta and
  Turing; our sm_103 layout matches theirs bit for bit.
* S. Gray, *maxas* (Maxwell assembler); D. Yan, *turingas*; *CuAssembler*.
  Patching control codes and registers in cubins.
* R. Huerta, M. Abaie Shoushtary, J.-L. Cruz, A. González, *Analyzing Modern
  NVIDIA GPU Cores* (2025). Issue logic, stall counters, the yield bit and the
  register-file/reuse design of recent NVIDIA cores, reverse-engineered for
  simulation. Our Blackwell measurements agree on the structure (dependence
  through compiler stalls, a small reuse cache, banked operand reads).
* H. Abdelkhalik et al., *Demystifying the Nvidia Ampere Architecture through
  Microbenchmarking and Instruction-level Analysis* (2022); W. Luo et al.,
  *Benchmarking and Dissecting the Nvidia Hopper GPU Architecture* (2024);
  A. Jarmusch et al., *Dissecting the NVIDIA Blackwell Architecture with
  Microbenchmarks* (2025). Per-instruction latencies and tensor-core rates on
  the previous parts and on B200.
* M. Khairy et al., *Accel-Sim* (2020); O. Villa et al., *NVBit* (2019).
  Trace-driven simulation and binary instrumentation; the warp-path recorder in
  research/flash_attention is built on NVBit.

## 12. Reproducing

```bash
# on the GPU node (GPU 3 is shared: every run takes the lock)
cd calibration/dissect && ./build.sh                      # kernels/*.cu -> build/*.cubin + .sass
flock /home/as0714/.gpu3.lock python3 stall_sweep.py build/chain.cubin build/chain.sass out.json
GPU_LOCK=/home/as0714/.gpu3.lock python3 isolate.py pair_sweep.py build/pair.cubin build/pair.sass out.json
flock ... python3 rfport.py build/issue_mw.cubin build/issue_mw.sass out.json
flock ... python3 timeline.py build/timeline.cubin timeline_hw.json
cd ../ && make && ./build/bench_tc05x all && ./build/bench_tlb && ./build/bench_dvfs 2000
# FA4 probes (flash_attn.cute environment, research/flash_attention/remote/env.sh)
python calibration/dissect/fa4/run_fa4_probe.py fa4_probe.json

# locally
python3 calibration/dissect/harness.py --sass build/ --hw results/ --calibration <cal.json>
python3 calibration/dissect/timeline_compare.py timeline_hw.json build/timeline.sass out.json --calibration <cal.json>
python3 calibration/dissect/fa4/fa4_compare.py fa4_probe.json research/flash_attention/rpath/b300_fa4_s4096.rpath.gz out.json --calibration <cal.json>
```
