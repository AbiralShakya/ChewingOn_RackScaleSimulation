# Dissecting Blackwell Ultra (B300, sm_103) for cycle-level simulation

This document records what we measured on one B300 SXM6 (della-tridao GPU 3,
driver 610.57, CUDA 13.3) to move `rsim` from "kernel totals within 7%" toward
per-instruction accuracy, and what changed in the simulator because of it.
Everything here is reproducible with `calibration/dissect/` (instruction-level
experiments on patched cubins) and `calibration/src/bench_tc05x.cu`,
`bench_tlb.cu`, `bench_dvfs.cu` (new benches in the suite).

The short version:

* **A kernel-total error hides much larger per-warp errors.** On straight-line
  microkernels the simulator matched ncu totals while its per-warp timelines
  between clock64 probes were off by 30 to 50 percent. FA4 is the target that
  matters, so the headline check is FA4's per-role timelines. Those are now
  measured inside the real kernel, and the per-role error dropped from
  **4.1% to 2.8%** of each warp's run time (section 9).
* Things that were not modelled and now are: register-file banks, operand
  reuse, the yield bit, the tcgen05 issue queue depth, and TMEM ld/st timing
  by shape (section 8).
* Things that were measured and need no new model: TLB reach (flat to
  16 GB), operand-slot-independent ALU latency (4 cycles), stall counts
  honoured exactly (1 to 15 with yield, 1 to 11 without).

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

Per SMSP, warp-instructions per cycle at saturation (8 warps per SMSP):

| pipe / op | cycles per warp-instruction | note |
|---|---|---|
| FFMA, FMUL, FADD (no bank conflict) | 1 | two 16-lane FMA pipes |
| IMAD, LOP3, SHF, HFMA2, I2FP, F2FP | 2 | 16 lanes |
| MUFU.EX2 | 4 | twice the other MUFU functions |
| MUFU.RSQ / TANH, POPC, F2I | 8 | |
| SHFL | 4 | per SMSP, independent of warp count |
| HMMA m16n8k16 BF16 | 8 | 2037 FLOP/clk/SM |
| DFMA | 64 | 1.2 TFLOP/s chip |

A single warp issues an FMUL every cycle at stall 1. HMMA interleaved with
FP32 work is expensive: HMMA+FMUL pairs take ~15 cycles per pair per SMSP
against 8 for HMMA alone, so every FP32 instruction between HMMAs costs
about 7 cycles of tensor-pipe time (A100: about 4). This is the softmax
penalty an FA2-style kernel pays on B300.

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

Measured, 8 warps per SMSP, cycles per instruction:

| FFMA sources | cycles | FMUL sources | cycles |
|---|---|---|---|
| 3 in one bank | 3.0 | 2 in one bank | 2.0 |
| 2 + 1 | 2.0 | 1 + 1 | 1.0 |
| 2 + 1 with a reuse hit | 1.0 | | |
| a == b, plus c in the other bank | 1.0 | | |
| alternating banks between instructions | 1.5 | | |

Without reuse, an FFMA can never issue faster than every 2 cycles on an
SMSP. ptxas marks reuse aggressively for that reason. bench_alu's FFMA
throughput (124 FLOP/clk/SM) is reached only because its operands are
reuse-cached.

**Simulator:** `issue.rf_banks = 2`, `issue.reuse_cache = true`; per-bank
read queues per SMSP, deduplicated reads, a per-warp reuse cache, and a new
warp state `dispatch` (Nsight's `dispatch_stall`) while a bank is busy.

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

## 9. Validation

### 9.1 FA4 per-role timelines (the headline)

`calibration/dissect/fa4/` measures timelines inside the real FA4 kernel
without editing its source. `fa4_probe.py` wraps every pipeline method FA4
uses (consumer_wait, producer_acquire, producer_commit, consumer_release, for
every pipeline class) so that each warp stores (%clock64, tag) after each
call. Records go to a per-warp ring indexed by `clock64 >> 4`, so no counter,
atomic or loop-carried state is needed. The probes cost FA4 about 3%.

`fa4_compare.py` replays the NVBit recording of the same configuration
(b=4, h=16, s=4096, d=128, BF16, non-causal) and compares each warp's
progress through its synchronisation events with hardware. "Timeline error"
is the mean |t_sim(q) − t_hw(q)| over q = 5%..100% of a warp's events, as a
fraction of its hardware run time. CTAs 0 to 3, 116 warps.

| role | end, before | end, after | timeline error, before | timeline error, after |
|---|---|---|---|---|
| MMA | −5.2% | −2.9% | 2.6% | 0.9% |
| load (TMA) | −5.6% | −3.2% | 1.6% | 0.7% |
| softmax (2 warpgroups) | −7.8 / −5.7% | −5.4 / −3.3% | 3.9 / 2.5% | 2.7 / 1.2% |
| correction | −5.7% | −3.3% | 2.4% | 1.1% |
| all warps | | | 4.1% | 2.8% |

FA4_V2_PLACEHOLDER

The epilogue warp still ends ~37% early in the replay. It never gates the
kernel, but its per-tile waits release too early. Replaying its waits with
try_wait.parity semantics made that worse (−48%), so the cause is elsewhere,
probably the TMA store's wait_group.read.

### 9.2 Microbenchmark replay and reference set

HARNESS_PLACEHOLDER

## 10. Prior art

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

## 11. Reproducing

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
