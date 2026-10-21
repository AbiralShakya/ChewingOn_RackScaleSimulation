# 1. Rooflines, and why the roof moves

A GPU kernel spends its time doing one of three things: arithmetic, moving bytes, or waiting
for something else to finish. The roofline model turns the first two into a single number per
kernel and tells you which one sets the time. This chapter derives it for the B300 and the
A100 from measured peaks, checks it against measured cuBLAS GEMMs, and then adds the part
that datasheet rooflines leave out: on a power-capped Blackwell, the compute roof itself drops
as a kernel gets larger.

## 1.1 Two ways to be slow

Take a kernel that performs *W* FLOP and moves *Q* bytes between HBM and the chip. With peak
compute *π* (FLOP/s) and HBM bandwidth *β* (bytes/s), it cannot finish faster than either

```
T_compute = W / π            T_memory = Q / β
```

If the two overlap perfectly (which a well pipelined kernel approaches), the time is

```
T ≈ max(T_compute, T_memory)
```

and if they do not overlap at all, it is the sum. The real answer sits between the two, and
much of this guide is about which side of that range a kernel lands on and why.

Divide through by *Q*. The ratio *I = W / Q* is the kernel's **arithmetic intensity** in
FLOP per byte. The kernel is compute bound when

```
I > π / β  ≡  I*      (the ridge point)
```

and memory bound below it. Achievable throughput is `min(π, I·β)`, the classic roofline.

## 1.2 The ridge points of our GPUs

We use measured numbers: SM count × measured clock × measured tensor FLOP per clock per SM
for *π*, and the measured HBM read bandwidth for *β*.

| GPU, precision | *π* | *β* | ridge *I** |
|---|---|---|---|
| A100 80GB PCIe, BF16 (`mma.sync`) | 108 × 1.395 GHz × 2029 = **306 TFLOP/s** | 1.70 TB/s | **180 FLOP/B** |
| B300, BF16 (`tcgen05`) | 148 × 2.032 GHz × 8192 = **2.46 PFLOP/s** | 6.82 TB/s | **361 FLOP/B** |
| B300, FP8 | 4.93 PFLOP/s | 6.82 TB/s | 723 FLOP/B |
| B300, FP4 (1-SM K=96 UMMA) | 148 × 2.032 GHz × 43691 = 13.1 PFLOP/s | 6.82 TB/s | 1928 FLOP/B |

Two things stand out. First, from A100 to B300 the BF16 peak grew 8× and the bandwidth 4×,
so the ridge doubled: a kernel needs twice the reuse per byte to stay compute bound on
Blackwell. Second, lower precision moves the ridge further out. Halving the bytes per
element doubles *I* for the same math, but the tensor rate doubles too, so in units of
*reuse per element* the bar stays where it was.

## 1.3 The GEMM, worked

Nearly all the FLOPs in a transformer are matrix multiplies of an activation *A* (*M* × *K*)
by a weight *B* (*K* × *N*), with *M* the number of tokens in the step. In BF16:

```
W = 2·M·N·K
Q = 2·(M·K + K·N + M·N)       (read A and B once, write C once)
```

When *M* is much smaller than *N* and *K*, the weight dominates *Q*:

```
I ≈ 2·M·N·K / (2·K·N) = M        FLOP per byte, in BF16
```

**The arithmetic intensity of a weight-stationary GEMM is about the number of tokens that
share each weight.** That one line explains most of inference (chapter 6): a decode step with
*M* = 64 tokens has *I* ≈ 63, far below either ridge, so it runs at HBM speed no matter how
fast the tensor cores are.

Setting *I* = *I** gives the **critical batch**: the number of tokens per weight read at which
a GEMM turns compute bound. For BF16 it is *M** ≈ *I**, so about **180 tokens on A100** and
**361 on B300**. With weights stored in *b* bytes per element and math at the matching
precision, `M* = I*·b/2`: 361 for FP8 on B300 and about 480 for FP4.

For the scaling book's TPU v5e the same number is about 240. B300 sits higher because its
compute grew faster than its memory.

## 1.4 Checking it against real GEMMs

The Mamba-vs-attention study timed cuBLAS BF16 GEMMs on both GPUs with CUDA graphs and a
cold L2 ([`research/ssm_vs_attention/results/validation_*.json`](../../research/ssm_vs_attention/results/)).
Here are three shapes per GPU, with the roofline prediction alongside:

| GPU | M × N × K | *I* (FLOP/B) | roofline | measured (min) | achieved |
|---|---|---|---|---|---|
| B300 | 1 × 28672 × 4096 | 1.0 | 34 µs (memory) | 41.0 µs | 5.7 TB/s, 84% of *β* |
| B300 | 128 × 18560 × 4096 | 123 | 23 µs (memory) | 30.5 µs | 638 TFLOP/s |
| B300 | 32768 × 6144 × 4096 | 2286 | 669 µs (compute) | 900 µs | **1.83 PFLOP/s, 74% of *π*** |
| A100 | 1 × 28672 × 4096 | 1.0 | 138 µs (memory) | 154 µs | 1.52 TB/s, 90% of *β* |
| A100 | 256 × 6144 × 4096 | 232 | 42 µs (compute) | 58.9 µs | 219 TFLOP/s, 72% of *π* |
| A100 | 4096 × 4096 × 8192 | 1638 | 899 µs (compute) | 1035 µs | **266 TFLOP/s, 87% of *π*** |

The roofline is a bound, and real GEMMs land 10 to 30% above it. The gap is the subject of
chapters 2 and 3: wave quantisation, pipeline fill and drain, tile shapes that waste part of
the tensor core, and on B300 a clock that drops under load (section 1.6).

Notice also that the largest B300 GEMMs reach about 74% of the measured peak while the A100
reaches 87%. Part of that is the power cap, which the A100 PCIe never hits in these runs.

## 1.5 The simulator's version

rackscale's synthetic GEMM is a TMA-fed, warp-specialised `tcgen05` kernel on Blackwell (and a
CUTLASS-style multistage `mma.sync` kernel on Ampere). It is simulated cycle by cycle on all
148 SMs. Sweeping *M* with *N* = *K* = 8192:

```bash
python docs/book/scripts/roofline_sweep.py            # B300, BF16, about 20 s
```

```
b300 bf16: peak 2464 TFLOP/s, HBM 6.82 TB/s, ridge 361 FLOP/byte
     M      AI roofline_us    sim_us  TFLOP/s  %peak    GB/s
     1     1.0        19.7      27.1        5    0.2    4958
    64    63.0        20.0      27.1      317   12.9    5034
   128   124.1        20.3      27.1      634   25.7    5111
   256   240.9        20.9      33.3     1033   41.9    4286
   512   455.1        27.9      40.9     1682   68.3    3696
  1024   819.2        55.8      78.8     1744   70.8    2129
  4096  2048.0       223.1     277.6     1980   80.4     967
  8192  2730.7       446.2     556.0     1977   80.2     724
```

Read it like a roofline plot. Up to *M* = 128 the time does not move at all (27.1 µs): the
kernel streams the 134 MB weight at 5.0 TB/s and the tensor core is mostly idle. Past the
ridge (*M* ≈ 361) time grows linearly with *M* and throughput flattens at 70 to 80% of peak.
The knee in the simulator sits where the algebra puts it.

The simulator is not perfect here. Against the measured B300 GEMMs it is within about 10% at
decode shapes (*M* ≤ 64) and 15 to 35% *fast* on large square-ish GEMMs, because the real
cuBLAS kernel loses time the model does not have yet. On A100 its multistage `mma.sync` GEMM is
30 to 90% *slow* at large *M* (the synthetic Ampere GEMM reaches about 55% of peak, cuBLAS
reaches 87%). Chapter 9 has the full comparison. For this chapter the point is the shape of
the curve, and the shape is right on both GPUs.

## 1.6 Why the roof moves: the power cap

A datasheet roofline assumes a fixed clock. A B300 board has an 1100 W limit, and a large
FP4 GEMM can draw more than that at the full 2032 MHz, so the GPU lowers its clock until it
fits. The calibration suite measured three CUTLASS NVFP4 GEMMs (K=96 UMMA) while sampling
board power and SM clock every 100 ms:

| GEMM (FP4) | median clock | board power | measured | peak at that clock | efficiency |
|---|---|---|---|---|---|
| 4096³ | 2032 MHz | 1012 W | 4.49 PFLOP/s | 13.1 PFLOP/s | 34% |
| 8192³ | 1717 MHz | 1064 W | **6.44 PFLOP/s** | 11.1 PFLOP/s | 58% |
| 16384³ | 1455 MHz | 1077 W | 4.42 PFLOP/s | 9.4 PFLOP/s | 47% |

The biggest GEMM is *slower* than the middle one. It keeps the tensor cores busier, which
costs more power per clock, so the clock has to fall further.

The simulator models this with the usual DVFS approximation: dynamic power scales as
frequency times voltage squared, and voltage tracks frequency, so

```
P(f) = P_idle + D · (f / f_max)³
```

Each measured run gives the *D* its kernel needs: `D = (P_board − P_idle) / (f/f_max)³`. With
the measured idle of 261 W that is 751 W for 4096³, 1331 W for 8192³ and 2224 W for 16384³.
For any other FP4 kernel the simulator interpolates *D* in log(FLOPs) between the runs, solves
`P_idle + D·(f/f_max)³ = 1100 W` for the sustained clock, and stretches only the tensor-bound
part of the kernel by `f_max/f` (memory time is unchanged). For 8192³ that gives 2032 ×
(839/1331)<sup>1/3</sup> = **1742 MHz**, against 1717 MHz measured.

```bash
python docs/book/scripts/roofline_sweep.py --dtype fp4 --n 16384 --k 16384 --ms 1,256,1024,4096,16384
```

```
b300 fp4: peak 13142 TFLOP/s, HBM 6.82 TB/s, ridge 1928 FLOP/byte
     M      AI roofline_us    sim_us  TFLOP/s  %peak    GB/s  clock
     1     4.0        19.7      26.8       20    0.2    5015 2032.0
   256   949.8        21.2      27.2     5047   38.4    5314 2032.0
  1024  3120.8        41.8     107.9     5094   38.8    1632 1924.0
  4096  7281.8       167.3     401.8     5473   41.6     752 1764.0
 16384 10922.7       669.3    1715.5     5128   39.0     469 1643.0
```

So the honest roofline for B300 FP4 has a ceiling that slopes down with kernel size, from 13.1
PFLOP/s for small kernels to about 10.6 at the clock the simulator predicts for 16384³ (9.4 at
the measured clock). The decode GEMMs of chapter 6 never reach it: they are memory bound and
leave the tensor cores idle, so they stay under the cap.

Only FP4 has measured power runs today, so the simulator throttles only FP4. BF16 kernels run
at the full clock in every simulation in this guide. That flatters BF16 a little; measuring a
BF16 and an FP8 GEMM under the power monitor would fix it.

## 1.7 What the roofline leaves out

Real kernels miss the roofline for reasons that are invisible at this level:

- **Latency, not bandwidth.** A kernel with too few bytes in flight cannot reach *β* even when
  it is memory bound (chapter 2.5 uses Little's law to size this).
- **Wave quantisation.** 2048 output tiles on 148 SMs is 13.8 waves, so the last wave runs
  with 80% of SMs idle.
- **Other pipes.** Attention needs one `exp` per score. When the exp unit (MUFU) is slower
  than the tensor core, the kernel is bound by neither memory nor tensor math (chapter 2.4).
- **Fixed costs.** Launch, prologue and epilogue: about 7000 cycles (3.5 µs) per kernel on
  B300, measured (chapter 3.6).

These are what a cycle-level simulator is for. The roofline tells you where to look; the
simulator tells you how much each of these costs.

## Exercises

1. **Decode on A100 vs B300.** A Llama-3-8B decode step at batch 32 reads about 16 GB of BF16
   weights. What is the lower bound on step time on each GPU, ignoring the KV cache? What batch
   would make the MLP GEMMs compute bound?
2. **FP8 weights.** If those weights are stored in FP8 and multiplied in FP8, what happens to
   the decode bound and to the critical batch on B300?
3. **The power cap.** Using `D` = 1331 W, what sustained clock would the 8192³ FP4 GEMM reach on
   a board capped at 1000 W? At 1400 W?
4. **Roofline vs measurement.** The B300 shape 128 × 18560 × 4096 measures 30.5 µs. What
   fraction of the HBM bandwidth does it achieve? What does that suggest about how many bytes
   are in flight?

<details><summary>Answers</summary>

1. A100: 16 GB / 1.70 TB/s = 9.4 ms. B300: 16 GB / 6.82 TB/s = 2.3 ms. The MLP GEMMs turn
   compute bound at about 180 tokens on A100 and 361 on B300 (section 1.3).
2. The bytes halve, so the bound halves (1.2 ms on B300). FP8 math is also twice as fast, so
   the critical batch stays about 361.
3. Budget = cap − 261 W. For 1000 W: 2032 × (739/1331)<sup>1/3</sup> ≈ 1670 MHz. For 1400 W the
   budget (1139 W) is below *D*, so 2032 × (1139/1331)<sup>1/3</sup> ≈ 1928 MHz; still throttled.
4. *Q* = 2·(128·4096 + 4096·18560 + 128·18560) ≈ 157 MB, so 157 MB / 30.5 µs = 5.2 TB/s, 76% of
   *β*. At 418 ns HBM latency that needs about 2.2 MB in flight across the chip, or 15 KB per
   SM; chapter 2 shows how a TMA pipeline gets there.

</details>

## Reproduce

| number | command |
|---|---|
| B300 / A100 peaks and ridge points | `calibration/results/*/calibration.json` (`model.tensor`, `model.memory.bandwidth`) |
| measured GEMM table | `research/ssm_vs_attention/results/validation_{b300,a100}.json`, `kind == "gemm"` |
| simulated BF16 sweep | `python docs/book/scripts/roofline_sweep.py` |
| simulated A100 sweep | `python docs/book/scripts/roofline_sweep.py --gpu a100` (about 4 minutes) |
| power-capped FP4 sweep | `python docs/book/scripts/roofline_sweep.py --dtype fp4 --n 16384 --k 16384` |
| one kernel in detail | `rsim kernel --gpu b300 gemm 8192 8192 8192` |

Next: [2. Inside one GPU](02-inside-the-gpu.md).
