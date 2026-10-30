# 16. Stragglers, flaps and tail latency

Every chapter so far simulated a perfect machine. Every GPU ran each kernel in exactly the
modelled time, and every link ran at its full rate for the whole run. Real clusters break both
assumptions all the time. Clocks wander under the power cap, one GPU in a rack runs a few
percent slow, a 224G SerDes lane loses margin and its link flaps, and a hash puts two elephant
flows on one spine link. None of this changes the average much. What it changes is the
**maximum**, and a synchronous step waits for the maximum.

This chapter adds those effects to the simulator, gives each one a closed form, and checks
the two against each other. The simulator gained two inputs for it: `variability` (slowdowns
per GPU and per kernel) and `net.faults` (links that are dead, degraded or flapping). Both are
off by default, so no earlier number changes.

## 16.1 The slowest rank sets the step

Take a step of *P* phases. In each phase every one of *N* ranks runs a kernel, and then all of
them meet in a collective. If rank *r*'s kernel takes *t* (1 + σ |*Z*<sub>r</sub>|), with *Z*
a standard normal drawn anew for every kernel, the collective can't finish until the last rank
arrives. So the phase takes the maximum of *N* draws, and the step takes the sum of *P*
maxima:

```
E[step compute] = P · t · (1 + σ · E[max of N |Z|])
E[max of N |Z|] = ∫ (1 − erf(x/√2)^N) dx  ≈  √(2 ln 2N) for large N
```

One GPU on its own loses only the mean, σ · E|*Z*| = 0.80 σ. Seventy-two GPUs in lockstep
lose 2.64 σ, more than three times as much, and the tax keeps growing as √(ln *N*) however
many GPUs are added. Every number in the closed-form column below is `tail.straggler_overhead`.

The simulator checks it with a loop of 16 × (one 2048 × 4096 × 4096 GEMM, then a 1 MiB
all-reduce over *N* ranks) on GB300 NVL72. Each GEMM span (38.7 µs at the power-capped clock)
is slowed by 1 + 0.05 |*Z*|, and each row is the mean of three seeds.

```python
rs.simulate("gb300_nvl72", wl, fast=True, config={"variability": {"kernel_sigma": 0.05, "seed": 1}})
```

| ranks *N* | step, no jitter | step, jitter | added | added / compute | closed form σ · E[max<sub>N</sub>] | one GPU alone |
|---|---|---|---|---|---|---|
| 2 | 857.1 µs | 893.3 µs | 36.2 µs | 5.8% | 5.6% | 4.0% |
| 4 | 924.8 | 970.6 | 45.8 | 7.4% | 7.3% | 4.0% |
| 8 | 1013.2 | 1062.6 | 49.4 | 8.0% | 8.9% | 4.0% |
| 18 | 1037.7 | 1096.0 | 58.3 | 9.4% | 10.6% | 4.0% |
| 36 | 1033.1 | 1094.9 | 61.7 | 10.0% | 12.0% | 4.0% |
| 72 | 1029.6 | 1098.7 | 69.1 | 11.2% | 13.2% | 4.0% |

A 5% jitter that costs one GPU 4% costs a 72-GPU group 11%. Up to 4 ranks the simulator and
the formula agree within 0.2 points. From 8 ranks up, the simulator comes in 1–2 points under
the formula, because a collective is not an instantaneous barrier. The ranks that arrive early
post their buffers, and the all-reduce's fixed costs (launch, fences, the first protocol
steps among the ranks already present) run before the last rank shows up. About 15% of the
skew hides there, and the formula, which treats the collective as a wall, is the upper bound.

### Synchronise less often

The sum of maxima is larger than the maximum of sums. If each rank runs *K* kernels between
collectives, their jitters partly cancel before anyone has to wait. The same 16 GEMMs on 72
ranks:

| GEMMs between all-reduces | added / compute (simulated) | closed form |
|---|---|---|
| 1 | 11.2% | 13.2% |
| 4 | 7.8% | 8.2% |
| 16 | 5.6% | 5.9% |

This is the case for gradient accumulation, for overlapping the collective with the *next*
phase rather than ending each phase with one, and for pipeline schedules that don't
synchronise every micro-batch. It is also why the tail tax is a property of the *schedule*,
not just the hardware.

### One slow GPU

A fixed slowdown is worse than jitter, because it never averages out. With 72 ranks
synchronising every GEMM (60% of the step is compute):

| case | step | vs exact |
|---|---|---|
| all exact | 1029.6 µs | |
| rank 5 is 10% slow | 1082.3 µs | +5.1% |
| every GPU fixed at 1 + 0.02 \|*Z*\| | 1058.9 µs | +2.9% |

One GPU in 72 that is 10% slow costs the whole group about 10% of its compute time. With
the collective's slack, the step loses 5.1%. A 2% spread between GPUs, the kind of thing
silicon binning and coolant temperature produce, costs 4.7% of compute here against a closed
form of 0.02 × 2.64 = 5.3%. At rack scale, *the distribution of GPU speeds matters more than
its mean*. That's why large training jobs screen GPUs before a run and drain the slowest
nodes.

## 16.2 A link is a stripe, and the slowest stripe sets the collective

An NVL72 GPU reaches the NVSwitch trays over 18 links, one per switch chip. NVLink spreads a
GPU's traffic over all 18 by address, and NVLS reduces in the switches plane by plane. So a
collective is striped in lockstep across 18 planes, and its bandwidth term is set by the
slowest plane. Two closed forms follow:

```
one link removed from the stripe:       slowdown = 18 / 17 = 1.059
one link kept in the stripe at rate s:   slowdown = 1 / s   (on the plane's share)
```

```python
config={"net": {"faults": [{"a": "gpu0", "b": "nvsw0.0", "rate_scale": 0.5}]}}
```

A 64 MiB all-reduce on 72 GPUs, with one of GPU 0's links faulted:

| link gpu0 – nvsw0.0 | NVLS | vs healthy | ring | vs healthy |
|---|---|---|---|---|
| healthy | 174.4 µs | | 566.6 µs | |
| dead (never trained, routed around) | 181.8 | +4.2% | 566.5 | 0.0% |
| at half rate | 261.6 | **+50.0%** | 567.1 | +0.1% |
| down for 50 µs from t = 20 µs | 191.6 | +9.9% | 613.4 | +8.3% |

Three lessons follow, and none of them is obvious from a datasheet.

1. **A degraded link is worse than a dead one.** A dead link is removed from the stripe, and
   the other 17 carry its share: +4.2% against a closed form of +5.9%, where the difference is
   the part of the collective that isn't bandwidth bound. A link that retrains at half width
   stays in the stripe. Its plane takes twice as long, every GPU waits for it at every NVLS
   step, and a 3% loss of rack bandwidth becomes a 50% slower all-reduce. This is why
   fabric management that *disables* a marginal link beats one that keeps it, and why "zero
   flap" retimers and AECs are sold on stability more than on reach.
2. **The ring doesn't care about link bandwidth here.** With 20 SMs of NCCL channels, a ring
   all-reduce moves 64 MiB at about 230 GB/s per GPU, a quarter of the 900 GB/s, so losing
   1/18 of the links, or half of one, is invisible. It is SM and protocol bound (chapter 11.6),
   and a fault only hurts the algorithm that was using the bandwidth.
3. **A flap stalls everything behind it.** NVLink's address interleave keeps sending to the
   dead port, so the packets queued for it wait out the outage. A 50 µs flap costs the ring
   47 µs, almost all of it, and NVLS 17 µs, because the other planes keep moving and the
   stalled one drains afterwards. On real hardware, a flap also triggers link retraining (tens
   of microseconds to milliseconds) and, past the transport's retry budget, an error.
   InfiniBand's retry timeout in recent NCCL releases (`NCCL_IB_TIMEOUT` = 20) is
   4.096 µs × 2<sup>20</sup> ≈ 4.3 s per retry, so a flap shorter than that shows up as a
   stall rather than a crash. The simulator models the stall, not the crash.

## 16.3 Hashing, adaptive routing and a sick spine link

Scale-out fabrics have many equal-cost paths, and how a switch picks among them decides
what one bad link costs. Static ECMP hashes each flow onto one path for its lifetime.
Adaptive routing (InfiniBand AR, Spectrum-X, and the packet spraying in Ultra Ethernet)
picks, per packet, the output with the least queued and in-flight bytes.

The closed form for ECMP is balls into bins. A rail-optimised 8-node H800 fabric with 8
spines has 8 uplinks per leaf. An all-to-all over 64 GPUs without PXN sends 448 cross-rail
flows through each leaf, in each direction. Hashed at random, the busiest of those
16 × 8 links carries 1.34× its fair share (`tail.ecmp_max_load(448, 8, groups=16)`), so ECMP
should run the spine-bound part of the all-to-all about a third slower than a perfect spread.

A 32 MiB all-to-all, packet level, with `nccl.pxn: false` so cross-rail traffic crosses the
spines:

| link leaf0 – spine0 | adaptive | vs healthy | ECMP | vs healthy adaptive |
|---|---|---|---|---|
| healthy | 617.5 µs | | 779.8 µs | **+26%** |
| dead | 619.4 | +0.3% | 773.7 | +25% |
| at half rate | 647.9 | +4.9% | 1102.1 | **+78%** |
| down for 200 µs from t = 50 µs | 631.7 | +2.3% | 832.5 | +35% |

- **ECMP costs a quarter even when everything works.** The +26% is the hash collision of the
  closed form, diluted by the part of the collective that isn't spine bound.
- **Adaptive routing turns a fault into a rounding error.** The switch sees the sick link's
  queue grow and sends new packets elsewhere: +0.3% for a dead link, +4.9% for one at half
  rate, +2.3% for a 200 µs outage, which works out to about 7% of the outage itself.
- **ECMP pins flows onto the sick link.** Every flow hashed onto it stays there. At half rate
  those flows take twice as long, and so does the collective's tail: +78%. A dead link is
  re-hashed by the routing tables, so it costs nothing beyond the collisions ECMP already
  had.

This is the tail-latency argument of the article that motivated this chapter, made exact.
The longest packet of the collective sets its completion, and a fabric that can steer around
trouble keeps that tail short.

## 16.4 What sits underneath: SerDes, FEC and the cable

The faults above have physical causes, and those causes set their rates.

- **The symbol rate.** A 448 Gb/s lane with PAM4 (2 bits per symbol) runs at 224 GBd, so its
  Nyquist frequency is 112 GHz, where copper loses tens of dB per metre. PAM6 (≈ 2.58 bits per
  symbol) brings that down to about 87 GHz, but its eye is 1/5 of the swing instead of 1/3,
  4.4 dB less margin. Every option trades loss against noise margin, and a lane near its margin
  is the one that flaps.
- **FEC and replay.** The bandwidths in this book are after FEC: the link layer in
  `rsim-net/src/phy.rs` charges RS-FEC, CRC, credits and replay as a framing fraction fitted to
  NCCL's measured per-link rates (NVLink 5 delivers 40.1 of 50 GB/s for SM traffic). What the
  model doesn't have is the *variance*: a lane whose pre-FEC error rate rises costs extra
  latency first, then replays, then a retrain. In this chapter's terms that's a link at
  `rate_scale` below 1, or a short outage.
- **The cable.** Passive copper (DAC) has no retimer and the shortest reach. Active copper
  (ACC) adds equalisers, and active electrical cables (AEC) add retimers with clock recovery.
  Optics (AOC, and co-packaged optics) reach furthest at the highest power. Inside an NVL72,
  the copper spine between trays is passive. Bend radius and skew between the pair's two wires
  decide which links in a rack are marginal.

## 16.5 Congestion control: what is modelled, and what is not

The simulator's scale-out fabric is **lossless and credit based**, like InfiniBand. A port
sends only when the downstream buffer has room, so congestion spreads backwards as
head-of-line blocking (chapter 4.5), which is the same behaviour that PFC gives RoCE. What it
doesn't model is a **sender-side rate controller**:

- **DCQCN** (RoCEv2). Switches mark packets with ECN once a queue passes a threshold. The
  receiver returns congestion notification packets, and the sender cuts its rate
  multiplicatively, then recovers in stages. It trades queueing delay, so tail latency, against
  utilisation, and it is slow to react, with timers of tens of microseconds.
- **Ultra Ethernet Transport.** UET adds receiver credit-based congestion control (RCCC) and a
  sender-side, network-signal-based window (NSCC), and specification 1.1 adds in-band
  congestion signalling (CSIG) from switches on the path. Meta's MetaRoCE is a third design in
  the same space.

A faithful model of any of these needs per-flow pacing at the sender's NIC, ECN marking in
switch queues and a feedback loop with the right delays. That's the next piece of network
work, and we haven't shipped a version we could not check. Until then, read the simulator's
scale-out results as what a well-tuned lossless fabric does, which is the regime where DCQCN
keeps queues near its marking threshold. A badly tuned one can be much worse.

## 16.6 What to take away

| effect | closed form | simulated | knob |
|---|---|---|---|
| per-kernel jitter σ across *N* ranks | σ · E[max<sub>N</sub> \|Z\|]: 5.6% (*N* = 2) to 13.2% (72) at σ = 5% | 5.8% to 11.2% | sync less often, overlap the collective |
| one GPU 10% slow among 72 | 10% of compute | 8.5% of compute (+5.1% step) | screen and drain slow GPUs |
| one NVLink plane dead | 18/17 = +5.9% | +4.2% (NVLS) | disable marginal links |
| one NVLink plane at half rate | 2× on its share | +50% (NVLS), +0.1% (ring) | same; the algorithm matters |
| ECMP hash collisions, 448 flows / 8 links | 1.34× busiest link | +26% | adaptive routing, PXN |
| a spine link at half rate | 2× on its flows (ECMP) | +78% ECMP, +4.9% adaptive | adaptive routing |

The common thread: at scale you pay for the worst component, not the average one. Every
mechanism that helps, whether adaptive routing, disabling bad links, fewer synchronisation
points or screening out slow GPUs, works by keeping the maximum close to the mean.

## Exercises

1. A training step synchronises 200 times (one all-reduce per layer, forward and backward),
   and each phase's compute jitters by σ = 3% on 1024 GPUs. Roughly what fraction of compute
   goes to stragglers?

   *Answer:* E[max of 1024 |*Z*|] ≈ 3.4 (`tail.expected_max_half_normal(1024)`), so about
   0.03 × 3.4 = 10%, against 2.4% on one GPU.

2. Why does a dead NVLink cost NVLS less than the 18/17 of the closed form?

   *Answer:* Only the bandwidth term scales. The latency, launch and protocol costs of the
   all-reduce are the same on 17 planes as on 18.

3. A collective is striped across 8 planes and one is at 75% rate. What's the slowdown of
   its bandwidth term? What if the link were disabled instead?

   *Answer:* 1/0.75 = 1.33× kept, 8/7 = 1.14× disabled.

## Reproduce

| what | command |
|---|---|
| every table (about 10 minutes on 4 cores) | `python docs/book/scripts/tail_latency.py` |
| one experiment | `python docs/book/scripts/tail_latency.py stragglers` (or `sync`, `slow`, `nvlink`, `ib`) |
| the closed forms | `rackscale.tail`: `expected_max_half_normal`, `straggler_overhead`, `dead_plane_slowdown`, `degraded_plane_slowdown`, `ecmp_max_load` |
| jitter in any run | `config={"variability": {"kernel_sigma": 0.05, "gpu_sigma": 0.02, "slow_ranks": [[5, 1.1]], "seed": 1}}` |
| a link fault in any run | `config={"net": {"faults": [{"a": "gpu0", "b": "nvsw0.0", "rate_scale": 0.5, "start_us": 0, "duration_us": 0, "repeat": 1, "period_us": 0}]}}` |

Link endpoints are `gpu<rank>` or a switch label: `nvsw<base>.<i>` for NVSwitch chips,
`leaf<r>` and `spine<j>` in multi-node systems, `sw<j>` in explicit topologies. A fault from
time 0 for the whole run with `rate_scale: 0` is a link that never trained, and routing
leaves it out. Any other window with `rate_scale: 0` is a flap, and frames wait for it.

Back to [the ladder](README.md#the-ladder).
