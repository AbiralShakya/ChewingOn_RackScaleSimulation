# 17. Power, cooling and the clock

Every rung of the ladder shares one envelope: watts go in, and heat has to come out. Chapter 1
showed the first consequence. A B300 board has an 1100 W limit, a large GEMM wants more, and
so the GPU lowers its clock. This chapter follows those watts up the ladder, from one kernel
to the board, the rack's power trace, the energy storage that trace asks for, the voltage it
is delivered at and the coolant that carries it away. Then it comes back down, because the
coolant temperature sets the board limit, and the board limit sets the clock.

The simulator gained three things for it:

- every kernel's board power, from the measured power model;
- `Report["power"]`, the summed board power of all GPUs over time;
- `rackscale.power`, closed forms from the board to the facility.

The GPU numbers come from measured B300 power data. The facility constants (thermal
resistance, conductor resistance, coolant properties) are estimates. Each one is a parameter
and is marked as an estimate where it appears.

## 17.1 What one kernel draws

The power model of chapter 1, `P(f) = P_idle + D · (f / f_max)³`, gives the clock of a fully
tensor-bound kernel at the cap. Turned around, it gives any kernel's power:

```
P = P_idle  +  u · D(dtype, FLOPs) · (f / f_max)³  +  B_HBM · 8 · e_HBM      (capped at the board limit)
```

| term | B300 value | where it comes from |
|---|---|---|
| `P_idle` | 261 W | measured (board power with nothing running) |
| `D` | 751–2224 W (FP4), interpolated in log(FLOPs) | measured FP4 runs; BF16 derived from ncu cycles over CUDA-graph wall time (chapter 1.6) |
| `u` | tensor-core busy fraction / 0.7, at most 1 | the simulated kernel; the measured GEMMs sit at about 70% |
| `f` | the clock the kernel ran at | the power cap (chapter 1) or the tcgen05 clock |
| `e_HBM` | 4 pJ/bit | estimate (`power.hbm_pj_per_bit`); HBM3e devices are quoted at 3–4 pJ/bit |
| collectives | idle + 15% of (cap − idle) | estimate (`COLLECTIVE_DYNAMIC_FRACTION`): copy engines, NVLink SerDes, HBM |

Three kernels show the range:

- **A 4096³ BF16 GEMM** keeps the tensor cores busy, so `u` = 1. At the full 2032 MHz it would
  draw 261 + 1567 = 1.8 kW. The cap holds it at 1650 MHz and 1100 W, and it takes 91.6 µs
  instead of 74.1 µs (`rsim kernel gemm 4096 4096 4096 --gpu b300`).
- **An elementwise kernel streaming HBM** leaves the tensor cores idle. It draws idle plus
  its bandwidth times the energy per bit, 6.82 TB/s × 8 × 4 pJ = 218 W, so about 479 W.
- **A collective** draws about 387 W. A GPU waiting on a collective is mostly idle silicon
  with its NVLinks busy.

The same GPU therefore spans 260 W to 1100 W from one microsecond to the next. The rest of
the chapter follows from that range.

## 17.2 A rack's power over time

Run 72 GB300 GPUs through 8 phases. In each phase every GPU runs eight 4096³ BF16 GEMMs and
then a 64 MiB NVLS all-reduce, all in lockstep. `Report["power"]` bins the summed board power
into 1000 bins over the run.

```python
r = rs.simulate("gb300_nvl72", wl, fast=True)
t = power.rack_trace(r)                       # {"bin_us", "watts"}
power.smoothing(t["watts"], t["bin_us"], floor_fraction=0.9)
```

| quantity | value |
|---|---|
| step | 7131 µs (bins of 7.1 µs) |
| mean | 68.2 kW (947 W per GPU) |
| peak / trough | 79.2 / 25.7 kW |
| swing | 53.5 kW |
| storage to flatten it to its mean | 7.3 J |
| a floor at 90% of peak instead | burns 13.1% more energy |

One phase, sampled every 43 µs, in kW:

```
74 79 65 79 76 79 79 79 79 79 79 79 79 79 79 79 79 28 28 28 28
```

The GEMMs sit at the cap (72 × 1100 W = 79.2 kW). The all-reduce drops the rack to 28 kW,
because 72 GPUs go from full power to nearly idle together in one bin. A real board's power
changes more slowly, through its voltage regulators and firmware ramp limits. The simulator
switches power instantly, so the ramp it reports (7.5 MW/ms) is set by the bin width. The swing
is the number to use.

### How much storage

A step that alternates between a high level for a fraction *d* of each period *T* and a low
level has its mean *d* · swing above the low level. To flatten it, a store gives
(1 − *d*) · swing for *d* · *T* of every period:

```
buffer = d (1 − d) · swing · T
```

Here *d* = 0.79 (the GEMMs' share of the step) and *T* = 891 µs, so the closed form gives
8.0 J, against 7.3 J from the trace. The trace isn't quite a square wave: the first samples
of each phase dip below the cap.

Seven joules is nothing. The formula is linear in *T*, though, and a training step's phases
are not microseconds long. Take forward and backward for 1 s, then 0.25 s of gradient
reduction, optimizer and data loading, at the same swing. Then *d* = 0.8 and *T* = 1.25 s,
which is 0.16 × 53.5 kW × 1.25 s = **10.7 kJ per rack**. A thousand such racks swing by
53 MW in step, a few times a second, and turning a whole data centre's load up and down that
fast is what grid operators and on-site generators can't absorb. The options are the ones
the simulator can cost:

- **Store it.** Capacitors and batteries in the rack's power shelves hold the
  `d (1 − d) · swing · T` above, so the feed sees only the mean.
- **Burn it.** A power floor keeps the GPUs drawing a fraction of peak when they have no
  work. At 90% of peak this trace burns 13.1% more energy. The closed form is
  (floor − low)(1 − *d*) / mean = (71.3 − 28)(0.21) / 68.2 = 13%.
- **Overlap it.** Every technique in chapters 7 and 11 that hides communication under
  compute also raises *d* and shrinks the swing.

## 17.3 Trading clock for energy: the board limit

The same loop on one HGX B300 (8 GPUs), with the board limit lowered from 1100 W to 600 W:

```python
rs.simulate(power.with_power_cap("hgx_b300_8", 800), wl, fast=True)
```

| board limit | step | vs 1100 W | closed form | GPU power, mean | energy per step | TFLOP per J | vs 1100 W |
|---|---|---|---|---|---|---|---|
| 1100 W | 6887 µs | | | 967 W | 53.25 J | 1.32 | |
| 1000 W | 7116 | +3.3% | +3.5% | 886 | 50.45 | 1.39 | +5.5% |
| 900 W | 7467 | +8.4% | +7.8% | 806 | 48.16 | 1.46 | +10.6% |
| 800 W | 7829 | +13.7% | +13.0% | 725 | 45.38 | 1.55 | +17.3% |
| 700 W | 8297 | +20.5% | +19.7% | 642 | 42.61 | 1.65 | +25.0% |
| 600 W | 8905 | +29.3% | +28.8% | 558 | 39.77 | 1.77 | **+33.9%** |

The closed form needs no fitted constant. A kernel at the cap satisfies
`P_idle + D (f/f_max)³ = cap`, so its clock scales as (cap − *P*<sub>idle</sub>)<sup>1/3</sup>
whatever *D* is:

```
clock ratio  = ((cap − P_idle) / (1100 − P_idle))^(1/3)
energy ratio = (cap / 1100) / clock ratio
```

At 600 W the clock ratio is (339/839)<sup>1/3</sup> = 0.74, so the GEMMs take 35% longer.
They are 82% of the step, which gives +28.8%, within a point of the simulated +29.3%
(`power.cap_tradeoff`).

The cube is why perf per watt gets better as the cap comes down. Halving the dynamic budget
costs only a fifth of the clock. This trade decides how to fill a site whose power, not its
GPU count, is fixed. At 600 W each GPU runs the loop at 0.77 of its 1100 W speed, but a megawatt
feeds 1.73 times as many of them, so the site does **1.34 times the work**. Whether that is
worth it depends on what is scarce. When GPUs are the constraint, run them at the cap. When
megawatts are, run more GPUs at a lower one. Chapter 16 adds the counterweight: more ranks in
lockstep pay a larger straggler tax, and more of them have to communicate.

## 17.4 Decode lives under the cap

A decode step is memory bound (chapter 6), so its kernels leave the tensor cores mostly idle.
Llama-3-8B, TP8 on one HGX B300, batch 64, context 4096, all 32 layers:

| board limit | step | vs 1100 W | GPU power, mean / peak | tokens per J |
|---|---|---|---|---|
| 1100 W | 3507 µs | | 408 / 829 W | 5.59 |
| 800 W | 3518 | +0.3% | 393 / 767 | 5.78 |
| 600 W | 3528 | +0.6% | 381 / 600 | 5.96 |

The mean is 408 W, which is 37% of the limit. The closed form for a kernel streaming HBM is
idle + 218 W = 479 W, and the step falls below it because launches, norms and collectives
don't stream at full bandwidth. The peak is set by the attention and the split-K GEMMs, which
are partly tensor bound. Capping the board at 600 W slows them slightly, costs the step 0.6%
and saves 6% of the energy per token.

So a rack provisioned for training's 1100 W per GPU is provisioned for 2.7 times what this
decode step draws on average. This is the case for planning inference racks against their measured draw (power
oversubscription), and for separating prefill (compute bound, at the cap) from decode
(memory bound, well under it) in the power plan as well as the scheduler (chapter 12).

The clock model throttles on tensor power only. HBM power adds to the reported draw, but it
never lowers the clock, so a cap below idle + HBM power (about 480 W here) is reported at the
cap with no slowdown. The table stops at 600 W for that reason.

## 17.5 The coolant sets the clock

A cold plate moves heat from the die to the coolant through a thermal resistance *R*<sub>th</sub>
(die, thermal interface, lid and plate together). In steady state the junction sits at

```
T_junction = T_coolant + P · R_th
```

The GPU slows down when the junction reaches its limit *T*<sub>max</sub>. So the coolant
temperature sets a second board limit, `(T_max − T_coolant) / R_th`, and whichever of the two
is lower is the cap. With *R*<sub>th</sub> = 0.04 K/W and *T*<sub>max</sub> = 90 °C (both
estimates, typical of a 1 kW-class liquid-cooled GPU), on the loop of 17.3:

| coolant at the cold plate | board limit it allows | step | vs 25 °C |
|---|---|---|---|
| 25 °C | 1100 W (electrical) | 6887 µs | |
| 35 °C | 1100 W (electrical) | 6887 | |
| 45 °C | 1100 W (thermal: 1125 W) | 6887 | |
| 50 °C | 1000 W | 7116 | +3.3% |
| 55 °C | 875 W | 7550 | +9.6% |
| 60 °C | 750 W | 8039 | +16.7% |

Up to about 45 °C the electrical limit binds and the coolant temperature doesn't matter.
Past that point each kelvin takes 1/*R*<sub>th</sub> = 25 W off the budget, and the cube law
turns that into

```
d ln f / dT = −1 / (3 · R_th · (P − P_idle))  ≈  1.1% of clock per K at 1000 W
```

The simulator gives 1.2–1.3% of step per kelvin between 50 and 60 °C.

The temperature at the cold plate is the end of a chain:

```
T_coolant at the plate = T_facility water + CDU approach + rise along the rack loop
```

The CDU's approach temperature is how far its secondary loop (the one that reaches the
racks) sits above the facility water feeding it, a few kelvin for a good heat exchanger. The
rack loop then warms as it passes each cold plate. Warmer facility water lets the site cool
with outside air instead of chillers, which saves facility energy, and every kelvin of that
saving, approach or rise lands at the cold plate. Once the thermal limit is the one that
binds, it costs about 1% of throughput per kelvin. The numbers above rest on two
estimates, but the shape doesn't: a flat region, then a cliff whose slope is
1/(3 *R*<sub>th</sub> (*P* − *P*<sub>idle</sub>)).

## 17.6 Getting the watts in: 54 V or 800 V

A rack drawing *P* through a conductor path of resistance *R* at voltage *V* carries current
*I* = *P*/*V* and loses *I*²*R*. At fixed copper the loss falls as 1/*V*², so 800 V loses
(800/54)² = 219 times less than 54 V, or can use 219 times less copper for the same loss.
For a 140 kW rack and 0.5 mΩ round trip (an estimate):

| feed | current | conduction loss |
|---|---|---|
| 54 V | 2593 A | 3.36 kW (2.40%) |
| 400 V | 350 A | 0.06 kW (0.04%) |
| 800 V | 175 A | 0.02 kW (0.01%) |

Today's NVL72 racks take AC into power shelves that convert to 54 V DC on a busbar. At
140 kW that busbar already carries 2.6 kA. A 1 MW rack would need 18.5 kA at 54 V, against
1.25 kA at 800 V, which is why the next generation of racks moves to 800 V DC. That move
brings problems the simulator doesn't model, in engineering that's well understood:

- **Fault protection.** DC current has no zero crossing, so an arc doesn't put itself out the
  way it does on AC. An 800 V bus with large capacitors can deliver an enormous fault current
  in microseconds. Protection splits the bus into zones, each with solid-state breakers that
  trip in microseconds, so a fault in one tray doesn't take down its neighbours.
- **Grounding.** An IT system (isolated from earth) keeps running after one insulation fault
  and relies on an insulation monitor to find it before a second fault does. A TN-S system
  (one point bonded to earth) trips on the first fault. The choice trades availability for
  simplicity.
- **Conversion.** Solid-state transformers convert medium-voltage AC to 800 V DC at high
  frequency, which replaces a 60 Hz transformer and a rectifier stage with one smaller box.

## 17.7 Getting the heat out: flow and CDUs

Liquid carries heat at *ṁ* = *P* / (*c*<sub>p</sub> Δ*T*). For PG25 (25% propylene glycol
in water; *c*<sub>p</sub> ≈ 3950 J/kg·K and 1.02 kg/L near 30 °C, handbook values), for the
same 140 kW:

| coolant rise across the rack | flow | per kW |
|---|---|---|
| 5 K | 417 L/min | 2.98 L/min/kW |
| 10 K | 208 L/min | 1.49 L/min/kW |
| 15 K | 139 L/min | 0.99 L/min/kW |

A larger rise needs less flow, so smaller pipes and pumps, but by 17.5 the last cold plate
on the loop then runs warmer and has less headroom. A 2 MW CDU serves 16 such racks when 85%
of each rack's heat goes to liquid, and the rest goes to air (`power.racks_per_cdu`). Two-phase
cooling (the coolant boils on the plate, so it absorbs heat at nearly constant temperature)
and immersion change *R*<sub>th</sub> and the rise. In this chapter's terms they move the
cliff of 17.5 to a higher coolant temperature.

## 17.8 What is modelled, and what is not

| | modelled | estimate or not modelled |
|---|---|---|
| kernel power | the measured idle and FP4 power runs; BF16 runs derived from measured clocks; tensor utilisation and clock from the simulated kernel | HBM energy per bit (4 pJ/bit, est); collective power (est); FP8 has no power runs, so FP8 kernels draw only idle + HBM |
| the clock | power-capped DVFS on tensor power, re-simulating the kernel at its clock (chapter 1) | HBM power never lowers the clock; no thermal time constant (the junction follows power instantly) |
| GB300 | the calibrated B300 die at its measured 1100 W board limit | GB300 boards are specified with a higher limit, which `gpu.power.cap_w` can set |
| the rack trace | GPU boards, summed, binned at makespan / 1000 | Grace CPUs, NVSwitch trays, NICs, fans and pumps, conversion losses; voltage-regulator slew |
| the facility | closed forms in `rackscale.power` | the 800 V protection and grounding of 17.6 (described, not computed) |

A GB200 or GB300 NVL72 rack is quoted at roughly 120–140 kW in total. The GPU boards in 17.2
draw 79 kW at the 1100 W limit, and the rest of the rack is in the right-hand column.

## 17.9 What to take away

| quantity | closed form | simulated | knob |
|---|---|---|---|
| board power of a kernel | idle + *u* *D* (*f*/*f*<sub>max</sub>)³ + *B* · 8 · *e*<sub>HBM</sub> | 1100 W (GEMM) to 387 W (collective) | the dtype, the kernel's arithmetic intensity |
| storage to flatten a step | *d* (1 − *d*) · swing · *T* | 8.0 J vs 7.3 J | overlap communication (raise *d*), shorter phases |
| board limit 1100 → 600 W | clock × (339/839)<sup>1/3</sup> = 0.74 | +29.3% step, +33.9% work per joule | cap low when megawatts are scarce, high when GPUs are |
| a decode rack | idle + HBM power ≈ 480 W | 408 W mean, 829 W peak | oversubscribe inference power |
| coolant past the knee | −1/(3 *R*<sub>th</sub> (*P* − *P*<sub>idle</sub>)) ≈ −1.1%/K | −1.2 to −1.3%/K | colder water, lower *R*<sub>th</sub>, smaller rise |
| 54 V → 800 V | loss × (54/800)² | — | voltage, copper |
| coolant flow | *P* / (*c*<sub>p</sub> Δ*T*) = 1.49 L/min/kW at 10 K | — | the rise across the rack |

The ladder ends where it began. A GEMM's clock was set in chapter 1 by a power budget, and
that budget is set by how much power the building can deliver and how cold its water is. A
rack that is 3 kelvin warmer or 100 W per GPU poorer is a slower rack, and the simulator can
now say by how much.

## Exercises

1. A site has 10 MW for GPU boards, running the loop of 17.3. Its GPUs run at a board limit of
   1100 W, or of 800 W with the step 13.7% longer. How much more work does the site do at
   800 W, if the job scales perfectly?

   *Answer:* at the mean draws of 17.3, 10 MW feeds 10 341 GPUs at 967 W or 13 793 at 725 W,
   1.33 times as many. Each runs 1/1.137 as fast, so the site does 1.33/1.137 = 1.17 times the
   work, which is the +17.3% of work per joule in the table.

2. A training step spends 0.6 s at 1100 W per GPU and 0.4 s at 400 W, on 4608 GPUs (64 racks).
   How much energy storage flattens it, and how much does a floor at 90% of peak burn instead?

   *Answer:* swing = 4608 × 700 W = 3.23 MW, *d* = 0.6, *T* = 1 s: 0.24 × 3.23 MW × 1 s =
   775 kJ, about 215 Wh. The floor is 990 W, so it burns 590 W × 0.4 s per GPU per step,
   against a mean of 820 W × 1 s: 29% more energy.

3. With *R*<sub>th</sub> = 0.04 K/W and *T*<sub>max</sub> = 90 °C, what is the warmest coolant
   that keeps an 1100 W board at its electrical limit? What does a cold plate with
   *R*<sub>th</sub> = 0.03 K/W buy?

   *Answer:* 90 − 1100 × 0.04 = 46 °C. With 0.03 K/W, 90 − 33 = 57 °C, 11 K more of
   facility water, approach and rise.

4. Why does a cap of 600 W cost a decode step 0.6% but a GEMM loop 29%?

   *Answer:* the GEMMs are tensor bound and at the cap, so their clock falls as
   (cap − idle)<sup>1/3</sup>. The decode kernels are bound by HBM, which keeps its speed in
   nanoseconds when the SM clock drops (chapter 1.6). Only the few partly tensor-bound kernels
   slow down.

## Reproduce

| what | command |
|---|---|
| every table (about 75 s on 4 cores) | `python docs/book/scripts/power_cooling.py` |
| one experiment | `python docs/book/scripts/power_cooling.py trace` (or `cap`, `thermal`, `decode`, `facility`) |
| the power trace of any run | `rs.simulate(...)["power"]`: `bin_us`, `gpus_w`, `avg_w`, `peak_w`, `min_w`, `energy_j`, `max_step_w_per_us`; each rank's `energy_j` and `avg_power_w` in `["ranks"]` |
| a different board limit | `rs.simulate(power.with_power_cap("hgx_b300_8", 800), wl)`: any system, every GPU's `gpu.power.cap_w` set |
| the closed forms | `rackscale.power`: `cap_tradeoff`, `smoothing`, `square_wave_buffer_j`, `thermal_power_limit`, `junction_temp_c`, `distribution`, `coolant_flow`, `racks_per_cdu`, `tokens_per_joule` |

Back to [the ladder](README.md#the-ladder).
