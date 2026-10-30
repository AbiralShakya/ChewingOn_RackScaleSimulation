"""Chapter 17: power, cooling and the clock, from the board to the facility.

    maturin develop --release
    python docs/book/scripts/power_cooling.py             # everything (a few minutes on 4 cores)
    python docs/book/scripts/power_cooling.py trace cap   # or: thermal, decode, facility

trace     board power over time of 72 GB300 GPUs running 8 x (eight 4096^3 GEMMs, then a
          64 MiB NVLS all-reduce) in lockstep, and what that trace asks of energy storage
cap       the same loop on one HGX B300 board (8 GPUs) with the board limit at 1100 ... 600 W:
          step time, energy per step and work per joule
thermal   the board limit a cold plate allows at each coolant temperature, and the step it gives
decode    a memory-bound decode step for contrast (Llama-3-8B, TP8, batch 64), and the cap
facility  closed forms: 54 V vs 800 V feed, coolant flow, racks per CDU
"""

import concurrent.futures as cf
import os
import sys

import rackscale as rs
from rackscale import power

GEMM = {"type": "gemm", "m": 4096, "n": 4096, "k": 4096}
FLOPS_PER_GEMM = 2 * 4096 ** 3


def loop(world, phases=8, gemms=8, ar_bytes=64 << 20):
    body = [{"op": "kernel", "kernel": GEMM} for _ in range(gemms)]
    body.append({"op": "collective", "kind": "all_reduce", "bytes": ar_bytes, "group": "world", "sms": 20})
    return {"name": "power loop", "spmd": {"world": world, "program": {"streams": [
        [{"op": "loop", "times": phases, "body": body}]]}}}


def trace():
    r = rs.simulate("gb300_nvl72", loop(72), fast=True)
    t = power.rack_trace(r)
    s = power.smoothing(t["watts"], t["bin_us"], floor_fraction=0.9)
    w = t["watts"]
    print(f"\n## Board power of 72 GPUs, 8 x (8 GEMMs + 64 MiB all-reduce): step {r.makespan_us:.0f} µs, "
          f"bins of {t['bin_us']:.1f} µs\n")
    print("| quantity | value |")
    print("|---|---|")
    print(f"| mean | {s['mean_w'] / 1e3:.1f} kW ({s['mean_w'] / 72:.0f} W per GPU) |")
    print(f"| peak / trough | {s['peak_w'] / 1e3:.1f} / {s['trough_w'] / 1e3:.1f} kW |")
    print(f"| swing | {s['swing_w'] / 1e3:.1f} kW |")
    print(f"| largest step between bins | {s['max_ramp_w_per_ms'] / 1e6:.1f} MW/ms |")
    print(f"| storage to flatten it to its mean | {s['buffer_j']:.1f} J |")
    print(f"| a 90%-of-peak floor instead | burns {100 * s['floor_burn_fraction']:.1f}% more energy |")
    duty = r.raw["ranks"][0]["kernel_us"] / r.makespan_us
    period = r.makespan_us * 1e-6 / 8
    print(f"\nclosed form d (1 - d) swing T, d = {duty:.2f} of a {period * 1e6:.0f} µs phase in GEMMs: "
          f"{power.square_wave_buffer_j(s['swing_w'], period, duty):.1f} J")
    phase = len(w) // 8
    step = max(1, phase // 20)
    samples = " ".join(f"{x / 1e3:.0f}" for x in w[phase:2 * phase:step])
    print(f"\none phase (kW, every {step * t['bin_us']:.0f} µs): {samples}")


def cap():
    caps = [1100, 1000, 900, 800, 700, 600]
    res = pmap(lambda c: rs.simulate(power.with_power_cap("hgx_b300_8", c), loop(8), fast=True), caps)
    flops = 8 * 8 * 8 * FLOPS_PER_GEMM
    print("\n## Board limit on one HGX B300 (8 GPUs), the same loop\n")
    base_t, base_e = res[0].makespan_us, res[0]["power"]["energy_j"]
    share = res[0].raw["ranks"][0]["kernel_us"] / base_t
    idle = idle_w()
    print(f"closed form: the GEMMs' {100 * share:.0f}% of the step times power.cap_tradeoff (idle {idle:.0f} W)\n")
    print("| board limit | step (µs) | vs 1100 W | closed form | GPU power, mean (W) | energy per step (J) "
          "| TFLOP per J | vs 1100 W |")
    print("|---|---|---|---|---|---|---|---|")
    for c, r in zip(caps, res):
        e = r["power"]["energy_j"]
        cf_ = share * (power.cap_tradeoff(c, 1100, idle)["time"] - 1)
        print(f"| {c} W | {r.makespan_us:.0f} | {100 * (r.makespan_us / base_t - 1):+.1f}% | {100 * cf_:+.1f}% | "
              f"{r['power']['avg_w'] / 8:.0f} | {e:.2f} | {flops / e / 1e12:.2f} | {100 * (base_e / e - 1):+.1f}% |")


def thermal(r_th=0.04, t_max=90.0):
    temps = [25, 35, 45, 50, 55, 60]
    limits = [min(1100.0, power.thermal_power_limit(t, r_th, t_max)) for t in temps]
    res = pmap(lambda c: rs.simulate(power.with_power_cap("hgx_b300_8", c), loop(8), fast=True), limits)
    print(f"\n## Coolant temperature sets the clock (R_th = {r_th} K/W junction to coolant, "
          f"throttle at {t_max:.0f} °C; both estimates)\n")
    print("| coolant at the cold plate | board limit it allows | step (µs) | vs 25 °C |")
    print("|---|---|---|---|")
    for t, c, r in zip(temps, limits, res):
        print(f"| {t} °C | {c:.0f} W | {r.makespan_us:.0f} | {100 * (r.makespan_us / res[0].makespan_us - 1):+.1f}% |")


def decode(batch=64):
    from rackscale import lower, models
    wl = lower.decode_step(models.LLAMA3_8B, rs.Parallel(tp=8), batch=batch, context=4096)
    caps = [1100, 800, 600]
    res = pmap(lambda c: rs.simulate(power.with_power_cap("hgx_b300_8", c), wl, fast=True), caps)
    print(f"\n## A decode step for contrast (Llama-3-8B, TP8, batch {batch}, context 4096)\n")
    print("| board limit | step (µs) | vs 1100 W | GPU power, mean / peak (W) | tokens per J |")
    print("|---|---|---|---|---|")
    for c, r in zip(caps, res):
        p = r["power"]
        print(f"| {c} W | {r.makespan_us:.0f} | {100 * (r.makespan_us / res[0].makespan_us - 1):+.1f}% | "
              f"{p['avg_w'] / 8:.0f} / {p['peak_w'] / 8:.0f} | {power.tokens_per_joule(batch, r):.2f} |")
    g = rs.presets()["gpus"]["b300"]
    bw = g["hbm"]["bw_gbps"] * g["hbm"]["efficiency"] * 1e9
    print(f"\nclosed form for a kernel streaming HBM: idle + bandwidth x 8 x pJ/bit = {g['power']['idle_w']:.0f} + "
          f"{bw * 8 * g['power']['hbm_pj_per_bit'] * 1e-12:.0f} W")


def idle_w():
    return rs.presets()["gpus"]["b300"]["power"]["idle_w"]


def facility(rack_w=140e3):
    print(f"\n## Facility closed forms for a {rack_w / 1e3:.0f} kW rack\n")
    print("| feed | current | conduction loss (0.5 mΩ round trip, est) |")
    print("|---|---|---|")
    for v in (54, 400, 800):
        d = power.distribution(rack_w, v)
        print(f"| {v} V | {d['amps']:.0f} A | {d['loss_w'] / 1e3:.2f} kW ({100 * d['loss_fraction']:.2f}%) |")
    print("\n| coolant rise | PG25 flow | per kW |")
    print("|---|---|---|")
    for dt in (5, 10, 15):
        f = power.coolant_flow(rack_w, dt)
        print(f"| {dt} K | {f['l_per_min']:.0f} L/min | {f['l_per_min_per_kw']:.2f} L/min/kW |")
    print(f"\nracks per 2 MW CDU at 85% liquid-cooled heat: {power.racks_per_cdu(2e6, rack_w, 0.85)}")


def pmap(f, xs):
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        return list(ex.map(f, xs))


EXPERIMENTS = {"trace": trace, "cap": cap, "thermal": thermal, "decode": decode, "facility": facility}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(EXPERIMENTS):
        EXPERIMENTS[name]()
