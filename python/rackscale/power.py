"""Power, cooling and the clock: closed forms from the board to the facility.

Chapter 17 of the book. The simulator gives each GPU's board power over time
(``Report["power"]``, from the measured power-capped clock model of chapter 1);
this module carries it up to the rack and the building:

* ``thermal_power_limit``: the board power at which a cold plate holds the
  junction at its limit, for a coolant temperature. Below the electrical cap
  it becomes the cap, so the coolant temperature sets the clock.
* ``with_power_cap``: a system with every GPU's board limit changed, to
  simulate a rack budget or a thermal limit; ``cap_tradeoff`` is the closed
  form for a kernel that sits at the cap.
* ``distribution``: current and conduction loss of feeding a rack at 54 V
  or 800 V through the same copper.
* ``coolant_flow``: the flow that carries a heat load at a temperature rise,
  and how many racks one CDU serves.
* ``smoothing``: what a power trace asks of energy storage: the swing, the
  ramp, the energy a buffer must hold to flatten it to its mean, and the
  energy a power floor burns to hide it instead; ``square_wave_buffer_j``
  is the closed form for a step that alternates compute and communication.

Constants that are not measured by this project carry ``est`` in their
docstring and are parameters, never hidden.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

# PG25 (25% propylene glycol in water) near 30 C: specific heat and density (est,
# handbook values).
PG25_CP_J_PER_KG_K = 3950.0
PG25_DENSITY_KG_PER_L = 1.02


def junction_temp_c(board_w: float, coolant_c: float, r_th_k_per_w: float) -> float:
    """Steady-state junction temperature: coolant + P * R_th (junction to coolant)."""
    return coolant_c + board_w * r_th_k_per_w


def thermal_power_limit(coolant_c: float, r_th_k_per_w: float, t_max_c: float = 90.0) -> float:
    """Largest steady board power that keeps the junction at or below `t_max_c`.

    `r_th_k_per_w` lumps die, TIM, lid and cold plate (est: about 0.03-0.05 K/W
    for a 1 kW-class liquid-cooled GPU). `t_max_c` is where the GPU starts to
    slow down (est)."""
    return max(0.0, (t_max_c - coolant_c) / r_th_k_per_w)


def with_power_cap(system: str, cap_w: float, overrides: Optional[dict] = None) -> str:
    """System text with every GPU's board power limit set to `cap_w`, for
    ``rackscale.simulate``. The measured power runs stay as they are, so the
    clock each kernel sustains follows from the lower budget."""
    from . import api

    o = dict(overrides or {})
    o["gpu.power.cap_w"] = cap_w
    return api.resolve(system, o)["yaml"]


def cap_tradeoff(cap_w: float, ref_cap_w: float, idle_w: float) -> Dict[str, float]:
    """A kernel that draws the cap at `ref_cap_w`, moved to `cap_w`.

    With P = idle + D (f / f_max)^3 and the kernel at the cap, the clock it
    sustains scales as (cap - idle)^(1/3), whatever D is. If its time is all
    SM bound, time scales as 1 / clock and energy as cap * time:

        clock  = ((cap - idle) / (ref - idle))^(1/3)
        time   = 1 / clock
        energy = (cap / ref) * time

    Parts of the kernel bound by HBM or L2 keep their speed. For a step,
    weight the time change by the kernel's share of the step (chapter 17
    gets the simulated cap sweep within a point that way)."""
    clock = ((cap_w - idle_w) / (ref_cap_w - idle_w)) ** (1.0 / 3.0)
    return {"clock": clock, "time": 1.0 / clock, "energy": cap_w / ref_cap_w / clock}


def distribution(rack_w: float, volts: float, r_milliohm: float = 0.5) -> Dict[str, float]:
    """Feeding `rack_w` at `volts` through a conductor path of `r_milliohm`
    (round trip; est). Current is P / V and conduction loss I^2 R, so at a
    fixed copper cross-section the loss falls as 1 / V^2: 800 V loses
    (800 / 54)^2 = 219x less than 54 V. Equivalently, for the same loss the
    copper can be 219x thinner."""
    i = rack_w / volts
    loss = i * i * r_milliohm * 1e-3
    return {"amps": i, "loss_w": loss, "loss_fraction": loss / rack_w}


def coolant_flow(heat_w: float, rise_k: float, cp: float = PG25_CP_J_PER_KG_K,
                 density_kg_per_l: float = PG25_DENSITY_KG_PER_L) -> Dict[str, float]:
    """Flow that carries `heat_w` with a coolant temperature rise of `rise_k`:
    m_dot = P / (c_p dT)."""
    kg_s = heat_w / (cp * rise_k)
    return {"kg_per_s": kg_s, "l_per_min": kg_s / density_kg_per_l * 60.0,
            "l_per_min_per_kw": kg_s / density_kg_per_l * 60.0 / (heat_w / 1e3)}


def racks_per_cdu(cdu_w: float, rack_w: float, liquid_fraction: float = 1.0) -> int:
    """Racks one CDU of `cdu_w` can serve when `liquid_fraction` of each rack's
    heat goes to liquid."""
    return int(cdu_w // (rack_w * liquid_fraction))


def smoothing(watts: Sequence[float], bin_us: float, floor_fraction: float = 0.0) -> Dict[str, float]:
    """What a power trace asks of the supply.

    * swing: peak - trough; ramp: the largest change between bins, per ms.
    * buffer_j: the energy storage needed for the supply to see only the mean
      power. The store absorbs P - mean when the load is above it and gives it
      back below, so it must span max S(t) - min S(t), S the running integral
      of P - mean.
    * floor_burn_j: the alternative of holding the draw at
      ``floor_fraction * peak`` by burning power when the load is lighter
      (what a GPU power floor does): the energy wasted per trace.
    """
    w = list(watts)
    if not w:
        return {}
    dt = bin_us * 1e-6
    mean = sum(w) / len(w)
    s, smax, smin = 0.0, 0.0, 0.0
    for x in w:
        s += (x - mean) * dt
        smax, smin = max(smax, s), min(smin, s)
    peak, trough = max(w), min(w)
    ramp = max((abs(b - a) for a, b in zip(w, w[1:])), default=0.0) / (bin_us / 1e3)
    floor = floor_fraction * peak
    burn = sum(max(0.0, floor - x) for x in w) * dt
    return {"mean_w": mean, "peak_w": peak, "trough_w": trough, "swing_w": peak - trough,
            "max_ramp_w_per_ms": ramp, "buffer_j": smax - smin,
            "floor_burn_j": burn, "floor_burn_fraction": burn / max(1e-12, sum(w) * dt)}


def square_wave_buffer_j(swing_w: float, period_s: float, duty: float) -> float:
    """Energy storage that flattens a load alternating between high and low
    (high for a fraction `duty` of each `period_s`) to its mean.

    The mean sits duty * swing above the low level, so during the high part
    the store gives (1 - duty) * swing for duty * period: d (1 - d) swing T.
    It grows with the period, so a training step's seconds-long phases need
    a million times more than the microsecond phases of one kernel loop."""
    return duty * (1.0 - duty) * swing_w * period_s


def rack_trace(report: Union[dict, "object"], replicas: int = 1) -> Dict[str, object]:
    """GPU board power over time from a simulation report, times `replicas`
    identical copies in lockstep (a simulated 8-GPU group standing for the
    nine of an NVL72, say). Returns ``{"bin_us", "watts"}``."""
    raw = report.raw if hasattr(report, "raw") else report
    p = raw.get("power")
    if not p:
        raise ValueError("the report has no power trace: the GPUs need a power model")
    return {"bin_us": p["bin_us"], "watts": [x * replicas for x in p["gpus_w"]]}


def tokens_per_joule(tokens: float, report: Union[dict, "object"]) -> float:
    """Tokens per joule of GPU board energy over the run."""
    raw = report.raw if hasattr(report, "raw") else report
    return tokens / raw["power"]["energy_j"]


__all__ = [
    "junction_temp_c", "thermal_power_limit", "with_power_cap", "cap_tradeoff", "distribution",
    "coolant_flow", "racks_per_cdu", "smoothing", "square_wave_buffer_j", "rack_trace",
    "tokens_per_joule",
]
