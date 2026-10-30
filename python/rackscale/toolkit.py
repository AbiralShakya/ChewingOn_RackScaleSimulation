"""Rack-scale answers: every simulated number beside its closed form, an error
bar, and the bottleneck with the reason for it.

The agent tools (``rackscale.tools``) are thin wrappers over these functions.
Each one returns an ``answer`` block of the same shape, so an agent (or a
person) can read any tool the same way::

    {"quantity": "step time", "value": 3151.0, "unit": "us",
     "band": {"lo": ..., "hi": ..., "band_pct": 23.0},     # from measured validation
     "closed_form": {"value": 953.0, "formula": "..."},    # the textbook number
     "sim_over_closed_form": 3.31,                          # how far reality departs
     "bottleneck": "idle", "why": "..."}                     # what bounds it, and why

Chapters 15-17 of docs/book derive the closed forms used here; chapter 11
(section 11.5) measures how far the textbook sits from the simulator.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from . import accuracy

# What bounds a step, and why that part costs time. Keys are the parts of
# ``analysis.bottlenecks``' time split.
WHY = {
    "idle": "GPUs sit between kernels: launch gaps, dependencies on other ranks or pipeline bubbles. Nothing "
            "computes, so neither the FLOPs nor the bytes are the limit; CUDA graphs, a megakernel, more work per "
            "launch (batch) or overlapping streams remove it.",
    "collectives": "communication is on the critical path: GPUs wait for all-reduces / all-to-alls whose time is "
                   "alpha + bytes / link bandwidth. Overlap them with compute, shrink the group (less TP / EP), use "
                   "in-switch reduction (NVLS) or a faster fabric.",
    "kernels:hbm": "kernels stream bytes at the HBM roof (time = bytes / bandwidth): weights and KV cache are read "
                   "for few FLOPs each. More tokens per weight read (batch), fewer bytes (FP8 / FP4, smaller KV) or "
                   "more HBM bandwidth help; more FLOP/s does not.",
    "kernels:tensor": "kernels run near the tensor-core roof (time = FLOPs / peak). Only fewer FLOPs (lower "
                      "precision, sparsity, less recompute) or more / faster GPUs help.",
    "kernels:memory_latency": "kernels are too small to cover memory latency: by Little's law a GPU needs "
                              "bandwidth x latency bytes in flight, and these kernels keep fewer. Bigger tiles or "
                              "batches, deeper pipelines or fusing kernels help; more bandwidth does not.",
    "kernels:tensor_pipeline": "the tensor core is busy but stalls on its own pipeline (issue, accumulator "
                               "hand-off); a better kernel schedule (more stages, 2-CTA MMA) helps.",
    "kernels:sync": "warps wait on each other at barriers: the work per CTA is unbalanced or the pipeline hand-off "
                    "is too fine.",
    "kernels:smem_mio": "shared-memory / MIO traffic limits: bank conflicts or too many small loads per FLOP.",
    "kernels:alu": "the CUDA-core pipes (softmax exponentials, normalisation, address math) bound the kernel "
                   "while the tensor core waits: FlashAttention-4-style exp emulation or fewer non-matmul ops help.",
    "kernels:tail": "the last wave of CTAs leaves SMs idle: the grid does not divide the GPU (wave "
                    "quantisation); split-K, a different tile or a persistent schedule helps.",
}


def answer(quantity: str, value: float, unit: str, *, band_pct: Optional[float] = None,
           worst_pct: Optional[float] = None, closed_form: Optional[float] = None, formula: Optional[str] = None,
           bottleneck: Optional[str] = None, why: Optional[str] = None, **extra) -> dict:
    """The common answer block (see the module docstring)."""
    out: Dict[str, Any] = {"quantity": quantity, "value": _r(value), "unit": unit}
    if band_pct is not None:
        out["band"] = {"lo": _r(value * (1 - band_pct / 100)), "hi": _r(value * (1 + band_pct / 100)),
                       "band_pct": round(band_pct, 2)}
        if worst_pct is not None:
            out["band"]["worst_seen_pct"] = round(worst_pct, 1)
    if closed_form is not None:
        out["closed_form"] = {"value": _r(closed_form), "formula": formula}
        if closed_form:
            out["sim_over_closed_form"] = round(value / closed_form, 3)
    if bottleneck:
        out["bottleneck"] = bottleneck
        out["why"] = why or WHY.get(bottleneck, "")
    out.update(extra)
    return out


def _r(x: Any) -> Any:
    if isinstance(x, float) and math.isfinite(x):
        return float(f"{x:.4g}") if abs(x) < 1e4 else round(x, 1)
    return x


# ---------------------------------------------------------------------------
# a simulated run: the textbook beside it, and what bounds it


def _launch_ns(rec: dict) -> float:
    if rec.get("launch_gap_ns") is not None:
        return float(rec["launch_gap_ns"])
    return float(rec["system"]["gpus"][0].get("launch_overhead_ns", 0.0))


def textbook(rec: dict) -> Optional[dict]:
    """The roofline replay of a run's exact program: every kernel at max(F / C,
    B / W) plus the launch gap, every collective at its closed form, perfect
    overlap (``scaling.replay``). ``None`` when the run's kernels have no
    first-order work (recorded paths, SASS, captures)."""
    from . import scaling
    wl = rec.get("workload")
    if not isinstance(rec.get("system_input"), str) or not wl or not (wl.get("spmd") or wl.get("ranks")):
        return None
    try:
        hw = scaling.hardware(rec["system_input"], launch=_launch_ns(rec) * 1e-9, overrides=rec["overrides"] or None)
        rp = scaling.replay(wl, hw)
    except (KeyError, ValueError, TypeError, ZeroDivisionError, RuntimeError):
        return None
    rep = rec["report"]
    r0 = rep["ranks"][0]
    k_rp = rp.ranks[0]["kernel_us"]
    c_rp = rp.ranks[0]["collective_us"] + rp.ranks[0]["p2p_us"]
    step_rp = rp.step * 1e6
    sim = rep["makespan_us"]
    return {
        "step_us": _r(step_rp),
        "method": "roofline replay of the same program: kernels at max(FLOPs / peak, bytes / HBM BW) + launch gap, "
                  "collectives at alpha + bytes / link BW (ring 2(n-1)/n, NVLS 1x), perfect overlap",
        "kernels_us": {k: _r(v * 1e6) for k, v in rp.kernels.items()},
        "collectives_us": {k: _r(v * 1e6) for k, v in rp.collectives.items()},
        "sim_over_textbook": round(sim / step_rp, 3) if step_rp else None,
        "gap_split_us": {
            "kernels_slower_than_roofline": _r(r0["kernel_us"] - k_rp),
            "collectives_slower_than_closed_form": _r(r0["collective_us"] - c_rp),
            "idle_and_lost_overlap": _r(sim - step_rp - (r0["kernel_us"] - k_rp) - (r0["collective_us"] - c_rp)),
        },
        "gap_note": "rank 0; kernel and collective time summed over streams, so overlap shows up in the last term",
    }


def step_bottleneck(rec: dict) -> dict:
    """The biggest part of the step (kernels by bound, collectives, idle), with why it costs time.

    The parts partition GPU time: kernels by bound, time inside collectives, and idle with
    neither (launch gaps, waits on other ranks)."""
    from . import analysis
    split = analysis.bottlenecks(rec, 1)["time_split"]
    parts = {f"kernels:{k}": v for k, v in split["kernels_by_bound"].items()}
    parts["collectives"] = split["collectives"]
    parts["idle"] = split["idle_without_collectives"]
    name, share = max(parts.items(), key=lambda kv: kv[1])
    return {"name": name, "share_of_gpu_time": round(share, 3), "why": WHY.get(name, ""),
            "split": {k: round(v, 3) for k, v in sorted(parts.items(), key=lambda kv: -kv[1])}}


def run_answer(summary: dict, rec: dict) -> dict:
    """``answer`` block for a simulated run, plus the textbook beside it."""
    tb = textbook(rec)
    bn = step_bottleneck(rec)
    st = summary["step_time_us"]
    return answer("step time", st["value"], "us", band_pct=st["band_pct"], worst_pct=summary["accuracy"]["worst_pct"],
                  closed_form=tb["step_us"] if tb else None, formula=tb["method"] if tb else None,
                  bottleneck=bn["name"], why=bn["why"], share_of_gpu_time=bn["share_of_gpu_time"])


# ---------------------------------------------------------------------------
# inside one GPU: GEMM tiling (chapter 15)

_TILING = ("tile_n", "stages", "cluster", "split_k", "reserve_sms")


def gemm_tiling(system: str = "hgx_b300_8", m: int = 4096, n: int = 4096, k: int = 4096, dtype: str = "bf16",
                vary: Optional[Dict[str, Sequence[Any]]] = None, base: Optional[dict] = None) -> dict:
    """One GEMM on one GPU under different tiling choices.

    ``base`` fixes tiling choices for every row; ``vary`` maps one or more of
    ``tile_n`` (64 / 128 / 256), ``stages`` (shared-memory pipeline depth),
    ``cluster`` (1 / 2: B multicast across a CTA pair), ``split_k`` and
    ``reserve_sms`` (SMs held back for communication) to the values to try.
    Without ``vary`` the lowering's own choice is the only row. Blackwell
    GPUs take every knob; others ignore all but ``reserve_sms``.
    """
    from . import api, scaling
    base = {k_: v for k_, v in (base or {}).items() if v is not None}
    bad = set(base) - set(_TILING) | set(vary or {}) - set(_TILING)
    if bad:
        raise ValueError(f"unknown tiling knobs {sorted(bad)}; known: {list(_TILING)}")
    variants = [("default" if not base else ", ".join(f"{a}={b}" for a, b in base.items()), dict(base))]
    for knob, vals in (vary or {}).items():
        for v in vals:
            variants.append((f"{knob}={v}", {**base, knob: v}))
    if vary:
        variants = variants[1:] if not base else variants
    hw = scaling.hardware(system)
    g = api.resolve(system)["gpus"][0]
    sms = g["sm_count"]
    lat_s = g["hbm"]["latency_ns"] * 1e-9
    eb = scaling.DTYPE_BYTES.get(dtype, 2.0)
    rows = []
    for label, t in variants:
        t = dict(t)
        reserve = int(t.pop("reserve_sms", 0) or 0)
        spec = {"type": "gemm", "m": m, "n": n, "k": k, "dtype": dtype, **t}
        d = api.kernel(system=system, spec=spec, overrides={"sm_count": sms - reserve} if reserve else None, top_pcs=3)
        us = d["time_us"]["value"]
        stretch = d.get("dvfs_stretch") or 1.0
        flops, nbytes = scaling.kernel_work({"type": "gemm", "m": m, "n": n, "k": k, "dtype": dtype})
        split = int(t.get("split_k") or 1)
        if split > 1:
            nbytes += 2 * split * m * n * 4.0  # fp32 partials written and read back
        live = sms - reserve
        peak = hw.peak(dtype) * live / sms
        t_c, t_b = flops / peak, nbytes / hw.hbm_bw
        roof = max(t_c, t_b) * 1e6
        slots = live * max(1, d["ctas_per_sm"])
        waves = d["ctas"] / slots
        quant = math.ceil(waves) / waves if waves > 0 else 1.0
        cf = roof * (quant if t_c >= t_b else 1.0)
        need = hw.hbm_bw / sms * lat_s  # Little's law, bytes in flight per SM to stream at the HBM roof
        staged = d["smem_per_cta"] * d["ctas_per_sm"]
        rows.append({
            "variant": label, "kernel": d["name"],
            "time_us": d["time_us"], "time_us_full_clock": _r(us / stretch), "power_clock_stretch": round(stretch, 3),
            "pct_of_peak_full_clock": round(100 * flops / (us / stretch * 1e-6) / peak, 1),
            "tflops": round(d["tflops"], 1), "ctas": d["ctas"], "ctas_per_sm": d["ctas_per_sm"], "waves": round(waves, 3),
            "smem_per_cta": d["smem_per_cta"], "dram_mb": round(d["dram_bytes"] / 1e6, 1),
            "bound": d["bound"], "flags": d["flags"], "diagnosis": d["diagnosis"],
            "closed_form": {
                "roofline_us": _r(roof), "wave_quantised_us": _r(cf),
                "formula": "max(2MNK / peak, bytes / HBM BW) x ceil(waves) / waves when compute bound; "
                           "waves = CTAs / (SMs x CTAs per SM)",
                "bytes_in_flight_needed_per_sm": round(need), "smem_staged_per_sm": staged,
                "littles_law": "HBM BW / SMs x HBM latency bytes must be in flight per SM to reach the HBM roof",
            },
        })
    best = min(rows, key=lambda r: r["time_us"]["value"])
    cls = accuracy.CLASSES[("synthetic", True)]
    return {
        "gemm": {"m": m, "n": n, "k": k, "dtype": dtype, "system": system, "gpu": g["name"]},
        "rows": rows, "best": best["variant"],
        "answer": answer("GEMM time, best variant", best["time_us"]["value"], "us", band_pct=cls["typical_pct"],
                         worst_pct=cls["worst_pct"], closed_form=best["closed_form"]["wave_quantised_us"],
                         formula=best["closed_form"]["formula"], bottleneck=f"kernels:{best['bound']}",
                         variant=best["variant"]),
        "note": "times are at the power-capped clock the board sustains; time_us_full_clock removes that stretch. "
                "Differences between variants are more reliable than absolute times (same model, same errors).",
        "chapter": "docs/book/15-parallelism-inside-one-gpu.md",
    }


# ---------------------------------------------------------------------------
# the tail: stragglers, link faults, hash collisions (chapter 16)

_GEMM = {"type": "gemm", "m": 2048, "n": 4096, "k": 4096}


def _bsp(world: int, group: str, phases: int, per: int, gemm: dict) -> dict:
    body: List[dict] = [{"op": "kernel", "kernel": gemm} for _ in range(per)]
    body.append({"op": "collective", "kind": "all_reduce", "bytes": 1 << 20, "group": group, "sms": 16})
    return {"name": f"bsp {group} every {per}", "spmd": {"world": world, "program": {"streams": [
        [{"op": "loop", "times": max(1, phases // per), "body": body}]]}}}


def tail_latency(ranks: int = 72, sigma: float = 0.05, sync_every: int = 1, phases: int = 16,
                 planes: int = 18, plane_scale: Optional[float] = None, flows: Optional[int] = None,
                 links: Optional[int] = None, groups: int = 1, simulate: bool = False,
                 system: str = "gb300_nvl72", seeds: Sequence[int] = (1, 2, 3), allreduce_mib: float = 64) -> dict:
    """Closed forms for the tail, optionally checked by simulation.

    * stragglers: ``ranks`` GPUs synchronise every ``sync_every`` kernels and
      each kernel takes ``1 + sigma |Z|`` of its time; the step pays the
      expected maximum over ranks, ``sigma E[max_N |Z|]`` (about
      ``sigma sqrt(2 ln N)``), against ``sigma E|Z|`` on one GPU.
    * a link of a striped collective at ``plane_scale`` of its rate
      (0 = dead) out of ``planes``.
    * ``flows`` equal flows hashed (ECMP) onto ``links`` links in ``groups``
      places: the busiest link's load over the mean.

    ``simulate`` runs the straggler loop (``phases`` GEMMs and 1 MiB
    all-reduces on ``system``) with and without jitter, mean over ``seeds``,
    and, with ``plane_scale``, an ``allreduce_mib`` all-reduce over the whole
    system with GPU 0's first NVLink degraded (NVLS and ring).
    """
    from . import api, tail
    out: Dict[str, Any] = {"chapter": "docs/book/16-stragglers-flaps-tail-latency.md"}
    trials = 2000 if sync_every > 1 else 0
    cf = tail.straggler_overhead(ranks, sigma, sync_every, trials=trials or 20000)
    out["stragglers"] = {
        "ranks": ranks, "sigma": sigma, "sync_every": sync_every,
        "overhead_closed_form": round(cf, 4), "one_gpu_alone": round(tail.mean_overhead(sigma), 4),
        "expected_max_half_normal": round(tail.expected_max_half_normal(ranks), 3),
        "formula": "added compute time / compute = sigma x E[max over N ranks of |Z|] (sync every kernel; "
                   "~ sigma sqrt(2 ln N)); sync every s kernels: E[max_N of the mean of s |Z|] shrinks it",
        "validation": "simulated 72-rank loop: 11.2% vs 13.2% closed form at sigma = 5%; 5.6% vs 5.9% syncing every "
                      "16 GEMMs (the collective absorbs part of the spread, so the closed form is an upper bound)",
    }
    if plane_scale is not None:
        out["link"] = {
            "planes": planes, "rate_scale": plane_scale,
            "nvls_striped_slowdown_closed_form": round(tail.dead_plane_slowdown(planes), 4) if plane_scale <= 0
            else round(tail.degraded_plane_slowdown(planes, plane_scale), 4),
            "formula": "striped in lockstep over P planes: a dead plane gives P / (P - 1); one at rate s gives "
                       "max(1, 1 / s) on the bandwidth term (the slow plane sets the finish). This bounds the "
                       "bandwidth term only: the whole collective also pays latency and protocol costs, so it slows "
                       "less (64 MiB on GB300 NVL72: +50% simulated against 2x on the bandwidth term)",
            "ring_note": "a ring routes each hop over all of a GPU's links and is barely affected by one slow plane "
                         "(+0.1% simulated at half rate, against +50% for NVLS)",
        }
    if flows and links:
        out["ecmp"] = {"flows": flows, "links": links, "groups": groups,
                       "busiest_over_mean": round(tail.ecmp_max_load(flows, links, groups, trials=2000), 3),
                       "formula": "balls into bins: E[max load] / (flows / links); adaptive routing removes it",
                       "validation": "448 flows on 8 links in 16 places (groups=16): 1.34x closed form, +26% simulated all-to-all with ECMP on 64 H800s"}
    if simulate:
        world = api.resolve(system)["num_gpus"]
        if ranks > world:
            raise ValueError(f"{system} has {world} GPUs; ranks must be <= {world}")
        grp = "world" if ranks == world else f"block:{ranks}"
        wl = _bsp(world, grp, phases, sync_every, _GEMM)
        base = api.run(system, wl, label="tail: no jitter")
        rec0 = api._rec(base["run_id"])
        compute = rec0["report"]["ranks"][0]["kernel_us"]
        steps = []
        for s in seeds:
            r = api.run(system, wl, variability={"kernel_sigma": sigma, "seed": int(s)}, label=f"tail: sigma {sigma} seed {s}")
            steps.append(api._rec(r["run_id"])["report"]["ranks"][0]["finish_us"])
        b = rec0["report"]["ranks"][0]["finish_us"]
        sim = (sum(steps) / len(steps) - b) / compute
        out["stragglers"]["simulated"] = {"step_us_no_jitter": _r(b), "step_us_jitter_mean": _r(sum(steps) / len(steps)),
                                          "overhead": round(sim, 4), "compute_us": _r(compute), "seeds": list(seeds),
                                          "run_id_no_jitter": base["run_id"]}
        out["answer"] = answer("straggler overhead (added / compute)", sim, "fraction", closed_form=cf,
                               formula=out["stragglers"]["formula"], bottleneck="stragglers",
                               why="every synchronisation waits for the slowest of N ranks; the expected maximum "
                                   "of N draws grows like sqrt(2 ln N), so the loss grows with scale even though "
                                   "each GPU is only sigma E|Z| slow on average. Synchronise less often, overlap "
                                   "the collective, or screen slow GPUs.")
        if plane_scale is not None:
            fault = {"a": "gpu0", "b": "nvsw0.0"}
            if plane_scale > 0:
                fault["rate_scale"] = plane_scale
            res = {}
            for algo in ("nvls", "ring"):
                ar = {"name": f"all_reduce {algo}", "spmd": {"world": world, "program": {"streams": [[
                    {"op": "collective", "kind": "all_reduce", "bytes": int(allreduce_mib * 2 ** 20), "group": "world",
                     "algo": algo,
                     "sms": 20}]]}}}
                h = api.run(system, ar, label=f"healthy {algo}")["step_time_us"]["value"]
                f = api.run(system, ar, net={"faults": [fault]}, label=f"fault {algo}")["step_time_us"]["value"]
                res[algo] = {"healthy_us": _r(h), "fault_us": _r(f), "slowdown": round(f / h, 4)}
            out["link"]["simulated"] = res
    else:
        out["answer"] = answer("straggler overhead (added / compute)", cf, "fraction", closed_form=cf,
                               formula=out["stragglers"]["formula"], bottleneck="stragglers",
                               why="the expected maximum of N half-normal draws grows like sqrt(2 ln N); pass "
                                   "simulate=true to check it on the simulated rack",
                               band_pct=None, note="closed form only; it was 0-2 points above simulation in chapter 16")
    return out


# ---------------------------------------------------------------------------
# power, cooling and the clock (chapter 17)


def power_thermal(run_id: str, caps_w: Optional[Sequence[float]] = None, coolant_c: Optional[float] = None,
                  r_th_k_per_w: float = 0.04, t_max_c: float = 90.0) -> dict:
    """Board power of a run, what a power cap or a warm coolant does to it, and
    what the rack asks of its supply.

    ``caps_w``: board power limits to re-simulate (the clock each kernel
    sustains follows the measured power model). ``coolant_c``: the cold-plate
    coolant temperature; the GPU's thermal limit, (T_max - T_coolant) / R_th,
    is added as a cap when it is below the electrical one.
    """
    from . import api, power
    rec = api._rec(run_id)
    rep = rec["report"]
    p = rep.get("power")
    if not p:
        raise ValueError("this run has no power trace: its GPUs have no power model (B300 / GB300 do)")
    g = rec["system"]["gpus"][0]
    pm = g.get("power") or {}
    cap0, idle = pm.get("cap_w"), pm.get("idle_w", 0.0)
    ngpu = rec["system"]["num_gpus"]
    sm = power.smoothing(p["gpus_w"], p["bin_us"])
    tok = api.tokens_per_step(rec.get("spec"))
    out: Dict[str, Any] = {
        "run_id": run_id, "gpus": ngpu, "board_cap_w": cap0, "idle_w": idle,
        "per_gpu": {"mean_w": _r(p["avg_w"] / ngpu), "peak_w": _r(p["peak_w"] / ngpu)},
        "all_gpus": {"mean_w": _r(p["avg_w"]), "peak_w": _r(p["peak_w"]), "energy_j_per_step": _r(p["energy_j"]),
                     "swing_w": _r(sm["swing_w"]), "max_ramp_w_per_ms": _r(sm["max_ramp_w_per_ms"]),
                     "buffer_j_to_flatten": _r(sm["buffer_j"])},
        "chapter": "docs/book/17-power-cooling-and-the-clock.md",
    }
    if tok:
        out["tokens_per_joule"] = _r(tok / p["energy_j"])
    caps = list(caps_w or [])
    if coolant_c is not None:
        lim = power.thermal_power_limit(coolant_c, r_th_k_per_w, t_max_c)
        out["thermal"] = {"coolant_c": coolant_c, "r_th_k_per_w": r_th_k_per_w, "t_max_c": t_max_c,
                          "thermal_power_limit_w": _r(lim),
                          "junction_at_cap_c": _r(power.junction_temp_c(cap0 or 0.0, coolant_c, r_th_k_per_w)),
                          "formula": "P_max = (T_max - T_coolant) / R_th; below the electrical cap it becomes the cap",
                          "binding": bool(cap0 and lim < cap0)}
        if cap0 and lim < cap0:
            caps.append(round(lim, 1))
    if caps and cap0:
        from . import analysis
        tensor_share = analysis.bottlenecks(rec, 1)["time_split"]["kernels_by_bound"].get("tensor", 0.0)
        base = rec["report"]["makespan_us"]
        rows = []
        for c in caps:
            w = api.what_if(run_id, {"hardware": {"gpu.power.cap_w": float(c)}}, label=f"power cap {c} W")
            r = w["result"]
            ct = power.cap_tradeoff(float(c), cap0, idle)
            pred = base * (1 + tensor_share * (ct["time"] - 1))
            nrec = api._rec(r["run_id"])
            e = nrec["report"].get("power", {}).get("energy_j")
            rows.append({
                "cap_w": c, "step_time_us": r["step_time_us"], "delta_pct": w["comparison"]["step_time_us"]["delta_pct"],
                "significant": w["comparison"]["difference"]["significant"],
                "energy_j_per_step": _r(e) if e else None,
                "tokens_per_joule": _r(tok / e) if tok and e else None,
                "closed_form": {"clock_ratio": round(ct["clock"], 4), "step_time_us": _r(pred),
                                "formula": "clock ~ ((cap - idle) / (cap0 - idle))^(1/3) for kernels at the cap; "
                                           "only the tensor-bound share of the step slows by 1 / clock"},
                "run_id": r["run_id"]})
        out["caps"] = rows
        last = rows[-1]
        out["answer"] = answer(f"step time at {last['cap_w']} W", last["step_time_us"]["value"], "us",
                               band_pct=last["step_time_us"]["band_pct"], closed_form=last["closed_form"]["step_time_us"],
                               formula=last["closed_form"]["formula"],
                               bottleneck="power" if tensor_share > 0.3 else "not power",
                               why=("the step is mostly tensor-bound kernels that draw the cap, so the clock and the "
                                    "time follow the cube root of the power budget" if tensor_share > 0.3 else
                                    "little of the step runs at the power cap (memory- or latency-bound kernels "
                                    "and idle gaps draw less), so the cap barely moves the step time"),
                               tensor_bound_share=round(tensor_share, 3))
    return out


# ---------------------------------------------------------------------------
# tokens or bytes: H-Net against BPE (chapter 13)


def tokens_or_bytes(model: str = "hnet-1stage-xl", baseline: str = "bpe-transformer-xl", batch: int = 64,
                    context: int = 8192, system: str = "hgx_b300_8", simulate: bool = False, steps: int = 8) -> dict:
    """Byte-level H-Net against a BPE (or byte) Transformer per byte of text.

    Closed form: FLOPs per byte of a prefill of ``context`` bytes, and HBM
    bytes moved per byte of generated text at decode (weights once per step,
    KV cache, Mamba state), divided by HBM bandwidth for a memory-bound
    time per byte. ``simulate`` runs both decodes on one GPU of ``system``.
    ``context`` is in bytes of text for both models.
    """
    from . import api, hnet, lower, scaling
    if model not in hnet.ALL:
        raise ValueError(f"model must be one of {sorted(hnet.ALL)}")
    if baseline not in hnet.BASELINES:
        raise ValueError(f"baseline must be one of {sorted(hnet.BASELINES)}")
    h, m = hnet.ALL[model], hnet.BASELINES[baseline]
    hw = scaling.hardware(system)
    bpt = hnet.text_bytes_per_step(m)
    toks = max(1, int(context / bpt))
    base_fpb = lower._model_flops(m, toks, toks // 2, False, lm_head=True) / context
    h_fpb = hnet.flops_per_byte(h, context)
    th, tm = hnet.decode_traffic(h, batch, context), hnet.decode_traffic(m, batch, context)
    # per_byte is HBM bytes per byte of each sequence's text, so a step that adds
    # batch x bytes_per_step bytes takes step / BW: text bytes/s = batch x BW / per_byte.
    rate_h = batch * hw.hbm_bw / th["per_byte"]
    rate_m = batch * hw.hbm_bw / tm["per_byte"]
    out: Dict[str, Any] = {
        "model": model, "baseline": baseline, "batch": batch, "context_bytes": context,
        "bytes_per_bpe_token": hnet.BYTES_PER_BPE_TOKEN,
        "prefill_flops_per_byte": {model: _r(h_fpb), baseline: _r(base_fpb), "ratio": round(h_fpb / base_fpb, 3)},
        "decode_hbm_bytes_per_text_byte": {
            model: {k: _r(v) for k, v in th.items()}, baseline: {k: _r(v) for k, v in tm.items()},
            "ratio": round(th["per_byte"] / tm["per_byte"], 3)},
        "decode_text_bytes_per_s_closed_form": {model: _r(rate_h), baseline: _r(rate_m), "gpu": hw.gpu,
                                                "ratio": round(rate_h / rate_m, 3)},
        "formula": "decode is memory bound: text bytes/s ~ batch x bytes of text per step x HBM BW / (weights + "
                   "KV + state bytes per step); a BPE step adds ~4.45 bytes, a byte model's step adds 1, so it "
                   "must read its weights ~4.45x as often per byte unless the main network steps less (H-Net's "
                   "router: ~1 step in 4.7 or 6.9 bytes)",
        "chapter": "docs/book/13-tokens-or-bytes.md",
    }
    if simulate:
        rh = api.run(system, {"model": model, "phase": "decode", "batch": batch, "context": context, "steps": steps},
                     label=f"{model} decode")
        rm = api.run(system, {"model": baseline, "phase": "decode", "batch": batch, "context": int(context / bpt)},
                     label=f"{baseline} decode")
        bh = rh["tokens_per_s"]["value"]                  # bytes/s: H-Net steps are bytes
        bm = rm["tokens_per_s"]["value"] * bpt            # tokens/s x bytes per token
        out["simulated"] = {"text_bytes_per_s": {model: _r(bh), baseline: _r(bm)}, "ratio": round(bh / bm, 3),
                            "run_ids": [rh["run_id"], rm["run_id"]],
                            "bottleneck": {model: rh.get("bottleneck"), baseline: rm.get("bottleneck")}}
        cf_ratio = rate_h / rate_m
        out["answer"] = answer(f"{model} / {baseline} text bytes per second at decode", bh / bm, "ratio",
                               band_pct=accuracy.diff_band(1, rh["step_time_us"]["band_pct"], 1,
                                                           rm["step_time_us"]["band_pct"], "workload")["band_us"] * 100,
                               closed_form=cf_ratio, formula=out["formula"],
                               bottleneck=(rh.get("bottleneck") or {}).get("bottleneck"))
    else:
        out["answer"] = answer(f"{model} / {baseline} HBM bytes per text byte at decode",
                               th["per_byte"] / tm["per_byte"], "ratio", closed_form=th["per_byte"] / tm["per_byte"],
                               formula=out["formula"], bottleneck="kernels:hbm",
                               note="closed form; pass simulate=true for the simulated decode")
    return out


# ---------------------------------------------------------------------------
# a GPU we don't have


def predict_gpu(system: str, workload: dict, changes: Dict[str, Any], *, name: Optional[str] = None,
                mode: str = "fast") -> dict:
    """Simulate ``workload`` on ``system`` and on a hypothetical GPU made by
    ``changes`` (hardware knobs, e.g. ``{"hbm_bw_gbps": "x1.5",
    "tensor_fp4_flops_per_cycle_per_sm": "x2"}``), and set the result beside a
    first-order prediction: every kernel scaled by its roofline ratio, every
    collective by the link-bandwidth ratio, idle time unchanged.

    The gap between the two is the answer to "what does the roofline miss":
    an L2 ceiling that binds once HBM is faster, latency-bound kernels that
    do not care about bandwidth, the power cap, waves.
    """
    from . import api, scaling
    base = api.run(system, workload, mode=mode, label="base GPU")
    w = api.what_if(base["run_id"], {"hardware": changes}, label=name or "hypothetical GPU")
    new = w["result"]
    rec = api._rec(base["run_id"])
    rep = rec["report"]
    world = rep["world"]
    h0 = scaling.hardware(system, overrides=rec["overrides"] or None)
    h1 = scaling.hardware(system, overrides={**rec["overrides"], **changes})
    k_old = k_new = 0.0
    for kk in rep["kernels"]:
        t = kk["total_us"] / world
        f = kk["tflops"] * 1e12 * kk["total_us"] * 1e-6
        b = kk["gbps"] * 1e9 * kk["total_us"] * 1e-6
        dt = "bf16"
        for spec_dt in ("fp4", "fp8"):
            if spec_dt in kk["name"]:
                dt = spec_dt
        r0 = max(f / h0.peak(dt), b / h0.hbm_bw)
        r1 = max(f / h1.peak(dt), b / h1.hbm_bw)
        k_old += t
        k_new += t * (r1 / r0 if r0 > 0 else 1.0)
    r0 = rep["ranks"][0]
    c_old = r0["collective_us"]
    c_new = c_old * h0.nvlink_bw / h1.nvlink_bw
    step = rep["makespan_us"]
    pred = step - k_old + k_new - c_old + c_new
    pred = max(pred, k_new)
    sim = new["step_time_us"]["value"]
    nrec = api._rec(new["run_id"])
    b_old, b_new = step_bottleneck(rec), step_bottleneck(nrec)
    cmp = w["comparison"]
    return {
        "base": {"run_id": base["run_id"], "step_time_us": base["step_time_us"], "tokens_per_s": base.get("tokens_per_s"),
                 "bottleneck": b_old},
        "hypothetical": {"name": name, "changes": changes, "run_id": new["run_id"], "step_time_us": new["step_time_us"],
                         "tokens_per_s": new.get("tokens_per_s"), "bottleneck": b_new},
        "speedup": round(step / sim, 4),
        "difference": cmp["difference"], "verdict": cmp["verdict"],
        "answer": answer("step time on the hypothetical GPU", sim, "us", band_pct=new["step_time_us"]["band_pct"],
                         closed_form=pred,
                         formula="per kernel: time x max(F / C', B / W') / max(F / C, B / W) (F, B the kernel's "
                                 "simulated FLOPs and bytes); collectives x old / new link BW; idle unchanged",
                         bottleneck=b_new["name"], why=b_new["why"],
                         bottleneck_moved=b_old["name"] != b_new["name"]),
        "caveat": "the changed GPU has no measured calibration: its absolute error is at least the base GPU's band; "
                  "the base-to-hypothetical difference is more reliable (correlation 0.9, see accuracy)",
    }


# ---------------------------------------------------------------------------
# reports


def html_report(run_ids: Sequence[str], path: str, *, labels: Optional[Sequence[str]] = None,
                title: Optional[str] = None) -> dict:
    """Write the HTML report (time split, kernels by bound, stalls, roofline,
    collectives) for one run or several side by side."""
    from . import api, report
    recs = [api._rec(r) for r in run_ids]
    reps = [r["report"] for r in recs]
    sys0 = recs[0]["system"]["gpus"][0]
    kw: Dict[str, Any] = {}
    if labels:
        kw["labels"] = list(labels)
    if title:
        kw["title"] = title
    out = report.write(reps if len(reps) > 1 else reps[0], path, system=sys0, **kw)
    return {"path": out, "runs": list(run_ids), "bytes": __import__("os").path.getsize(out)}
