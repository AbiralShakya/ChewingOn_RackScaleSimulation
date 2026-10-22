"""Turn simulator reports into diagnoses: what bounds each kernel, and what to try.

The rules are deliberately simple and visible, so an engineer (or an agent)
can check them against the numbers they come with:

* a kernel at >= 70% of achievable HBM bandwidth is **hbm**-bound; at >= 60%
  of the dense tensor peak for its dtype it is **tensor**-bound;
* otherwise it is bound by whatever its scheduler cycles wait on most, after
  setting aside ``async_wait`` (warps parked on mbarriers, which is how every
  warp-specialised kernel idles) and issue cycles: **memory_latency**
  (TMA / L1TEX round trips), **tensor_pipeline**, **sync**, **smem_mio**,
  **alu** or **tail**;
* on top of that, two shape problems are flagged: **underfilled** (fewer CTAs
  than SMs, or a last wave mostly empty) and **tile padding** (a GEMM whose M
  is far below its tile, so most tensor work is on zeros).

Each diagnosis comes with what-ifs in the exact form ``api.what_if`` takes.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

#: What a scheduler stall means, in one line.
STALL_MEANING = {
    "tma_wait": "warps wait for TMA loads to land: the tensor core is starved by memory",
    "long_scoreboard": "warps wait on global/local memory results (L1TEX round trips)",
    "lg_throttle": "the LSU queue for global memory is full",
    "tensor_wait": "warps wait for issued tensor-core work to complete",
    "tensor_throttle": "the tensor-core queue is full",
    "async_wait": "warps parked on an mbarrier for another warp's or engine's arrival (pipeline handoff)",
    "barrier": "warps wait at a CTA barrier (__syncthreads)",
    "short_scoreboard": "warps wait on shared memory / SFU / shuffle results",
    "mio_throttle": "the shared-memory / special-op queue is full",
    "math_pipe_throttle": "an ALU pipe is busy",
    "wait": "fixed-latency dependencies and compiler stall counts",
    "branch_resolving": "branch target resolution",
    "drain": "warps finishing their stores before exit",
    "selected": "issuing",
    "not_selected": "eligible but another warp issued",
}

STALL_CLASS = {
    "tma_wait": "memory_latency", "long_scoreboard": "memory_latency", "lg_throttle": "memory_latency",
    "tensor_wait": "tensor_pipeline", "tensor_throttle": "tensor_pipeline",
    "barrier": "sync", "short_scoreboard": "smem_mio", "mio_throttle": "smem_mio",
    "math_pipe_throttle": "alu", "wait": "alu", "branch_resolving": "alu", "drain": "tail",
}
_IGNORED = {"selected", "not_selected", "async_wait"}


def _dtype(name: str, spec: Optional[dict]) -> str:
    if spec and spec.get("dtype"):
        return spec["dtype"]
    m = re.search(r"_(bf16|fp16|fp8|fp6|fp4|int8|tf32)_", name)
    return m.group(1) if m else "bf16"


def peaks(gpu: dict, dtype: str) -> Dict[str, float]:
    fpc = gpu["tensor"]["flops_per_cycle_per_sm"].get(dtype) or gpu["tensor"]["flops_per_cycle_per_sm"]["bf16"]
    return {"tflops": fpc * gpu["sm_count"] * gpu["sm_clock_mhz"] * 1e6 / 1e12,
            "hbm_gbps": gpu["hbm"]["bw_gbps"] * gpu["hbm"].get("efficiency", 1.0),
            "l2_gbps": gpu.get("l2_bytes_per_cycle", 0) * gpu["sm_clock_mhz"] / 1e3}


def _dominant(sched: Dict[str, float], n: int = 4) -> List[dict]:
    tot = sum(sched.values()) or 1.0
    rows = sorted(((k, v / tot) for k, v in sched.items()), key=lambda kv: -kv[1])
    return [{"stall": k, "fraction": round(v, 4), "meaning": STALL_MEANING.get(k, k)} for k, v in rows[:n] if v > 0]


def _limiter(sched: Dict[str, float]) -> Optional[str]:
    rest = {k: v for k, v in sched.items() if k not in _IGNORED and v > 0}
    if not rest:
        return None
    return max(rest, key=rest.get)


def _padding(spec: Optional[dict], name: str) -> Optional[dict]:
    if not spec or spec.get("type") not in ("gemm", "grouped_gemm"):
        return None
    t = re.search(r"tile(\d+)x(\d+)", name)
    if not t:
        return None
    tm = int(t.group(1))
    ms = [spec["m"]] if spec["type"] == "gemm" else spec["ms"]
    used = sum(ms)
    padded = sum(math.ceil(m / tm) * tm for m in ms if m > 0) or 1
    eff = used / padded
    return {"tile_m": tm, "m": ms if len(ms) <= 8 else f"{len(ms)} groups, sum {used}", "useful_fraction": round(eff, 3)}


def _waves(info: Optional[dict], ctas_per_sm: int, sm_count: int) -> Optional[dict]:
    if not info or "grid" not in info:
        return None
    ctas = info["grid"][0] * info["grid"][1] * info["grid"][2]
    slots = max(1, ctas_per_sm) * sm_count
    waves = ctas / slots
    eff = waves / math.ceil(waves) if waves > 0 else 0.0
    return {"ctas": ctas, "slots": slots, "waves": round(waves, 3), "sm_fill": round(min(1.0, eff), 3)}


def _can_scale(rec: dict, field: str) -> bool:
    spec = rec.get("spec") or {}
    return "model" in spec and (field != "batch" or spec.get("phase", "decode") in ("decode", "prefill"))


def _world_fits(rec: dict, par: dict) -> bool:
    from .lower import Parallel
    p = Parallel(**{**(rec["spec"].get("parallel") or {}), **par})
    return p.world() <= rec["system"]["num_gpus"]


def kernel_levers(rec: dict, bound: str, flags: List[str], spec: Optional[dict], dtype: str) -> List[dict]:
    out: List[dict] = []
    s = rec.get("spec") or {}
    model = "model" in s
    batch = int(s.get("batch", 0) or 0)
    if bound == "hbm":
        if model and batch and _can_scale(rec, "batch"):
            out.append({"what_if": {"workload": {"batch": batch * 2}},
                        "why": "weights are streamed once per step; twice the batch reuses each byte twice (throughput up, "
                               "per-step latency up)"})
        if model and dtype in ("bf16", "fp16"):
            out.append({"what_if": {"workload": {"dtype": "fp8"}}, "why": "FP8 weights halve the bytes streamed"})
        par = s.get("parallel") or {}
        tp = par.get("tp", 1) * 2
        if model and par.get("ep", 1) == 1 and _world_fits(rec, {"tp": tp}):
            out.append({"what_if": {"workload": {"parallel": {"tp": tp}}},
                            "why": "more tensor parallelism splits the weights over more HBM stacks (adds all-reduces)"})
        out.append({"what_if": {"hardware": {"hbm_bw_gbps": "x1.5"}}, "why": "sensitivity: what 1.5x HBM bandwidth buys"})
        out.append({"what_if": {"hardware": {"hbm_bw_gbps": "x1.5", "l2_bytes_per_cycle": "x1.5"}},
                    "why": "sensitivity: HBM and L2 bandwidth together (streaming traffic crosses L2, so HBM alone can "
                           "hit the L2 ceiling)"})
    elif bound == "tensor":
        if model and dtype in ("bf16", "fp16"):
            out.append({"what_if": {"workload": {"dtype": "fp8"}}, "why": "FP8 doubles the dense tensor rate"})
        elif model and dtype == "fp8":
            out.append({"what_if": {"workload": {"dtype": "fp4"}}, "why": "FP4 doubles the dense tensor rate again"})
        out.append({"what_if": {"hardware": {"sm_clock_mhz": "x1.1"}}, "why": "sensitivity: a 10% higher sustained clock"})
    elif bound == "memory_latency":
        out.append({"what_if": {"hardware": {"hbm_latency_ns": "x0.8"}},
                    "why": "sensitivity: if 20% lower memory latency helps, more bytes in flight (stages, CTAs) would too"})
        out.append({"what_if": {"hardware": {"l2_hit_latency": "x0.8"}}, "why": "sensitivity to L2 latency"})
    elif bound == "smem_mio":
        out.append({"what_if": {"hardware": {"smem_bytes_per_cycle": "x2"}}, "why": "sensitivity: twice the shared-memory bandwidth"})
    elif bound == "sync":
        out.append({"what_if": {"hardware": {"sm_clock_mhz": "x1.1"}}, "why": "barrier-bound: only fewer syncs help; clock shows the floor"})
    if "underfilled" in flags and model and batch and _can_scale(rec, "batch"):
        out.append({"what_if": {"workload": {"batch": batch * 2}},
                    "why": "the grid leaves SMs idle; a larger batch gives the kernel more CTAs"})
    if "tile_padding" in flags and model and batch:
        out.append({"what_if": {"workload": {"batch": batch * 2}},
                    "why": "M is far below the GEMM tile: most tensor work multiplies padding; more rows fill the tile"})
    seen, uniq = set(), []
    for lv in out:
        key = repr(lv["what_if"])
        if key not in seen:
            seen.add(key)
            uniq.append(lv)
    return uniq


def classify(row: dict, spec: Optional[dict], info: Optional[dict], gpu: dict) -> dict:
    dtype = _dtype(row["name"], spec)
    pk = peaks(gpu, dtype)
    t = row["tflops"] / pk["tflops"] if pk["tflops"] else 0.0
    h = row["gbps"] / pk["hbm_gbps"] if pk["hbm_gbps"] else 0.0
    if h >= 0.7 and h >= t:
        bound = "hbm"
    elif t >= 0.6:
        bound = "tensor"
    else:
        lim = _limiter(row.get("sched", {}))
        bound = STALL_CLASS.get(lim, "latency") if lim else "pipeline_handoff"
    flags = []
    waves = _waves(info, row.get("ctas_per_sm", 1), gpu["sm_count"])
    if waves and waves["sm_fill"] < 0.75:
        flags.append("underfilled")
    pad = _padding(spec, row["name"])
    if pad and pad["useful_fraction"] < 0.5:
        flags.append("tile_padding")
    return {"dtype": dtype, "bound": bound, "flags": flags, "waves": waves, "padding": pad,
            "achieved": {"tflops": round(row["tflops"], 1), "pct_tensor_peak": round(100 * t, 1),
                         "gbps": round(row["gbps"]), "pct_hbm_achievable": round(100 * h, 1),
                         "pct_l2_bandwidth": round(100 * row["gbps"] / pk["l2_gbps"], 1) if pk["l2_gbps"] else None}}


def _diagnosis(name: str, c: dict, dom: List[dict]) -> str:
    a = c["achieved"]
    b = c["bound"]
    if b == "hbm":
        s = f"HBM-bound: moves model bytes at {a['gbps']} GB/s, {a['pct_hbm_achievable']}% of achievable HBM"
    elif b == "tensor":
        s = f"tensor-bound: {a['tflops']} TFLOPS, {a['pct_tensor_peak']}% of the dense {c['dtype']} peak"
    else:
        s = (f"{b.replace('_', ' ')}-bound: {a['pct_tensor_peak']}% of tensor peak and {a['pct_hbm_achievable']}% of HBM, "
             "so neither roof is reached")
    real = [d for d in dom if d["stall"] not in _IGNORED]
    if real:
        s += f"; past the mbarrier parking, schedulers wait on {real[0]['stall']} ({100 * real[0]['fraction']:.0f}%: {real[0]['meaning']})"
    if "tile_padding" in c["flags"]:
        s += f"; only {100 * c['padding']['useful_fraction']:.0f}% of each {c['padding']['tile_m']}-row tile is real rows"
    if "underfilled" in c["flags"]:
        w = c["waves"]
        s += f"; {w['ctas']} CTAs over {w['slots']} SM slots leave {100 * (1 - w['sm_fill']):.0f}% of the last wave empty"
    return s


def bottlenecks(rec: dict, top: int = 8) -> dict:
    rep = rec["report"]
    gpu = rec["system"]["gpus"][0]
    world = rep["world"]
    span = world * rep["makespan_us"] or 1.0
    items = []
    split: Dict[str, float] = {}
    for row in rep["kernels"]:
        e = rec["kernels"].get(row["name"], {})
        spec = e.get("spec")
        c = classify(row, spec, e.get("info"), gpu)
        share = row["total_us"] / span
        split[c["bound"]] = split.get(c["bound"], 0.0) + share
        if len(items) < top:
            dom = _dominant(row.get("sched", {}))
            items.append({"kernel": row["name"], "ops": e.get("ops", [])[:4], "count": row["count"],
                          "share_of_gpu_time": round(share, 4), "mean_us": round(row["mean_us"], 2),
                          **c, "tensor_util": round(row["tensor_util"], 3), "l2_hit_rate": round(row["l2_hit_rate"], 3),
                          "dominant_stalls": dom, "diagnosis": _diagnosis(row["name"], c, dom),
                          "levers": kernel_levers(rec, c["bound"], c["flags"], spec, c["dtype"])})
    coll = sum(r["collective_us"] for r in rep["ranks"]) / span
    idle = rep["bubbles"]["gpu_idle_fraction"]
    step_levers = []
    s = rec.get("spec") or {}
    if idle > 0.05:
        step_levers.append({"what_if": {"launch_gap_ns": 300},
                            "why": f"GPUs sit idle {100 * idle:.0f}% of the step between kernels; CUDA-graph replay "
                                   "(~0.3 us gaps) shows how much of that is launch overhead"})
    if coll > 0.05:
        if s.get("phase", "decode") == "decode" and "model" in s and not s.get("overlap") and (s.get("parallel") or {}).get("ep", 1) > 1:
            step_levers.append({"what_if": {"workload": {"overlap": True}},
                                "why": "two-micro-batch overlap hides all-to-all behind the other half's compute "
                                       "(but re-reads weights per half)"})
        step_levers.append({"what_if": {"hardware": {"nvlink_gbps_per_dir": "x2"}},
                            "why": f"collectives occupy {100 * coll:.0f}% of GPU time; sensitivity to 2x scale-up bandwidth"})
    order = sorted(split.items(), key=lambda kv: -kv[1])
    text = [f"Step {rep['makespan_us']:.0f} us on {rep['system']} ({world} GPUs)."]
    for b, sh in order[:3]:
        text.append(f"{100 * sh:.0f}% of GPU time is in {b.replace('_', ' ')}-bound kernels.")
    text.append(f"Collectives {100 * coll:.0f}%, idle between kernels {100 * idle:.0f}%.")
    if items:
        text.append(f"Biggest single cost: {items[0]['kernel']} ({100 * items[0]['share_of_gpu_time']:.0f}%), "
                    f"{items[0]['diagnosis']}.")
    ranked = []
    seen = set()
    for it in items:
        for lv in it["levers"]:
            k = repr(lv["what_if"])
            if k not in seen:
                seen.add(k)
                ranked.append({**lv, "targets": it["kernel"], "target_share": it["share_of_gpu_time"]})
    for lv in step_levers:
        k = repr(lv["what_if"])
        if k not in seen:
            seen.add(k)
            ranked.append({**lv, "targets": "step", "target_share": max(idle, coll)})
    ranked.sort(key=lambda lv: -lv["target_share"])
    return {"summary": " ".join(text), "time_split": {"kernels_by_bound": {k: round(v, 4) for k, v in order},
                                                      "collectives": round(coll, 4), "idle": round(idle, 4)},
            "items": items, "step_levers": step_levers, "suggested_what_ifs": ranked[:10],
            "rules": "hbm-bound at >=70% of achievable HBM, tensor-bound at >=60% of dense peak, otherwise the "
                     "dominant non-async scheduler stall; see rackscale.analysis"}


def kernel_diagnosis(d: dict, gpu: dict, spec: dict) -> dict:
    """Per-role reading of a ``kernel_detail`` result."""
    roles = []
    for role, r in (d.get("by_role") or {}).items():
        st = r["stalls"]
        tot = sum(st.values()) or 1
        waits = {k: v for k, v in st.items() if k != "selected"}
        top = max(waits, key=waits.get) if waits else None
        roles.append({"role": role, "issued": r["issued"], "warp_cycles": tot,
                      "dominant_wait": top, "fraction": round(waits.get(top, 0) / tot, 3) if top else 0,
                      "meaning": STALL_MEANING.get(top, top) if top else "issuing"})
    roles.sort(key=lambda r: -r["warp_cycles"])
    warps = []
    for i, st in enumerate(d.get("warp_by_index") or []):
        tot = sum(st.values()) or 1
        waits = {k: v for k, v in st.items() if k != "selected"}
        top = max(waits, key=waits.get) if waits else None
        issued = (d.get("issued_by_index") or [0] * (i + 1))[i] if i < len(d.get("issued_by_index") or []) else 0
        warps.append({"warp": i, "issued": issued, "dominant_wait": top,
                      "fraction": round(waits.get(top, 0) / tot, 3) if top else 0})
    if len(roles) <= 1 and warps:
        # Recorded paths carry no role names: the warp index within the CTA is the role.
        roles = [{"role": f"warp {w['warp']}", "issued": w["issued"], "warp_cycles": None,
                  "dominant_wait": w["dominant_wait"], "fraction": w["fraction"],
                  "meaning": STALL_MEANING.get(w["dominant_wait"], "issuing")} for w in warps]
    row = {"name": d["name"], "tflops": d["tflops"], "gbps": d["gbps"], "sched": d["sched"],
           "ctas_per_sm": d["ctas_per_sm"], "tensor_util": d["tensor_util"]}
    c = classify(row, spec, {"grid": d["grid"]}, gpu)
    dom = _dominant(d["sched"])
    text = _diagnosis(d["name"], c, dom)
    if not d.get("flops"):
        # Recorded kernels do not say what work they stand for; the tensor-core busy
        # fraction is the direct measure.
        text = (f"tensor cores busy {100 * d['tensor_util']:.0f}% of SMSP-cycles (a recorded kernel carries no FLOP "
                f"count, so the roofline is not available); " + text.split("; ", 1)[-1])
    mma = next((r for r in roles if "mma" in r["role"].lower()), None)
    if mma and mma["dominant_wait"] and mma["dominant_wait"] != "async_wait":
        text += f". The MMA-issuing role spends {100 * mma['fraction']:.0f}% of its cycles on {mma['dominant_wait']}"
    hot = d["top_pcs"][0] if d.get("top_pcs") else None
    if hot:
        text += f". Hottest instruction: [{hot['role']}] {hot['text'][:60]}"
    return {"bound": c["bound"], "flags": c["flags"], "achieved": c["achieved"], "waves": c["waves"],
            "padding": c["padding"], "role_anatomy": roles, "dominant_stalls": dom, "diagnosis": text}
