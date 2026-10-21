#!/usr/bin/env python3
"""Rack-scale sweeps (simulator only): Transformer vs Mamba-2 vs hybrid on GB200 NVL72.

Decode: one step of a tensor-parallel replica (TP GPUs of the NVL72, all-reduces
over NVLink5/NVLS) across batch x context. Every point records step time,
per-GPU tokens/s, whether the batch fits in HBM, MFU, the GPU-idle fraction, the
scheduler-cycle stall anatomy and the time per kernel class. Prefill: one
prompt of length L. Replicas are independent (data parallel), so per-GPU
throughput of the full rack equals that of one replica; ``--full-rack`` checks
this with all 72 GPUs simulated together.

    python research/ssm_vs_attention/sweep.py --family 70b --out results/sweep_70b.json
    python research/ssm_vs_attention/sweep.py --quick          # small grid, 8B models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

import rackscale as rs  # noqa: E402
from rackscale import lower, models  # noqa: E402

FAMILIES = {
    "8b": [models.LLAMA3_8B, models.MAMBA2_8B, models.MAMBA2_HYBRID_8B],
    "70b": [models.LLAMA3_70B, models.MAMBA2_70B, models.MAMBA2_HYBRID_70B],
}
HBM_GB = 186.0          # GB200 per-GPU HBM3e
HBM_USABLE = 0.90       # leave room for activations / allocator slack
# Decode is served from CUDA graphs (vLLM/SGLang default), so each kernel pays
# the graph-replay gap measured on B300 (0.55 us), not an eager launch.
DECODE_GAP_NS = 554.0
SYSTEM = "gb200_nvl72"


def kernel_class(name: str) -> str:
    n = name.lower()
    if "fmha" in n or "fa2_fwd" in n:
        return "attention"
    if "selective_state_update" in n:
        return "ssm_state_update"
    if n.startswith("ssd_"):
        return "ssd_" + n.split("_")[1] + ("_" + n.split("_")[2] if n.split("_")[1] in ("chunk", "state", "bmm") else "")
    if "conv1d" in n:
        return "conv1d"
    if "gemm" in n:
        return "gemm"
    return "other"


def fits(m, tp, batch, context) -> tuple:
    weights = 2.0 * (m.active_params(lm_head=True) + m.hidden * m.vocab) / tp  # + embedding table
    cache = batch * m.cache_bytes_per_seq(context) / tp
    return weights + cache <= HBM_USABLE * HBM_GB * 1e9, weights / 1e9, cache / 1e9


def summarize(rep: dict, wall: float) -> dict:
    by_class: dict = {}
    for k in rep["kernels"]:
        c = kernel_class(k["name"])
        by_class[c] = by_class.get(c, 0.0) + k["total_us"]
    # Collective time on the critical path: last rank posting -> completion.
    coll = sum((c["done_ps"] - c["last_post_ps"]) / 1e6 for c in rep.get("collectives", []))
    world = max(1, rep["world"])
    return dict(
        step_us=rep["makespan_us"], mfu=rep.get("mfu"), gpu_idle=rep["bubbles"]["gpu_idle_fraction"],
        tensor_idle=rep["bubbles"]["tensor_idle_fraction"], sched=rep["bubbles"]["sched"],
        kernel_us_per_gpu={k: v / world for k, v in by_class.items()},
        collective_us=coll, n_collectives=len(rep.get("collectives", [])),
        busbw_gbps=(sum(c["busbw_gbps"] for c in rep["collectives"]) / len(rep["collectives"])) if rep.get("collectives") else None,
        top_kernels=[dict(name=k["name"], us=k["total_us"] / world, count=k["count"] / world,
                                              tflops=k["tflops"], gbps=k["gbps"], tensor_util=k["tensor_util"])
                                         for k in rep["kernels"][:8]],
        sim_wall_s=wall)


def run_point(p: dict) -> dict:
    m = models.ALL[p["model"]]
    par = rs.Parallel(tp=p["tp"], comm_sms=20)
    if p["phase"] == "decode":
        ok, w, c = fits(m, p["tp"], p["batch"], p["context"])
        if not ok:
            # Does not fit in HBM: report the capacity limit, simulate nothing.
            return dict(p, fits_hbm=False, weights_gb_per_gpu=w, cache_gb_per_gpu=c)
    if p["phase"] == "decode":
        wl = lower.decode_step(m, par, p["batch"], p["context"], layers=p.get("layers"))
    else:
        wl = lower.prefill(m, par, p["batch"], p["seqlen"], layers=p.get("layers"))
    if p.get("world"):
        wl["spmd"]["world"] = p["world"]
    if p["phase"] == "decode" and p.get("launch_gap_ns") is None:
        p = dict(p, launch_gap_ns=DECODE_GAP_NS)
    t0 = time.time()
    try:
        rep = rs.simulate(SYSTEM, wl, fast=True, launch_gap_ns=p.get("launch_gap_ns")).raw
    except Exception as e:  # noqa: BLE001
        return dict(p, error=repr(e)[:300])
    out = dict(p, **summarize(rep, time.time() - t0))
    if p.get("layers"):
        # Scale a truncated stack back to the full depth (layers are identical
        # per type, so time is linear in the block count) -- not used by default.
        out["layers_simulated"] = p["layers"]
    if p["phase"] == "decode":
        out["tokens_per_s_per_gpu"] = p["batch"] / (out["step_us"] * 1e-6) / p["tp"]
        ok, w, c = fits(m, p["tp"], p["batch"], p["context"])
        out.update(fits_hbm=ok, weights_gb_per_gpu=w, cache_gb_per_gpu=c)
    else:
        out["tokens_per_s_per_gpu"] = p["batch"] * p["seqlen"] / (out["step_us"] * 1e-6) / p["tp"]
        out["ttft_ms"] = out["step_us"] / 1e3
    return out


def grid(family: str, quick: bool) -> list:
    ms = [m.name for m in FAMILIES[family]]
    tps = [1] if family == "8b" else [4, 8]
    batches = [1, 16, 64, 256] if quick else [1, 8, 32, 64, 128, 256, 512]
    ctxs = [1024, 16384, 131072] if quick else [1024, 4096, 16384, 32768, 65536, 131072]
    # Prefill beyond 32k is O(L^2) attention work that takes hours to simulate
    # cycle by cycle; the crossover sits well below it.
    lens = [1024, 16384] if quick else [1024, 2048, 4096, 8192, 16384, 32768]
    pts = []
    for m in ms:
        for tp in tps:
            for b in batches:
                for c in ctxs:
                    pts.append(dict(phase="decode", model=m, tp=tp, batch=b, context=c))
            for L in lens:
                pts.append(dict(phase="prefill", model=m, tp=tp, batch=1, seqlen=L))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), default="8b")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--full-rack", action="store_true", help="also simulate one point with all 72 GPUs (TP x DP)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pts = grid(a.family, a.quick)
    if a.full_rack:
        m = FAMILIES[a.family][0].name
        tp = 8 if a.family == "70b" else 1
        pts.append(dict(phase="decode", model=m, tp=tp, batch=64, context=16384, world=72, full_rack=True))
    # Small points first so results stream in; big ones run last.
    pts.sort(key=lambda p: p.get("batch", 1) * p.get("context", p.get("seqlen", 1)))
    os.chdir(ROOT)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(run_point, pts, chunksize=1):
            results.append(r)
            tag = f"{r['phase']:7s} {r['model']:20s} tp{r['tp']} " + (
                f"b{r['batch']:<4d} ctx{r['context']:<6d}" if r["phase"] == "decode" else f"L{r['seqlen']:<6d}")
            print(f"[{time.time() - t0:7.0f}s] {tag} " + (f"{r['step_us']:10.1f} us  {r['tokens_per_s_per_gpu']:10.0f} tok/s/GPU"
                                                         if "step_us" in r else r.get("error", "does not fit in HBM")), flush=True)
    out = a.out or os.path.join(HERE, "results", f"sweep_{a.family}{'_quick' if a.quick else ''}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(system=SYSTEM, family=a.family, hbm_gb=HBM_GB, hbm_usable=HBM_USABLE, points=results), f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
