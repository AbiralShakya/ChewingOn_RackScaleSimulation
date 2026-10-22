#!/usr/bin/env python3
"""Synthetic GEMMs against measured cuBLAS BF16 GEMMs, by size class.

Simulates every GEMM case of the Mamba-vs-attention study
(research/ssm_vs_attention/results/validation_<gpu>.json) on the calibrated
GPU, back to back with the CUDA-graph launch gap as it was timed, and
compares with the measured CUDA-graph time. The study's own ``sim_us`` (the
simulator at the commit that produced the file) is shown as "before".

    maturin develop --release
    python calibration/scripts/check_gemms.py --gpu b300 [--min-flops 1e10]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "research", "ssm_vs_attention"))

import validate as V  # noqa: E402

CLASSES = [(0, 1e10, "decode (< 1e10 FLOP)"), (1e10, 1e11, "mid"), (1e11, float("inf"), "large (>= 1e11 FLOP)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="b300", choices=sorted(V.GPUS))
    ap.add_argument("--min-flops", type=float, default=0.0)
    a = ap.parse_args()
    d = json.load(open(os.path.join(ROOT, "research", "ssm_vs_attention", "results", f"validation_{a.gpu}.json")))
    g = V.GPUS[a.gpu]
    rows = []
    for name, r in d["cases"].items():
        flops = 2.0 * r.get("m", 0) * r.get("n", 0) * r.get("k", 0)
        if r["kind"] != "gemm" or flops < a.min_flops:
            continue
        us = V.simulate_case(g["system"], V.ops_for(name, r), V.GRAPH_GAP_NS[a.gpu])["us"]
        now, before = us / r["measured_us"] - 1, r["sim_us"] / r["measured_us"] - 1
        rows.append((flops, now, before))
        print(f"{name:30s} measured {r['measured_us']:8.1f} us  sim {us:8.1f} us  {100 * now:+6.1f}%"
              f"   (before {100 * before:+6.1f}%)", flush=True)
    print(f"\n{'class':22s} {'n':>3s}  {'median |err|':>12s}  {'mean err':>9s}   before: {'median':>6s} {'mean':>7s}")
    for lo, hi, label in CLASSES:
        sel = [x for x in rows if lo <= x[0] < hi]
        if sel:
            med = lambda i: 100 * statistics.median(abs(x[i]) for x in sel)
            mean = lambda i: 100 * statistics.mean(x[i] for x in sel)
            print(f"{label:22s} {len(sel):3d}  {med(1):11.1f}%  {mean(1):+8.1f}%   {med(2):13.1f}% {mean(2):+6.1f}%")


if __name__ == "__main__":
    main()
