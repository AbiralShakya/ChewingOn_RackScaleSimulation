#!/usr/bin/env python3
"""Sustained SM clock of back-to-back BF16 GEMMs, derived from data we already have.

The B300 power sweep (cutlass_power.sh) only recorded FP4 runs, so the simulator
ran every BF16 kernel at the full 2032 MHz. The Mamba-vs-attention study timed 54
cuBLAS BF16 GEMMs two ways on the same GPU (research/ssm_vs_attention):

* Nsight Compute gives each kernel's SM cycles (``sm__cycles_elapsed``), which do
  not depend on the clock;
* a CUDA-graph replay of the same GEMM back to back (rotating cold-L2 copies,
  20 iterations) gives its sustained wall time.

cycles / wall time is the clock the GPU held under sustained load. It falls from
the full clock for small GEMMs to about 1.2-1.3 GHz for the largest ones, which
is the board power cap at work. This script bins that clock by GEMM size (half a
decade of FLOPs per bin, median per bin) and writes the bins as power runs that
sit at the cap. ``rsim_hw::PowerSpec`` then reproduces each bin's clock exactly
and interpolates between them in log(FLOPs).

    python calibration/scripts/fit_sustained_clock.py \
        research/ssm_vs_attention/results/validation_b300.json \
        calibration/results/b300-sxm6-ac-sm103/raw/sustained_clock_bf16.json

merge.py picks the output up as ``model.power`` runs with ``"derived": true``.
Small GEMMs time slower under ncu than in a graph (per-kernel replay overhead),
which would put their clock above the maximum; those bins are clamped to it.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict

MIN_FLOPS = 1e10  # below this the kernel is weight-streaming; its clock does not matter


def fit(validation: dict, max_mhz: float, cap_w: float) -> dict:
    bins = defaultdict(list)
    for name, r in validation["cases"].items():
        if r.get("kind") != "gemm" or not r.get("graphed") or not r.get("ncu_kernels"):
            continue
        flops = 2.0 * r["m"] * r["n"] * r["k"]
        if flops < MIN_FLOPS:
            continue
        cycles = sum(k["cycles"] for k in r["ncu_kernels"])
        mhz = min(max_mhz, cycles / r["measured_us"])
        bins[math.floor(2 * math.log10(flops))].append((flops, mhz, name))
    runs = []
    for b in sorted(bins):
        pts = bins[b]
        runs.append({
            "gemm": "cublas_bf16_sustained",
            "dtype": "bf16",
            "derived": True,
            "flops": float(f"{statistics.median(p[0] for p in pts):.4g}"),
            "sm_mhz_median": round(statistics.median(p[1] for p in pts), 1),
            "board_w_median": cap_w,
            "cases": sorted(p[2] for p in pts),
        })
    return {
        "source": "Nsight Compute cycles / CUDA-graph wall time of cuBLAS BF16 GEMMs "
                  "(research/ssm_vs_attention, B300 GPU 3); board assumed at the power cap",
        "runs": runs,
    }


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    validation = json.load(open(argv[1]))
    out = fit(validation, max_mhz=2032.0, cap_w=1100.0)
    with open(argv[2], "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for r in out["runs"]:
        print(f"  {r['flops']:.2e} FLOP  {r['sm_mhz_median']:7.1f} MHz  ({len(r['cases'])} GEMMs)")


if __name__ == "__main__":
    main(sys.argv)
