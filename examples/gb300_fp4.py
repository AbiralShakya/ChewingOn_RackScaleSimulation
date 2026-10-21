"""GB200 vs GB300 NVL72: Llama-3-405B prefill (TP8 x 9 replicas) in BF16 and FP4.

GB300 uses the measured B300 calibration: the K=96 "ultra" FP4 UMMA, and
board-power-capped clocks fitted to CUTLASS FP4 runs. The power cap is why
FP4 does not deliver its paper 1.5x over GB200.

    python examples/gb300_fp4.py --layers 2
"""

import argparse
import copy

import rackscale as rs
from rackscale import lower, models

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--seqlen", type=int, default=8192)
a = ap.parse_args()


def with_dtype(wl: dict, dtype: str) -> dict:
    """Run every GEMM of the workload in `dtype` (weights + activations)."""
    wl = copy.deepcopy(wl)
    for stream in wl["spmd"]["program"]["streams"]:
        for op in stream:
            k = op.get("kernel", {})
            if k.get("type") in ("gemm", "grouped_gemm"):
                k["dtype"] = dtype
    wl["name"] += f" [{dtype}]"
    return wl


base = lower.prefill(models.LLAMA3_405B, rs.Parallel(tp=8, dp=9), batch=1, seqlen=a.seqlen, layers=a.layers)
rows = []
for system in ("gb200_nvl72", "gb300_nvl72"):
    for dtype in ("bf16", "fp4"):
        rep = rs.simulate(system, with_dtype(base, dtype), fast=True)
        gemm_us = sum(k["total_us"] for k in rep["kernels"] if "gemm" in k["name"]) / rep["world"]
        rows.append((system, dtype, rep.makespan_us, gemm_us))
        print(f"{system:<12} {dtype:<5} step {rep.makespan_us:9.1f} us   GEMM time/GPU {gemm_us:9.1f} us")
