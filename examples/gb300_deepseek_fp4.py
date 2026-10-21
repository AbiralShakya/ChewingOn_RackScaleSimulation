"""GB200 vs GB300 NVL72: DeepSeek-V3 decode (EP72) with BF16 vs FP4 weights.

Decode GEMMs are weight-bandwidth bound, so FP4 helps through bytes, not
FLOPs, and should stay clear of the GB300 power cap that limits FP4 prefill
(examples/gb300_fp4.py). Batch is per rank; context is 4k; fast mode.

    python examples/gb300_deepseek_fp4.py --layers 4
"""

import argparse
import copy

import rackscale as rs
from rackscale import lower, models

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--context", type=int, default=4096)
ap.add_argument("--batches", default="64,256")
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


print(f"{'system':<12} {'batch':>5} {'dtype':<5} {'step us':>9} {'GEMM us/GPU':>12} {'tok/s/GPU':>10}")
for batch in (int(b) for b in a.batches.split(",")):
    base = lower.decode_step(models.DEEPSEEK_V3, rs.Parallel(ep=72), batch, a.context, layers=a.layers)
    for system in ("gb200_nvl72", "gb300_nvl72"):
        for dtype in ("bf16", "fp4"):
            rep = rs.simulate(system, with_dtype(base, dtype), fast=True)
            gemm_us = sum(k["total_us"] for k in rep["kernels"] if "gemm" in k["name"]) / rep["world"]
            print(
                f"{system:<12} {batch:>5} {dtype:<5} {rep.makespan_us:9.1f} {gemm_us:12.1f} "
                f"{batch / rep.makespan_us * 1e6:10.0f}"
            )
