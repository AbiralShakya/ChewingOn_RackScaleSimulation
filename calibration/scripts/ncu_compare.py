#!/usr/bin/env python3
"""Cross-check bench_mma_sync throughput against Nsight Compute counters.

The bench times each throughput kernel with %clock64 (CTA span, CtaTimer) and
reports FLOPs/clk/SM. Here ncu profiles the same launches and we recompute the
rate from hardware counters alone:

    rate_ncu = tensor_inst_per_sm * flops_per_inst / sm__cycles_active.avg

where tensor_inst_per_sm = sm__inst_executed_pipe_tensor_op_*.sum / sm_count
(warp-level MMA instructions). The ratio rate_ncu / rate_clock should sit near
1; a large gap means the clock-based timing window is wrong (e.g. it misses
warps, or the kernel has a long tail). ncu also gives the tensor pipe's
active-cycle utilization, which should approach 100% at the bench's best
(warps, ILP) point.

Usage: ncu_compare.py build/bench_mma_sync results/<slug>/raw/mma_sync.json [out.json]
"""
import csv
import io
import json
import os
import re
import subprocess
import sys

WANT = [
    "sm__cycles_active.avg",
    "sm__cycles_elapsed.max",
    "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "sm__inst_executed_pipe_tensor_op_imma.sum",
    "sm__inst_executed_pipe_tensor_op_dmma.sum",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
]

# Ops whose MMA counts as two pipe instructions or none: filled from the data
# by checking inst counts against the launch's known MMA count.
KNAME = re.compile(r"mma_thr_kernel<(\w+), (\d+)>")


def available(binary):
    out = subprocess.run(["ncu", "--query-metrics", "--csv"], capture_output=True, text=True).stdout
    have = set()
    for line in out.splitlines():
        name = line.split(",")[0].strip('"')
        have.add(name)
    keep = []
    for m in WANT:
        base = m.split(".")[0]
        if base in have:
            keep.append(m)
    return keep


def run_ncu(binary, metrics):
    cmd = ["ncu", "--csv", "--page", "raw", "--metrics", ",".join(metrics),
           "-k", "regex:mma_thr_kernel", binary]
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CAL_THR_REPS": "1"})
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-4000:])
        sys.exit(p.returncode)
    text = p.stdout
    start = text.find('"ID"')
    rows = list(csv.DictReader(io.StringIO(text[start:])))
    return [r for r in rows if r.get("ID", "").isdigit()]


def num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return float("nan")


def main():
    binary, bench_json = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    bench = json.load(open(bench_json))
    sm_count = bench["device"]["sm_count"]
    metrics = available(binary)
    rows = run_ncu(binary, metrics)
    # The bench runs every (warps, ilp) point 3 times; keep the best ncu repeat.
    best = {}
    for r in rows:
        m = KNAME.search(r["Kernel Name"])
        if not m:
            continue
        op, ilp = m.group(1), int(m.group(2))
        block = r.get("Block Size", "")
        warps = int(re.findall(r"\d+", block)[0]) // 32 if block else None
        cyc = num(r.get("sm__cycles_active.avg"))
        inst = sum(num(r.get(k, 0)) for k in metrics if k.startswith("sm__inst_executed_pipe_tensor"))
        util = max([num(r.get(k)) for k in metrics if "cycles_active.avg.pct" in k] or [float("nan")])
        info = bench["mma"].get(op)
        if not info or warps is None:
            continue
        rate = inst / sm_count * info["flops_per_mma"] / cyc
        key = (op, warps, ilp)
        if key not in best or rate > best[key]["rate_ncu"]:
            best[key] = {"rate_ncu": rate, "tensor_pipe_util_pct": util, "sm_cycles_active": cyc,
                         "tensor_inst_per_sm": inst / sm_count}
    report = {"bench": "ncu_compare_mma_sync", "metrics": metrics, "ops": {}}
    for op, info in bench["mma"].items():
        pts = []
        for p in info["sweep"]:
            b = best.get((op, p["warps"], p["ilp"]))
            if not b:
                continue
            pts.append({"warps": p["warps"], "ilp": p["ilp"], "rate_clock": p["flops_per_clk_per_sm"],
                        **b, "ratio_ncu_over_clock": b["rate_ncu"] / p["flops_per_clk_per_sm"]})
        if not pts:
            continue
        top = max(pts, key=lambda q: q["rate_clock"])
        report["ops"][op] = {"points": pts, "best": top}
        print(f"{op:22s} clock {top['rate_clock']:8.1f}  ncu {top['rate_ncu']:8.1f}  "
              f"ratio {top['ratio_ncu_over_clock']:.3f}  pipe util {top['tensor_pipe_util_pct']:.1f}%")
    s = json.dumps(report, indent=1)
    if out_path:
        open(out_path, "w").write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()
