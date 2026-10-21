#!/usr/bin/env python3
"""Nsight Compute ground truth for one attention implementation.

Runs `fa_bench.py --once` under ncu, profiles every kernel of the timed call
(launches after the JIT/warm-up call), and writes one JSON record per kernel:
elapsed SM cycles, duration, instructions, DRAM/L2 bytes and the warp-stall
breakdown (warps stalled per issued instruction, by reason). Clocks are not
locked (`--clock-control none`) so the numbers match the timing runs; caches
are flushed before each kernel, which is also how the simulator starts.

  python ncu_attn.py --impl fa4 --b 4 --h 16 --s 4096 --d 128 --out results/b300/ncu_fa4_s4096.json
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = [
    "sm__cycles_elapsed.max", "sm__cycles_elapsed.avg", "smsp__cycles_active.avg", "gpu__time_duration.sum",
    "smsp__inst_executed.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum", "lts__t_sector_hit_rate.pct",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread", "launch__cluster_size",
    "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_xu.sum", "smsp__inst_executed_pipe_fma.sum", "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_lsu.sum", "smsp__inst_executed_pipe_uniform.sum",
]
STALLS = ["barrier", "branch_resolving", "dispatch_stall", "drain", "lg_throttle", "long_scoreboard",
          "math_pipe_throttle", "membar", "mio_throttle", "misc", "no_instruction", "not_selected", "selected",
          "short_scoreboard", "sleeping", "tex_throttle", "wait"]
METRICS += [f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio" for s in STALLS]


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return s


def run(ncu, bench_args, metrics):
    cmd = [ncu, "--clock-control", "none", "--cache-control", "all", "--profile-from-start", "off",
           "--csv", "--page", "raw", "--metrics", ",".join(metrics),
           sys.executable, os.path.join(HERE, "fa_bench.py"), "--once"] + bench_args
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout
    start = text.find('"ID"')
    if start < 0:
        raise SystemExit("ncu failed:\n" + p.stderr[-3000:] + text[-3000:])
    rows = list(csv.reader(io.StringIO(text[start:])))
    header, data = rows[0], rows[2:]
    return [dict(zip(header, r)) for r in data]


def main():
    # No -h/--help: `--h` is fa_bench's head count and would abbreviate to --help.
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--out", required=True)
    a, bench_args = ap.parse_known_args()
    ncu = os.environ.get("NCU", "ncu")
    # fa_bench --once wraps only the timed call in cudaProfilerStart/Stop.
    rows = run(ncu, bench_args, METRICS)
    out = []
    for r in rows:
        m = {k: num(v) for k, v in r.items() if "__" in k}
        out.append({
            "kernel": r.get("Kernel Name"),
            "ncu": {k: v for k, v in m.items() if not k.startswith("smsp__average_warps_issue_stalled")},
            "stalls_per_issue": {k.split("issue_stalled_")[1].split("_per_issue")[0]: v for k, v in m.items()
                                 if k.startswith("smsp__average_warps_issue_stalled") and isinstance(v, float)},
        })
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"bench_args": bench_args, "kernels": out}, f, indent=1)
    for k in out:
        print(f"{k['ncu'].get('gpu__time_duration.sum', 0)/1e3:9.1f} us  {k['kernel'][:100]}")


if __name__ == "__main__":
    main()
