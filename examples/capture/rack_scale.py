"""Scale a one-GPU capture of one tensor-parallel rank to a rack.

    python rack_scale.py CAPTURE --tp 8 [--steps 1:] [--graph-gap-ns 1000]

The capture must be one TP rank's share of the model (vllm_decode.py
--shape llama3-70b --shard-tp 8). Each configuration replays the same kernels
(re-simulated from their warp paths); the plan adds an all-reduce after every
row-parallel layer and the vocabulary-parallel embedding, sized from the
captured module outputs, and replicates the program over data-parallel
replicas.
"""
import argparse
import json

from rackscale import record, replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--steps", default="1:")
    ap.add_argument("--graph-gap-ns", type=float, default=1000.0)
    ap.add_argument("--mode", default="paths")
    ap.add_argument("--json")
    a = ap.parse_args()
    lo, _, hi = a.steps.partition(":")
    steps = (int(lo or 0), int(hi) if hi else None)
    cap = record.load(a.capture)
    n = sum(1 for o in cap["processes"][0]["regions"][0] if o["op"] == "step")
    nsteps = len(range(n)[steps[0]:steps[1]]) or 1
    configs = [
        ("1 GPU, TP rank alone (no communication)", "hgx_b300_8", {}),
        (f"HGX B300 8-GPU, TP{a.tp}", "hgx_b300_8", {"tp": a.tp}),
        (f"GB300 NVL72, TP{a.tp} x DP{72 // a.tp}", "gb300_nvl72", {"tp": a.tp, "dp": 72 // a.tp}),
        (f"GB300 NVL72, TP{a.tp} x DP{72 // a.tp}, NVLS all-reduce", "gb300_nvl72",
         {"tp": a.tp, "dp": 72 // a.tp, "algo": "nvls"}),
    ]
    rows = []
    for label, system, plan in configs:
        rep = replay.replay(cap, system, plan, mode=a.mode, steps=steps, launch_gap_ns=a.graph_gap_ns)
        r = rep.raw
        rank0 = r["ranks"][0]
        row = {"config": label, "system": system, "plan": plan, "world": r["world"],
               "step_us": rep.makespan_us / nsteps, "rank0_kernel_us": rank0["kernel_us"] / nsteps,
               "rank0_collective_us": rank0["collective_us"] / nsteps,
               "collectives": [{k: c[k] for k in ("kind", "algo", "bytes", "busbw_gbps") if k in c}
                               for c in r["collectives"][:3]]}
        rows.append(row)
        print(f"{label:<52} {row['world']:>3} GPUs  step {row['step_us']:8.0f} us  kernels {row['rank0_kernel_us']:7.0f} us"
              f"  all-reduce {row['rank0_collective_us']:7.0f} us")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"capture": a.capture, "steps": nsteps, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
