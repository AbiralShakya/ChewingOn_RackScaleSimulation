"""Replay a capture on its own GPU and compare with what was measured.

    python validate_capture.py CAPTURE [--steps 1:] [--native native.json] [--system hgx_b300_8]

Prints, for the selected steps:

* every recorded kernel simulated alone vs its measured median duration;
* the step simulated four ways: measured kernel durations or re-simulated
  kernels (paths), each with the GPU free-running (CUDA-graph view, launch gap
  `--graph-gap-ns`) or with the captured host launch timeline;
* the measured GPU span of the captured steps (eager, under the tool), and the
  native step time when `--native` gives a JSON line from vllm_decode.py /
  sglang_decode.py --measure.
"""
import argparse
import json

from rackscale import record, replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--steps", default="1:")
    ap.add_argument("--system", default="hgx_b300_8")
    ap.add_argument("--graph-gap-ns", type=float, default=1000.0, help="launch gap inside a CUDA graph")
    ap.add_argument("--native", action="append", default=[], help="JSON line(s) from --measure runs")
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    lo, _, hi = a.steps.partition(":")
    steps = (int(lo or 0), int(hi) if hi else None)
    cap = record.load(a.capture)
    nsteps = None
    for p in cap["processes"]:
        n = sum(1 for o in p["regions"][0] if o["op"] == "step")
        if n:
            nsteps = len(range(n)[steps[0]:steps[1]])
    out = {"capture": a.capture, "device": cap["device"], "steps": nsteps, "kernels": [], "replay": {}}

    rows = replay.kernel_table(cap, a.system)
    out["kernels"] = rows
    print(f"{'kernel':<7} {'name':<60} {'n':>5} {'meas_us':>9} {'sim_us':>9} {'err%':>7}")
    for r in sorted(rows, key=lambda r: -(r.get("measured_us") or 0) * r.get("count", 0)):
        if "error" in r:
            print(f"{r['kernel']:<7} {r['name'][:60]:<60} error {r['error'][:80]}")
            continue
        print(f"{r['kernel']:<7} {r['name'][:60]:<60} {r['count']:>5} {r['measured_us'] or 0:>9.2f} "
              f"{r['sim_us']:>9.2f} {r['error_pct'] or 0:>+7.1f}")
    ok = [r for r in rows if r.get("error_pct") is not None]
    if ok:
        wsum = sum(r["measured_us"] * r["count"] for r in ok)
        werr = sum(abs(r["error_pct"]) * r["measured_us"] * r["count"] for r in ok) / wsum
        out["kernel_time_weighted_abs_err_pct"] = werr
        print(f"time-weighted mean |error| over {len(ok)} recorded kernels: {werr:.1f}%")

    m = replay.measured(cap, steps=steps)
    out["measured"] = {k: v for k, v in m.items() if k != "per_kernel_ns"}
    for mode in ("measured", "paths"):
        for host in (False, True):
            gap = None if host else a.graph_gap_ns
            rep = replay.replay(cap, a.system, mode=mode, host=host, steps=steps, launch_gap_ns=gap)
            key = f"{mode}{'+host' if host else ''}"
            out["replay"][key] = {"makespan_us": rep.makespan_us,
                                  "per_step_us": rep.makespan_us / nsteps if nsteps else None,
                                  "gpu_idle": rep.bubbles["gpu_idle_fraction"]}
    print(f"\ncaptured steps {a.steps} ({nsteps}): measured under the tool GPU span {m['gpu_span_us']:.0f} us, "
          f"kernel busy {m['kernel_busy_us']:.0f} us, host span {m['host_span_us']:.0f} us")
    for k, v in out["replay"].items():
        print(f"  replay {k:<15} {v['makespan_us']:>10.0f} us  per step {v['per_step_us'] or 0:>8.0f} us  "
              f"GPU idle {100 * v['gpu_idle']:.0f}%")
    for line in a.native:
        nat = json.loads(open(line).read().strip().splitlines()[-1]) if not line.startswith("{") else json.loads(line)
        out.setdefault("native", []).append(nat)
        print(f"  native ({'eager' if nat.get('eager') else 'graphs'}) decode step {nat['decode_step_us']:.0f} us")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
