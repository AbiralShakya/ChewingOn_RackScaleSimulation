"""Compare a capture's replay with a native CUDA-graph run of the same engine.

    python compare_graph_run.py CAPTURE graph_run.sqlite [--steps 1:] [--system hgx_b300_8] [--json out.json]
    python compare_graph_run.py - graph_run.sqlite --summarize graph_run.json   # small summary to copy

The capture (taken without CUDA graphs) and the graph run (Nsight Systems with
`--cuda-graph-trace=node`, see nsys_graph_steps.py) execute the same kernels;
the graph run measures them back to back on a busy GPU, the way they are
served. This joins the two by kernel name and grid and reports:

* per kernel: the graph run's median duration against the replay of the
  captured warp path (stream timing: no profiler window);
* per decode step: the graph run's GPU span and busy time against the replay
  of one captured step, with the launch gap set to the graph run's.
"""
import argparse
import json
import sqlite3
import statistics
from collections import defaultdict

from nsys_graph_steps import steps as graph_steps
from rackscale import record, replay


def graph_kernels(path):
    """(mangled or demangled name, grid, block) -> durations (ns) of kernels launched by graphs."""
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    graph_corr = {corr for corr, nid in c.execute("select correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")
                  if names.get(nid, "").startswith("cudaGraphLaunch")}
    out = defaultdict(list)
    q = ("select correlationId, start, end, mangledName, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ "
         "from CUPTI_ACTIVITY_KIND_KERNEL")
    for corr, a, b, m, dm, gx, gy, gz, bx, by, bz in c.execute(q):
        if corr in graph_corr:
            out[(names.get(m) or names.get(dm), (gx, gy, gz), (bx, by, bz))].append(b - a)
    return out


def load_graph_run(path):
    """Kernel durations and steps of a graph run, from its sqlite or a summary written by --summarize."""
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        return {(m, tuple(g), tuple(b)): v for m, g, b, v in d["kernels"]}, d["steps"]
    return graph_kernels(path), graph_steps(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("graph_sqlite")
    ap.add_argument("--steps", default="1:")
    ap.add_argument("--system", default="hgx_b300_8")
    ap.add_argument("--json")
    ap.add_argument("--summarize", help="only write the graph run's summary JSON here (small, portable)")
    a = ap.parse_args()
    if a.summarize:
        gk, gs = load_graph_run(a.graph_sqlite)
        with open(a.summarize, "w") as f:
            json.dump({"kernels": [[m, list(g), list(b), v] for (m, g, b), v in gk.items()], "steps": gs}, f)
        return
    lo, _, hi = a.steps.partition(":")
    steps = (int(lo or 0), int(hi) if hi else None)
    cap = record.load(a.capture)
    gk, gs = load_graph_run(a.graph_sqlite)
    rows = {r["kernel"]: r for r in replay.check_kernels(cap, a.system)}

    # kernels of one captured step, with their launch counts
    region = cap["processes"][0]["regions"][0]
    sel = replay._select(region, steps)
    nsteps = max(1, sum(1 for o in sel if o["op"] == "step"))
    per_step = defaultdict(int)
    for o in sel:
        if o["op"] == "kernel":
            per_step[o["kernel"]] += 1
    table = []
    for kid, n in per_step.items():
        k = cap["kernels"][kid]
        key = (k["mangled"], tuple(k["grid"]), tuple(k["block"]))
        g = gk.get(key)
        r = rows.get(kid, {})
        table.append({"kernel": kid, "name": replay.short_name(k["name"]), "per_step": n / nsteps,
                      "graph_us": statistics.median(g) / 1e3 if g else None,
                      "eager_us": (k.get("measured_ns") or {}).get("median", 0) / 1e3,
                      "sim_us": r.get("sim_stream_us"), "variant": r.get("variant"), "error": r.get("error")})
    table.sort(key=lambda t: -(t["graph_us"] or t["eager_us"] or 0) * t["per_step"])
    print(f"{'kernel':<7} {'name':<52} {'/step':>5} {'graph_us':>9} {'eager_us':>9} {'sim_us':>8} {'err%':>7}")
    tw = te = 0.0
    for t in table:
        err = (100 * (t["sim_us"] - t["graph_us"]) / t["graph_us"]) if t["sim_us"] and t["graph_us"] else None
        t["error_pct"] = err
        if err is not None:
            tw += t["graph_us"] * t["per_step"]
            te += abs(err) * t["graph_us"] * t["per_step"]
        sim = f"{t['sim_us']:8.2f}" if t["sim_us"] else ("  failed" if t["error"] else "       -")
        print(f"{t['kernel']:<7} {t['name'][:52]:<52} {t['per_step']:>5.1f} "
              f"{(t['graph_us'] or 0):>9.2f} {t['eager_us']:>9.2f} {sim} {('%+7.1f' % err) if err is not None else '      -'}")
    matched = sum((t["graph_us"] or 0) * t["per_step"] for t in table)
    print(f"graph-run kernel time per step matched to the capture: {matched:.0f} us; time-weighted |error| of "
          f"replayed kernels {te / tw if tw else 0:.1f}% over {100 * tw / matched if matched else 0:.0f}% of it")

    from collections import Counter
    common = Counter(r["kernels"] for r in gs).most_common(1)[0][0]
    gsel = [r for r in gs if r["kernels"] == common]
    span = statistics.median(r["span_us"] for r in gsel)
    busy = statistics.median(r["busy_us"] for r in gsel)
    gap_ns = max(0.0, 1e3 * (span - busy) / max(1, common - 1))
    out = {"graph_step": {"kernels": common, "span_us": span, "busy_us": busy, "gap_ns": gap_ns},
           "kernels": table, "replay": {}}
    for mode in ("measured", "paths"):
        rep = replay.replay(cap, a.system, mode=mode, steps=steps, launch_gap_ns=gap_ns)
        out["replay"][mode] = rep.makespan_us / nsteps
    print(f"graph run: {common} kernels per step, GPU span {span:.0f} us, busy {busy:.0f} us "
          f"(gap {gap_ns:.0f} ns per launch)")
    print(f"replay of the capture, per step, at that gap: measured kernels {out['replay']['measured']:.0f} us, "
          f"re-simulated kernels {out['replay']['paths']:.0f} us")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
