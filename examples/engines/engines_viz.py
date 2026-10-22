"""vLLM's many-kernel decode against the Hazy Research Llama-1B megakernel, on one B300, an HGX
B300 node and a GB300 NVL72 rack, in one viewer page.

    python examples/engines/engines_viz.py --caps DIR -o docs/viz/engines_llama1b_b300.html

DIR is the data/engine-captures branch's rackscale_caps/ with its tarballs unpacked:
  vllm_l1b_b1/{vllm_l1b_b1/, graphrun_vllm_l1b_b1.json}         vLLM batch 1 (`rackscale capture`)
  vllm_l1b_b32_nographs/{vllm_l1b_c/, graphrun_vllm_l1b.json}   vLLM batch 32
  megakernel_timing*.json                                        the megakernel's own per-task timing
  mk_allcta/pass_*.rpath.gz (optional)                           every CTA of one megakernel launch

What is simulated and what is measured:

* vLLM: the captured streams of one decode step are replayed. Every kernel whose warp path
  replays is re-simulated at cycle level; the rest (and every kernel's launch window) keep their
  duration from the native CUDA-graph run (Nsight Systems), so the step matches what vLLM serves.
* Megakernel: the per-SM task schedule comes from the kernel's own clock64 stamps on hardware
  (Hazy's timing build). With an all-CTA recording, the launch is also replayed at cycle level:
  the inter-SM counters become `GlobalWait`/`GlobalAdd`, and the instruction-fetch, counter and
  timing-flush accesses are probed to give the simulated schedule of the same tasks.
* Node and rack: data-parallel replicas (each GPU serves its own batch). Llama-1B is too small to
  shard, and the megakernel has no tensor-parallel version.
"""
import argparse
import glob
import gzip
import json
import os
import statistics
import subprocess
import sys
from collections import Counter

import rackscale as rs
from rackscale import record, replay

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "examples", "capture"))
from compare_graph_run import load_graph_run  # noqa: E402

B300 = "configs/hgx_b300_8_calibrated.yaml"
SYSTEMS = [("1 GPU", B300, 1), ("HGX B300 node", B300, 8), ("GB300 NVL72", "gb300_nvl72", 72)]
HAZY_OPS = ["NoOp", "RMS + QKV matvec + RoPE + KV append", "partial attention", "attention reduction",
            "O proj + residual", "RMS + up/gate matvec + SiLU", "down proj + residual", "RMS + LM head"]
HAZY_CATS = ["other", "gemm", "attn", "attn", "gemm", "gemm", "gemm", "gemm"]


def graph_calibrated(cap_dir, graphrun, system="hgx_b300_8"):
    """The capture with every kernel that the CUDA-graph run launches at its graph-run duration
    (used when its warp path does not replay), the rest left out; and the graph run's launch gap."""
    cap = record.load(cap_dir)
    replay.check_kernels(cap, system)  # writes kernel_check.json: which recordings replay
    cap = record.load(cap_dir)
    gk, gs = load_graph_run(graphrun)
    for k in cap["kernels"].values():
        g = gk.get((k["mangled"], tuple(k["grid"]), tuple(k["block"])))
        if g:
            k["measured_ns"] = {"median": statistics.median(g)}
            k["measured_override"] = True
        else:
            k["skip"] = True
    common = Counter(r["kernels"] for r in gs).most_common(1)[0][0]
    sel = [r for r in gs if r["kernels"] == common]
    span = statistics.median(r["span_us"] for r in sel)
    busy = statistics.median(r["busy_us"] for r in sel)
    gap = max(0.0, 1e3 * (span - busy) / max(1, common - 1))
    return cap, gap, {"kernels": common, "span_us": span, "busy_us": busy}


def bundle_runs(path):
    with open(path) as f:
        return json.load(f)["runs"]


def vllm_runs(cap_dir, graphrun, label, tmp, warp_sms):
    cap, gap, native = graph_calibrated(cap_dir, graphrun)
    runs = []
    for sname, system, dp in SYSTEMS:
        wl = replay.to_workload(cap, {"dp": dp}, steps=(1, 2), name=f"{label} · {sname}")
        out = os.path.join(tmp, f"{label}_{dp}.json".replace(" ", "_"))
        rs.viz(system, wl, out, name=f"{label} · {sname}", detail_ranks=[0], warp_sms=warp_sms,
               launch_gap_ns=gap)
        run = bundle_runs(out)[0]
        run["span_sms"] = {o["name"]: o["kernel"]["sms"] for st in wl["spmd"]["program"]["streams"] for o in st
                           if o.get("op") == "kernel" and o["kernel"].get("type") == "fixed"}
        run["note"] = (f"One vLLM decode step replayed from its capture. Native (CUDA graphs, Nsight Systems): "
                       f"{native['kernels']} kernels, GPU span {native['span_us']:.0f} µs, busy "
                       f"{native['busy_us']:.0f} µs per step. Kernels whose warp path replays are "
                       f"re-simulated at cycle level and carry SM, warp and SASS detail; the others keep "
                       f"their graph-run duration and show as measured spans.")
        runs.append(run)
    return runs


def u64(lo, hi):
    return (int(hi) & 0xFFFFFFFF) << 32 | (int(lo) & 0xFFFFFFFF)


def hazy_measured(path):
    """Per-SM task schedule from the megakernel's own timing slots (see hazy_src in the captures):
    0 controller start, 4 controller end, 45 inter-SM wait done, 5 loader start. Times are clock64
    cycles from each CTA's start; CTAs are aligned with their absolute clock64 (one clock domain)."""
    d = json.load(open(path))
    mhz = (d.get("timed_token", {}).get("sm_clock_MHz_from_clock64_vs_globaltimer") or {}).get("median") \
        or float(d.get("nvidia_smi_before", {}).get("clocks.sm_MHz") or 1920)
    t0 = min(s["cta_start_clock64"] for s in d["sms"])
    tasks, ops = [], HAZY_OPS
    for s in d["sms"]:
        base = s["cta_start_clock64"] - t0
        for ins in s["instructions"]:
            op, ts = ins["opcode"], ins["timings"]
            if op == 0 or not ts[0] or not ts[4]:
                continue
            start, end = ts[0], ts[4]
            ready = ts[45] if ts[45] else (ts[5] or start)
            layer = ins["args"].get("layer_idx", 0) if isinstance(ins.get("args"), dict) else 0
            ns = lambda c: round((base + c) * 1e3 / mhz, 2)  # noqa: E731
            tasks += [s["sm"], op, layer, ns(start), ns(max(start, ready)), ns(end)]
    span = max(tasks[i + 5] for i in range(0, len(tasks), 6))
    return {"source": "measured", "sm_count": d["num_sms"], "ops": ops, "cats": HAZY_CATS, "tasks": tasks,
            "span_ns": span, "clock_mhz": mhz, "wall": d.get("wall"), "prompt_len": d.get("prompt_len")}


def megakernel_runs(timing_json, tmp, label="Hazy megakernel, batch 1", wall_json=None):
    mk = hazy_measured(timing_json)
    us = mk["span_ns"] / 1e3
    if wall_json and os.path.exists(wall_json):
        w = json.load(open(wall_json)).get("wall") or {}
        us = w.get("megakernel_us_median", us)
    runs = []
    for sname, system, dp in SYSTEMS:
        wl = {"name": f"{label} · {sname}", "spmd": {"world": dp, "program": {"streams": [[
            {"op": "kernel", "name": "hazy_llama1b_megakernel (persistent, 148 CTAs)",
             "kernel": {"type": "fixed", "ns": us * 1e3, "sms": mk["sm_count"]}}]]}}}
        out = os.path.join(tmp, f"mk_{dp}.json")
        rs.viz(system, wl, out, name=f"{label} · {sname}", detail_ranks=[0], warp_sms=[])
        run = bundle_runs(out)[0]
        # a fixed-duration kernel has no entry in run["kernels"]: the schedule hangs off the run (kernel -1)
        run["megakernels"] = [dict(mk, kernel=-1, rank=0, label="Llama-3.2-1B decode token")]
        run["note"] = (f"One decode token of the Hazy Research low-latency Llama-1B megakernel: a single "
                       f"launch of {mk['sm_count']} persistent CTAs. Its per-SM task schedule is measured "
                       f"on B300 by the kernel's own clock64 stamps. Open the megakernel from the table below.")
        runs.append(run)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tmp", default="/tmp/engines_viz")
    ap.add_argument("--warp-sms", default="0,1")
    ap.add_argument("--only", choices=["vllm", "mk"], help="build only one engine (testing)")
    a = ap.parse_args()
    os.makedirs(a.tmp, exist_ok=True)
    warp_sms = [int(x) for x in a.warp_sms.split(",") if x]
    runs = []
    if a.only != "vllm":
        t = os.path.join(a.caps, "megakernel_timing_ctx512.json")
        t = t if os.path.exists(t) else os.path.join(a.caps, "megakernel_timing.json")
        if os.path.exists(t):
            runs += megakernel_runs(t, a.tmp, wall_json=os.path.join(a.caps, os.path.basename(t).replace(
                "timing", "wall_stock")))
        else:
            print(f"no megakernel timing in {a.caps}; the page has vLLM only", file=sys.stderr)
    if a.only != "mk":
        runs += vllm_runs(os.path.join(a.caps, "vllm_l1b_b1", "vllm_l1b_b1"),
                          os.path.join(a.caps, "vllm_l1b_b1", "graphrun_vllm_l1b_b1.json"),
                          "vLLM, batch 1", a.tmp, warp_sms)
        runs += vllm_runs(os.path.join(a.caps, "vllm_l1b_b32_nographs", "vllm_l1b_c"),
                          os.path.join(a.caps, "vllm_l1b_b32_nographs", "graphrun_vllm_l1b.json"),
                          "vLLM, batch 32", a.tmp, warp_sms)
    # one GPU first, then the node, then the rack: the engine comparison card uses GPU 0 of every run
    order = {s[0]: i for i, s in enumerate(SYSTEMS)}
    runs.sort(key=lambda r: order.get(r["name"].split(" · ")[-1], 9))
    bundle = os.path.join(a.tmp, "engines.json")
    with open(bundle, "w") as f:
        json.dump({"format": "rsviz/1", "title": "Engines on B300: vLLM vs a megakernel", "generator": "engines_viz.py",
                   "runs": runs}, f)
    rsim = os.path.join(REPO, "target", "release", "rsim")
    subprocess.run([rsim, "viz", "merge", bundle, "--title", "vLLM vs a megakernel, Llama-3.2-1B decode on B300",
                    "-o", a.out], check=True)


if __name__ == "__main__":
    main()
