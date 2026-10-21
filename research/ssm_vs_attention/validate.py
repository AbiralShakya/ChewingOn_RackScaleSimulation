#!/usr/bin/env python3
"""Single-GPU validation: simulate every case of ``hw/bench_kernels.py`` on the
calibrated GPU model and compare with the measured time (CUDA-graph timing,
cold L2) and, where available, Nsight Compute's per-kernel cycles.

    python research/ssm_vs_attention/validate.py --gpu b300
    python research/ssm_vs_attention/validate.py --gpu a100

Writes ``results/validation_<gpu>.json``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

import rackscale as rs  # noqa: E402

GPUS = {
    "b300": dict(system=os.path.join(HERE, "configs", "b300_calibrated.yaml"), data="b300"),
    "a100": dict(system=os.path.join(HERE, "configs", "a100_calibrated.yaml"), data="a100pcie"),
    "b300tp8": dict(system=os.path.join(HERE, "configs", "b300_calibrated.yaml"), data="b300_tp8"),
    "a100sxm4": dict(system=os.path.join(HERE, "configs", "a100sxm4_calibrated.yaml"), data="a100sxm4"),
}

D_MODEL = 4096
M2 = dict(d_inner=8192, headdim=64, nheads=128, d_state=128, ngroups=8, d_conv=4, chunk=256)
ATT = dict(hq=32, hkv=8, d=128)
SSD_STAGES = ("chunk_cumsum", "chunk_state", "state_passing", "bmm_chunk", "chunk_scan")


def ops_for(cid: str, rec: dict) -> list:
    """The simulator kernels that stand for one measured case."""
    k = rec["kind"]
    conv_dim = M2["d_inner"] + 2 * M2["ngroups"] * M2["d_state"]
    ssm = dict(heads=M2["nheads"], headdim=M2["headdim"], d_state=M2["d_state"], groups=M2["ngroups"])
    if k == "ssm_state_update":
        ssm = dict(ssm, heads=rec.get("nheads", ssm["heads"]), groups=rec.get("ngroups", ssm["groups"]))
        return [dict(type="ssm_state_update", batch=rec["batch"], **ssm)]
    if k == "conv1d_update":
        return [dict(type="conv1d", batch=rec["batch"], dim=rec.get("dim", conv_dim), seqlen=1, width=M2["d_conv"])]
    if k == "conv1d_fwd":
        return [dict(type="conv1d", batch=rec["batch"], dim=conv_dim, seqlen=rec["seqlen"], width=M2["d_conv"])]
    if k == "ssd_prefill":
        return [dict(type="ssd", stage=s, batch=rec["batch"], seqlen=rec["seqlen"], chunk=M2["chunk"], **ssm)
                for s in SSD_STAGES]
    if k == "attn_decode":
        return [dict(type="attention", batch=rec["batch"], heads_q=rec.get("hq", ATT["hq"]), heads_kv=rec.get("hkv", ATT["hkv"]),
                     sq=1, skv=rec["context"], d=ATT["d"]),
                # FA2 and FA4 both finish split-KV decode with a combine kernel
                # over the fp32 partial outputs and log-sum-exps.
                dict(type="elementwise", n=rec["batch"] * rec.get("hq", ATT["hq"]) * ATT["d"] * 4, inputs=1, elem_bytes=4.0,
                     flops_per_elem=4)]
    if k == "attn_prefill":
        return [dict(type="attention", batch=rec["batch"], heads_q=ATT["hq"], heads_kv=ATT["hkv"], sq=rec["seqlen"],
                     skv=rec["seqlen"], d=ATT["d"], causal=True)]
    if k == "gemm":
        return [dict(type="gemm", m=rec["m"], n=rec["n"], k=rec["k"])]
    raise ValueError(k)


GRAPH_GAP_NS = {"b300": 554.0, "b300tp8": 554.0, "a100": 940.0, "a100sxm4": 940.0}


def simulate_case(system: str, ops: list, gap_ns: float) -> dict:
    """Kernels back to back as in a CUDA-graph replay (the measured timing),
    so each launch pays the graph's per-kernel gap."""
    wl = {"name": "case", "spmd": {"world": 1, "program": {"streams": [
        [{"op": "kernel", "name": f"k{i}", "kernel": o} for i, o in enumerate(ops)]]}}}
    r = rs.simulate(system, wl, launch_gap_ns=gap_ns).raw
    ks = [dict(name=k["name"], us=k["total_us"], tflops=k["tflops"], gbps=k["gbps"], tensor_util=k["tensor_util"])
          for k in r["kernels"]]
    return dict(us=r["makespan_us"], kernel_us=sum(k["us"] for k in ks), kernels=ks, sched=r["bubbles"]["sched"])


def read_ncu(path: str) -> dict:
    """ncu --csv --page raw with --print-nvtx-rename kernel: one row per kernel,
    kernel name = the enclosing NVTX range (case/<id>)."""
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    start = text.find('"ID"')
    if start < 0:
        return {}
    rows = list(csv.reader(io.StringIO(text[start:])))
    header, data = rows[0], rows[2:]
    out: dict = {}
    for row in data:
        r = dict(zip(header, row))
        name = r.get("Kernel Name", "")
        m = re.search(r"case_(\w+)", name)
        if not m:
            continue

        def num(key):
            try:
                return float(r.get(key, "nan").replace(",", ""))
            except ValueError:
                return float("nan")

        out.setdefault(m.group(1), []).append(dict(
            ns=num("gpu__time_duration.sum"), cycles=num("sm__cycles_elapsed.max"),
            insts=num("smsp__inst_executed.sum"), dram_read=num("dram__bytes_read.sum"),
            dram_write=num("dram__bytes_write.sum"), grid=num("launch__grid_size"), block=num("launch__block_size"),
            regs=num("launch__registers_per_thread"),
            smem=num("launch__shared_mem_per_block_dynamic") + num("launch__shared_mem_per_block_static"),
            warps_active_pct=num("sm__warps_active.avg.pct_of_peak_sustained_active"),
            stalls={k.split("issue_stalled_")[1].split("_per_issue")[0]: num(k) for k in r
                    if "issue_stalled_" in k}))
    return out


def read_names(path: str) -> list:
    """Kernel names (mangled) in launch order from the second ncu pass."""
    if not os.path.exists(path):
        return []
    text = open(path).read()
    start = text.find('"ID"')
    if start < 0:
        return []
    rows = list(csv.reader(io.StringIO(text[start:])))
    header = rows[0]
    return [dict(zip(header, r)).get("Kernel Name", "") for r in rows[2:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=sorted(GPUS), required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None, help="regex of cases to leave out (e.g. the largest, slow to simulate)")
    a = ap.parse_args()
    cfg = GPUS[a.gpu]
    ddir = os.path.join(HERE, "data", cfg["data"])
    timing = json.load(open(os.path.join(ddir, "timing.json")))
    # FA4 decode re-timed with split-KV (num_splits=0) supersedes the first pass.
    redo = os.path.join(ddir, "timing_decode.json")
    if os.path.exists(redo):
        timing["cases"].update(json.load(open(redo))["cases"])
    ncu = read_ncu(os.path.join(ddir, "ncu_raw.csv"))
    names = read_names(os.path.join(ddir, "ncu_names.csv"))
    # Attach mangled names in order (both ncu passes profile the same launches).
    flat = [(cid, i) for cid, ks in ncu.items() for i in range(len(ks))]
    if len(names) == len(flat):
        for (cid, i), n in zip(flat, names):
            ncu[cid][i]["kernel"] = n
    os.chdir(ROOT)  # calibration paths in the system YAMLs are repo-relative
    results = {"gpu": a.gpu, "device": timing["device"], "cases": {}}
    for cid, rec in timing["cases"].items():
        if "error" in rec or (a.only and not re.search(a.only, cid)) or (a.skip and re.search(a.skip, cid)):
            continue
        sim = simulate_case(cfg["system"], ops_for(cid, rec), GRAPH_GAP_NS[a.gpu])
        meas = rec["us_median"]
        nk = ncu.get(cid, [])
        ncu_us = sum(k["ns"] for k in nk) / 1e3 if nk else None
        res = dict(rec, measured_us=meas, ncu_us=ncu_us, ncu_kernels=nk, sim_us=sim["us"], sim=sim,
                   err_vs_measured=sim["us"] / meas - 1.0,
                   err_vs_ncu=(sim["us"] / ncu_us - 1.0) if ncu_us else None)
        results["cases"][cid] = res
        print(f"{cid:34s} meas {meas:9.2f}  ncu {ncu_us if ncu_us else float('nan'):9.2f}  sim {sim['us']:9.2f}  "
              f"err {100 * res['err_vs_measured']:+6.1f}%", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"validation_{a.gpu}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
