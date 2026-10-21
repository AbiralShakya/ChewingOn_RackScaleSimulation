"""One-hardware-number what-ifs on B300 / GB300 NVL72 (simulator only).

Each variant changes one number of the calibrated B300 and re-runs:
  mufu_2x       every MUFU function (EX2, RCP, ...) issues twice as fast
  smem_2x       shared memory serves 256 B/clk per SM instead of 128
  no_power_cap  no 1100 W board limit (clocks stay at f_max)
on single kernels (synthetic FA-style attention, BF16 and FP4 GEMMs) and on
two rack workloads (Llama-3-405B FP4 prefill, DeepSeek-V3 FP4 decode).

    python examples/gb300_whatifs.py
"""

import copy
import json

import rackscale as rs
from rackscale import lower, models

B300 = rs.presets()["gpus"]["b300"]


def mufu_2x():
    t = {}
    for op, v in B300["opcode_timing"].items():
        if op.startswith("MUFU"):
            ex = v["issue_interval_exact"] / 2
            t[op] = dict(v, issue_interval=max(1, round(ex)), issue_interval_exact=ex)
    return {"opcode_timing": t}


VARIANTS = {
    "baseline": {},
    "mufu_2x": mufu_2x(),
    "smem_2x": {"pipes": {"l1_bytes_per_cycle": 256}},
    "no_power_cap": {"power": None},
}
VARIANTS["all_three"] = {**VARIANTS["mufu_2x"], **VARIANTS["smem_2x"], **VARIANTS["no_power_cap"]}

NVL72_TOPOLOGY = {"kind": "switched", "switches": 18, "links_per_gpu_per_switch": [1] * 18}


def system(variant: str, n: int) -> str:
    g = {"preset": "b300", "count": n}
    if VARIANTS[variant]:
        g["overrides"] = VARIANTS[variant]
    if n == 72:
        sysd = {"preset": "gb200_nvl72", "name": f"GB300 NVL72 [{variant}]", "gpus": [g], "topology": NVL72_TOPOLOGY}
    else:
        sysd = {"preset": "hgx_b200_8", "name": f"B300 [{variant}]", "gpus": [g]}
    return json.dumps(sysd)  # JSON is YAML


def one_kernel(name: str, **k) -> dict:
    return {"name": name, "spmd": {"world": 1, "program": {"streams": [[{"op": "kernel", "name": name, "kernel": k}]]}}}


def with_dtype(wl: dict, dtype: str) -> dict:
    wl = copy.deepcopy(wl)
    for stream in wl["spmd"]["program"]["streams"]:
        for op in stream:
            k = op.get("kernel", {})
            if k.get("type") in ("gemm", "grouped_gemm"):
                k["dtype"] = dtype
    return wl


KERNELS = {
    "attention 8k causal (Llama-70B heads)": one_kernel(
        "attention", type="attention", batch=1, heads_q=64, heads_kv=8, sq=8192, skv=8192, d=128, causal=True),
    "GEMM bf16 8192^3": one_kernel("gemm_bf16", type="gemm", m=8192, n=8192, k=8192, dtype="bf16"),
    "GEMM fp4 16384^3": one_kernel("gemm_fp4", type="gemm", m=16384, n=16384, k=16384, dtype="fp4"),
}
RACK = {
    "Llama-3-405B fp4 prefill 8k (TP8x9, 2 layers)": with_dtype(
        lower.prefill(models.LLAMA3_405B, rs.Parallel(tp=8, dp=9), batch=1, seqlen=8192, layers=2), "fp4"),
    "DeepSeek-V3 fp4 decode b256 (EP72, 4 layers)": with_dtype(
        lower.decode_step(models.DEEPSEEK_V3, rs.Parallel(ep=72), 256, 4096, layers=4), "fp4"),
}

rows = []
for title, wl, n in [(t, w, 1) for t, w in KERNELS.items()] + [(t, w, 72) for t, w in RACK.items()]:
    base = None
    for v in VARIANTS:
        rep = rs.simulate(system(v, 8 if n == 1 else 72), wl, fast=True)
        us = rep.makespan_us
        base = base or us
        rows.append({"workload": title, "variant": v, "us": us, "speedup": base / us})
        print(f"{title:<48} {v:<13} {us:10.1f} us  x{base / us:.3f}", flush=True)
json.dump(rows, open("out/gb300_whatifs.json", "w"), indent=1)
