#!/usr/bin/env python3
"""Build results/report.html: the findings and an explorer over every simulated point.

    python research/model_sweep/build_page.py

Reads results/<study>.json written by run.py. The page is self-contained (data
inline, no libraries) and is what gets published as the study's artifact.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from findings import findings  # noqa: E402

KEEP = ("study", "model", "system", "phase", "plan", "batch", "context", "seqlen", "chunk", "micro_batches", "launch",
        "overlap", "dtype", "tag", "world", "fits", "memory_total_gb", "hbm_gb", "step_us", "tok_s_gpu", "tok_s_user",
        "ttft_ms", "mfu", "dominant", "error")


def slim(r: dict) -> dict:
    o = {k: r[k] for k in KEEP if k in r and r[k] not in (None, "", False) or k in ("fits",) and k in r}
    o["plan"] = "x".join(f"{k}{v}" for k, v in r["plan"].items() if v > 1) or "1gpu"
    if "breakdown_us" in r:
        o["bd"] = [round(r["breakdown_us"][k], 1) for k in ("compute", "hbm", "latency", "comm", "launch")]
        o["cls"] = {k: round(v, 1) for k, v in r["classes_us"].items()}
        o["sched"] = {k: round(v, 3) for k, v in sorted(r["sched"].items(), key=lambda kv: -kv[1])[:5]}
        o["top"] = [[k["name"], round(k["us"], 1), k["bound"]] for k in r["top_kernels"][:3]]
    for k in ("step_us", "tok_s_gpu", "tok_s_user", "ttft_ms", "memory_total_gb"):
        if k in o:
            o[k] = round(o[k], 3 if k == "tok_s_user" else 1)
    if "mfu" in o:
        o["mfu"] = round(o["mfu"], 4)
    return o


def _key(r: dict) -> str:
    import dataclasses
    import rackscale.sweep as sw
    return sw.Point(**{f.name: r[f.name] for f in dataclasses.fields(sw.Point) if f.name in r}).key()


def main() -> None:
    data, systems, archs, pending = {}, {}, {}, {}
    import rackscale.sweep as sw
    from rackscale import models
    for st in sw.STUDIES:
        p = os.path.join(RES, f"{st}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            systems = d["systems"]
        elif os.path.exists(p + "l"):
            # A sweep still running: preview from its checkpoint.
            d = dict(points=list({r["_key"]: r for r in map(json.loads, filter(str.strip, open(p + "l")))}.values()))
        else:
            continue
        done = {_key(r) for r in d["points"]}
        left = [pt for pt in sw.study_points(st) if pt.key() not in done]
        if left:
            pending[st] = sorted({f"{pt.model} {pt.system} {pt.phase}" for pt in left})
        data[st] = [slim(r) for r in d["points"]]
        for r in d["points"]:
            if "arch" in r:
                archs[r["model"]] = r["arch"]
    for name in {r["model"] for pts in data.values() for r in pts} - set(archs):
        archs[name] = sw.arch_of(models.ALL[name])
    sysinfo = {k: dict(label=v["label"], gpus=v["gpus"], provenance=v["provenance"], caveat=v["caveat"],
                       hbm=v["peaks"]["hbm_gb"], bf16=round(v["peaks"]["tflops"]["bf16"]),
                       hbm_tbps=round(v["peaks"]["hbm_gbps"] / 1000, 2)) for k, v in systems.items()}
    blob = dict(points=data, systems=sysinfo, archs=archs, measured=sw.MEASURED_B300, bound=sw.BOUND,
                pending=pending)
    fnd = findings(data, sysinfo, archs)
    tpl = open(os.path.join(HERE, "page_template.html")).read()
    html = (tpl.replace("/*__DATA__*/null", json.dumps(blob, separators=(",", ":")))
               .replace("<!--__FINDINGS__-->", fnd["html"])
               .replace("__NPOINTS__", f"{sum(len(v) for v in data.values()):,}")
               .replace("__NSIM__", f"{sum(1 for v in data.values() for r in v if 'step_us' in r):,}"))
    out = os.path.join(RES, "report.html")
    with open(out, "w") as f:
        f.write(html)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(fnd["numbers"], f, indent=1)
    print("wrote", out, f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
