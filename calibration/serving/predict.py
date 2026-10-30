"""Simulate every published serving case in published.json and score it.

    python calibration/serving/predict.py            # all cases -> predictions.json, RESULTS.md
    python calibration/serving/predict.py --only sglang-h100-decode

Each case is reproduced as the simulator's own decode or prefill step at the
published configuration (model, GPUs, parallelism, batch, KV length, GEMM
precision). Nothing is fitted: the simulator runs at a clean commit, recorded
with every prediction, and the signed error (predicted - measured) / measured
is reported per case. These are throughputs, so a positive error means the
simulator is optimistic (it predicts more tokens/s than were published).

Two kinds of case:

* PD-disaggregated decode or prefill (SGLang, DeepSeek): one simulated step,
  tokens/s per GPU = tokens in the step / step time / GPUs.
* In-flight batching (TensorRT-LLM ``trtllm-bench``): prefill and decode
  share iterations. With ISL/OSL tokens per request, every decode step of a
  ``batch``-sequence rank also prefills batch/OSL prompts on average, so the
  per-token time is t_decode + (batch / OSL) * t_prefill(one prompt). Running
  the two as separate kernels is pessimistic (a fused mixed batch shares the
  weight reads); the decode-only number is reported beside it as the bound
  on the other side.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))

from rackscale import sweep  # noqa: E402
from rackscale.sweep import Point, SystemDef, SYSTEMS  # noqa: E402

# Systems the published cases ran on (the 8-GPU presets are already in sweep.SYSTEMS).
EXTRA = [
    SystemDef("h100_4n", "configs/serving/h100_4node_ib.yaml", "4x HGX H100, 400G IB", "h100_sxm", 32, 600.0,
              1500.0, "datasheet preset (no H100 measurements)"),
    SystemDef("h100_9n", "configs/serving/h100_9node_ib.yaml", "9x HGX H100, 400G IB", "h100_sxm", 72, 600.0,
              1500.0, "datasheet preset (no H100 measurements)"),
    SystemDef("h800_18n", "configs/serving/h800_18node_ib.yaml", "18x HGX H800, 400G IB", "h800", 144, 600.0,
              1500.0, "datasheet preset (H100 with 8 NVLink links)"),
    SystemDef("h200", "configs/serving/hgx_h200_8.yaml", "HGX H200 x8", "h200_sxm", 8, 600.0, 1500.0,
              "datasheet preset (no H200 measurements)"),
    SystemDef("gb200", "gb200_nvl72", "GB200 NVL72", "gb200", 72, 554.0, 2600.0,
              "B200 die preset; NVL72 fabric from spec"),
]
for s in EXTRA:
    SYSTEMS.setdefault(s.key, s)

DS = "deepseek-v3"  # R1 has V3's architecture
P = sweep.plan

# How each case is simulated. "steps" are Points; "combine" turns their results into tok/s per GPU.
SIM = {
    "sglang-h100-decode": dict(
        steps=dict(decode=Point(DS, "h100_9n", "decode", P(ep=72), batch=256, context=2000, dtype="fp8",
                                overlap=True)),
        alt=dict(no_overlap=Point(DS, "h100_9n", "decode", P(ep=72), batch=256, context=2000, dtype="fp8"))),
    "sglang-h100-prefill": dict(
        steps=dict(prefill=Point(DS, "h100_4n", "prefill", P(ep=32), batch=4, seqlen=4096, dtype="fp8"))),
    "sglang-gb200-decode": dict(
        steps=dict(decode=Point(DS, "gb200", "decode", P(ep=48), batch=1408, context=2048, dtype="fp8",
                                overlap=True)),
        alt=dict(no_overlap=Point(DS, "gb200", "decode", P(ep=48), batch=1408, context=2048, dtype="fp8"))),
    "trtllm-b200-r1-fp4": dict(
        steps=dict(decode=Point(DS, "b200", "decode", P(ep=8), batch=384, context=2048, dtype="fp4"),
                   prefill=Point(DS, "b200", "prefill", P(ep=8), batch=1, seqlen=1024, dtype="fp4")),
        isl=1024, osl=2048),
    "trtllm-h200-r1": dict(
        steps=dict(decode=Point(DS, "h200", "decode", P(ep=8), batch=128, context=2048, dtype="fp8"),
                   prefill=Point(DS, "h200", "prefill", P(ep=8), batch=1, seqlen=1024, dtype="fp8")),
        isl=1024, osl=2048),
    "deepseek-production-decode": dict(
        steps=dict(decode=Point(DS, "h800_18n", "decode", P(ep=144), batch=88, context=4989, dtype="fp8",
                                overlap=True))),
}


def git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "python", "crates", "configs"], cwd=ROOT)
    return sha + ("-dirty" if dirty else "")


def summarize(r: dict) -> dict:
    keep = ("error", "step_us", "tok_s_gpu", "tpot_ms", "ttft_ms", "world", "fits", "memory_total_gb", "hbm_gb",
            "classes_us", "breakdown_us", "dominant", "mfu", "sim_wall_s")
    return {k: r.get(k) for k in keep if k in r}


def predict(case: dict) -> dict:
    sim = SIM[case["id"]]
    runs = {k: sweep.run_point(pt) for k, pt in {**sim["steps"], **sim.get("alt", {})}.items()}
    out = dict(points={k: dict(point=pt.__dict__, result=summarize(runs[k]))
                       for k, pt in {**sim["steps"], **sim.get("alt", {})}.items()})
    bad = [k for k, r in runs.items() if k in sim["steps"] and (r.get("error") or not r.get("step_us"))]
    if bad:
        out["error"] = "; ".join(f"{k}: {runs[k].get('error') or 'does not fit in HBM'}" for k in bad)
        return out
    if "decode" in sim["steps"] and "prefill" in sim["steps"]:
        b = sim["steps"]["decode"].batch
        t_dec = runs["decode"]["step_us"]
        t_pf = runs["prefill"]["step_us"]
        per_step = t_dec + b / sim["osl"] * t_pf
        out["predicted"] = b / (per_step * 1e-6)
        out["predicted_decode_only"] = b / (t_dec * 1e-6)
        out["model"] = f"batch / (t_decode + batch/OSL * t_prefill) = {b} / ({t_dec:.0f} + {b}/{sim['osl']} x " \
                       f"{t_pf:.0f}) us"
    else:
        (k, r), = [(k, runs[k]) for k in sim["steps"]]
        out["predicted"] = r["tok_s_gpu"]
        out["model"] = f"one {k} step: {r['step_us']:.0f} us"
    for k in sim.get("alt", {}):
        if runs[k].get("tok_s_gpu"):
            out[f"predicted_{k}"] = runs[k]["tok_s_gpu"]
    return out


def table(doc: dict) -> str:
    lines = [f"# Published serving benchmarks vs the simulator\n",
             f"Simulator commit `{doc['git_sha']}`, run {doc['created_utc']}. Error = (predicted − measured) / "
             f"measured on output (or input) tokens/s per GPU: positive means the simulator predicts more "
             f"throughput than was published. Sources and configs: [published.json](published.json).\n",
             "| case | engine, hardware | measured tok/s/GPU | predicted | error | how |",
             "|---|---|---:|---:|---:|---|"]
    errs = []
    for c in doc["cases"]:
        p = c["prediction"]
        if p.get("error"):
            lines.append(f"| {c['id']} | {c['engine']}, {c['hardware']} | {c['measured']:,.0f} | — | — | "
                         f"not simulated: {p['error'][:120]} |")
            continue
        e = (p["predicted"] - c["measured"]) / c["measured"]
        c["error"] = e
        if c["role"] == "primary":
            errs.append(e)
        tag = "" if c["role"] == "primary" else " (soft)"
        lines.append(f"| [{c['id']}]({c['source']}){tag} | {c['engine']}, {c['hardware']} | {c['measured']:,.0f} | "
                     f"{p['predicted']:,.0f} | {e:+.1%} | {p['model']} |")
    if errs:
        s = sorted(abs(e) for e in errs)
        med = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
        lines += ["", f"Primary cases: {len(errs)}, median |error| {med:.1%}, signed errors "
                  + ", ".join(f"{e:+.1%}" for e in errs) + "."]
    lines += ["", "## Second references", ""]
    for c in doc["cases"]:
        p = c["prediction"]
        for k in ("predicted_decode_only", "predicted_no_overlap"):
            if p.get(k):
                lines.append(f"- {c['id']} {k.replace('predicted_', '').replace('_', ' ')}: {p[k]:,.0f} tok/s/GPU "
                             f"({(p[k] - c['measured']) / c['measured']:+.1%})")
        if p.get("predicted_decode_only"):
            # Added after the first comparison (so not a frozen prediction): with attention DP,
            # every rank's step waits for the slowest, so a step costs a prefill whenever any of
            # the W ranks has one, probability 1 - (1 - batch/OSL)^W, not batch/OSL.
            pts = p["points"]
            b, w = pts["decode"]["point"]["batch"], pts["decode"]["result"]["world"]
            q = b / SIM[c["id"]]["osl"]
            t = pts["decode"]["result"]["step_us"] + (1 - (1 - q) ** w) * pts["prefill"]["result"]["step_us"]
            v = b / (t * 1e-6)
            lines.append(f"- {c['id']} with attention-DP synchronisation (post hoc, same simulated steps): "
                         f"{v:,.0f} tok/s/GPU ({(v - c['measured']) / c['measured']:+.1%})")
        if c.get("measured_alt") and p.get("predicted"):
            lines.append(f"- {c['id']} against the alternative published figure {c['measured_alt']:,.0f}: "
                         f"{(p['predicted'] - c['measured_alt']) / c['measured_alt']:+.1%}")
    lines += ["", FINDINGS]
    return "\n".join(lines) + "\n"


FINDINGS = """## What the misses say (read from the simulated breakdowns)

- **SGLang H100 decode, -2.7%.** The one case run exactly as published (PD-disaggregated decode,
  two-batch overlap) lands within 3%. Without two-batch overlap the simulator gives -13.6%, so
  the overlap is doing real work in the published number.
- **SGLang H100 prefill, -45.8%.** Communication is 2.57 s of the simulated 4.07 s step. The
  lowering sends every token to each of its top-8 experts as a plain all-to-all
  (`python/rackscale/lower.py`, `moe_dispatch`). DeepSeek-V3 limits each token to 4 nodes and
  DeepEP sends it once per node over InfiniBand, then fans it out over NVLink; SGLang also overlaps
  the two micro-batches in prefill. Neither is modelled, so the simulator is pessimistic here.
  This is a lowering gap, not a fabric one.
- **TensorRT-LLM, +95% (B200) and +67% (H200).** The decode step alone is close to what the
  hardware can do; the miss is in how prefill shares the iterations. With attention DP, all 8
  ranks wait for any rank that is prefilling, so prefill costs far more than its token share.
  Accounting for that with the same simulated steps gives +0.9% on H200 and +25% on B200. This
  composition was added after seeing the first result, so it is reported as post hoc. The frozen
  errors above stay the headline.
- **SGLang GB200 decode, not simulated.** 1,408 sequences of 2,048 tokens need 203 GB of BF16 MLA
  KV cache per GPU, more than a GB200 holds, so the published run must have used a shorter
  average KV length or an FP8 cache. The post does not say which, so the case is left unscored.
- **DeepSeek production, -12.4% (soft).** A 24-hour average with load swings and an inferred batch;
  shown for scale only.
"""


def rebuild() -> None:
    """Rewrite RESULTS.md from predictions.json without simulating."""
    doc = json.load(open(os.path.join(HERE, "predictions.json")))
    open(os.path.join(HERE, "RESULTS.md"), "w").write(table(doc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--table", action="store_true", help="only rebuild RESULTS.md")
    a = ap.parse_args()
    if a.table:
        return rebuild()
    pub = json.load(open(os.path.join(HERE, "published.json")))
    path = os.path.join(HERE, "predictions.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    done = {c["id"]: c for c in old.get("cases", [])}
    sha = git_sha()
    if a.only and old and old.get("git_sha") != sha:
        sys.exit(f"predictions.json is from {old.get('git_sha')}, HEAD is {sha}: rerun every case")
    cases = []
    for c in pub["cases"]:
        if a.only and c["id"] not in a.only and c["id"] in done:
            cases.append(done[c["id"]])
            continue
        print(f"  {c['id']} ...", flush=True)
        cases.append({**c, "prediction": predict(c)})
    doc = dict(git_sha=sha, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               cases=cases)
    md = table(doc)
    json.dump(doc, open(path, "w"), indent=1)
    open(os.path.join(HERE, "RESULTS.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
