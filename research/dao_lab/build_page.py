#!/usr/bin/env python3
"""Build results/report.html from the study results (self-contained: data inline, no libraries).

    python research/dao_lab/build_page.py
"""
import json
import os
import subprocess

from common import HERE, RESULTS, load

STUDIES = {
    "extrap": "extrap_check.json",
    "prefill": "prefill_mixers.json",
    "dec_k": "decode_attention_kernels.json",
    "dec_r": "decode_attention_rack.json",
    "cp": "cp_prefill.json",
}


def slim_prefill(r):
    return {"mixer": r["mixer"], "s": r["seqlen"], "ms": round(r["block_us"] / 1e3, 4),
            "fast": sorted({v for v in r["rules"].values() if v != "exact"}),
            "parts": {k: round(v, 1) for k, v in sorted(r["parts_us"].items(), key=lambda kv: -kv[1])[:4]}}


def slim_dec_k(r):
    return {k: r[k] for k in ("label", "tp", "batch", "context")} | {
        "wg": r.get("kv_warpgroups", 1), "us": round(r["attn_us"], 2), "gbps": round(r["kv_gbps"]),
        "ai": round(r["flops_per_byte"], 1), "kv_mb": round(r["kv_bytes"] / 1e6, 1), "d": r["spec"]["d"],
        "hq": r["spec"]["heads_q"], "hkv": r["spec"]["heads_kv"]}


def slim_dec_r(r):
    o = {"label": r.get("label"), "tp": r["plan"]["tp"], "ctx": r["context"], "batch": r["batch"],
         "max_batch": r.get("max_batch"), "wg": r.get("kv_warpgroups", 1)}
    for k in ("step_us", "tok_s_gpu", "tok_s_user", "memory_total_gb", "dominant", "error"):
        if k in r:
            o[k] = round(r[k], 2) if isinstance(r[k], float) else r[k]
    if "classes_us" in r:
        o["cls"] = {k: round(v, 1) for k, v in r["classes_us"].items()}
    return o


def slim_cp(r):
    return {"model": r["model"], "cp": r["cp"], "s": r["seqlen"], "ttft_ms": round(r["ttft_ms"], 2),
            "mfu": round(r["mfu"], 4), "tpr": r["tokens_per_rank"], "top": r["kernels"][:5]}


def main():
    data = {}
    for key, fn in STUDIES.items():
        rows = load(fn)
        f = {"prefill": slim_prefill, "dec_k": slim_dec_k, "dec_r": slim_dec_r, "cp": slim_cp}.get(key)
        data[key] = [f(r) for r in rows] if f else [
            {"type": r["spec"]["type"], "spec": r["spec"], "err": round(r["error"], 5),
             "est": round(r["extrapolated_us"], 1), "sim": round(r["simulated_us"], 1)} for r in rows]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    data["commit"] = commit
    tpl = open(os.path.join(HERE, "page_template.html")).read()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    path = os.path.join(RESULTS, "report.html")
    open(path, "w").write(out)
    print(path, len(out) // 1024, "KB")


if __name__ == "__main__":
    main()
