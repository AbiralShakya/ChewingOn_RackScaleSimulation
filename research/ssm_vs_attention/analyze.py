#!/usr/bin/env python3
"""Turn measurements, validation and sweeps into results/summary.json and the
HTML write-up (results/report.html, inline SVG figures).

    python research/ssm_vs_attention/analyze.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from svgchart import legend, line_chart, scatter, stacked_bars  # noqa: E402

DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
GPUS = {"b300": "B300 SXM6 (sm_103)", "a100pcie": "A100 80GB PCIe (sm_80)"}
KFMT = lambda v: f"{v / 1024:g}k"  # noqa: E731  token counts are powers of two
S = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)"]


def load_timing(g):
    t = json.load(open(os.path.join(DATA, g, "timing.json")))
    p = os.path.join(DATA, g, "timing_decode.json")
    if os.path.exists(p):
        t["cases"].update(json.load(open(p))["cases"])
    return t


def ncu_rows(g):
    p = os.path.join(DATA, g, "ncu_raw.csv")
    text = open(p).read()
    text = text[text.find('"ID"'):]
    rows = list(csv.reader(io.StringIO(text)))
    h = rows[0]
    out = []
    for r in rows[2:]:
        d = dict(zip(h, r))
        m = re.match(r"case_(\w+)/(.*)", d["Kernel Name"])
        if not m:
            continue

        def num(k):
            try:
                return float(d.get(k, "nan").replace(",", ""))
            except ValueError:
                return float("nan")
        out.append(dict(case=m.group(1), kernel=m.group(2), ns=num("gpu__time_duration.sum"),
                        cycles=num("sm__cycles_elapsed.max"), grid=num("launch__grid_size"),
                        block=num("launch__block_size"), insts=num("smsp__inst_executed.sum"),
                        dram=num("dram__bytes_read.sum") + num("dram__bytes_write.sum"),
                        stalls={k.split("stalled_")[1].split("_per")[0]: num(k) for k in d if "stalled_" in k}))
    return out


def kclass(cid):
    for pre, c in (("gemm", "GEMM"), ("attn", "attention"), ("ssm_state", "Mamba-2 decode"), ("conv1d", "conv1d"),
                   ("ssd", "SSD prefill")):
        if cid.startswith(pre):
            return c
    return "other"


def crossover(xs, a, b):
    """First x where a(x) >= b(x) by log-linear interpolation; None if never,
    the first x if a already exceeds b there."""
    if a and a[0] >= b[0]:
        return float(xs[0])
    for i in range(1, len(xs)):
        d0, d1 = a[i - 1] - b[i - 1], a[i] - b[i]
        if d0 < 0 <= d1:
            t = -d0 / (d1 - d0)
            return math.exp(math.log(xs[i - 1]) + t * (math.log(xs[i]) - math.log(xs[i - 1])))
    return None


def measured_summary():
    out = {}
    for g in GPUS:
        t = load_timing(g)["cases"]
        dec = {}
        for b in [1, 8, 32, 64, 128, 256]:
            mam = t[f"ssm_state_update_b{b}"]["us_median"] + t[f"conv1d_update_b{b}"]["us_median"]
            ctxs = [c for c in [1024, 4096, 16384, 65536] if f"attn_decode_b{b}_ctx{c}" in t
                    and "us_median" in t[f"attn_decode_b{b}_ctx{c}"]]
            att = [t[f"attn_decode_b{b}_ctx{c}"]["us_median"] for c in ctxs]
            st = t[f"ssm_state_update_b{b}"]
            dec[b] = dict(mamba_us=mam, state_update_us=st["us_median"], conv_us=t[f"conv1d_update_b{b}"]["us_median"],
                          state_tbps=st["min_bytes"] / st["us_median"] / 1e6,
                          ctx=ctxs, attn_us=att,
                          kv_tbps=[t[f"attn_decode_b{b}_ctx{c}"]["kv_bytes"] / t[f"attn_decode_b{b}_ctx{c}"]["us_median"] / 1e6
                                   for c in ctxs],
                          crossover_ctx=crossover(ctxs, att, [mam] * len(ctxs)))
        pre = {}
        Ls = [1024, 2048, 4096, 8192, 16384, 32768]
        ssd = [t[f"ssd_prefill_L{L}"]["us_median"] + t[f"conv1d_fwd_L{L}"]["us_median"] for L in Ls]
        att = [t[f"attn_prefill_L{L}"]["us_median"] for L in Ls]
        pre = dict(L=Ls, ssd_us=ssd, attn_us=att, crossover_L=crossover(Ls, att, ssd))
        out[g] = dict(decode=dec, prefill=pre)
    return out


def ssd_stages():
    out = {}
    for g in GPUS:
        rows = ncu_rows(g)
        st = {}
        for L in [4096, 16384]:
            ks = [r for r in rows if r["case"] == f"ssd_prefill_L{L}"]
            st[L] = {re.sub(r"^_|_fwd_kernel$", "", r["kernel"]): dict(us=r["ns"] / 1e3, grid=r["grid"], block=r["block"],
                                                                      inst_per_warp=r["insts"] / max(1, r["grid"] * r["block"] / 32),
                                                                      dram_mb=r["dram"] / 1e6,
                                                                      top_stall=max(r["stalls"].items(), key=lambda kv: kv[1] if kv[1] == kv[1] else -1)[0])
                     for r in ks}
        su = [r for r in rows if r["case"] == "ssm_state_update_b64"]
        st["state_update_b64"] = dict(us=su[0]["ns"] / 1e3, inst_per_warp=su[0]["insts"] / (su[0]["grid"] * su[0]["block"] / 32),
                                      stalls=su[0]["stalls"]) if su else None
        out[g] = st
    return out


def validation():
    out = {}
    for g, key in (("b300", "b300"), ("a100pcie", "a100")):
        p = os.path.join(RES, f"validation_{key}.json")
        if not os.path.exists(p):
            continue
        v = json.load(open(p))["cases"]
        by = {}
        for cid, r in v.items():
            by.setdefault(kclass(cid), []).append(abs(r["err_vs_measured"]))
        en = [abs(r["sim_us"] / r["ncu_us"] - 1) for r in v.values() if r.get("ncu_us")]
        out[g] = dict(n=len(v), median_abs_err={k: statistics.median(e) for k, e in by.items()},
                      overall_vs_ncu=statistics.median(en) if en else None,
                      count={k: len(e) for k, e in by.items()},
                      overall=statistics.median([abs(r["err_vs_measured"]) for r in v.values()]),
                      points=[(r["measured_us"], r["sim_us"], cid) for cid, r in v.items()])
    return out


def sweeps():
    out = {}
    for fam in ("70b", "8b", "whatif_ssu8"):
        p = os.path.join(RES, f"sweep_{fam}.json")
        if os.path.exists(p):
            out[fam] = json.load(open(p))["points"]
    return out


MODEL_NAMES = {"mamba2-70b-ssu8": "Mamba-2 70B, 8-row state update (what-if)", "llama3-70b": "Transformer (Llama-3-70B)", "mamba2-70b": "Mamba-2 70B", "mamba2-hybrid-70b": "Hybrid 70B",
               "llama3-8b": "Transformer (Llama-3-8B)", "mamba2-8b": "Mamba-2 8B", "mamba2-hybrid-8b": "Hybrid 8B"}


def rack_frontier(points, tp):
    """Per model and context: best tokens/s/GPU over the batches that fit in HBM."""
    fr = {}
    for p in points:
        if p["phase"] != "decode" or p["tp"] != tp or not p.get("fits_hbm") or "step_us" not in p or p.get("full_rack"):
            continue
        key = (p["model"], p["context"])
        if key not in fr or p["tokens_per_s_per_gpu"] > fr[key]["tokens_per_s_per_gpu"]:
            fr[key] = p
    return fr


def main():
    meas = measured_summary()
    stages = ssd_stages()
    val = validation()
    sw = sweeps()
    summary = dict(measured=meas, ssd_stages=stages,
                   validation={g: {k: v for k, v in d.items() if k != "points"} for g, d in val.items()})
    if "whatif_ssu8" in sw:
        summary["whatif_ssu8"] = {f"tp{p['tp']}_b{p['batch']}": p["tokens_per_s_per_gpu"] for p in sw["whatif_ssu8"]
                                  if p["context"] == 1024 and "step_us" in p}
    if "70b" in sw:
        for tp in (4, 8):
            fr = rack_frontier(sw["70b"], tp)
            summary[f"rack_frontier_tp{tp}"] = {f"{m}@{c}": dict(batch=p["batch"], tokens_per_s_per_gpu=p["tokens_per_s_per_gpu"],
                                                                step_us=p["step_us"]) for (m, c), p in sorted(fr.items())}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    figs = make_figures(meas, stages, val, sw)
    with open(os.path.join(RES, "figures.json"), "w") as f:
        json.dump(figs, f)
    print(json.dumps({k: v for k, v in summary.items() if k != "ssd_stages"}, indent=1, default=str)[:4000])


def make_figures(meas, stages, val, sw):
    figs = {}
    # Decode: per-layer mixer time vs context at batch 64.
    for g in GPUS:
        d = meas[g]["decode"][64]
        att = [(c, u, f"{u:.0f} us at {c // 1024}k tokens, KV read at {bw:.1f} TB/s") for c, u, bw in zip(d["ctx"], d["attn_us"], d["kv_tbps"])]
        mam = [(c, d["mamba_us"], f"{d['mamba_us']:.0f} us (state update {d['state_update_us']:.0f} us at {d['state_tbps']:.2f} TB/s)") for c in d["ctx"]]
        mk = [dict(x=d["crossover_ctx"], label=f"crossover {d['crossover_ctx'] / 1024:.1f}k")] if d["crossover_ctx"] else []
        figs[f"decode_{g}"] = line_chart(
            [dict(name="Attention (FA4)" if g == "b300" else "Attention (FA2)", color=S[0], points=att),
             dict(name="Mamba-2 mixer", color=S[1], points=mam)],
            xlabel="context length (tokens)", ylabel="µs per layer, batch 64", xlog=True, ylog=True,
            xticks=d["ctx"], xfmt=KFMT, markers=mk, aria=f"Decode mixer time vs context on {GPUS[g]}")
        pr = meas[g]["prefill"]
        mk = [dict(x=pr["crossover_L"], label=f"crossover {pr['crossover_L'] / 1024:.1f}k")] if pr["crossover_L"] else []
        figs[f"prefill_{g}"] = line_chart(
            [dict(name="Attention (FA4)" if g == "b300" else "Attention (FA2)", color=S[0],
                  points=[(L, u, f"{u:.0f} us at L={L}") for L, u in zip(pr["L"], pr["attn_us"])]),
             dict(name="Mamba-2 SSD + conv", color=S[1],
                  points=[(L, u, f"{u:.0f} us at L={L}") for L, u in zip(pr["L"], pr["ssd_us"])])],
            xlabel="prompt length (tokens)", ylabel="µs per layer", xlog=True, ylog=True, xticks=pr["L"], xfmt=KFMT, markers=mk,
            aria=f"Prefill mixer time vs length on {GPUS[g]}")
    # SSD stage breakdown at 16k.
    order = ["chunk_cumsum", "chunk_state", "state_passing", "bmm_chunk", "chunk_scan"]
    bars = []
    for g in GPUS:
        for L in (4096, 16384):
            bars.append((f"{'B300' if g == 'b300' else 'A100'}|L={L // 1024}k", {k: stages[g][L][k]["us"] for k in order if k in stages[g][L]}))
    figs["ssd_stages"] = stacked_bars(bars, order, S[:5], ylabel="µs per layer (Nsight Compute)", aria="SSD kernel breakdown")
    figs["ssd_legend"] = legend(list(zip(order, S[:5])))
    # Validation scatter.
    for g, d in val.items():
        cls = {"GEMM": S[0], "attention": S[1], "Mamba-2 decode": S[2], "SSD prefill": S[3], "conv1d": S[4]}
        pts = [(m, s, f"{cid}: measured {m:.1f} us, simulated {s:.1f} us ({100 * (s / m - 1):+.0f}%)", cls.get(kclass(cid), S[5]))
               for m, s, cid in d["points"]]
        figs[f"val_{g}"] = scatter(pts, xlabel="measured µs", ylabel="simulated µs",
                                   aria=f"Simulated vs measured kernel time on {GPUS[g]}")
    figs["val_legend"] = legend([("GEMM", S[0]), ("attention", S[1]), ("Mamba-2 decode", S[2]), ("SSD prefill", S[3]), ("conv1d", S[4])])
    # Rack sweeps.
    if "70b" in sw:
        pts = sw["70b"]
        models = ["llama3-70b", "mamba2-70b", "mamba2-hybrid-70b"]
        for tp in (4, 8):
            fr = rack_frontier(pts, tp)
            series = []
            for i, m in enumerate(models):
                p = sorted([(c, q) for (mm, c), q in fr.items() if mm == m])
                if p:
                    series.append(dict(name=MODEL_NAMES[m], color=S[i], points=[
                        (c, q["tokens_per_s_per_gpu"], f"{q['tokens_per_s_per_gpu']:.0f} tok/s/GPU at {c // 1024}k, batch {q['batch']}")
                        for c, q in p]))
            wi = [p for p in sw.get("whatif_ssu8", []) if p.get("tp") == tp and p.get("batch") == 512 and "step_us" in p]
            if wi and series:
                ctxs_all = sorted({c for (_, c) in fr})
                v = wi[0]["tokens_per_s_per_gpu"]
                series.append(dict(name="Mamba-2, faster state update", color=S[3], dash=True,
                                   points=[(c, v, f"{v:.0f} tok/s/GPU (what-if, batch 512)") for c in ctxs_all]))
            if series:
                figs[f"rack_tput_tp{tp}"] = line_chart(series, xlabel="context length (tokens)", ylabel="decode tokens/s per GPU",
                                                      xlog=True, ylog=True, xticks=sorted({c for (_, c) in fr}), xfmt=KFMT,
                                                      aria=f"NVL72 decode throughput, TP{tp}")
            pre = [p for p in pts if p["phase"] == "prefill" and p["tp"] == tp and "step_us" in p]
            series = []
            for i, m in enumerate(models):
                q = sorted([(p["seqlen"], p["step_us"] / 1e3) for p in pre if p["model"] == m])
                if q:
                    series.append(dict(name=MODEL_NAMES[m], color=S[i], points=[(L, t, f"{t:.1f} ms TTFT at L={L}") for L, t in q]))
            if series:
                figs[f"rack_prefill_tp{tp}"] = line_chart(series, xlabel="prompt length (tokens)", ylabel="time to first token (ms)",
                                                         xlog=True, ylog=True, xticks=sorted({p["seqlen"] for p in pre}), xfmt=KFMT,
                                                         aria=f"NVL72 prefill TTFT, TP{tp}")
        # Step breakdown at one representative point.
        keys = ["gemm", "attention", "ssm_state_update", "conv1d", "other", "collectives"]
        bars = []
        for c in (4096, 32768):
            for m in models:
                p = next((q for q in pts if q["phase"] == "decode" and q["model"] == m and q["tp"] == 8 and q["batch"] == 64
                          and q["context"] == c and "step_us" in q and not q.get("full_rack")), None)
                if p:
                    v = {k: p["kernel_us_per_gpu"].get(k, 0.0) / 1e3 for k in keys if k != "collectives"}
                    v["collectives"] = p["collective_us"] / 1e3
                    bars.append((f"{m.split('-')[0].replace('llama3', 'Transformer').replace('mamba2', 'Mamba-2' if 'hybrid' not in m else 'Hybrid')}|{c // 1024}k", v))
        if bars:
            figs["rack_breakdown"] = stacked_bars(bars, keys, S[:6], ylabel="ms per decode step (per GPU)",
                                                  yfmt=lambda v: f"{v:.1f}", aria="Decode step breakdown")
            figs["rack_breakdown_legend"] = legend(list(zip(["GEMMs", "attention", "Mamba-2 state update", "conv1d", "norms, residuals, launch-bound", "TP all-reduce"], S[:6])))
    return figs


if __name__ == "__main__":
    main()
