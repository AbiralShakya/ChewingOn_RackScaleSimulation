"""Score a node day against the committed predictions: one signed error per measurement.

    python calibration/nodeday/compare.py ~/nodeday/results/<host>-<gpu>
    python calibration/nodeday/compare.py <results> --predictions predictions/hgx_b200_8.json

Reads ``predictions/hgx_<gpu>_8.json`` as committed (the GPU comes from the
run's ``env/gpu-key.txt``) and every block's output, and writes
``COMPARE.md`` and ``compare.json`` into the results directory. Errors are
signed ``(predicted − measured) / measured``: negative means the simulator is
fast (optimistic). Training and decode steps are also scored for the
textbook formula and the roofline replay, so the article's "formulas rank
plans, the machine sets the price" is checked on hardware, not only in
simulation. Nothing here re-runs the simulator.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import statistics
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cases import DECODE, NCCL_HEADLINE_SIZES, NCCL_TESTS_BIN, TRAIN  # noqa: E402

E_MAX = {2: 0.5642, 4: 1.0294, 8: 1.4236}  # E[max of N standard normals]


def err(pred: Optional[float], meas: Optional[float]) -> Optional[float]:
    if pred is None or meas is None or meas == 0:
        return None
    return (pred - meas) / meas


def pct(e: Optional[float]) -> str:
    return "–" if e is None else f"{100 * e:+.1f}%"


def us(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x / 1000:.2f} ms" if x >= 10000 else f"{x:.1f} µs"


def load(path: str):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# collectives


def parse_nccl_tests(path: str) -> List[dict]:
    """Out-of-place rows of an nccl-tests log: size (bytes), time (µs), busbw (GB/s)."""
    rows = []
    for line in open(path, errors="replace"):
        t = line.split()
        if len(t) >= 8 and t[0].isdigit() and t[1].isdigit():
            try:
                rows.append(dict(bytes=int(t[0]), time_us=float(t[5]), busbw_gbps=float(t[7])))
            except ValueError:
                pass
    return rows


def nccl(res: str, pred: dict) -> dict:
    by_bin = {v.replace("_perf", ""): k for k, v in NCCL_TESTS_BIN.items()}
    idx = {}
    for p in pred.get("nccl") or []:
        if "bytes" in p:
            idx[(p["coll"], p["ranks"], p["algo_forced"], p["bytes"])] = p
    points, curves = [], {}
    for path in sorted(glob.glob(os.path.join(res, "nccl", "*.txt"))):
        m = re.match(r"(\w+?)_n(\d)(?:_(Ring|Tree|NVLS))?\.txt$", os.path.basename(path))
        if not m:
            continue
        coll = by_bin.get(m.group(1))
        n, algo = int(m.group(2)), (m.group(3) or "auto").lower()
        for r in parse_nccl_tests(path):
            p = idx.get((coll, n, algo, r["bytes"]))
            if p:
                pt = dict(coll=coll, ranks=n, algo=algo, bytes=r["bytes"], meas_us=r["time_us"],
                          pred_us=p["time_us"], meas_busbw=r["busbw_gbps"], pred_busbw=p["busbw_gbps"],
                          pred_algo=f"{p['algo']}/{p['proto']}", err=err(p["time_us"], r["time_us"]))
                points.append(pt)
                curves.setdefault((coll, n, algo), []).append(pt)
    summary = []
    for (coll, n, algo), pts in curves.items():
        big = [p["err"] for p in pts if p["bytes"] >= 1 << 20]
        small = [p for p in pts if p["bytes"] <= 1024]
        summary.append(dict(coll=coll, ranks=n, algo=algo, points=len(pts),
                            median_err_ge_1MiB=statistics.median(big) if big else None,
                            median_abs_err_ge_1MiB=statistics.median(abs(e) for e in big) if big else None,
                            latency_err=statistics.median(p["err"] for p in small) if small else None,
                            peak_busbw_meas=max(p["meas_busbw"] for p in pts),
                            peak_busbw_pred=max(p["pred_busbw"] for p in pts)))
    return dict(points=points, summary=summary)


# ---------------------------------------------------------------------------
# steps


def _pred_step(pred: dict, block: str, name: str, layers) -> Optional[dict]:
    for r in pred.get(block) or []:
        if r["case"] == name and str(r["layers"]) == str(layers) and "sim_us" in r:
            return r
    return None


def train(res: str, pred: dict) -> List[dict]:
    rows = []
    for c in TRAIN:
        for layers in c["layers"]:
            m = load(os.path.join(res, "train", f"{c['script']}_L{layers}.json"))
            p = _pred_step(pred, "train", c["name"], layers)
            if not m or not p:
                continue
            rows.append(dict(case=c["name"], layers=layers, meas_us=m["step_us"], meas_sd_us=m.get("stdev_us"),
                             sim_us=p["sim_us"], replay_us=p["replay_us"], textbook_us=p["textbook_us"],
                             err_sim=err(p["sim_us"], m["step_us"]), err_replay=err(p["replay_us"], m["step_us"]),
                             err_textbook=err(p["textbook_us"], m["step_us"]), zb_split=m.get("zb_split")))
    return rows


def decode(res: str, pred: dict) -> List[dict]:
    from_model = {}
    for r in pred.get("decode") or []:
        from_model.setdefault(r["case"], []).append(r["layers"])
    rows = []
    for c in DECODE:
        for layers in c["layers"]:
            slug = "".join(ch if ch.isalnum() else "_" for ch in c["name"]).strip("_").lower()
            m = load(os.path.join(res, "decode", f"{slug}_L{layers}.json"))
            # Predictions store the full depth as a number.
            want = max(from_model.get(c["name"], [0])) if layers == "full" else layers
            p = _pred_step(pred, "decode", c["name"], want)
            if not m or "step_us" not in m or not p:
                continue
            rows.append(dict(case=c["name"], layers=want, meas_us=m["step_us"], p90_us=1e3 * m.get("p90_itl_ms", 0),
                             sim_us=p["sim_us"], replay_us=p["replay_us"], textbook_us=p["textbook_us"],
                             err_sim=err(p["sim_us"], m["step_us"]), err_replay=err(p["replay_us"], m["step_us"]),
                             err_textbook=err(p["textbook_us"], m["step_us"])))
    return rows


# ---------------------------------------------------------------------------
# kernels: GEMM time and clock under power; GEMM beside NCCL


def gemm_power(res: str, pred: dict, max_mhz: Optional[float]) -> dict:
    out = {}
    for tag in ("gpu0", "all"):
        doc = load(os.path.join(res, "power", f"{tag}.json"))
        if not doc:
            continue
        rows = []
        for gpu, results in doc["results"].items():
            for r in results:
                if "error" in r:
                    continue
                row = dict(gpu=int(gpu), case=r["case"], meas_us=r["time_us"], meas_mhz=r.get("sm_mhz"),
                           power_w=r.get("power_w"), capped=r.get("power_capped_frac"), reasons=r.get("reasons"))
                if r["kind"] == "gemm":
                    p = next((g for g in pred.get("gemm") or [] if g["dtype"] == r["dtype"] and g["n"] == r["n"]
                              and "time_us" in g), None)
                    if p:
                        pm = p.get("sustained_mhz") or max_mhz
                        row.update(pred_us=p["time_us"], pred_mhz=pm, err_time=err(p["time_us"], r["time_us"]),
                                   err_mhz=err(pm, r.get("sm_mhz")))
                else:
                    # The simulator gives attention no power-cap drop (CALIBRATION.md); score that claim.
                    row.update(pred_mhz=max_mhz, err_mhz=err(max_mhz, r.get("sm_mhz")))
                rows.append(row)
        out[tag] = rows
    return out


def overlap(res: str, pred: dict) -> List[dict]:
    pr = {p["reserve_sms"]: p for p in pred.get("overlap") or [] if "time_us" in p}
    meas = {}
    for path in glob.glob(os.path.join(res, "overlap", "R*.json")):
        d = load(path)
        if d:
            meas[d["reserve"]] = d
    if 0 not in meas or 0 not in pr:
        return []
    rows = []
    for r in sorted(meas):
        if r not in pr:
            continue
        m_rel = meas[r]["gemm_us"] / meas[0]["gemm_us"]
        p_rel = pr[r]["time_us"] / pr[0]["time_us"]
        rows.append(dict(reserve=r, meas_us=meas[r]["gemm_us"], pred_us=pr[r]["time_us"], waves=pr[r].get("waves"),
                         meas_rel=m_rel, pred_rel=p_rel, err_rel=err(p_rel, m_rel),
                         kernel=(meas[r]["ranks"][0].get("kernel") or [None])[0]))
    return rows


# ---------------------------------------------------------------------------
# variance and stragglers


def variance(res: str) -> dict:
    out = {}
    for tag in ("isolated", "concurrent", "isolated_end"):
        d = load(os.path.join(res, "variance" if tag != "isolated_end" else "variance_end", f"{tag}.json"))
        if d:
            out[tag] = d["node"]
    return out


def straggler(res: str, pred: dict) -> List[dict]:
    d = load(os.path.join(res, "straggler", "straggler.json"))
    if not d:
        return []
    ranks = d["ranks"]
    rows = []
    rng = random.Random(0)
    for key, s in sorted(d["summary"].items(), key=lambda kv: (kv[1]["every"], kv[1]["ranks"])):
        n, every = s["ranks"], s["every"]
        free = [r["free_us"] for r in ranks[:n]]
        pooled = [x for f in free for x in f]
        mu = statistics.fmean(pooled)
        sigma = statistics.pstdev(pooled) / mu
        # Bootstrap: E[max over ranks of a sum of `every` kernels] / (every * mean) - 1, from the samples.
        trials = []
        for _ in range(2000):
            trials.append(max(sum(rng.choice(f) for _ in range(every)) for f in free))
        boot = statistics.fmean(trials) / (every * mu) - 1
        closed = sigma * E_MAX.get(n, math.sqrt(2 * math.log(n))) / math.sqrt(every)
        sim5 = pred.get("straggler", {}).get("sim_tax_sync_every_kernel", {}).get(str(n)) if every == 1 else None
        rows.append(dict(ranks=n, every=every, meas_tax=s["tax"], sigma=sigma, closed_form_tax=closed,
                         bootstrap_tax=boot, err_closed=err(closed, s["tax"]) if s["tax"] > 0 else None,
                         sim_at_5pct=sim5))
    return rows


# ---------------------------------------------------------------------------
# H-Net


def hnet(res: str, pred: dict) -> dict:
    d = load(os.path.join(res, "fanout", "hnet.json"))
    h = pred.get("hnet")
    if not d or not h:
        return {}
    out = {"batch1": [], "prefill": [], "bpe_batched": []}
    bpe_b1 = next((r for r in h.get("batch1", []) if r["model"] == "bpe-transformer-xl"), None)
    for name, key in (("hnet_1stage_XL", "hnet-1stage-xl"), ("hnet_2stage_XL", "hnet-2stage-xl")):
        m = d.get(name, {})
        p = next((r for r in h.get("batch1", []) if r["model"] == key and r.get("routing") == "sampled"), None)
        if "decode_b1_us_per_byte" in m and p:
            out["batch1"].append(dict(model=key, meas_us_per_byte=m["decode_b1_us_per_byte"], pred_us_per_byte=p["us_per_byte"],
                                      err=err(p["us_per_byte"], m["decode_b1_us_per_byte"]),
                                      meas_ratio=m.get("ratio_vs_bpe_graph"), pred_ratio=p["ratio"],
                                      err_ratio=err(p["ratio"], m.get("ratio_vs_bpe_graph")),
                                      meas_kernels_per_byte=m.get("kernels_per_byte"), pred_ops_per_byte=p.get("ops_per_byte"),
                                      main_net_rate=m.get("main_net_rate_per_stage")))
        pp = next((r for r in h.get("prefill", []) if r["model"] == key), None)
        if "prefill_us" in m and pp:
            out["prefill"].append(dict(model=key, meas_us=m["prefill_us"], pred_us=pp["us"], err=err(pp["us"], m["prefill_us"])))
    bpe = d.get("bpe", {})
    pp = next((r for r in h.get("prefill", []) if r["model"] == "bpe-transformer-xl"), None)
    if "prefill_us" in bpe and pp:
        out["prefill"].append(dict(model="bpe-transformer-xl", meas_us=bpe["prefill_us"], pred_us=pp["us"],
                                   err=err(pp["us"], bpe["prefill_us"])))
    preds = {r["batch"]: r["step_us"] for r in h.get("bpe_steps", [])}
    if bpe_b1:
        preds[1] = bpe_b1["step_us"]
    for b, p in sorted(preds.items()):
        m = bpe.get(str(b))
        if m:
            out["bpe_batched"].append(dict(batch=b, meas_graph_us=m["step_us_graph"], meas_eager_us=m["step_us_eager"],
                                           pred_us=p, err=err(p, m["step_us_graph"])))
    return out


# ---------------------------------------------------------------------------


def report(r: dict) -> str:
    L = [f"# Node day vs predictions: {r['gpu']}", "",
         f"Predictions: `{r['predictions']}` (commit `{r['pred_sha']}`, {r['pred_time']}). "
         f"Measured on {r['measured_on']}. Errors are (predicted − measured) / measured: negative means "
         "the simulator is optimistic.", ""]
    s = r["nccl"]["summary"]
    if s:
        L += ["## Collectives (nccl-tests, out-of-place)", "",
              "| collective | GPUs | algorithm | median error ≥ 1 MiB | median abs error | small-message error | peak busbw meas / pred |",
              "|---|---|---|---|---|---|---|"]
        for x in sorted(s, key=lambda x: (x["coll"], -x["ranks"], x["algo"])):
            L.append(f"| {x['coll']} | {x['ranks']} | {x['algo']} | {pct(x['median_err_ge_1MiB'])} | "
                     f"{pct(x['median_abs_err_ge_1MiB'])} | {pct(x['latency_err'])} | "
                     f"{x['peak_busbw_meas']:.0f} / {x['peak_busbw_pred']:.0f} GB/s |")
        L += ["", "Headline sizes, 8 GPUs, default algorithm:", "", "| collective | size | measured | predicted | error |",
              "|---|---|---|---|---|"]
        for p in r["nccl"]["points"]:
            if p["ranks"] == 8 and p["algo"] == "auto" and p["bytes"] in NCCL_HEADLINE_SIZES:
                L.append(f"| {p['coll']} | {p['bytes'] / 2**20:g} MiB | {us(p['meas_us'])} | {us(p['pred_us'])} "
                         f"({p['pred_algo']}) | {pct(p['err'])} |")
    for block, title in (("train", "Training steps"), ("decode", "Decode steps (median inter-token latency)")):
        if r[block]:
            L += ["", f"## {title}", "", "| case | layers | measured | sim | replay | textbook | sim error | replay error | textbook error |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for x in r[block]:
                L.append(f"| {x['case']} | {x['layers']} | {us(x['meas_us'])} | {us(x['sim_us'])} | {us(x['replay_us'])} | "
                         f"{us(x['textbook_us'])} | {pct(x['err_sim'])} | {pct(x['err_replay'])} | {pct(x['err_textbook'])} |")
    for tag, rows in r["power"].items():
        if rows:
            L += ["", f"## Clock and power ({'GPU 0 alone' if tag == 'gpu0' else 'all GPUs at once'})", "",
                  "| GPU | case | measured | predicted | time error | MHz meas / pred | MHz error | W | capped |", "|---|---|---|---|---|---|---|---|---|"]
            for x in rows:
                if tag == "all" and x["gpu"] not in (0, 7):
                    continue
                L.append(f"| {x['gpu']} | {x['case']} | {us(x['meas_us'])} | {us(x.get('pred_us'))} | {pct(x.get('err_time'))} | "
                         f"{x['meas_mhz'] or 0:.0f} / {x.get('pred_mhz') or 0:.0f} | {pct(x.get('err_mhz'))} | "
                         f"{x['power_w'] or 0:.0f} | {x['capped'] or 0:.0%} |")
    if r["overlap"]:
        L += ["", "## GEMM beside NCCL (time relative to no all-reduce)", "",
              "| SMs held by NCCL | measured | predicted (waves) | relative meas / pred | error | cuBLAS kernel |", "|---|---|---|---|---|---|"]
        for x in r["overlap"]:
            L.append(f"| {x['reserve']} | {us(x['meas_us'])} | {us(x['pred_us'])} ({x['waves']}) | "
                     f"{x['meas_rel']:.3f} / {x['pred_rel']:.3f} | {pct(x['err_rel'])} | `{(x['kernel'] or '')[:60]}` |")
    if r["variance"]:
        L += ["", "## Variance across the node", "", "| run | workload | GPU-to-GPU spread | GPU-to-GPU σ | slowest GPU | within-GPU σ |",
              "|---|---|---|---|---|---|"]
        for tag, node in r["variance"].items():
            for w, x in node.items():
                L.append(f"| {tag} | {w} | {x['gpu_spread']:.2%} | {x['gpu_cv']:.2%} | {x['slowest_gpu']} "
                         f"(+{x['slowest_over_mean'] - 1:.2%}) | {x['within_gpu_cv_median']:.2%} |")
    if r["straggler"]:
        L += ["", "## Straggler tax (BF16 4096³ GEMM, all-reduce every k GEMMs)", "",
              "| ranks | k | measured | σ per kernel | closed form σ·E[max]/√k | bootstrap | closed-form error | sim at σ = 5% |",
              "|---|---|---|---|---|---|---|---|"]
        for x in r["straggler"]:
            sim5 = "" if x["sim_at_5pct"] is None else f"{x['sim_at_5pct']:+.1%}"
            L.append(f"| {x['ranks']} | {x['every']} | {x['meas_tax']:+.2%} | {x['sigma']:.2%} | {x['closed_form_tax']:+.2%} | "
                     f"{x['bootstrap_tax']:+.2%} | {pct(x['err_closed'])} | {sim5} |")
    h = r["hnet"]
    if h:
        L += ["", "## H-Net vs BPE", ""]
        if h.get("batch1"):
            L += ["| model | µs/byte meas / pred | error | ÷ BPE meas / pred | ratio error | kernels/byte meas / pred |",
                  "|---|---|---|---|---|---|"]
            for x in h["batch1"]:
                L.append(f"| {x['model']} (batch 1) | {x['meas_us_per_byte']:.0f} / {x['pred_us_per_byte']:.0f} | {pct(x['err'])} | "
                         f"{(x['meas_ratio'] or 0):.2f} / {x['pred_ratio']:.2f} | {pct(x['err_ratio'])} | "
                         f"{x['meas_kernels_per_byte']} / {(x['pred_ops_per_byte'] or 0):.0f} |")
        if h.get("prefill"):
            L += ["", "| prefill, 8 × 8 KiB | measured | predicted | error |", "|---|---|---|---|"]
            for x in h["prefill"]:
                L.append(f"| {x['model']} | {us(x['meas_us'])} | {us(x['pred_us'])} | {pct(x['err'])} |")
        if h.get("bpe_batched"):
            L += ["", "| BPE decode step | measured (graph / eager) | predicted | error |", "|---|---|---|---|"]
            for x in h["bpe_batched"]:
                L.append(f"| batch {x['batch']} | {us(x['meas_graph_us'])} / {us(x['meas_eager_us'])} | {us(x['pred_us'])} | {pct(x['err'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--predictions", default="")
    a = ap.parse_args()
    res = os.path.abspath(os.path.expanduser(a.results))
    key = open(os.path.join(res, "env", "gpu-key.txt")).read().strip() if os.path.exists(
        os.path.join(res, "env", "gpu-key.txt")) else "b300"
    ppath = a.predictions or os.path.join(HERE, "predictions", f"hgx_{key}_8.json")
    pred = json.load(open(ppath))
    max_mhz = None
    gpus_csv = os.path.join(res, "env", "gpus.csv")
    if os.path.exists(gpus_csv):
        m = re.findall(r"(\d+) MHz", open(gpus_csv).read())
        max_mhz = float(m[0]) if m else None
    r = dict(gpu=key, predictions=os.path.relpath(ppath, os.getcwd()), pred_sha=pred.get("git_sha"),
             pred_time=pred.get("created_utc"), measured_on=os.path.basename(res),
             nccl=nccl(res, pred), train=train(res, pred), decode=decode(res, pred),
             power=gemm_power(res, pred, max_mhz), overlap=overlap(res, pred), variance=variance(res),
             straggler=straggler(res, pred), hnet=hnet(res, pred))
    with open(os.path.join(res, "compare.json"), "w") as f:
        json.dump(r, f, indent=1)
    text = report(r)
    with open(os.path.join(res, "COMPARE.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
