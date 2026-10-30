"""Pre-register the simulator's predictions for the 8-GPU node day.

Everything the node day measures is predicted here first, from the code on
``main``, and the output is committed before the node is touched. ``compare.py``
later reads these files, never a fresh simulation, so a prediction cannot be
tuned after the fact. Each file records the git commit it came from.

    python calibration/nodeday/predict.py                       # all systems, all blocks
    python calibration/nodeday/predict.py --system b300 --only nccl,gemm
    python calibration/nodeday/predict.py --table               # rebuild PREDICTIONS.md

Systems are the three nodes we might rent: HGX B300 (calibrated die), HGX
B200 and HGX H100 (presets, not calibrated). Output goes to
``predictions/<system>.json`` and a readable summary to ``PREDICTIONS.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "predictions")
RSIM = os.environ.get("RSIM", os.path.join(ROOT, "target", "release", "rsim"))

#: key -> (rackscale.sweep system key, rsim system preset, rsim GPU preset, SMs)
SYSTEMS = {
    "b300": ("b300", "hgx_b300_8", "b300", 148),
    "b200": ("b200", "hgx_b200_8", "b200", 148),
    "h100": ("h100", "hgx_h100_8", "h100_sxm", 132),
}

# --------------------------------------------------------------------------
# cases; the node-day scripts read the same lists (cases.py), so a measured
# run and its prediction are the same shape by construction.

sys.path.insert(0, HERE)
from cases import DECODE, GEMM_SIZES, NCCL_ALGOS, NCCL_COLLS, NCCL_RANKS, OVERLAP_RESERVE, TRAIN  # noqa: E402


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "python", "crates", "configs"], cwd=ROOT)
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# training and decode steps (rackscale.scaling.check: textbook, replay, sim)


def _plan(d: dict):
    from rackscale.scaling import Plan
    return Plan(**d)


def run_step(job) -> dict:
    """One training or decode case at one depth on one system."""
    key, case, layers = job
    from rackscale.scaling import check
    sweep_key = SYSTEMS[key][0]
    spec = dict(case["spec"])
    model, phase = spec.pop("model"), spec.pop("phase")
    if layers == "full":
        from rackscale.models import ALL
        layers_n = ALL[model].layers
    else:
        layers_n = int(layers)
    try:
        r = check(model, sweep_key, phase, _plan(case["plan"]), layers=layers_n, **spec)
    except Exception as e:  # a case that does not fit or does not lower on this system
        return dict(case=case["name"], layers=layers_n, error=f"{type(e).__name__}: {e}")
    keep = ("textbook_us", "replay_us", "sim_us", "world", "label", "sim_kernel_us", "sim_collective_us",
            "replay_kernel_us", "replay_collective_us", "gap_us", "sim_mfu", "dominant")
    out = {k: r[k] for k in keep if k in r}
    out["collectives"] = [{k: c[k] for k in ("kind", "algo", "bytes", "ranks", "n", "sim_us", "closed_form_us")}
                          for c in r.get("collectives", [])]
    out["top_kernels"] = r.get("sim_kernels", [])[:8]
    return dict(case=case["name"], layers=layers_n, **out)


def steps(key: str, which: str, jobs: int) -> List[dict]:
    cases = TRAIN if which == "train" else DECODE
    todo = []
    for c in cases:
        if key in c.get("skip", ()):
            continue
        for layers in c["layers"]:
            todo.append((key, c, layers))
    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(run_step, todo))
    for r in rows:
        s = f"{r['sim_us']:10.0f} us" if "sim_us" in r else r["error"]
        print(f"  [{key}] {which:5s} {r['case']:<34} L{r['layers']:<3} {s}", flush=True)
    return rows


# --------------------------------------------------------------------------
# collectives (rsim net sweep, the nccl-tests convention for size and busbw)


def rsim_json(args: List[str]) -> list:
    out = subprocess.run([RSIM] + args + ["--json"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def nccl(key: str) -> List[dict]:
    system = SYSTEMS[key][1]
    rows = []
    jobs = [(c, n, "auto") for c in NCCL_COLLS for n in NCCL_RANKS]
    jobs += [(c, 8, a) for c, algos in NCCL_ALGOS.items() for a in algos]
    for coll, n, algo in jobs:
        try:
            pts = rsim_json(["net", "sweep", "--system", system, "--coll", coll, "--ranks", str(n), "--algo", algo,
                             "-b", "8", "-e", "8G", "-f", "2"])
        except subprocess.CalledProcessError as e:
            rows.append(dict(coll=coll, ranks=n, algo_forced=algo, error=e.stderr.strip()[-300:]))
            continue
        for p in pts:
            rows.append(dict(coll=coll, ranks=n, algo_forced=algo, bytes=p["bytes"],
                             time_us=(p["done_ps"] - p["last_post_ps"]) * 1e-6, busbw_gbps=p["busbw_gbps"],
                             algo=p["algo"], proto=p["proto"], channels=p["channels"],
                             nccl_model_us=p.get("nccl_model_us")))
        print(f"  [{key}] nccl  {coll:<15} n={n} algo={algo:<5} {len(pts)} sizes", flush=True)
    return rows


# --------------------------------------------------------------------------
# single kernels (rsim kernel): GEMM time and sustained clock, reserved SMs

_TIME = re.compile(r"time\s+([\d.]+) us")
_CLOCK = re.compile(r"sustained SM clock (\d+) MHz")
_NAME = re.compile(r"^(sm\S+_gemm\S+)", re.M)


def rsim_kernel(gpu: str, dims: List[int], dtype: str = "bf16", reserve: int = 0) -> dict:
    args = [RSIM, "kernel", "--gpu", gpu, "--dtype", dtype, "gemm"] + [str(d) for d in dims]
    if reserve:
        args += ["--reserve-sms", str(reserve)]
    p = subprocess.run(args, capture_output=True, text=True)
    text = p.stdout + p.stderr
    t = _TIME.search(text)
    if p.returncode or not t:
        return dict(error=text.strip()[-300:])
    clk = _CLOCK.search(text)
    name = _NAME.search(text)
    us = float(t.group(1))
    flops = 2.0 * dims[0] * dims[1] * dims[2]
    return dict(time_us=us, tflops=flops / us * 1e-6, sustained_mhz=float(clk.group(1)) if clk else None,
                kernel=name.group(1) if name else None)


def _gemm_job(job):
    gpu, dtype, n = job
    return dict(dtype=dtype, m=n, n=n, k=n, **rsim_kernel(gpu, [n, n, n], dtype))


def _overlap_job(job):
    gpu, sms, r = job
    m = OVERLAP_RESERVE["gemm"]
    out = dict(reserve_sms=r, m=m[0], n=m[1], k=m[2], **rsim_kernel(gpu, m, "bf16", r))
    # The wave rule the article states, for 128x256 tiles: ceil(tiles / free SMs).
    tiles = math.ceil(m[0] / 128) * math.ceil(m[1] / 256)
    out["waves"] = math.ceil(tiles / max(1, sms - r))
    return out


def gemm(key: str, jobs: int) -> List[dict]:
    gpu = SYSTEMS[key][2]
    todo = [(gpu, d, n) for d in ("bf16", "fp8") for n in GEMM_SIZES]
    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(_gemm_job, todo))
    print(f"  [{key}] gemm  {len(rows)} shapes", flush=True)
    return rows


def overlap(key: str, jobs: int) -> List[dict]:
    gpu, sms = SYSTEMS[key][2], SYSTEMS[key][3]
    with ProcessPoolExecutor(jobs) as ex:
        rows = list(ex.map(_overlap_job, [(gpu, sms, r) for r in OVERLAP_RESERVE["reserve"]]))
    print(f"  [{key}] overlap {len(rows)} points", flush=True)
    return rows


# --------------------------------------------------------------------------
# H-Net vs BPE: the committed chapter-13 results (B300 only)


def hnet(key: str) -> Optional[dict]:
    if key != "b300":
        return None
    d = json.load(open(os.path.join(ROOT, "research", "bytes", "results", "bytes.json")))
    by = {}
    for r in d["batch"]:
        by.setdefault(r["batch"], {})[r["model"]] = r["us_per_byte"]
    ratios = []
    for b, m in sorted(by.items()):
        bpe = m.get("bpe-transformer-xl")
        for name in ("hnet-1stage-xl", "hnet-2stage-xl"):
            if bpe and name in m:
                ratios.append(dict(batch=b, model=name, us_per_byte=m[name], bpe_us_per_byte=bpe,
                                   ratio=m[name] / bpe))
    closed = [dict(batch=r["batch"], model=r["model"], us_per_byte=r["us_per_byte"]) for r in d["closed"]]
    # A BPE row's "us" is one decode step of the whole batch.
    bpe_steps = [dict(batch=r["batch"], step_us=r["us"]) for r in d["batch"] if r["model"] == "bpe-transformer-xl"]
    # Batch 1 is what the released H-Net code can decode; simulate it now (sampled routing,
    # 64 bytes), with the BPE model at the same context.
    sys.path.insert(0, os.path.join(ROOT, "research", "bytes"))
    import run as bytes_run
    b1 = bytes_run.decode(batch=1, steps=64)
    b1_bpe = next(r for r in b1 if r["model"] == "bpe-transformer-xl")
    batch1 = [dict(model=r["model"], routing=r["routing"], us_per_byte=r["us_per_byte"],
                   ratio=r["us_per_byte"] / b1_bpe["us_per_byte"], ops_per_byte=r["ops"] / 64)
              for r in b1 if r["model"].startswith("hnet")]
    batch1.append(dict(model="bpe-transformer-xl", us_per_byte=b1_bpe["us_per_byte"], step_us=b1_bpe["us"],
                       ops_per_token=b1_bpe["ops"]))
    prefill = [dict(model=r["model"], us=r["us"], ops=r["ops"]) for r in d["prefill"]]
    return dict(source="research/bytes/results/bytes.json and research/bytes/run.py decode(batch=1)",
                ratios=ratios, closed_form=closed, bpe_steps=bpe_steps, batch1=batch1, prefill=prefill,
                kernels_per_step={r["model"]: r.get("kernels") for r in d["decode"]})


# --------------------------------------------------------------------------
# stragglers: the closed form compare.py evaluates with the measured sigma


STRAGGLER = dict(
    rule="tax(N) = sigma * E[max of N standard normals]; E[max_8] = 1.4236, E[max_4] = 1.0294, E[max_2] = 0.5642",
    book="docs/book/16-stragglers-flaps-tail-latency.md section 16.1",
    assumed_sigma=0.05,
    sim_tax_sync_every_kernel={"2": 0.058, "4": 0.074, "8": 0.080},
)


# --------------------------------------------------------------------------


BLOCKS = ("nccl", "train", "decode", "gemm", "overlap", "hnet")


def predict(key: str, only: List[str], jobs: int) -> dict:
    path = os.path.join(OUT, f"hgx_{key}_8.json")
    doc = json.load(open(path)) if os.path.exists(path) else {}
    doc.update(schema="rackscale-nodeday-predictions/v1", system=SYSTEMS[key][1], gpu=SYSTEMS[key][2],
               git_sha=git_sha(), created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    for b in only:
        if b == "nccl":
            doc["nccl"] = nccl(key)
        elif b in ("train", "decode"):
            doc[b] = steps(key, b, jobs)
        elif b == "gemm":
            doc["gemm"] = gemm(key, jobs)
        elif b == "overlap":
            doc["overlap"] = overlap(key, jobs)
        elif b == "hnet":
            doc["hnet"] = hnet(key)
    doc["straggler"] = STRAGGLER
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def _us(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{x / 1000:.2f} ms" if x >= 10000 else f"{x:.1f} µs"


def table() -> str:
    from cases import NCCL_HEADLINE_SIZES
    docs = {}
    for key in SYSTEMS:
        p = os.path.join(OUT, f"hgx_{key}_8.json")
        if os.path.exists(p):
            docs[key] = json.load(open(p))
    keys = list(docs)
    lines = ["# Node-day predictions", "",
             "Written by `predict.py` before any measurement. `compare.py` scores the node day against these",
             "files as committed; they are not regenerated afterwards. Commits they came from:", ""]
    for k in keys:
        lines.append(f"- `{docs[k]['system']}`: `{docs[k]['git_sha']}` at {docs[k]['created_utc']}")
    head = "| case | " + " | ".join(f"{docs[k]['system']}" for k in keys) + " |"
    sep = "|---|" + "---|" * len(keys)

    def by(doc, block, **kw):
        for r in doc.get(block) or []:
            if all(r.get(a) == b for a, b in kw.items()):
                return r
        return None

    lines += ["", "## Collectives (nccl-tests out-of-place time, default algorithm, 8 GPUs)", "", head, sep]
    for coll in NCCL_COLLS:
        for size in NCCL_HEADLINE_SIZES:
            cells = []
            for k in keys:
                r = by(docs[k], "nccl", coll=coll, ranks=8, algo_forced="auto", bytes=size)
                cells.append(f"{_us(r['time_us'])} · {r['busbw_gbps']:.0f} GB/s · {r['algo']}/{r['proto']}" if r else "–")
            lines.append(f"| {coll} {size / 2**20:g} MiB | " + " | ".join(cells) + " |")
    for block, title in (("train", "Training steps (forward + backward of the transformer layers, sim)"),
                         ("decode", "Decode steps (one token for every sequence, sim)")):
        lines += ["", f"## {title}", "", head, sep]
        names = []
        for k in keys:
            for r in docs[k].get(block) or []:
                if (r["case"], r["layers"]) not in names:
                    names.append((r["case"], r["layers"]))
        for name, layers in names:
            cells = []
            for k in keys:
                r = by(docs[k], block, case=name, layers=layers)
                if not r:
                    cells.append("–")
                elif "error" in r:
                    cells.append("does not lower")
                else:
                    cells.append(f"{_us(r['sim_us'])} (replay {_us(r['replay_us'])})")
            lines.append(f"| {name}, {layers} layers | " + " | ".join(cells) + " |")
    lines += ["", "## GEMMs (square, one GPU; time and sustained SM clock)", "", head, sep]
    for dtype in ("bf16", "fp8"):
        for n in GEMM_SIZES:
            cells = []
            for k in keys:
                r = by(docs[k], "gemm", dtype=dtype, n=n)
                if not r or "error" in r:
                    cells.append("–")
                else:
                    clk = f", {r['sustained_mhz']:.0f} MHz" if r.get("sustained_mhz") else ", no clock drop"
                    cells.append(f"{_us(r['time_us'])} · {r['tflops']:.0f} TFLOP/s{clk}")
            lines.append(f"| {dtype} {n}³ | " + " | ".join(cells) + " |")
    lines += ["", "## GEMM beside NCCL (4096³ BF16 with R SMs held by communication)", "", head, sep]
    for r0 in docs[keys[0]].get("overlap") or []:
        cells = []
        for k in keys:
            r = by(docs[k], "overlap", reserve_sms=r0["reserve_sms"])
            cells.append(f"{_us(r['time_us'])} · {r['waves']} waves" if r and "time_us" in r else "–")
        lines.append(f"| R = {r0['reserve_sms']} | " + " | ".join(cells) + " |")
    h = docs.get("b300", {}).get("hnet")
    if h:
        lines += ["", "## H-Net ÷ BPE, time per byte (B300, sim)", "", "| batch | model | ratio |", "|---|---|---|"]
        for r in h["ratios"]:
            lines.append(f"| {r['batch']} | {r['model']} | {r['ratio']:.2f}× |")
    lines += ["", "## Stragglers", "", STRAGGLER["rule"] + ".", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="all", help="b300, b200, h100 or all")
    ap.add_argument("--only", default=",".join(BLOCKS))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--table", action="store_true", help="only rebuild PREDICTIONS.md")
    a = ap.parse_args()
    if not a.table:
        if not os.path.exists(RSIM):
            sys.exit(f"{RSIM} not found: cargo build --release first")
        keys = list(SYSTEMS) if a.system == "all" else a.system.split(",")
        for key in keys:
            predict(key, [b for b in a.only.split(",") if b], a.jobs)
    with open(os.path.join(HERE, "PREDICTIONS.md"), "w") as f:
        f.write(table())
    print(f"wrote {os.path.relpath(OUT, ROOT)}/ and PREDICTIONS.md")


if __name__ == "__main__":
    main()
