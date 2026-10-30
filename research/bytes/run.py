"""Tokens vs bytes on Blackwell: H-Net against a BPE Transformer and a byte-level Transformer.

Everything is per *byte* of text, so the three model families are compared on
the same work. The BPE model reads ~4.45 bytes per token (Llama-3 tokenizer on
English, Cartesia part 1).

    python research/bytes/run.py            # all studies -> results/bytes.json, results/bytes.md
    python research/bytes/run.py --quick    # smaller decode run
    python research/bytes/run.py --only decode   # rerun one study, keep the others

Studies:
  flops    FLOPs per byte of each model vs the H-Net paper's Table 2 (signed error)
  prefill  bytes/s of one GB300 on 8 prompts of 8 KiB
  decode   per-byte latency and bytes/s at batch 64, 8 KiB context; H-Net with the
           routers drawn per step, with the same draw spread evenly over the steps
           (its mean without its variability), and with the expected batch (the
           mean-field assumption an analytic model makes)
  batch    per-byte decode time at batch 16, 64 and 256 (H-Net at the expected batch)
  closed   the bytes each decode step moves, in closed form (hnet.decode_traffic), and
           the time per byte at HBM bandwidth, at the same batches (no simulation)

Every simulated run also records where its time goes (outer stages, main-network
attention, matmuls and elementwise ops, LM head, launch gaps) from the per-op trace.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import rackscale as rs
from rackscale import hnet, lower, scaling
from rackscale.lower import Parallel, _model_flops
from rackscale.sweep import SYSTEMS, _system_text

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = _system_text(SYSTEMS["b300"])
GAP = SYSTEMS["b300"].graph_gap_ns
BPT = hnet.BYTES_PER_BPE_TOKEN
BETA = scaling.hardware("b300").hbm_bw
#: H-Net paper Table 2, GFLOPs per byte (XL scale).
PAPER = {"hnet-1stage-xl": 0.72, "hnet-2stage-xl": 0.69, "bpe-transformer-xl": 0.69}


def _part(op: str) -> str:
    n = op.split(".", 1)[1] if op[:1] == "t" and op.split(".", 1)[0][1:].isdigit() else op
    if n.startswith("s") and n[1:2].isdigit():
        return "outer stages"
    n = n.removeprefix("main.")
    if n in ("attention", "attn_combine"):
        return "main attention"
    if n.endswith("_proj"):
        return "main matmuls"
    if n == "lm_head":
        return "LM head"
    return "main norms and elementwise"


PARTS = ("outer stages", "main attention", "main matmuls", "main norms and elementwise", "LM head", "launch gaps")


def sim(wl: dict) -> dict:
    """Simulate on one B300; ``parts_us`` splits the makespan by part of the
    network from the per-op trace (one stream, so launch gaps are the
    makespan minus the ops)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        r = rs.simulate(SYS, wl, fast=True, launch_gap_ns=GAP, trace=path).raw
        ev = json.load(open(path))
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    parts = dict.fromkeys(PARTS, 0.0)
    ops = 0
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            parts[_part(e["name"])] += e["dur"]
            ops += 1
    parts["launch gaps"] = r["makespan_us"] - sum(parts.values())
    return dict(us=r["makespan_us"], busy_us=r["ranks"][0]["gpu_busy_us"], kernels=len(r["kernels"]), ops=ops,
                parts_us=parts, top=[(k["name"], round(k["total_us"], 1)) for k in r["kernels"][:6]])


def flops(nbytes: int = 8192) -> list:
    rows = []
    for h in hnet.ALL.values():
        f = hnet.flops_per_byte(h, nbytes) / 1e9
        rows.append(dict(model=h.name, params_b=h.total_params() / 1e9, gflops_per_byte=f, paper=PAPER.get(h.name)))
    for m, bpb in ((hnet.BPE_XL, BPT), (hnet.BYTE_XL, 1.0)):
        t = int(nbytes / bpb)
        f = _model_flops(m, t, t // 2, False, True) / nbytes / 1e9
        rows.append(dict(model=m.name, params_b=m.active_params(False) / 1e9, gflops_per_byte=f, paper=PAPER.get(m.name)))
    for r in rows:
        r["err"] = (r["gflops_per_byte"] - r["paper"]) / r["paper"] if r["paper"] else None
    return rows


def prefill(batch: int = 8, nbytes: int = 8192) -> list:
    rows = []
    for h in hnet.ALL.values():
        s = sim(hnet.prefill(h, Parallel(), batch, nbytes))
        rows.append(dict(model=h.name, **s, bytes_per_s=batch * nbytes / (s["us"] * 1e-6)))
    t = int(nbytes / BPT)
    s = sim(lower.prefill(hnet.BPE_XL, Parallel(), batch, t))
    rows.append(dict(model=hnet.BPE_XL.name, **s, bytes_per_s=batch * nbytes / (s["us"] * 1e-6), tokens=t))
    s = sim(lower.prefill(hnet.BYTE_XL, Parallel(), batch, nbytes))
    rows.append(dict(model=hnet.BYTE_XL.name, **s, bytes_per_s=batch * nbytes / (s["us"] * 1e-6)))
    return rows


def decode(batch: int = 64, context: int = 8192, steps: int = 32) -> list:
    rows = []
    for h in hnet.ALL.values():
        drawn = hnet.step_counts(h, batch, steps)
        for routing, counts in (("sampled", drawn), ("sampled, evened", hnet.even_counts(drawn)),
                                ("mean field", hnet.step_counts(h, batch, steps, mean_field=True))):
            s = sim(hnet.decode_steps(h, Parallel(), batch, context, counts=counts))
            per = s["us"] / steps
            main = [c[-1] for c in counts]
            rows.append(dict(model=h.name, routing=routing, us_per_byte=per,
                             bytes_per_s=batch / (per * 1e-6), main_batch_mean=sum(main) / len(main),
                             main_batch_min=min(main), main_batch_max=max(main), **s))
    t = int(context / BPT)
    s = sim(lower.decode_step(hnet.BPE_XL, Parallel(), batch, t))
    rows.append(dict(model=hnet.BPE_XL.name, routing="token", us_per_token=s["us"], us_per_byte=s["us"] / BPT,
                     bytes_per_s=batch * BPT / (s["us"] * 1e-6), **s))
    s = sim(lower.decode_step(hnet.BYTE_XL, Parallel(), batch, context))
    rows.append(dict(model=hnet.BYTE_XL.name, routing="byte", us_per_byte=s["us"], bytes_per_s=batch / (s["us"] * 1e-6), **s))
    return rows


def batch_sweep(batches=(16, 64, 256), context: int = 8192, steps: int = 4) -> list:
    rows = []
    for b in batches:
        for h in hnet.ALL.values():
            s = sim(hnet.decode_steps(h, Parallel(), b, context, counts=hnet.step_counts(h, b, steps, mean_field=True)))
            rows.append(dict(model=h.name, batch=b, us_per_byte=s["us"] / steps, bytes_per_s=b * steps / (s["us"] * 1e-6),
                             **s))
        s = sim(lower.decode_step(hnet.BPE_XL, Parallel(), b, int(context / BPT)))
        rows.append(dict(model=hnet.BPE_XL.name, batch=b, us_per_byte=s["us"] / BPT,
                         bytes_per_s=b * BPT / (s["us"] * 1e-6), **s))
    return rows


def closed(batches=(16, 64, 256), context: int = 8192) -> list:
    rows = []
    for b in batches:
        for m in (*hnet.ALL.values(), hnet.BPE_XL, hnet.BYTE_XL):
            t = hnet.decode_traffic(m, b, context)
            parts = {k: v / 1e9 for k, v in t.items() if k not in ("step", "per_byte")}
            rows.append(dict(model=m.name, batch=b, gb_per_step=parts, gb_per_byte=t["per_byte"] / 1e9,
                             us_per_byte=t["per_byte"] / BETA * 1e6))
    return rows


def table(res: dict) -> str:
    out = ["## FLOPs per byte (8 KiB sequences)", "", "| model | params | GFLOP/byte | paper | error |", "|---|---|---|---|---|"]
    for r in res["flops"]:
        e = f"{100 * r['err']:+.1f}%" if r["err"] is not None else ""
        out.append(f"| {r['model']} | {r['params_b']:.2f} B | {r['gflops_per_byte']:.3f} | {r['paper'] or ''} | {e} |")
    out += ["", "## Prefill, one B300, 8 prompts of 8 KiB", "", "| model | step µs | MB/s of text |", "|---|---|---|"]
    for r in res["prefill"]:
        out.append(f"| {r['model']} | {r['us']:.0f} | {r['bytes_per_s'] / 1e6:.2f} |")
    out += ["", "## Decode, one B300, batch 64, 8 KiB context", "",
            "| model | routing | µs per byte | KB/s of text | main-net batch mean (min-max) |", "|---|---|---|---|---|"]
    for r in res["decode"]:
        mb = (f"{r['main_batch_mean']:.1f} ({r['main_batch_min']}-{r['main_batch_max']})" if "main_batch_mean" in r else "")
        out.append(f"| {r['model']} | {r['routing']} | {r['us_per_byte']:.1f} | {r['bytes_per_s'] / 1e3:.0f} | {mb} |")
    if "batch" in res:
        bs = sorted({r["batch"] for r in res["batch"]})
        out += ["", "## Decode by batch (µs per byte; KB/s of text per GPU)", "",
                "| model | " + " | ".join(f"batch {b}" for b in bs) + " |", "|---" * (len(bs) + 1) + "|"]
        for m in dict.fromkeys(r["model"] for r in res["batch"]):
            cells = []
            for b in bs:
                r = next(x for x in res["batch"] if x["model"] == m and x["batch"] == b)
                cells.append(f"{r['us_per_byte']:.0f} ({r['bytes_per_s'] / 1e3:.0f})")
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        if "closed" in res:
            def ratio(rows, m, b):
                pick = lambda name: next(x for x in rows if x["model"] == name and x["batch"] == b)["us_per_byte"]
                return pick(m) / pick(hnet.BPE_XL.name)
            for m in hnet.ALL:
                out.append(f"| {m} ÷ BPE, simulated | " + " | ".join(f"{ratio(res['batch'], m, b):.2f}x" for b in bs) + " |")
                out.append(f"| {m} ÷ BPE, closed form | " + " | ".join(f"{ratio(res['closed'], m, b):.2f}x" for b in bs) + " |")
    if "closed" in res:
        out += ["", f"## Decode in closed form (bytes moved; µs at {BETA / 1e12:.2f} TB/s)", "",
                "| model | batch | GB per step | GB per byte | µs per byte |", "|---|---|---|---|---|"]
        for r in res["closed"]:
            parts = " + ".join(f"{v:.2f} {k}" for k, v in r["gb_per_step"].items() if v > 0.005)
            out.append(f"| {r['model']} | {r['batch']} | {parts} | {r['gb_per_byte']:.2f} | {r['us_per_byte']:.0f} |")
    out += ["", "## Where the time goes", "", "Prefill: ms per step of 8 prompts. Decode: µs per byte of every "
            "sequence (a BPE step makes 4.45 bytes; H-Net rows are the mean-field run).", "",
            "| run | model | ops | " + " | ".join(PARTS) + " |", "|---" * (len(PARTS) + 3) + "|"]
    for r in res["prefill"]:
        if "parts_us" in r:
            out.append(f"| prefill | {r['model']} | {r['ops']} | " +
                       " | ".join(f"{r['parts_us'][k] / 1e3:.1f}" for k in PARTS) + " |")
    for r in res["decode"]:
        if "parts_us" in r and r["routing"] in ("mean field", "token", "byte"):
            per = r["us"] / r["us_per_byte"]  # bytes per sequence in the run
            out.append(f"| decode | {r['model']} | {r['ops'] / per:.0f} per byte | " +
                       " | ".join(f"{r['parts_us'][k] / per:.0f}" for k in PARTS) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated studies to (re)run; keep the others' results")
    a = ap.parse_args()
    path = os.path.join(HERE, "results", "bytes.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    studies = dict(flops=flops, prefill=prefill, decode=lambda: decode(steps=8 if a.quick else 32), batch=batch_sweep,
                   closed=closed)
    for name, fn in studies.items():
        if a.only and name not in a.only.split(","):
            continue
        res[name] = fn()
        print(f"== {name}", flush=True)
        # Saved after each study, so a later failure keeps the earlier ones.
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
    if all(k in res for k in studies):
        md = table(res)
        with open(os.path.join(HERE, "results", "bytes.md"), "w") as f:
            f.write(md)
        print(md)


if __name__ == "__main__":
    main()
