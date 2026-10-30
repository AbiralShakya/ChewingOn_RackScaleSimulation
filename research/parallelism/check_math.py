"""Every parallelism on GB300 NVL72: closed-form math vs roofline replay vs cycle-level simulation.

For each case this runs :func:`rackscale.scaling.check`: the textbook estimate
(one formula per axis), the roofline replay of the exact lowered program, and
the simulator in fast mode, all on the same truncated model (``layers``).
Errors are signed, ``(sim - x) / x``: positive means the simulated step is
slower than the math.

    python research/parallelism/check_math.py            # all cases, 2 workers
    python research/parallelism/check_math.py --only fsdp
    python research/parallelism/check_math.py --table    # rebuild the table from results/check.jsonl

Results go to results/check.jsonl (one JSON object per case, resumable) and
results/check.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from rackscale.scaling import Plan, check

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "check.jsonl")

T8 = dict(model="llama3-8b", phase="train", layers=2, micro_batch=1, seqlen=4096)
CASES = [
    ("1 GPU", T8, Plan()),
    ("DDP", T8, Plan(dp=8, zero=0)),
    ("ZeRO-2", T8, Plan(dp=8, zero=2)),
    ("ZeRO-2 + optimizer", {**T8, "optimizer": True}, Plan(dp=8, zero=2)),
    ("FSDP", T8, Plan(dp=8, zero=3)),
    ("HSDP 4x2", T8, Plan(dp=8, zero=3, shard=4)),
    ("TP8", T8, Plan(tp=8)),
    ("TP8 + SP", T8, Plan(tp=8, sp=True)),
    ("TP2 x DP4", T8, Plan(tp=2, dp=4)),
    ("TP2 x FSDP4 (HSDP 2)", T8, Plan(tp=2, dp=4, zero=3, shard=2)),
    ("PP2 GPipe m8", {**T8, "micro_batches": 8}, Plan(pp=2, schedule="gpipe")),
    ("PP2 1F1B m8", {**T8, "micro_batches": 8}, Plan(pp=2, schedule="1f1b")),
    ("PP2 ZB-H1 m8", {**T8, "micro_batches": 8}, Plan(pp=2, schedule="zb")),
    ("TP2 x PP2 x DP2 1F1B m4", {**T8, "micro_batches": 4}, Plan(tp=2, pp=2, dp=2)),
    ("prefill TP4", dict(model="llama3-8b", phase="prefill", layers=2, batch=1, seqlen=8192), Plan(tp=4)),
    ("prefill TP4 + SP", dict(model="llama3-8b", phase="prefill", layers=2, batch=1, seqlen=8192), Plan(tp=4, sp=True)),
    ("prefill CP8 ring", dict(model="llama3-8b", phase="prefill", layers=2, seqlen=65536), Plan(cp=8)),
    ("prefill CP8 Ulysses", dict(model="llama3-8b", phase="prefill", layers=2, seqlen=65536), Plan(cp=8, cp_kind="ulysses")),
    ("decode 1 GPU b64", dict(model="llama3-8b", phase="decode", layers=2, batch=64, context=8192), Plan()),
    ("decode TP8 b64 (70B)", dict(model="llama3-70b", phase="decode", layers=2, batch=64, context=8192), Plan(tp=8)),
    ("decode MoE TP4 (Qwen3-30B-A3B)", dict(model="qwen3-30b-a3b", phase="decode", layers=2, batch=64, context=8192),
     Plan(tp=4)),
    ("decode EP8 (DeepSeek-V3)", dict(model="deepseek-v3", phase="decode", layers=4, batch=64, context=4096), Plan(ep=8)),
]


def run_case(case) -> dict:
    name, spec, plan = case
    spec = dict(spec)
    model, phase, layers = spec.pop("model"), spec.pop("phase"), spec.pop("layers")
    r = check(model, "gb300", phase, plan, layers=layers, **spec)
    r["case"] = name
    return r


def table(rows) -> str:
    out = ["| case | GPUs | textbook µs | replay µs | sim µs | sim vs replay | sim vs textbook | kernel efficiency µs | power clock µs | collectives µs | overlap + idle µs |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["gap_us"]
        out.append(f"| {r['case']} | {r['world']} | {r['textbook_us']:.0f} | {r['replay_us']:.0f} | {r['sim_us']:.0f} | "
                   f"{100 * r['err_vs_replay']:+.1f}% | {100 * r['err_vs_textbook']:+.1f}% | {g['kernel_efficiency_us']:+.0f} | {g['power_clock_us']:+.0f} | "
                   f"{g['collectives_us']:+.0f} | {g['overlap_and_idle_us']:+.0f} |")
    out += ["", "Collectives, simulated vs closed form (mean per call):", "",
            "| case | kind | algo | size | ranks | n | closed form µs | sim µs | error |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for c in r.get("collectives", []):
            out.append(f"| {r['case']} | {c['kind']} | {c['algo']} | {c['bytes'] / 2**20:.1f} MiB | {c['ranks']} | {c['n']} | "
                       f"{c['closed_form_us']:.1f} | {c['sim_us']:.1f} | {100 * c['err']:+.1f}% |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    done = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done[r["case"]] = r
    if not a.table:
        todo = [c for c in CASES if a.only.lower() in c[0].lower() and (a.only or c[0] not in done)]
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(run_case, todo):
                done[r["case"]] = r
                with open(OUT, "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{r['case']:<34} replay {r['replay_us']:9.0f}  sim {r['sim_us']:9.0f}  {100 * r['err_vs_replay']:+6.1f}%",
                      flush=True)
    rows = [done[c[0]] for c in CASES if c[0] in done]
    with open(os.path.join(HERE, "results", "check.md"), "w") as f:
        f.write(table(rows))
    print(table(rows))


if __name__ == "__main__":
    sys.exit(main())
