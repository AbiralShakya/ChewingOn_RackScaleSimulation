"""Check the page's JavaScript closed forms (rsmath.js) against rackscale.scaling.

    python research/learn/verify.py [--cases 600]

Draws random models, phases and plans, evaluates each in Python and in Node,
and fails on any relative difference above 1e-9 in step time, MFU, memory,
throughput or any term. Needs ``node`` and research/learn/data/learn.json
(``export.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

from rackscale import scaling
from rackscale.scaling import Plan

HERE = os.path.dirname(os.path.abspath(__file__))

NODE = r"""
const fs = require("fs");
require(process.argv[2]);
const data = JSON.parse(fs.readFileSync(process.argv[3]));
const cases = JSON.parse(fs.readFileSync(process.argv[4]));
const out = cases.map((c) => {
  const e = rsmath.estimate(data.models[c.model], data.hardware[c.hw], c.phase, c.plan, c.opts);
  return { step_us: e.step_us, mfu: e.mfu, mem: e.memory_total_gb, tok: e.tok_s_gpu, label: e.label,
           terms: e.terms.map((t) => [t.name, t.us]), dominant: e.dominant, fits: e.fits };
});
process.stdout.write(JSON.stringify(out));
"""


def draw(rng: random.Random, data: dict) -> dict:
    model = rng.choice(list(data["models"]))
    md = data["models"][model]
    phase = rng.choice(["train", "decode", "prefill"])
    tp = rng.choice([t for t in md["valid_tp"] if t <= 8] or [1])
    p = dict(tp=tp, sp=rng.random() < 0.3 and tp > 1, pp=rng.choice([1, 1, 2, 4]))
    opts = {}
    if phase == "train":
        p["cp"] = rng.choice([1, 1, 2, 4])
        p["cp_kind"] = rng.choice(["ring", "ulysses"])
        p["dp"] = rng.choice([1, 2, 4, 8])
        p["zero"] = rng.choice([0, 1, 2, 3])
        if p["zero"] == 3 and p["dp"] >= 4 and rng.random() < 0.5:
            p["shard"] = 2
        p["schedule"] = rng.choice(["gpipe", "1f1b", "zb", "interleaved"])
        p["virtual"] = 2 if p["schedule"] == "interleaved" else 1
        if md["moe"] and rng.random() < 0.5:
            p["ep"] = rng.choice([2, 4, 8])
        opts = dict(micro_batch=rng.choice([1, 2]), seqlen=rng.choice([2048, 8192, 32768]),
                    micro_batches=rng.choice([1, 4, 16]), optimizer=rng.random() < 0.7)
    else:
        if md["moe"] and rng.random() < 0.4:
            p = dict(ep=rng.choice([8, 16, 64]))
        if phase == "prefill" and rng.random() < 0.4:
            p["cp"] = rng.choice([2, 4, 8])
            p["cp_kind"] = rng.choice(["ring", "ulysses"])
        opts = dict(batch=rng.choice([1, 8, 64, 256]), context=rng.choice([1024, 8192, 32768]),
                    seqlen=rng.choice([2048, 8192]))
    opts["dtype"] = rng.choice(["bf16", "bf16", "fp8"])
    return dict(model=model, hw=rng.choice(list(data["hardware"])), phase=phase, plan=p, opts=opts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "data", "learn.json")))
    rng = random.Random(a.seed)
    cases = [draw(rng, data) for _ in range(a.cases)]
    hws = {k: scaling.hardware(k) for k in data["hardware"]}
    py = []
    for c in cases:
        e = scaling.estimate(c["model"], hws[c["hw"]], c["phase"], Plan(**c["plan"]), **c["opts"])
        py.append(e)
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "run.js")
        open(script, "w").write(NODE)
        cpath = os.path.join(d, "cases.json")
        json.dump(cases, open(cpath, "w"))
        r = subprocess.run(["node", script, os.path.join(HERE, "rsmath.js"), os.path.join(HERE, "data", "learn.json"), cpath],
                           capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        return 1
    js = json.loads(r.stdout)
    bad = 0

    def close(x, y):
        return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))

    for c, p, j in zip(cases, py, js):
        diffs = []
        for k, pk in (("step_us", "step_us"), ("mfu", "mfu"), ("mem", "memory_total_gb"), ("tok", "tok_s_gpu")):
            if not close(j[k], p[pk]):
                diffs.append(f"{k}: js {j[k]} py {p[pk]}")
        if j["label"] != p["label"] or j["dominant"] != p["dominant"] or j["fits"] != p["fits"]:
            diffs.append(f"label/dominant/fits: js {j['label']}/{j['dominant']}/{j['fits']} py {p['label']}/{p['dominant']}/{p['fits']}")
        pt = [(t["name"], t["us"]) for t in p["terms"]]
        if len(pt) != len(j["terms"]) or any(a[0] != b[0] or not close(a[1], b[1]) for a, b in zip(pt, j["terms"])):
            diffs.append(f"terms: js {j['terms']} py {pt}")
        if diffs:
            bad += 1
            if bad <= 5:
                print(json.dumps(c), *diffs, sep="\n  ")
    print(f"{len(cases) - bad}/{len(cases)} cases identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
