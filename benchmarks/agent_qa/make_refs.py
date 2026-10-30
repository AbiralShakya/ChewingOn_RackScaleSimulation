"""Compute the reference answers into refs.json (simulated ones take ~10 minutes).

    maturin develop --release
    python benchmarks/agent_qa/make_refs.py [question ids...]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from questions import HERE, QUESTIONS, ROOT  # noqa: E402

OUT = os.path.join(HERE, "refs.json")


def main(ids):
    refs = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            refs = json.load(f)["refs"]
    for q in QUESTIONS:
        if ids and q["id"] not in ids:
            continue
        t = time.time()
        r = q["ref"]()
        r["truth"] = q["truth"]
        refs[q["id"]] = r
        print(f"{q['id']:<18} {time.time() - t:6.1f}s  {json.dumps(r, default=str)}", flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(OUT, "w") as f:
        json.dump({"commit": commit, "refs": refs}, f, indent=1, default=str)


if __name__ == "__main__":
    main(set(sys.argv[1:]))
