#!/usr/bin/env python3
"""Run the model-size and architecture study (simulator only).

    python research/model_sweep/run.py                       # every study, results/<study>.json
    python research/model_sweep/run.py --study moe --jobs 4
    python research/model_sweep/run.py --quick               # smoke test of every grid

Each study is a grid from ``rackscale.sweep.study_points``; see README.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

import rackscale.sweep as sw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", action="append", choices=sw.STUDIES, help="default: all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    for st in a.study or sw.STUDIES:
        t0 = time.time()
        pts = sw.study_points(st, quick=a.quick)
        print(f"== {st}: {len(pts)} points", flush=True)
        res = sw.run(pts, jobs=a.jobs, progress=True)
        path = os.path.join(a.out, f"{st}{'_quick' if a.quick else ''}.json")
        sw.save(res, path, study=st, quick=a.quick, wall_s=time.time() - t0)
        print(f"wrote {path} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
