"""Decode steps of a CUDA-graph run, from an Nsight Systems sqlite export.

    nsys profile -t cuda --cuda-graph-trace=node -o run python vllm_decode.py --measure ...
    nsys export --type sqlite -o run.sqlite run.nsys-rep
    python nsys_graph_steps.py run.sqlite

With graph node tracing, every kernel a graph launch ran carries the launch's
correlation id. For each `cudaGraphLaunch` this reports the GPU span of its
kernels and their busy time, and the period to the next launch; the summary
is the median over the launches that ran the most common number of kernels
(the decode graph of the measured batch size).
"""
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict


def steps(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    launches = {}
    for corr, start, nid in c.execute("select correlationId, start, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        if names.get(nid, "").startswith("cudaGraphLaunch"):
            launches[corr] = start
    ks = defaultdict(list)
    for corr, a, b in c.execute("select correlationId, start, end from CUPTI_ACTIVITY_KIND_KERNEL"):
        if corr in launches:
            ks[corr].append((a, b))
    rows = []
    for corr in sorted(ks, key=lambda x: launches[x]):
        v = sorted(ks[corr])
        busy, end = 0, 0
        for a, b in v:  # union of kernel intervals
            busy += max(0, b - max(a, end))
            end = max(end, b)
        rows.append({"corr": corr, "kernels": len(v), "start": v[0][0], "span_us": (end - v[0][0]) / 1e3,
                     "busy_us": busy / 1e3})
    for r, n in zip(rows, rows[1:]):
        r["period_us"] = (n["start"] - r["start"]) / 1e3
    return rows


def main():
    rows = steps(sys.argv[1])
    common, _ = Counter(r["kernels"] for r in rows).most_common(1)[0]
    sel = [r for r in rows if r["kernels"] == common and "period_us" in r]
    out = {"graph_launches": len(rows), "kernels_per_step": common, "steps": len(sel),
           "span_us": statistics.median(r["span_us"] for r in sel),
           "busy_us": statistics.median(r["busy_us"] for r in sel),
           "period_us": statistics.median(r["period_us"] for r in sel)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
