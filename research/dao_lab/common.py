"""Shared helpers for the Dao Lab architecture studies."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(ROOT, "python"))
os.makedirs(RESULTS, exist_ok=True)
os.environ.setdefault("RACKSCALE_EXTRAP_CACHE", os.path.join(RESULTS, "kernel_cache.json"))

#: One calibrated B300 (the HGX B300 preset, rank 0) for kernel-level points.
GPU_SYSTEM = "hgx_b300_8"


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else []


def save(name, rows):
    p = os.path.join(RESULTS, name)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(tmp, p)


def kernels_of(wl, types=None):
    """Kernel ops of rank 0, stream 0 of an SPMD workload."""
    ops = wl["spmd"]["program"]["streams"][0]
    return [o for o in ops if o.get("op") == "kernel" and (types is None or o["kernel"]["type"] in types)]
