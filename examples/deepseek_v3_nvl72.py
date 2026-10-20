"""DeepSeek-V3 decode on a GB200 NVL72 with expert parallelism over all 72 GPUs.

Compares one-stream execution with DeepSeek-style two-micro-batch overlap
and prints the bubble anatomy. ``--layers`` keeps the run short: the first 3
layers are dense, so use >= 4 to include MoE (all-to-all dispatch/combine).

    python examples/deepseek_v3_nvl72.py --layers 4 --trace out/ds.json
"""

import argparse

import rackscale as rs
from rackscale import lower, models

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="gb200_nvl72")
ap.add_argument("--batch", type=int, default=64, help="sequences per GPU")
ap.add_argument("--context", type=int, default=4096)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--exact", action="store_true", help="co-simulate every rank at cycle level (slow)")
ap.add_argument("--trace")
a = ap.parse_args()

par = rs.Parallel(ep=72, comm_sms=20)
for overlap in (False, True):
    wl = lower.decode_step(models.DEEPSEEK_V3, par, a.batch, a.context, overlap=overlap, layers=a.layers)
    rep = rs.simulate(a.system, wl, fast=not a.exact, trace=a.trace if overlap else None)
    print(rep.summary(), "\n")
