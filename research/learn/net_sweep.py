"""Collective timings for the learning page's collectives panel.

    python research/learn/net_sweep.py      # -> research/learn/data/net_sweep.json

nccl-tests-style sweeps (all ranks post at t = 0) on the GB300 NVL72 preset
with ``rsim net sweep``: all-reduce, all-gather, reduce-scatter and
all-to-all over 2, 8 and 72 GPUs with NCCL's own algorithm choice, ring and
NVLS all-reduces forced, and two-GPU send/recv, from 1 KiB to 1 GiB. Needs
the CLI (``cargo build --release -p rsim-cli``).
"""

from __future__ import annotations

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSIM = os.path.join(ROOT, "target", "release", "rsim")

SWEEPS = [(c, n, "auto") for c in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all") for n in (2, 8, 72)]
SWEEPS += [("all_reduce", 2, "ring"), ("all_reduce", 8, "ring"), ("all_reduce", 72, "ring"),
           ("all_reduce", 8, "nvls"), ("all_reduce", 72, "nvls"), ("send_recv", 2, "auto")]


def main() -> None:
    rows = []
    for coll, ranks, algo in SWEEPS:
        out = subprocess.run([RSIM, "net", "sweep", "--system", "gb300_nvl72", "--coll", coll, "--ranks", str(ranks),
                              "--algo", algo, "-b", "1K", "-e", "1G", "-f", "4", "--json"],
                             check=True, capture_output=True, text=True).stdout
        for r in json.loads(out):
            us = (r["done_ps"] - r["first_post_ps"]) / 1e6
            rows.append(dict(coll=coll, ranks=ranks, algo_req=algo, algo=r["algo"], proto=r["proto"],
                             channels=r["channels"], bytes=r["bytes"], us=us, busbw=r["busbw_gbps"],
                             nccl_model_us=r["nccl_model_us"]))
        print(coll, ranks, algo, len(rows), flush=True)
    with open(os.path.join(HERE, "data", "net_sweep.json"), "w") as f:
        json.dump(rows, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
