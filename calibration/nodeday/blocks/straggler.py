"""The straggler tax, measured: what synchronising N GPUs after every kernel costs.

    torchrun --nproc-per-node 8 straggler.py --out straggler.json

Every rank runs the same BF16 4096³ GEMM ``--iters`` times in three ways:

* ``free``: no synchronisation, each rank timing its own kernels. Gives each
  GPU's per-kernel distribution (its mean and σ).
* ``sync1``: a 4-byte all-reduce after every GEMM, so every iteration waits
  for the slowest rank (chapter 16's "synchronise after every GEMM").
* ``sync16``: the all-reduce after every 16 GEMMs.

The tax is (synchronised time per GEMM) / (mean free time per GEMM) − 1, at
N = 2, 4 and 8 (sub-groups of the first N ranks). ``compare.py`` sets it
beside chapter 16's closed form σ·E[max_N] evaluated with the measured σ,
and beside an empirical bootstrap (the mean of the max of N draws from the
free samples), which needs no normality assumption.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", local)
    x, y = (torch.randn(a.n, a.n, device=dev, dtype=torch.bfloat16) for _ in range(2))
    flag = torch.zeros(1, device=dev)
    groups = {n: dist.new_group(list(range(n))) for n in (2, 4, 8) if n <= world}

    for _ in range(20):
        torch.mm(x, y)
    torch.cuda.synchronize()

    # free: per-kernel times on this rank
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.iters)]
    dist.barrier()
    for e0, e1 in ev:
        e0.record()
        torch.mm(x, y)
        e1.record()
    torch.cuda.synchronize()
    free = [e0.elapsed_time(e1) * 1e3 for e0, e1 in ev]

    res = dict(rank=rank, free_us=free, free_mean_us=statistics.fmean(free),
               free_cv=statistics.pstdev(free) / statistics.fmean(free), sync={})
    for n, g in groups.items():
        for every in (1, 16):
            dist.barrier()
            if rank < n:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(a.iters):
                    torch.mm(x, y)
                    if (i + 1) % every == 0:
                        dist.all_reduce(flag, group=g)
                torch.cuda.synchronize()
                res["sync"][f"{n}:{every}"] = (time.perf_counter() - t0) * 1e6 / a.iters
    out = [None] * world if rank == 0 else None
    dist.gather_object(res, out, dst=0)
    if rank == 0:
        means = [r["free_mean_us"] for r in out]
        summary = {}
        for key in out[0]["sync"]:
            n = int(key.split(":")[0])
            free_n = statistics.fmean(means[:n])
            t = max(r["sync"][key] for r in out[:n])
            summary[key] = dict(ranks=n, every=int(key.split(":")[1]), per_gemm_us=t, free_mean_us=free_n,
                                tax=t / free_n - 1)
            print(f"  N={n} sync every {summary[key]['every']:>2}: {t:8.1f} us/GEMM vs {free_n:8.1f} free "
                  f"-> tax {summary[key]['tax']:+.2%}")
        with open(a.out, "w") as f:
            json.dump(dict(n=a.n, iters=a.iters, ranks=out, summary=summary), f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
