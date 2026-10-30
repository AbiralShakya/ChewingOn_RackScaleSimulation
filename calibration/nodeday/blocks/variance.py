"""Run-to-run and GPU-to-GPU variance of the same kernels on every GPU of the node.

    python variance.py --mode isolated   --out var_isolated.json    # one GPU at a time
    python variance.py --mode concurrent --out var_concurrent.json  # all GPUs at once

Each GPU runs the same four workloads, ``--reps`` timed repetitions each:

* ``gemm8k``   BF16 8192³: tensor-bound and power-capped on Blackwell
* ``gemm4k``   BF16 4096³ x 8 per repetition: tensor-bound, below the cap
* ``copy``     a 2 GiB device-to-device copy: HBM-bound
* ``small``    1000 tiny elementwise kernels back to back: launch- and latency-bound

The output gives per-GPU mean, median, σ/mean and p99/p50, the per-GPU median
clock and power, and the spread of per-GPU means across the node. Chapter 16's
straggler term assumes 5% per-kernel jitter; this is what replaces it.
``isolated`` separates silicon from the node's shared power and cooling;
``concurrent`` includes them. Running ``isolated`` again at the end of the day
gives the drift over the day.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

WORKLOADS = ("gemm8k", "gemm4k", "copy", "small")


def run_gpu(gpu: int, reps: int, start=None) -> dict:
    import torch
    from nvml import Sampler
    torch.cuda.set_device(gpu)
    dev = torch.device("cuda", gpu)
    a8, b8 = (torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16) for _ in range(2))
    a4, b4 = (torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16) for _ in range(2))
    src = torch.empty(1 << 30, device=dev, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    tiny = torch.zeros(1024, device=dev)

    def gemm8k():
        torch.mm(a8, b8)

    def gemm4k():
        for _ in range(8):
            torch.mm(a4, b4)

    def copy():
        dst.copy_(src)

    def small():
        for _ in range(1000):
            tiny.add_(1.0)

    fns = dict(gemm8k=gemm8k, gemm4k=gemm4k, copy=copy, small=small)
    out = dict(gpu=gpu, name=torch.cuda.get_device_name(gpu),
               uuid=str(torch.cuda.get_device_properties(gpu).uuid) if hasattr(
                   torch.cuda.get_device_properties(gpu), "uuid") else None)
    with Sampler([gpu], period=0.02) as s:
        for w in WORKLOADS:
            f = fns[w]
            for _ in range(5):
                f()
            torch.cuda.synchronize()
            if start is not None:
                start.wait()
            s.mark(w)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
            for e0, e1 in ev:
                e0.record()
                f()
                e1.record()
            torch.cuda.synchronize()
            s.mark("")
            t = [e0.elapsed_time(e1) * 1e3 for e0, e1 in ev]
            ts = sorted(t)
            mean = statistics.fmean(t)
            out[w] = dict(us=t, mean_us=mean, median_us=ts[len(ts) // 2], cv=statistics.pstdev(t) / mean,
                          p99_over_p50=ts[min(len(ts) - 1, int(0.99 * len(ts)))] / ts[len(ts) // 2],
                          **s.summary(w, gpu, settle=0.0))
    return out


def _worker(gpu, reps, start, q):
    q.put(run_gpu(gpu, reps, start))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("isolated", "concurrent"), required=True)
    ap.add_argument("--gpus", default="")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else list(range(torch.cuda.device_count()))
    if a.mode == "isolated":
        per = []
        for g in gpus:
            per.append(run_gpu(g, a.reps))
            print(f"  gpu {g}: " + "  ".join(f"{w} {per[-1][w]['median_us']:.1f}us cv {per[-1][w]['cv']:.2%}"
                                            for w in WORKLOADS), flush=True)
    else:
        ctx = mp.get_context("spawn")
        start, q = ctx.Barrier(len(gpus)), ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(g, a.reps, start, q)) for g in gpus]
        for p in procs:
            p.start()
        per = sorted((q.get() for _ in procs), key=lambda r: r["gpu"])
        for p in procs:
            p.join()
    node = {}
    for w in WORKLOADS:
        means = [r[w]["mean_us"] for r in per]
        m = statistics.fmean(means)
        node[w] = dict(mean_us=m, gpu_spread=(max(means) - min(means)) / m, gpu_cv=statistics.pstdev(means) / m,
                       slowest_gpu=per[means.index(max(means))]["gpu"],
                       slowest_over_mean=max(means) / m,
                       within_gpu_cv_median=statistics.median(r[w]["cv"] for r in per))
        print(f"  node {w:<7} spread {node[w]['gpu_spread']:.2%}  gpu cv {node[w]['gpu_cv']:.2%}  "
              f"slowest gpu {node[w]['slowest_gpu']} (+{node[w]['slowest_over_mean'] - 1:.2%})  "
              f"within-gpu cv {node[w]['within_gpu_cv_median']:.2%}")
    with open(a.out, "w") as f:
        json.dump(dict(mode=a.mode, reps=a.reps, gpus=per, node=node), f, indent=1)


if __name__ == "__main__":
    main()
