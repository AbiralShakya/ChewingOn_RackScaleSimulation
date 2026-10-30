"""A GEMM while NCCL holds R SMs: chapter 15's reserved-SM step, measured.

    NCCL_MIN_CTAS=24 NCCL_MAX_CTAS=24 torchrun --nproc-per-node 8 overlap.py --out overlap_R24.json
    torchrun --nproc-per-node 8 overlap.py --alone --out overlap_R0.json

Each rank keeps a stream of 256 MiB all-reduces running on a side stream (so
NCCL's R CTAs sit on R SMs the whole time) and times BF16 4096³ GEMMs on the
main stream. The simulator predicts no cost up to 16 reserved SMs and +34% at
24 on a 148-SM Blackwell (⌈512 tiles / free SMs⌉ goes from 4 to 5 waves) if
the GEMM uses 128×256 tiles; the profiler records which cuBLAS kernel ran,
since its tile decides the wave count. Hardware also shares L2 and HBM with
the all-reduce, which the reserved-SM simulation does not; the GEMM is
compute-bound, so that should be small, and the R = 2 point measures it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alone", action="store_true", help="no all-reduce beside the GEMM")
    ap.add_argument("--gemms", type=int, default=200)
    ap.add_argument("--ar-mib", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank = dist.get_rank()
    dev = torch.device("cuda", local)
    x, y = (torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16) for _ in range(2))
    buf = torch.zeros(a.ar_mib << 19, device=dev, dtype=torch.bfloat16)
    side = torch.cuda.Stream()

    def all_reduces(n):
        with torch.cuda.stream(side):
            return [dist.all_reduce(buf, async_op=True) for _ in range(n)]

    for _ in range(10):
        torch.mm(x, y)
    all_reduces(2)
    torch.cuda.synchronize()
    # One all-reduce alone, for the comm time per call.
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(side):
        e0.record(side)
        dist.all_reduce(buf)
        e1.record(side)
    torch.cuda.synchronize()
    ar_us = e0.elapsed_time(e1) * 1e3

    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.gemms)]
    dist.barrier()
    # Enough all-reduces queued to outlast the GEMMs (they are long; the GEMMs are ~100 us each).
    n_ar = 0 if a.alone else max(4, int(a.gemms * 150 / max(ar_us, 1.0)) + 4)
    works = all_reduces(n_ar) if n_ar else []
    for e0, e1 in ev:
        e0.record()
        torch.mm(x, y)
        e1.record()
    torch.cuda.synchronize()
    for w in works:
        w.wait()
    t = [e0.elapsed_time(e1) * 1e3 for e0, e1 in ev]
    t_mid = sorted(t[a.gemms // 4: 3 * a.gemms // 4])
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.mm(x, y)
        torch.cuda.synchronize()
    kernel = [e.key for e in prof.key_averages() if "gemm" in e.key.lower() or "cutlass" in e.key.lower()
              or "nvjet" in e.key.lower()]
    res = dict(rank=rank, gemm_us=statistics.median(t_mid), gemm_us_all=t, all_reduce_us=ar_us, all_reduces=n_ar,
               kernel=kernel[:3], nccl_min_ctas=os.environ.get("NCCL_MIN_CTAS"),
               nccl_max_ctas=os.environ.get("NCCL_MAX_CTAS"), alone=a.alone, sms=torch.cuda.get_device_properties(
                   dev).multi_processor_count)
    out = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(res, out, dst=0)
    if rank == 0:
        med = statistics.median(r["gemm_us"] for r in out)
        doc = dict(gemm_us=med, ranks=out, reserve=0 if a.alone else int(os.environ.get("NCCL_MAX_CTAS", "0") or 0))
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"  R={doc['reserve']:<3} GEMM {med:8.1f} us  all-reduce {ar_us:8.1f} us  kernel {out[0]['kernel'][:1]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
