"""Clock and power under sustained load: GEMMs by dtype and size, and attention.

    python power.py --gpus 0 --out power_gpu0.json          # one GPU, the others idle
    python power.py --gpus 0,1,2,3,4,5,6,7 --out power_all.json   # every GPU at once

For each case the kernel runs back to back for ``--seconds`` while NVML samples
SM clock, board power, temperature and throttle reasons every 10 ms. Only
samples after the first quarter of the window count (the clock settles in a
few hundred ms). TFLOP/s is measured over the same steady window with CUDA
events. This replaces the BF16 power numbers that were derived from
Nsight-vs-wall clocks, adds FP8, and is CALIBRATION.md microbenchmark 11
(clock under attention): FA4 and cuDNN attention with no ncu clock control.

With several GPUs, each GPU runs in its own process and all start together,
so the node's shared power and cooling are part of the measurement.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "research", "flash_attention", "remote"))

from cases import GEMM_SIZES  # noqa: E402

ATTN = dict(b=4, h=16, s=4096, d=128)  # the b300_fa4_s4096 reference shape


def make_case(kind: str, dtype: str, n: int, dev):
    """Returns (callable, FLOPs per call) or raises if the case is unsupported here."""
    import torch
    if kind == "gemm":
        a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        flops = 2.0 * n ** 3
        if dtype == "bf16":
            return (lambda: torch.mm(a, b)), flops
        if dtype == "fp8":
            a8 = a.to(torch.float8_e4m3fn)
            b8 = b.t().contiguous().t().to(torch.float8_e4m3fn)  # column-major B, as cuBLASLt wants
            one = torch.ones((), device=dev, dtype=torch.float32)
            return (lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)), flops
        raise ValueError(dtype)
    if kind == "attn":
        import fa_bench
        q = torch.randn(ATTN["b"], ATTN["s"], ATTN["h"], ATTN["d"], device=dev, dtype=torch.bfloat16)
        k, v = (torch.randn_like(q) for _ in range(2))
        f = fa_bench.make(dtype, q, k, v, False)
        f()
        return f, fa_bench.attn_flops(ATTN["b"], ATTN["h"], ATTN["s"], ATTN["s"], ATTN["d"], False)
    raise ValueError(kind)


def cases():
    out = [("gemm", d, n) for d in ("bf16", "fp8") for n in GEMM_SIZES]
    out += [("attn", impl, ATTN["s"]) for impl in ("fa4", "sdpa_cudnn", "fa2")]
    return out


def worker(gpu: int, seconds: float, start, q, csv: str):
    import torch
    from nvml import Sampler
    torch.cuda.set_device(gpu)
    dev = torch.device("cuda", gpu)
    sys.argv = sys.argv[:1]  # the CuTe DSL reads argv at JIT time
    results = []
    with Sampler([gpu], period=0.01, csv=csv) as s:
        for kind, dtype, n in cases():
            label = f"{kind}:{dtype}:{n}"
            try:
                f, flops = make_case(kind, dtype, n, dev)
                for _ in range(3):
                    f()
                torch.cuda.synchronize()
            except Exception as e:
                results.append(dict(case=label, error=f"{type(e).__name__}: {e}"[:300]))
                start.wait()
                continue
            # Calls per event window: about 50 ms of work.
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            f()
            e1.record()
            e1.synchronize()
            per = max(1, int(50.0 / max(1e-3, e0.elapsed_time(e1))))
            start.wait()  # every GPU starts each case together
            s.mark(label)
            t_end = time.time() + seconds
            windows = []
            while time.time() < t_end:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(per):
                    f()
                e1.record()
                windows.append((time.time(), e0, e1))
            torch.cuda.synchronize()
            s.mark("")
            t0 = windows[0][0]
            steady = [e0.elapsed_time(e1) for (t, e0, e1) in windows if t >= t0 + 0.25 * seconds] or \
                [e0.elapsed_time(e1) for (_, e0, e1) in windows]
            ms = sorted(steady)[len(steady) // 2] / per
            results.append(dict(case=label, kind=kind, dtype=dtype, n=n, time_us=ms * 1e3,
                                tflops=flops / (ms * 1e-3) * 1e-12, calls_per_window=per,
                                **s.summary(label, gpu)))
            torch.cuda.synchronize()
            time.sleep(2.0)  # let the clock and power recover before the next case
    q.put((gpu, results))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    ctx = mp.get_context("spawn")
    start, q = ctx.Barrier(len(gpus)), ctx.Queue()
    base = os.path.splitext(a.out)[0]
    procs = [ctx.Process(target=worker, args=(g, a.seconds, start, q, f"{base}.gpu{g}.csv")) for g in gpus]
    for p in procs:
        p.start()
    got = dict(q.get() for _ in procs)
    for p in procs:
        p.join()
    from nvml import limits
    doc = dict(gpus=gpus, concurrent=len(gpus) > 1, seconds=a.seconds, limits=limits(gpus[0]),
               results={str(g): got[g] for g in gpus})
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    for r in got[gpus[0]]:
        if "error" in r:
            print(f"  {r['case']:<22} skipped: {r['error'][:80]}")
        else:
            print(f"  {r['case']:<22} {r['tflops']:8.1f} TFLOP/s  {r.get('sm_mhz', 0):6.0f} MHz  "
                  f"{r.get('power_w', 0):6.0f} W  capped {r.get('power_capped_frac', 0):.0%}")


if __name__ == "__main__":
    main()
