#!/usr/bin/env python3
"""Run one attention forward implementation on the local GPU.

  python fa_bench.py --impl fa4 --b 4 --h 16 --s 4096 --d 128            # timing (median of --iters)
  python fa_bench.py --impl fa2 ... --once                                # one call (for ncu / NVBit)

Implementations
  naive         softmax(QK^T)V written out in PyTorch: two GEMMs and a softmax, S materialised
  sdpa_math     torch SDPA, math backend (same as naive, fused by PyTorch's decomposition)
  sdpa_flash    torch SDPA, PyTorch's bundled FlashAttention-2 backend
  sdpa_cudnn    torch SDPA, cuDNN backend
  fa2           Dao-AILab flash_attn (FlashAttention-2, csrc/flash_attn), mma.sync on every arch
  fa3           Dao-AILab flash_attn_interface (FlashAttention-3, hopper/), sm_90 only
  fa4           Dao-AILab flash_attn.cute (FlashAttention-4, CuTe DSL), sm_90 / sm_100 / sm_103

Prints one JSON line: implementation, shape, median time, TFLOP/s.
"""
import argparse
import json
import math
import sys

import torch
import torch.nn.functional as F


def attn_flops(b, h, s_q, s_k, d, causal):
    f = 4.0 * b * h * s_q * s_k * d
    return f / 2 if causal else f


def make(impl, q, k, v, causal):
    # q, k, v: (b, s, h, d)
    scale = 1.0 / math.sqrt(q.shape[-1])
    if impl == "naive":
        def f():
            qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
            s = torch.matmul(qt, kt.transpose(-1, -2)) * scale
            if causal:
                n = s.shape[-1]
                s = s.masked_fill(torch.ones(n, n, device=s.device, dtype=torch.bool).triu(1), float("-inf"))
            p = torch.softmax(s.float(), dim=-1).to(q.dtype)
            return torch.matmul(p, vt).transpose(1, 2)
        return f
    if impl.startswith("sdpa"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backend = {"sdpa_math": SDPBackend.MATH, "sdpa_flash": SDPBackend.FLASH_ATTENTION,
                   "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION, "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION}[impl]
        qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))

        def f():
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
        return f
    if impl == "fa2":
        from flash_attn import flash_attn_func
        return lambda: flash_attn_func(q, k, v, causal=causal)
    if impl == "fa3":
        from flash_attn_interface import flash_attn_func
        return lambda: flash_attn_func(q, k, v, causal=causal)
    if impl == "fa4":
        from flash_attn.cute import flash_attn_func
        return lambda: flash_attn_func(q, k, v, causal=causal)
    raise SystemExit(f"unknown impl {impl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--hkv", type=int, default=0, help="KV heads (GQA); 0 = same as --h")
    ap.add_argument("--s", type=int, default=4096)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--once", action="store_true", help="warm up (JIT), then exactly one timed call")
    ap.add_argument("--check", action="store_true", help="compare against the math backend")
    a = ap.parse_args()
    # The CuTe DSL parses sys.argv for its own diagnostic flags at JIT time.
    sys.argv = sys.argv[:1]

    torch.manual_seed(0)
    dev = "cuda"
    hkv = a.hkv or a.h
    q = torch.randn(a.b, a.s, a.h, a.d, device=dev, dtype=torch.bfloat16)
    k, v = (torch.randn(a.b, a.s, hkv, a.d, device=dev, dtype=torch.bfloat16) for _ in range(2))
    f = make(a.impl, q, k, v, a.causal)
    out = f()
    out = out[0] if isinstance(out, tuple) else out
    torch.cuda.synchronize()
    err = None
    if a.check:
        ref = make("sdpa_math", q, k, v, a.causal)()
        err = (out.float() - ref.float()).abs().max().item()
    if a.once:
        # The JIT/warm-up call above is launch 0 of the kernel; this is launch 1.
        # Only this call is inside the profiler range (ncu --profile-from-start off).
        torch.cuda.cudart().cudaProfilerStart()
        f()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(json.dumps({"impl": a.impl, "once": True}))
        return
    for _ in range(a.warmup):
        f()
    torch.cuda.synchronize()
    # Back-to-back calls between one pair of events, so host-side wrapper overhead is
    # hidden behind the GPU work (a single call per event pair would charge it to the kernel).
    reps, per = 7, max(1, a.iters // 7)
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(per):
            f()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / per)
    ts.sort()
    med = ts[len(ts) // 2]
    fl = attn_flops(a.b, a.h, a.s, a.s, a.d, a.causal)
    print(json.dumps({
        "impl": a.impl, "gpu": torch.cuda.get_device_name(), "cc": "%d.%d" % torch.cuda.get_device_capability(),
        "b": a.b, "h": a.h, "hkv": hkv, "s": a.s, "d": a.d, "causal": a.causal, "us_median": med, "us_min": ts[0],
        "tflops": fl / med / 1e6, "max_abs_err_vs_math": err,
    }))


if __name__ == "__main__":
    sys.exit(main())
