#!/usr/bin/env python3
"""Time the per-layer kernels of Mamba-2, attention and the shared GEMMs on a real GPU.

Every case is one kernel launch sequence from the open-source implementations:

* Mamba-2 (state-spaces/mamba): ``selective_state_update`` (decode),
  ``mamba_chunk_scan_combined`` (SSD prefill, 5 Triton kernels),
* causal-conv1d (Dao-AILab/causal-conv1d): ``causal_conv1d_update`` / ``causal_conv1d_fn``,
* attention: FlashAttention-2 (``flash_attn``) on sm_80, FlashAttention-4 (``flash_attn.cute``)
  on sm_100+, both prefill (causal) and decode against a KV cache,
* the projection GEMMs of both layer types (cuBLAS via ``torch.matmul``).

Output: one JSON with the median wall time per case (CUDA events). Each case's
measured call is wrapped in an NVTX range ``case_<id>`` so a single Nsight
Compute run with ``--nvtx --nvtx-include "regex:^case_.*/"`` profiles exactly those
kernels (see ``run_ncu.sh``).

Usage: python bench_kernels.py --out results.json [--quick] [--only REGEX]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time

import torch

DT = torch.bfloat16

# 8B-class layer shapes (d_model 4096), see configs.py for where they come from.
D_MODEL = 4096
# Mamba-2: expand 2, headdim 64, d_state 128, ngroups 8, d_conv 4, chunk 256.
M2 = dict(d_inner=8192, headdim=64, nheads=128, d_state=128, ngroups=8, d_conv=4, chunk=256)
# Attention: 32 query heads, 8 KV heads, head_dim 128 (GQA 4).
ATT = dict(hq=32, hkv=8, d=128)


def bench(fns, iters=20, warmup=3):
    """Median per-call GPU time, cold L2.

    ``fns`` are independent copies of the same call (own tensors) whose total
    footprint exceeds the L2, called round-robin, so every call streams its
    weights / state / KV cache from HBM as it would inside a full model. Calls
    are captured in a CUDA graph so host launch overhead (large for Triton and
    CuTe launches from Python) does not leak into the kernel time; falls back
    to eager timing."""
    reps = len(fns) * max(1, 10 // len(fns))

    def fn():
        for f in fns:
            f()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(reps // len(fns)):
                fn()
        graph.replay()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  (graph capture failed, eager timing: {e!r})"[:200], file=sys.stderr)
        graph = None
    times = []
    for _ in range(iters):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        if graph is not None:
            graph.replay()
        else:
            for _ in range(reps // len(fns)):
                fn()
        en.record()
        torch.cuda.synchronize()
        times.append(st.elapsed_time(en) * 1e3 / reps)  # us per call
    return statistics.median(times), min(times), graph is not None


def nvtx_once(name, fn):
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"case_{name}")
    fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


# ----------------------------------------------------------------------------- cases


def case_state_update(b, nh=None, g=None):
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    nh, hd, ds, g = nh or M2["nheads"], M2["headdim"], M2["d_state"], g or M2["ngroups"]
    dev = "cuda"
    state = torch.randn(b, nh, hd, ds, device=dev, dtype=DT)
    x = torch.randn(b, nh, hd, device=dev, dtype=DT)
    dt = torch.randn(b, nh, hd, device=dev, dtype=DT)
    A = -torch.rand(nh, hd, ds, device=dev, dtype=torch.float32)
    B = torch.randn(b, g, ds, device=dev, dtype=DT)
    C = torch.randn(b, g, ds, device=dev, dtype=DT)
    D = torch.randn(nh, hd, device=dev, dtype=torch.float32)
    z = torch.randn(b, nh, hd, device=dev, dtype=DT)
    dt_bias = torch.randn(nh, hd, device=dev, dtype=torch.float32)
    # Mamba2.step broadcasts per-head dt/A/D over headdim with stride-0 views.
    dt = dt[:, :, :1].expand(-1, -1, hd)
    A = A[:, :1, :1].expand(-1, hd, ds)
    D = D[:, :1].expand(-1, hd)
    dt_bias = dt_bias[:, :1].expand(-1, hd)

    def fn():
        selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    bytes_ = 2 * state.numel() * state.element_size()
    return fn, dict(kind="ssm_state_update", batch=b, nheads=nh, headdim=hd, d_state=ds, ngroups=g,
                    state_bytes=state.numel() * state.element_size(), min_bytes=bytes_)


def case_conv_update(b, dim=None):
    from causal_conv1d import causal_conv1d_update
    dim = dim or M2["d_inner"] + 2 * M2["ngroups"] * M2["d_state"]
    w = M2["d_conv"]
    x = torch.randn(b, dim, device="cuda", dtype=DT)
    st = torch.randn(b, dim, w - 1, device="cuda", dtype=DT)
    wt = torch.randn(dim, w, device="cuda", dtype=DT)
    bias = torch.randn(dim, device="cuda", dtype=DT)

    def fn():
        causal_conv1d_update(x, st, wt, bias, "silu")

    return fn, dict(kind="conv1d_update", batch=b, dim=dim, width=w)


def case_conv_fwd(b, L):
    from causal_conv1d import causal_conv1d_fn
    dim = M2["d_inner"] + 2 * M2["ngroups"] * M2["d_state"]
    w = M2["d_conv"]
    # Mamba2 passes xBC as (b, L, dim) transposed to channel-last (b, dim, L) view.
    x = torch.randn(b, L, dim, device="cuda", dtype=DT).transpose(1, 2)
    wt = torch.randn(dim, w, device="cuda", dtype=DT)
    bias = torch.randn(dim, device="cuda", dtype=DT)

    def fn():
        causal_conv1d_fn(x, wt, bias, activation="silu")

    return fn, dict(kind="conv1d_fwd", batch=b, seqlen=L, dim=dim, width=w)


def case_ssd(b, L):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    nh, hd, ds, g, cs = M2["nheads"], M2["headdim"], M2["d_state"], M2["ngroups"], M2["chunk"]
    dev = "cuda"
    x = torch.randn(b, L, nh, hd, device=dev, dtype=DT)
    dt = torch.randn(b, L, nh, device=dev, dtype=DT)
    A = -torch.rand(nh, device=dev, dtype=torch.float32)
    B = torch.randn(b, L, g, ds, device=dev, dtype=DT)
    C = torch.randn(b, L, g, ds, device=dev, dtype=DT)
    D = torch.randn(nh, device=dev, dtype=torch.float32)
    z = torch.randn(b, L, nh, hd, device=dev, dtype=DT)
    dt_bias = torch.randn(nh, device=dev, dtype=torch.float32)

    def fn():
        mamba_chunk_scan_combined(x, dt, A, B, C, cs, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    return fn, dict(kind="ssd_prefill", batch=b, seqlen=L, nheads=nh, headdim=hd, d_state=ds, ngroups=g, chunk=cs)


def _fa(cc):
    if cc[0] >= 10:
        from flash_attn.cute import flash_attn_func
        return "fa4", flash_attn_func
    from flash_attn import flash_attn_func
    return "fa2", flash_attn_func


def case_attn_prefill(b, L, cc):
    impl, f = _fa(cc)
    hq, hkv, d = ATT["hq"], ATT["hkv"], ATT["d"]
    q = torch.randn(b, L, hq, d, device="cuda", dtype=DT)
    k = torch.randn(b, L, hkv, d, device="cuda", dtype=DT)
    v = torch.randn(b, L, hkv, d, device="cuda", dtype=DT)

    def fn():
        f(q, k, v, causal=True)

    return fn, dict(kind="attn_prefill", impl=impl, batch=b, seqlen=L, hq=hq, hkv=hkv, d=d)


def case_attn_decode(b, ctx, cc, hq=None, hkv=None):
    hq, hkv, d = hq or ATT["hq"], hkv or ATT["hkv"], ATT["d"]
    q = torch.randn(b, 1, hq, d, device="cuda", dtype=DT)
    kc = torch.randn(b, ctx, hkv, d, device="cuda", dtype=DT)
    vc = torch.randn(b, ctx, hkv, d, device="cuda", dtype=DT)
    if cc[0] >= 10:
        from flash_attn.cute import flash_attn_func
        impl = "fa4"

        # num_splits=0 selects FA4's split-KV heuristic (its default, 1, runs
        # one CTA per KV head and leaves most SMs idle at small batch).
        def fn():
            flash_attn_func(q, kc, vc, causal=False, num_splits=0, pack_gqa=True)
    else:
        from flash_attn import flash_attn_with_kvcache
        impl = "fa2"
        cache_seqlens = torch.full((b,), ctx, device="cuda", dtype=torch.int32)

        def fn():
            flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cache_seqlens, causal=True)

    kv_bytes = 2 * kc.numel() * kc.element_size()
    return fn, dict(kind="attn_decode", impl=impl, batch=b, context=ctx, hq=hq, hkv=hkv, d=d, kv_bytes=kv_bytes)


def case_gemm(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=DT)
    w = torch.randn(n, k, device="cuda", dtype=DT)

    def fn():
        torch.matmul(a, w.t())

    return fn, dict(kind="gemm", m=m, n=n, k=k)


# ----------------------------------------------------------------------------- sweep


def cases_tp8_70b(cc, mem_gb):
    """Per-GPU shapes of the 70B-class models at TP=8 (what the NVL72 sweep simulates):
    Llama-3-70B attention 64/8 heads -> 8/1 per GPU; Mamba-2-70B 256 heads, 8 groups -> 32/1."""
    h, ffn, hd = 8192, 28672, 128
    nh, g, d_in = 32, 1, 2048
    conv = d_in + 2 * g * 128
    for b in [16, 64, 256]:
        yield f"tp8_ssm_state_update_b{b}", lambda b=b: case_state_update(b, nh=nh, g=g)
        yield f"tp8_conv1d_update_b{b}", lambda b=b: case_conv_update(b, dim=conv)
        for ctx in [4096, 16384, 65536]:
            if 2 * b * ctx * 1 * hd * 2 / 1e9 < 0.5 * mem_gb:
                yield f"tp8_attn_decode_b{b}_ctx{ctx}", lambda b=b, ctx=ctx: case_attn_decode(b, ctx, cc, hq=8, hkv=1)
        for (n, k, tag) in [(2 * d_in + 2 * g * 128 + nh, h, "mamba_in_proj"), (h, d_in, "mamba_out_proj"),
                            ((8 + 2) * hd, h, "attn_qkv"), (h, 8 * hd, "attn_o"),
                            (2 * ffn // 8, h, "mlp_gate_up"), (h, ffn // 8, "mlp_down")]:
            yield f"tp8_gemm_{tag}_m{b}", lambda b=b, n=n, k=k: case_gemm(b, n, k)


def cases(cc, quick, mem_gb):
    batches = [1, 8, 32, 128] if quick else [1, 8, 32, 64, 128, 256]
    ctxs = [1024, 4096, 16384] if quick else [1024, 4096, 16384, 65536]
    lens = [1024, 4096, 16384] if quick else [1024, 2048, 4096, 8192, 16384, 32768]
    ssm_in = M2["d_inner"] * 2 + 2 * M2["ngroups"] * M2["d_state"] + M2["nheads"]
    qkv = (ATT["hq"] + 2 * ATT["hkv"]) * ATT["d"]
    for b in batches:
        yield f"ssm_state_update_b{b}", lambda b=b: case_state_update(b)
        yield f"conv1d_update_b{b}", lambda b=b: case_conv_update(b)
        for ctx in ctxs:
            kv_gb = 2 * b * ctx * ATT["hkv"] * ATT["d"] * 2 / 1e9
            if kv_gb < 0.5 * mem_gb:
                yield f"attn_decode_b{b}_ctx{ctx}", lambda b=b, ctx=ctx: case_attn_decode(b, ctx, cc)
        for (n, k, tag) in [(ssm_in, D_MODEL, "mamba_in_proj"), (D_MODEL, M2["d_inner"], "mamba_out_proj"),
                            (qkv, D_MODEL, "attn_qkv"), (D_MODEL, D_MODEL, "attn_o"),
                            (2 * 14336, D_MODEL, "mlp_gate_up"), (D_MODEL, 14336, "mlp_down")]:
            yield f"gemm_{tag}_m{b}", lambda b=b, n=n, k=k: case_gemm(b, n, k)
    for L in lens:
        yield f"ssd_prefill_L{L}", lambda L=L: case_ssd(1, L)
        yield f"conv1d_fwd_L{L}", lambda L=L: case_conv_fwd(1, L)
        yield f"attn_prefill_L{L}", lambda L=L: case_attn_prefill(1, L, cc)
        for (n, k, tag) in [(ssm_in, D_MODEL, "mamba_in_proj"), (D_MODEL, M2["d_inner"], "mamba_out_proj"),
                            (qkv, D_MODEL, "attn_qkv")]:
            yield f"gemm_{tag}_m{L}", lambda L=L, n=n, k=k: case_gemm(L, n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--nvtx-only", action="store_true", help="one warm-up + one NVTX-wrapped call per case (for ncu)")
    ap.add_argument("--set", default="8b", choices=["8b", "tp8_70b"])
    a = ap.parse_args()
    cc = torch.cuda.get_device_capability()
    props = torch.cuda.get_device_properties(0)
    out = dict(device=props.name, cc=f"{cc[0]}.{cc[1]}", sms=props.multi_processor_count,
               torch=torch.__version__, cases={}, generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        import mamba_ssm, causal_conv1d
        out["mamba_ssm"], out["causal_conv1d"] = mamba_ssm.__version__, causal_conv1d.__version__
    except Exception as e:  # noqa: BLE001
        print("import:", e, file=sys.stderr)
    l2 = props.L2_cache_size
    out["l2_bytes"] = l2
    gen = cases(cc, a.quick, props.total_memory / 1e9) if a.set == "8b" else cases_tp8_70b(cc, props.total_memory / 1e9)
    for cid, mk in gen:
        if a.only and not re.search(a.only, cid):
            continue
        try:
            m0 = torch.cuda.memory_allocated()
            fn, meta = mk()
            footprint = torch.cuda.memory_allocated() - m0
            copies = 1 if a.nvtx_only else int(min(32, max(1, math.ceil(4 * l2 / max(footprint, 1)))))
            fns = [fn] + [mk()[0] for _ in range(copies - 1)]
            meta = dict(meta, footprint_bytes=footprint, copies=copies)
            if a.nvtx_only:
                fn()
                nvtx_once(cid, fn)
                rec = dict(meta)
            else:
                med, mn, graphed = bench(fns)
                rec = dict(meta, us_median=med, us_min=mn, graphed=graphed)
                nvtx_once(cid, fn)
            out["cases"][cid] = rec
            print(f"{cid:40s} {rec.get('us_median', float('nan')):10.2f} us", file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001
            out["cases"][cid] = dict(error=repr(e)[:400])
            print(f"{cid:40s} ERROR {e!r}"[:300], file=sys.stderr, flush=True)
        finally:
            fn = fns = None
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
