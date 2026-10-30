"""H-Net against a BPE Transformer on one real GPU (book chapter 13 on hardware).

    python hnet_bench.py --hnet-repo ~/nodeday/ext/hnet --out hnet.json

Measures, on one GPU:

* prefill: 8 prompts of 8 KiB of English text (H-Net reads bytes, BPE reads
  ~1841 tokens), the simulator's committed ``prefill`` study;
* decode at batch 1: H-Net's own ``step`` (the released code only steps batch
  1) against the BPE model, eager and as a CUDA graph, 8 KiB of context;
* BPE decode at batch 16, 64 and 256 (the denominator of the chapter's batched
  ratios; H-Net's batched decode needs a batched step the repo does not have,
  so those ratios stay simulated);
* kernels launched per decoded byte, from torch.profiler (the chapter says
  328 for H-Net against 60 for BPE);
* how many sequences reach the main network per step (the chunking ratio on
  real text), from H-Net's routers.

The BPE model is the chapter's ``bpe-transformer-xl``: hidden 2048, 24 layers,
16 heads of 128, SwiGLU 5376, vocabulary 128256, random weights (timing does not
depend on them). H-Net uses the released XL checkpoints, because its routers
need trained weights to chunk text the way it does in the paper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

BYTES_PER_TOKEN = 4.45  # rackscale.hnet.BYTES_PER_BPE_TOKEN
CTX = 8192


# ---------------------------------------------------------------------------
# text


def english(nbytes: int, n: int) -> list:
    """n strings of exactly nbytes UTF-8 bytes of English web text."""
    out = []
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        for row in ds:
            b = row["text"].encode()
            if len(b) >= nbytes:
                out.append(b[:nbytes])
            if len(out) == n:
                return out
    except Exception as e:  # offline node: fall back to the Python docs that ship with the interpreter
        print(f"  fineweb-edu unavailable ({type(e).__name__}); using stdlib docstrings", flush=True)
    import pydoc
    import inspect
    import collections
    import json as _json
    import textwrap
    text = "\n".join(inspect.getdoc(m) or "" for m in (pydoc, collections, _json, textwrap, inspect, os, math)).encode()
    while len(out) < n:
        k = len(out) * 997 % max(1, len(text) - nbytes)
        out.append((text * (1 + nbytes // len(text) + 1))[k:k + nbytes])
    return out


# ---------------------------------------------------------------------------
# the BPE baseline


class BPE(torch.nn.Module):
    def __init__(self, d=2048, layers=24, heads=16, ffn=5376, vocab=128256):
        super().__init__()
        self.d, self.h, self.hd, self.L = d, heads, d // heads, layers
        self.emb = torch.nn.Embedding(vocab, d)
        self.qkv = torch.nn.ModuleList(torch.nn.Linear(d, 3 * d, bias=False) for _ in range(layers))
        self.o = torch.nn.ModuleList(torch.nn.Linear(d, d, bias=False) for _ in range(layers))
        self.gu = torch.nn.ModuleList(torch.nn.Linear(d, 2 * ffn, bias=False) for _ in range(layers))
        self.dn = torch.nn.ModuleList(torch.nn.Linear(ffn, d, bias=False) for _ in range(layers))
        self.n1 = torch.nn.ParameterList(torch.nn.Parameter(torch.ones(d)) for _ in range(layers))
        self.n2 = torch.nn.ParameterList(torch.nn.Parameter(torch.ones(d)) for _ in range(layers))
        self.nf = torch.nn.Parameter(torch.ones(d))
        self.head = torch.nn.Linear(d, vocab, bias=False)

    def alloc(self, b, maxlen, dev):
        shape = (self.L, b, maxlen, self.h, self.hd)
        self.k = torch.zeros(shape, device=dev, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)

    def forward(self, ids, pos0: int = 0, seqlens=None):
        """ids (B, T). Prefill when T > 1 (fills the cache from pos0); one decode step when T == 1."""
        x = self.emb(ids)
        b, t, _ = x.shape
        for i in range(self.L):
            h = F.rms_norm(x, (self.d,), self.n1[i])
            q, k, v = self.qkv[i](h).view(b, t, 3, self.h, self.hd).unbind(2)
            if t > 1:
                self.k[i, :b, pos0:pos0 + t], self.v[i, :b, pos0:pos0 + t] = k, v
                a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                   is_causal=True).transpose(1, 2)
            else:
                from flash_attn import flash_attn_with_kvcache
                a = flash_attn_with_kvcache(q, self.k[i, :b], self.v[i, :b], k, v, cache_seqlens=seqlens)
            x = x + self.o[i](a.reshape(b, t, self.d))
            g, u = self.gu[i](F.rms_norm(x, (self.d,), self.n2[i])).chunk(2, -1)
            x = x + self.dn[i](F.silu(g) * u)
        return self.head(F.rms_norm(x[:, -1:], (self.d,), self.nf))


def bpe_runs(dev, prompts, steps: int, batches) -> dict:
    torch.manual_seed(0)
    m = BPE().to(dev, torch.bfloat16).eval()
    t = int(CTX / BYTES_PER_TOKEN)
    out = {}
    with torch.inference_mode():
        # prefill: 8 prompts of t tokens
        m.alloc(8, t + 8, dev)
        ids = torch.randint(0, 128256, (8, t), device=dev)
        out["prefill_us"] = _median_us(lambda: m(ids), 5)
        for b in batches:
            m.alloc(b, t + steps + 8, dev)
            ids = torch.randint(0, 128256, (b, t), device=dev)
            m(ids)
            tok = torch.randint(0, 128256, (b, 1), device=dev)
            lens = torch.full((b,), t, device=dev, dtype=torch.int32)
            eager = _median_us(lambda: m(tok, seqlens=lens), steps)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                m(tok, seqlens=lens)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                m(tok, seqlens=lens)
            graph = _median_us(g.replay, steps)
            row = dict(batch=b, step_us_eager=eager, step_us_graph=graph,
                       us_per_byte_eager=eager / BYTES_PER_TOKEN, us_per_byte_graph=graph / BYTES_PER_TOKEN)
            if b == batches[0]:
                row["kernels_per_step"] = _kernels(lambda: m(tok, seqlens=lens), 8)
            out[str(b)] = row
            print(f"  bpe b{b}: step {eager:.0f} us eager, {graph:.0f} us graph", flush=True)
    return out


# ---------------------------------------------------------------------------
# H-Net


def hnet_runs(dev, repo: str, name: str, prompts, steps: int) -> dict:
    sys.path.insert(0, repo)
    from generate import load_from_pretrained
    from huggingface_hub import hf_hub_download
    from hnet.modules.dc import ChunkLayer
    ckpt = hf_hub_download(f"cartesia-ai/{name}", f"{name}.pt")
    model = load_from_pretrained(ckpt, os.path.join(repo, "configs", f"{name}.json"))
    out = {}
    rows_per_step = []
    orig = ChunkLayer.step

    def counting_step(self, hidden_states, boundary_mask):
        rows_per_step.append(int(boundary_mask.sum()))
        return orig(self, hidden_states, boundary_mask)

    with torch.inference_mode():
        ids = torch.tensor([[254] + list(p[:CTX - 1]) for p in prompts[:8]], device=dev)  # 254 = BOS
        mask = torch.ones(ids.shape, device=dev, dtype=torch.bool)
        out["prefill_us"] = _median_us(lambda: model.forward(ids, mask=mask), 5)
        print(f"  {name} prefill 8x8KiB: {out['prefill_us']:.0f} us", flush=True)
        # decode at batch 1, greedy, on real text so the routers chunk as trained
        cache = model.allocate_inference_cache(1, CTX + steps + 8, dtype=torch.bfloat16)
        x = ids[:1]
        o = model.forward(x, mask=mask[:1], inference_params=cache)
        tok = o.logits[:, -1:].argmax(-1)
        ChunkLayer.step = counting_step
        ts = []
        try:
            for _ in range(steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                o = model.step(tok, cache)
                tok = o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e6)
        finally:
            ChunkLayer.step = orig
        ts = ts[steps // 8:]
        out["decode_b1_us_per_byte"] = statistics.median(ts)
        out["decode_b1_us_mean"] = statistics.fmean(ts)
        # Fraction of steps that reach each inner stage (one ChunkLayer.step per stage per step).
        n_stage = max(1, len(rows_per_step) // steps)
        out["main_net_rate_per_stage"] = [statistics.fmean(rows_per_step[i::n_stage]) for i in range(n_stage)]
        out["kernels_per_byte"] = _kernels(lambda: model.step(tok, cache), 16)
        print(f"  {name} decode b1: {out['decode_b1_us_per_byte']:.0f} us/byte, "
              f"{out['kernels_per_byte']} kernels/byte", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------


def _median_us(f, n: int) -> float:
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        f()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)


def _kernels(f, n: int) -> float:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            f()
        torch.cuda.synchronize()
    k = sum(e.count for e in prof.key_averages() if (getattr(e, "self_device_time_total", None)
                                                     or getattr(e, "self_cuda_time_total", 0)) > 0)
    return round(k / n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hnet-repo", required=True, help="a clone of github.com/goombalab/hnet")
    ap.add_argument("--models", default="hnet_1stage_XL,hnet_2stage_XL")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--bpe-batches", default="1,16,64,256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    prompts = english(CTX, 8)
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, context_bytes=CTX)
    try:
        res["bpe"] = bpe_runs(dev, prompts, a.steps // 2, [int(b) for b in a.bpe_batches.split(",")])
    except Exception as e:
        res["bpe"] = dict(error=f"{type(e).__name__}: {e}"[:400])
        print(f"  bpe failed: {res['bpe']['error']}", flush=True)
    for name in a.models.split(","):
        try:
            res[name] = hnet_runs(dev, os.path.expanduser(a.hnet_repo), name, prompts, a.steps)
        except Exception as e:
            res[name] = dict(error=f"{type(e).__name__}: {e}"[:400])
            print(f"  {name} failed: {res[name]['error']}", flush=True)
    b1 = res.get("bpe", {}).get("1", {})
    for name in a.models.split(","):
        h = res.get(name, {})
        if "decode_b1_us_per_byte" in h and b1:
            h["ratio_vs_bpe_graph"] = h["decode_b1_us_per_byte"] / b1["us_per_byte_graph"]
            h["ratio_vs_bpe_eager"] = h["decode_b1_us_per_byte"] / b1["us_per_byte_eager"]
            print(f"  {name} / BPE at batch 1: {h['ratio_vs_bpe_graph']:.2f}x (BPE graphed), "
                  f"{h['ratio_vs_bpe_eager']:.2f}x (BPE eager)")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
