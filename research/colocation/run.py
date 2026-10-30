"""Several models on one GB300 NVL72 at once.

    python research/colocation/run.py          # all scenarios -> results/colocation.json, .md

Scenarios (each tenant cut to a few layers; every tenant loops several steps so they overlap):
  partition   DeepSeek-V3 EP64 decode beside Llama-3-8B TP8 decode on the other 8 GPUs
  mixed       a 70B TP8 trainer (FSDP x2), a Qwen3-30B-A3B TP4 server and four 1B servers sharing one rack
  disagg      KV caches streaming from a prefill pool into a Llama-3-70B TP8 decode pool, 2 to 100 GB/s per GPU
  shared      two models on the SAME 4 GPUs (MPS-style), fast mode vs exact (cycle-level co-run)
"""

from __future__ import annotations

import argparse
import json
import os
import time

from rackscale import colocate as co
from rackscale.scaling import Plan

HERE = os.path.dirname(os.path.abspath(__file__))


def scenarios():
    yield "partition", dict(tenants=[
        co.Tenant("deepseek", "deepseek-v3", "decode", Plan(ep=64), gpus=range(64), batch=64, context=4096, steps=3),
        co.Tenant("llama8b", "llama3-8b", "decode", Plan(tp=8), gpus=range(64, 72), batch=64, context=8192, steps=3)],
        layers=4)
    yield "mixed", dict(tenants=[
        co.Tenant("trainer", "llama3-70b", "train", Plan(tp=8, dp=2, zero=3), gpus=range(16), seqlen=4096, steps=1),
        co.Tenant("moe", "qwen3-30b-a3b", "decode", Plan(tp=4), gpus=range(16, 20), batch=128, context=8192, steps=4),
        *[co.Tenant(f"1b-{i}", "llama3.2-1b", "decode", Plan(), gpus=[20 + i], batch=64, context=4096, steps=6)
          for i in range(4)]], layers=2)
    # Disaggregated serving: KV caches stream from a prefill pool (GPUs 0-7)
    # into a TP8 decode pool (GPUs 8-15), one 4 MiB chunk (one layer's KV of
    # an 8k prompt, per GPU) at a time over NCCL send/recv on 4 SMs, issued
    # on a clock that sets the rate per decode GPU. A 1:1 pool of Llama-3-70B
    # produces ~2 GB/s per decode GPU (8k-token prompts, 320 KB of KV per
    # token, ~0.2 s per prefill); the sweep goes far past that to find where
    # the KV stream starts to hurt decode. 4 MiB send/recv tops out near
    # 145 GB/s in the network model.
    decode_us = 8 * 362.0
    for gbps in (2, 10, 25, 50, 100):
        chunk = 4 << 20
        every = chunk / (gbps * 1e9) * 1e9
        yield f"disagg {gbps} GB/s", dict(tenants=[
            co.Tenant("decode", "llama3-70b", "decode", Plan(tp=8), gpus=range(8, 16), batch=64, context=8192,
                      steps=8)],
            layers=2, flows=[co.Flow("kv", src=range(8), dst=range(8, 16), bytes=chunk, sms=4,
                                     times=1 + int(decode_us * 1e3 / every), every_ns=every)])
    for fast in (True, False):
        yield f"shared ({'fast' if fast else 'exact'})", dict(tenants=[
            co.Tenant("8b", "llama3-8b", "decode", Plan(tp=4), gpus=range(4), batch=32, context=4096, steps=2),
            co.Tenant("1b", "llama3.2-1b", "decode", Plan(tp=4), gpus=range(4), batch=64, context=4096, steps=4)],
            layers=1, fast=fast)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run scenarios whose name contains this; keep the others' results")
    a = ap.parse_args()
    path = os.path.join(HERE, "results", "colocation.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name, kw in scenarios():
        if a.only and a.only not in name:
            continue
        t0 = time.time()
        res = co.run(kw.pop("tenants"), "gb300", **kw)
        res.pop("report")
        res["wall_s"] = time.time() - t0
        out[name] = res
        print(f"== {name} ({res['wall_s']:.0f} s)\n{co.summary(res)}\n", flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    md = []
    for name, res in out.items():
        md += [f"### {name}", "", "```", co.summary(res), "```", ""]
    with open(os.path.join(HERE, "results", "colocation.md"), "w") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
