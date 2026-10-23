"""Study B: decode attention variants (MHA/GQA/MQA, MLA, GTA, GLA) at kernel, GPU and NVL72 scale.

After "Hardware-Efficient Attention for Fast Decoding" (Zadouri, Strauss and Dao, 2025). All
variants keep Llama-3-70B's body (64 query heads of 128, d_model 8192); only the attention block
and its cache change.

  python research/dao_lab/decode_attention.py kernels   # per-rank attention kernel vs TP, batch, context
  python research/dao_lab/decode_attention.py rack      # full decode steps on GB300 NVL72, TP8 and TP4 replicas
"""
import itertools
import sys
import time

from common import GPU_SYSTEM, kernels_of, load, save
from rackscale import extrap, lower, models, sweep

VARIANTS = ["llama3-70b", "llama3-70b-mqa", "llama3-70b-gta8", "llama3-70b-mla", "llama3-70b-gla2"]
LABEL = {"llama3-70b": "GQA-8", "llama3-70b-mqa": "MQA", "llama3-70b-gta8": "GTA-8", "llama3-70b-mla": "MLA",
         "llama3-70b-gla2": "GLA-2"}
# (sequences per rank, context): 256k cached tokens per rank, plus one 1M-token sequence.
SHAPES = [(64, 4096), (16, 16384), (4, 65536), (1, 262144), (1, 1048576)]
TPS = [1, 2, 4, 8]


def split_kv(wl, wg=2):
    """Use the FlashMLA-style split-block schedule for every decode attention kernel."""
    for prog in ([wl["spmd"]["program"]] if "spmd" in wl else wl["ranks"]):
        for st in prog["streams"]:
            for o in st:
                k = o.get("kernel")
                if o.get("op") == "kernel" and k and k["type"] == "attention" and k["sq"] == 1:
                    k["kv_warpgroups"] = wg
    return wl


def kernels():
    rows = load("decode_attention_kernels.json")
    done = {(r["model"], r["tp"], r["batch"], r["context"], r.get("kv_warpgroups", 1)) for r in rows}
    for wg, name, tp, (b, ctx) in itertools.product((1, 2), VARIANTS, TPS, SHAPES):
        m = models.ALL[name]
        if (name, tp, b, ctx, wg) in done:
            continue
        wl = lower.decode_step(m, lower.Parallel(tp=tp), b, ctx, layers=1, lm_head=False)
        if wg > 1:
            split_kv(wl, wg)
        attn = kernels_of(wl, ("attention",))[0]["kernel"]
        comb = [o["kernel"] for o in kernels_of(wl, ("elementwise",)) if "combine" in o["name"]]
        t0 = time.time()
        us = extrap.simulate_us(GPU_SYSTEM, attn)
        us_comb = sum(extrap.simulate_us(GPU_SYSTEM, c) for c in comb)
        kv_bytes = b * ctx * m.kv_elems_per_token_layer(tp) * 2
        d = attn["d"]
        flops = 4.0 * b * attn["heads_q"] * ctx * d
        r = dict(model=name, label=LABEL[name], tp=tp, kv_warpgroups=wg, batch=b, context=ctx, spec=attn, attn_us=us,
                 combine_us=us_comb, kv_bytes=kv_bytes, kv_gbps=kv_bytes / us / 1e3,
                 flops=flops, tflops=flops / us / 1e6, flops_per_byte=flops / kv_bytes,
                 wall_s=time.time() - t0)
        print(f"{LABEL[name]:6s} wg{wg} tp{tp} b{b:<3d} ctx{ctx:<8d} {us:9.1f} us  {r['kv_gbps']:6.0f} GB/s "
              f"{r['tflops']:6.0f} TF/s  AI {r['flops_per_byte']:.0f}", flush=True)
        rows.append(r)
        save("decode_attention_kernels.json", rows)


def rack():
    """Whole decode steps (all 80 layers + LM head) on GB300 NVL72: 9 TP8 or 18 TP4 replicas.
    Batch per replica is the largest power of two up to 512 that fits in HBM."""
    rows = load("decode_attention_rack.json")
    done = {(r["model"], r["plan"]["tp"], r["context"], r["batch"], r.get("kv_warpgroups", 1)) for r in rows}
    for wg, ctx in itertools.product((2,), (8192, 32768, 131072)):
        for tp in (8, 4):
            for name in VARIANTS:
                m = models.ALL[name]
                p = lower.Parallel(tp=tp)
                best = None
                b = 512
                while b >= 1:
                    mem = sum(lower.memory(m, p, phase="decode", batch=b, context=ctx).values()) / 1e9
                    if mem <= sweep.HBM_USABLE * 288:
                        best = b
                        break
                    b //= 2
                for batch in sorted({best or 0, 32} - {0}):
                    if batch > (best or 0) or (name, tp, ctx, batch, wg) in done:
                        continue
                    pt = sweep.Point(name, "gb300", "decode", sweep.plan(tp=tp), batch=batch, context=ctx,
                                     study="dao_decode_attention")
                    t0 = time.time()
                    r = sweep.run_point(pt, transform=(lambda wl: split_kv(wl, wg)) if wg > 1 else None)
                    r["max_batch"] = best
                    r["kv_warpgroups"] = wg
                    r["label"] = LABEL[name]
                    print(f"{LABEL[name]:6s} wg{wg} tp{tp} ctx{ctx:<7d} b{batch:<4d} step {r.get('step_us', 0):8.0f} us "
                          f"tok/s/gpu {r.get('tok_s_gpu', 0):7.0f}  ({time.time() - t0:.0f}s) {r.get('error', '')}",
                          flush=True)
                    rows.append(r)
                    save("decode_attention_rack.json", rows)


if __name__ == "__main__":
    {"kernels": kernels, "rack": rack}[sys.argv[1]]()
