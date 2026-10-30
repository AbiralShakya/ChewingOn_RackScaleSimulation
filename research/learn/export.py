"""Export what the learning page needs: model tables for the in-browser
closed forms, the hardware constants, and the simulator results it shows
next to them.

    python research/learn/export.py        # -> research/learn/data/learn.json

The page recomputes every estimate in JavaScript (``rsmath.js``) from these
tables; ``verify.py`` checks the port against ``rackscale.scaling`` case by
case.
"""

from __future__ import annotations

import json
import os
import sys

from rackscale import hnet, lower, models, scaling
from rackscale.lower import Parallel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TPS = (1, 2, 4, 8, 16, 32, 64, 72)

#: Models offered in the explorer (a size ladder, an MoE pair, a hybrid SSM).
MODELS = ["llama3.2-1b", "llama3-8b", "qwen2.5-32b", "llama3-70b", "llama3-405b", "qwen3-30b-a3b",
          "qwen3-235b-a22b", "deepseek-v3", "mamba2-hybrid-8b"]


def model_table(name: str) -> dict:
    m = models.ALL[name]
    n_layers = scaling._layers(m)
    # Per-layer weight shares for lower._weights_per_gpu: [TP-sharded, EP x TP-sharded, unsharded].
    entries = []
    if m.blocks is not None:
        per = m._block_params()
        entries = [[per[b], 0, 0] for b in m.blocks]
    else:
        for l in range(m.layers):
            if m.moe and l >= m.moe.first_dense_layers:
                mo = m.moe
                entries.append([m.attn_params() + 3 * m.hidden * mo.expert_ffn * mo.shared_experts,
                                3 * m.hidden * mo.expert_ffn * mo.experts, m.hidden * mo.experts])
            else:
                entries.append([m.attn_params() + m.ffn_params(l), 0, 0])
    one = lower._model_flops(m, 1, 1, False, lm_head=False)
    attn_coeff = 4.0 * m.attn_layers() * m.heads * m.head_dim
    dense = 2.0 * m.active_params(lm_head=False)
    ssm_dec = one - dense - attn_coeff * 1
    ssm_pre = lower._model_flops(m, 1, 2, False, lm_head=False) - dense - attn_coeff * 2
    state = 0.0
    if m.mamba is not None:
        mb = m.mamba
        state += m.block_list().count("M") * (mb.nheads * mb.headdim * mb.d_state * 4 + mb.conv_dim * (mb.d_conv - 1) * 2)
    if m.linear is not None:
        la = m.linear
        state += m.block_list().count("L") * (la.state_elems() * 4 + la.conv_dim * (la.d_conv - 1) * 2)
    valid_tp = []
    for tp in TPS:
        try:
            lower.check(m, Parallel(tp=tp))
            valid_tp.append(tp)
        except lower.ParallelError:
            pass
    expert = 0.0
    if m.moe:
        expert = 3.0 * m.hidden * m.moe.expert_ffn * m.moe.experts * scaling._moe_layers(m)
    return dict(
        name=m.name, hidden=m.hidden, layers=n_layers, heads=m.heads, kv_heads=m.kv_heads, head_dim=m.head_dim,
        vocab=m.vocab, ffn=m.ffn, bytes_per_param=m.bytes_per_param(),
        total_params=m.total_params(), expert_params=expert, active_no_head=m.active_params(lm_head=False),
        attn_layers=m.attn_layers(), mixers=scaling._mixers(m), moe_layers=scaling._moe_layers(m),
        moe=None if not m.moe else dict(experts=m.moe.experts, topk=m.moe.topk, expert_ffn=m.moe.expert_ffn,
                                        dispatch_bytes=m.moe.dispatch_bytes),
        attn_coeff=attn_coeff, ssm_dec=ssm_dec, ssm_pre=ssm_pre, ssm_state_tp1=state,
        kv_elems={str(tp): m.kv_elems_per_token_layer(tp) for tp in TPS}, layer_entries=entries,
        ffn_max=max(m.ffn, m.moe.expert_ffn * m.moe.topk if m.moe else 0),
        tp_crit_params=m.attn_params() + m.mlp_params(), valid_tp=valid_tp,
        kind="moe" if m.moe else ("hybrid" if m.blocks else "dense"),
        mla=bool(m.mla),
    )


def hardware_table(key: str) -> dict:
    hw = scaling.hardware(key)
    return dict(key=key, name=hw.name, gpu=hw.gpu, gpus=hw.gpus, domain=hw.domain, flops=hw.flops, hbm_bw=hw.hbm_bw,
                hbm_bytes=hw.hbm_bytes, nvlink_bw=hw.nvlink_bw, nvls=hw.nvls, ib_bw=hw.ib_bw, alpha=hw.alpha,
                alpha_p2p=hw.alpha_p2p, launch=hw.launch)


def _blocks(spans: list, rank: int) -> list:
    """Rank ``rank``'s compute as pipeline blocks [kind, micro-batch, start us, end us]:
    F (forward of the stage's layers), B (input gradients) and W (weight
    gradients), recovered from kernel names: backward kernels carry ``.bwd.``,
    a new block starts where the layer index jumps back (next micro-batch, or
    the weight-gradient pass restarting at the stage's last layer), and a
    backward block of GEMMs only is a W block."""
    ks = sorted((s for s in spans if s[0] == rank and s[2] == "kernel"), key=lambda s: s[4])
    raw = []
    for s in ks:
        name = s[3]
        bwd = ".bwd." in name
        layer = int(name.split(".")[0][1:]) if name.startswith("L") else -1
        op = name.split(".")[-1]
        if raw and raw[-1]["bwd"] == bwd and ((layer >= raw[-1]["layer"]) if not bwd else (layer <= raw[-1]["layer"])):
            b = raw[-1]
            b["end"] = s[5]
            b["layer"] = layer
            b["gemm_only"] &= op.endswith("_proj")
        else:
            raw.append(dict(bwd=bwd, layer=layer, start=s[4], end=s[5], gemm_only=op.endswith("_proj")))
    out, count = [], {"F": 0, "B": 0, "W": 0}
    for b in raw:
        kind = "F" if not b["bwd"] else ("W" if b["gemm_only"] else "B")
        out.append([kind, count[kind], round(b["start"] * 1e6, 2), round(b["end"] * 1e6, 2)])
        count[kind] += 1
    return out


def pipelines(model: str = "llama3-8b", layers: int = 8, pp: int = 4, m: int = 8, seqlen: int = 4096) -> dict:
    """Ideal-machine timelines of GPipe, 1F1B and ZB-H1 for the pipeline panel."""
    hw = scaling.hardware("gb300")
    mod = models.ALL[model].with_layers(layers)
    out = dict(model=model, layers=layers, pp=pp, micro_batches=m, seqlen=seqlen, schedules={})
    for sched in ("gpipe", "1f1b", "zb"):
        wl = lower.train_step(mod, Parallel(pp=pp), 1, seqlen, micro_batches=m, schedule=sched)
        r = scaling.replay(wl, hw, spans=True)
        p2p = [[s[0], s[2], round(s[4] * 1e6, 2), round(s[5] * 1e6, 2)] for s in r.spans if s[2] in ("send", "recv")]
        out["schedules"][sched] = dict(step_us=r.step * 1e6, ranks=[_blocks(r.spans, k) for k in range(pp)], p2p=p2p)
    return out


def colocation() -> dict:
    """research/colocation results, with each tenant's and flow's GPU ids
    (from the scenario definitions) for the rack map."""
    res = _load("research/colocation/results/colocation.json")
    if res is None:
        return None
    sys.path.insert(0, os.path.join(ROOT, "research", "colocation"))
    import run as colocation_run  # noqa: E402
    for name, kw in colocation_run.scenarios():
        if name not in res:
            continue
        ids = {t.name: list(t.gpus) for t in kw["tenants"]}
        for t in res[name]["tenants"]:
            t["gpu_ids"] = ids[t["name"]]
        for f, spec in zip(res[name].get("flows", []), kw.get("flows", [])):
            f["src_ids"], f["dst_ids"] = list(spec.src), list(spec.dst)
    return res


def _load(path: str):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    if p.endswith(".jsonl"):
        return [json.loads(l) for l in open(p)]
    return json.load(open(p))


def main() -> None:
    out = dict(
        models={n: model_table(n) for n in MODELS},
        hardware={k: hardware_table(k) for k in ("gb300", "b300", "h100")},
        check=_load("research/parallelism/results/check.jsonl"),
        net=_load("research/learn/data/net_sweep.json"),
        colocation=colocation(),
        bytes=_load("research/bytes/results/bytes.json"),
        ladders={},
        pipelines=pipelines(),
        hnet=dict(bytes_per_bpe_token=hnet.BYTES_PER_BPE_TOKEN,
                  models={h.name: dict(params=h.params(), compression=h.compression,
                                       stages=[dict(width=s.width, enc=s.enc, dec=s.dec, ratio=s.ratio) for s in h.stages],
                                       main=dict(hidden=h.main.hidden, layers=h.main.layers))
                          for h in hnet.ALL.values()}),
    )
    for m in ("llama3-8b", "llama3-70b", "llama3-405b", "deepseek-v3"):
        out["ladders"][m] = dict(train=scaling.ladder(m, "gb300", "train"),
                                 decode=scaling.ladder(m, "gb300", "decode", min_tps_user=30))
    with open(os.path.join(HERE, "data", "learn.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"), default=float)
    print({k: (len(json.dumps(v)) if v is not None else 0) for k, v in out.items()})


if __name__ == "__main__":
    main()
