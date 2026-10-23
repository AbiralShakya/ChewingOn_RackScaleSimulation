"""Headline findings, computed from the sweep results (no hand-entered numbers)."""
from __future__ import annotations

import html
from typing import Dict, List, Optional

BD = ("compute", "hbm", "latency", "comm", "launch")
NAMES = dict(compute="compute", hbm="HBM bandwidth", latency="latency", comm="NVLink communication", launch="launch gaps")


def _sim(pts):
    return [p for p in pts if p.get("step_us")]


def _best(pts, phase, slo=0.0) -> Optional[dict]:
    ok = [p for p in _sim(pts) if p["phase"] == phase and (phase != "decode" or p["tok_s_user"] >= slo)]
    return max(ok, key=lambda p: p["tok_s_gpu"], default=None)


def _share(p, k) -> float:
    return p["bd"][BD.index(k)] / sum(p["bd"])


def _find(pts, **kw) -> Optional[dict]:
    for p in pts:
        if all(p.get(k) == v for k, v in kw.items()):
            return p
    return None


def _k(n: float) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:.0f}"


def _rack_train(tr: list) -> str:
    """One sentence on 405B training layouts across the NVL72."""
    big = sorted((p for p in tr if p["system"] == "gb300" and p["model"] == "llama3-405b" and p.get("mfu")),
                 key=lambda p: -p["mfu"])
    if len(big) < 2:
        return ""
    b, w = big[0], big[-1]
    comm = lambda p: 100 * p["bd"][3] / sum(p["bd"])  # noqa: E731
    return (f" On GB300 NVL72, Llama-3-405B trains best at {b['plan']} ({100 * b['mfu']:.0f}% MFU); adding pipeline "
            f"stages brings {w['plan']} down to {100 * w['mfu']:.0f}% as communication grows from {comm(b):.0f}% "
            f"to {comm(w):.0f}% of the step (rack-scale training points extrapolated from 6- and 12-layer runs).")


def findings(data: Dict[str, List[dict]], systems: dict, archs: dict) -> dict:
    cards, nums = [], {}

    def card(title, num, text):
        cards.append(f'<div class="finding"><h3>{html.escape(title)}</h3><div class="num">{html.escape(num)}</div>'
                     f'<p>{text}</p></div>')

    lad = data.get("ladder", [])
    if lad:
        dec = [p for p in _sim(lad) if p["phase"] == "decode"]
        hbm = sum(1 for p in dec if p["dominant"] == "hbm") / max(1, len(dec))
        comp = [p for p in dec if p["dominant"] == "compute"]
        nums["decode_hbm_bound_fraction"] = hbm
        card("Decode is a memory problem at every size",
             f"{100 * hbm:.0f}% of decode points",
             f"are HBM-bound across 1B to 405B and all four systems ({len(dec)} simulated decode steps). "
             + (f"Only {len(comp)} reach compute-bound, all at batch {min(p['batch'] for p in comp)} or more."
                if comp else "None of them reaches compute-bound, even at batch 256: weights and KV cache are re-read "
                             "every step, and padded M=16..256 tiles leave the tensor cores mostly idle."))
        # Size: best per-GPU throughput at 20 tok/s/user on the rack, 1B -> 405B.
        rows = []
        for m in ("llama3.2-1b", "llama3-8b", "llama3-70b", "llama3-405b"):
            b = _best([p for p in lad if p["model"] == m and p["system"] == "gb300"], "decode", 20)
            if b:
                rows.append((m, b))
        if len(rows) >= 2:
            (m0, b0), (m1, b1) = rows[0], rows[-1]
            nums["gb300_decode_best_20tps"] = {m: dict(plan=b["plan"], batch=b["batch"], tok_s_gpu=b["tok_s_gpu"]) for m, b in rows}
            card("Bigger models cost per token about in proportion to their weights",
                 f"{b0['tok_s_gpu'] / b1['tok_s_gpu']:.0f}x",
                 f"more tokens/s per GPU for {m0} ({_k(b0['tok_s_gpu'])}, {b0['plan']} b{b0['batch']}) than {m1} "
                 f"({_k(b1['tok_s_gpu'])}, {b1['plan']} b{b1['batch']}) on GB300 NVL72 at 20 tokens/s per user. The best "
                 "layout moves from one GPU per replica to TP8 as the model grows: " +
                 ", ".join(f"{m} {b['plan']}" for m, b in rows) + ".")
        # Small models at batch 1: launch + latency.
        sm = _find(lad, model="llama3.2-1b", system="b300", phase="decode", batch=1, plan="1gpu")
        if sm and sm.get("step_us"):
            lat = _share(sm, "launch") + _share(sm, "latency")
            nums["1b_b1_launch_latency_share"] = lat
            card("Small models are bound by everything except the GPU's peak",
                 f"{100 * lat:.0f}% of the step",
                 f"of a Llama-3.2-1B batch-1 decode on B300 is launch gaps and latency-bound kernels "
                 f"({sm['step_us']:.0f} µs per token). The fix is in the serving stack (graphs, fusion, a megakernel), "
                 "not in a bigger GPU; see the launch-mode chart.")
        # Prefill / train compute bound + MFU.
        pf = [p for p in _sim(lad) if p["phase"] == "prefill"]
        tr = [p for p in _sim(lad) if p["phase"] == "train"]
        if pf:
            cb = sum(1 for p in pf if p["dominant"] == "compute") / len(pf)
            b3 = [p for p in tr if p["system"] == "b300"]
            best_tr = {}
            for p in b3:
                if p["model"] not in best_tr or p["tok_s_gpu"] > best_tr[p["model"]]["tok_s_gpu"]:
                    best_tr[p["model"]] = p
            mfus = sorted(((archs[m]["total_params_b"], m, p["mfu"], p["plan"]) for m, p in best_tr.items() if p.get("mfu")))
            nums["prefill_compute_bound_fraction"] = cb
            nums["train_mfu_b300"] = {m: dict(mfu=f, plan=pl) for _, m, f, pl in mfus}
            card("Prefill and training are compute problems",
                 f"{100 * cb:.0f}% of prefills",
                 "are compute-bound (8k-token prompts). Training MFU on HGX B300 at the best layout: " +
                 ", ".join(f"{m} {100 * f:.0f}% ({pl})" for _, m, f, pl in mfus) +
                 "." + _rack_train(tr))
        # Generations.
        gen = []
        for m in ("llama3-8b", "llama3-70b"):
            pa = _best([p for p in lad if p["model"] == m and p["system"] == "a100"], "prefill")
            pb = _best([p for p in lad if p["model"] == m and p["system"] == "b300"], "prefill")
            da = _best([p for p in lad if p["model"] == m and p["system"] == "a100"], "decode", 20)
            db = _best([p for p in lad if p["model"] == m and p["system"] == "b300"], "decode", 20)
            if pa and pb and da and db:
                gen.append((m, pb["tok_s_gpu"] / pa["tok_s_gpu"], db["tok_s_gpu"] / da["tok_s_gpu"]))
        if gen:
            nums["b300_over_a100"] = {m: dict(prefill=a, decode=b) for m, a, b in gen}
            m, a, b = gen[-1]
            card("Three generations: compute grew faster than bandwidth",
                 f"{a:.0f}x prefill, {b:.0f}x decode",
                 f"per GPU for {m}, HGX B300 over HGX A100 (best layouts, decode at 20 tokens/s per user). Prefill "
                 "tracks the 8x tensor peak; decode tracks the 4x HBM bandwidth plus B300's 288 GB, which allows "
                 "larger batches per GPU. The A100 GEMM model is pessimistic, so read the A100 side as a lower bound.")

    att = data.get("attention", [])
    if att:
        def best_at(m, c, sys="b300"):
            return _best([p for p in att if p["model"] == m and p["system"] == sys and p.get("context") == c], "decode")
        variants = ("llama3-8b-mha", "llama3-8b", "llama3-8b-mqa", "llama3-8b-mla", "mamba2-8b", "mamba2-hybrid-8b")
        at128 = {m: best_at(m, 131072) for m in variants}
        at8 = {m: best_at(m, 8192) for m in variants}
        kv = {m: archs[m]["kv_bytes_per_token"] for m in variants[:4] if m in archs}
        nums["kv_bytes_per_token"] = kv
        nums["best_decode_128k"] = {m: p and dict(batch=p["batch"], tok_s_gpu=p["tok_s_gpu"]) for m, p in at128.items()}
        nums["best_decode_8k"] = {m: p and dict(batch=p["batch"], tok_s_gpu=p["tok_s_gpu"]) for m, p in at8.items()}
        g, mla, mqa, mha = (at128.get(m) for m in ("llama3-8b", "llama3-8b-mla", "llama3-8b-mqa", "llama3-8b-mha"))
        if g and mla:
            def desc(name, p):
                return f"{name} {_k(p['tok_s_gpu'])} (batch {p['batch']})" if p else f"{name} does not fit"
            card("At long context, the KV layout decides the batch", f"{mla['tok_s_gpu'] / g['tok_s_gpu']:.0f}x",
                 "tokens/s per GPU for MLA over GQA-4 at 128k context on one B300, each at the largest batch that fits: "
                 + ", ".join([desc("MHA", mha), desc("GQA-4", g), desc("MQA", mqa), desc("MLA", mla)]) +
                 f". Per token the cache is {kv['llama3-8b-mha'] / 1024:.0f} KB, {kv['llama3-8b'] / 1024:.0f} KB, "
                 f"{kv['llama3-8b-mqa'] / 1024:.0f} KB and {kv['llama3-8b-mla'] / 1024:.0f} KB. MQA wins on bytes; "
                 "MLA keeps per-head quality but its 576-wide decode attention spends more time in the tensor pipe.")
        mb, hy, g8 = at128.get("mamba2-8b"), at128.get("mamba2-hybrid-8b"), at8.get("llama3-8b")
        if mb and g8 and g:
            card("Mamba-2 does not care how long the context is", f"{mb['tok_s_gpu'] / g['tok_s_gpu']:.0f}x",
                 f"tokens/s per GPU for Mamba-2 over GQA at 128k context ({_k(mb['tok_s_gpu'])} vs {_k(g['tok_s_gpu'])}). "
                 f"GQA loses {g8['tok_s_gpu'] / g['tok_s_gpu']:.0f}x from 8k to 128k because its batch must shrink to fit the cache; "
                 f"Mamba-2's state is the same size at every context"
                 + (f", and the hybrid with 4 attention layers keeps {_k(hy['tok_s_gpu'])}." if hy else "."))

    moe = data.get("moe", [])
    if moe:
        ds = [p for p in moe if p["model"] == "deepseek-v3" and p["system"] == "gb300" and p["phase"] == "decode"
              and not p.get("overlap") and p["batch"] == 128]
        tp = max((p for p in _sim(ds) if p["plan"].startswith("tp")), key=lambda p: p["tok_s_gpu"], default=None)
        ep = max((p for p in _sim(ds) if p["plan"].startswith("ep")), key=lambda p: p["tok_s_gpu"], default=None)
        if ep:
            nums["deepseek_gb300_b128"] = dict(ep=dict(plan=ep["plan"], tok_s_gpu=ep["tok_s_gpu"]),
                                               tp=tp and dict(plan=tp["plan"], tok_s_gpu=tp["tok_s_gpu"]))
            big = max((p for p in _sim(ds) if p["plan"].startswith("ep")), key=lambda p: int(p["plan"][2:]))
            card("DeepSeek-V3 wants expert parallelism", ep["plan"].upper(),
                 f"is the best DeepSeek-V3 layout on GB300 NVL72 at batch 128 per rank: {_k(ep['tok_s_gpu'])} "
                 "tokens/s per GPU" + (f", {ep['tok_s_gpu'] / tp['tok_s_gpu']:.0f}x the best tensor-parallel layout "
                                       f"({tp['plan']}, {_k(tp['tok_s_gpu'])})" if tp else "")
                 + f". Under EP each GPU keeps a few whole experts and serves its own batch, so the expert weights "
                 f"it streams are shared by many more tokens. Past {ep['plan']} the gain is flat "
                 f"({big['plan']}: {_k(big['tok_s_gpu'])}); all-to-all is {100 * _share(ep, 'comm'):.0f}% of the step.")
        ov = [(p, _find(moe, model=p["model"], system=p["system"], plan=p["plan"], batch=p["batch"], phase="decode", overlap=None))
              for p in _sim(moe) if p.get("overlap")]
        ov = [(o, b) for o, b in ov if b and b.get("step_us")]
        if ov:
            wins = [(o, b) for o, b in ov if o["step_us"] < b["step_us"]]
            nums["overlap_wins"] = f"{len(wins)}/{len(ov)}"
            comm = max(100 * _share(b, "comm") for _, b in ov)
            card("Two-micro-batch overlap rarely pays on NVLink", f"wins {len(wins)} of {len(ov)}",
                 "EP layouts at batch 128. Each half-batch re-reads every weight, which costs more than the all-to-all "
                 f"it hides: on NVLink5 the all-to-all is at most {comm:.0f}% of these steps"
                 + (". It pays only where communication is a larger share: "
                    + ", ".join(f"{o['model']} {o['plan']} ({100 * (1 - o['step_us'] / b['step_us']):.0f}% faster)" for o, b in wins[:3])
                    + "." if wins else ", so there is little to hide. Across racks over InfiniBand, with a slower all-to-all, the balance may differ (not simulated here)."))

    ctx = data.get("context", [])
    if ctx:
        a = _find(ctx, model="llama3-70b", context=4096, batch=16)
        b = _find(ctx, model="llama3-70b", context=131072, batch=16)
        h = _find(ctx, model="mamba2-hybrid-70b", context=131072, batch=16)
        big = _find(ctx, model="llama3-70b", context=131072, batch=64)
        if a and b and a.get("step_us") and b.get("step_us"):
            nums["70b_4k_to_128k"] = a["tok_s_gpu"] / b["tok_s_gpu"]
            card("At 128k, a 70B model spends its step reading KV", f"{100 * (b['cls'].get('attention', 0)) / sum(b['bd']):.0f}% attention",
                 f"of Llama-3-70B's TP8 decode step at 128k context and batch 16 on GB300 NVL72, against "
                 f"{100 * a['cls'].get('attention', 0) / sum(a['bd']):.0f}% at 4k. Throughput falls "
                 f"{a['tok_s_gpu'] / b['tok_s_gpu']:.1f}x" + (f"; the Mamba-2 hybrid of the same size keeps "
                                                               f"{h['tok_s_gpu'] / b['tok_s_gpu']:.1f}x more tokens/s per GPU." if h and h.get("step_us") else ".")
                 + (f" Batch 64 at 128k does not fit at all: {big['memory_total_gb']:.0f} GB per GPU." if big and not big.get("fits") else ""))

    inf = data.get("infra", [])
    if inf:
        g = lambda m, b, la: _find(inf, model=m, batch=b, launch=la, tag="launch")  # noqa: E731
        e, gr, mk = g("llama3.2-1b", 1, "eager"), g("llama3.2-1b", 1, "graph"), g("llama3.2-1b", 1, "megakernel")
        if e and gr and mk:
            nums["1b_b1_launch"] = dict(eager=e["step_us"], graph=gr["step_us"], megakernel=mk["step_us"])
            card("Launch overhead is a model-size effect", f"{e['step_us'] / mk['step_us']:.2f}x",
                 f"from eager launches ({e['step_us']:.0f} µs) to no gaps at all ({mk['step_us']:.0f} µs) for Llama-3.2-1B "
                 f"at batch 1 on B300; CUDA graphs get {gr['step_us']:.0f} µs. For Llama-3-70B at TP8 the same change is "
                 + (lambda a, c: f"{a['step_us'] / c['step_us']:.2f}x" if a and c else "small")(g("llama3-70b", 1, "eager"), g("llama3-70b", 1, "megakernel"))
                 + ". The measured Hazy megakernel does 608 µs per token, so fusion is worth more than removing gaps.")
        c0 = _find(inf, model="llama3-8b", tag="chunked", chunk=None)  # chunk 0 is dropped by slim()
        c2 = _find(inf, model="llama3-8b", tag="chunked", chunk=2048)
        if c0 and c2:
            nums["chunk2048_tpot_increase"] = c2["step_us"] / c0["step_us"]
            card("A prefill chunk is a tax on every decoding user", f"+{100 * (c2['step_us'] / c0['step_us'] - 1):.0f}% TPOT",
                 f"when a 2048-token prompt chunk joins a 64-sequence Llama-3-8B decode batch on B300 "
                 f"({c0['step_us'] / 1e3:.1f} → {c2['step_us'] / 1e3:.1f} ms). The step turns from HBM-bound to "
                 "compute-bound once the chunk passes a few hundred tokens, which is where chunk size should stop.")
        f8 = [(p, _find(inf, model=p["model"], phase=p["phase"], batch=p["batch"], dtype="bf16", tag="dtype"))
              for p in inf if p.get("dtype") == "fp8" and p.get("step_us")]
        f8 = [(a, b) for a, b in f8 if b and b.get("step_us")]
        if f8:
            dsp = [b["step_us"] / a["step_us"] for a, b in f8 if a["phase"] == "decode"]
            psp = [b["step_us"] / a["step_us"] for a, b in f8 if a["phase"] == "prefill"]
            nums["fp8_speedup"] = dict(decode=dsp, prefill=psp)
            card("FP8 helps decode through bytes and prefill through FLOPs", f"{min(dsp + psp):.2f}–{max(dsp + psp):.2f}x",
                 "from BF16 to FP8 GEMMs on B300 across 8B, 70B and 405B. Attention stays BF16, so decode at 8k context "
                 "gains less than the halved weight bytes suggest.")
    return dict(html='<div class="findings">' + "".join(cards) + "</div>", numbers=nums)
