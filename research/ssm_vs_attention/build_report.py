#!/usr/bin/env python3
"""Assemble results/report.html from summary.json, figures.json and the sweeps."""
from __future__ import annotations

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def k(x):
    return f"{x / 1024:.1f}k" if x >= 1024 else f"{x:.0f}"


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    s = json.load(open(os.path.join(RES, "summary.json")))
    f = json.load(open(os.path.join(RES, "figures.json")))
    sw = {}
    for fam in ("70b",):
        p = os.path.join(RES, f"sweep_{fam}.json")
        if os.path.exists(p):
            sw[fam] = json.load(open(p))["points"]
    mb, ma = s["measured"]["b300"], s["measured"]["a100pcie"]
    dec_b = statistics.median([mb["decode"][b]["crossover_ctx"] for b in ("32", "64", "128", "256")])
    dec_a = statistics.median([ma["decode"][b]["crossover_ctx"] for b in ("32", "64", "128", "256")])
    pre_b, pre_a = mb["prefill"]["crossover_L"], ma["prefill"]["crossover_L"]
    st_b, st_a = mb["decode"]["256"]["state_tbps"], ma["decode"]["256"]["state_tbps"]
    kv_b = max(mb["decode"]["256"]["kv_tbps"])
    kv_a = max(ma["decode"]["256"]["kv_tbps"])
    stg_b, stg_a = s["ssd_stages"]["b300"], s["ssd_stages"]["a100pcie"]
    su_b = stg_b["state_update_b64"]
    L16b = stg_b["16384"]
    L16a = stg_a["16384"]
    tot16b = sum(v["us"] for v in L16b.values())
    tot16a = sum(v["us"] for v in L16a.values())
    val = s.get("validation", {})
    tp8_txt = ""
    p8 = os.path.join(RES, "validation_b300tp8.json")
    if os.path.exists(p8):
        v8 = json.load(open(p8))["cases"]
        big = [abs(r["err_vs_measured"]) for c, r in v8.items() if "state_update" in c or "attn_decode" in c]
        tp8_txt = (f"<p>The rack sweeps run per-GPU shapes that differ from the 8B ones (TP8 leaves 8 query heads and 1 KV head per GPU "
                   f"for attention, 32 heads and 1 group for Mamba-2, and narrower projections), so those were measured on the B300 too: "
                   f"{len(v8)} cases, median error {pct(statistics.median(abs(r['err_vs_measured']) for r in v8.values()))}, and "
                   f"{pct(statistics.median(big))} for the state updates and attention kernels that decide the rack results.</p>")

    def val_rows():
        rows = []
        classes = ["GEMM", "attention", "Mamba-2 decode", "conv1d", "SSD prefill"]
        for c in classes:
            cells = []
            for g in ("b300", "a100pcie"):
                v = val.get(g)
                if v and c in v["median_abs_err"]:
                    cells.append(f"<td>{pct(v['median_abs_err'][c])}</td><td class='n'>{v['count'][c]}</td>")
                else:
                    cells.append("<td>–</td><td class='n'>–</td>")
            rows.append(f"<tr><th>{c}</th>{''.join(cells)}</tr>")
        cells = "".join(f"<td><b>{pct(val[g]['overall'])}</b></td><td class='n'>{val[g]['n']}</td>" if g in val else "<td>–</td><td>–</td>"
                        for g in ("b300", "a100pcie"))
        rows.append(f"<tr class='tot'><th>all cases</th>{cells}</tr>")
        cells = "".join(f"<td>{pct(val[g]['overall_vs_ncu'])}</td><td class='n'></td>" if val.get(g, {}).get("overall_vs_ncu") is not None
                        else "<td>–</td><td></td>" for g in ("b300", "a100pcie"))
        rows.append(f"<tr><th>all cases, vs Nsight Compute time</th>{cells}</tr>")
        return "".join(rows)

    def stage_rows():
        out = []
        for st in ["chunk_cumsum", "chunk_state", "state_passing", "bmm_chunk", "chunk_scan"]:
            b, a = L16b.get(st), L16a.get(st)
            out.append(f"<tr><th><code>{st}</code></th><td>{b['us']:.0f}</td><td class='n'>{int(b['grid'])}</td>"
                       f"<td class='n'>{b['inst_per_warp']:.0f}</td><td>{b['top_stall'].replace('_', ' ')}</td>"
                       f"<td>{a['us']:.0f}</td><td>{a['top_stall'].replace('_', ' ')}</td></tr>")
        return "".join(out)

    rack = ""
    if "70b" in sw:
        pts = sw["70b"]
        fr8 = s.get("rack_frontier_tp8", {})
        ctxs = sorted({int(key.split("@")[1]) for key in fr8})
        models = [("llama3-70b", "Transformer"), ("mamba2-70b", "Mamba-2"), ("mamba2-hybrid-70b", "Hybrid")]
        trs = []
        for c in ctxs:
            row = f"<tr><th>{k(c)}</th>"
            best = max((fr8.get(f"{m}@{c}", {}).get("tokens_per_s_per_gpu", 0) for m, _ in models))
            for m, _ in models:
                q = fr8.get(f"{m}@{c}")
                if q:
                    cls = " class='win'" if q["tokens_per_s_per_gpu"] == best else ""
                    row += f"<td{cls}>{q['tokens_per_s_per_gpu']:,.0f}</td><td class='n'>{q['batch']}</td>"
                else:
                    row += "<td>does not fit</td><td class='n'>–</td>"
            trs.append(row + "</tr>")
        # Cycle-level bubble anatomy of one decode step per model (TP8, batch 64, 16k).
        stall_keys = ["async_wait", "tma_wait", "tensor_wait", "long_scoreboard", "lg_throttle", "math_pipe_throttle",
                      "barrier", "short_scoreboard", "selected"]
        brow = []
        for m, nm in models:
            p = next((q for q in pts if q["phase"] == "decode" and q["model"] == m and q["tp"] == 8 and q["batch"] == 64
                      and q["context"] == 16384 and "step_us" in q and not q.get("full_rack")), None)
            if not p:
                continue
            cells = "".join(f"<td>{pct(p['sched'].get(sk, 0.0))}</td>" for sk in stall_keys)
            brow.append(f"<tr><th>{nm}</th><td>{p['step_us'] / 1e3:.2f}</td><td>{pct(p['gpu_idle'])}</td>{cells}</tr>")
        bubbles = ""
        if brow:
            heads = "".join(f"<th>{sk.replace('_', ' ')}</th>" for sk in stall_keys)
            bubbles = (f"<h3>Bubble anatomy of one decode step</h3><p>Share of all warp-scheduler cycles by the reason the "
                       f"critical warp could not issue (Nsight Compute's stall taxonomy), plus the time the GPU sits idle "
                       f"between kernels. TP8, batch 64, 16k context.</p>"
                       f"<div class='tablewrap'><table class='data'><thead><tr><th>model</th><th>step ms</th><th>GPU idle</th>{heads}</tr></thead>"
                       f"<tbody>{''.join(brow)}</tbody></table></div>")
        def fr(m, c):
            return fr8.get(f"{m}@{c}", {}).get("tokens_per_s_per_gpu", 0.0)
        wi = s.get("whatif_ssu8", {})
        rack_text = (
            f"<ul><li><b>Short contexts belong to the Transformer.</b> At 1k tokens it decodes {fr('llama3-70b', 1024):,.0f} tokens/s per GPU, "
            f"{fr('llama3-70b', 1024) / fr('mamba2-hybrid-70b', 1024):.1f}× the hybrid and {fr('llama3-70b', 1024) / fr('mamba2-70b', 1024):.1f}× pure Mamba-2: "
            f"the KV cache is small and FA4 reads it near HBM speed, while every Mamba-2 layer pays the issue-bound state update.</li>"
            f"<li><b>From 16k tokens the hybrid wins</b> ({fr('mamba2-hybrid-70b', 16384):,.0f} vs {fr('llama3-70b', 16384):,.0f} tokens/s per GPU). With 9 attention "
            f"layers instead of 80 its KV cache is a ninth of the Transformer's, so it keeps batch 512 in HBM where the Transformer drops to "
            f"{fr8.get('llama3-70b@16384', {}).get('batch', '?')}.</li>"
            f"<li><b>Pure Mamba-2 is flat at {fr('mamba2-70b', 1024):,.0f} tokens/s per GPU</b> at every context, because its cost per token does not "
            f"depend on context. It passes the hybrid only at 128k tokens. Its ceiling is the state update: at batch 512 a TP8 decode step reads and "
            f"writes about 88 GB of state per GPU, at under 2 TB/s.</li>"
            f"<li><b>What-if: a state update with 8 headdim rows per CTA</b> instead of 4 (same math, twice the bytes per CTA, half the CTAs) "
            f"lifts pure Mamba-2 to {wi.get('tp8_b512', 0):,.0f} tokens/s per GPU at TP8 ({wi.get('tp8_b512', 0) / max(1.0, fr('mamba2-70b', 1024)):.1f}×), "
            f"ahead of the Transformer from about 4k tokens and of the hybrid from about 8k. The simulator lets this kernel stream at about 7.5 TB/s, above the 6.4 TB/s copy "
            f"bandwidth measured on B300, so read it as an upper bound that is about 15% high.</li></ul>")
        full = next((p for p in pts if p.get("full_rack") and "step_us" in p), None)
        same = next((p for p in pts if not p.get("full_rack") and full and p["phase"] == "decode" and p["model"] == full["model"]
                     and p["tp"] == full["tp"] and p["batch"] == full["batch"] and p["context"] == full["context"] and "step_us" in p), None)
        full_txt = ""
        if full and same:
            full_txt = (f"<p>To check that one tensor-parallel group stands for the whole rack, the same decode step "
                        f"(Transformer, TP8, batch 64, 16k context) was also simulated with all 72 GPUs at once (9 data-parallel "
                        f"replicas sharing the NVLink switches): {full['step_us'] / 1e3:.2f} ms against {same['step_us'] / 1e3:.2f} ms "
                        f"for a single 8-GPU group, so the all-reduces of different replicas do not contend on the fabric.</p>")
        rack = f"""
<section id="rack">
<h2>3. GB200 NVL72: what the rack sees</h2>
<p class="lede">The 70B-class models run as tensor-parallel replicas inside one NVL72 domain. For every model, context
length and batch the simulator reports the decode step at cycle level: weights streamed per GPU, the attention or
Mamba-2 kernels, the NVLink all-reduces, and the launch gaps of a CUDA-graph replay. Each batch is also checked against the
GPU's 186 GB of HBM (90% usable), so the throughput below is the best batch that actually fits.</p>
<div class="figpair">
<figure><figcaption>Decode throughput per GPU, TP4</figcaption>{f.get('rack_tput_tp4', '')}</figure>
<figure><figcaption>Decode throughput per GPU, TP8</figcaption>{f.get('rack_tput_tp8', '')}</figure>
</div>
<div class="tablewrap"><table class="data">
<thead><tr><th>context</th><th>Transformer<br><span>tok/s/GPU</span></th><th class="n">batch</th><th>Mamba-2<br><span>tok/s/GPU</span></th><th class="n">batch</th><th>Hybrid<br><span>tok/s/GPU</span></th><th class="n">batch</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
<p class="note">TP8 on GB200 NVL72, best batch from 1 to 512 that fits in HBM. Bold is the fastest model at that context.</p>
{rack_text}
<figure class="wide"><figcaption>Where a decode step goes (TP8, batch 64)</figcaption>{f.get('rack_breakdown', '')}{f.get('rack_breakdown_legend', '')}</figure>
<div class="figpair">
<figure><figcaption>Prefill time to first token, TP4</figcaption>{f.get('rack_prefill_tp4', '')}</figure>
<figure><figcaption>Prefill time to first token, TP8</figcaption>{f.get('rack_prefill_tp8', '')}</figure>
</div>
{bubbles}
{full_txt}
</section>"""

    html = f"""<title>Mamba-2 vs Attention on Blackwell</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: one reading column (~70ch) with figures allowed to run wider; data set in mono. */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --fg: #0d1117; --fg2: #4a5160; --muted: #7d8494; --rule: #dde1e8; --accent: #2a78d6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #6250d6;
  --win: #e6f0fb;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0e1013; --panel: #16191e; --fg: #eef0f3; --fg2: #b9bfca; --muted: #8b92a0; --rule: #2a2f37; --accent: #3987e5;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #9085e9; --win: #1b2a3d; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0e1013; --panel: #16191e; --fg: #eef0f3; --fg2: #b9bfca; --muted: #8b92a0; --rule: #2a2f37; --accent: #3987e5;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #9085e9; --win: #1b2a3d; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.6 var(--body); }}
main {{ max-width: 1040px; margin: 0 auto; padding-block: 40px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 56px; }}
header {{ display: grid; gap: 14px; }}
.kicker {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 700 clamp(34px, 6vw, 56px)/1.02 var(--display); letter-spacing: -.01em; margin: 0; text-wrap: balance; }}
h2 {{ font: 600 28px/1.15 var(--display); margin: 0 0 12px; text-wrap: balance; }}
h3 {{ font: 600 19px/1.3 var(--display); margin: 24px 0 6px; }}
p, li {{ max-width: 70ch; color: var(--fg); }}
.lede {{ color: var(--fg2); font-size: 17px; }}
.dek {{ font-size: 19px; color: var(--fg2); max-width: 62ch; margin: 0; }}
code, .mono {{ font-family: var(--mono); font-size: .88em; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 10px; }}
.tile {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 16px 18px; display: grid; gap: 6px; }}
.tile .lab {{ font: 500 12px/1.3 var(--mono); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }}
.tile .val {{ font: 600 30px/1.1 var(--display); font-variant-numeric: tabular-nums; }}
.tile .val small {{ font: 500 15px var(--body); color: var(--fg2); }}
.tile .sub {{ font-size: 13.5px; color: var(--fg2); }}
section {{ display: grid; gap: 6px; }}
figure {{ margin: 18px 0 6px; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 14px 10px; min-width: 0; }}
figcaption {{ font: 600 14px/1.3 var(--body); margin-bottom: 6px; }}
.figpair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
.figpair figure {{ margin: 8px 0; }}
svg.chart {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg .axis {{ stroke: var(--muted); stroke-width: 1; }}
svg .tick {{ fill: var(--muted); font: 11px var(--mono); }}
svg .alabel {{ fill: var(--fg2); font: 12px var(--body); }}
svg .dlabel {{ fill: var(--fg); font: 600 12px var(--body); }}
svg .note {{ fill: var(--fg2); font: 11px var(--mono); }}
svg .mark-line {{ stroke: var(--muted); stroke-dasharray: 3 3; }}
svg .diag {{ stroke: var(--muted); stroke-width: 1; }}
svg .band {{ fill: var(--rule); opacity: .55; }}
svg .dot {{ stroke: var(--panel); stroke-width: 1.5; }}
svg .dot:hover {{ stroke: var(--fg); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--fg2); margin-top: 8px; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
.tablewrap {{ overflow-x: auto; margin: 10px 0; }}
table.data {{ border-collapse: collapse; font: 14px var(--body); font-variant-numeric: tabular-nums; min-width: 520px; }}
table.data th, table.data td {{ padding: 7px 12px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }}
table.data th:first-child {{ text-align: left; }}
table.data thead th {{ font: 600 12.5px var(--body); color: var(--fg2); vertical-align: bottom; }}
table.data thead th span {{ font-weight: 400; color: var(--muted); }}
table.data td.n {{ color: var(--muted); font-family: var(--mono); font-size: 12.5px; }}
table.data td.win {{ background: var(--win); font-weight: 600; }}
table.data tr.tot th, table.data tr.tot td {{ border-top: 1.5px solid var(--muted); }}
.note {{ font-size: 13.5px; color: var(--muted); }}
.callout {{ border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; }}
pre {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; overflow-x: auto; font: 13px/1.5 var(--mono); }}
ul {{ padding-left: 20px; }}
a {{ color: var(--accent); }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<main>
<header>
<div class="kicker">rackscale · cycle-level study · B300, A100, GB200 NVL72</div>
<h1>Mamba-2 vs attention on Blackwell</h1>
<p class="dek">Measured on a real B300 and A100, explained instruction by instruction, then scaled to a 72-GPU rack in
the simulator. Blackwell pushes the point where Mamba-2 beats attention to roughly three times longer sequences than on
Ampere, because FlashAttention-4 rides the new tensor cores and HBM while the Mamba-2 kernels do not.</p>
<div class="tiles">
  <div class="tile"><div class="lab">Decode crossover, B300</div><div class="val">{k(dec_b)} <small>tokens</small></div><div class="sub">A100: {k(dec_a)}. Context where one attention layer's KV read costs more than a Mamba-2 state update (batch 32 to 256).</div></div>
  <div class="tile"><div class="lab">Prefill crossover, B300</div><div class="val">{k(pre_b)} <small>tokens</small></div><div class="sub">A100: {k(pre_a)}. Prompt length where FlashAttention costs more than the SSD chunked scan.</div></div>
  <div class="tile"><div class="lab">Mamba-2 state traffic, B300</div><div class="val">{st_b:.1f} <small>TB/s</small></div><div class="sub">vs {kv_b:.1f} TB/s for FA4 reading the KV cache. A100: {st_a:.1f} vs {kv_a:.1f} TB/s.</div></div>
</div>
</header>

<section id="setup">
<h2>Setup</h2>
<p>Three 8B-class models share d_model 4096 and Llama 3's vocabulary: a Transformer with the Llama-3-8B shape
(32 query heads, 8 KV heads, head dim 128), a pure Mamba-2 stack (expand 2, 128 heads of 64, d_state 128, 8 groups,
chunk 256; 64 blocks so its 7.0B non-embedding parameters match the Transformer's), and the hybrid layout of Waleffe et al.
(2024): 24 Mamba-2, 4 attention and 28 MLP blocks. The rack-scale runs use 70B-class versions of the same three at
d_model 8192, each with about 68.5B non-embedding parameters.</p>
<p>Every kernel comes from the open-source implementations: <code>selective_state_update</code> and
<code>mamba_chunk_scan_combined</code> from state-spaces/mamba 2.3.2 (Triton), <code>causal_conv1d</code> 1.7.0, FlashAttention-4
(CuTe DSL) on B300 and FlashAttention-2 2.8.3 on A100, and cuBLAS for the projections. Each kernel was timed on a B300
SXM6 (della-tridao, GPU 3) and an A100 80GB PCIe (Della) with CUDA graphs and a cold L2, then profiled with Nsight
Compute for cycles, instruction counts, DRAM bytes and stall reasons, and its SASS was dumped. The simulator then
runs the same shapes on its calibrated B300 and A100 models (the measured clocks, latencies, tensor-core and
<code>mma.sync</code> rates from <code>calibration/</code>). Beyond one GPU, only the simulator is used.</p>
</section>

<section id="decode">
<h2>1. Decode: the state update is issue-bound</h2>
<p class="lede">At decode time an attention layer reads its KV cache, which grows with context; a Mamba-2 layer reads and
writes a fixed recurrent state (2 MB per sequence per layer in bf16). On paper the state wins once the KV cache is
larger than twice the state. On B300 it takes much longer than that.</p>
<div class="figpair">
<figure><figcaption>B300: one layer's mixer at batch 64 (measured)</figcaption>{f['decode_b300']}</figure>
<figure><figcaption>A100: one layer's mixer at batch 64 (measured)</figcaption>{f['decode_a100pcie']}</figure>
</div>
<p class="callout"><code>selective_state_update</code> moves the state at only {st_b:.1f} TB/s on B300, about a quarter of its
7.7 TB/s, while FlashAttention-4 streams the KV cache at up to {kv_b:.1f} TB/s. On A100 the state update reaches
{st_a:.1f} TB/s, more than half of the part's bandwidth. That gap is why the crossover moves from {k(dec_a)} tokens on A100 to
{k(dec_b)} on B300.</p>
<p>The cause is visible in the SASS. With d_state 128 the kernel's hand-tuned table picks <code>BLOCK_SIZE_M = 4</code> and
4 warps, so each CTA owns 4 rows of headdim and moves just 2 KB of state. The compiled kernel (sm_103a, Triton 3.8) is about
270 instructions, mostly address arithmetic, an integer division for the head-to-group mapping, five
<code>BAR.SYNC</code>s and an <code>STSM</code>/<code>LDS</code> round trip that broadcasts B and C through shared memory.
Nsight Compute counts {su_b['inst_per_warp']:.0f} instructions per warp and reports the warps as mostly
<i>not selected</i> and <i>math pipe throttle</i>: the SMs are busy issuing instructions, and the DRAM pipe sits at
11 to 25% utilisation. On A100 the same work is a smaller multiple of the time HBM needs, so the kernel looks
bandwidth-bound there. A larger <code>BLOCK_SIZE_M</code> or a fused state update with the out projection would recover most of
the gap; the simulator runs the real Triton SASS for this kernel, so it can test such a change before it is written.</p>
</section>

<section id="prefill">
<h2>2. Prefill: FlashAttention-4 got faster than SSD did</h2>
<div class="figpair">
<figure><figcaption>B300: one layer's mixer, batch 1 (measured)</figcaption>{f['prefill_b300']}</figure>
<figure><figcaption>A100: one layer's mixer, batch 1 (measured)</figcaption>{f['prefill_a100pcie']}</figure>
</div>
<p>Moving from A100 to B300, causal FlashAttention at 16k tokens gets {ma['prefill']['attn_us'][4] / mb['prefill']['attn_us'][4]:.1f}× faster, but the
Mamba-2 SSD scan gets only {ma['prefill']['ssd_us'][4] / mb['prefill']['ssd_us'][4]:.1f}× faster, so the prefill crossover moves from {k(pre_a)} to {k(pre_b)} tokens.
The SSD is five Triton kernels, and none of them is a large, dense GEMM that can keep <code>tcgen05</code> busy:</p>
<figure class="wide"><figcaption>SSD kernels per layer at 4k and 16k tokens (Nsight Compute)</figcaption>{f['ssd_stages']}{f['ssd_legend']}</figure>
<div class="tablewrap"><table class="data">
<thead><tr><th>kernel (16k tokens)</th><th>B300 µs</th><th class="n">CTAs</th><th class="n">inst/warp</th><th>B300 top stall</th><th>A100 µs</th><th>A100 top stall</th></tr></thead>
<tbody>{stage_rows()}</tbody></table></div>
<ul>
<li><b><code>chunk_scan</code> is {pct(L16b['chunk_scan']['us'] / tot16b)} of the SSD on B300.</b> The autotuner settles on a 2-warp,
64x32 tile; each CTA issues a few small <code>UTCHMMA</code>s per k-step and spends the rest on the decay mask
<code>exp(dA_i - dA_j)</code>: one <code>MUFU.EX2</code> per element of the 256x256 <code>C B^T</code> block, plus the SiLU gate.
Tensor pipes stay mostly idle, which is the inverse of FlashAttention-4, which keeps two 128-row Q tiles in flight to hide its exponentials behind the MMAs.</li>
<li><b><code>chunk_cumsum</code> costs a flat {L16b['chunk_cumsum']['us']:.0f} µs</b> whatever the length: 32 to 128 CTAs on 148 SMs, one warp per
scheduler, {L16b['chunk_cumsum']['inst_per_warp']:.0f} instructions per warp of fully unrolled scan, stalled on instruction fetch.
At 4k tokens that is {pct(stg_b['4096']['chunk_cumsum']['us'] / sum(v['us'] for v in stg_b['4096'].values()))} of the SSD.</li>
<li><b><code>state_passing</code> is a sequential scan over chunks</b>; every iteration waits on a DRAM load (long scoreboard), so it
is latency-bound on both parts.</li>
</ul>
</section>
{rack}
<section id="validation">
<h2>4. How close is the simulator?</h2>
<p class="lede">Every measured case was simulated on the calibrated GPU model with the same shapes, back to back as in the
CUDA-graph replay that produced the measurement.</p>
<div class="figpair">
<figure><figcaption>B300: simulated vs measured (µs, shaded band ±25%)</figcaption>{f.get('val_b300', '<p class="note">pending</p>')}</figure>
<figure><figcaption>A100 PCIe: simulated vs measured</figcaption>{f.get('val_a100pcie', '<p class="note">pending</p>')}</figure>
</div>
{f.get('val_legend', '')}
<div class="tablewrap"><table class="data">
<thead><tr><th>median |error|</th><th>B300</th><th class="n">cases</th><th>A100</th><th class="n">cases</th></tr></thead>
<tbody>{val_rows()}</tbody></table></div>
{tp8_txt}
<p>Two references are reported: the CUDA-graph timing, and Nsight Compute's per-kernel duration, which replays one
kernel at a time. Under Nsight Compute the tensor-bound kernels already run at about 1.79 GHz on the B300 (2.0 GHz for the
memory-bound ones); in a sustained back-to-back replay the power cap pulls the clock lower still, and the large GEMMs take 15 to 30%
longer than Nsight Compute reports. The simulator's large-GEMM times sit close to the Nsight Compute numbers.</p>
<p>What it took to get there, all on this branch: FlashAttention-4's two-Q-tile ping-pong on Blackwell; HMMAs with independent
accumulators and double-buffered <code>ldmatrix</code> fragments (the synthetic kernels had chained every HMMA and every
exponential through one register); split-K and narrow N tiles for decode-sized GEMMs, as cuBLAS uses; the autotuned SSD
tile shapes and the L2 reuse of <code>C B^T</code> across heads; a separate, measured <code>mma.sync</code> rate (on B300 it
is 2037 BF16 FLOP/clk/SM, a quarter of <code>tcgen05</code>); and the real Triton SASS for the state update.</p>
<p>The A100 model is the weaker of the two. Its synthetic <code>mma.sync</code> GEMMs land a median 33% from cuBLAS, and it
runs the SSD scan 60 to 70% slower than the hardware, almost all of it in <code>chunk_scan</code>. The crossovers above come
from the hardware measurements, and the rack results use only the Blackwell model, so neither depends on the A100
simulation. Small kernels under about 10 µs (launch-bound conv1d updates, the split-KV combine) are the other known gap on
both parts.</p>
</section>

<section id="repro">
<h2>Reproduce</h2>
<pre># on the GPU (B300: della-tridao GPU 3; A100: Della Slurm)
research/ssm_vs_attention/hw/setup_env.sh          # torch, mamba_ssm, causal-conv1d, FA4/FA2
python research/ssm_vs_attention/hw/bench_kernels.py --out timing.json
research/ssm_vs_attention/hw/run_ncu.sh out/        # Nsight Compute + SASS

# anywhere (simulator)
maturin develop --release
python research/ssm_vs_attention/validate.py --gpu b300
python research/ssm_vs_attention/sweep.py --family 70b --full-rack
python research/ssm_vs_attention/analyze.py && python research/ssm_vs_attention/build_report.py</pre>
<p class="note">Raw data: <code>research/ssm_vs_attention/data/</code> (timings, Nsight Compute CSVs, Triton SASS and metadata);
results: <code>results/*.json</code>. The MFU over-count of the LM head for prefill and training with <code>--layers</code> is
fixed on the same branch.</p>
</section>
</main>"""
    with open(os.path.join(RES, "report.html"), "w") as fh:
        fh.write(html)
    print("wrote", os.path.join(RES, "report.html"), len(html))


if __name__ == "__main__":
    main()
