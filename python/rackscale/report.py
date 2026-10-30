"""Publication-quality run reports: one self-contained HTML page, and each figure as its own SVG.

    rep = rs.simulate("gb200_nvl72", wl, fast=True)
    report.write(rep, "ds.html", system="gb200_nvl72")                    # one run
    report.write([rep_h100, rep_b200], "cmp.html", system=["hgx_h100_8", "hgx_b200_8"])   # side by side
    report.figures(rep, "figs/", system="gb200_nvl72")                    # SVGs for an article

From the command line: ``--html PAGE.html`` (and ``--figures DIR``) on every simulate command,
``--system a,b,c`` to compare systems, or ``rackscale report report.json`` for a saved run.

The page answers, in order: how long is the step and how efficient (headline tiles), where the
GPU time goes (one segmented bar: tensor-bound, HBM-bound and latency-bound kernels,
collectives, idle), where each kernel sits against the GPU's roofline, which kernels cost the
most and how close each gets to its roof, and what the schedulers wait on. Kernels are classed
by the rules in :mod:`rackscale.analysis` (HBM-bound at >= 70% of achievable HBM, tensor-bound at
>= 60% of dense peak, otherwise latency or pipeline bound).

Charts are inline SVG with no dependencies. Colours follow a CVD-validated categorical order
(blue, orange, aqua), with light and dark steps; exported SVGs carry their own styles.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Union

from . import analysis

# Categorical slots 1-3 (validated all-pairs in both modes) plus a neutral and slots 4-5 for
# the segmented bar, whose segments are adjacent-only.
_CSS_VARS_LIGHT = """
  --surface:#fcfcfb; --surface-2:#f3f2ef; --grid:#e6e5e0; --axis:#c9c7c0;
  --text:#0b0b0b; --text-2:#52514e; --text-3:#7a7873;
  --c-tensor:#2a78d6; --c-hbm:#eb6834; --c-latency:#1baf7a; --c-coll:#eda100; --c-idle:#b9b7b0;
  --c-roof:#0b0b0b;"""
_CSS_VARS_DARK = """
  --surface:#1a1a19; --surface-2:#232321; --grid:#2f2f2c; --axis:#4a4945;
  --text:#f0efec; --text-2:#c3c2b7; --text-3:#8f8d86;
  --c-tensor:#3987e5; --c-hbm:#d95926; --c-latency:#199e70; --c-coll:#c98500; --c-idle:#5d5c57;
  --c-roof:#f0efec;"""

_SVG_STYLE = ("<style>.rs{font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;"
              + _CSS_VARS_LIGHT + "}"
              "@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])) .rs{" + _CSS_VARS_DARK + "}}"
              ":root[data-theme=dark] .rs{" + _CSS_VARS_DARK + "}"
              ".rs text{fill:var(--text-2);font-size:12px}.rs .t1{fill:var(--text);}.rs .t3{fill:var(--text-3);font-size:11px}"
              ".rs .grid{stroke:var(--grid);stroke-width:1}.rs .axis{stroke:var(--axis);stroke-width:1}"
              ".rs .roof{stroke:var(--c-roof);stroke-width:2;fill:none;stroke-linecap:round}"
              ".rs .bg{fill:var(--surface)}"
              ".rs .tensor{fill:var(--c-tensor)}.rs .hbm{fill:var(--c-hbm)}.rs .latency{fill:var(--c-latency)}"
              ".rs .coll{fill:var(--c-coll)}.rs .idle{fill:var(--c-idle)}.rs .bar{fill:var(--c-tensor)}"
              ".rs .dot{stroke:var(--surface);stroke-width:2;fill-opacity:.85}"
              ".rs [data-tip]:hover{opacity:.8}</style>")

#: What the three kernel classes mean in the report.
CLASSES = [
    ("tensor", "Tensor-bound", "at least 60% of the dense tensor peak"),
    ("hbm", "HBM-bound", "at least 70% of achievable HBM bandwidth"),
    ("latency", "Latency or pipeline bound", "neither roof reached: waits on memory latency, handoffs, ALUs"),
]
_SEGMENTS = CLASSES + [("coll", "Collectives", "time inside communication kernels"),
                       ("idle", "Idle", "GPU waiting between kernels: launches, dependencies, stragglers")]


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt_us(us: float) -> str:
    if us >= 10_000:
        return f"{us / 1000:.1f} ms"
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.1f} µs"


def _short(name: str, n: int = 44) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _gpu_spec(system) -> Optional[dict]:
    if system is None:
        return None
    if isinstance(system, dict):
        return system
    from . import _need_native, _system_yaml
    return json.loads(_need_native().resolve_system(_system_yaml(system)))["gpus"][0]


def _raw(rep) -> dict:
    return rep.raw if hasattr(rep, "raw") else rep


def analyze(rep, system=None, label: Optional[str] = None) -> dict:
    """Everything the figures need from one report: classified kernels, the time split, stalls."""
    r = _raw(rep)
    gpu = _gpu_spec(system) or r.get("gpu_spec")
    world, mk = r["world"], r["makespan_us"]
    span = world * mk or 1.0
    kernels = []
    split = {k: 0.0 for k, *_ in _SEGMENTS}
    for row in r["kernels"]:
        if gpu:
            c = analysis.classify(row, None, None, gpu)
            bound = c["bound"] if c["bound"] in ("tensor", "hbm") else "latency"
            pk = analysis.peaks(gpu, c["dtype"])
        else:
            bound, pk = "latency", None
        share = row["total_us"] / span
        split[bound] += share
        ai = row["tflops"] * 1e12 / (row["gbps"] * 1e9) if row["gbps"] > 0 else None
        roof_frac = None
        if pk:
            roof_frac = max(row["tflops"] / pk["tflops"] if pk["tflops"] else 0,
                            row["gbps"] / pk["hbm_gbps"] if pk["hbm_gbps"] else 0)
        kernels.append({"name": row["name"], "dtype": analysis._dtype(row["name"], None), "count": row["count"], "total_us": row["total_us"],
                        "mean_us": row["mean_us"], "share": share, "tflops": row["tflops"], "gbps": row["gbps"],
                        "ai": ai, "bound": bound, "roof_frac": roof_frac, "tensor_util": row.get("tensor_util", 0)})
    coll = sum(x.get("collective_us", 0) for x in r["ranks"]) / span
    idle = r["bubbles"]["gpu_idle_fraction"]
    # Collective kernels are also in the kernel list on some paths; keep the bar summing to 1.
    split["coll"] = coll
    split["idle"] = idle
    tot = sum(split.values()) or 1.0
    split = {k: v / tot for k, v in split.items()}
    pk = analysis.peaks(gpu, "bf16") if gpu else None
    return {"raw": r, "gpu": gpu, "peaks": pk, "kernels": kernels, "split": split,
            "sched": r["bubbles"]["sched"], "label": label or r["system"]}


# ---------------------------------------------------------------- SVG figures

def _svg(w: int, h: int, body: str, title: str) -> str:
    return (f'<svg class="rs" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
            f'role="img" aria-label="{_esc(title)}" style="max-width:{w}px;height:auto">{_SVG_STYLE}'
            f'<rect class="bg" x="0" y="0" width="{w}" height="{h}"/>{body}</svg>')


def _rounded_hbar(x: float, y: float, w: float, h: float, cls: str, tip: str = "") -> str:
    """Horizontal bar: square at the baseline (left), 4px rounded data-end (right)."""
    if w <= 0:
        return ""
    r = min(4.0, w / 2, h / 2)
    d = (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
         f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")
    t = f' data-tip="{_esc(tip)}"' if tip else ""
    return f'<path class="{cls}" d="{d}"{t}/>'


def fig_time_split(a: dict, w: int = 760) -> str:
    """Where the GPU time goes: one segmented bar, 2px surface gaps, legend below with values."""
    h, bh, x0, y0 = 118, 28, 16, 34
    inner = w - 2 * x0
    parts, x = [], x0
    segs = [(k, lab, a["split"].get(k, 0.0)) for k, lab, _ in _SEGMENTS]
    mk = a["raw"]["makespan_us"]
    vis = [s for s in segs if s[2] > 0.0005]
    gap = 2.0
    usable = inner - gap * (len(vis) - 1)
    for i, (k, lab, f) in enumerate(vis):
        sw = usable * f
        r_l = 4 if i == 0 else 0
        r_r = 4 if i == len(vis) - 1 else 0
        tip = f"{lab}: {100 * f:.1f}% of GPU time ({_fmt_us(f * mk)} of the {_fmt_us(mk)} step per GPU)"
        parts.append(f'<rect class="{k}" x="{x:.1f}" y="{y0}" width="{max(sw, 0.5):.1f}" height="{bh}" '
                     f'rx="{max(r_l, r_r)}" data-tip="{_esc(tip)}"/>')
        if r_l == 0 and r_r == 0 or (r_l and not r_r) or (r_r and not r_l):
            # square off the interior corners of the end segments
            if r_l and not r_r and sw > 8:
                parts.append(f'<rect class="{k}" x="{x + sw - 6:.1f}" y="{y0}" width="6" height="{bh}"/>')
            if r_r and not r_l and sw > 8:
                parts.append(f'<rect class="{k}" x="{x:.1f}" y="{y0}" width="6" height="{bh}"/>')
        if sw > 46:
            dark_text = k in ("latency", "coll", "idle")
            parts.append(f'<text x="{x + sw / 2:.1f}" y="{y0 + bh / 2 + 4:.1f}" text-anchor="middle" '
                         f'style="fill:{"#0b0b0b" if dark_text else "#ffffff"};font-weight:600">{100 * f:.0f}%</text>')
        x += sw + gap
    lx, ly = x0, y0 + bh + 30
    for k, lab, f in segs:
        txt = f"{lab} {100 * f:.1f}%"
        parts.append(f'<rect class="{k}" x="{lx}" y="{ly - 9}" width="10" height="10" rx="2"/>'
                     f'<text x="{lx + 15}" y="{ly}" class="t1">{_esc(txt)}</text>')
        lx += 22 + 7.0 * len(txt)
    parts.append(f'<text x="{x0}" y="20" class="t1" style="font-weight:600;font-size:13px">Where the GPU time goes</text>'
                 f'<text x="{w - x0}" y="20" text-anchor="end" class="t3">share of all GPUs × step time</text>')
    return _svg(w, h, "".join(parts), "Where the GPU time goes")


def fig_roofline(a: dict, w: int = 760, h: int = 430, labels: int = 4) -> str:
    """Every kernel on the GPU's roofline (dense BF16 peak, achievable HBM), sized by its share of time."""
    pk = a["peaks"]
    ks = [k for k in a["kernels"] if k["ai"] and k["tflops"] > 0]
    ml, mr, mt, mb = 64, 20, 40, 48
    pw, ph = w - ml - mr, h - mt - mb
    body = [f'<text x="16" y="22" class="t1" style="font-weight:600;font-size:13px">Kernels on the roofline</text>']
    if not ks or not pk:
        body.append('<text x="16" y="60">No kernels with both FLOPs and bytes, or no GPU spec for the roof.</text>')
        return _svg(w, 90, "".join(body), "Roofline")
    hbm_peak = a["gpu"]["hbm"]["bw_gbps"]
    ridge = pk["tflops"] * 1e12 / (hbm_peak * 1e9)
    xmin = 10 ** math.floor(math.log10(min(min(k["ai"] for k in ks), ridge / 10)))
    xmax = 10 ** math.ceil(math.log10(max(max(k["ai"] for k in ks), ridge * 10)))
    # Kernels in a narrower type (FP8, FP4) have a higher compute roof: draw it too, thin.
    extra = []
    for dt in sorted({k["dtype"] for k in ks} - {"bf16", "fp16"}):
        p2 = (analysis.peaks(a["gpu"], dt) or {}).get("tflops") or 0
        if p2 > pk["tflops"] * 1.01:
            extra.append((dt, p2))
    ymax = 10 ** math.ceil(math.log10(max([pk["tflops"]] + [p for _, p in extra]) * 1.05))
    ymin = 10 ** math.floor(math.log10(max(1e-3, min(k["tflops"] for k in ks))))
    lx = lambda v: ml + pw * (math.log10(v) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))  # noqa: E731
    ly = lambda v: mt + ph - ph * (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))  # noqa: E731
    e = math.log10(xmin)
    while e <= math.log10(xmax) + 1e-9:
        v = 10 ** e
        body.append(f'<line class="grid" x1="{lx(v):.1f}" y1="{mt}" x2="{lx(v):.1f}" y2="{mt + ph}"/>'
                    f'<text x="{lx(v):.1f}" y="{mt + ph + 18}" text-anchor="middle" class="t3">{v:g}</text>')
        e += 1
    e = math.log10(ymin)
    while e <= math.log10(ymax) + 1e-9:
        v = 10 ** e
        body.append(f'<line class="grid" x1="{ml}" y1="{ly(v):.1f}" x2="{ml + pw}" y2="{ly(v):.1f}"/>'
                    f'<text x="{ml - 8}" y="{ly(v) + 4:.1f}" text-anchor="end" class="t3">{v:g}</text>')
        e += 1
    body.append(f'<line class="axis" x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}"/>')
    body.append(f'<text x="{ml + pw / 2}" y="{h - 10}" text-anchor="middle" class="t3">'
                f'arithmetic intensity (FLOP per byte of HBM traffic, log)</text>'
                f'<text transform="translate(16,{mt + ph / 2}) rotate(-90)" text-anchor="middle" class="t3">TFLOPS (log)</text>')
    # The roof: bandwidth slope up to the ridge, then the flat compute peak.
    x_start = max(xmin, ymin * 1e12 / (hbm_peak * 1e9))
    pts = [(x_start, hbm_peak * 1e9 * x_start / 1e12), (ridge, pk["tflops"]), (xmax, pk["tflops"])]
    body.append('<polyline class="roof" points="' + " ".join(f"{lx(x):.1f},{ly(y):.1f}" for x, y in pts) + '"/>')
    body.append(f'<text x="{ml + pw - 4}" y="{ly(pk["tflops"]) - 7:.1f}" text-anchor="end" class="t1">'
                f'dense BF16 peak {pk["tflops"]:,.0f} TFLOPS</text>')
    for dt, p2 in extra:
        r2 = p2 * 1e12 / (hbm_peak * 1e9)
        pts2 = [(ridge, pk["tflops"]), (min(r2, xmax), min(p2, hbm_peak * 1e9 * xmax / 1e12)), (xmax, p2)]
        body.append('<polyline class="roof" style="stroke-width:1;stroke-dasharray:4 3" points="'
                    + " ".join(f"{lx(x):.1f},{ly(y):.1f}" for x, y in pts2) + '"/>'
                    f'<text x="{ml + pw - 4}" y="{ly(p2) - 7:.1f}" text-anchor="end" class="t1">'
                    f'dense {dt.upper()} peak {p2:,.0f} TFLOPS</text>')
    # Label the bandwidth roof along its slope, low on the chart where kernels rarely sit.
    xs = x_start * 1.6
    ang = -math.degrees(math.atan2(ly(hbm_peak * 1e9 * xs * 10 / 1e12) - ly(hbm_peak * 1e9 * xs / 1e12), lx(xs * 10) - lx(xs)))
    body.append(f'<text transform="translate({lx(xs) + 6:.1f},{ly(hbm_peak * 1e9 * xs / 1e12) + 16:.1f}) rotate({-ang:.1f})" '
                f'class="t1">HBM {hbm_peak / 1000:.1f} TB/s</text>')
    smax = max(k["share"] for k in ks) or 1.0
    order = sorted(ks, key=lambda k: -k["share"])
    for k in sorted(ks, key=lambda k: -k["share"]):
        r = 4 + 14 * math.sqrt(k["share"] / smax)
        tip = (f"{k['name']}\n{k['count']} launches, {100 * k['share']:.1f}% of GPU time\n"
               f"{k['tflops']:,.0f} TFLOPS at {k['ai']:.1f} FLOP/B ({k['gbps']:,.0f} GB/s)"
               + (f"\n{100 * k['roof_frac']:.0f}% of its roof" if k["roof_frac"] is not None else ""))
        body.append(f'<circle class="dot {k["bound"]}" cx="{lx(k["ai"]):.1f}" cy="{ly(k["tflops"]):.1f}" '
                    f'r="{r:.1f}" data-tip="{_esc(tip)}"/>')
    # Top kernels labelled in a column at the lower right, with leader lines to their dots.
    col_x = ml + pw * 0.50
    for i, k in enumerate(sorted(order[:labels], key=lambda k: -k["tflops"])):
        x, y = lx(k["ai"]), ly(k["tflops"])
        ty = mt + ph - 14 - 18 * (min(labels, len(order)) - 1 - i)
        body.append(f'<polyline class="axis" fill="none" points="{x:.1f},{y:.1f} {col_x - 8:.1f},{ty - 4:.1f}"/>'
                    f'<text x="{col_x:.1f}" y="{ty:.1f}" class="t1">{_esc(_short(k["name"], 36))}'
                    f' <tspan class="t3">{100 * k["share"]:.0f}%</tspan></text>')
    lgx = ml + 8
    for key, lab, _ in CLASSES:
        body.append(f'<circle class="{key}" cx="{lgx + 5}" cy="{mt + 10}" r="5"/>'
                    f'<text x="{lgx + 14}" y="{mt + 14}" class="t1">{_esc(lab)}</text>')
        lgx += 26 + 7.0 * len(lab)
    return _svg(w, h, "".join(body), "Kernels on the roofline")


def fig_top_kernels(a: dict, w: int = 760, n: int = 10) -> str:
    """The costliest kernels by share of GPU time, coloured by class, with how close each gets to its roof."""
    ks = sorted(a["kernels"], key=lambda k: -k["share"])[:n]
    row, top, lw = 30, 40, 300
    h = top + row * len(ks) + 30
    bw = w - lw - 150
    smax = max((k["share"] for k in ks), default=1.0) or 1.0
    body = [f'<text x="16" y="22" class="t1" style="font-weight:600;font-size:13px">Costliest kernels</text>'
            f'<text x="{lw + 4}" y="22" class="t3">share of GPU time</text>'
            f'<text x="{w - 16}" y="22" text-anchor="end" class="t3">of its roof</text>']
    for i, k in enumerate(ks):
        y = top + i * row
        body.append(f'<text x="{lw - 10}" y="{y + 15}" text-anchor="end" class="t1">{_esc(_short(k["name"], 42))}</text>')
        bwid = bw * k["share"] / smax
        tip = (f"{k['name']}: {k['count']} launches, {_fmt_us(k['mean_us'])} each, {100 * k['share']:.1f}% of GPU time; "
               f"{k['tflops']:,.0f} TFLOPS, {k['gbps']:,.0f} GB/s")
        body.append(_rounded_hbar(lw, y + 4, max(bwid, 1), 16, k["bound"], tip))
        body.append(f'<text x="{lw + bwid + 6:.1f}" y="{y + 16}">{100 * k["share"]:.1f}%</text>')
        if k["roof_frac"] is not None:
            body.append(f'<text x="{w - 16}" y="{y + 16}" text-anchor="end" class="t1">{100 * k["roof_frac"]:.0f}%</text>')
    body.append(f'<line class="axis" x1="{lw}" y1="{top - 2}" x2="{lw}" y2="{top + row * len(ks) - 6}"/>')
    lgx, ly = lw, h - 10
    for key, lab, _ in CLASSES:
        body.append(f'<rect class="{key}" x="{lgx}" y="{ly - 9}" width="10" height="10" rx="2"/>'
                    f'<text x="{lgx + 15}" y="{ly}" class="t3">{_esc(lab)}</text>')
        lgx += 26 + 6.2 * len(lab)
    return _svg(w, h, "".join(body), "Costliest kernels")


def fig_stalls(a: dict, w: int = 760) -> str:
    """Scheduler-cycle anatomy: what every SM sub-partition's scheduler did, cycle by cycle."""
    s = sorted(((k, v) for k, v in a["sched"].items() if v >= 0.001), key=lambda kv: -kv[1])
    row, top, lw = 26, 40, 170
    h = top + row * len(s) + 14
    bw = w - lw - 390
    vmax = max((v for _, v in s), default=1.0) or 1.0
    body = [f'<text x="16" y="22" class="t1" style="font-weight:600;font-size:13px">What the warp schedulers wait on</text>'
            f'<text x="{w - 16}" y="22" text-anchor="end" class="t3">share of scheduler cycles, all SMs</text>']
    for i, (k, v) in enumerate(s):
        y = top + i * row
        meaning = analysis.STALL_MEANING.get(k, "")
        body.append(f'<text x="{lw - 10}" y="{y + 13}" text-anchor="end" class="t1">{_esc(k)}</text>')
        body.append(_rounded_hbar(lw, y + 2, max(bw * v / vmax, 1), 14, "bar", f"{k}: {100 * v:.1f}%. {meaning}"))
        body.append(f'<text x="{lw + bw * v / vmax + 6:.1f}" y="{y + 13}">{100 * v:.1f}%</text>')
        body.append(f'<text x="{lw + bw + 56}" y="{y + 13}" class="t3">{_esc(_short(meaning, 50))}</text>')
    return _svg(w, h, "".join(body), "Scheduler-cycle anatomy")


def fig_compare(runs: List[dict], w: int = 760) -> str:
    """Several systems: step time per GPU, split the same way as the single-run bar."""
    row, top = 44, 44
    lw = int(max(150, 16 + 7.2 * max(len(_short(r["label"], 28)) for r in runs) + 10))
    h = top + row * len(runs) + 44
    bw = w - lw - 110
    tmax = max(r["raw"]["makespan_us"] for r in runs) or 1.0
    body = [f'<text x="16" y="22" class="t1" style="font-weight:600;font-size:13px">Step time, split by where it goes</text>']
    for i, r in enumerate(runs):
        y = top + i * row
        mk = r["raw"]["makespan_us"]
        body.append(f'<text x="{lw - 10}" y="{y + 17}" text-anchor="end" class="t1">{_esc(_short(r["label"], 28))}</text>'
                    f'<text x="{lw - 10}" y="{y + 31}" text-anchor="end" class="t3">{r["raw"]["world"]} GPUs</text>')
        segs = [(k, lab, r["split"].get(k, 0.0)) for k, lab, _ in _SEGMENTS if r["split"].get(k, 0.0) > 0.0005]
        x = float(lw)
        total_w = bw * mk / tmax
        usable = total_w - 2.0 * (len(segs) - 1)
        for j, (k, lab, f) in enumerate(segs):
            sw = max(usable * f, 0.5)
            tip = f"{r['label']}: {lab} {_fmt_us(f * mk)} ({100 * f:.0f}%)"
            if j == len(segs) - 1:
                body.append(_rounded_hbar(x, y + 4, sw, 22, k, tip))
            else:
                body.append(f'<rect class="{k}" x="{x:.1f}" y="{y + 4}" width="{sw:.1f}" height="22" data-tip="{_esc(tip)}"/>')
            x += sw + 2
        body.append(f'<text x="{lw + total_w + 8:.1f}" y="{y + 20}" class="t1" style="font-weight:600">{_fmt_us(mk)}</text>')
    lgx, ly = lw, h - 12
    for k, lab, _ in _SEGMENTS:
        body.append(f'<rect class="{k}" x="{lgx}" y="{ly - 9}" width="10" height="10" rx="2"/>'
                    f'<text x="{lgx + 15}" y="{ly}" class="t3">{_esc(lab)}</text>')
        lgx += 30 + 6.6 * len(lab)
    return _svg(w, h, "".join(body), "Step time by system")


# ---------------------------------------------------------------- the page

_PAGE_CSS = """
:root{color-scheme:light;""" + _CSS_VARS_LIGHT + """}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;""" + _CSS_VARS_DARK + """}}
:root[data-theme="dark"]{color-scheme:dark;""" + _CSS_VARS_DARK + """}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 64px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-3);margin:0 0 6px}
h1{font-size:28px;line-height:1.2;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:18px;margin:40px 0 4px;letter-spacing:-.01em}
.sub{color:var(--text-2);margin:0 0 24px}
.note{color:var(--text-2);margin:4px 0 14px;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0 8px}
.tile{background:var(--surface-2);border-radius:10px;padding:14px 16px}
.tile .l{font-size:12px;color:var(--text-3)}
.tile .v{font-size:24px;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tile .d{font-size:12px;color:var(--text-2)}
.warn{border-left:3px solid var(--c-coll);background:var(--surface-2);padding:10px 14px;border-radius:6px;margin:12px 0;font-size:14px}
figure{margin:12px 0 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;word-break:break-all}
th{color:var(--text-3);font-weight:500}
details{margin-top:10px}summary{cursor:pointer;color:var(--text-2);font-size:14px}
.tablewrap{overflow-x:auto}
footer{margin-top:48px;color:var(--text-3);font-size:12px}
code{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:13px}
#tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);font-size:12px;line-height:1.4;
  padding:6px 9px;border-radius:6px;max-width:380px;white-space:pre-line;opacity:0;transition:opacity .08s;z-index:9}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
"""

_PAGE_JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');
 if(!t){tip.style.opacity=0;return;}tip.textContent=t.getAttribute('data-tip');tip.style.opacity=1;
 const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8),y=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8);
 tip.style.left=x+'px';tip.style.top=y+'px';});
"""


def _tiles(a: dict) -> str:
    r = a["raw"]
    b = r["bubbles"]
    mem = r.get("memory")
    tiles = [("Step time", _fmt_us(r["makespan_us"]), f"{r['world']} GPUs"),
             ("MFU", f"{100 * r['mfu']:.1f}%" if r.get("mfu") else "–", "model FLOPs / dense BF16 peak"),
             ("GPU idle", f"{100 * b['gpu_idle_fraction']:.1f}%", "between kernels"),
             ("Tensor cores idle", f"{100 * b['tensor_idle_fraction']:.1f}%", "inside kernels")]
    if mem:
        tiles.append(("Memory, fullest GPU", f"{mem['total_gb']:.0f} GB", f"of {mem['capacity_gb']:.0f} GB HBM"))
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="l">{_esc(l)}</div><div class="v">{_esc(v)}</div><div class="d">{_esc(d)}</div></div>'
        for l, v, d in tiles) + "</div>"


def _headline(a: dict) -> str:
    """One sentence that says what bounds the step."""
    sp = a["split"]
    names = {k: (lab if k in ("tensor", "hbm") else lab.lower()) for k, lab, _ in _SEGMENTS}
    top = max(sp, key=sp.get)
    ks = sorted(a["kernels"], key=lambda k: -k["share"])
    s = f"{100 * sp[top]:.0f}% of GPU time is {names[top]}"
    if top in ("tensor", "hbm", "latency"):
        s = f"{100 * sp[top]:.0f}% of GPU time is in {names[top]} kernels"
    if ks:
        k = ks[0]
        s += f"; the costliest kernel, <code>{_esc(_short(k['name'], 60))}</code>, takes {100 * k['share']:.0f}%"
        if k["roof_frac"] is not None:
            s += f" and reaches {100 * k['roof_frac']:.0f}% of its roof"
    return s + "."


def _kernel_table(a: dict) -> str:
    rows = []
    for k in sorted(a["kernels"], key=lambda k: -k["share"]):
        ai = "" if k["ai"] is None else f"{k['ai']:.1f}"
        roof = "" if k["roof_frac"] is None else f"{100 * k['roof_frac']:.0f}%"
        rows.append(f"<tr><td><span class='sw' style='background:var(--c-{k['bound']})'></span>{_esc(k['name'])}</td>"
                    f"<td>{k['count']}</td><td>{k['mean_us']:.2f}</td><td>{100 * k['share']:.2f}%</td>"
                    f"<td>{k['tflops']:,.0f}</td><td>{k['gbps']:,.0f}</td><td>{ai}</td><td>{roof}</td></tr>")
    return ("<details><summary>All kernels as a table</summary><div class='tablewrap'><table><tr><th>kernel</th><th>n</th>"
            "<th>mean µs</th><th>GPU time</th><th>TFLOPS</th><th>GB/s</th><th>FLOP/B</th><th>of roof</th></tr>"
            + "".join(rows) + "</table></div></details>")


def _coll_table(a: dict) -> str:
    from . import _bytes, _group_collectives
    cs = _group_collectives(a["raw"].get("collectives", []))
    if not cs:
        return ""
    rows = "".join(f"<tr><td>{_esc(c['kind'])} · {_esc(c['algo'])}</td><td>{_bytes(c['bytes'])}</td><td>{c['ranks']}</td>"
                   f"<td>{c['n']}</td><td>{c['mean_us']:.1f}</td><td>{c['busbw_gbps']:,.0f}</td></tr>" for c in cs[:20])
    return ("<h2>Collectives</h2><p class='note'>Bus bandwidth is NCCL's convention: algorithm bandwidth scaled so it "
            "is comparable to one link's speed.</p><div class='tablewrap'><table><tr><th>collective</th><th>size</th>"
            "<th>ranks</th><th>n</th><th>mean µs</th><th>busbw GB/s</th></tr>" + rows + "</table></div>")


def _analyze_all(runs, system, labels) -> List[dict]:
    if not isinstance(runs, (list, tuple)):
        runs = [runs]
    systems = list(system) if isinstance(system, (list, tuple)) else [system] * len(runs)
    labels = list(labels) if labels else [None] * len(runs)
    if not (len(systems) == len(labels) == len(runs)):
        raise ValueError(f"{len(runs)} runs, {len(systems)} systems and {len(labels)} labels")
    return [analyze(r, s, l) for r, s, l in zip(runs, systems, labels)]


def page(runs: Union[dict, Sequence], system=None, *, title: Optional[str] = None, note: Optional[str] = None,
         labels: Optional[Sequence[str]] = None) -> str:
    """The HTML report for one run, or for several side by side (systems, dtypes, layouts: name them with
    ``labels``; by default each is named by its system)."""
    an = _analyze_all(runs, system, labels)
    a0 = an[0]
    wl = a0["raw"]["workload"]
    parts = [f"<p class='eyebrow'>rackscale run report</p><h1>{_esc(title or wl)}</h1>"]
    if len(an) == 1:
        parts.append(f"<p class='sub'>{_esc(a0['raw']['system'])}, {a0['raw']['world']} GPUs"
                     f"{'' if title is None else ' · ' + _esc(wl)}</p>")
    else:
        same = len({a["raw"]["workload"] for a in an}) == 1
        parts.append(f"<p class='sub'>{_esc(wl) + ' on ' if same else ''}{len(an)} runs side by side</p>")
    if note:
        parts.append("".join(f"<p class='note'>{_esc(p)}</p>" for p in note.split("\n\n")))
    if len(an) > 1:
        rows = []
        for a in an:
            mfu = f"{100 * a['raw']['mfu']:.1f}%" if a["raw"].get("mfu") else ""
            rows.append(f"<tr><td>{_esc(a['label'])}</td><td>{a['raw']['world']}</td><td>{_fmt_us(a['raw']['makespan_us'])}</td>"
                        f"<td>{a0['raw']['makespan_us'] / a['raw']['makespan_us']:.2f}×</td><td>{mfu}</td>"
                        f"<td>{100 * a['raw']['bubbles']['gpu_idle_fraction']:.1f}%</td></tr>")
        rows = "".join(rows)
        parts.append("<h2>Side by side</h2><p class='note'>Speedup is against the first run. Bars are drawn to the "
                     "same time scale.</p>" + f"<figure>{fig_compare(an)}</figure>"
                     "<div class='tablewrap'><table><tr><th>run</th><th>GPUs</th><th>step</th><th>speedup</th>"
                     "<th>MFU</th><th>GPU idle</th></tr>" + rows + "</table></div>")
    for a in an:
        if len(an) > 1:
            parts.append(f"<h2 style='margin-top:56px;font-size:22px'>{_esc(a['label'])}</h2>")
        parts.append(_tiles(a))
        for wmsg in a["raw"].get("warnings", []):
            parts.append(f"<div class='warn'>{_esc(wmsg)}</div>")
        parts.append(f"<p class='sub'>{_headline(a)}</p>")
        parts.append(f"<figure>{fig_time_split(a)}</figure>")
        parts.append("<h2>Kernels on the roofline</h2><p class='note'>Each circle is one kernel, sized by its share of GPU "
                     "time. Left of the ridge a kernel can at best stream HBM; right of it, saturate the tensor cores. "
                     "Distance below the roof is efficiency left on the table. Hover for numbers.</p>"
                     f"<figure>{fig_roofline(a)}</figure>")
        parts.append(f"<h2>Where the time goes, kernel by kernel</h2><figure>{fig_top_kernels(a)}</figure>{_kernel_table(a)}")
        parts.append("<h2>What the warp schedulers wait on</h2><p class='note'>Every scheduler cycle of every SM, "
                     "attributed the way Nsight Compute samples warp state. <code>async_wait</code> is warps parked on "
                     "mbarriers by design in warp-specialised kernels; the bubbles that matter are usually the next ones."
                     f"</p><figure>{fig_stalls(a)}</figure>")
        parts.append(_coll_table(a))
    from . import __version__
    sim = a0["raw"].get("sim") or {}
    parts.append(f"<footer>rackscale {__version__} · simulated in {sim.get('wallclock_s', 0):.1f} s · kernel classes: "
                 + "; ".join(f"{lab.lower()}: {desc}" for _, lab, desc in CLASSES)
                 + ". Throughput figures use the preset clock; reports carry the power-capped stretch in step time.</footer>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{_esc(title or 'rackscale report')}</title><style>{_PAGE_CSS}</style></head><body>"
            f"<main>{''.join(parts)}</main><div id='tip'></div><script>{_PAGE_JS}</script></body></html>")


def write(runs, path: str, system=None, **kw) -> str:
    """Write the report page; returns the path."""
    with open(path, "w") as f:
        f.write(page(runs, system, **kw))
    return path


def figures(runs, out_dir: str, system=None, labels: Optional[Sequence[str]] = None) -> List[str]:
    """Write each figure as a standalone SVG (light and dark aware) for articles and slides."""
    an = _analyze_all(runs, system, labels)
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def put(name, svg):
        p = os.path.join(out_dir, name)
        with open(p, "w") as f:
            f.write(svg)
        out.append(p)

    if len(an) > 1:
        put("compare_step.svg", fig_compare(an))
    for i, a in enumerate(an):
        tag = "" if len(an) == 1 else f"_{i}_" + "_".join(re.findall(r"[a-z0-9]+", a["label"].lower()))
        put(f"time_split{tag}.svg", fig_time_split(a))
        put(f"roofline{tag}.svg", fig_roofline(a))
        put(f"top_kernels{tag}.svg", fig_top_kernels(a))
        put(f"stalls{tag}.svg", fig_stalls(a))
    return out
