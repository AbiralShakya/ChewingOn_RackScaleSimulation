"""Minimal, dependency-free SVG charts for the write-up.

Colors are CSS custom properties (``var(--s1)`` ...), so the charts follow the
page's light/dark tokens. Every mark carries a ``<title>`` tooltip.
"""
from __future__ import annotations

import math
from html import escape


def _ticks_log(lo, hi):
    out = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.0001:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo <= v <= hi * 1.0001:
                out.append(v)
        e += 1
    return out


def _ticks_lin(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n)) if span > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    v = math.ceil(lo / step) * step
    out = []
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def fmt(v):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


class Axis:
    def __init__(self, lo, hi, log, px0, px1):
        self.lo, self.hi, self.log, self.p0, self.p1 = lo, hi, log, px0, px1

    def __call__(self, v):
        if self.log:
            t = (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        else:
            t = (v - self.lo) / (self.hi - self.lo)
        return self.p0 + t * (self.p1 - self.p0)

    def ticks(self):
        return _ticks_log(self.lo, self.hi) if self.log else _ticks_lin(self.lo, self.hi)


def line_chart(series, *, xlabel, ylabel, xlog=False, ylog=False, w=560, h=300, xticks=None, xfmt=fmt, yfmt=fmt,
               ylim=None, markers=(), aria=""):
    """series: list of dict(name, color (css var), points [(x, y, tooltip)], dash=False)."""
    pts = [p for s in series for p in s["points"]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    L, R, T, B = 58, 18, 14, 44
    xa = Axis(min(xs), max(xs), xlog, L, w - R)
    ylo, yhi = ylim or ((min(ys) * 0.8, max(ys) * 1.25) if ylog else (0, max(ys) * 1.1))
    ya = Axis(ylo, yhi, ylog, h - B, T)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(aria)}" class="chart">']
    for t in ya.ticks():
        y = ya(t)
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{yfmt(t)}</text>')
    for t in (xticks or xa.ticks()):
        x = xa(t)
        o.append(f'<text x="{x:.1f}" y="{h - B + 18}" class="tick" text-anchor="middle">{xfmt(t)}</text>')
    o.append(f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" class="axis"/>')
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 6}" class="alabel" text-anchor="middle">{escape(xlabel)}</text>')
    o.append(f'<text x="14" y="{(T + h - B) / 2}" class="alabel" text-anchor="middle" '
             f'transform="rotate(-90 14 {(T + h - B) / 2})">{escape(ylabel)}</text>')
    for mk in markers:
        x = xa(mk["x"])
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{h - B}" class="mark-line"/>'
                 f'<text x="{x + 5:.1f}" y="{T + 12}" class="note">{escape(mk["label"])}</text>')
    for s in series:
        p = s["points"]
        d = " ".join(f"{'M' if i == 0 else 'L'}{xa(x):.1f},{ya(y):.1f}" for i, (x, y, _) in enumerate(p))
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        o.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="2"{dash}/>')
        for x, y, tip in p:
            o.append(f'<circle cx="{xa(x):.1f}" cy="{ya(y):.1f}" r="4" fill="{s["color"]}" class="dot">'
                     f'<title>{escape(s["name"])}: {escape(tip)}</title></circle>')
        x, y, _ = p[-1]
        if s.get("label", True):
            o.append(f'<text x="{min(xa(x) + 6, w - R - 4):.1f}" y="{ya(y) - 8:.1f}" class="dlabel" '
                     f'text-anchor="end">{escape(s["name"])}</text>')
    o.append("</svg>")
    return "".join(o)


def scatter(points, *, xlabel, ylabel, w=420, h=380, lo=None, hi=None, band=0.25, aria=""):
    """Log-log measured vs simulated with y = x and a +/-band corridor. points: [(x, y, tip, colorvar)]."""
    xs = [p[0] for p in points] + [p[1] for p in points]
    lo = lo or min(xs) * 0.7
    hi = hi or max(xs) * 1.4
    L, R, T, B = 58, 16, 14, 44
    xa = Axis(lo, hi, True, L, w - R)
    ya = Axis(lo, hi, True, h - B, T)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(aria)}" class="chart">']
    for t in xa.ticks():
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{ya(t):.1f}" y2="{ya(t):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{ya(t) + 4:.1f}" class="tick" text-anchor="end">{fmt(t)}</text>'
                 f'<text x="{xa(t):.1f}" y="{h - B + 18}" class="tick" text-anchor="middle">{fmt(t)}</text>')
    band_pts = [(lo, lo * (1 + band)), (hi / (1 + band), hi), (hi, hi / (1 + band)), (lo * (1 + band), lo)]
    poly = " ".join(f"{xa(min(max(a, lo), hi)):.1f},{ya(min(max(b, lo), hi)):.1f}" for a, b in band_pts)
    o.append(f'<polygon points="{poly}" class="band"/>')
    o.append(f'<line x1="{xa(lo):.1f}" y1="{ya(lo):.1f}" x2="{xa(hi):.1f}" y2="{ya(hi):.1f}" class="diag"/>')
    o.append(f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" class="axis"/>')
    for x, y, tip, col in points:
        o.append(f'<circle cx="{xa(x):.1f}" cy="{ya(y):.1f}" r="4.5" fill="{col}" class="dot">'
                 f'<title>{escape(tip)}</title></circle>')
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 6}" class="alabel" text-anchor="middle">{escape(xlabel)}</text>')
    o.append(f'<text x="14" y="{(T + h - B) / 2}" class="alabel" text-anchor="middle" '
             f'transform="rotate(-90 14 {(T + h - B) / 2})">{escape(ylabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def stacked_bars(bars, keys, colors, *, ylabel, w=560, h=300, yfmt=fmt, aria="", horizontal_labels=True):
    """bars: [(label, {key: value})]; keys in stack order (bottom first)."""
    L, R, T, B = 58, 12, 14, 52
    tot = max(sum(v.get(k, 0) for k in keys) for _, v in bars) * 1.1
    ya = Axis(0, tot, False, h - B, T)
    n = len(bars)
    slot = (w - L - R) / n
    bw = min(56, slot * 0.6)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(aria)}" class="chart">']
    for t in ya.ticks():
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{ya(t):.1f}" y2="{ya(t):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{ya(t) + 4:.1f}" class="tick" text-anchor="end">{yfmt(t)}</text>')
    for i, (lab, vals) in enumerate(bars):
        x = L + slot * i + (slot - bw) / 2
        acc = 0.0
        for k, c in zip(keys, colors):
            v = vals.get(k, 0.0)
            if v <= 0:
                continue
            y0, y1 = ya(acc), ya(acc + v)
            hgt = max(0.0, y0 - y1 - 2)
            o.append(f'<rect x="{x:.1f}" y="{y1 + 1:.1f}" width="{bw:.1f}" height="{hgt:.1f}" rx="2" fill="{c}" class="dot">'
                     f'<title>{escape(lab)} / {escape(k)}: {yfmt(v)}</title></rect>')
            acc += v
        o.append(f'<text x="{x + bw / 2:.1f}" y="{ya(acc) - 6:.1f}" class="dlabel" text-anchor="middle">{yfmt(acc)}</text>')
        parts = lab.split("|")
        for j, ptxt in enumerate(parts):
            o.append(f'<text x="{x + bw / 2:.1f}" y="{h - B + 16 + 13 * j}" class="tick" text-anchor="middle">{escape(ptxt)}</text>')
    o.append(f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" class="axis"/>')
    o.append(f'<text x="14" y="{(T + h - B) / 2}" class="alabel" text-anchor="middle" '
             f'transform="rotate(-90 14 {(T + h - B) / 2})">{escape(ylabel)}</text>')
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{escape(n)}</span>' for n, c in items) + "</div>"
