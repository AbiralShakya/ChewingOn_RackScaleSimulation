#!/usr/bin/env python3
"""Build the study write-up (writeup.html) from results/study.json and figures/*.png."""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
st = json.load(open(os.path.join(HERE, "results", "study.json")))


def img(name, alt):
    p = os.path.join(HERE, "figures", name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure class="plate"><img src="data:image/png;base64,{b}" alt="{alt}"></figure>'


def pct(x):
    return f"{100 * x:+.1f}%"


def fmt_m(c):
    return f"{c / 1e6:.2f} M" if c >= 1e6 else f"{c / 1e3:.0f} k"


val = {(r["gpu"], r["impl"]): r for r in st.get("validation", [])}
GPU = {"a100": "A100 80GB PCIe", "a100sxm": "A100 SXM4 80GB", "b300": "B300 SXM6"}
IMPL = {"fa2": "FlashAttention-2", "fa4": "FlashAttention-4"}
ROLE = {("a100", "fa2"): "mma.sync (HMMA), cp.async", ("a100", "fa4"): "CuTe DSL Sm80 path, HMMA",
        ("b300", "fa2"): "legacy mma.sync on Blackwell", ("b300", "fa4"): "2-CTA tcgen05, TMEM, 6 warp roles"}

rows = []
for k in [("a100", "fa2"), ("a100", "fa4"), ("b300", "fa2"), ("b300", "fa4")]:
    r = val.get(k)
    if not r or "ncu" not in r:
        continue
    n, s = r["ncu"], r["sim"]
    rows.append(f"<tr><td>{GPU[k[0]]}</td><td>{IMPL[k[1]]}<span class='sub'>{ROLE[k]}</span></td>"
                f"<td class='num'>{n['us']:.0f}</td><td class='num'>{fmt_m(n['cycles'])}</td>"
                f"<td class='num'>{fmt_m(s['cycles'])}</td><td class='num err'>{pct(r['cycle_error'])}</td>"
                f"<td class='num'>{pct(r['inst_error'])}</td>"
                f"<td class='num'>{(s['dram_read_bytes'] or 0) / 1e6:.0f} / {(n['dram_read_bytes'] or 0) / 1e6:.0f}</td></tr>")
val_rows = "\n".join(rows)

cp_rows = []
for r in st.get("validation_cp", []):
    if "ncu" not in r:
        continue
    sh = r["shape"].replace("cp_c", "chunk ").replace("_kv", ", KV heads ")
    ok = abs(r["cycle_error"]) <= 0.10
    cp_rows.append(f"<tr><td>{GPU[r['gpu']]}</td><td>{IMPL[r['impl']]}</td><td>{sh}</td>"
                   f"<td class='num'>{r['ncu']['us']:.0f}</td><td class='num {'err' if ok else 'bad'}'>{pct(r['cycle_error'])}</td>"
                   f"<td class='num'>{pct(r['inst_error'])}</td></tr>")
cp_rows = "\n".join(cp_rows)

wi = {(r["gpu"], r["impl"], r["variant"]): r for r in st.get("whatif", [])}


def w(g, i, v):
    r = wi.get((g, i, v))
    return f"{r['speedup_vs_baseline']:.2f}×" if r and r.get("speedup_vs_baseline") else "n/a"


ring = st.get("scale", [])
ring_rows = []
SYS = {"gb300_nvl72": "GB300 NVL72", "b300_9x8_ib": "9× HGX B300, 400G IB", "a100_9x8_ib": "9× DGX A100, 200G IB"}
for r in sorted(ring, key=lambda r: (r["system"], r["hkv"], r["n"], r["chunk"])):
    if r.get("eff_alone") is None or r["n"] != 72:
        continue
    ring_rows.append(f"<tr><td>{SYS[r['system']]}</td><td class='num'>{r['n']}</td><td class='num'>{r['context'] // 1024}k</td>"
                     f"<td class='num'>{r['hkv']}</td><td class='num'>{r['kernel_alone_us']:.0f}</td><td class='num'>{r['kernel_us']:.0f}</td>"
                     f"<td class='num'>{(r['sendrecv_us'] or 0):.0f}</td><td class='num'>{r['makespan_us'] / r['n']:.0f}</td>"
                     f"<td class='num'>{100 * r['eff_alone']:.0f}%</td><td class='num'>{r['tflops_per_gpu']:.0f}</td></tr>")
ring_rows = "\n".join(ring_rows)

def R(system, hkv, chunk, key="eff_alone"):
    r = next((r for r in ring if r["system"] == system and r["hkv"] == hkv and r["chunk"] == chunk and r["n"] == 72), None)
    return r[key] if r else float("nan")


if ring:
    finding = (
        "<p><b>B300 turns a compute-bound ring into a network-bound one, unless the ring stays inside NVLink.</b> "
        f"With grouped-query attention at 72k context, the A100 ring over InfiniBand is {100 * R('a100_9x8_ib', 8, 1024):.0f}% "
        f"efficient: its {R('a100_9x8_ib', 8, 1024, 'kernel_alone_us'):.0f} µs FA2 step hides each KV transfer. "
        f"The same ring on B300s over 400G InfiniBand drops to {100 * R('b300_9x8_ib', 8, 1024):.0f}%, because FA4's step "
        f"is {R('b300_9x8_ib', 8, 1024, 'kernel_alone_us'):.0f} µs while the slowest inter-node hop takes "
        f"{R('b300_9x8_ib', 8, 1024, 'sendrecv_us'):.0f} µs. On GB300 NVL72 every hop is NVLink, and the ring "
        f"is back to {100 * R('gb300_nvl72', 8, 1024):.0f}% at 72k and {100 * R('gb300_nvl72', 8, 2048):.0f}% at 144k.</p>"
        "<p><b>Without grouped-query attention, even NVL72 cannot hide the ring.</b> With 64 KV heads each step moves "
        f"8× more K and V: {R('gb300_nvl72', 64, 1024, 'sendrecv_us'):.0f} µs per hop against a "
        f"{R('gb300_nvl72', 64, 1024, 'kernel_alone_us'):.0f} µs kernel at 72k. Efficiency falls to "
        f"{100 * R('gb300_nvl72', 64, 1024):.0f}% at 72k and {100 * R('gb300_nvl72', 64, 2048):.0f}% at 144k. "
        "A second effect appears in the simulator: FA4 launches exactly one persistent CTA per SM. The send and receive "
        "kernels take SMs during the step, so a few of FA4's CTAs wait for a second wave. The kernel then runs "
        f"{R('gb300_nvl72', 64, 2048, 'kernel_us'):.0f} µs inside the ring against "
        f"{R('gb300_nvl72', 64, 2048, 'kernel_alone_us'):.0f} µs alone. Copy-engine or NVSHMEM transfers, or a grid "
        "sized to the free SMs, would remove that part.</p>")
else:
    finding = ""

TEMPLATE = open(os.path.join(HERE, "writeup_template.html")).read()
html = (TEMPLATE
        .replace("{{VAL_ROWS}}", val_rows)
        .replace("{{CP_ROWS}}", cp_rows)
        .replace("{{RING_ROWS}}", ring_rows)
        .replace("{{RING_FINDING}}", finding)
        .replace("{{FIG_THROUGHPUT}}", img("throughput.png", "TFLOP/s by sequence length on A100 and B300"))
        .replace("{{FIG_VALIDATION}}", img("validation.png", "Simulated vs measured cycles"))
        .replace("{{FIG_STALLS}}", img("stalls.png", "Stall reasons, hardware vs simulator"))
        .replace("{{FIG_ROLES}}", img("fa4_roles.png", "FA4 warp roles on B300"))
        .replace("{{FIG_WHATIF}}", img("whatif.png", "What-if speedups"))
        .replace("{{FIG_RING}}", img("ring.png", "Ring attention efficiency at 72 GPUs")))
for (g, i, v) in list(wi):
    html = html.replace("{{W:%s:%s:%s}}" % (g, i, v), w(g, i, v))
open(os.path.join(HERE, "writeup.html"), "w").write(html)
print("wrote writeup.html", len(html) // 1024, "KB")
