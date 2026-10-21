"""Render a per-SM CTA timeline PNG from a `--per-sm` Perfetto trace.

    python docs/img/plot_per_sm.py out/deepseek_v3_ep8_b200_per_sm.json docs/img/deepseek_v3_ep8_b200_per_sm.png

Needs matplotlib. Plots GPU0 (pid 1): stream track, one row per SM, tensor and DRAM counters.
"""
import gzip, json, sys, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
t = json.load(gzip.open(sys.argv[1]) if sys.argv[1].endswith(".gz") else open(sys.argv[1])); ev = t["traceEvents"] if isinstance(t, dict) else t
out = sys.argv[2]
PID = 1
kern = sorted([e for e in ev if e.get("cat") == "kernel" and e["pid"] == PID], key=lambda e: e["ts"])
ctas = [e for e in ev if e.get("cat") == "cta" and e["pid"] == PID]
coll = [e for e in ev if e.get("cat") == "collective"]
cnt = collections.defaultdict(list)
for e in ev:
    if e.get("ph") == "C" and e["pid"] == PID:
        for k, v in e["args"].items(): cnt[k].append((e["ts"], v))
def group(n):
    n = n.lower()
    if "attn" in n or "fmha" in n or "mla" in n or "attention" in n: return "attention"
    if "expert" in n or "moe" in n or "grouped" in n or "router" in n or "topk" in n: return "MoE / router"
    if "dispatch" in n or "combine" in n or "all" in n: return "comm"
    if "norm" in n or "rope" in n or "act" in n or "silu" in n or "add" in n: return "elementwise / norm"
    if "lm_head" in n or "head" in n: return "LM head"
    return "dense GEMM"
pal = {"dense GEMM": "#3b6fb6", "attention": "#d9822b", "MoE / router": "#2a9d8f", "elementwise / norm": "#9aa0a6",
       "LM head": "#8e5ea2", "comm": "#c0392b"}
# map CTA -> kernel by time containment on stream
starts = [k["ts"] for k in kern]
import bisect
def kgroup(ts):
    i = bisect.bisect_right(starts, ts) - 1
    while i >= 0:
        k = kern[i]
        if k["ts"] <= ts <= k["ts"] + k["dur"]: return group(k["name"])
        i -= 1
        if i >= 0 and kern[i]["ts"] + kern[i]["dur"] < ts - 50: break
    return "comm"
T1 = max(k["ts"] + k["dur"] for k in kern)
fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [0.6, 6, 1.2, 1.2]})
ax = axes[0]
for k in kern:
    ax.broken_barh([(k["ts"], k["dur"])], (0, 1), color=pal[group(k["name"])], lw=0)
for c in coll:
    ax.broken_barh([(c["ts"], c["dur"])], (1.1, 0.7), color=pal["comm"], lw=0)
ax.set_yticks([]); ax.set_ylabel("stream", rotation=0, ha="right", va="center")
ax = axes[1]
bysm = collections.defaultdict(lambda: collections.defaultdict(list))
for c in ctas:
    bysm[c["tid"] - 1000][kgroup(c["ts"] + c["dur"] / 2)].append((c["ts"], c["dur"]))
for sm, g in bysm.items():
    for name, spans in g.items():
        ax.broken_barh(spans, (sm, 0.9), color=pal[name], lw=0)
ax.set_ylim(148, 0); ax.set_ylabel("SM id (one row per SM,\none bar per CTA)")
for ax, key, lab, col in [(axes[2], "tensor", "tensor util", "#2a9d8f"), (axes[3], "GB/s", "DRAM GB/s", "#3b6fb6")]:
    pts = sorted(cnt[key]); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.step(xs, ys, where="post", color=col, lw=0.8); ax.fill_between(xs, ys, step="post", color=col, alpha=0.25)
    ax.set_ylabel(lab)
axes[3].axhline(7700, ls="--", lw=0.8, color="#555"); axes[3].text(5, 7900, "B200 HBM peak 7.7 TB/s", fontsize=8, color="#555")
axes[3].set_xlabel("time (us)"); axes[3].set_xlim(0, T1)
axes[0].legend(handles=[Patch(color=v, label=k) for k, v in pal.items()], ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=9)
fig.suptitle("DeepSeek-V3 decode, 4 layers, HGX B200 x8, EP8, batch 32/GPU, ctx 4096: GPU0, per-SM CTA timeline (simulated)", y=0.995, fontsize=11)
for a in axes: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=150)
print("ok", len(kern), len(ctas), len(coll), T1, collections.Counter(group(k["name"]) for k in kern), set(k["name"] for k in kern))
