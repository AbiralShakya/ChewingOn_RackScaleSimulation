"""Figures for the FlashAttention study (results/study.json -> figures/*.png)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One categorical palette, one ink color, light grid; same look in every figure.
C = {"naive": "#9aa0a6", "sdpa_flash": "#c58af9", "sdpa_cudnn": "#f29900", "fa2": "#1a73e8", "fa4": "#d93025",
     "sim": "#1a73e8", "ncu": "#202124"}
LABEL = {"naive": "naive (3 kernels)", "sdpa_flash": "PyTorch SDPA flash", "sdpa_cudnn": "cuDNN SDPA",
         "fa2": "FlashAttention-2", "fa4": "FlashAttention-4 (CuTe)"}
STALL_COLORS = {
    "wait": "#8ab4f8", "math_pipe_throttle": "#1a73e8", "short_scoreboard": "#fbbc04", "long_scoreboard": "#f29900",
    "barrier": "#a142f4", "mbarrier_wait": "#d93025", "async_wait": "#d93025", "tma_wait": "#f28b82",
    "tensor_wait": "#b31412", "mio_throttle": "#34a853", "lg_throttle": "#0d652d", "branch_resolving": "#5f6368",
    "membar": "#3c4043", "dispatch_stall": "#bdc1c6", "sleeping": "#e8710a", "no_instruction": "#dadce0",
    "other": "#e8eaed",
}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8eaed", "grid.linewidth": 0.6, "axes.edgecolor": "#5f6368",
        "axes.labelcolor": "#202124", "xtick.color": "#5f6368", "ytick.color": "#5f6368",
        "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 10,
    })


def fig_throughput(study, out):
    t = study.get("timing", {})
    panels = [(k, v) for k, v in sorted(t.items()) if any(r.get("file", "").startswith("timing") for r in v)]
    def pick(v):
        # Prefer the back-to-back sweep (timing2) where a GPU has one.
        files = {r.get("file") for r in v}
        f = "timing2.jsonl" if "timing2.jsonl" in files else "timing.jsonl"
        return [r for r in v if r.get("file") == f and not r.get("causal") and r.get("h") == 16]
    panels = [(k, pick(v)) for k, v in panels]
    panels = [(k, v) for k, v in panels if v]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), sharey=False)
    axes = axes if len(panels) > 1 else [axes]
    for ax, (gpu, rows) in zip(axes, panels):
        for impl in ("naive", "sdpa_flash", "sdpa_cudnn", "fa2", "fa4"):
            pts = {}
            for r in rows:
                if r["impl"] == impl:
                    pts[r["s"]] = r["tflops"]  # later sweeps overwrite earlier ones
            if pts:
                xs = sorted(pts)
                ax.plot(xs, [pts[x] for x in xs], marker="o", ms=3, lw=1.6, color=C[impl], label=LABEL[impl])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 2048, 4096, 8192, 16384], ["1k", "2k", "4k", "8k", "16k"])
        ax.set_title(rows[0].get("gpu", gpu).replace("NVIDIA ", ""))
    axes[0].set_ylabel("TFLOP/s (forward, non-causal)")
    fig.supxlabel("sequence length (batch x seq = 16k tokens, 16 heads, head dim 128, BF16)", fontsize=9)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "throughput.png"), bbox_inches="tight")
    plt.close(fig)


def fig_validation(study, out):
    rows = [r for r in study.get("validation", []) if "ncu" in r] + study.get("validation_extra", [])
    if not rows:
        return
    labels = [f"{r['gpu']} {r['impl']}\n{r['shape']}" for r in rows]
    sim = [r["sim"]["cycles"] / 1e6 for r in rows]
    hw = [r["ncu"]["cycles"] / 1e6 for r in rows]
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(rows) + 2), 3.2))
    x = range(len(rows))
    w = 0.38
    ax.bar([i - w / 2 for i in x], hw, w, color=C["ncu"], label="Nsight Compute (hardware)")
    ax.bar([i + w / 2 for i in x], sim, w, color=C["sim"], label="rsim (recorded SASS paths)")
    for i, r in enumerate(rows):
        ax.text(i + w / 2, sim[i], f"{100 * r['cycle_error']:+.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("SM cycles (millions)")

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "validation.png"), bbox_inches="tight")
    plt.close(fig)


# Nsight counts a warp blocked in an mbarrier try_wait as long_scoreboard; the simulator
# names what it waits for (async_wait / tma_wait / tensor_wait). One bucket for both.
SIM2NCU = {"async_wait": "long_scoreboard", "tma_wait": "long_scoreboard", "tensor_wait": "long_scoreboard",
           "tensor_throttle": "math_pipe_throttle"}
STALL_LABEL = {"long_scoreboard": "long scoreboard / mbarrier wait"}
SKIP = {"selected", "not_selected", "drain", "idle"}


def stall_shares(d, mapping=None):
    agg = {}
    for k, v in d.items():
        if k in SKIP or not isinstance(v, (int, float)):
            continue
        k = (mapping or {}).get(k, k)
        agg[k] = agg.get(k, 0) + v
    tot = sum(agg.values()) or 1
    return {k: v / tot for k, v in agg.items()}


def fig_stalls(study, out):
    rows = [r for r in study.get("validation", []) if "ncu" in r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(rows) + 2), 3.4))
    keys = list(STALL_COLORS)
    pos = 0
    ticks, labels = [], []
    for r in rows:
        for src, d in (("ncu", stall_shares(r["ncu"]["stalls_per_issue"])),
                       ("sim", stall_shares(r["sim"]["warp_stalls_per_issue"], SIM2NCU))):
            bottom = 0
            other = sum(v for k, v in d.items() if k not in keys)
            d = dict(d, other=d.get("other", 0) + other)
            for k in keys:
                v = d.get(k, 0)
                if v > 0:
                    ax.bar(pos, v, 0.8, bottom=bottom, color=STALL_COLORS[k],
                           label=STALL_LABEL.get(k, k) if k not in labels else None)
                    if k not in labels:
                        labels.append(k)
                    bottom += v
            ticks.append((pos, f"{src}\n{r['gpu']} {r['impl']}"))
            pos += 1
        pos += 0.6
    ax.set_xticks([p for p, _ in ticks])
    ax.set_xticklabels([t for _, t in ticks], fontsize=7)
    ax.set_ylabel("share of stalled warp-cycles")

    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stalls.png"), bbox_inches="tight")
    plt.close(fig)


FA4_ROLES = [("softmax WG0 (warps 0-3)", [0, 1, 2, 3]), ("softmax WG1 (warps 4-7)", [4, 5, 6, 7]),
             ("correction WG (warps 8-11)", [8, 9, 10, 11]), ("MMA issue (warp 12)", [12]),
             ("epilogue / TMA store (warp 13)", [13]), ("TMA load (warp 14)", [14])]


def fig_roles(study, out):
    r = next((r for r in study.get("validation", []) if r["gpu"] == "b300" and r["impl"] == "fa4"), None)
    if not r or not r["sim"].get("warp_by_index"):
        return
    wb = r["sim"]["warp_by_index"]
    keys = ["selected", "wait", "math_pipe_throttle", "short_scoreboard", "tensor_wait", "async_wait", "tma_wait",
            "barrier", "mio_throttle", "membar", "drain"]
    lab = {"selected": "issuing", "wait": "fixed-latency dependency", "math_pipe_throttle": "math pipe busy",
           "short_scoreboard": "short scoreboard", "tensor_wait": "mbarrier: tensor core (S ready / O done)",
           "async_wait": "mbarrier: other warps", "tma_wait": "mbarrier: TMA data", "barrier": "named barrier",
           "mio_throttle": "MIO throttle", "membar": "fence", "drain": "exited"}
    col = dict(STALL_COLORS, selected="#34a853", drain="#e8eaed", tensor_wait="#b31412", async_wait="#f28b82",
               tma_wait="#fbbc04", short_scoreboard="#fdd663")
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    used = set()
    for gi, (name, g) in enumerate(FA4_ROLES):
        tot = {k: sum(wb[i].get(k, 0) for i in g if i < len(wb)) for k in keys}
        s_ = sum(tot.values()) or 1
        left = 0
        for k in keys:
            v = tot[k] / s_
            if v > 0.002:
                ax.barh(gi, v, left=left, color=col[k], label=lab[k] if k not in used else None)
                used.add(k)
                left += v
    ax.set_yticks(range(len(FA4_ROLES)))
    ax.set_yticklabels([n for n, _ in FA4_ROLES], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the role's resident warp-cycles (all CTAs; CTA-pair peers included)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fa4_roles.png"), bbox_inches="tight")
    plt.close(fig)


def fig_whatif(study, out):
    rows = [r for r in study.get("whatif", []) if r["variant"] != "baseline" and r["speedup_vs_baseline"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(rows) + 1.2))
    vl = {"ex2_half_rate": "MUFU.EX2 at half rate (B200-like)", "ex2_double_rate": "MUFU.EX2 at 2x",
          "tensor_1.5x": "tensor core 1.5x", "tensor_2x": "tensor core 2x",
          "tensor_2x+ex2_2x": "tensor 2x + EX2 2x", "tensor_2x+fma2_2x": "tensor 2x + FMA2 2x",
          "tensor_2x+ex2_2x+fma2_2x": "tensor 2x + EX2 2x + FMA2 2x", "fma2_half_rate": "FFMA2/FADD2 at half rate",
          "hbm_half": "HBM bandwidth halved", "no_hmma_fp32_coupling": "no HMMA/FP32-pipe coupling"}
    gl = {"a100": "A100", "b300": "B300"}
    labels = [f"{gl.get(r['gpu'], r['gpu'])} {r['impl'].upper()}: {vl.get(r['variant'], r['variant'])}" for r in rows]
    vals = [r["speedup_vs_baseline"] for r in rows]
    ax.barh(range(len(rows)), [v - 1 for v in vals], left=1, color=["#1a73e8" if v >= 1 else "#d93025" for v in vals])
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}x", va="center", fontsize=7)
    ax.axvline(1, color="#202124", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("kernel speed vs the measured hardware (1.0 = calibrated)")

    fig.tight_layout()
    fig.savefig(os.path.join(out, "whatif.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ring(study, out):
    rows = [r for r in study.get("scale", []) if r["n"] == 72 and r.get("eff_alone") is not None]
    if not rows:
        return
    systems = [("gb300_nvl72", "GB300 NVL72", "#0f6b5c"), ("b300_9x8_ib", "9× HGX B300 (IB)", "#e8a33d"),
               ("a100_9x8_ib", "9× DGX A100 (IB)", "#6f7f8f")]
    groups = [(8, 1024, "GQA, 72k"), (8, 2048, "GQA, 144k"), (64, 1024, "MHA, 72k"), (64, 2048, "MHA, 144k")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.26
    for si, (s, lab, col) in enumerate(systems):
        xs, ys = [], []
        for gi, (hkv, c, _) in enumerate(groups):
            r = next((r for r in rows if r["system"] == s and r["hkv"] == hkv and r["chunk"] == c), None)
            if r:
                xs.append(gi + (si - 1) * w)
                ys.append(100 * r["eff_alone"])
        bars = ax.bar(xs, ys, w * 0.92, color=col, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.5, f"{y:.0f}", ha="center", fontsize=7, color="#202124")
    ax.set_xticks(range(len(groups)), [g[2] for g in groups])
    ax.set_ylim(0, 110)
    ax.set_ylabel("ring efficiency (%)\n72 x standalone kernel / ring time")
    ax.set_xlabel("Llama-3-70B attention (64 query heads), 72-way context parallel; context = 72 x chunk")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ring.png"), bbox_inches="tight")
    plt.close(fig)


def make_figures(study, out):
    os.makedirs(out, exist_ok=True)
    style()
    for f in (fig_throughput, fig_validation, fig_stalls, fig_roles, fig_whatif, fig_ring):
        try:
            f(study, out)
        except Exception as e:  # keep the other figures if one input is missing
            print(f"{f.__name__}: {e}")
