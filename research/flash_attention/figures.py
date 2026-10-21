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
    panels = [(k, [r for r in v if r.get("file") in ("timing2.jsonl", "timing.jsonl") and not r.get("causal")
                   and r.get("h") == 16]) for k, v in panels]
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
        ax.set_xlabel("sequence length (batch x seq = 16k tokens, 16 heads, d=128)")
        ax.set_ylabel("TFLOP/s (forward, non-causal)")
        ax.set_title(rows[0].get("gpu", gpu))
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
    ax.set_title("Kernel cycles: simulator vs hardware")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "validation.png"), bbox_inches="tight")
    plt.close(fig)


SIM2NCU = {"async_wait": "mbarrier_wait", "tma_wait": "mbarrier_wait", "tensor_wait": "mbarrier_wait",
           "tensor_throttle": "math_pipe_throttle"}
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
                    ax.bar(pos, v, 0.8, bottom=bottom, color=STALL_COLORS[k], label=k if k not in labels else None)
                    if k not in labels:
                        labels.append(k)
                    bottom += v
            ticks.append((pos, f"{src}\n{r['gpu']} {r['impl']}"))
            pos += 1
        pos += 0.6
    ax.set_xticks([p for p, _ in ticks])
    ax.set_xticklabels([t for _, t in ticks], fontsize=7)
    ax.set_ylabel("share of stalled warp-cycles")
    ax.set_title("Why warps are not issuing: hardware vs simulator")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stalls.png"), bbox_inches="tight")
    plt.close(fig)


def fig_roles(study, out):
    r = next((r for r in study.get("validation", []) if r["gpu"] == "b300" and r["impl"] == "fa4"), None)
    if not r or not r["sim"].get("warp_by_index"):
        return
    roles = study.get("fa4_roles", {})
    wb = r["sim"]["warp_by_index"]
    groups = roles.get("groups") or [[i] for i in range(len(wb))]
    names = roles.get("names") or [f"warp {g[0]}" for g in groups]
    keys = ["selected", "wait", "math_pipe_throttle", "short_scoreboard", "async_wait", "tma_wait", "tensor_wait",
            "barrier", "mio_throttle", "branch_resolving", "drain"]
    col = dict(STALL_COLORS, selected="#34a853", drain="#e8eaed")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for gi, g in enumerate(groups):
        tot = {k: sum(wb[i].get(k, 0) for i in g) for k in keys}
        s = sum(tot.values()) or 1
        left = 0
        for k in keys:
            v = tot[k] / s
            if v > 0:
                ax.barh(gi, v, left=left, color=col[k], label=k if gi == 0 else None)
                left += v
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of the warp's resident cycles")
    ax.set_title("FA4 on B300: what each warp role does with its cycles (simulator)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fa4_roles.png"), bbox_inches="tight")
    plt.close(fig)


def fig_whatif(study, out):
    rows = [r for r in study.get("whatif", []) if r["variant"] != "baseline" and r["speedup_vs_baseline"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(rows) + 1.2))
    labels = [f"{r['gpu']} {r['impl']}: {r['variant']}" for r in rows]
    vals = [r["speedup_vs_baseline"] for r in rows]
    ax.barh(range(len(rows)), [v - 1 for v in vals], left=1, color=["#1a73e8" if v >= 1 else "#d93025" for v in vals])
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f}x", va="center", fontsize=7)
    ax.axvline(1, color="#202124", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("kernel speed vs the measured hardware (1.0 = calibrated)")
    ax.set_title("One number at a time: what limits each kernel (simulator)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "whatif.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ring(study, out):
    rows = study.get("scale", [])
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    sys_col = {"gb300_nvl72": "#d93025", "b300_9x8_ib": "#f29900", "a100_9x8_ib": "#1a73e8"}
    sys_lab = {"gb300_nvl72": "GB300 NVL72", "b300_9x8_ib": "9x HGX B300 (IB)", "a100_9x8_ib": "9x DGX A100 (IB)"}
    for ax, hkv in zip(axes, (8, 64)):
        for s in sys_col:
            pts = sorted((r["context"], r["eff"]) for r in rows if r["system"] == s and r["hkv"] == hkv and r["n"] == 72
                         and r.get("eff") is not None)
            if pts:
                ax.plot([p[0] / 1024 for p in pts], [100 * p[1] for p in pts], marker="o", ms=3, color=sys_col[s],
                        label=sys_lab[s])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (k tokens), 72-way context parallel")
        ax.set_title(f"Llama-3-70B attention, {hkv} KV heads" + (" (GQA)" if hkv < 64 else " (MHA)"))
    axes[0].set_ylabel("ring efficiency: kernel time / step time (%)")
    axes[1].legend()
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
