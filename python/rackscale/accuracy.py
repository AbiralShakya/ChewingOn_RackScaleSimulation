"""Error bars for simulator answers, from our own validation record.

Every number the agent API returns carries a band. The band does not come from
the simulator itself, which is deterministic. It comes from how far the model
has been from hardware on comparable work, measured against Nsight Compute
cycles at the same clock (docs/CALIBRATION.md, docs/BLACKWELL.md, docs/CAPTURE.md).

A run's band is built from parts:

* every kernel is put in a *class* by how it was described (a recorded warp
  path, a SASS listing, a synthetic template, a measured duration), and by
  whether its GPU has measured calibration data;
* each class has a typical |error| and the worst error we have seen;
* the run's band is the time-weighted sum of its parts (kernels, collectives,
  idle gaps), plus the fast-mode bias. Summing assumes the errors are fully
  correlated, so the band is conservative.

Numbers marked ``heuristic`` have no direct measurement behind them yet (no
multi-GPU fabric has been measured; uncalibrated presets are datasheet values).
Update ``CLASSES`` when new validation lands.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

#: Kernel spec type -> accuracy class.
KIND_CLASS = {
    "path": "recorded_path",
    "sass": "sass",
    "trace": "sass",
    "fixed": "measured",
    "gemm": "synthetic",
    "grouped_gemm": "synthetic",
    "attention": "synthetic",
    "elementwise": "synthetic",
    "row_norm": "synthetic",
    "conv1d": "synthetic",
    "ssm_state_update": "synthetic",
    "ssd": "synthetic",
}

#: (class, calibrated?) -> typical |error| %, worst seen %, bias, evidence.
CLASSES: Dict[tuple, dict] = {
    ("recorded_path", True): {
        "typical_pct": 15.0, "worst_pct": 124.0, "bias": "none",
        "evidence": "B300 microbenchmarks 12/12 within 7% (docs/CALIBRATION.md); FlashAttention 16 recorded "
                    "kernels within 6%, mean 3.3% (docs/BLACKWELL.md); vLLM Llama-3.2-1B engine kernels 39% "
                    "time-weighted, worst TRT-LLM decode attention +124% (docs/CAPTURE.md)",
    },
    ("sass", True): {
        "typical_pct": 8.0, "worst_pct": 23.5, "bias": "none",
        "evidence": "A100 reference kernels 9/11 within 8%, hbm_copy -23%, l2_pointer_chase +16% (docs/CALIBRATION.md)",
    },
    ("synthetic", True): {
        "typical_pct": 12.0, "worst_pct": 85.0, "bias": "none",
        "evidence": "synthetic BF16 GEMMs vs 54 measured cuBLAS GEMMs on B300: median |error| 3.9% (decode), 19.1% "
                    "(mid), 11.6% (large) (docs/CALIBRATION.md); ThunderKittens GEMMs -6% to +85% (docs/CAPTURE.md)",
    },
    ("measured", True): {
        "typical_pct": 1.0, "worst_pct": 5.0, "bias": "none",
        "evidence": "kernel keeps its measured duration; replaying a vLLM capture with measured durations gave +0.3% "
                    "(docs/CAPTURE.md)",
    },
    ("uncalibrated", False): {
        "typical_pct": 25.0, "worst_pct": 60.0, "bias": "unknown", "heuristic": True,
        "evidence": "GPU preset built from datasheet numbers without measured microbenchmarks; Accel-Sim-class "
                    "simulators are typically 15-25% off even with measured configs",
    },
    ("collective", None): {
        "typical_pct": 25.0, "worst_pct": 50.0, "bias": "unknown", "heuristic": True,
        "evidence": "transfer-level fabric model with preset link numbers; no multi-GPU system measured yet",
    },
    ("idle", None): {
        "typical_pct": 30.0, "worst_pct": 100.0, "bias": "optimistic", "heuristic": True,
        "evidence": "launch gaps use preset launch overheads; host-side scheduling is not modelled unless a capture "
                    "replays host issue times (vLLM: 942 us GPU span vs 2143 us end-to-end, docs/CAPTURE.md)",
    },
}

#: Synthetic BF16 GEMMs on the calibrated B300 by size: median |error| against 54 measured
#: cuBLAS GEMMs (docs/CALIBRATION.md). Mid-sized GEMMs are the weak spot.
GEMM_SIZE_CLASSES = [
    (1e10, 3.9, "decode-sized (< 1e10 FLOP): median |error| 3.9% over 27 cuBLAS GEMMs"),
    (1e11, 19.1, "mid-sized (1e10-1e11 FLOP): median |error| 19.1% over 11 cuBLAS GEMMs; worst -33% "
                 "(256 x 18560 x 4096)"),
    (float("inf"), 11.6, "large (>= 1e11 FLOP): median |error| 11.6% over 16 cuBLAS GEMMs"),
]


def gemm_typical_pct(flops: float) -> tuple:
    """(typical |error| %, evidence) of a synthetic GEMM on a calibrated GPU, by size."""
    for hi, pct, ev in GEMM_SIZE_CLASSES:
        if flops < hi:
            return pct, ev
    return GEMM_SIZE_CLASSES[-1][1:]


#: Fast mode vs exact co-simulation on the DeepSeek-V3 decode demos (docs/RESULTS.md).
FAST_MODE = {"bias_pct": 1.0, "evidence": "fast vs exact: +0.5% (HGX B200 EP8), +1.0% (GB200 NVL72 EP72), docs/RESULTS.md"}

#: How far the cycle-level simulator sits from the closed forms (``scaling.estimate``), as
#: sim / textbook over the 22 plans of docs/book/11-every-parallelism.md section 11.5 on B300.
#: The textbook is a lower bound for training and prefill (power-capped clock, kernels at
#: 46-81% of peak, collectives at 25-75% of the link) and close for decode.
CLOSED_FORM = {
    "train": {"lo": 1.31, "median": 2.03, "hi": 2.23, "cases": 14},
    "prefill": {"lo": 1.87, "median": 2.11, "hi": 2.12, "cases": 4},
    "decode": {"lo": 0.96, "median": 1.15, "hi": 1.50, "cases": 4},
    "evidence": "docs/book/11-every-parallelism.md section 11.5 (scaling.check on B300: textbook, roofline replay "
                "and cycle-level simulation of the same plans); about two thirds of the training gap is the "
                "power-capped clock (2032 -> ~1455 MHz)",
}

#: How strongly the errors of two runs move together, by what changed between them.
#: Heuristic: the same kernels on the same model err the same way.
CORRELATION = {"hardware": 0.9, "run": 0.9, "workload": 0.6, "system": 0.3, "unrelated": 0.0}


def kernel_class(kind: Optional[str], calibrated: bool) -> tuple:
    cls = KIND_CLASS.get(kind or "", "synthetic")
    if cls == "measured":
        return ("measured", True)
    if not calibrated:
        return ("uncalibrated", False)
    return (cls, True)


def band(value: float, pct: float) -> dict:
    return {"value": value, "lo": value * (1 - pct / 100), "hi": value * (1 + pct / 100), "band_pct": round(pct, 2)}


def run_band(parts: Dict[tuple, float], collective_share: float, idle_share: float, fast: bool) -> dict:
    """``parts``: kernel class -> share of GPU-time. Returns the run's band and its make-up."""
    terms = []
    for cls, share in parts.items():
        c = CLASSES[cls]
        terms.append({"part": f"kernels:{cls[0]}", "share": share, "typical_pct": c["typical_pct"],
                      "worst_pct": c["worst_pct"], "bias": c["bias"]})
    for name, share in (("collective", collective_share), ("idle", idle_share)):
        if share > 0:
            c = CLASSES[(name, None)]
            terms.append({"part": name, "share": share, "typical_pct": c["typical_pct"], "worst_pct": c["worst_pct"],
                          "bias": c["bias"]})
    tot = sum(t["share"] for t in terms) or 1.0
    typical = sum(t["share"] * t["typical_pct"] for t in terms) / tot
    worst = sum(t["share"] * t["worst_pct"] for t in terms) / tot
    if fast:
        typical += FAST_MODE["bias_pct"]
        worst += FAST_MODE["bias_pct"]
        terms.append({"part": "fast_mode", "share": 1.0, "typical_pct": FAST_MODE["bias_pct"],
                      "worst_pct": FAST_MODE["bias_pct"], "bias": "pessimistic"})
    return {"typical_pct": round(typical, 2), "worst_pct": round(worst, 2), "terms": terms,
            "method": "time-weighted sum of per-class errors (fully correlated, conservative)"}


def diff_band(a_us: float, a_pct: float, b_us: float, b_pct: float, change: str) -> dict:
    """Band on ``b - a`` when the runs differ by ``change`` (a CORRELATION key)."""
    rho = CORRELATION.get(change, 0.0)
    sa, sb = a_us * a_pct / 100, b_us * b_pct / 100
    sd = math.sqrt(max(0.0, sa * sa + sb * sb - 2 * rho * sa * sb))
    return {"delta_us": b_us - a_us, "band_us": sd, "correlation": rho, "change": change,
            "significant": abs(b_us - a_us) > sd,
            "method": "sqrt(a^2 + b^2 - 2*rho*a*b) on the absolute bands, rho by kind of change (heuristic)"}


def table() -> dict:
    """The whole accuracy table, for display."""
    return {
        "classes": [{"class": k[0], "calibrated_gpu": k[1], **v} for k, v in CLASSES.items()],
        "fast_mode": FAST_MODE,
        "closed_form_vs_sim": CLOSED_FORM,
        "correlation": CORRELATION,
        "notes": "Errors are simulated vs measured time for the same work at the same clock. Calibrated GPUs: "
                 "B300/GB300 (built in), A100 SXM4/PCIe (calibration files attached automatically). Everything "
                 "else is a datasheet preset.",
    }
