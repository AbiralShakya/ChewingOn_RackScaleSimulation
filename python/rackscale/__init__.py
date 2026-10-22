"""rackscale: rack-scale, cycle-level GPU simulation.

The heavy lifting happens in Rust (``rackscale._native``); this package
builds workloads (from model configs or captured profiler traces), runs them
and presents the results.

    >>> import rackscale as rs
    >>> wl = rs.lower.decode_step(rs.models.DEEPSEEK_V3, rs.Parallel(ep=72), batch=64, context=4096, layers=2)
    >>> rep = rs.simulate("gb200_nvl72", wl)
    >>> print(rep.summary())
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

from . import capture, lower, models  # noqa: F401
from .lower import Parallel  # noqa: F401

try:
    from . import _native
except ImportError as e:  # pragma: no cover - only when the extension is not built
    _native = None
    _IMPORT_ERROR = e

__all__ = ["simulate", "viz", "kernel", "presets", "Report", "Parallel", "lower", "models", "capture"]


def _need_native():
    if _native is None:
        raise ImportError(
            "rackscale native extension not built; run `maturin develop --release` "
            f"(or `pip install .`) in the repository root: {_IMPORT_ERROR}"
        )
    return _native


def _system_yaml(system: Union[str, os.PathLike]) -> str:
    s = os.fspath(system)
    if os.path.exists(s):
        with open(s) as f:
            return f.read()
    if "\n" in s or ":" in s:
        return s  # already YAML text
    return f"preset: {s}\n"  # preset name


@dataclass
class Report:
    raw: dict

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    @property
    def makespan_us(self) -> float:
        return self.raw["makespan_us"]

    @property
    def bubbles(self) -> dict:
        return self.raw["bubbles"]

    @property
    def warnings(self) -> list:
        """Reasons the result may not be meaningful (e.g. the model does not fit in HBM)."""
        return self.raw.get("warnings", [])

    @property
    def memory(self) -> Optional[dict]:
        """Device memory of the fullest GPU: total_gb, capacity_gb, by_category_gb."""
        return self.raw.get("memory")

    def kernels(self, top: int = 20) -> list:
        return self.raw["kernels"][:top]

    def summary(self) -> str:
        r = self.raw
        lines = [f"== {r['workload']} on {r['system']} ({r['world']} GPUs) ==",
                 f"makespan {r['makespan_us']:.1f} us" + (f"   MFU {100 * r['mfu']:.1f}%" if r.get("mfu") else "")]
        mem = r.get("memory")
        if mem:
            parts = ", ".join(f"{k} {v:.1f}" for k, v in mem["by_category_gb"].items())
            lines.append(f"memory (rank {mem['rank']}): {mem['total_gb']:.1f} of {mem['capacity_gb']:.0f} GB  [{parts}]")
        for w in r.get("warnings", []):
            lines.append(f"WARNING: {w}")
        b = r["bubbles"]
        lines.append(f"GPU idle {100 * b['gpu_idle_fraction']:.1f}%   tensor-core idle in kernels {100 * b['tensor_idle_fraction']:.1f}%")
        lines.append("scheduler-cycle anatomy:")
        for k, v in sorted(b["sched"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<20} {100 * v:6.2f}%  " + "#" * round(50 * v))
        lines.append(f"{'kernel':<56} {'n':>5} {'total_us':>10} {'TFLOPS':>8} {'GB/s':>8} {'tc%':>6}")
        for k in r["kernels"][:12]:
            lines.append(f"{k['name'][:56]:<56} {k['count']:>5} {k['total_us']:>10.1f} {k['tflops']:>8.1f} {k['gbps']:>8.0f} {100 * k['tensor_util']:>6.1f}")
        for c in r["collectives"][:6]:
            lines.append(f"  {c['kind']:<14} {c['algo']:<6} {c['bytes']:>12} B ranks={c['ranks']:<3} busbw {c['busbw_gbps']:.0f} GB/s")
        return "\n".join(lines)


def simulate(system: Union[str, os.PathLike], workload: Union[dict, str], *, trace: Optional[str] = None,
             launch_gap_ns: Optional[float] = None, fast: bool = False, config: Optional[dict] = None) -> Report:
    """Run ``workload`` (dict / JSON text / path) on ``system`` (preset name, YAML text or path).

    ``fast=True`` simulates each distinct kernel once on a cold GPU and reuses
    the result across ranks/layers (see ``RunConfig::memoize_kernels``).
    """
    nat = _need_native()
    if isinstance(workload, dict):
        wl = json.dumps(workload)
    elif os.path.exists(workload):
        with open(workload) as f:
            wl = f.read()
    else:
        wl = workload
    cfg = dict(config or {})
    if launch_gap_ns is not None:
        cfg["launch_gap_ns"] = launch_gap_ns
    if fast:
        cfg["memoize_kernels"] = True
    cfg.setdefault("trace", trace is not None)
    out = nat.simulate(_system_yaml(system), wl, json.dumps(cfg), trace)
    return Report(json.loads(out))


def viz(system: Union[str, os.PathLike], workload: Union[dict, str], out: str, *, name: Optional[str] = None,
        fast: bool = True, detail_ranks=(0,), warp_sms=(0,), max_warp_events: int = 400_000,
        link_bin_ns: float = 1000.0, launch_gap_ns: Optional[float] = None, config: Optional[dict] = None) -> Report:
    """Simulate and write a zoomable viewer (``out`` ending in .html) or an rsviz/1 bundle (.json).

    The page goes from the rack's links down to per-warp timelines and per-instruction stall samples
    for the ranks in ``detail_ranks`` and the SMs in ``warp_sms``. See docs/VIZ.md.
    """
    nat = _need_native()
    if isinstance(workload, dict):
        wl = json.dumps(workload)
    elif os.path.exists(workload):
        with open(workload) as f:
            wl = f.read()
    else:
        wl = workload
    cfg = dict(config or {})
    cfg.update(trace=False, memoize_kernels=fast, detail_ranks=list(detail_ranks))
    if launch_gap_ns is not None:
        cfg["launch_gap_ns"] = launch_gap_ns
    cfg.setdefault("gpu", {})["detail"] = {"ctas": True, "pc_stats": True, "warp_sms": list(warp_sms),
                                           "max_warp_events": max_warp_events}
    cfg.setdefault("net", {})["link_bin_ns"] = link_bin_ns
    title = name or (json.loads(wl).get("name", "run") if wl.lstrip().startswith("{") else "run")
    rep = nat.viz(_system_yaml(system), wl, json.dumps(cfg), title, os.fspath(out))
    return Report(json.loads(rep))


def kernel(gpu: str, kind: str, *dims: int, dtype: str = "bf16") -> dict:
    """Simulate one synthetic kernel, e.g. ``kernel("b200", "gemm", 8192, 8192, 8192)``."""
    return json.loads(_need_native().kernel(gpu, kind, list(dims), dtype))


def presets() -> dict:
    return json.loads(_need_native().presets())
