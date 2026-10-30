"""Long-context fast path: steady-state extrapolation of long, regular kernels.

A causal FlashAttention-4 forward over 16k tokens takes about four minutes to
simulate at cycle level on four cores; at 1M tokens it would take days. The
kernels that grow with context are regular, though: every CTA of an attention
prefill runs the same K/V loop, and the SSD scan, the GEMMs, norms and convs
grow linearly in tokens once the GPU is full. So for those kernels this module
simulates two or three *reduced* instances at cycle level (on the same
calibrated GPU, through the same fast-mode path a rack run uses, DVFS
included) and extrapolates:

* **attention prefill** (``sq > 16``): per-wave time is linear in the K/V
  iterations a CTA runs, ``t_wave = a + b * k_iters``, measured on a grid
  that is an exact number of full waves; the full kernel is
  ``waves * t_wave``. A causal kernel is the non-causal one with half the
  keys per CTA, which is how the synthetic FA4 kernel balances it.
* **token-linear kernels** (GEMM ``m``, ``row_norm`` rows, ``elementwise``
  ``n``, ``conv1d`` / SSD ``seqlen``, ``embedding_bag`` bags):
  ``t = t(x1) + slope * (x - x1)`` from
  two sizes that already fill the GPU many times over.

:func:`check` compares each rule with a full simulation at a size where both
are affordable; ``research/dao_lab`` reports those errors next to every
result that uses the fast path. :func:`freeze` rewrites the large kernels of
a workload into ``fixed`` ops with the extrapolated time, so a rack run keeps
simulating every collective, stream dependency and small kernel exactly.

    from rackscale import extrap
    us = extrap.kernel_us("hgx_b300_8", {"type": "attention", "batch": 1, "heads_q": 32,
                          "heads_kv": 8, "sq": 1 << 20, "skv": 1 << 20, "d": 128, "causal": True})
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

#: Kernels at or below these sizes are simulated in full.
ATTN_EXACT_MAX = 8192 * 8192  # sq * skv per (batch, head)
LINEAR_EXACT_MAX_TOKENS = 32768

_CACHE: Dict[str, float] = {}
_CACHE_PATH = os.environ.get("RACKSCALE_EXTRAP_CACHE")


def _load_cache() -> None:
    if _CACHE_PATH and not _CACHE and os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH) as f:
            _CACHE.update(json.load(f))


def _save_cache() -> None:
    if _CACHE_PATH:
        # Merge with what other processes wrote since we loaded.
        if os.path.exists(_CACHE_PATH):
            try:
                with open(_CACHE_PATH) as f:
                    for k, v in json.load(f).items():
                        _CACHE.setdefault(k, v)
            except ValueError:
                pass
        tmp = f"{_CACHE_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(_CACHE, f)
        os.replace(tmp, _CACHE_PATH)


_BUILD: Optional[str] = None


def _system_key(system: str) -> str:
    """System YAML plus the simulator build, so a cache never outlives the
    model that filled it."""
    global _BUILD
    from . import _native, _system_yaml
    if _BUILD is None:
        with open(_native.__file__, "rb") as f:
            _BUILD = hashlib.sha256(f.read()).hexdigest()[:12]
    return _BUILD + hashlib.sha256(_system_yaml(system).encode()).hexdigest()[:16]


def simulate_us(system: str, spec: dict) -> float:
    """Time of one kernel alone on rank 0 of ``system``, as a fast-mode rack run
    places it on the timeline (cycle-level SM model, sustained clock)."""
    from . import simulate
    _load_cache()
    key = _system_key(system) + json.dumps(spec, sort_keys=True)
    if key not in _CACHE:
        wl = {"name": "k", "spmd": {"world": 1, "program": {"streams": [[{"op": "kernel", "name": "k",
                                                                             "kernel": spec}]]}}}
        rep = simulate(system, wl, fast=True).raw
        _CACHE[key] = rep["ranks"][0]["kernel_us"]
        _save_cache()
    return _CACHE[key]


def _gpu(system: str) -> dict:
    from . import api
    return api.resolve(system)["gpus"][0]


# ---------------------------------------------------------------------------
# attention prefill


def _attn_tiles(g: dict, d: int, sq: int, tied: bool = False) -> Tuple[int, int, int]:
    """(rows per CTA, K/V block, CTAs per SM) of the synthetic Blackwell
    attention kernel (``rsim_gpu::synth::attention``)."""
    smem = g["smem_per_sm_bytes"]
    tile_q, kv_blk = 128, 128
    kv_mult = 1 if tied else 2
    fits = lambda tq, kb: (kv_mult * 2 * kb * d + tq * d) * 2 + 1024 <= smem
    while not fits(tile_q, kv_blk) and kv_blk > 32:
        kv_blk //= 2
    while not fits(tile_q, kv_blk) and tile_q > 64:
        tile_q //= 2
    q_stage = 2 if sq > tile_q else 1
    return tile_q * q_stage, kv_blk, 1


def _is_long_attention(spec: dict) -> bool:
    return spec.get("type") == "attention" and spec["sq"] > 16 and spec["sq"] * spec["skv"] > ATTN_EXACT_MAX


def _attention_us(system: str, spec: dict, reserved: int = 0) -> Tuple[float, dict]:
    g = _gpu(system)
    if g.get("arch", "").lower() != "blackwell":
        raise ValueError("attention extrapolation is implemented for the Blackwell kernel only")
    sms = g["sm_count"]
    rows, kv_blk, cps = _attn_tiles(g, spec["d"], spec["sq"], spec.get("kv_tied", spec["d"] > 256))
    skv_eff = math.ceil(spec["skv"] / 2) if spec.get("causal") else spec["skv"]
    hq = spec["heads_q"]
    ctas = spec["batch"] * hq * math.ceil(spec["sq"] / rows)
    # SMs held by concurrent communication kernels (ring sends) are not
    # available to the attention grid, which changes its wave count.
    waves = math.ceil(ctas / ((sms - reserved) * cps))
    # Reduced grid: batch 1, the same heads, the fewest Q tiles that make an
    # exact number of full waves (whole heads, so GQA sharing in L2 is kept).
    per_wave = sms * cps
    n_tiles = per_wave // math.gcd(hq, per_wave)
    red_waves = hq * n_tiles // per_wave
    while red_waves < 2:
        n_tiles *= 2
        red_waves *= 2
    k_full = math.ceil(skv_eff / kv_blk)
    ks = [k for k in (16, 32) if k < k_full] or [max(1, k_full // 2), k_full]
    pts = []
    for k in ks:
        red = dict(spec, batch=1, sq=n_tiles * rows, skv=k * kv_blk, causal=False)
        pts.append((k, simulate_us(system, red) / red_waves))
    (k1, t1), (k2, t2) = pts[0], pts[-1]
    b = (t2 - t1) / (k2 - k1) if k2 != k1 else t1 / k1
    a = t1 - b * k1
    us = waves * (a + b * k_full)
    return us, {"rule": "attention_waves", "waves": waves, "reserved_sms": reserved, "k_iters": k_full, "t_wave_us": a + b * k_full,
                "fit": {"a_us": a, "b_us_per_iter": b, "points": pts, "reduced_waves": red_waves}}


# ---------------------------------------------------------------------------
# token-linear kernels

_AXIS = {"gemm": "m", "row_norm": "rows", "elementwise": "n", "conv1d": "seqlen", "ssd": "seqlen",
         "embedding_bag": "bags"}


def _tokens(spec: dict) -> Optional[int]:
    ax = _AXIS.get(spec.get("type"))
    if ax is None:
        return None
    if spec["type"] == "elementwise":
        return None  # no token axis of its own; small, simulated in full
    return spec[ax]


def _is_long_linear(spec: dict) -> bool:
    t = _tokens(spec)
    if t is None or t <= LINEAR_EXACT_MAX_TOKENS:
        return False
    return spec["type"] != "conv1d" or spec["seqlen"] > 1


def _clamp_flops(system: str, dtype: str) -> float:
    """FLOPs above which the power model's sustained clock no longer moves
    (its largest measured run of ``dtype``); 0 when it has no such run."""
    p = _gpu(system).get("power") or {}
    return max((r["flops"] for r in p.get("runs", []) if r["dtype"] == dtype), default=0.0)


def _linear_us(system: str, spec: dict) -> Tuple[float, dict]:
    ax = _AXIS[spec["type"]]
    x = spec[ax]
    x1, x2 = LINEAR_EXACT_MAX_TOKENS // 2, LINEAR_EXACT_MAX_TOKENS
    if spec["type"] == "gemm":
        # The DVFS model lowers the clock with kernel size up to its largest
        # measured run, so time is linear in M only past that size: fit there.
        per_row = 2.0 * spec["n"] * spec["k"]
        x1 = max(x1, -(-int(math.ceil(_clamp_flops(system, spec.get("dtype", "bf16")) / per_row)) // 128) * 128)
        x2 = 2 * x1
        if x <= x2:
            return simulate_us(system, spec), {"rule": "exact"}
    if spec["type"] == "ssd":
        c = spec.get("chunk", 256)
        x1, x2 = (x1 // c) * c, (x2 // c) * c
    t1 = simulate_us(system, dict(spec, **{ax: x1}))
    t2 = simulate_us(system, dict(spec, **{ax: x2}))
    slope = (t2 - t1) / (x2 - x1)
    return t1 + slope * (x - x1), {"rule": "token_linear", "axis": ax, "points": [(x1, t1), (x2, t2)],
                                   "slope_us_per_token": slope}


# ---------------------------------------------------------------------------


def is_long(spec: dict) -> bool:
    return _is_long_attention(spec) or _is_long_linear(spec)


def estimate(system: str, spec: dict, reserved: int = 0) -> Tuple[float, dict]:
    """``(microseconds, how)``: a full simulation when the kernel is small,
    otherwise the steady-state extrapolation and its fit. ``reserved`` SMs are
    held by concurrent communication kernels: the attention rule counts waves
    on the rest, token-linear kernels scale by the lost SM fraction."""
    if _is_long_attention(spec):
        return _attention_us(system, spec, reserved)
    if _is_long_linear(spec):
        us, how = _linear_us(system, spec)
        if reserved and how["rule"] != "exact":
            sms = _gpu(system)["sm_count"]
            us *= sms / (sms - reserved)
            how["reserved_sms"] = reserved
        return us, how
    return simulate_us(system, spec), {"rule": "exact"}


def kernel_us(system: str, spec: dict) -> float:
    return estimate(system, spec)[0]


def _walk(ops: List[dict]) -> Iterable[dict]:
    for op in ops:
        yield op
        if op.get("op") == "loop":
            yield from _walk(op["body"])


def _programs(wl: dict) -> Iterable[dict]:
    if "spmd" in wl:
        yield wl["spmd"]["program"]
    for r in wl.get("ranks", []):
        yield r


def freeze(system: str, workload: dict, reserved_sms: int = 0) -> Tuple[dict, List[dict]]:
    """Copy of ``workload`` whose long kernels are ``fixed`` ops with their
    extrapolated time. Returns ``(workload, log)``; the log lists every
    rewritten op with the rule used. ``reserved_sms``: SMs that concurrent
    communication holds while these kernels run (a ``fixed`` op does not see
    them, so the estimate must)."""
    wl = copy.deepcopy(workload)
    log = []
    done: Dict[str, Tuple[float, dict]] = {}
    for prog in _programs(wl):
        for stream in prog["streams"]:
            for op in _walk(stream):
                spec = op.get("kernel")
                if op.get("op") != "kernel" or not isinstance(spec, dict) or not is_long(spec):
                    continue
                key = json.dumps(spec, sort_keys=True)
                if key not in done:
                    done[key] = estimate(system, spec, reserved_sms)
                    log.append({"op": op.get("name"), "spec": spec, "us": done[key][0], **done[key][1]})
                op["kernel"] = {"type": "fixed", "ns": done[key][0] * 1e3}
    return wl, log


def check(system: str, spec: dict) -> dict:
    """Extrapolated vs fully simulated time for one kernel (validation)."""
    est, how = (_attention_us(system, spec) if spec.get("type") == "attention" else _linear_us(system, spec))
    full = simulate_us(system, spec)
    return {"spec": spec, "extrapolated_us": est, "simulated_us": full, "error": est / full - 1.0, "how": how}
