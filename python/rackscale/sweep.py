"""Model x hardware x phase sweeps, parallelism-plan search and bottleneck attribution.

This is the layer that turns single simulations into answers to questions like
"what does going from 8B to 405B do on an NVL72", "is MLA worth it at 128k
context" or "what is the best layout for Qwen3-235B decode at 30 tokens/s per
user". Every point is one simulated step (decode iteration, prefill pass or
training step) of one model replica in fast mode, so the SM model runs every
distinct kernel at cycle level and the fabric runs every collective.

    import rackscale.sweep as sw
    pts = sw.run([sw.Point("llama3-70b", "gb300", "decode", sw.plan(tp=8), batch=64, context=8192)])
    print(sw.best_plan("llama3-70b", "gb300", "decode", context=8192, min_tps_user=30))

Each point reports:

* ``step_us`` and the derived throughput (tokens/s per GPU, per user, TTFT, MFU);
* ``memory_gb`` against HBM, from :func:`rackscale.lower.memory` (points that
  do not fit are recorded but not simulated);
* ``breakdown_us``: the step split into **compute** (kernels running the
  tensor pipe at >= ``BOUND`` of peak), **hbm** (kernels moving HBM bytes at
  >= ``BOUND`` of achievable bandwidth), **latency** (kernels that reach
  neither: too few CTAs, launch tails, sync and dependency chains), **comm**
  (collective time not hidden behind kernels) and **launch** (the GPU idle
  between kernels). ``dominant`` is the largest of the five;
* ``classes_us``: the same step by kernel class (attention, GEMM, LM head,
  MoE experts, SSM, norm and elementwise);
* ``sched``: the scheduler-cycle stall anatomy over all kernels.

Throughput assumes every replica of a plan behaves like the simulated one
(true for TP and for EP with balanced routing): ``replicas = gpus // world``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

from . import lower, models
from .lower import Parallel, ParallelError

#: A kernel is compute- or HBM-bound when it runs that resource at this
#: fraction of its peak (tensor FLOPs at the max clock, HBM at the achievable
#: streaming bandwidth). Below it on both, it is latency-bound.
BOUND = 0.5
#: Leave this much HBM for the allocator, CUDA graphs and fragmentation.
HBM_USABLE = 0.92

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass(frozen=True)
class SystemDef:
    key: str
    #: Preset name or YAML path (relative to the repository root).
    system: str
    label: str
    #: GPU preset used for peaks (tensor FLOPs, HBM bandwidth, capacity).
    gpu: str
    gpus: int
    #: Per-kernel gap with CUDA graphs and eagerly launched, ns.
    graph_gap_ns: float
    eager_gap_ns: float
    #: How much of the GPU model comes from measurements.
    provenance: str
    caveat: str = ""


SYSTEMS: Dict[str, SystemDef] = {s.key: s for s in [
    SystemDef("a100", "configs/hgx_a100_8.yaml", "HGX A100 80GB x8", "a100_sxm", 8, 940.0, 1900.0,
              "calibrated A100 SXM4 microbenchmarks; NVLink3 / NVSwitch2 from spec",
              "The Ampere mma.sync GEMM model runs 1.5-2.4x slower than cuBLAS (decode GEMMs stream weights at "
              "~0.8 TB/s instead of ~1.7 TB/s), so A100 step times are pessimistic and over-weight GEMMs."),
    SystemDef("h100", "hgx_h100_8", "HGX H100 x8", "h100_sxm", 8, 600.0, 1500.0,
              "datasheet preset (no H100 measurements)", "Uncalibrated preset."),
    SystemDef("b300", "hgx_b300_8", "HGX B300 x8", "b300", 8, 554.0, 2600.0,
              "calibrated B300 SXM6 (clock, caches, HBM, tcgen05, launch gaps, DVFS)"),
    SystemDef("gb300", "gb300_nvl72", "GB300 NVL72", "gb300", 72, 554.0, 2600.0,
              "calibrated B300 die; NVL72 fabric from spec",
              "GB300 board power and clocks are taken from the measured HGX B300."),
]}


def plan(tp: int = 1, ep: int = 1, pp: int = 1, dp: int = 1) -> dict:
    return dict(tp=tp, ep=ep, pp=pp, dp=dp)


@dataclass
class Point:
    model: str
    system: str
    phase: str  # decode | prefill | train | mixed
    plan: dict = field(default_factory=plan)
    batch: int = 1  # decode: sequences per replica (per rank under EP); prefill: prompts
    context: int = 0  # decode: cached tokens per sequence
    seqlen: int = 0  # prefill / train: tokens per sequence
    chunk: int = 0  # mixed: prefill-chunk tokens added to the decode batch
    micro_batches: int = 1  # train
    schedule: str = "1f1b"  # train
    launch: str = "graph"  # graph | eager | megakernel | <ns>
    overlap: bool = False  # decode: two-micro-batch overlap
    dtype: str = "bf16"  # GEMM dtype (weights and activations)
    study: str = ""
    tag: str = ""

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# --------------------------------------------------------------------------
# plans


def _divides(m: models.Model, p: Parallel) -> bool:
    try:
        lower.check(m, p)
        return True
    except ParallelError:
        return False


def inference_plans(model: str, system: str, *, max_tp: int = 8) -> List[dict]:
    """Layouts of one inference replica: TP inside the NVLink domain, and for
    MoE models expert parallelism with data-parallel attention."""
    m = models.ALL[model]
    s = SYSTEMS[system]
    out = []
    for tp in (1, 2, 4, 8, 16):
        if tp <= min(max_tp if s.gpus <= 8 else max(max_tp, 16), s.gpus) and _divides(m, Parallel(tp=tp)):
            out.append(plan(tp=tp))
    if m.moe:
        for ep in (4, 8, 16, 32, 64, 72):
            if ep <= s.gpus and ep <= m.moe.experts and _divides(m, Parallel(ep=ep)):
                out.append(plan(ep=ep))
    return out


def train_plans(model: str, system: str) -> List[dict]:
    """TP x PP x DP layouts that use every GPU of the system."""
    m = models.ALL[model]
    s = SYSTEMS[system]
    out = []
    for tp in (1, 2, 4, 8):
        for pp in (1, 2, 3, 4, 6, 8, 9, 12, 18):
            if s.gpus % (tp * pp):
                continue
            p = Parallel(tp=tp, pp=pp, dp=s.gpus // (tp * pp))
            if _divides(m, p):
                out.append(plan(tp=tp, pp=pp, dp=p.dp))
    return out


# --------------------------------------------------------------------------
# lowering and simulation


def _par(pl: dict) -> Parallel:
    return Parallel(tp=pl.get("tp", 1), ep=pl.get("ep", 1), pp=pl.get("pp", 1), dp=pl.get("dp", 1))


def _launch_gap(pt: Point, s: SystemDef) -> float:
    if pt.launch == "graph":
        return s.graph_gap_ns
    if pt.launch == "eager":
        return s.eager_gap_ns
    if pt.launch == "megakernel":
        # One persistent kernel: no launch gaps between ops. (The intra-SM
        # fusion a real megakernel also does is not modelled.)
        return 0.0
    return float(pt.launch)


def _set_dtype(wl: dict, dtype: str) -> None:
    def walk(ops):
        for op in ops:
            if op.get("op") == "kernel" and op["kernel"]["type"] in ("gemm", "grouped_gemm"):
                op["kernel"]["dtype"] = dtype
            if op.get("op") == "loop":
                walk(op["body"])
    if "spmd" in wl:
        for st in wl["spmd"]["program"]["streams"]:
            walk(st)
    for r in wl.get("ranks", []):
        for st in r["streams"]:
            walk(st)


def workload(pt: Point) -> dict:
    m = models.ALL[pt.model]
    if pt.dtype != "bf16":
        from dataclasses import replace
        m = replace(m, dtype=pt.dtype)
    p = _par(pt.plan)
    if pt.phase == "decode":
        wl = lower.decode_step(m, p, pt.batch, pt.context, overlap=pt.overlap)
    elif pt.phase == "prefill":
        wl = lower.prefill(m, p, pt.batch, pt.seqlen)
    elif pt.phase == "train":
        wl = lower.train_step(m, p, pt.batch, pt.seqlen, micro_batches=pt.micro_batches, schedule=pt.schedule)
    elif pt.phase == "mixed":
        wl = lower.mixed_step(m, p, pt.batch, pt.context, pt.chunk)
    else:
        raise ValueError(f"unknown phase {pt.phase!r}")
    if pt.dtype != "bf16":
        _set_dtype(wl, pt.dtype)
    return wl


_PEAKS: Dict[str, dict] = {}


def _peaks(gpu: str) -> dict:
    if gpu not in _PEAKS:
        from . import presets
        g = presets()["gpus"][gpu]
        clk = g["sm_clock_mhz"] * 1e6
        f = g["tensor"]["flops_per_cycle_per_sm"]
        _PEAKS[gpu] = dict(
            tflops={k: v * g["sm_count"] * clk / 1e12 for k, v in f.items()},
            hbm_gbps=g["hbm"]["bw_gbps"] * g["hbm"]["efficiency"],
            hbm_gb=g["hbm"]["capacity_gb"], name=g["name"], sms=g["sm_count"])
    return _PEAKS[gpu]


def _system_text(s: SystemDef) -> str:
    path = os.path.join(_ROOT, s.system)
    if not s.system.endswith(".yaml"):
        return s.system
    with open(path) as f:
        text = f.read()
    # Calibration paths in system YAMLs are relative to the repository root.
    return re.sub(r"(calibration:\s*)(\S+\.json)", lambda mt: mt.group(1) + os.path.join(_ROOT, mt.group(2)), text)


_SHAPE = re.compile(r"_(\d+)x(\d+)x(\d+)")


def kernel_class(name: str, m: models.Model, tp: int) -> str:
    n = name.lower()
    if "fmha" in n or "fa2_" in n or "attention" in n or "flash" in n:
        return "attention"
    if n.startswith("grouped_gemm"):
        return "moe_experts"
    if "selective_state_update" in n or n.startswith("ssd_") or "conv1d" in n:
        return "ssm"
    if "gemm" in n:
        sh = _SHAPE.search(n)
        if sh and int(sh.group(2)) == m.vocab // max(1, tp):
            return "lm_head"
        return "gemm"
    if "row_norm" in n or "elementwise" in n:
        return "norm_elementwise"
    return "other"


def _bound(k: dict, pk: dict, dtype: str) -> str:
    c = k["tflops"] / max(1e-9, pk["tflops"].get(dtype, pk["tflops"]["bf16"]))
    b = k["gbps"] / pk["hbm_gbps"]
    if max(c, b) < BOUND:
        return "latency"
    return "compute" if c >= b else "hbm"


def analyse(rep: dict, pt: Point, s: SystemDef) -> dict:
    """Step-time breakdown of rank 0 (all ranks of these SPMD plans are alike)."""
    m = models.ALL[pt.model]
    pk = _peaks(s.gpu)
    r0 = rep["ranks"][0]
    step = r0["finish_us"]
    busy = r0["gpu_busy_us"]
    comm = max(0.0, min(r0["collective_us"], step - busy))
    launch = max(0.0, step - busy - comm)
    tp = pt.plan.get("tp", 1)
    tot = sum(k["total_us"] for k in rep["kernels"]) or 1.0
    scale = busy / tot  # aggregate over ranks -> rank 0's busy time
    bd = dict(compute=0.0, hbm=0.0, latency=0.0, comm=comm, launch=launch)
    classes: Dict[str, float] = {}
    for k in rep["kernels"]:
        us = k["total_us"] * scale
        bd[_bound(k, pk, pt.dtype)] += us
        c = kernel_class(k["name"], m, tp)
        classes[c] = classes.get(c, 0.0) + us
    if comm:
        classes["communication"] = comm
    if launch:
        classes["launch_idle"] = launch
    world = rep["world"]
    top = []
    for k in rep["kernels"][:6]:
        top.append(dict(name=k["name"], us=k["total_us"] / world, count=k["count"] / world, tflops=k["tflops"],
                        gbps=k["gbps"], bound=_bound(k, pk, pt.dtype), cls=kernel_class(k["name"], m, tp)))
    coll = rep.get("collectives", [])
    return dict(step_us=step, breakdown_us=bd, dominant=max(bd, key=bd.get), classes_us=classes,
                sched=rep["bubbles"]["sched"], mfu=rep.get("mfu"), top_kernels=top,
                busbw_gbps=(sum(c["busbw_gbps"] for c in coll) / len(coll)) if coll else None,
                n_collectives=len(coll))


def _derive(out: dict, pt: Point, s: SystemDef) -> None:
    world = out["world"]
    step = out["step_us"] * 1e-6
    tp = pt.plan.get("tp", 1)
    out["replicas"] = s.gpus // world
    if pt.phase in ("decode", "mixed"):
        tokens = pt.batch * world // tp
        out["tok_s_gpu"] = tokens / step / world
        out["tok_s_user"] = 1.0 / step
        out["tpot_ms"] = step * 1e3
        if pt.phase == "mixed":
            out["prefill_tok_s_gpu"] = pt.chunk * world // tp / step / world
    elif pt.phase == "prefill":
        out["tok_s_gpu"] = pt.batch * pt.seqlen * (world // tp) / step / world
        out["ttft_ms"] = step * 1e3
    else:
        dp = pt.plan.get("dp", 1)
        out["tok_s_gpu"] = pt.batch * pt.seqlen * pt.micro_batches * dp / step / world


def run_point(pt: Point, *, cache: bool = True, transform=None) -> dict:
    import rackscale as rs
    s = SYSTEMS[pt.system]
    m = models.ALL[pt.model]
    out = asdict(pt)
    out.update(system_label=s.label, gpu=_peaks(s.gpu)["name"], arch=arch_of(m))
    if m.mla and s.gpu.startswith("a100"):
        # MLA decode kernels (FlashMLA, FlashInfer MLA) target Hopper and later;
        # the simulator's Ampere FA2 cannot hold a 576-wide head in shared memory.
        out["error"] = "MLA has no Ampere kernel in the simulator"
        return out
    try:
        wl = workload(pt)
        if transform is not None:
            wl = transform(wl)
    except (ParallelError, ValueError) as e:
        out["error"] = str(e)
        return out
    world = wl["spmd"]["world"] if "spmd" in wl else len(wl["ranks"])
    out["world"] = world
    mem = wl.get("memory", {})
    need = sum(mem.values()) / 1e9
    cap = _peaks(s.gpu)["hbm_gb"]
    out.update(memory_gb={k: v / 1e9 for k, v in mem.items()}, memory_total_gb=need, hbm_gb=cap,
               fits=need <= HBM_USABLE * cap)
    if world > s.gpus:
        out["error"] = f"plan needs {world} GPUs, {s.label} has {s.gpus}"
        return out
    if not out["fits"]:
        return out
    t0 = time.time()
    try:
        rep = rs.simulate(_system_text(s), wl, fast=True, launch_gap_ns=_launch_gap(pt, s),
                          config={"persist_memo": cache}).raw
    except Exception as e:  # noqa: BLE001 - record and carry on with the sweep
        out["error"] = repr(e)[:300]
        return out
    out.update(analyse(rep, pt, s))
    out["sim_wall_s"] = time.time() - t0
    _derive(out, pt, s)
    return out


def arch_of(m: models.Model) -> dict:
    """Architecture tags used to group results."""
    if m.blocks is not None:
        kind = ("linear" if "L" in m.blocks else "ssm") if "A" not in m.blocks else "hybrid"
    else:
        kind = "moe" if m.moe else "dense"
    if m.attn_layers() == 0:
        attn = "none"
    elif m.mla:
        attn = "mla"
    elif m.gla:
        attn = f"gla{m.gla.latent_heads}"
    elif m.gta:
        attn = f"gta{m.kv_heads}"
    elif m.kv_heads == m.heads:
        attn = "mha"
    elif m.kv_heads == 1:
        attn = "mqa"
    else:
        attn = f"gqa{m.heads // m.kv_heads}"
    return dict(kind=kind, attention=attn, total_params_b=m.total_params() / 1e9,
                active_params_b=m.active_params() / 1e9, layers=len(m.block_list()) if m.blocks else m.layers,
                hidden=m.hidden, kv_bytes_per_token=lower._kv_bytes_per_token(m, Parallel()))


def _run_one(pt_dict: dict) -> dict:
    return run_point(Point(**pt_dict))


def run(points: Iterable[Point], *, jobs: int = 1, progress: bool = False,
        checkpoint: Optional[str] = None) -> List[dict]:
    """Simulate points, in parallel processes when ``jobs > 1``, cheapest
    first. With ``checkpoint`` (a JSON-lines file), every finished point is
    appended as it completes and points already in the file are not run
    again, so an interrupted sweep resumes. The output keeps the input order."""
    pts = list(points)
    done: Dict[str, dict] = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by an interruption
                done[r.pop("_key")] = r
    results: List[Optional[dict]] = [done.get(p.key()) for p in pts]
    todo = sorted((i for i in range(len(pts)) if results[i] is None), key=lambda i: _cost(pts[i]))
    t0 = time.time()
    ck = open(checkpoint, "a") if checkpoint else None

    def finish(i, r):
        results[i] = r
        if ck:
            ck.write(json.dumps(dict(r, _key=pts[i].key())) + "\n")
            ck.flush()
        if progress:
            what = r.get("error") or ("does not fit" if not r.get("fits", True) else
                                      f"{r['step_us']:10.1f} us  {r.get('tok_s_gpu', 0):9.0f} tok/s/GPU  {r['dominant']}")
            n = sum(x is not None for x in results)
            print(f"[{time.time() - t0:6.0f}s {n}/{len(pts)}] {r['study']:9s} {r['model']:17s} {r['system']:5s} "
                  f"{r['phase']:7s} {_plan_str(r['plan']):9s} b{r['batch']:<4d} ctx{r['context'] or r['seqlen']:<6d} "
                  f"{r['launch']:10s} {what}", flush=True)

    try:
        if jobs <= 1:
            for i in todo:
                finish(i, run_point(pts[i]))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(jobs) as ex:
                futs = {ex.submit(_run_one, asdict(pts[i])): i for i in todo}
                for fu in as_completed(futs):
                    finish(futs[fu], fu.result())
    finally:
        if ck:
            ck.close()
    return results  # type: ignore[return-value]


def _cost(pt: Point) -> float:
    m = models.ALL[pt.model]
    work = pt.batch * max(1, pt.context // 64 if pt.phase == "decode" else pt.seqlen) * max(1, pt.micro_batches)
    return m.active_params() * work / max(1, pt.plan.get("tp", 1))


def _plan_str(pl: dict) -> str:
    parts = [f"{k}{v}" for k, v in pl.items() if v > 1]
    return "x".join(parts) or "1gpu"


# --------------------------------------------------------------------------
# plan search


def best(results: List[dict], *, phase: str, min_tps_user: float = 0.0, max_ttft_ms: float = float("inf")) -> Optional[dict]:
    """The highest tokens/s per GPU among simulated points that meet the
    per-user speed (decode) or TTFT (prefill) target."""
    ok = [r for r in results if "step_us" in r and r["phase"] == phase]
    if phase in ("decode", "mixed"):
        ok = [r for r in ok if r["tok_s_user"] >= min_tps_user]
    if phase == "prefill":
        ok = [r for r in ok if r["ttft_ms"] <= max_ttft_ms]
    return max(ok, key=lambda r: r["tok_s_gpu"], default=None)


def pareto(results: List[dict]) -> List[dict]:
    """Decode points not beaten on both tokens/s per GPU and tokens/s per user."""
    pts = sorted((r for r in results if "tok_s_user" in r), key=lambda r: -r["tok_s_user"])
    out, best_gpu = [], -1.0
    for r in pts:
        if r["tok_s_gpu"] > best_gpu:
            out.append(r)
            best_gpu = r["tok_s_gpu"]
    return out


def best_plan(model: str, system: str, phase: str = "decode", *, context: int = 8192, seqlen: int = 8192,
              batches=(1, 8, 32, 128, 256), min_tps_user: float = 0.0, max_ttft_ms: float = float("inf"),
              launch: str = "graph", jobs: int = 1) -> dict:
    """Search layouts (and decode batch sizes) for one model on one system.

    Returns ``{"best": point, "points": [...], "pareto": [...]}``. Decode
    maximises tokens/s per GPU subject to ``min_tps_user``; prefill
    maximises prefill tokens/s per GPU subject to ``max_ttft_ms``; train
    maximises tokens/s per GPU over TP x PP x DP layouts of the whole system.
    """
    pts: List[Point] = []
    if phase == "train":
        for pl in train_plans(model, system):
            mbs = max(4, 2 * pl["pp"])
            pts.append(Point(model, system, "train", pl, batch=1, seqlen=seqlen, micro_batches=mbs, launch=launch))
    else:
        for pl in inference_plans(model, system):
            if phase == "decode":
                pts += [Point(model, system, "decode", pl, batch=b, context=context, launch=launch) for b in batches]
            else:
                pts.append(Point(model, system, "prefill", pl, batch=1, seqlen=seqlen, launch=launch))
    res = run(pts, jobs=jobs)
    return dict(best=best(res, phase=phase, min_tps_user=min_tps_user, max_ttft_ms=max_ttft_ms),
                points=res, pareto=pareto(res) if phase == "decode" else [])


_PLAN_KEEP = ("model", "system", "phase", "plan", "batch", "context", "seqlen", "fits", "memory_total_gb", "hbm_gb",
              "step_us", "tok_s_gpu", "tok_s_user", "ttft_ms", "mfu", "breakdown_us", "dominant", "classes_us",
              "top_kernels", "error")


def plan_report(out: dict, system: str) -> dict:
    """``best_plan``'s result as compact JSON, for scripts and agents."""
    def slim(r):
        return r and {k: r[k] for k in _PLAN_KEEP if k in r}
    return dict(best=slim(out["best"]), pareto=[slim(r) for r in out["pareto"]],
                points=[slim(r) for r in out["points"]], system=SYSTEMS[system].label,
                caveat=SYSTEMS[system].caveat)


def save(results: List[dict], path: str, **meta) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    systems = {k: dict(asdict(v), peaks=_peaks(v.gpu)) for k, v in SYSTEMS.items()}
    with open(path, "w") as f:
        json.dump(dict(meta, bound=BOUND, hbm_usable=HBM_USABLE, systems=systems, points=results), f, indent=0)


# --------------------------------------------------------------------------
# studies: the grids behind research/model_sweep


def _fitting_train_plans(model: str, system: str, seqlen: int, limit: int) -> List[dict]:
    """Training layouts that fit in HBM, least model parallelism first (the
    usual optimum: every extra TP or PP degree adds communication or bubbles)."""
    m = models.ALL[model]
    cap = _peaks(SYSTEMS[system].gpu)["hbm_gb"] * HBM_USABLE
    ok = []
    for pl in train_plans(model, system):
        mbs = max(4, 2 * pl["pp"])
        mem = lower.memory(m, _par(pl), phase="train", batch=1, seqlen=seqlen, micro_batches=mbs)
        if sum(mem.values()) / 1e9 <= cap:
            ok.append(pl)
    ok.sort(key=lambda pl: (pl["tp"] * pl["pp"], pl["pp"]))
    return ok[:limit]


def study_points(name: str, quick: bool = False) -> List[Point]:
    """Named grids. ``quick`` shrinks each to a smoke test."""
    P: List[Point] = []
    if name == "ladder":
        # Dense size ladder on four systems: decode (batch x layout), prefill, training.
        mods = ["llama3.2-1b", "llama3.2-3b", "llama3-8b", "qwen2.5-32b", "llama3-70b", "llama3-405b"]
        syss = ["a100", "h100", "b300", "gb300"]
        batches = [1, 16, 64, 256]
        if quick:
            mods, syss, batches = ["llama3.2-1b", "llama3-70b"], ["b300"], [1, 64]
        for mo in mods:
            for sy in syss:
                # The older generations get decode at every layout but prefill and
                # training only at their usual layouts: the Ampere model is slow to
                # simulate and H100 is an uncalibrated preset.
                full = sy in ("b300", "gb300")
                for pl in inference_plans(mo, sy):
                    P += [Point(mo, sy, "decode", pl, batch=b, context=8192, study=name) for b in batches]
                    if full or pl["tp"] in (1, 8):
                        P.append(Point(mo, sy, "prefill", pl, batch=1, seqlen=8192, study=name))
                if not quick or mo == "llama3.2-1b":
                    for pl in _fitting_train_plans(mo, sy, 4096, 1 if quick else (4 if full else 2)):
                        P.append(Point(mo, sy, "train", pl, batch=1, seqlen=4096, micro_batches=max(4, 2 * pl["pp"]),
                                       study=name))
    elif name == "attention":
        # Same 8B body, different sequence mixers, across context.
        mods = ["llama3-8b-mha", "llama3-8b", "llama3-8b-mqa", "llama3-8b-mla", "mamba2-8b", "mamba2-hybrid-8b"]
        ctxs, batches, lens = [2048, 8192, 32768, 131072], [1, 16, 64, 256], [2048, 8192, 32768]
        syss = ["b300", "a100"]
        if quick:
            ctxs, batches, lens, syss = [2048, 32768], [16], [2048], ["b300"]
        for mo in mods:
            for sy in syss:
                # The A100 model is several times slower to simulate and less
                # accurate: keep it to the contexts an 80 GB GPU serves.
                P += [Point(mo, sy, "decode", plan(), batch=b, context=c, study=name) for c in ctxs for b in batches
                      if sy != "a100" or (c <= 32768 and b <= 64)]
            P += [Point(mo, "b300", "prefill", plan(), batch=1, seqlen=L, study=name) for L in lens]
    elif name == "moe":
        mods = ["mixtral-8x7b", "qwen3-30b-a3b", "mixtral-8x22b", "qwen3-235b-a22b", "deepseek-v3"]
        batches = [8, 32, 128]
        syss = ["b300", "gb300"]
        if quick:
            mods, batches, syss = ["qwen3-30b-a3b", "deepseek-v3"], [32], ["gb300"]
        for mo in mods:
            for sy in syss:
                for pl in inference_plans(mo, sy):
                    P += [Point(mo, sy, "decode", pl, batch=b, context=4096, study=name) for b in batches]
                    if pl["ep"] > 1:
                        P.append(Point(mo, sy, "decode", pl, batch=128, context=4096, overlap=True, study=name,
                                       tag="two-batch overlap"))
                    if pl["tp"] == 8 or pl["ep"] in (8, 72):
                        P.append(Point(mo, sy, "prefill", pl, batch=1, seqlen=8192, study=name))
    elif name == "context":
        # Long-context decode at 70B scale on the rack.
        cases = [("llama3-70b", plan(tp=8)), ("mamba2-hybrid-70b", plan(tp=8)), ("mamba2-70b", plan(tp=8)),
                 ("deepseek-v3", plan(ep=72)), ("llama3-405b", plan(tp=8))]
        ctxs, batches = [4096, 16384, 65536, 131072], [16, 64]
        if quick:
            cases, ctxs, batches = cases[:2], [4096, 65536], [16]
        for mo, pl in cases:
            P += [Point(mo, "gb300", "decode", pl, batch=b, context=c, study=name) for c in ctxs for b in batches]
    elif name == "infra":
        # Launch mode (eager / CUDA graphs / one persistent kernel) across batch.
        cases = [("llama3.2-1b", plan()), ("llama3-8b", plan()), ("llama3-70b", plan(tp=8))]
        batches = [1, 8, 32, 128]
        if quick:
            cases, batches = cases[:1], [1, 32]
        for mo, pl in cases:
            for la in ("eager", "graph", "megakernel"):
                P += [Point(mo, "b300", "decode", pl, batch=b, context=1024, launch=la, study=name, tag="launch")
                      for b in batches]
        # Chunked prefill: what a prompt chunk riding along costs the decode batch.
        for mo, pl in [("llama3-8b", plan()), ("llama3-70b", plan(tp=8))][: 1 if quick else 2]:
            for ch in ([0, 2048] if quick else [0, 256, 1024, 2048, 4096, 8192]):
                P.append(Point(mo, "b300", "mixed", pl, batch=64, context=4096, chunk=ch, study=name, tag="chunked"))
        # Precision as a kernel-library choice: FP8 GEMMs (weights and activations).
        if not quick:
            for mo, pl in [("llama3-8b", plan()), ("llama3-70b", plan(tp=8)), ("llama3-405b", plan(tp=8))]:
                for dt in ("bf16", "fp8"):
                    P += [Point(mo, "b300", "decode", pl, batch=b, context=8192, dtype=dt, study=name, tag="dtype")
                          for b in (1, 64)]
                    P.append(Point(mo, "b300", "prefill", pl, batch=1, seqlen=8192, dtype=dt, study=name, tag="dtype"))
        # Measured-engine anchors (Llama-3.2-1B on one B300, see docs/CAPTURE.md).
        P.append(Point("llama3.2-1b", "b300", "decode", plan(), batch=32, context=520, study=name, tag="anchor:vllm"))
        P.append(Point("llama3.2-1b", "b300", "decode", plan(), batch=1, context=520, launch="megakernel", study=name,
                       tag="anchor:megakernel"))
    else:
        raise ValueError(f"unknown study {name!r} (ladder, attention, moe, context, infra)")
    return P


STUDIES = ("ladder", "attention", "moe", "context", "infra")

#: Native measurements on one B300 SXM6 (sm_103) from `rackscale capture`
#: runs (docs/CAPTURE.md), to set simulated engine results against.
MEASURED_B300 = [
    dict(engine="vLLM 0.32 (torch.compile, CUDA graphs)", model="llama3.2-1b", batch=32, context=520,
         gpu_step_us=942.0, e2e_step_us=2143.0, kernels_per_step=193,
         note="GPU span of one decode step; end-to-end includes the host between graph launches"),
    dict(engine="SGLang 0.5.22 (CUDA graphs)", model="llama3.2-1b", batch=32, context=520, e2e_step_us=1140.0,
         kernels_per_step=None, note="native decode step; eager is 15.7 ms"),
    dict(engine="Hazy Research low-latency megakernel", model="llama3.2-1b", batch=1, context=None,
         gpu_step_us=608.0, kernels_per_step=1, note="1,645 tokens/s, one 148-CTA persistent launch per token"),
    dict(engine="vLLM eager (one TP=8 rank, dummy weights)", model="llama3-70b", batch=32, context=None,
         gpu_step_us=6740.0, note="native decode step with CUDA graphs, one rank; replay on HGX B300 TP8: 6.39 ms"),
]
