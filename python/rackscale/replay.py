"""Replay a capture (``rackscale capture``) on any system, with a parallelism plan.

A capture is what one GPU ran. Replaying it re-simulates every recorded
kernel from its warp paths (or keeps its measured duration), and keeps the
captured streams, event dependencies and copies. A parallelism plan turns
that one-GPU program into a rack:

    tp: 8        # tensor parallel: an all-reduce after every row-parallel module
    ep: 1        # expert parallel: all-to-all dispatch/combine around MoE modules
    pp: 1        # pipeline parallel: layers split over stages, activations sent between them
    dp: 9        # data parallel: replicas (inference) / gradient all-reduce (training)
    microbatches: 1
    comm_sms: 16
    collectives: [...]   # where the collectives go; defaults below

The plan only adds communication and replicates or splits the program; it
does not change kernels. For tensor or expert parallelism the capture must
already be one rank's share of the work: run the engine on one GPU with the
model's per-rank shapes (:func:`shard_overrides` gives vLLM/SGLang
``hf_overrides`` for that, with dummy weights). Then every kernel is exactly
the kernel a rank of the real deployment runs.

What is not modelled: the engine's scheduler and batching policy (the capture
is one fixed batch), host overheads beyond the captured launch timeline
(``host=True`` replays the measured issue times; scaling assumes they do not
change), and load imbalance across experts.
"""

from __future__ import annotations

import copy as _copy
import fnmatch
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .record import load

# Where tensor/expert-parallel collectives go in vLLM and SGLang models (class
# names are the engines' own layer classes) and in plain Hugging Face models
# (module names). `bytes` is evaluated with `out` = the module's output bytes.
DEFAULT_COLLECTIVES = [
    {"after": {"cls": "RowParallelLinear"}, "kind": "all_reduce", "group": "tp", "bytes": "out"},
    {"after": {"cls": "VocabParallelEmbedding"}, "kind": "all_reduce", "group": "tp", "bytes": "out"},
    {"after": {"cls": "ParallelLMHead|LogitsProcessor"}, "kind": "all_gather", "group": "tp", "bytes": "out"},
    {"before": {"cls": "FusedMoE"}, "kind": "all_to_all", "group": "ep", "bytes": "out * topk"},
    {"after": {"cls": "FusedMoE"}, "kind": "all_to_all", "group": "ep", "bytes": "out * topk"},
    {"after": {"name": "*.o_proj|*.down_proj"}, "kind": "all_reduce", "group": "tp", "bytes": "out", "hf": True},
]

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def short_name(n: str) -> str:
    """`void ns::kernel<args...>(params)` -> `kernel<args...>`, cut to 72 chars."""
    n = n[5:] if n.startswith("void ") else n
    depth, cut = 0, len(n)
    for i, ch in enumerate(n):
        if ch in "<(":
            if ch == "(" and depth == 0:
                cut = i
                break
            depth += 1
        elif ch in ">)":
            depth -= 1
    head = n[:cut]
    lt = head.find("<")
    base = head[:lt] if lt >= 0 else head
    base = base.rsplit("::", 1)[-1]
    return (base + (head[lt:] if lt >= 0 else ""))[:72]


def shard_overrides(config: dict, tp: int = 1, ep: int = 1) -> dict:
    """``hf_overrides`` that give one GPU the per-rank shapes of a TP x EP deployment.

    `config` is the model's Hugging Face config (a dict). Heads, KV heads,
    MLP width and vocabulary are divided by `tp` (KV heads stay >= 1 and are
    replicated, as engines do); routed experts are divided by `ep`. Load the
    model with dummy weights (vLLM ``load_format="dummy"``, SGLang
    ``--load-format dummy``) and prompt with token ids below the new vocab size.
    """
    o = {}
    hd = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    if tp > 1:
        for k in ("num_attention_heads",):
            if k in config:
                assert config[k] % tp == 0, f"{k}={config[k]} not divisible by tp={tp}"
                o[k] = config[k] // tp
        if "num_key_value_heads" in config:
            o["num_key_value_heads"] = max(1, config["num_key_value_heads"] // tp)
        for k in ("intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size"):
            if config.get(k):
                o[k] = config[k] // tp
        if "vocab_size" in config:
            o["vocab_size"] = -(-config["vocab_size"] // tp)
        o["head_dim"] = hd
    if ep > 1:
        for k in ("num_experts", "n_routed_experts", "num_local_experts"):
            if config.get(k):
                assert config[k] % ep == 0, f"{k}={config[k]} not divisible by ep={ep}"
                o[k] = config[k] // ep
    return o


# --------------------------------------------------------------------- plan

class Plan:
    def __init__(self, d: Optional[dict] = None, **kw):
        d = dict(d or {}, **kw)
        self.tp = int(d.get("tp", 1))
        self.ep = int(d.get("ep", 1))
        self.pp = int(d.get("pp", 1))
        self.dp = int(d.get("dp", 1))
        self.microbatches = int(d.get("microbatches", 1))
        self.comm_sms = int(d.get("comm_sms", 16))
        self.topk = float(d.get("topk", 1))
        self.algo = d.get("algo", "auto")
        self.dp_allreduce_bytes = int(d.get("dp_allreduce_bytes", 0))
        self.collectives = d.get("collectives", DEFAULT_COLLECTIVES)
        self.world = self.tp * self.pp * self.dp
        if self.ep > 1 and self.world % self.ep:
            raise ValueError(f"ep={self.ep} must divide world={self.world}")

    def group(self, g: str) -> str:
        if g == "tp":
            return f"block:{self.tp}"
        if g == "ep":
            return f"block:{self.ep}"
        if g == "dp":
            return f"stride:{self.tp * self.pp}"
        return g

    def degree(self, g: str) -> int:
        return {"tp": self.tp, "ep": self.ep, "dp": self.dp}.get(g, 0)


def load_plan(p: Union[None, str, dict, Plan]) -> Plan:
    if p is None or isinstance(p, Plan):
        return p or Plan()
    if isinstance(p, dict):
        return Plan(p)
    import yaml  # noqa: WPS433 - only needed for plan files

    with open(p) as f:
        return Plan(yaml.safe_load(f))


def _match(rule: dict, name: str, cls: str) -> bool:
    if "cls" in rule and not re.fullmatch(rule["cls"], cls or ""):
        return False
    if "name" in rule and not any(fnmatch.fnmatch(name or "", pat) for pat in rule["name"].split("|")):
        return False
    return True


# ------------------------------------------------------------------ lowering

def _kernel_spec(k: dict, o: dict, mode: str, base: str, sms: int) -> dict:
    if mode == "paths" and k.get("path") and k.get("replay_ok", True):
        spec = {"type": "path", "path": os.path.join(base, k["path"])}
        if k.get("regs"):
            spec["regs"] = k["regs"]
        return spec
    t = _gpu(o)
    if mode == "measured" and t:
        ns = t[1] - t[0]
    else:
        ns = (k.get("measured_ns") or {}).get("median", 0.0)
    ctas = k["grid"][0] * k["grid"][1] * k["grid"][2]
    return {"type": "fixed", "ns": float(ns), "sms": int(max(1, min(ctas, sms)))}


def _gpu(o: dict):
    """GPU [start, end] of a captured op: native (timing pass) when matched, else from the tool's events."""
    return o.get("native_ns") or o.get("gpu_ns")


def _host(o: dict) -> float:
    return float(o.get("native_host_ns", o.get("host_ns", 0.0)))


def _select(ops: List[dict], steps: Optional[Tuple[int, Optional[int]]]) -> List[dict]:
    if steps is None:
        return ops
    starts = [i for i, o in enumerate(ops) if o["op"] == "step"]
    if not starts:
        return ops
    a, b = steps
    lo = starts[a] if a < len(starts) else len(ops)
    hi = starts[b] if b is not None and b < len(starts) else len(ops)
    return ops[lo:hi]


def program(cap: Union[str, dict], plan: Union[None, str, dict, Plan] = None, *, mode: str = "paths",
            host: bool = False, region: int = 0, steps: Optional[Tuple[int, Optional[int]]] = None,
            stage: int = 0, sms: Optional[int] = None) -> Tuple[List[List[dict]], dict]:
    """One rank's streams (rsim ops) from the capture, with the plan's collectives inserted.

    Returns ``(streams, info)``; `info` has the layer range and stage boundary
    activation sizes used for pipeline parallelism.
    """
    cap = load(cap) if isinstance(cap, str) else cap
    plan = load_plan(plan)
    base = cap["_dir"]
    sms = sms or cap["device"].get("sms") or 148
    kernels = cap["kernels"]
    hf = not any(o.get("cls", "").endswith("ParallelLinear") for p in cap["processes"]
                 for r in p["regions"] for o in r if o["op"] == "push")
    rules = [r for r in plan.collectives if r.get("hf", False) == hf or "hf" not in r]
    rules = [r for r in rules if plan.degree(r.get("group", "tp")) > 1]

    streams: Dict[Tuple[int, int], List[dict]] = {}
    stream_order: List[Tuple[int, int]] = []

    def st(pid, s):
        key = (pid, s)
        if key not in streams:
            streams[key] = []
            stream_order.append(key)
        return streams[key]

    regions = [(p["pid"], _select(p["regions"][region], steps)) for p in cap["processes"] if region < len(p["regions"])]
    h_base = min((_host(o) for _pid, ops in regions for o in ops if o["op"] in ("kernel", "copy")), default=0.0)
    owner = [_layer_of_ops(ops) for _pid, ops in regions]
    nlayers = 1 + max((l for lay in owner for l in lay if l is not None), default=-1)
    per = -(-nlayers // plan.pp) if nlayers else 0
    lo_layer, hi_layer = stage * per, min((stage + 1) * per, nlayers)
    info = {"layers": nlayers, "stage_layers": (lo_layer, hi_layer), "boundary_bytes": 0}

    def mine(layer):
        if plan.pp == 1:
            return True
        return lo_layer <= layer < hi_layer if layer is not None else stage == 0

    for (pid, ops), lay in zip(regions, owner):
        stack: List[dict] = []  # open modules: name, cls, layer, where an op before them would go
        last_stream = 0
        for o, layer in zip(ops, lay):
            t = o["op"]
            if t == "push":
                stack.append({"name": o.get("name", ""), "cls": o.get("cls", ""), "layer": layer,
                              "pos": (last_stream, len(st(pid, last_stream)))})
                continue
            if t == "pop":
                if not stack:
                    continue
                mod = stack.pop()
                out = o.get("bytes", 0)
                if mod["layer"] == hi_layer - 1 and re.search(r"layers\.\d+$", mod["name"]):
                    info["boundary_bytes"] = max(info["boundary_bytes"], out)
                if not mine(mod["layer"]):
                    continue
                for r in rules:
                    for where in ("before", "after"):
                        if where in r and _match(r[where], mod["name"], mod["cls"]):
                            nbytes = int(eval(str(r.get("bytes", "out")), {}, {"out": out, "topk": plan.topk}))  # noqa: S307
                            coll = {"op": "collective", "name": f"{r['kind']} {mod['name'] or mod['cls']}",
                                    "kind": r["kind"], "bytes": max(1, nbytes), "group": plan.group(r["group"]),
                                    "algo": plan.algo, "sms": plan.comm_sms}
                            if where == "after":
                                st(pid, last_stream).append(coll)
                            else:
                                s, i = mod["pos"]
                                st(pid, s).insert(i, coll)
                continue
            if t in ("kernel", "copy") and not mine(layer):
                continue
            if t == "kernel":
                k = kernels[o["kernel"]]
                s = st(pid, o["stream"])
                if host:
                    s.append({"op": "host", "at_ns": _host(o) - h_base})
                s.append({"op": "kernel", "name": short_name(k["name"]),
                          "kernel": _kernel_spec(k, o, mode, base, sms)})
                last_stream = o["stream"]
            elif t == "copy":
                s = st(pid, o["stream"])
                if host:
                    s.append({"op": "host", "at_ns": _host(o) - h_base})
                g = _gpu(o)
                ns = (g[1] - g[0]) if g else 0.0
                s.append({"op": "delay", "ns": float(max(0.0, ns))})
            elif t == "record":
                st(pid, o["stream"]).append({"op": "record", "event": f"p{pid}:{o['event']}"})
            elif t == "wait":
                st(pid, o["stream"]).append({"op": "wait", "event": f"p{pid}:{o['event']}"})
    return [streams[k] for k in stream_order], info


def _layer_of_ops(ops: List[dict]) -> List[Optional[int]]:
    """The decoder layer every op belongs to, for pipeline stages.

    Ops inside a ``layers.N`` module get N. Ops before the first layer get
    None (the first stage runs them); ops after the last layer get the last
    layer, and ops between layers the layer before them, so they run on the
    stage that holds that layer.
    """
    out: List[Optional[int]] = []
    stack: List[Optional[int]] = []
    last = None
    for o in ops:
        if o["op"] == "push":
            m = _LAYER.search(o.get("name", ""))
            stack.append(int(m.group(1)) if m else (stack[-1] if stack else None))
            cur = stack[-1]
        elif o["op"] == "pop":
            cur = stack.pop() if stack else None
        else:
            cur = stack[-1] if stack else None
        if cur is not None:
            last = cur
        out.append(cur if cur is not None else last)
    return out


def _uniq_events(ops: List[dict], tag: str) -> List[dict]:
    out = []
    for o in ops:
        if o["op"] in ("record", "wait"):
            o = dict(o, event=o["event"] + tag)
        out.append(o)
    return out


def to_workload(cap: Union[str, dict], plan: Union[None, str, dict, Plan] = None, *, mode: str = "paths",
                host: bool = False, region: int = 0, steps: Optional[Tuple[int, Optional[int]]] = None,
                name: Optional[str] = None) -> dict:
    """The capture as an rsim workload over ``plan.world`` ranks.

    `mode`: ``paths`` re-simulates every kernel that has warp paths and keeps
    the measured duration of the rest; ``measured`` keeps every kernel's own
    measured duration (only overlap, dependencies and communication are
    simulated). `host` replays the captured host issue time of every launch.
    """
    cap = load(cap) if isinstance(cap, str) else cap
    plan = load_plan(plan)
    title = name or f"capture {os.path.basename(cap['_dir'])} ({mode}) tp{plan.tp} ep{plan.ep} pp{plan.pp} dp{plan.dp}"
    mb = plan.microbatches

    def rank_streams(stage):
        streams, info = program(cap, plan, mode=mode, host=host, region=region, steps=steps, stage=stage)
        out = [[] for _ in streams]
        for m in range(mb):
            for i, s in enumerate(streams):
                body = _uniq_events(_copy.deepcopy(s), f"@{m}")
                if plan.pp > 1 and i == 0:
                    if stage > 0:
                        body.insert(0, {"op": "recv", "bytes": max(1, info["boundary_bytes"]), "peer": -plan.tp,
                                        "relative": True})
                    if stage < plan.pp - 1:
                        body.append({"op": "send", "bytes": max(1, info["boundary_bytes"]), "peer": plan.tp,
                                     "relative": True})
                out[i].extend(body)
        if plan.dp > 1 and plan.dp_allreduce_bytes:
            out[0].append({"op": "collective", "name": "dp gradient all-reduce", "kind": "all_reduce",
                           "bytes": plan.dp_allreduce_bytes, "group": plan.group("dp"), "algo": plan.algo,
                           "sms": plan.comm_sms})
        return {"streams": out}

    if plan.pp == 1:
        return {"name": title, "spmd": {"world": plan.world, "program": rank_streams(0)}}
    progs = [rank_streams(s) for s in range(plan.pp)]
    ranks = []
    for d in range(plan.dp):
        for s in range(plan.pp):
            for _t in range(plan.tp):
                ranks.append(progs[s])
    return {"name": title, "ranks": ranks}


def measured(cap: Union[str, dict], region: int = 0, steps: Optional[Tuple[int, Optional[int]]] = None) -> dict:
    """What was measured on hardware for the selected ops: GPU span, host span, kernel busy time.

    Times are native (the nsys timing pass) when it matched the region, else
    from the tool's CUDA events (``source`` says which).
    """
    cap = load(cap) if isinstance(cap, str) else cap
    t0, t1, h0, h1, busy = None, None, None, 0.0, 0.0
    per: Dict[str, List[float]] = {}
    native = all(p.get("native") for p in cap["processes"])
    for p in cap["processes"]:
        if region >= len(p["regions"]):
            continue
        for o in _select(p["regions"][region], steps):
            if o["op"] in ("kernel", "copy"):
                h = _host(o)
                h0 = h if h0 is None else min(h0, h)
                h1 = max(h1, h)
                g = _gpu(o)
                if not g:
                    continue
                a, b = g
                t0 = a if t0 is None else min(t0, a)
                t1 = b if t1 is None else max(t1, b)
                if o["op"] == "kernel":
                    busy += b - a
                    per.setdefault(o["kernel"], []).append(b - a)
    return {"source": "nsys" if native else "events", "gpu_span_us": ((t1 - t0) / 1e3) if t0 is not None else 0.0,
            "host_span_us": (h1 - h0) / 1e3 if h0 is not None else 0.0, "kernel_busy_us": busy / 1e3,
            "per_kernel_ns": per}


def replay(cap: Union[str, dict], system: str, plan: Union[None, str, dict, Plan] = None, *, mode: str = "paths",
           host: bool = False, region: int = 0, steps: Optional[Tuple[int, Optional[int]]] = None,
           fast: bool = True, launch_gap_ns: Optional[float] = None, trace: Optional[str] = None):
    """Simulate the capture on `system` with `plan`; returns a :class:`rackscale.Report`."""
    from . import simulate

    wl = to_workload(cap, plan, mode=mode, host=host, region=region, steps=steps)
    return simulate(system, wl, fast=fast, launch_gap_ns=launch_gap_ns, trace=trace)


def kernel_table(cap: Union[str, dict], gpu_system: str = "hgx_b300_8", *, only: Optional[Sequence[str]] = None,
                 launch_gap_ns: float = 0.0) -> List[dict]:
    """Every recorded kernel simulated alone vs its measured median duration.

    ``sim_us`` charges the measured kernel window (launch, rasterisation,
    drain), as Nsight Compute's kernel duration does; ``sim_stream_us`` does
    not, as back to back in a stream the windows overlap and Nsight Systems
    reports less. A replay that does not finish (``error``) usually means the
    recording uses an mbarrier pattern the converter does not model yet.
    """
    from . import simulate

    cap = load(cap) if isinstance(cap, str) else cap
    rows = []
    for kid, k in cap["kernels"].items():
        if not k.get("path") or (only and kid not in only):
            continue
        row = {"kernel": kid, "name": short_name(k["name"]), "count": k["count"]}
        meas = (k.get("measured_ns") or {}).get("median")
        row["measured_us"] = meas / 1e3 if meas else None
        variants = list((k.get("path_variants") or {"phases": k["path"]}).items())
        err = None
        for variant, path in variants:
            spec = _kernel_spec(dict(k, path=path, replay_ok=True), {}, "paths", cap["_dir"],
                                cap["device"].get("sms") or 148)
            wl = {"name": kid, "spmd": {"world": 1, "program": {"streams": [[{"op": "kernel", "name": kid,
                                                                              "kernel": spec}]]}}}
            try:
                t = {}
                for window in (True, False):
                    rep = simulate(gpu_system, wl, launch_gap_ns=launch_gap_ns,
                                   config={"gpu": {"kernel_window": window}})
                    t[window] = rep.raw["makespan_us"]
                    if not rep.raw["kernels"]:
                        raise RuntimeError("kernel did not finish (replay deadlock)")
            except Exception as e:  # try the other wait semantics, then report
                err = str(e)[:200]
                continue
            row.update(variant=variant, sim_us=t[True], sim_stream_us=t[False],
                       error_pct=(100 * (t[True] * 1e3 - meas) / meas) if meas else None,
                       stream_error_pct=(100 * (t[False] * 1e3 - meas) / meas) if meas else None)
            err = None
            break
        rows.append(dict(row, error=err) if err else row)
    return rows


def check_kernels(cap: Union[str, dict], gpu_system: str = "hgx_b300_8") -> List[dict]:
    """Simulate every recorded kernel alone and write ``kernel_check.json`` next to the capture.

    Replays then use the warp paths only of kernels that replayed, and keep
    the measured duration of the others.
    """
    cap = load(cap) if isinstance(cap, str) else cap
    rows = kernel_table(cap, gpu_system)
    with open(os.path.join(cap["_dir"], "kernel_check.json"), "w") as f:
        json.dump({"system": gpu_system, "kernels": rows}, f, indent=1)
    for r in rows:
        k = cap["kernels"][r["kernel"]]
        k["replay_ok"] = "error" not in r
        if r.get("variant") in (k.get("path_variants") or {}):
            k["path"] = k["path_variants"][r["variant"]]
    return rows
