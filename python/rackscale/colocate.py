"""Several models on one rack at once.

A *tenant* is one model with its own phase, parallel plan and GPUs: a large
MoE decoding on 64 GPUs of an NVL72 next to a small dense model on the other
8, a prefill pool feeding a decode pool, a trainer beside a sampler. Each
tenant is lowered on its own (:mod:`rackscale.lower`), then its ranks are
placed on the rack's GPUs and the programs are merged into one workload with
explicit per-rank programs, so every tenant runs on the same simulated
NVLink fabric and NVSwitch chips at the same time.

Tenants can also share GPUs: give two tenants overlapping ``gpus`` and the
shared GPU runs both programs as separate streams, the way two processes
under MPS do. In exact mode (``fast=False``) their kernels share the SMs, L2
and HBM cycle by cycle; fast mode replays each kernel's solo timing on the
SMs left free, so it captures SM time-slicing but not cache or bandwidth
interference.

Each tenant's program is looped ``steps`` times so the tenants overlap in
time; :func:`run` reports every tenant's mean step against the same tenant
simulated alone, which is the interference.

    from rackscale import colocate as co
    from rackscale.scaling import Plan
    ts = [co.Tenant("ds", "deepseek-v3", "decode", Plan(ep=64), gpus=range(64), batch=64, context=4096),
          co.Tenant("8b", "llama3-8b", "decode", Plan(tp=8), gpus=range(64, 72), batch=32, context=8192)]
    print(co.summary(co.run(ts, "gb300", layers=4)))
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Union

from . import models
from .models import Model
from .scaling import Plan, _group, lower_plan


@dataclass
class Tenant:
    name: str
    model: Union[str, Model]
    phase: str  # decode | prefill | train
    plan: Plan = field(default_factory=Plan)
    #: The rack GPUs its ranks run on, rank i on gpus[i].
    gpus: Sequence[int] = ()
    batch: int = 1
    context: int = 4096
    seqlen: int = 4096
    micro_batch: int = 1
    micro_batches: int = 1
    #: Steps to simulate back to back.
    steps: int = 4
    #: Delay before the first step, ns (to stagger tenants).
    start_ns: float = 0.0

    def model_obj(self) -> Model:
        return models.ALL[self.model] if isinstance(self.model, str) else self.model

    def workload(self, layers: Optional[int] = None) -> dict:
        m = self.model_obj()
        if layers is not None:
            m = m.with_layers(min(layers, len(m.block_list()) if m.blocks is not None else m.layers))
        return lower_plan(m, self.phase, self.plan, micro_batch=self.micro_batch, seqlen=self.seqlen,
                          micro_batches=self.micro_batches, batch=self.batch, context=self.context)

    def tokens_per_step(self) -> int:
        if self.phase == "decode":
            return self.batch * (self.plan.ep if self.plan.ep > 1 else 1)
        if self.phase == "prefill":
            return self.batch * self.seqlen
        return self.micro_batch * self.seqlen * self.micro_batches * self.plan.dp


@dataclass
class Flow:
    """Traffic between tenants that is not part of either model's program:
    KV caches moving from a prefill pool to a decode pool, weights pushed
    from a trainer to samplers. ``bytes`` go from each ``src`` GPU to the
    ``dst`` GPU at the same position (round robin when the lists differ in
    length), ``times`` times. With ``every_ns`` the transfers are issued on a
    fixed clock (one every ``every_ns`` from ``start_ns``), so the offered
    rate is ``bytes / every_ns``; a transfer that is still running when the
    next one is due delays it, and the report shows the rate delivered."""

    name: str
    src: Sequence[int]
    dst: Sequence[int]
    bytes: int
    times: int = 1
    every_ns: float = 0.0
    sms: int = 2
    start_ns: float = 0.0

    def pairs(self) -> List[tuple]:
        src, dst = list(self.src), list(self.dst)
        return [(src[i % len(src)], dst[i % len(dst)]) for i in range(max(len(src), len(dst)))]

    def program(self, kind: str, peer: int) -> List[dict]:
        op = {"op": kind, "bytes": int(self.bytes), "peer": peer, "sms": self.sms}
        if not self.every_ns:
            body = [op] if not self.start_ns else [{"op": "host", "at_ns": self.start_ns}, op]
            return body if self.times == 1 else [{"op": "loop", "times": self.times, "body": body}]
        out = []
        for j in range(self.times):
            out += [{"op": "host", "at_ns": self.start_ns + j * self.every_ns}, op]
        return out


def _rank_programs(wl: dict) -> List[List[List[dict]]]:
    if "spmd" in wl:
        world = wl["spmd"]["world"]
        return [copy.deepcopy(wl["spmd"]["program"]["streams"]) for _ in range(world)]
    return [copy.deepcopy(r["streams"]) for r in wl["ranks"]]


def _place(ops: List[dict], rank: int, world: int, gpus: List[int], tag: str) -> List[dict]:
    out = []
    for op in ops:
        op = dict(op)
        kind = op.get("op")
        if kind == "loop":
            op["body"] = _place(op["body"], rank, world, gpus, tag)
        elif kind == "collective":
            op["group"] = [gpus[r] for r in _group(op["group"], rank, world)]
            if op.get("name"):
                op["name"] = f"{tag}.{op['name']}"
        elif kind in ("send", "recv"):
            peer = (rank + op["peer"]) % world if op.get("relative") else op["peer"]
            op["peer"] = gpus[peer]
            op.pop("relative", None)
        elif kind in ("record", "wait"):
            op["event"] = f"{tag}.{op['event']}"
        elif kind == "kernel" and op.get("name"):
            op["name"] = f"{tag}.{op['name']}"
        out.append(op)
    return out


def build(tenants: Iterable[Tenant], num_gpus: int, *, layers: Optional[int] = None,
          flows: Sequence[Flow] = ()) -> dict:
    """One workload holding every tenant on its GPUs, plus ``flows`` between them."""
    return _build(tenants, num_gpus, layers=layers, flows=flows)[0]


#: Which streams of which GPUs a tenant or flow owns: {name: {gpu: [stream index]}}.
Owners = Dict[str, Dict[int, List[int]]]


def _build(tenants: Iterable[Tenant], num_gpus: int, *, layers: Optional[int] = None,
           flows: Sequence[Flow] = ()) -> tuple:
    tenants = list(tenants)
    names = [t.name for t in tenants] + [f.name for f in flows]
    if len(set(names)) != len(names):
        raise ValueError(f"tenant and flow names must be unique: {names}")
    ranks: List[List[List[dict]]] = [[] for _ in range(num_gpus)]
    owned: Owners = {}
    mem: List[Dict[str, float]] = [{} for _ in range(num_gpus)]
    flops = 0.0
    for t in tenants:
        wl = t.workload(layers)
        progs = _rank_programs(wl)
        gpus = list(t.gpus)
        if len(gpus) != len(progs):
            raise ValueError(f"tenant {t.name}: plan {t.plan.label()} needs {len(progs)} GPUs, given {len(gpus)}")
        if max(gpus) >= num_gpus:
            raise ValueError(f"tenant {t.name}: GPU {max(gpus)} is outside the {num_gpus}-GPU system")
        for r, streams in enumerate(progs):
            g = gpus[r]
            placed = []
            for s in streams:
                body = _place(s, r, len(progs), gpus, t.name)
                st = [{"op": "loop", "times": t.steps, "body": body}] if t.steps > 1 else body
                if t.start_ns:
                    st = [{"op": "delay", "ns": t.start_ns}] + st
                placed.append(st)
            # The tenant's streams on this GPU, so its time is read from them
            # alone and not from another tenant or flow sharing the GPU.
            owned.setdefault(t.name, {})[g] = list(range(len(ranks[g]), len(ranks[g]) + len(placed)))
            ranks[g] += placed
            for k, v in (wl.get("memory") or {}).items():
                mem[g][f"{t.name}.{k}"] = mem[g].get(f"{t.name}.{k}", 0.0) + v
        flops += wl.get("model_flops", 0.0) * t.steps
    for f in flows:
        for a, b in f.pairs():
            for g, kind, peer in ((a, "send", b), (b, "recv", a)):
                owned.setdefault(f.name, {}).setdefault(g, []).append(len(ranks[g]))
                ranks[g].append(f.program(kind, peer))
    for g in range(num_gpus):
        if not ranks[g]:
            ranks[g] = [[{"op": "delay", "ns": 0}]]
    names = " + ".join(f"{t.name}[{len(list(t.gpus))}]" for t in tenants)
    wl = {"name": f"colocated: {names}", "ranks": [{"streams": s} for s in ranks],
          "memory_per_rank": mem, "model_flops": flops}
    return wl, owned


def _finish(rep: dict, streams: Dict[int, List[int]]) -> float:
    """When the last of these streams finished, us."""
    return max(rep["ranks"][g]["stream_finish_us"][i] for g, idx in streams.items() for i in idx)


def run(tenants: Sequence[Tenant], system: str = "gb300", *, layers: Optional[int] = None, alone: bool = True,
        fast: bool = True, flows: Sequence[Flow] = ()) -> dict:
    """Simulate the tenants together (and, with ``alone``, each one by itself
    on the same GPUs of an otherwise idle rack). Per tenant: mean step time,
    tokens/s, and the slowdown from sharing the rack."""
    import rackscale as rs
    from .scaling import hardware
    from .sweep import SYSTEMS, _system_text
    hw = hardware(system)
    sd = SYSTEMS.get(system)
    sys_text = _system_text(sd) if sd else system
    wl, owned = _build(tenants, hw.gpus, layers=layers, flows=flows)
    rep = rs.simulate(sys_text, wl, fast=fast, launch_gap_ns=hw.launch * 1e9).raw
    out = dict(system=hw.name, layers=layers, together_makespan_us=rep["makespan_us"], tenants=[], flows=[],
               warnings=rep.get("warnings", []))
    for t in tenants:
        fin = _finish(rep, owned[t.name]) - t.start_ns / 1e3
        row = dict(name=t.name, model=t.model_obj().name, phase=t.phase, plan=t.plan.label(), gpus=len(list(t.gpus)),
                   steps=t.steps, step_us=fin / t.steps, tokens_per_step=t.tokens_per_step())
        row["tok_s"] = row["tokens_per_step"] / (row["step_us"] * 1e-6)
        if alone:
            solo, own = _build([t], hw.gpus, layers=layers)
            rs_ = rs.simulate(sys_text, solo, fast=fast, launch_gap_ns=hw.launch * 1e9).raw
            row["alone_step_us"] = (_finish(rs_, own[t.name]) - t.start_ns / 1e3) / t.steps
            row["slowdown"] = row["step_us"] / row["alone_step_us"] - 1.0
        out["tenants"].append(row)
    for f in flows:
        # Per GPU pair: bytes moved over the flow's schedule (times x every_ns), or
        # over the time it actually took when it fell behind.
        span_us = _finish(rep, owned[f.name]) - f.start_ns / 1e3
        window_us = max(span_us, f.times * f.every_ns / 1e3)
        row = dict(name=f.name, pairs=len(f.pairs()), bytes=f.bytes, times=f.times, sms=f.sms,
                   delivered_gbps=f.bytes * f.times / (window_us * 1e3), span_us=span_us)
        if f.every_ns:
            row["offered_gbps"] = f.bytes / f.every_ns
        out["flows"].append(row)
    out["report"] = rep
    return out


def summary(res: dict) -> str:
    lines = [f"{len(res['tenants'])} tenants on {res['system']}" + (f" ({res['layers']} layers each)" if res["layers"] else "")]
    # Throughput of a model cut to a few layers says nothing about the
    # whole model, so it is shown only for whole models.
    whole = not res["layers"]
    lines.append(f"{'tenant':<12} {'model':<18} {'phase':<8} {'plan':<16} {'GPUs':>4} {'step us':>10} {'alone us':>10} "
                 f"{'slowdown':>9}" + (f" {'tok/s':>12}" if whole else ""))
    for t in res["tenants"]:
        alone = f"{t['alone_step_us']:10.1f}" if "alone_step_us" in t else " " * 10
        slow = f"{100 * t['slowdown']:+8.2f}%" if "slowdown" in t else " " * 9
        lines.append(f"{t['name']:<12} {t['model']:<18} {t['phase']:<8} {t['plan']:<16} {t['gpus']:>4} "
                     f"{t['step_us']:10.1f} {alone} {slow}" + (f" {t['tok_s']:12,.0f}" if whole else ""))
    for f in res.get("flows", []):
        offered = f"{f['offered_gbps']:.1f} GB/s offered, " if "offered_gbps" in f else ""
        lines.append(f"flow {f['name']}: {f['pairs']} GPU pairs x {f['times']} x {f['bytes'] / 2**20:.1f} MiB on "
                     f"{f['sms']} SMs, {offered}{f['delivered_gbps']:.1f} GB/s delivered per pair")
    for w in res.get("warnings", []):
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)
