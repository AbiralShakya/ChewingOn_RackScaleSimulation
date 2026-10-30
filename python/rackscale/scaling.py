"""Closed-form scaling math for every parallelism, and the check against the simulator.

This is the arithmetic of DeepMind's *How to Scale Your Model*, done for
NVIDIA GPUs and NVLink domains, at three levels that can be compared case by
case:

1. **Textbook** (:func:`estimate`). One formula per parallelism axis: compute
   time ``F / C``, memory time ``B / beta``, and a communication time
   ``alpha + bytes / W`` for each collective, combined by stating which
   communication overlaps with what. Every term is returned with the formula
   that produced it.
2. **Roofline replay** (:func:`replay`). The *lowered program* (the exact
   kernels, collectives and stream dependencies the simulator runs) timed on
   an ideal machine: each kernel at ``max(F / C, B / beta)``, each collective
   at its closed form, streams that overlap perfectly and never contend. It
   differs from the textbook only where the program has work the formulas
   leave out (norms, residuals, the exact causal attention FLOPs, the
   schedule's real dependency chain).
3. **Cycle level** (the simulator). The same program on the SM, memory and
   NVLink/NVSwitch models.

``(sim - replay) / replay`` is then the part of reality that roofline
arithmetic misses, and :func:`check` attributes it to kernel efficiency,
collective efficiency, overlap that does not happen and launch gaps.

Units: seconds, bytes and FLOPs unless a key says otherwise (``_us``, ``_gb``).

Notation, per GPU: ``C`` peak dense FLOP/s at the model dtype, ``beta`` the
achievable HBM bandwidth, ``W`` the NVLink bandwidth per direction (all of a
GPU's links: 18 x 50 GB/s on NVLink 5), ``W_ib`` the scale-out NIC bandwidth,
``alpha`` the fixed cost of one collective. A collective's ``bytes`` is the
whole buffer on one rank (the input of an all-reduce or reduce-scatter, the
output of an all-gather), as in the workload format and nccl-tests.

    import rackscale.scaling as sc
    hw = sc.hardware("gb300")
    est = sc.estimate("llama3-70b", hw, "train", sc.Plan(tp=8, dp=9, zero=3), micro_batch=1, seqlen=8192)
    print(sc.explain(est))
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union

from . import lower, models
from .lower import Parallel
from .models import Model

#: Bytes per element of a GEMM operand for each dtype the lowering emits.
DTYPE_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "fp4": 0.5, "nvfp4": 0.5625, "fp32": 4.0, "tf32": 4.0}


# ---------------------------------------------------------------------------
# hardware


@dataclass(frozen=True)
class Hardware:
    """The handful of numbers the closed forms need, read from the same
    presets the simulator runs (so the comparison is like for like)."""

    name: str
    gpu: str
    gpus: int
    #: GPUs in one NVLink domain (8 on HGX, 72 on NVL72).
    domain: int
    #: Peak dense FLOP/s per dtype at the preset SM clock.
    flops: Dict[str, float]
    #: Achievable HBM bandwidth (spec x measured streaming efficiency), B/s.
    hbm_bw: float
    hbm_bytes: float
    #: NVLink bandwidth of one GPU, one direction, B/s.
    nvlink_bw: float
    #: NVSwitch in-network reduction (NVLS) available inside the domain.
    nvls: bool
    #: Scale-out bandwidth per GPU per direction (B/s), 0 when there is none.
    ib_bw: float
    #: Fixed cost of one collective (launch, handshake, protocol), s.
    alpha: float = 5e-6
    #: Fixed cost of one point-to-point transfer, s.
    alpha_p2p: float = 3e-6
    #: Per-kernel launch gap with CUDA graphs, s (measured on B300).
    launch: float = 0.0

    def peak(self, dtype: str = "bf16") -> float:
        return self.flops.get(dtype, self.flops["bf16"])

    def link_bw(self, n: int) -> float:
        """Per-GPU bandwidth of a group of ``n`` consecutive ranks: NVLink
        inside one domain, the NIC once the group spans domains."""
        return self.nvlink_bw if n <= self.domain or not self.ib_bw else self.ib_bw


def hardware(system: str = "gb300", *, launch: Optional[float] = None,
             overrides: Optional[dict] = None) -> Hardware:
    """:class:`Hardware` for a sweep system key (``gb300``, ``b300``, ``h100``,
    ``a100``) or any simulator system preset (``gb200_nvl72``, ``hgx_b200_8``, ...).
    ``overrides`` are agent-API hardware knobs (``rackscale.api.KNOBS``), so the
    closed forms see the same hypothetical machine the simulator does."""
    from . import api, presets
    from .sweep import SYSTEMS
    sd = SYSTEMS.get(system)
    sys_name = sd.system if sd else system
    r = api.resolve(sd and api._repo_path(sys_name) if sd and sys_name.endswith(".yaml") else sys_name, overrides)
    g = r["gpus"][0]
    clk = g["sm_clock_mhz"] * 1e6
    flops = {k: v * g["sm_count"] * clk for k, v in g["tensor"]["flops_per_cycle_per_sm"].items()}
    topo = r["spec"]["topology"]
    if topo["kind"] == "multi_node":
        intra = topo["intra"]
        domain = r["num_gpus"] // topo["nodes"]
        ib = topo["nic"]["gbps_per_dir"] * 1e9
    else:
        intra, domain, ib = topo, r["num_gpus"], 0.0
    nvls = bool(intra.get("switch", {}).get("in_network_reduction"))
    gap = launch if launch is not None else (sd.graph_gap_ns * 1e-9 if sd else 0.0)
    return Hardware(name=r["spec"]["name"], gpu=g["name"], gpus=r["num_gpus"], domain=domain, flops=flops,
                    hbm_bw=g["hbm"]["bw_gbps"] * g["hbm"].get("efficiency", 1.0) * 1e9,
                    hbm_bytes=g["hbm"]["capacity_gb"] * 1e9,
                    nvlink_bw=g["nvlink"]["gbps_per_port_per_dir"] * g["nvlink"]["ports"] * 1e9,
                    nvls=nvls, ib_bw=ib, launch=gap)


# ---------------------------------------------------------------------------
# collectives


def collective(kind: str, nbytes: float, n: int, hw: Hardware, *, algo: Optional[str] = None) -> Tuple[float, str]:
    """Time of one collective over ``n`` ranks and the formula behind it.

    Ring algorithms move ``(n-1)/n`` of the buffer per rank per phase (two
    phases for an all-reduce). With NVLS the switch does the reduction: each
    GPU sends its buffer into the switch once and reads the result once, on
    the two directions of its links at the same time, so an all-reduce costs
    ``bytes / W`` instead of ``2 (n-1)/n bytes / W``. An all-to-all on a
    full-bisection switch sends ``(n-1)/n`` of the buffer to the others.
    """
    if n <= 1 or nbytes <= 0:
        return 0.0, "0 (one rank)"
    w = hw.link_bw(n)
    f = (n - 1) / n
    nvls = hw.nvls and n <= hw.domain and algo in (None, "nvls")
    if kind == "all_reduce":
        if nvls:
            return hw.alpha + nbytes / w, "alpha + B/W (NVLS)"
        return hw.alpha + 2 * f * nbytes / w, "alpha + 2(n-1)/n B/W (ring)"
    if kind in ("all_gather", "reduce_scatter", "all_to_all"):
        return hw.alpha + f * nbytes / w, "alpha + (n-1)/n B/W"
    if kind == "broadcast":
        return hw.alpha + nbytes / w, "alpha + B/W"
    raise ValueError(f"unknown collective {kind!r}")


def p2p(nbytes: float, hw: Hardware, *, cross_domain: bool = False) -> float:
    """One point-to-point transfer between two GPUs (all of a GPU's NVLinks
    reach every switch, so one peer can use the full ``W``)."""
    w = hw.ib_bw if cross_domain and hw.ib_bw else hw.nvlink_bw
    return hw.alpha_p2p + nbytes / w


# ---------------------------------------------------------------------------
# kernels: work and roofline


def kernel_work(k: dict) -> Tuple[float, float]:
    """(FLOPs, HBM bytes) of one lowered kernel, at first order: every
    operand read once and every output written once."""
    ty = k["type"]
    eb = DTYPE_BYTES.get(k.get("dtype", "bf16"), 2.0)
    if ty == "gemm":
        m, n, kk = k["m"], k["n"], k["k"]
        return 2.0 * m * n * kk, (m * kk + kk * n) * eb + m * n * 2.0
    if ty == "grouped_gemm":
        ms, n, kk = k["ms"], k["n"], k["k"]
        rows = sum(ms)
        return 2.0 * rows * n * kk, len(ms) * kk * n * eb + rows * (kk * eb + n * 2.0)
    if ty == "attention":
        b, hq, hkv, sq, skv, d = k["batch"], k["heads_q"], k["heads_kv"], k["sq"], k["skv"], k["d"]
        frac = 1.0
        if k.get("causal"):
            # Query i sees skv - sq + i + 1 keys: the mean is skv - (sq - 1) / 2.
            frac = (skv - (sq - 1) / 2.0) / skv
        dv = 512 if d > 256 else d  # MLA: V is the 512-wide latent
        flops = 2.0 * b * hq * sq * skv * (d + dv) * frac
        kv = hkv * skv * (d if (k.get("kv_tied") or d > 256) else d + dv)
        return flops, b * (hq * sq * (d + dv) + kv) * 2.0
    if ty == "elementwise":
        n = k["n"]
        return n * k.get("flops_per_elem", 1), n * k.get("elem_bytes", 2.0) * (k.get("inputs", 1) + 1)
    if ty == "row_norm":
        return 4.0 * k["rows"] * k["cols"], 2.0 * k["rows"] * k["cols"] * k.get("elem_bytes", 2.0)
    if ty == "conv1d":
        return (2.0 * k["batch"] * k["dim"] * k["seqlen"] * k["width"],
                2.0 * k["batch"] * k["dim"] * (2 * k["seqlen"] + k["width"]))
    if ty == "ssm_state_update":
        st = k["batch"] * k["heads"] * k["headdim"] * k["d_state"]
        return 6.0 * st, 8.0 * st
    if ty == "ssd":
        b, s, h, p, n = k["batch"], k["seqlen"], k["heads"], k["headdim"], k["d_state"]
        stage = k["stage"]
        flops = {"chunk_state": 2.0 * b * s * h * p * n, "chunk_scan": 2.0 * b * s * h * (p * n + k["chunk"] * p),
                 "bmm_chunk": 2.0 * b * s * k["chunk"] * k["groups"] * n}.get(stage, 2.0 * b * s * h)
        return flops, 2.0 * b * s * h * p * 2.0
    raise ValueError(f"no closed form for kernel type {ty!r}")


def kernel_time(k: dict, hw: Hardware) -> float:
    """Roofline time of one kernel: ``max(F / C, B / beta)``."""
    if k["type"] == "fixed":
        return k["ns"] * 1e-9
    f, b = kernel_work(k)
    vector = k["type"] in ("elementwise", "row_norm", "conv1d", "ssm_state_update")
    # Vector kernels run on the FP32 pipes; they are memory bound long before
    # those matter, so only their bytes count.
    return max(0.0 if vector else f / hw.peak(k.get("dtype", "bf16")), b / hw.hbm_bw)


# ---------------------------------------------------------------------------
# roofline replay of a lowered program


def _expand(ops: List[dict]) -> List[dict]:
    out: List[dict] = []
    for op in ops:
        if op.get("op") == "loop":
            body = _expand(op["body"])
            out += body * int(op["times"])
        else:
            out.append(op)
    return out


def _group(g, rank: int, world: int) -> Tuple[int, ...]:
    if isinstance(g, list):
        return tuple(g)
    if g == "world":
        return tuple(range(world))
    kind, n = g.split(":")
    n = int(n)
    if kind == "block":
        b = rank // n * n
        return tuple(range(b, b + n))
    return tuple(range(rank % n, world, n))


@dataclass
class Replay:
    """Result of :func:`replay`."""

    step: float
    #: Per rank: finish time, kernel time (all streams), collective time, p2p time.
    ranks: List[dict]
    #: Rank 0's kernel time by kernel type, and collective time by kind.
    kernels: Dict[str, float]
    collectives: Dict[str, float]
    #: With ``spans=True``: (rank, stream, kind, name, start_s, end_s) of
    #: every kernel, collective and transfer, for timelines.
    spans: List[tuple] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"step_us": self.step * 1e6, "ranks": self.ranks[:8], "kernels_us": {k: v * 1e6 for k, v in self.kernels.items()},
                "collectives_us": {k: v * 1e6 for k, v in self.collectives.items()}}


def replay(wl: dict, hw: Hardware, *, spans: bool = False) -> Replay:
    """Time a lowered workload on an ideal machine.

    Every kernel takes its roofline time (plus the launch gap), every
    collective its closed form, every point-to-point ``alpha + bytes / W``;
    streams of a rank run concurrently without slowing each other, waits
    match records in order (CUDA events), a collective starts when its last
    member posts it and a transfer when both ends have posted. SPMD programs
    are symmetric, so one rank is replayed and its collectives start when it
    posts them.
    """
    if "spmd" in wl:
        world = wl["spmd"]["world"]
        progs = [wl["spmd"]["program"]["streams"]]
        symmetric = True
    else:
        world = len(wl["ranks"])
        progs = [r["streams"] for r in wl["ranks"]]
        symmetric = False
    streams = [[_expand(s) for s in p] for p in progs]
    nr = len(streams)
    t = [[0.0] * len(s) for s in streams]
    ip = [[0] * len(s) for s in streams]
    records: Dict[Tuple[int, str], List[float]] = {}
    waited: Dict[Tuple[int, int, str], int] = {}
    coll_seq: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    coll_post: Dict[Tuple[Tuple[int, ...], int], Dict[int, float]] = {}
    coll_done: Dict[Tuple[Tuple[int, ...], int], float] = {}
    pend_coll: Dict[Tuple[int, int], Tuple[Tuple[int, ...], int]] = {}
    p2p_seq: Dict[Tuple[int, int, str], int] = {}
    p2p_post: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    pend_p2p: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    stats = [dict(kernel=0.0, collective=0.0, p2p=0.0) for _ in range(nr)]
    out_spans: List[tuple] = []
    by_kernel: Dict[str, float] = {}
    by_coll: Dict[str, float] = {}

    def coll_cost(op) -> float:
        g = _group(op["group"], 0, world) if symmetric else None
        n = len(g) if g is not None else len(op["group"]) if isinstance(op["group"], list) else None
        if n is None:
            n = len(_group(op["group"], 0, world))
        return collective(op["kind"], op["bytes"], n, hw, algo=op.get("algo"))[0]

    progress = True
    while progress:
        progress = False
        for r in range(nr):
            for s, ops in enumerate(streams[r]):
                while ip[r][s] < len(ops):
                    op = ops[ip[r][s]]
                    kind = op["op"]
                    if kind == "kernel":
                        k = op["kernel"]
                        d = kernel_time(k, hw) + hw.launch
                        if spans:
                            out_spans.append((r, s, "kernel", op.get("name") or k["type"], t[r][s], t[r][s] + d))
                        t[r][s] += d
                        stats[r]["kernel"] += d
                        if r == 0:
                            by_kernel[k["type"]] = by_kernel.get(k["type"], 0.0) + d
                    elif kind == "delay":
                        t[r][s] += op["ns"] * 1e-9
                    elif kind == "record":
                        records.setdefault((r, op["event"]), []).append(t[r][s])
                    elif kind == "wait":
                        key = (r, s, op["event"])
                        n_done = waited.get(key, 0)
                        rec = records.get((r, op["event"]), [])
                        if len(rec) <= n_done:
                            break
                        t[r][s] = max(t[r][s], rec[n_done])
                        waited[key] = n_done + 1
                    elif kind == "collective":
                        d = coll_cost(op)
                        if symmetric:
                            if spans:
                                out_spans.append((r, s, "collective", op.get("name") or op["kind"], t[r][s], t[r][s] + d))
                            t[r][s] += d
                        else:
                            if (r, s) not in pend_coll:
                                g = _group(op["group"], r, world)
                                seq = coll_seq.get((r, g), 0)
                                coll_seq[(r, g)] = seq + 1
                                coll_post.setdefault((g, seq), {})[r] = t[r][s]
                                pend_coll[(r, s)] = (g, seq)
                            g, seq = pend_coll[(r, s)]
                            posts = coll_post[(g, seq)]
                            if len(posts) < len(g):
                                break
                            done = coll_done.setdefault((g, seq), max(posts.values()) + d)
                            if spans:
                                out_spans.append((r, s, "collective", op.get("name") or op["kind"], done - d, done))
                            t[r][s] = done
                            del pend_coll[(r, s)]
                        stats[r]["collective"] += d
                        if r == 0:
                            by_coll[op["kind"]] = by_coll.get(op["kind"], 0.0) + d
                    elif kind in ("send", "recv"):
                        peer = (r + op["peer"]) % world if op.get("relative") else op["peer"]
                        d = p2p(op["bytes"], hw, cross_domain=(r // hw.domain) != (peer // hw.domain))
                        if symmetric:
                            if spans:
                                out_spans.append((r, s, kind, f"{kind} {op['bytes'] / 2**20:.1f} MiB", t[r][s], t[r][s] + d))
                            t[r][s] += d
                        else:
                            src, dst = (r, peer) if kind == "send" else (peer, r)
                            if (r, s) not in pend_p2p:
                                seq = p2p_seq.get((src, dst, kind), 0)
                                p2p_seq[(src, dst, kind)] = seq + 1
                                p2p_post.setdefault((src, dst, seq), {})[kind] = t[r][s]
                                pend_p2p[(r, s)] = (src, dst, seq)
                            key = pend_p2p[(r, s)]
                            posts = p2p_post[key]
                            if len(posts) < 2:
                                break
                            t[r][s] = max(posts.values()) + d
                            if spans:
                                out_spans.append((r, s, kind, f"{kind} {op['bytes'] / 2**20:.1f} MiB", t[r][s] - d, t[r][s]))
                            del pend_p2p[(r, s)]
                        stats[r]["p2p"] += d
                    else:
                        raise ValueError(f"replay: unknown op {kind!r}")
                    ip[r][s] += 1
                    progress = True
    stuck = [(r, s) for r in range(nr) for s in range(len(streams[r])) if ip[r][s] < len(streams[r][s])]
    if stuck:
        r, s = stuck[0]
        raise RuntimeError(f"replay deadlock: rank {r} stream {s} blocked on {streams[r][s][ip[r][s]]}")
    ranks = [dict(rank=r, finish_us=max(t[r] or [0.0]) * 1e6, kernel_us=stats[r]["kernel"] * 1e6,
                  collective_us=stats[r]["collective"] * 1e6, p2p_us=stats[r]["p2p"] * 1e6) for r in range(nr)]
    return Replay(step=max(max(x) for x in t if x), ranks=ranks, kernels=by_kernel, collectives=by_coll,
                  spans=out_spans)


# ---------------------------------------------------------------------------
# the textbook estimate


@dataclass(frozen=True)
class Plan:
    """A parallel layout. ``world = tp * cp * pp * dp``; ``ep`` shards the
    experts of MoE layers over that many of the ``dp * cp`` ranks (attention
    stays data parallel, the DeepSeek layout), so it does not add GPUs."""

    tp: int = 1
    #: Megatron sequence parallelism alongside TP.
    sp: bool = False
    cp: int = 1
    #: ``ring`` (ring attention) or ``ulysses`` (all-to-all over heads).
    cp_kind: str = "ring"
    pp: int = 1
    #: ``gpipe``, ``1f1b``, ``interleaved`` (``virtual`` chunks per GPU) or ``zb`` (ZB-H1).
    schedule: str = "1f1b"
    virtual: int = 1
    dp: int = 1
    #: 0 DDP, 1 / 2 / 3 ZeRO stage (3 = FSDP).
    zero: int = 2
    #: HSDP shard-group size (0 = all of ``dp``).
    shard: int = 0
    ep: int = 1

    @property
    def world(self) -> int:
        # Inference EP (Plan(ep=72)) is its own replica group of ep ranks.
        return max(self.tp * self.cp * self.pp * self.dp, self.ep)

    def parallel(self, comm_sms: int = 20) -> Parallel:
        """The simulator lowering's :class:`~rackscale.lower.Parallel` for this plan."""
        return Parallel(tp=self.tp, ep=self.ep, dp=self.dp, pp=self.pp, zero=self.zero, shard=self.shard,
                        sp=self.sp, comm_sms=comm_sms)

    def label(self, train: bool = True) -> str:
        parts = []
        for k, d in (("tp", 1), ("cp", 1), ("pp", 1), ("dp", 1), ("ep", 1)):
            v = getattr(self, k)
            if v != d:
                parts.append(f"{k}{v}")
        if self.sp:
            parts.append("sp")
        if self.cp > 1:
            parts.append(self.cp_kind)
        if self.pp > 1 and train:
            parts.append(self.schedule + (f"v{self.virtual}" if self.virtual > 1 else ""))
        if self.dp > 1 and train:
            parts.append({0: "ddp", 1: "zero1", 2: "zero2", 3: "fsdp"}[self.zero] +
                         (f"-hsdp{self.shard}" if self.shard and self.shard != self.dp else ""))
        return " ".join(parts) or "1 GPU"


@dataclass
class Term:
    """One line of an estimate: a time, the formula and its inputs."""

    name: str
    seconds: float
    formula: str
    #: ``compute``, ``memory``, ``exposed`` (on the critical path) or ``overlapped``.
    kind: str
    detail: Dict[str, float] = field(default_factory=dict)


def _model(model: Union[str, Model]) -> Model:
    return models.ALL[model] if isinstance(model, str) else model


def _layers(m: Model) -> int:
    return len(m.block_list()) if m.blocks is not None else m.layers


def _mixers(m: Model) -> int:
    """Blocks that end in a TP all-reduce: every block of a block stack, or
    attention + FFN per layer."""
    return len(m.blocks) if m.blocks is not None else 2 * m.layers


def _moe_layers(m: Model) -> int:
    return 0 if not m.moe else m.layers - m.moe.first_dense_layers


def _param_split(m: Model, plan: Plan) -> Tuple[float, float]:
    """(dense, expert) parameters held per GPU before any DP sharding: TP
    splits everything, PP splits the layers, EP splits the routed experts."""
    total = m.total_params()
    expert = 0.0
    if m.moe:
        mo = m.moe
        expert = 3.0 * m.hidden * mo.expert_ffn * mo.experts * _moe_layers(m)
    dense = total - expert
    return dense / (plan.tp * plan.pp), expert / (plan.tp * plan.pp * plan.ep)


def estimate(model: Union[str, Model], hw: Hardware, phase: str, plan: Plan = Plan(), *, micro_batch: int = 1,
             seqlen: int = 4096, micro_batches: int = 1, batch: int = 1, context: int = 4096,
             dtype: str = "bf16", optimizer: bool = True) -> dict:
    """Closed-form step time, throughput, MFU and memory of one step.

    ``phase`` is ``train`` (``micro_batch`` sequences of ``seqlen`` tokens per
    micro-batch per DP replica, ``micro_batches`` of them per step),
    ``prefill`` (``batch`` prompts of ``seqlen``) or ``decode`` (``batch``
    sequences per model replica at ``context`` cached tokens; per rank under
    EP, as in :func:`rackscale.lower.decode_step`).

    The step is the sum of the ``compute``/``memory`` roofline (the larger of
    the two), every ``exposed`` term, and any ``overlapped`` term's excess
    over the compute it hides behind. Which communication overlaps follows
    the simulator's lowering: TP, EP and Ulysses collectives sit on the
    compute stream, DP gradient traffic and FSDP gathers run on their own
    streams, ring-attention transfers hide behind the attention steps.
    """
    m = _model(model)
    if phase == "train":
        return _estimate_train(m, hw, plan, micro_batch, seqlen, micro_batches, dtype, optimizer)
    if phase in ("decode", "prefill"):
        return _estimate_infer(m, hw, phase, plan, batch, seqlen, context, dtype)
    raise ValueError(f"unknown phase {phase!r} (train, prefill, decode)")


def _combine(terms: List[Term]) -> Tuple[float, float, float]:
    """(step, no-overlap bound, perfect-overlap bound)."""
    comp = sum(x.seconds for x in terms if x.kind == "compute")
    mem = sum(x.seconds for x in terms if x.kind == "memory")
    roof = max(comp, mem)
    exposed = sum(x.seconds for x in terms if x.kind == "exposed")
    over = [x for x in terms if x.kind == "overlapped"]
    hidden_excess = sum(max(0.0, x.seconds - x.detail.get("window", roof)) for x in over)
    step = roof + exposed + hidden_excess
    no_overlap = comp + mem + exposed + sum(x.seconds for x in over)
    perfect = max(roof, exposed + hidden_excess, max([x.seconds for x in over] or [0.0]))
    return step, no_overlap, perfect


def _estimate_train(m: Model, hw: Hardware, plan: Plan, mb: int, seqlen: int, mbs: int, dtype: str,
                    optimizer: bool) -> dict:
    tp, cp, pp, dp, ep = plan.tp, plan.cp, plan.pp, plan.dp, plan.ep
    world = plan.world
    C = hw.peak(dtype)
    eb = DTYPE_BYTES.get(dtype, 2.0)
    L = _layers(m)
    tokens = mb * seqlen * mbs * dp  # per step, global
    t_gpu = mb * seqlen // cp  # tokens one GPU's layers see per micro-batch
    flops_gpu = lower._model_flops(m, tokens, seqlen // 2, True, lm_head=False) / world
    dense_p, expert_p = _param_split(m, plan)
    p_gpu = dense_p + expert_p
    s = plan.shard or dp
    terms: List[Term] = []
    t_math = flops_gpu / C
    terms.append(Term("matmul FLOPs", t_math, "6 N_active T / world / C  (+ attention 12 L H d S/2 T)", "compute",
                      dict(flops=flops_gpu, C=C)))
    # HBM: every weight read in forward, dgrad and wgrad of each micro-batch;
    # activations are kernel-internal at first order.
    t_w = 3 * mbs * p_gpu * eb / hw.hbm_bw
    terms.append(Term("weight reads", t_w, "3 m P_gpu b / beta", "memory", dict(p_gpu=p_gpu)))
    # TP: two collectives per mixer per pass (fwd, bwd), on the compute stream.
    tp_t = 0.0
    if tp > 1:
        a = t_gpu * m.hidden * 2.0
        n_coll = 2 * _mixers(m) // pp * mbs
        if plan.sp:
            one = collective("reduce_scatter", a, tp, hw)[0] + collective("all_gather", a, tp, hw)[0]
            f = "2 L_stage m [RS + AG](T D 2)"
        else:
            one, f = collective("all_reduce", a, tp, hw)
            f = f"2 L_stage m x AR(T D 2): {f}"
        tp_t = n_coll * one
        terms.append(Term("TP activations", tp_t, f, "exposed", dict(bytes=a, count=n_coll, each=one)))
    # EP: dispatch + combine per MoE layer per pass.
    ep_t = 0.0
    if m.moe and ep > 1:
        mo = m.moe
        a = t_gpu * mo.topk * m.hidden * 2.0
        n_coll = 2 * 2 * _moe_layers(m) // pp * mbs
        one = collective("all_to_all", a, ep, hw)[0]
        ep_t = n_coll * one
        terms.append(Term("EP all-to-all", ep_t, "4 L_moe m x A2A(T k D 2)", "exposed", dict(bytes=a, count=n_coll)))
    # CP.
    cp_t = 0.0
    n_attn = m.attn_layers() // pp
    if cp > 1 and n_attn:
        c = t_gpu
        hq, hkv, d = m.heads // tp, max(1, m.kv_heads // tp), m.head_dim
        if plan.cp_kind == "ulysses":
            a_in = c * (hq + 2 * hkv) * d * 2.0
            a_out = c * hq * d * 2.0
            one = collective("all_to_all", a_in, cp, hw)[0] + collective("all_to_all", a_out, cp, hw)[0]
            cp_t = 2 * n_attn * mbs * one
            terms.append(Term("Ulysses all-to-all", cp_t, "2 L_attn m [A2A(c(H+2Hkv)d2) + A2A(c H d 2)]", "exposed"))
        else:
            kv = c * 2 * hkv * d * 2.0
            x = p2p(kv, hw)
            step_c = 4.0 * c * (c / 2) * hq * d / C  # zig-zag: half a chunk of keys per step
            # forward K,V; backward K,V out and dK,dV back: 3 transfers vs 3.5x the compute
            exposed = (cp - 1) * (max(0.0, x - step_c) + max(0.0, 2 * x - 2.5 * step_c))
            cp_t = n_attn * mbs * exposed
            terms.append(Term("ring attention (exposed part)", cp_t,
                              "L_attn m (cp-1) [max(0, KV/W - 4c(c/2)Hd/C) + bwd]", "exposed",
                              dict(kv_bytes=kv, transfer=x, step_compute=step_c)))
    # Pipeline.
    stage_mb = (max(t_math, t_w) + tp_t + ep_t + cp_t) / mbs
    bubble = 0.0
    if pp > 1:
        if plan.schedule == "zb":
            bubble = (pp - 1) * stage_mb / 3.0
            f = "(p-1)(t_F + t_B - t_W), t_F = t_B = t_W"
        else:
            v = plan.virtual if plan.schedule == "interleaved" else 1
            bubble = (pp - 1) * stage_mb / v
            f = "(p-1)/v (t_F + t_B + t_W)"
        terms.append(Term("pipeline bubble", bubble, f, "exposed",
                          dict(fraction=bubble / (bubble + mbs * stage_mb))))
        a = t_gpu * m.hidden * 2.0 / tp
        fill = 2 * (pp - 1) * p2p(a, hw, cross_domain=world > hw.domain)
        terms.append(Term("pipeline fill/drain transfers", fill, "2 (p-1) (alpha + T D 2 / tp / W)", "exposed"))
    # DP / ZeRO / FSDP / HSDP.
    window_bwd = 2.0 / 3.0 * (max(t_math, t_w) + tp_t + ep_t + cp_t)
    if dp > 1:
        g_dense = 2.0 * dense_p
        g_exp = 2.0 * expert_p
        rep_e = max(1, dp * cp // ep)  # replicas of each expert
        if plan.zero == 0:
            t_g = collective("all_reduce", g_dense, dp * cp, hw)[0]
            if g_exp:
                t_g += collective("all_reduce", g_exp, rep_e, hw)[0]
            terms.append(Term("DP gradient all-reduce", t_g, "AR(2 P_gpu) over dp", "overlapped",
                              dict(window=window_bwd)))
        elif plan.zero in (1, 2):
            t_rs = collective("reduce_scatter", g_dense, dp * cp, hw)[0]
            t_ag = collective("all_gather", g_dense, dp * cp, hw)[0]
            if g_exp:
                t_rs += collective("reduce_scatter", g_exp, rep_e, hw)[0]
                t_ag += collective("all_gather", g_exp, rep_e, hw)[0]
            terms.append(Term("ZeRO gradient reduce-scatter", t_rs, "RS(2 P_gpu) over dp", "overlapped",
                              dict(window=window_bwd)))
            if optimizer:
                terms.append(Term("ZeRO parameter all-gather", t_ag, "AG(2 P_gpu) over dp, after the optimizer",
                                  "exposed"))
        else:
            n = s
            t_ag = 2 * mbs * collective("all_gather", g_dense + g_exp, n, hw)[0]
            t_rs = mbs * collective("reduce_scatter", g_dense + g_exp, n, hw)[0]
            if dp // n > 1:
                t_rs += mbs * collective("all_reduce", (g_dense + g_exp) / n, dp // n, hw)[0]
            terms.append(Term("FSDP parameter all-gathers (fwd + bwd)", t_ag, "2 m AG(2 P_gpu) over shard group",
                              "overlapped", dict(window=max(t_math, t_w) + tp_t + ep_t + cp_t)))
            terms.append(Term("FSDP gradient reduce-scatter" + (" + HSDP all-reduce" if dp // n > 1 else ""), t_rs,
                              "m RS(2 P_gpu) over shard [+ AR(2 P_gpu / s) over replicas]", "overlapped",
                              dict(window=window_bwd)))
            first = collective("all_gather", (g_dense + g_exp) / max(1, L // pp), n, hw)[0]
            terms.append(Term("FSDP first-layer gather", first, "AG(2 P_layer): nothing to hide behind", "exposed"))
    if optimizer:
        p_opt = p_gpu / (s if (dp > 1 and plan.zero >= 1) else 1)
        terms.append(Term("Adam update", 30.0 * p_opt / hw.hbm_bw, "30 B/param (4 FP32 in, 3 FP32 + 1 BF16 out) / beta",
                          "exposed", dict(params=p_opt)))
    step, no_ov, perfect = _combine(terms)
    mem = lower.memory(m, plan.parallel(), phase="train", batch=mb, seqlen=seqlen, micro_batches=mbs,
                       schedule="gpipe" if plan.schedule == "gpipe" else "1f1b") if not (m.moe and ep > 1) else {}
    if mem and cp > 1:
        mem["activations"] /= cp  # each CP rank holds 1/cp of every sequence
    out = _finish(m, hw, plan, "train", terms, step, no_ov, perfect, flops_gpu, dtype, mem)
    out.update(tokens_per_step=tokens, tok_s_gpu=tokens / step / world, tok_s=tokens / step,
               micro_batch=mb, seqlen=seqlen, micro_batches=mbs)
    # The DP critical batch (scaling book ch. 5): gradient traffic hides
    # behind the backward pass when tokens per GPU exceed C / W (ring, BF16).
    out["critical"] = dict(
        dp_tokens_per_gpu=dp_critical_tokens(hw, plan.zero, dp * cp, dtype),
        tokens_per_gpu=tokens / world,
        tp_max=_tp_crit(m, hw, dtype),
    )
    return out


def dp_critical_tokens(hw: Hardware, zero: int, n: int, dtype: str = "bf16") -> float:
    """Tokens per GPU per step above which data-parallel traffic hides
    behind compute (BF16 gradients and weights, large ``n``).

    The backward pass does ``4 P T`` FLOPs per GPU in ``4 P T / C``. DDP's
    gradient all-reduce moves ``2 x 2P`` bytes over a ring (``2P`` with
    NVLS), ZeRO-1/2's reduce-scatter ``2P``: hidden when ``T >= C / W`` or
    ``C / 2W``. FSDP also gathers ``2P`` in the forward (``2 P T / C``) and
    ``2P`` + scatters ``2P`` in the backward, so it needs ``T >= C / W``.
    """
    C, W = hw.peak(dtype), hw.link_bw(n)
    nvls = hw.nvls and n <= hw.domain
    if zero == 0:
        return C / (2 * W) if nvls else C / W
    if zero in (1, 2):
        return C / (2 * W)
    return C / W


def _tp_crit(m: Model, hw: Hardware, dtype: str) -> float:
    """TP degree at which a layer's all-reduces take as long as its matmuls
    (dense transformer, NVLS all-reduce: ``n - 1 = W_layer W / (4 C)``)."""
    per_tok = m.attn_params() + m.mlp_params()
    # matmul time 2 T P / (n C) vs two all-reduces 2 x 2 T D / W (NVLS)
    w = hw.nvlink_bw
    if hw.nvls:
        return 2.0 * per_tok * w / (4.0 * m.hidden * hw.peak(dtype))
    return 1.0 + 2.0 * per_tok * w / (8.0 * m.hidden * hw.peak(dtype))


def _estimate_infer(m: Model, hw: Hardware, phase: str, plan: Plan, batch: int, seqlen: int, context: int,
                    dtype: str) -> dict:
    tp, pp, ep, cp = plan.tp, plan.pp, plan.ep, plan.cp
    par = Parallel(tp=tp, ep=ep, pp=pp)
    world = plan.world
    C = hw.peak(dtype)
    eb = DTYPE_BYTES.get(dtype, 2.0)
    decode = phase == "decode"
    sq = 1 if decode else seqlen
    skv = context if decode else seqlen
    rows = batch * sq // (cp if not decode else 1)  # tokens through one GPU's GEMMs
    # Sequences the step serves across the whole replica group (each DP
    # group of the plan serves its own batch).
    seqs = batch * (ep if ep > 1 else plan.dp)
    tokens = seqs * sq
    flops = lower._model_flops(m, tokens, skv if decode else seqlen // 2, False, lm_head=decode) / world
    terms: List[Term] = [Term("matmul + attention FLOPs", flops / C, "2 N_active T (+ attention) / world / C", "compute",
                              dict(flops=flops))]
    # Weights actually read: everything but routed experts nobody routed to.
    dense_p, expert_p = _param_split(m, plan)
    if m.moe:
        mo = m.moe
        local = max(1, mo.experts // max(1, ep))
        recv = rows * mo.topk * (1 if ep > 1 else 1)
        touched = min(local, recv) / local
        expert_read = expert_p * touched
    else:
        expert_read = 0.0
    head = (m.vocab * m.hidden / tp) if decode else 0.0
    w_bytes = (dense_p - (m.vocab * m.hidden / tp if not decode else 0.0) + expert_read) * eb
    kv = batch * skv * lower._kv_bytes_per_token(m, par) / (1 if decode else cp) if m.attn_layers() else 0.0
    if decode:
        kv_read = kv
        f_kv = "B K kv_bytes_per_token / beta"
    else:
        kv_read = kv / 2  # written once (read back by attention inside the kernel)
        f_kv = "KV written once"
    terms.append(Term("weight reads", w_bytes / hw.hbm_bw, "P_gpu b / beta", "memory", dict(bytes=w_bytes)))
    if kv_read:
        terms.append(Term("KV cache", kv_read / hw.hbm_bw, f_kv, "memory", dict(bytes=kv_read)))
    del head
    if tp > 1:
        a = rows * m.hidden * 2.0
        one, f = collective("all_reduce", a, tp, hw)
        n = _mixers(m) // pp
        terms.append(Term("TP all-reduces", n * one, f"2 L x AR(T D 2): {f}", "exposed", dict(bytes=a, count=n, each=one)))
    if m.moe and ep > 1:
        mo = m.moe
        a = rows * mo.topk * m.hidden * 2.0
        n = 2 * _moe_layers(m) // pp
        one = collective("all_to_all", a, ep, hw)[0]
        terms.append(Term("EP dispatch + combine", n * one, "2 L_moe A2A(T k D 2)", "exposed", dict(bytes=a, count=n)))
    if pp > 1:
        a = rows * m.hidden * 2.0
        terms.append(Term("PP hops", (pp - 1) * p2p(a, hw), "(p-1)(alpha + T D 2 / W)", "exposed"))
    if cp > 1 and not decode and m.attn_layers():
        c = seqlen // cp
        hq, hkv, d = m.heads, m.kv_heads, m.head_dim
        if plan.cp_kind == "ulysses":
            one = (collective("all_to_all", c * (hq + 2 * hkv) * d * 2.0, cp, hw)[0]
                   + collective("all_to_all", c * hq * d * 2.0, cp, hw)[0])
            terms.append(Term("Ulysses all-to-all", m.attn_layers() * one, "L_attn [A2A(qkv) + A2A(out)]", "exposed"))
        else:
            x = p2p(c * 2 * hkv * d * 2.0, hw)
            step_c = 4.0 * c * (c / 2) * hq * d / C
            terms.append(Term("ring attention (exposed part)", m.attn_layers() * (cp - 1) * max(0.0, x - step_c),
                              "L_attn (cp-1) max(0, KV/W - step)", "exposed", dict(transfer=x, step_compute=step_c)))
    step, no_ov, perfect = _combine(terms)
    mem = lower.memory(m, par, phase=phase, batch=batch, context=context, seqlen=seqlen)
    if pp > 1:
        # The terms above are one stage's share of one batch. A token
        # crosses all pp stages; with pp batches in flight every stage stays
        # busy, so a stage-time later the next batch's token is out.
        stage = step
        step = pp * stage
        no_ov, perfect = pp * no_ov, pp * perfect
        tokens *= pp
        flops *= pp
        for k in ("kv_cache", "ssm_state"):
            if k in mem:
                mem[k] *= pp
    out = _finish(m, hw, plan, phase, terms, step, no_ov, perfect, flops, dtype, mem)
    replicas = max(1, hw.gpus // world)
    out.update(batch=batch, context=context, seqlen=seqlen, replicas=replicas, tokens_per_step=tokens)
    if decode:
        out.update(tok_s_gpu=tokens / step / world, tok_s_user=1.0 / step, tpot_ms=step * 1e3,
                   tok_s_rack=tokens / step * replicas)
        # Weight-bandwidth / compute crossover (scaling book ch. 7): the
        # batch at which 2 N B / C = N b / beta.
        out["critical"] = dict(batch=C * eb / (2.0 * hw.hbm_bw), batch_per_gpu=rows,
                               note="decode is weight-bandwidth bound below this many sequences per GPU")
    else:
        out.update(ttft_ms=step * 1e3, tok_s_gpu=tokens / step / world, tok_s_rack=tokens / step * replicas)
    return out


# ---------------------------------------------------------------------------
# plan search and the single-GPU-to-rack ladder


def _divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def train_plans(model: Union[str, Model], gpus: int, *, seqlen: int = 8192) -> List[Plan]:
    """Every training plan on ``gpus`` GPUs that divides the model: TP in
    {1, 2, 4, 8} (with and without SP), PP stages, CP for long sequences,
    DP with each ZeRO stage, HSDP groups of 8 when DP spans more."""
    m = _model(model)
    out = []
    n_layers = _layers(m)
    for tp in (1, 2, 4, 8):
        if gpus % tp or m.heads % tp or (m.kv_heads % tp and tp % m.kv_heads):
            continue
        for cp in ((1, 2, 4, 8) if seqlen >= 32768 else (1,)):
            for pp in (1, 2, 4, 8, 9, 16):
                if gpus % (tp * cp * pp) or pp > n_layers:
                    continue
                dp = gpus // (tp * cp * pp)
                zeros = (0, 1, 3) if pp == 1 else (1,)
                for zero in (zeros if dp > 1 else (1,)):
                    shards = [0] + ([8] if zero == 3 and dp > 8 and dp % 8 == 0 else [])
                    for shard in shards:
                        for sp in ((False, True) if tp > 1 else (False,)):
                            for sched in (("1f1b", "zb") if pp > 1 else ("1f1b",)):
                                out.append(Plan(tp=tp, sp=sp, cp=cp, pp=pp, schedule=sched, dp=dp, zero=zero,
                                                shard=shard))
    return out


def decode_plans(model: Union[str, Model], gpus: int) -> List[Plan]:
    """Decode replica layouts that fit in ``gpus``: TP up to the domain, PP
    for what TP cannot hold, EP with data-parallel attention for MoE."""
    m = _model(model)
    out = []
    for tp in (1, 2, 4, 8, 16, 32, 64, 72):
        if tp > gpus or m.heads % tp or (m.kv_heads % tp and tp % m.kv_heads):
            continue
        for pp in (1, 2, 4, 8):
            if tp * pp <= gpus:
                out.append(Plan(tp=tp, pp=pp))
    if m.moe:
        for ep in _divisors(gpus):
            if 1 < ep <= m.moe.experts:
                out.append(Plan(ep=ep))
    return out


def best_train(model: Union[str, Model], hw: Hardware, gpus: int, *, seqlen: int = 8192,
               global_batch: int = 256, dtype: str = "bf16") -> Optional[dict]:
    """The training plan with the most tokens/s per GPU on ``gpus`` GPUs
    that fits in HBM, at a fixed global batch of ``global_batch``
    sequences (micro-batches of one sequence; a plan's DP replicas split
    the global batch). ``None`` when nothing fits."""
    best = None
    hwg = replace(hw, gpus=gpus)
    for plan in train_plans(model, gpus, seqlen=seqlen):
        m = max(1, -(-global_batch // plan.dp))
        if plan.pp > 1 and m < plan.pp:
            continue
        e = estimate(model, hwg, "train", plan, micro_batch=1, seqlen=seqlen, micro_batches=m, dtype=dtype)
        if not e["fits"]:
            continue
        if best is None or e["tok_s_gpu"] > best["tok_s_gpu"]:
            best = e
    return best


def best_decode(model: Union[str, Model], hw: Hardware, gpus: int, *, context: int = 8192,
                min_tps_user: float = 0.0, batches=(1, 4, 16, 32, 64, 128, 256, 512), dtype: str = "bf16") -> Optional[dict]:
    """The decode layout and batch with the most tokens/s per GPU on
    ``gpus`` GPUs (whole replicas only) that fits in HBM and gives every
    user at least ``min_tps_user`` tokens/s."""
    best = None
    hwg = replace(hw, gpus=gpus)
    for plan in decode_plans(model, gpus):
        for b in batches:
            e = estimate(model, hwg, "decode", plan, batch=b, context=context, dtype=dtype)
            if not e["fits"] or e["tok_s_user"] < min_tps_user:
                continue
            e["tok_s_total"] = e["tok_s_rack"]
            if best is None or e["tok_s_total"] > best["tok_s_total"]:
                best = e
    return best


def ladder(model: Union[str, Model], system: str = "gb300", phase: str = "train", *,
           gpus=(1, 2, 4, 8, 16, 36, 72), seqlen: int = 8192, global_batch: int = 256, context: int = 8192,
           min_tps_user: float = 0.0, dtype: str = "bf16") -> dict:
    """From one GPU to the rack: the best plan at each GPU count, with
    step time, throughput, per-user speed, MFU and memory, and the scaling
    efficiency against the smallest count that fits (closed form)."""
    hw = hardware(system)
    rows = []
    for n in gpus:
        if n > hw.gpus:
            continue
        if phase == "train":
            e = best_train(model, hw, n, seqlen=seqlen, global_batch=global_batch, dtype=dtype)
        elif phase == "decode":
            e = best_decode(model, hw, n, context=context, min_tps_user=min_tps_user, dtype=dtype)
        else:
            raise ValueError("ladder phases: train, decode")
        if e is None:
            rows.append(dict(gpus=n, fits=False))
            continue
        row = dict(gpus=n, fits=True, plan=e["label"], plan_spec=e["plan"], step_ms=e["step_us"] / 1e3,
                   tok_s_gpu=e["tok_s_gpu"], mfu=e["mfu"], memory_gb=e["memory_total_gb"], dominant=e["dominant"],
                   terms={t["name"]: t["us"] / 1e3 for t in e["terms"]})
        if phase == "train":
            row.update(tok_s=e["tok_s"], tokens_per_step=e["tokens_per_step"])
        else:
            # Whole replicas only: GPUs left over count against throughput.
            row.update(tok_s=e["tok_s_total"], tok_s_gpu=e["tok_s_total"] / n, tok_s_user=e["tok_s_user"],
                       batch=e["batch"], replicas=e["replicas"])
        rows.append(row)
    base = next((r for r in rows if r["fits"]), None)
    for r in rows:
        if r["fits"] and base:
            r["scaling_efficiency"] = r["tok_s_gpu"] / base["tok_s_gpu"]
    return dict(model=_model(model).name, system=hw.name, phase=phase, rows=rows,
                assumptions=dict(seqlen=seqlen, global_batch=global_batch, context=context,
                                 min_tps_user=min_tps_user, dtype=dtype))


def _finish(m: Model, hw: Hardware, plan: Plan, phase: str, terms: List[Term], step: float, no_ov: float,
            perfect: float, flops_gpu: float, dtype: str, mem: dict) -> dict:
    mem_gb = {k: v / 1e9 for k, v in mem.items()}
    total = sum(mem_gb.values())
    dom = max(terms, key=lambda x: x.seconds if x.kind != "overlapped" else
              max(0.0, x.seconds - x.detail.get("window", 0.0)))
    return dict(
        model=m.name, hardware=hw.name, phase=phase, plan=asdict(plan), label=plan.label(phase == "train"),
        world=plan.world,
        dtype=dtype, step_us=step * 1e6, bounds_us=dict(no_overlap=no_ov * 1e6, perfect_overlap=perfect * 1e6),
        mfu=flops_gpu / (hw.peak(dtype) * step), dominant=dom.name,
        terms=[dict(name=x.name, us=x.seconds * 1e6, kind=x.kind, formula=x.formula,
                    **{k: v for k, v in x.detail.items()}) for x in terms],
        memory_gb=mem_gb, memory_total_gb=total, fits=total <= hw.hbm_bytes / 1e9 * 0.92,
    )


def explain(est: dict) -> str:
    """The estimate as a table a person can read."""
    lines = [f"{est['model']} {est['phase']} on {est['hardware']}: {est['label']} ({est['world']} GPUs)",
             f"step {est['step_us']:.0f} us   MFU {100 * est['mfu']:.1f}%   dominant: {est['dominant']}",
             f"bounds: perfect overlap {est['bounds_us']['perfect_overlap']:.0f} us, no overlap "
             f"{est['bounds_us']['no_overlap']:.0f} us"]
    for x in est["terms"]:
        lines.append(f"  {x['name']:<40} {x['us']:>11.1f} us  {x['kind']:<10}  {x['formula']}")
    if est.get("memory_gb"):
        parts = ", ".join(f"{k} {v:.1f}" for k, v in est["memory_gb"].items())
        lines.append(f"memory per GPU {est['memory_total_gb']:.1f} GB [{parts}]" + ("" if est["fits"] else "  DOES NOT FIT"))
    for k in ("tok_s_gpu", "tok_s_user", "ttft_ms", "tok_s_rack"):
        if k in est:
            lines.append(f"{k} = {est[k]:,.1f}")
    if est.get("critical"):
        lines.append("critical: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in est["critical"].items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# lowering + check against the simulator


def lower_plan(model: Union[str, Model], phase: str, plan: Plan, *, micro_batch: int = 1, seqlen: int = 4096,
               micro_batches: int = 1, batch: int = 1, context: int = 4096, layers: Optional[int] = None,
               optimizer: bool = False) -> dict:
    """The simulator workload for a plan (what :func:`check` runs)."""
    m = _model(model)
    if phase == "train":
        if plan.cp > 1:
            raise ValueError("context parallelism is lowered for prefill (lower.prefill_cp); training CP is analytic only")
        return lower.train_step(m, plan.parallel(), micro_batch, seqlen, layers=layers, micro_batches=micro_batches,
                                schedule="zb" if plan.schedule == "zb" else ("gpipe" if plan.schedule == "gpipe" else "1f1b"),
                                optimizer=optimizer)
    if phase == "prefill":
        if plan.cp > 1:
            return lower.prefill_cp(m, plan.cp, seqlen, layers=layers, kind=plan.cp_kind)
        return lower.prefill(m, plan.parallel(), batch, seqlen, layers=layers)
    if phase == "decode":
        return lower.decode_step(m, plan.parallel(), batch, context, layers=layers)
    raise ValueError(phase)


def check(model: Union[str, Model], system: str, phase: str, plan: Plan, *, layers: int = 2, **kw) -> dict:
    """Textbook vs roofline replay vs cycle-level simulation for one case.

    The model is cut to ``layers`` layers for all three (the simulator's
    fast mode still runs every distinct kernel at cycle level and every
    collective on the fabric). Returns the three step times, the signed
    errors ``(sim - replay) / replay`` and ``(sim - textbook) / textbook``,
    and the gap split into kernel, collective, exposure and launch parts.
    """
    import rackscale as rs
    from .sweep import SYSTEMS, _system_text
    m = _model(model).with_layers(layers)
    hw = hardware(system)
    wl = lower_plan(m, phase, plan, layers=None, **{k: v for k, v in kw.items() if k != "dtype"})
    rp = replay(wl, hw)
    est = estimate(m, hw, phase, plan, optimizer=kw.get("optimizer", False),
                   **{k: v for k, v in kw.items() if k in ("micro_batch", "seqlen", "micro_batches", "batch", "context")})
    sd = SYSTEMS.get(system)
    sys_text = _system_text(sd) if sd else system
    rep = rs.simulate(sys_text, wl, fast=True, launch_gap_ns=hw.launch * 1e9).raw
    sim_step = rep["makespan_us"] * 1e-6
    world = rep["world"]
    mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
    sim_k = mean([r["kernel_us"] for r in rep["ranks"]])
    # The kernel table counts cycles at the preset clock; rank spans include
    # the power-capped (DVFS) stretch. The difference is the clock.
    sim_k_table = sum(k["total_us"] for k in rep["kernels"]) / world
    rp_k = mean([r["kernel_us"] for r in rp.ranks])
    rp_c = mean([r["collective_us"] + r["p2p_us"] for r in rp.ranks])
    colls = rep.get("collectives", [])
    # Mean time one rank spends inside collectives, from post to done.
    sim_coll = sum((c["done_ps"] - c["last_post_ps"]) * 1e-6 * c["ranks"] for c in colls) / max(1, world)
    table: Dict[tuple, dict] = {}
    for c in colls:
        kind = _KIND.get(c["kind"], c["kind"])
        row = table.setdefault((kind, c["bytes"], c["ranks"]), dict(kind=kind, bytes=c["bytes"], ranks=c["ranks"],
                                                                    algo=c["algo"], n=0, sim_us=0.0))
        row["n"] += 1
        row["sim_us"] += (c["done_ps"] - c["last_post_ps"]) * 1e-6
    coll_rows = []
    for row in table.values():
        row["sim_us"] /= row["n"]
        if row["kind"] == "send_recv":
            row["closed_form_us"] = p2p(row["bytes"], hw) * 1e6
        else:
            row["closed_form_us"] = collective(row["kind"], row["bytes"], row["ranks"], hw)[0] * 1e6
        row["err"] = (row["sim_us"] - row["closed_form_us"]) / row["closed_form_us"]
        coll_rows.append(row)
    gap = dict(kernel_efficiency_us=sim_k_table - rp_k, power_clock_us=sim_k - sim_k_table,
               collectives_us=sim_coll - rp_c)
    gap["overlap_and_idle_us"] = (sim_step - rp.step) * 1e6 - sum(gap.values())
    return dict(
        model=_model(model).name, layers=layers, system=system, phase=phase, plan=asdict(plan), label=plan.label(),
        world=rep["world"], textbook_us=est["step_us"], replay_us=rp.step * 1e6, sim_us=sim_step * 1e6,
        err_vs_replay=(sim_step - rp.step) / rp.step, err_vs_textbook=(sim_step * 1e6 - est["step_us"]) / est["step_us"],
        textbook_vs_replay=(est["step_us"] - rp.step * 1e6) / (rp.step * 1e6),
        gap_us=gap, sim_kernel_us=sim_k, sim_kernel_table_us=sim_k_table, replay_kernel_us=rp_k,
        sim_collective_us=sim_coll, replay_collective_us=rp_c,
        sim_mfu=rep.get("mfu"), textbook_mfu=est["mfu"], dominant=est["dominant"], collectives=coll_rows,
        sim_kernels=[dict(name=k["name"], us=k["total_us"] / world, tflops=k["tflops"], gbps=k["gbps"])
                     for k in rep["kernels"][:12]],
        **{k: v for k, v in kw.items()},
    )


_KIND = {"AllReduce": "all_reduce", "AllGather": "all_gather", "ReduceScatter": "reduce_scatter",
         "AllToAll": "all_to_all", "Broadcast": "broadcast", "SendRecv": "send_recv"}
