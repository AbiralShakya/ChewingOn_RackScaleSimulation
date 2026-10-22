"""Persistent megakernels, from the schedule their own scheduler builds, at cycle level.

A megakernel runs a whole forward pass as one launch with one CTA per SM. A host-side
scheduler cuts every operation into tasks ("instructions") and gives each SM an ordered queue;
SMs order themselves through counters in global memory (a producer adds one after its store, a
consumer spins until the count is reached).

This module rebuilds that schedule for the Hazy Research low-latency Llama megakernel
(github.com/HazyResearch/Megakernels, `demos/low-latency-llama`: `make_dag`, then round-robin
assignment to SMs in DAG order, exactly as its `scheduler.py`) and turns it into a recorded-path
kernel (`.rpath`) that the simulator replays at cycle level:

* every CTA runs a loader warp, a storer warp and 16 consumer warps, as in Hazy's
  `matvec_pipeline.cuh`: the loader streams the task's weights in 64 KB iterations through a
  3-stage ring of 16 KB pages without waiting on other SMs; consumer warp 0 spins on the task's
  inter-SM counters (`GlobalWait`), then the consumers load and normalise the activation and work
  through the stages; the storer writes the outputs, waits for them to land and increments the
  counters it owns (`GlobalAdd`);
* stores to a marker region are timestamped by the simulator (`probes`), which gives the
  simulated per-SM task schedule: inter-SM wait reached, inputs ready, done.

What is exact and what is approximate:

* exact: the task list, its split into blocks, which SM runs which task in what order, every
  inter-SM dependency and its count, and the bytes every task moves (from the model shapes);
* approximate: the instruction stream inside a task is synthetic, with compute thinned to
  `fma_per_iter` FFMA per consumer warp per iteration (decode matvecs are bandwidth-bound). The
  SM, L2, HBM and inter-SM counter timing is the simulator's.

    from rackscale import megakernel
    sched = megakernel.hazy_llama_schedule(megakernel.LLAMA_3_2_1B, sm_count=148, context=40)
    megakernel.write_rpath(sched, "mk.rpath")
    # rsim viz path mk.rpath --gpu b300 -o mk.json; then megakernel.tasks_from_probes(...)
"""
import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LlamaShape:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int


LLAMA_3_2_1B = LlamaShape("Llama-3.2-1B", 2048, 16, 32, 8, 64, 8192, 128256)

OPS = ["NoOp", "RMS + QKV matvec + RoPE + KV append", "partial attention", "attention reduction",
       "O proj + residual", "RMS + up/gate matvec + SiLU", "down proj + residual", "RMS + LM head"]
CATS = ["other", "gemm", "attn", "attn", "gemm", "gemm", "gemm", "gemm"]
QKV, PARTIAL, OPROJ, UPGATE, DOWN, LMHEAD = 1, 2, 4, 5, 6, 7


@dataclass
class Task:
    op: int
    layer: int
    weight_bytes: int        # streamed from HBM (weights, KV cache)
    act_bytes: int           # activation vector read after the inputs are ready
    out_bytes: int
    waits: List[Tuple[int, int]] = field(default_factory=list)   # (counter, increments needed)
    adds: List[int] = field(default_factory=list)                 # counters this task increments
    sm: int = -1
    slot: int = -1


@dataclass
class Schedule:
    shape: LlamaShape
    sm_count: int
    context: int
    tasks: List[Task]
    queues: List[List[int]]
    counters: int


def _assert_div(a, b):
    assert a % b == 0, (a, b)
    return a // b


def hazy_llama_schedule(m: LlamaShape, sm_count: int = 148, context: int = 40, block: int = 16,
                        dtype_bytes: int = 2) -> Schedule:
    """The low-latency demo's DAG (`make_dag`) with its block sizes (16) and one attention partition
    per KV head (`skip_attn_reduction`), assigned round-robin to SMs in DAG order."""
    H, D = m.hidden, m.head_dim
    tasks: List[Task] = []
    counters = 0

    def counter():
        nonlocal counters
        counters += 1
        return counters - 1

    prev_out: Optional[int] = None   # counter of the previous layer's down projection
    prev_n = 0
    for L in range(m.layers):
        # QKV: (heads + 2 kv_heads) * head_dim outputs in blocks of 16, split evenly over the SMs
        n_qkv = _assert_div((m.heads + 2 * m.kv_heads) * D, block)
        per = n_qkv / sm_count
        kv_ctr = [counter() for _ in range(m.kv_heads)]   # K and V blocks of each KV head
        qkv_blocks = []
        for s in range(sm_count):
            a, b = round(s * per), round((s + 1) * per)
            t = Task(QKV, L, (b - a) * block * H * dtype_bytes, H * dtype_bytes, (b - a) * block * dtype_bytes,
                     waits=[(prev_out, prev_n)] if prev_out is not None else [])
            qkv_blocks.append((a, b))
            tasks.append(t)
        dims = _assert_div(D, block)
        need = [0] * m.kv_heads
        for h in range(m.kv_heads):
            kb = (m.heads + h) * D // block
            vb = (m.heads + m.kv_heads + h) * D // block
            mine = set(range(kb, kb + dims)) | set(range(vb, vb + dims))
            for i, (a, b) in enumerate(qkv_blocks):
                if mine & set(range(a, b)):
                    tasks[len(tasks) - sm_count + i].adds.append(kv_ctr[h])
                    need[h] += 1
        # partial attention: one task per KV head (its query heads), reading its K and V cache
        part_ctr = counter()
        for h in range(m.kv_heads):
            kv = 2 * context * D * dtype_bytes
            q = (m.heads // m.kv_heads) * D * dtype_bytes
            tasks.append(Task(PARTIAL, L, kv, q, q, waits=[(kv_ctr[h], need[h])], adds=[part_ctr]))
        # O projection: one task per 16-row block of the hidden size
        n_o = _assert_div(H, block)
        o_ctr = counter()
        for _ in range(n_o):
            tasks.append(Task(OPROJ, L, block * H * dtype_bytes, H * dtype_bytes, block * dtype_bytes,
                              waits=[(part_ctr, m.kv_heads)], adds=[o_ctr]))
        # up and gate: interleaved 16-row blocks of the FFN, block i on SM i % sm_count
        n_ug = _assert_div(m.ffn, block)
        ug_ctr = counter()
        for s in range(sm_count):
            nb = len(range(s, n_ug, sm_count))
            tasks.append(Task(UPGATE, L, 2 * nb * block * H * dtype_bytes, H * dtype_bytes,
                              nb * block * dtype_bytes, waits=[(o_ctr, n_o)], adds=[ug_ctr]))
        # down: (ffn / hidden) reduction columns x (hidden / 16) blocks, contiguous runs per SM
        n_down = _assert_div(H, block)
        cols = m.ffn // H
        jobs = [(c, b) for c in range(cols) for b in range(n_down)]
        done = 0
        down_ctr = counter()
        for s in range(sm_count):
            take = round((len(jobs) - done) / (sm_count - s))
            sl = jobs[done:done + take]
            sl = [j for j in sl if j[0] == sl[0][0]]
            tasks.append(Task(DOWN, L, len(sl) * block * H * dtype_bytes, H * dtype_bytes,
                              len(sl) * block * 4, waits=[(ug_ctr, sm_count)], adds=[down_ctr]))
            done += len(sl)
        prev_out, prev_n = down_ctr, sm_count
    # LM head: vocab in 16-row blocks, split evenly
    n_lm = _assert_div(m.vocab, block)
    per = n_lm / sm_count
    for s in range(sm_count):
        a, b = round(s * per), round((s + 1) * per)
        tasks.append(Task(LMHEAD, m.layers - 1, (b - a) * block * H * dtype_bytes, H * dtype_bytes,
                          (b - a) * block * dtype_bytes, waits=[(prev_out, prev_n)]))
    queues: List[List[int]] = [[] for _ in range(sm_count)]
    for i, t in enumerate(tasks):
        t.sm, t.slot = i % sm_count, len(queues[i % sm_count])
        queues[i % sm_count].append(i)
    return Schedule(m, sm_count, context, tasks, queues, counters)


# ---------------------------------------------------------------- .rpath ---
def _ctrl(hi: int, stall: int = 1, wbar: int = 7, rbar: int = 7, wait: int = 0, yld: int = 1) -> str:
    """The high word of an instruction with its control bits (stall, yield, scoreboards) set."""
    bits = stall | yld << 4 | wbar << 5 | rbar << 8 | wait << 11
    return f"0x{(hi & ((1 << 41) - 1)) | bits << 41:016x}"


# The synthetic listing: (text, low word, high word). Scoreboards: SB1 activation load,
# SB2 shared-memory loads, SB4 store read-back before the counter add.
_LISTING = [
    ("UBLKCP.S.G [UR4], [UR6], UR8 ;", "0x0000000806047ba6", _ctrl(0x8000000, 1)),
    ("SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR4], RZ ;", "0x000000ffff0079a7", _ctrl(0x8000004, 1)),
    ("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4], R2 ;", "0x00000002ff0075a7", _ctrl(0x8000144, 2)),
    ("FFMA R4, R5, R6, R4 ;", "0x0000000605047223", _ctrl(0x0000004, 4)),
    ("LDG.E.STRONG.GPU R8, desc[UR4][R6.64] ;", "0x0000000406087981", _ctrl(0xc1f5900, 1, wbar=0)),
    ("RED.E.ADD.STRONG.GPU desc[UR4][R6.64], R5 ;", "0x000000050600798e", _ctrl(0xc10e384, 1, wait=1 << 4)),
    ("STG.E desc[UR4][R6.64], R9 ;", "0x0000000902007986", _ctrl(0xc101904, 1)),
    ("EXIT ;", "0x000000000000794d", _ctrl(0x3800000, 5)),
    ("BAR.SYNC.DEFER_BLOCKING 0x3, 0x200 ;", "0x000c000000007b1d", _ctrl(0x0010000, 6)),
    ("LDG.E.64 R10, desc[UR4][R6.64] ;", "0x000000040606a981", _ctrl(0xc1e1b00, 1, wbar=1)),
    ("FFMA R4, R10, R10, R4 ;", "0x0000000a0a047223", _ctrl(0x0000004, 4, wait=1 << 1)),
    ("LDS.128 R12, [R2] ;", "0x00000000020c7984", _ctrl(0x0000c00, 1, wbar=2)),
    ("FFMA R4, R12, R5, R4 ;", "0x000000050c047223", _ctrl(0x0000004, 4, wait=1 << 2)),
    ("SHFL.BFLY PT, R5, R4, 0x10, 0x1f ;", "0x0e001f0004057f89", _ctrl(0x00e4000, 2)),
    ("STS [R3], R4 ;", "0x0000000403007388", _ctrl(0x0000800, 1)),
    ("LDG.E.STRONG.GPU R20, desc[UR4][R6.64] ;", "0x0000000406147981", _ctrl(0xc1f5900, 1, wbar=4)),
]
(I_BLK, I_ARRIVE, I_WAIT, I_FMA, I_FLAG, I_RED, I_STG, I_EXIT, I_BAR, I_LDACT, I_FMA_ACT, I_LDS,
 I_FMA_SM, I_SHFL, I_STS, I_READBACK) = range(len(_LISTING))


def listing(name: str) -> str:
    out = ["\tcode for sm_103a", f"\t\tFunction : {name}", "\t.headerflags\t@\"EF_CUDA_SM103\""]
    for i, (text, lo, hi) in enumerate(_LISTING):
        out.append(f"        /*{16 * i:04x}*/                   {text:<44} /* {lo} */")
        out.append(f"                                                                             /* {hi} */")
    return "\n".join(out) + "\n"


WEIGHTS = 0x7000_0000_0000
ACTS = 0x7100_0000_0000
OUTS = 0x7200_0000_0000
FLAGS = 0x7300_0000_0000
MARKS = 0x7400_0000_0000
MARK_START, MARK_READY, MARK_DONE = 0, 1, 2


def mark_addr(sm: int, slot: int, kind: int) -> int:
    return MARKS + ((sm << 12 | slot) * 4 + kind) * 32


def decode_mark(addr: int) -> Tuple[int, int, int]:
    i = (addr - MARKS) // 32
    kind, k = i % 4, i // 4
    return k >> 12, k & 0xFFF, kind


_OP_FMT = "<IIQBIIIHIH"
_OP_SIZE = struct.calcsize(_OP_FMT)


def _op(buf: bytearray, idx, active=1, payload=0, kind=0, mid=0, arg=0, n=0, run_bytes=0, addrs=()):
    buf += struct.pack(_OP_FMT, idx, active, payload, kind, mid, arg, n, 0, run_bytes, len(addrs))
    if addrs:
        buf += struct.pack(f"<{len(addrs)}Q", *addrs)


K_WAIT, K_ARRIVE, K_TXDONE, K_NAMED, K_GWAIT, K_GADD = 1, 2, 3, 5, 7, 8


def write_rpath(s: Schedule, out: str, *, iter_bytes: int = 65536, page: int = 16384, stages: int = 3,
                out_stages: int = 3, consumers: int = 16, inst_stages: int = 2, fma_per_iter: int = 16,
                lds_per_iter: int = 2, shfl_per_iter: int = 4) -> dict:
    """Write the schedule as a recorded-path kernel (every CTA explicit). Returns a summary.

    The warp roles and their synchronisation follow Hazy's `matvec_pipeline.cuh`:

    * loader: per 64 KB iteration (16 weight rows x 2048), waits for the stage to be released,
      expects its bytes on the stage's mbarrier and issues four 16 KB bulk copies. It does not wait
      on other SMs, so a task's first stages are already in flight while its inputs are pending;
      it runs at most `inst_stages` tasks ahead of the storer (the instruction ring).
    * consumers (16 warps): warp 0 spins on the task's global counters, then all consumers sync,
      load the activation vector from global memory, normalise it (RMS, one more sync) and, per
      iteration, wait for the stage, read it from shared memory, multiply-accumulate, reduce across
      lanes, write partial sums and release the stage and an output slot.
    * storer: per iteration waits for the 16 consumers and stores the output rows; after the last
      it waits for its stores to land (read-back) before adding one to each counter it owns.

    Compute is thinned to `fma_per_iter` FFMA per consumer warp per iteration (the real kernel
    does ~64 plus conversions): decode matvecs are bandwidth-bound and the issue slots this drops
    are idle in the real kernel too, but the per-iteration latency chain is kept.
    """
    pages_per_iter = iter_bytes // page
    full = list(range(stages))                               # weights arrived (tx)
    free = list(range(stages, 2 * stages))                    # weights finished (consumers)
    oarr = list(range(2 * stages, 2 * stages + out_stages))  # outputs arrived (consumers)
    ofree = [oarr[-1] + 1 + i for i in range(out_stages)]     # outputs finished (storer)
    islot = [ofree[-1] + 1 + i for i in range(inst_stages)]   # instruction slot free (storer)
    lanes = (1 << 32) - 1
    warps = []
    weight_off = 0
    total_ops = 0
    for sm, q in enumerate(s.queues):
        loader, storer = bytearray(), bytearray()
        cons = [bytearray() for _ in range(consumers)]
        c = 0   # iterations through the weight ring
        o = 0   # iterations through the output ring
        for k, ti in enumerate(q):
            t = s.tasks[ti]
            iters = max(1, -(-t.weight_bytes // iter_bytes))
            # loader
            if k >= inst_stages:
                _op(loader, I_WAIT, kind=K_WAIT, mid=islot[k % inst_stages], arg=k // inst_stages)
            left = t.weight_bytes
            for it in range(iters):
                st = (c + it) % stages
                if c + it >= stages:
                    _op(loader, I_WAIT, kind=K_WAIT, mid=free[st], arg=(c + it) // stages)
                nb = min(iter_bytes, left)
                left -= nb
                _op(loader, I_ARRIVE, kind=K_ARRIVE, mid=full[st], arg=nb, n=1)
                for p in range(pages_per_iter):
                    pb = min(page, nb - p * page)
                    if pb <= 0:
                        break
                    _op(loader, I_BLK, payload=pb, kind=K_TXDONE, mid=full[st], arg=pb, run_bytes=pb,
                        addrs=[WEIGHTS + weight_off])
                    weight_off += pb
            # consumers: inputs, activation, RMS, then the iterations
            for wi, w in enumerate(cons):
                if wi == 0:
                    _op(w, I_STG, addrs=[mark_addr(sm, k, MARK_START)])
                    for ctr, need in t.waits:
                        if need:
                            _op(w, I_FLAG, kind=K_GWAIT, mid=ctr, arg=need, addrs=[FLAGS + 128 * ctr])
                    _op(w, I_STG, addrs=[mark_addr(sm, k, MARK_READY)])
                _op(w, I_BAR, active=lanes, kind=K_NAMED, mid=3)
                act = ACTS + (t.layer * 8 + t.op) * 65536 + wi * (t.act_bytes // consumers)
                _op(w, I_LDACT, active=lanes, addrs=[act + 8 * l for l in range(32)])
                for _ in range(4):
                    _op(w, I_FMA_ACT, active=lanes)
                for _ in range(5):
                    _op(w, I_SHFL, active=lanes)
                _op(w, I_STS, active=lanes, addrs=[0x30000 + 4 * wi])
                _op(w, I_BAR, active=lanes, kind=K_NAMED, mid=3)
                for it in range(iters):
                    st, ost = (c + it) % stages, (o + it) % out_stages
                    _op(w, I_WAIT, kind=K_WAIT, mid=full[st], arg=(c + it) // stages + 1)
                    if o + it >= out_stages:
                        _op(w, I_WAIT, kind=K_WAIT, mid=ofree[ost], arg=(o + it) // out_stages)
                    base = (st * pages_per_iter + wi * pages_per_iter // consumers) * page
                    for j in range(lds_per_iter):
                        _op(w, I_LDS, active=lanes, addrs=[base + 512 * j + 16 * l for l in range(32)])
                    _op(w, I_FMA_SM, active=lanes)
                    for _ in range(fma_per_iter - 1):
                        _op(w, I_FMA, active=lanes)
                    for _ in range(shfl_per_iter):
                        _op(w, I_SHFL, active=lanes)
                    _op(w, I_STS, active=lanes, addrs=[0x30000 + 1024 * ost + 64 * wi + 4 * (l % 16)
                                                        for l in range(32)])
                    _op(w, I_ARRIVE, kind=K_ARRIVE, mid=oarr[ost], n=1)
                    _op(w, I_ARRIVE, kind=K_ARRIVE, mid=free[st], n=1)
            # storer
            for it in range(iters):
                ost = (o + it) % out_stages
                _op(storer, I_WAIT, kind=K_WAIT, mid=oarr[ost], arg=(o + it) // out_stages + 1)
                _op(storer, I_LDS, active=lanes, addrs=[0x30000 + 1024 * ost + 16 * l for l in range(32)])
                _op(storer, I_FMA_SM, active=lanes)
                _op(storer, I_STG, active=0xFFFF, addrs=[OUTS + ti * 4096 + it * 64 + 4 * l for l in range(16)])
                _op(storer, I_ARRIVE, kind=K_ARRIVE, mid=ofree[ost], n=1)
            if t.adds:
                _op(storer, I_READBACK, addrs=[OUTS + ti * 4096])
                for ctr in t.adds:
                    _op(storer, I_RED, kind=K_GADD, mid=ctr, n=1, addrs=[FLAGS + 128 * ctr])
            _op(storer, I_STG, addrs=[mark_addr(sm, k, MARK_DONE)])
            _op(storer, I_ARRIVE, kind=K_ARRIVE, mid=islot[k % inst_stages], n=1)
            c += iters
            o += iters
        for w, ops in [(0, loader), (1, storer)] + [(2 + i, b) for i, b in enumerate(cons)]:
            _op(ops, I_EXIT, active=lanes)
            n = _count_ops(ops)
            total_ops += n
            warps.append((sm, w, n, bytes(ops)))
    bars = {str(i): 1 for i in full + ofree + islot}
    bars.update({str(i): consumers for i in free + oarr})
    header = {"kernel": f"hazy_megakernel_{s.shape.name}", "grid": [s.sm_count, 1, 1],
              "block": [32 * (2 + consumers), 1, 1], "cluster": [1, 1, 1], "regs": 64,
              "smem": stages * iter_bytes + 16384 + 8192, "primary": 0, "barriers": bars, "strides": {},
              "sass": listing(f"hazy_megakernel_{s.shape.name}"),
              "flops": float(sum(t.weight_bytes for t in s.tasks)), "bytes": float(weight_off),
              "probes": [[MARKS, MARKS + (s.sm_count << 12) * 4 * 32]]}
    hb = json.dumps(header).encode()
    opener = open
    if out.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(out, "wb") as f:
        f.write(b"RSIMPTH1" + struct.pack("<I", len(hb)) + hb + struct.pack("<I", len(warps)))
        for sm, w, n, ops in warps:
            f.write(struct.pack("<III", sm, w, n))
            f.write(ops)
        f.write(struct.pack("<I", 0))
    return {"tasks": len(s.tasks), "per_sm": max(len(q) for q in s.queues), "ops": total_ops,
            "weight_bytes": weight_off, "counters": s.counters}


def _count_ops(b: bytes) -> int:
    n, p = 0, 0
    while p < len(b):
        na = struct.unpack_from("<H", b, p + _OP_SIZE - 2)[0]
        p += _OP_SIZE + 8 * na
        n += 1
    return n


def tasks_from_probes(s: Schedule, probes: List[int], clock_mhz: float) -> List[float]:
    """Simulated task schedule, flat [sm, op, layer, t_start, t_ready, t_end] in ns, from the
    probe hits of a replay (`details[i].probes`: flat [sm, cta, warp, address, cycle])."""
    marks: Dict[Tuple[int, int], List[Optional[int]]] = {}
    for i in range(0, len(probes), 5):
        cta, addr, cyc = probes[i + 1], probes[i + 3], probes[i + 4]
        sm, slot, kind = decode_mark(addr)
        marks.setdefault((cta, slot), [None, None, None])[kind] = cyc
    out: List[float] = []
    ns = 1e3 / clock_mhz
    for (cta, slot), (a, r, d) in sorted(marks.items()):
        if a is None or d is None:
            continue
        t = s.tasks[s.queues[cta][slot]]
        out += [cta, t.op, t.layer, round(a * ns, 2), round((r if r is not None else a) * ns, 2), round(d * ns, 2)]
    return out


def megakernel_entry(s: Schedule, tasks: List[float], source: str, **kw) -> dict:
    """A `megakernels` entry of an rsviz run (see docs/VIZ.md)."""
    return dict({"source": source, "sm_count": s.sm_count, "ops": OPS, "cats": CATS, "tasks": tasks,
                 "rank": 0, "kernel": -1, "label": f"{s.shape.name} decode token"}, **kw)
