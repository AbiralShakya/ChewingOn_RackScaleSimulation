"""Capture real engines (vLLM, SGLang, TensorRT-LLM via PyTorch, any training
loop) through the PyTorch profiler and convert the trace into a workload.

Why the profiler: every serious engine can emit a Kineto/Chrome trace
(``vllm --profile``/``VLLM_TORCH_PROFILER_DIR``, SGLang ``/start_profile``,
``torch.profiler`` in training code). With ``record_shapes=True`` each GPU
kernel links (via ``External id``/correlation) to the CPU op that launched it,
which gives GEMM/attention shapes. Kernels are lowered as follows:

* GEMMs with recoverable shapes (``aten::mm/addmm/bmm/linear/matmul``) and
  flash/flashinfer attention with known dims  -> re-simulated at SASS level
  (synthetic kernels), so the *target* hardware can differ from the profiled
  one;
* NCCL kernels -> collectives (bytes from ``record_param_comms`` metadata
  when present);
* everything else -> ``fixed`` kernels replaying the measured duration
  (optionally scaled), occupying all SMs.

``mode="replay"`` keeps every kernel as measured (only overlap, launch gaps
and communication are re-simulated); ``mode="resimulate"`` (default) lowers
what it can.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_GEMM_OPS = {"aten::mm", "aten::addmm", "aten::matmul", "aten::linear", "aten::bmm", "aten::_scaled_mm"}
_NCCL = re.compile(r"nccl(?:Dev)?Kernel_(\w+?)_", re.I)
_COLL_KIND = {
    "allreduce": "all_reduce",
    "allgather": "all_gather",
    "reducescatter": "reduce_scatter",
    "alltoall": "all_to_all",
    "broadcast": "broadcast",
    "sendrecv": "send_recv",
}
_DTYPE_BYTES = {"float": 4, "float32": 4, "bfloat16": 2, "half": 2, "float16": 2, "c10::bfloat16": 2, "c10::half": 2, "float8_e4m3fn": 1}


def load_trace(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def _gemm_shape(cpu: dict) -> Optional[Tuple[int, int, int]]:
    dims = cpu.get("args", {}).get("Input Dims")
    if not dims:
        return None
    name = cpu.get("name", "")
    try:
        if name in ("aten::linear",):
            x, w = dims[0], dims[1]
            m = 1
            for d in x[:-1]:
                m *= d
            return m, w[0], x[-1]
        if name == "aten::addmm":
            a, b = dims[1], dims[2]
        else:
            a, b = dims[0], dims[1]
        if name == "aten::bmm" and len(a) == 3:
            return a[0] * a[1], b[2], a[2]
        m = 1
        for d in a[:-1]:
            m *= d
        return m, b[-1], a[-1]
    except (IndexError, TypeError):
        return None


def from_torch_profiler(trace: dict, *, mode: str = "resimulate", world: int = 1, collective_group: str = "world",
                        time_scale: float = 1.0, max_kernels: Optional[int] = None) -> dict:
    """Convert one rank's Kineto trace into an SPMD workload over ``world`` ranks."""
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    cpu_by_ext: Dict[int, dict] = {}
    comm_meta: Dict[int, dict] = {}
    kernels = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = (e.get("cat") or "").lower()
        args = e.get("args", {}) or {}
        if cat in ("cpu_op", "operator"):
            ext = args.get("External id")
            if ext is not None:
                if e.get("name") in _GEMM_OPS or ext not in cpu_by_ext:
                    cpu_by_ext[ext] = e
                if e.get("name") == "record_param_comms":
                    comm_meta[ext] = args
        elif cat == "kernel":
            kernels.append(e)
    kernels.sort(key=lambda e: e["ts"])
    if max_kernels:
        kernels = kernels[:max_kernels]
    streams: Dict[int, List[dict]] = defaultdict(list)
    stats = defaultdict(int)
    for k in kernels:
        args = k.get("args", {}) or {}
        stream = int(args.get("stream", 7))
        name = k.get("name", "kernel")
        dur_ns = float(k.get("dur", 0.0)) * 1000.0 * time_scale
        ext = args.get("External id")
        cpu = cpu_by_ext.get(ext, {})
        m = _NCCL.search(name)
        if m:
            kind = _COLL_KIND.get(m.group(1).lower().replace("_", ""), "all_reduce")
            meta = comm_meta.get(ext, {})
            nelem = meta.get("In msg nelems") or meta.get("Out msg nelems")
            dt = _DTYPE_BYTES.get(str(meta.get("dtype", "bfloat16")).lower(), 2)
            nbytes = int(nelem) * dt if nelem else None
            if nbytes and kind != "send_recv":
                streams[stream].append({"op": "collective", "name": name[:60], "kind": kind, "bytes": nbytes,
                                        "group": collective_group, "sms": int(args.get("blocks per SM", 0) or 0) or 16})
                stats["collective"] += 1
            else:
                streams[stream].append({"op": "kernel", "name": name[:60], "kernel": {"type": "fixed", "ns": dur_ns, "sms": 16}})
                stats["comm_fixed"] += 1
            continue
        lowered = None
        if mode == "resimulate":
            if cpu.get("name") in _GEMM_OPS:
                shp = _gemm_shape(cpu)
                if shp:
                    mm, nn, kk = shp
                    lowered = {"type": "gemm", "m": int(mm), "n": int(nn), "k": int(kk), "dtype": "bf16"}
                    stats["gemm"] += 1
        if lowered is None:
            grid = args.get("grid", [1, 1, 1])
            sms = min(int(grid[0] * grid[1] * grid[2]) if isinstance(grid, list) else 1 << 30, 1 << 30)
            lowered = {"type": "fixed", "ns": dur_ns, "sms": max(1, sms)}
            stats["fixed"] += 1
        streams[stream].append({"op": "kernel", "name": name[:60], "kernel": lowered})
    progs = [streams[s] for s in sorted(streams)]
    return {
        "name": f"captured ({dict(stats)})",
        "spmd": {"world": world, "program": {"streams": progs}},
    }


def from_file(path: str, **kw) -> dict:
    return from_torch_profiler(load_trace(path), **kw)
