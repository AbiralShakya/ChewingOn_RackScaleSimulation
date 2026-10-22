"""Capture any CUDA program on one GPU and turn it into a replayable capture.

    rackscale capture -o cap/ --region forward:8:4 -- python serve.py --model ...
    rackscale replay cap/ --system gb300_nvl72 --plan plan.yaml

The program runs unmodified under the NVBit tool ``capture/nvbit/rsim_capture``
twice:

1. **graph pass**: every kernel launch (function, grid, block, cluster,
   shared memory, registers, stream), event record and stream wait, copy and
   memset in the capture region, in host order, with the host time of each
   call and the GPU start/end of each kernel and copy. Nothing is
   instrumented, so the GPU timeline is native apart from one CUDA event on
   each side of every launch. The Python site package also marks engine steps
   and module calls (name, class, output size), which the parallelism plan
   uses to place collectives.
2. **paths pass**: the per-warp SASS path (the ``warp_path`` format, with lane
   addresses, mbarrier, TMA and UMMA operands) of the first in-region launch of
   every distinct kernel. A kernel is distinct by function and launch
   configuration (``--key func`` merges configurations), so a decode step of an
   80-layer model records each GEMM, attention and norm kernel once.

:func:`build` joins the two passes into ``capture.json`` and converts every
recording into ``kernels/<id>.rpath.gz``. Kernels that were not recorded (or
whose recording was too large) keep their measured duration.

The two passes must launch the same kernels in the region, which holds for
eager PyTorch engines with a fixed seed. Kernels launched inside CUDA graphs
are not expanded: capture with graphs disabled (vLLM ``enforce_eager=True``,
SGLang ``--disable-cuda-graph``) and replay with a graph-mode launch gap.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_capture_site")


def tool_path() -> str:
    """The rsim_capture NVBit tool: $RSIM_CAPTURE_TOOL, else the build-rsim output of $NVBIT."""
    p = os.environ.get("RSIM_CAPTURE_TOOL")
    if p:
        return p
    nv = os.environ.get("NVBIT")
    if nv:
        return os.path.join(nv, "tools", "build-rsim", "rsim_capture.so")
    raise RuntimeError("set RSIM_CAPTURE_TOOL to rsim_capture.so (or NVBIT; build it with capture/nvbit/build.sh)")


def run(cmd: Sequence[str], out: str, *, region: str = "marked",
        passes: Sequence[str] = ("graph", "paths", "timing"),
        modules: bool = True, key: str = "config", max_paths: int = 1024, max_mb: int = 2048,
        kernel_filter: str = "", gpu_lock: Optional[str] = None, tool: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None, max_path_passes: int = 6) -> str:
    """Run `cmd` once per pass under the capture tool, then :func:`build` the capture in `out`.

    A paths pass records each kernel function at most once (NVBit cannot
    re-enable instrumentation it has turned off), so paths passes repeat until
    every configuration in the region has a recording.
    """
    tool = tool or tool_path()
    os.makedirs(out, exist_ok=True)
    for p in passes:
        if p == "paths":
            for old in glob.glob(os.path.join(out, "paths*")):
                shutil.rmtree(old, ignore_errors=True) if os.path.isdir(old) else os.remove(old)
            for i in range(max_path_passes):
                d = os.path.join(out, "paths" if i == 0 else f"paths{i}")
                done = os.path.join(out, "paths_done.txt")
                keys = set(_recordings(out))
                with open(done, "w") as f:
                    f.write("".join(k + "\n" for k in sorted(keys)))
                _run_pass(cmd, d, "paths", tool, region, modules, key, max_paths, max_mb, kernel_filter, gpu_lock,
                          extra_env, done)
                new = set(_recordings(out)) - keys
                if not new:
                    shutil.rmtree(d, ignore_errors=True)
                    break
                print(f"rackscale capture: paths pass {i + 1} recorded {len(new)} kernels", file=sys.stderr)
        elif p == "timing":
            _timing_pass(cmd, os.path.join(out, "timing"), gpu_lock, extra_env, region)
        else:
            _run_pass(cmd, os.path.join(out, p), p, tool, region, modules, key, max_paths, max_mb, kernel_filter,
                      gpu_lock, extra_env, None)
    with open(os.path.join(out, "command.json"), "w") as f:
        json.dump({"cmd": list(cmd), "region": region, "key": key, "passes": list(passes)}, f, indent=1)
    return build(out, key=key)


def _run_pass(cmd, d, mode, tool, region, modules, key, max_paths, max_mb, kernel_filter, gpu_lock, extra_env,
              done):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    env = dict(os.environ, **(extra_env or {}))
    env.update({
        "CUDA_INJECTION64_PATH": tool,
        "RSIM_CAP_OUT": d,
        "RSIM_CAP_MODE": mode,
        "RSIM_CAP_REGION": region,
        "RSIM_CAP_MODULES": "1" if modules and mode == "graph" else "0",
        "RSIM_CAP_KEY": key,
        "RSIM_CAP_MAX_PATHS": str(max_paths),
        "RSIM_CAP_MAX_MB": str(max_mb),
        "RSIM_CAP_FILTER": kernel_filter,
        "RSIM_CAP_ARM_FILE": os.path.join(d, "armed"),
        "RSIM_CAP_DONE": done or "",
        "PYTHONPATH": SITE_DIR + os.pathsep + env.get("PYTHONPATH", ""),
        "NOBANNER": "1",
    })
    full = (["flock", gpu_lock] if gpu_lock else []) + list(cmd)
    for attempt in range(3):  # engines on a shared GPU fail start-up when its free memory moves
        print(f"rackscale capture: {mode} pass: {' '.join(cmd)}", file=sys.stderr)
        r = subprocess.run(full, env=env)
        if r.returncode == 0:
            return
        print(f"rackscale capture: {mode} pass exited with {r.returncode}"
              + (", retrying" if attempt < 2 else ""), file=sys.stderr)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    raise RuntimeError(f"{mode} pass exited with {r.returncode}")


def nsys_path() -> Optional[str]:
    """Nsight Systems: $RSIM_NSYS, else `nsys` on PATH (CUDA toolkits ship a stub that only prints an error)."""
    if os.environ.get("RSIM_NSYS"):
        return os.environ["RSIM_NSYS"]
    for c in ("/usr/local/bin/nsys", shutil.which("nsys")):
        if c and os.path.exists(c):
            return c
    return None


def _timing_pass(cmd, d, gpu_lock, extra_env, region="marked"):
    """Run the program natively under Nsight Systems: exact kernel times and the host launch timeline.

    Unless the region is the whole program, the site package starts and stops
    the profiler at the region marks (cudaProfilerStart/Stop), so only the
    region is recorded and the data of engine subprocesses is flushed before
    the engine tears them down.
    """
    nsys = nsys_path()
    if not nsys:
        print("rackscale capture: no nsys, skipping the timing pass (kernel times come from CUDA events)",
              file=sys.stderr)
        return
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    rep = os.path.join(d, "timing")
    env = dict(os.environ, **(extra_env or {}))
    rng = []
    if region != "all":
        env.update({"RSIM_CAP_PROFILER": "1", "RSIM_CAP_REGION": region, "RSIM_CAP_ARM_FILE": os.path.join(d, "armed"),
                    "PYTHONPATH": SITE_DIR + os.pathsep + env.get("PYTHONPATH", "")})
        rng = ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
    full = (["flock", gpu_lock] if gpu_lock else []) + [nsys, "profile", "-t", "cuda", "-s", "none",
                                                         "--cpuctxsw=none"] + rng + ["-o", rep, "-f", "true"] + list(cmd)
    print(f"rackscale capture: timing pass (nsys): {' '.join(cmd)}", file=sys.stderr)
    for attempt in range(3):
        if subprocess.run(full, env=env).returncode == 0:
            break
    else:
        raise RuntimeError("timing pass failed")
    subprocess.run([nsys, "export", "--type", "sqlite", "-o", rep + ".sqlite", "-f", "true", rep + ".nsys-rep"],
                   stdout=subprocess.DEVNULL, check=True)


def _native(sqlite_path: str) -> Dict[int, List[dict]]:
    """globalPid -> GPU ops (kernels, copies, memsets) in host launch order, from an nsys sqlite export."""
    import sqlite3

    c = sqlite3.connect(sqlite_path)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    strings = dict(c.execute("select id, value from StringIds"))
    host = {}
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t in tables:
            for corr, start in c.execute(f"select correlationId, start from {t}"):
                host.setdefault(corr, start)
    procs: Dict[int, List[dict]] = defaultdict(list)
    for row in c.execute("select correlationId, globalPid, start, end, mangledName, demangledName, gridX, gridY, "
                         "gridZ, blockX, blockY, blockZ from CUPTI_ACTIVITY_KIND_KERNEL"):
        corr, pid, a, b, m, dm, gx, gy, gz, bx, by, bz = row
        procs[pid].append({"corr": corr, "t": "K", "m": strings.get(m) or strings.get(dm), "g": [gx, gy, gz],
                           "b": [bx, by, bz], "a": a, "e": b, "h": host.get(corr)})
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if t in tables:
            for corr, pid, a, b in c.execute(f"select correlationId, globalPid, start, end from {t}"):
                procs[pid].append({"corr": corr, "t": "C", "a": a, "e": b, "h": host.get(corr)})
    for v in procs.values():
        v.sort(key=lambda r: r["corr"])
    return procs


def _attach_native(regions: List[List[dict]], funcs: Dict[int, dict], native: Dict[int, List[dict]]) -> bool:
    """Give the ops of the first region the native GPU times and host issue times of the same launches.

    The native run executes the same program without the tool, so its kernel
    sequence contains the region's. The region is matched as the last
    contiguous occurrence of its (function, grid, block) sequence; copies are
    matched in order between matched kernels.
    """
    if not regions or not native:
        return False
    ops = regions[0]
    rk = [o for o in ops if o["op"] == "kernel"]
    want = [(funcs[o["fid"]]["m"], list(o["grid"]), list(o["block"])) for o in rk]
    if not want:
        return False
    for pid, nat in native.items():
        nk = [i for i, r in enumerate(nat) if r["t"] == "K"]
        keys = [(nat[i]["m"], nat[i]["g"], nat[i]["b"]) for i in nk]
        n = len(want)
        hit = None
        for j in range(len(keys) - n, -1, -1):
            if keys[j] == want[0] and keys[j:j + n] == want:
                hit = j
                break
        if hit is None:
            continue
        idx = nk[hit:hit + n]
        t0 = nat[idx[0]]["a"]
        h0 = min(nat[i]["h"] for i in idx if nat[i]["h"] is not None) if any(nat[i]["h"] for i in idx) else None
        ki = 0
        prev = idx[0]
        pending_copies: List[dict] = []

        def flush(lo, hi):
            cs = [r for r in nat[lo + 1:hi] if r["t"] == "C"]
            for o, r in zip(pending_copies, cs):
                o["native_ns"] = [r["a"] - t0, r["e"] - t0]
                if h0 is not None and r["h"] is not None:
                    o["native_host_ns"] = r["h"] - h0
            pending_copies.clear()

        for o in ops:
            if o["op"] == "kernel":
                i = idx[ki]
                flush(prev, i)
                r = nat[i]
                o["native_ns"] = [r["a"] - t0, r["e"] - t0]
                if h0 is not None and r["h"] is not None:
                    o["native_host_ns"] = r["h"] - h0
                prev = i
                ki += 1
            elif o["op"] == "copy" and ki > 0:
                pending_copies.append(o)
        flush(prev, len(nat))
        return True
    return False


# --------------------------------------------------------------------- parse

def _events(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # a line cut off by a crash
                    continue


def _key(fn: dict, L: dict, key: str) -> str:
    if key == "func":
        return fn["m"]
    g, b, c = L["g"], L["b"], L["c"]
    return "%s|%d,%d,%d|%d,%d,%d|%d,%d,%d|%d" % (fn["m"], *g, *b, *c, L["sm"])


def _process(path: str, key: str) -> dict:
    """One process's graph-pass events -> regions of ops, plus its functions and device."""
    dev, funcs, times = {}, {}, {}
    regions: List[List[dict]] = []
    cur: Optional[List[dict]] = None
    streams: Dict[str, int] = {}
    records: Dict[str, int] = defaultdict(int)  # event handle -> records so far
    live: set = set()  # event record names seen in the current region
    graphs = 0

    def sid(h):
        return streams.setdefault(h, len(streams))

    for e in _events(path):
        t = e["e"]
        if t == "D":
            dev = e
        elif t == "F":
            funcs[e["id"]] = e
        elif t == "T":
            times[e["i"]] = (e["a"], e["b"])
        elif t == "M":
            k = e["k"]
            if k == "region_begin":
                cur = []
                regions.append(cur)
                live = set()
            if cur is None:
                continue
            if k == "region_end":
                cur.append({"op": "end", "host_ns": e["h"]})
                cur = None
            elif k == "push":
                name, _, cls = e["p"].partition("\t")
                cur.append({"op": "push", "name": name, "cls": cls})
            elif k == "pop":
                parts = e["p"].split("\t")
                o = {"op": "pop"}
                if len(parts) == 3:
                    o.update(shape=[int(x) for x in parts[0].split("x") if x], dtype=parts[1], bytes=int(parts[2]))
                cur.append(o)
            elif k == "step":
                cur.append({"op": "step", "host_ns": e["h"]})
            elif k != "region_begin":
                cur.append({"op": "mark", "kind": k, "payload": e["p"]})
        else:
            if cur is None:
                if not regions and dev.get("region") == "all" and t in ("L", "C"):
                    # RSIM_CAP_REGION=all: no begin mark, the whole program is the region
                    cur = []
                    regions.append(cur)
                else:
                    continue
            if t == "L":
                fn = funcs[e["f"]]
                cur.append({"op": "kernel", "seq": e["i"], "stream": sid(e["s"]), "key": _key(fn, e, key),
                            "fid": e["f"], "grid": e["g"], "block": e["b"], "cluster": e["c"], "smem": e["sm"],
                            "host_ns": e["h"]})
            elif t == "C":
                cur.append({"op": "copy", "seq": e["i"], "stream": sid(e["s"]), "kind": e["k"], "bytes": e["n"],
                            "host_ns": e["h"]})
            elif t == "R":
                records[e["ev"]] += 1
                name = "%s#%d" % (e["ev"], records[e["ev"]])
                live.add(name)
                cur.append({"op": "record", "stream": sid(e["s"]), "event": name})
            elif t == "W":
                name = "%s#%d" % (e["ev"], records[e["ev"]])
                if name in live:  # a wait on a record from before the region is already satisfied
                    cur.append({"op": "wait", "stream": sid(e["s"]), "event": name})
            elif t == "S":
                cur.append({"op": "sync", "kind": e["k"], "host_ns": e["h"]})
            elif t == "G":
                graphs += 1
    for ops in regions:
        for o in ops:
            if o["op"] in ("kernel", "copy") and o["seq"] in times:
                o["gpu_ns"] = list(times[o["seq"]])
    return {"device": dev, "functions": funcs, "regions": regions, "streams": len(streams), "graph_launches": graphs}


def _recordings(out_dir: str) -> Dict[str, dict]:
    """key -> {dir, bytes, truncated} from every process of every paths pass."""
    out = {}
    for ev in sorted(glob.glob(os.path.join(out_dir, "paths*", "p*", "events.jsonl"))):
        pdir = os.path.dirname(ev)
        for e in _events(ev):
            if e["e"] == "P" and e["key"] not in out:
                out[e["key"]] = {"dir": os.path.join(pdir, e["dir"]), "bytes": e["bytes"], "truncated": e["truncated"]}
    return out


# --------------------------------------------------------------------- build

def build(out: str, *, key: str = "config", convert: bool = True) -> str:
    """Join the passes in `out` into ``capture.json`` and convert recordings to ``kernels/``."""
    procs = []
    for ev in sorted(glob.glob(os.path.join(out, "graph", "p*", "events.jsonl"))):
        p = _process(ev, key)
        if any(r for r in p["regions"]):
            p["pid"] = int(os.path.basename(os.path.dirname(ev))[1:])
            procs.append(p)
    native_db = os.path.join(out, "timing", "timing.sqlite")
    try:
        native = _native(native_db) if os.path.exists(native_db) else {}
    except Exception as e:  # an empty or partial export: fall back to the tool's events
        print(f"rackscale capture: warning: timing pass unreadable ({e})", file=sys.stderr)
        native = {}
    for p in procs:
        p["native"] = _attach_native(p["regions"], p["functions"], native)
        if native and not p["native"]:
            print(f"rackscale capture: warning: process {p['pid']}: region not found in the timing pass",
                  file=sys.stderr)
    if not procs:
        raise RuntimeError(f"no kernels captured in {out}/graph: was the region ever entered? "
                           "(use --region all, forward:S:N, or rsim_capture.region() in the workload)")
    recs = _recordings(out)
    kernels: Dict[str, dict] = {}
    kid_of: Dict[str, str] = {}
    durations: Dict[str, List[float]] = defaultdict(list)  # native (nsys) kernel durations
    tool_durations: Dict[str, List[float]] = defaultdict(list)  # CUDA events around launches under the tool
    for p in procs:
        for ops in p["regions"]:
            for o in ops:
                if o["op"] != "kernel":
                    continue
                k = o.pop("key")
                if k not in kid_of:
                    kid = "k%04d" % len(kid_of)
                    kid_of[k] = kid
                    fn = p["functions"][o["fid"]]
                    kernels[kid] = {"key": k, "name": fn["n"], "mangled": fn["m"], "grid": o["grid"],
                                    "block": o["block"], "cluster": o["cluster"],
                                    "smem": o["smem"] + fn.get("ss", 0), "regs": fn.get("r", 0), "count": 0}
                kid = kid_of[k]
                o["kernel"] = kid
                kernels[kid]["count"] += 1
                if "native_ns" in o:
                    durations[kid].append(o["native_ns"][1] - o["native_ns"][0])
                if "gpu_ns" in o:
                    tool_durations[kid].append(o["gpu_ns"][1] - o["gpu_ns"][0])
                for f in ("fid", "grid", "block", "cluster", "smem"):
                    o.pop(f, None)
    kdir = os.path.join(out, "kernels")
    os.makedirs(kdir, exist_ok=True)
    # Conversions from an earlier build, reused only for the same recording.
    prev = {}
    try:
        with open(os.path.join(out, "capture.json")) as f:
            for k in json.load(f)["kernels"].values():
                r = k.get("recording") or {}
                if k.get("path") and r.get("dir"):
                    prev[k["key"]] = (r["dir"], r["bytes"], k["path"], r.get("summary"))
    except (OSError, ValueError, KeyError):
        pass
    for kid, k in kernels.items():
        for name, src in (("measured_ns", durations.get(kid) or tool_durations.get(kid)),
                          ("tool_ns", tool_durations.get(kid))):
            if src:
                k[name] = {"median": statistics.median(src), "mean": statistics.fmean(src), "min": min(src),
                           "n": len(src)}
        k["measured_by"] = "nsys" if durations.get(kid) else "events"
        r = recs.get(k["key"])
        if r is None:
            continue
        rdir = os.path.relpath(r["dir"], out)
        k["recording"] = {"dir": rdir, "bytes": r["bytes"], "truncated": r["truncated"]}
        if r["truncated"] or not convert:
            continue
        dst = os.path.join(kdir, kid + ".rpath")
        rel = {v: os.path.relpath(_variant(dst, v) + ".gz", out) for v, _p, _i in VARIANTS}
        old = prev.get(k["key"])
        if old and old[0] == rdir and old[1] == r["bytes"] and all(os.path.exists(os.path.join(out, x))
                                                                    for x in rel.values()):
            k["recording"]["summary"] = old[3]
        else:
            try:
                summary = _convert(r["dir"], dst, k["regs"])
            except Exception as e:  # keep the measured duration for this kernel
                k["recording"]["error"] = f"{type(e).__name__}: {e}"
                continue
            k["recording"]["summary"] = {x: summary.get(x) for x in ("ctas", "stats", "mma_shapes")}
        k["path"] = rel["phases"]
        k["path_variants"] = rel
    dev = procs[0]["device"]
    doc = {
        "format": "rackscale-capture/v1",
        "device": {x: dev.get(x) for x in ("name", "cc", "sms", "clock_khz")},
        "key": key,
        "processes": [{"pid": p["pid"], "streams": p["streams"], "graph_launches": p["graph_launches"],
                       "native": p["native"], "regions": p["regions"]} for p in procs],
        "kernels": kernels,
    }
    path = os.path.join(out, "capture.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    n_rec = sum(1 for k in kernels.values() if k.get("path"))
    n_launch = sum(k["count"] for k in kernels.values())
    print(f"rackscale capture: {n_launch} launches of {len(kernels)} distinct kernels, {n_rec} with warp paths "
          f"-> {path}", file=sys.stderr)
    if any(p["graph_launches"] for p in procs):
        print("rackscale capture: warning: CUDA graph launches in the region were not expanded; "
              "capture with CUDA graphs disabled", file=sys.stderr)
    return path


# Wait semantics of the converted path, tried in this order by rackscale.replay.check_kernels:
# phase counts (validated on FlashAttention), phase counts with barriers that start completed
# (cuBLAS), and the hardware's parity rule.
VARIANTS = (("phases", False, False), ("init", False, True), ("parity", True, True))


def _variant(dst: str, v: str) -> str:
    return dst if v == "phases" else dst[: -len(".rpath")] + f".{v}.rpath"


def _convert(trace_dir: str, dst: str, regs: int) -> dict:
    """Convert a recording once per wait semantics in `VARIANTS` (``kNNNN[.v].rpath.gz``).

    `rackscale.replay.check_kernels` keeps the first variant that replays.
    """
    from .nvbit.trace_to_rsim import convert

    summary = {}
    for v, parity, init in VARIANTS:
        d = _variant(dst, v)
        sm = convert(trace_dir, d, regs=regs or None, parity_waits=parity, detect_completed_init=init)
        summary = summary or sm
        with open(d, "rb") as f, gzip.open(d + ".gz", "wb", compresslevel=6) as g:
            shutil.copyfileobj(f, g)
        os.remove(d)
    return summary


def load(path: str) -> dict:
    """Read a capture (the directory or its ``capture.json``)."""
    p = os.path.join(path, "capture.json") if os.path.isdir(path) else path
    with open(p) as f:
        doc = json.load(f)
    doc["_dir"] = os.path.dirname(os.path.abspath(p))
    chk = os.path.join(doc["_dir"], "kernel_check.json")
    if os.path.exists(chk):  # written by rackscale.replay.check_kernels
        with open(chk) as f:
            for r in json.load(f)["kernels"]:
                k = doc["kernels"].get(r["kernel"])
                if k is not None:
                    k["replay_ok"] = "error" not in r
                    if r.get("variant") in (k.get("path_variants") or {}):
                        k["path"] = k["path_variants"][r["variant"]]
    return doc
