#!/usr/bin/env python3
"""Capture Accel-Sim / NVBit traces of the ncu reference kernels at small size.

`rsim validate` replays a `.traceg` exactly (real control flow and addresses)
instead of guessing loop trip counts from the static SASS. A full-size
reference launch is far too large to trace, so each kernel is re-run with
CAL_MAX_CTAS (grid cap), CAL_MAX_ITERS (loop cap) and, for streaming kernels,
CAL_MAX_BYTES (stream length); see cal_grid()/cal_iters()/cal_bytes() in
src/common.cuh. ncu is re-measured at
that same size, because the trace and the cycles it is checked against must
describe the same launch.

For every entry of <outdir>/ncu_reference.json that this GPU can run:
  * ncu at trace size replaces "ncu" and "stalls_per_issue"; the full-size
    measurement moves to "ncu_full_size",
  * the NVBit tracer records every launch matching the kernel, and the one at
    the entry's launch index becomes traces/<label>.traceg,
  * "trace_file", "trace_env" are added.

Needs the Accel-Sim tracer (util/tracer_nvbit) built for this GPU:
  ACCELSIM_TRACER=<...>/tracer_tool/tracer_tool.so
  ACCELSIM_POST=<...>/tracer_tool/traces-processing/post-traces-processing

Usage: trace_reference.py results/<gpu>-sm<cc> [label ...]
With labels, only those entries are traced. With GPU_LOCK set (a flock file),
the lock is held per entry rather than for the whole run, so other jobs on a
shared GPU can interleave.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import ncu_reference as nr  # noqa: E402

# Grid / byte caps per reference kernel. One CTA keeps the trace of a
# 1024-thread throughput kernel near 10 MB; the HBM streams get 2 CTAs and
# 1 MB, which still issues long runs of LDG.128 per warp.
SIZE = {
    "ffma_throughput": {"CAL_MAX_CTAS": "1"},
    "ex2_throughput": {"CAL_MAX_CTAS": "1"},
    "dfma_latency_chain": {},
    "smem_conflict_free": {"CAL_MAX_CTAS": "1", "CAL_MAX_ITERS": "32", "CAL_SECTIONS": "smem"},
    "smem_32way_conflict": {"CAL_MAX_CTAS": "1", "CAL_MAX_ITERS": "32", "CAL_SECTIONS": "smem"},
    "hbm_stream_read": {"CAL_MAX_CTAS": "2", "CAL_MAX_BYTES": str(1 << 20), "CAL_MAX_ITERS": "1", "CAL_SECTIONS": "bandwidth"},
    "hbm_copy": {"CAL_MAX_CTAS": "2", "CAL_MAX_BYTES": str(1 << 20), "CAL_MAX_ITERS": "1", "CAL_SECTIONS": "bandwidth"},
    "l2_pointer_chase": {"CAL_SECTIONS": "latency"},
    "hmma_bf16_throughput": {"CAL_MAX_CTAS": "1"},
    "cp_async_cg16_dram_stream": {"CAL_MAX_CTAS": "2", "CAL_MAX_BYTES": str(1 << 20)},
    "hmma_f16_f32_m16n8k16_w32_ilp4": {"CAL_MAX_CTAS": "1"},
    # B300 (sm_103): tcgen05 UMMA issue loops (one issuing thread per CTA) and
    # the bulk-copy stream.
    "umma_bf16_m128n256k16": {"CAL_MAX_CTAS": "2"},
    "umma_nvfp4_m128n256k64": {"CAL_MAX_CTAS": "2"},
    "tma_bulk_stream": {"CAL_MAX_CTAS": "2", "CAL_MAX_BYTES": str(1 << 20)},
}


def locked(cmd):
    lock = os.environ.get("GPU_LOCK")
    return ["flock", lock] + cmd if lock else cmd


def ncu_at(binary, args, regex, skip, env):
    old = dict(os.environ)
    os.environ.update(env)
    try:
        return nr.run_ncu(binary, args, regex, skip, 1)
    finally:
        os.environ.clear()
        os.environ.update(old)


def trace(binary, args, mangled, skip, env, workdir):
    tracer, post = os.environ["ACCELSIM_TRACER"], os.environ["ACCELSIM_POST"]
    folder = os.path.join(workdir, "run")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    e = dict(os.environ, **env)
    e.update({
        "CUDA_INJECTION64_PATH": tracer,
        "DYNAMIC_KERNEL_RANGE": "@" + re.escape(mangled),
        "USER_DEFINED_FOLDERS": "1",
        "TRACES_FOLDER": folder,
        "TRACE_FILE_COMPRESS": "0",
        "TOOL_COMPRESS": "0",
    })
    try:
        p = subprocess.run(locked([binary] + args), env=e, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "tracer run timed out after 900 s"
    if p.returncode != 0:
        return None, p.stderr[-2000:]
    tdir = os.path.join(folder, "traces")
    lists = [f for f in os.listdir(tdir) if f.startswith("kernelslist")]
    if not lists:
        return None, "tracer wrote no kernelslist"
    kernels = [l.strip() for l in open(os.path.join(tdir, lists[0])) if l.strip().startswith("kernel-")]
    if skip >= len(kernels):
        return None, f"only {len(kernels)} matching launches traced, wanted index {skip}"
    q = subprocess.run([post, os.path.join(tdir, lists[0]), "--text"], capture_output=True, text=True)
    if q.returncode != 0:
        return None, q.stderr[-2000:]
    stem = kernels[skip].split(".trace")[0]
    out = os.path.join(tdir, stem + ".traceg")
    if not os.path.exists(out):
        return None, f"post-processing did not produce {out}"
    return out, None


def main(outdir, only=()):
    ref_path = os.path.join(outdir, "ncu_reference.json")
    doc = json.load(open(ref_path))
    tdir = os.path.join(outdir, "traces")
    os.makedirs(tdir, exist_ok=True)
    only = set(sys.argv[2:])
    work = tempfile.mkdtemp(prefix="caltrace-", dir=os.environ.get("TMPDIR"))
    for entry in doc["kernels"]:
        label = entry["label"]
        if label not in SIZE or "error" in entry or not entry.get("kernel_mangled") or (only and label not in only):
            continue
        binary, args, regex, skip = entry["binary"], entry["args"], entry["kernel"], entry["launch_index"]
        env = SIZE[label]
        rec, err = ncu_at(binary, args, regex, skip, env)
        if rec is None:
            print(f"{label}: ncu at trace size failed: {err[-300:]}", file=sys.stderr)
            continue
        path, err = trace(binary, args, entry["kernel_mangled"], skip, env, work)
        if path is None:
            print(f"{label}: trace failed: {err[-300:]}", file=sys.stderr)
            continue
        dst = os.path.join(tdir, f"{label}.traceg")
        shutil.move(path, dst)
        metrics = {k: nr.num(v) for k, v in rec.items() if "__" in k}
        if "ncu_full_size" not in entry:
            entry["ncu_full_size"] = entry.get("ncu")
        entry["ncu"] = {k: v for k, v in metrics.items() if k in nr.METRICS}
        entry["stalls_per_issue"] = {
            k.split("issue_stalled_")[1].split("_per_issue")[0]: v
            for k, v in metrics.items()
            if "average_warps_issue_stalled_" in k and k.endswith("per_issue_active.ratio") and isinstance(v, float)
        }
        entry["trace_file"] = os.path.relpath(dst, outdir)
        entry["trace_env"] = env
        mb = os.path.getsize(dst) / 2**20
        print(f"{label:32s} traced {mb:6.1f} MB, ncu sm cycles {entry['ncu'].get('sm__cycles_elapsed.avg')}",
              file=sys.stderr)
        json.dump(doc, open(ref_path, "w"), indent=1)  # after every entry: jobs can time out
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], tuple(sys.argv[2:]))
