#!/usr/bin/env python3
"""Ground-truth kernels for validating the simulator against Nsight Compute.

For each reference kernel this records the launch configuration, the kernel's
SASS (exactly what the simulator's SASS front end should consume) and Nsight
Compute's view of it: elapsed SM cycles, instructions, achieved occupancy,
DRAM/L2 traffic and the warp-stall breakdown. It also records the kernel's own
%clock64 measurement where the bench reports one, so the two clocks can be
compared.

The simulator's check is then: simulate <sass> with <launch>, compare predicted
cycles with ncu's sm__cycles_elapsed.max, and stall shares with the ncu
breakdown.

Usage: ncu_reference.py results/<gpu>-sm<cc>
"""
import csv
import io
import json
import os
import re
import subprocess
import sys

# (label, binary, args, kernel regex, launch-skip, launch-count)
# Skips are chosen to land on a timed (post-warm-up) launch; for tc05_kernel,
# launch 12 is the first all-SM 4096-MMA throughput run.
REFERENCE = [
    ("ffma_throughput", "build/bench_alu", [], r"thr_kernel<OpFFMA>", 0, 1),
    ("ex2_throughput", "build/bench_alu", [], r"thr_kernel<OpEX2>", 0, 1),
    ("dfma_latency_chain", "build/bench_alu", [], r"lat_kernel<OpDFMA>", 0, 1),
    ("smem_conflict_free", "build/bench_mem", [], r"smem_bw_kernel", 0, 1),
    ("smem_32way_conflict", "build/bench_mem", [], r"smem_bw_kernel", 5, 1),
    ("hbm_stream_read", "build/bench_mem", [], r"bw_read_kernel", 1, 1),
    ("hbm_copy", "build/bench_mem", [], r"bw_copy_kernel", 1, 1),
    ("l2_pointer_chase", "build/bench_mem", [], r"chase_kernel<1>", 20, 1),
    ("hmma_bf16_throughput", "build/bench_tc", ["mma_sync"], r"mma_sync_kernel<MmaBF16, 4>", 0, 1),
    ("umma_bf16_m128n256k16", "build/bench_tc", ["tcgen05_f16_bf16_m128n256k16"], r"tc05_kernel", 12, 1),
    ("umma_nvfp4_m128n256k64", "build/bench_tc", ["tcgen05_nvfp4_m128n256k64"], r"tc05_kernel", 12, 1),
    ("tma_bulk_stream", "build/bench_tma", [], r"tma_bw_kernel", 4, 1),
    # sm_80+ async-copy and warp-MMA paths (Ampere's main data movers).
    ("cp_async_cg16_dram_stream", "build/bench_cpasync", [], r"stream_kernel<1, 4>", 68, 1),  # DRAM pass, 256 thr x 4 CTA/SM
    ("hmma_f16_f32_m16n8k16_w32_ilp4", "build/bench_mma_sync", [], r"mma_thr_kernel<F16_F32_m16n8k16, 4>", 17, 1),
]

# Reference kernels that need hardware newer than sm_80.
MIN_CC = {"umma_bf16_m128n256k16": 100, "umma_nvfp4_m128n256k64": 100, "tma_bulk_stream": 90}


def gpu_index():
    """nvidia-smi index of the visible GPU; a MIG UUID maps to its parent (0 in the job)."""
    d = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return "0" if d.startswith("MIG-") else d


def name_regex(regex):
    """Escape a kernel signature, letting integer template args match either
    `<1>` or `<(int)1>`: cu++filt from CUDA 12 prints the cast, CUDA 13 does not."""
    return re.sub(r"(<|,\\ )(\d)", r"\1(?:\\(int\\))?\2", re.escape(regex))

METRICS = [
    "sm__cycles_elapsed.max", "sm__cycles_elapsed.avg", "smsp__cycles_active.avg", "gpu__time_duration.sum",
    "smsp__inst_executed.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum",
    "launch__grid_size", "launch__block_size", "launch__registers_per_thread",
    "launch__shared_mem_per_block_dynamic", "launch__shared_mem_per_block_static",
]
STALLS = ["barrier", "branch_resolving", "dispatch_stall", "drain", "lg_throttle", "long_scoreboard", "math_pipe_throttle",
          "membar", "mio_throttle", "misc", "no_instruction", "not_selected", "selected", "short_scoreboard", "sleeping",
          "tex_throttle", "wait"]
# Requested by name: no ncu section files are needed (Della's install has none).
METRICS += [f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio" for s in STALLS]


def sass_of(binary, mangled):
    out = subprocess.run(["cuobjdump", "-sass", "-fun", mangled, binary], capture_output=True, text=True).stdout
    return out


def mangled_names(binary):
    out = subprocess.run(["cuobjdump", "-sass", binary], capture_output=True, text=True).stdout
    names = re.findall(r"Function : (\S+)", out)
    dm = subprocess.run(["cu++filt"], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(dm, names))


def run_ncu(binary, args, regex, skip, count):
    # With --kernel-name-base demangled the regex sees the full signature,
    # template arguments included; the launch is then picked by index.
    cmd = ["ncu", "--clock-control", "none", "--csv", "--page", "raw", "--kernel-name-base", "demangled", "--kernel-name", f"regex:{name_regex(regex)}",
           "--launch-skip", str(skip), "--launch-count", str(count), "--metrics", ",".join(METRICS), binary] + args
    if os.environ.get("GPU_LOCK"):  # shared GPU: hold the lock per profiled run
        cmd = ["flock", os.environ["GPU_LOCK"]] + cmd
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout
    start = text.find('"ID"')
    if start < 0:
        return None, p.stderr[-2000:] + text[-2000:]
    rows = list(csv.reader(io.StringIO(text[start:])))
    header, units, data = rows[0], rows[1], rows[2:]
    if not data:
        return None, "no kernel matched"
    rec = dict(zip(header, data[0]))
    return rec, None


def num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return s


def main(outdir):
    sass_dir = os.path.join(outdir, "sass")
    os.makedirs(sass_dir, exist_ok=True)
    results = []
    names_cache = {}
    cc = int(subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i",
                             gpu_index()],
                            capture_output=True, text=True).stdout.split()[0].replace(".", ""))
    for label, binary, args, regex, skip, count in REFERENCE:
        if cc < MIN_CC.get(label, 0) or not os.access(binary, os.X_OK):
            continue
        rec, err = run_ncu(binary, args, regex, skip, count)
        entry = {"label": label, "binary": binary, "args": args, "kernel": regex, "launch_index": skip}
        if rec is None:
            entry["error"] = err
            print(f"{label}: ncu failed", file=sys.stderr)
            results.append(entry)
            continue
        entry["kernel_demangled"] = rec.get("Kernel Name")
        metrics = {k: num(v) for k, v in rec.items() if "__" in k}
        entry["ncu"] = {k: v for k, v in metrics.items() if k in METRICS}
        entry["stalls_per_issue"] = {
            k.split("issue_stalled_")[1].split("_per_issue")[0]: v
            for k, v in metrics.items()
            if "average_warps_issue_stalled_" in k and k.endswith("per_issue_active.ratio") and isinstance(v, float)
        }
        names = names_cache.setdefault(binary, mangled_names(binary))
        full = re.compile(r"(?:void )?" + name_regex(regex) + r"\(")
        mangled = next((m for d, m in names.items() if full.match(d)), None)
        if mangled is None:
            mangled = next((m for d, m in names.items() if regex.split("<")[0] in d), None)
        if mangled:
            path = os.path.join(sass_dir, f"{label}.sass")
            open(path, "w").write(sass_of(binary, mangled))
            entry["sass_file"] = os.path.relpath(path, outdir)
            entry["kernel_mangled"] = mangled
        results.append(entry)
        c = entry["ncu"].get("sm__cycles_elapsed.max")
        print(f"{label:28s} sm cycles {c}", file=sys.stderr)
    json.dump({"schema": "rackscale-ncu-reference/v1", "kernels": results}, open(os.path.join(outdir, "ncu_reference.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
