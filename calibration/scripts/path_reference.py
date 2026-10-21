#!/usr/bin/env python3
"""Record ncu reference kernels with the NVBit warp_path tool and convert them
to the simulator's per-warp path format (.rpath).

A `.rpath` is the exact dynamic SASS path of a few CTAs with every lane
address, mbarrier operand and tcgen05 descriptor, so `rsim validate` replays
what the kernel really did: the throughput loop of a UMMA kernel whose mode is
a runtime argument, the bank conflicts of a strided smem loop. CTAs that were
not recorded replay the primary CTA's path, so the recording is taken of the
full-size reference launch and the existing full-size ncu numbers still apply.

For each label: run the entry's binary with the tool injected, on the same
launch index ncu measured; convert with research/flash_attention/
trace_to_rsim.py; write paths/<label>.rpath.gz and add "path_file" to the entry.

  WARP_PATH_SO=<nvbit>/tools/build/warp_path/warp_path.so \
  TRACE_TO_RSIM=<repo>/research/flash_attention/trace_to_rsim.py \
  [GPU_LOCK=/path/lock] path_reference.py results/<gpu>-sm<cc> label [label ...]
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile


def locked(cmd):
    lock = os.environ.get("GPU_LOCK")
    return ["flock", lock] + cmd if lock else cmd


def main(outdir, labels):
    so, conv = os.environ["WARP_PATH_SO"], os.environ["TRACE_TO_RSIM"]
    ref_path = os.path.join(outdir, "ncu_reference.json")
    doc = json.load(open(ref_path))
    pdir = os.path.join(outdir, "paths")
    os.makedirs(pdir, exist_ok=True)
    for entry in doc["kernels"]:
        label = entry["label"]
        if label not in labels:
            continue
        if not entry.get("kernel_mangled") or "error" in entry:
            print(f"{label}: no mangled name / ncu entry, skipped", file=sys.stderr)
            continue
        work = tempfile.mkdtemp(prefix=f"wp-{label}-", dir=os.environ.get("TMPDIR"))
        # warp_path matches WP_KERNEL against NVBit's demangled name: use the
        # function name with its template arguments, without the signature.
        dm = entry.get("kernel_demangled") or entry["kernel"]
        key = dm.split("(")[0].replace("void ", "").strip()
        env = dict(os.environ, CUDA_INJECTION64_PATH=so, WP_KERNEL=key,
                   WP_LAUNCH=str(entry["launch_index"]), WP_CTAS="0,1", WP_OUT=work)
        p = subprocess.run(locked([entry["binary"]] + entry["args"]), env=env, capture_output=True, text=True)
        if p.returncode != 0 or not any(f.endswith(".wp") for f in os.listdir(work)):
            print(f"{label}: recording failed: {(p.stderr or p.stdout)[-600:]}", file=sys.stderr)
            shutil.rmtree(work, ignore_errors=True)
            continue
        dst = os.path.join(pdir, f"{label}.rpath")
        regs = entry.get("ncu", {}).get("launch__registers_per_thread")
        cmd = [sys.executable, conv, work, dst] + (["--regs", str(int(regs))] if isinstance(regs, (int, float)) else [])
        q = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(work, ignore_errors=True)
        if q.returncode != 0 or not os.path.exists(dst):
            print(f"{label}: conversion failed: {q.stderr[-600:]}", file=sys.stderr)
            continue
        # Recordings of long loops compress ~70x; rsim reads .rpath.gz directly.
        with open(dst, "rb") as f, gzip.open(dst + ".gz", "wb", compresslevel=9) as g:
            shutil.copyfileobj(f, g)
        os.remove(dst)
        dst += ".gz"
        entry["path_file"] = os.path.relpath(dst, outdir)
        entry["path_ctas_recorded"] = [0, 1]
        print(f"{label:28s} {os.path.getsize(dst) / 2**20:7.1f} MB  {q.stdout.strip().splitlines()[-1] if q.stdout.strip() else ''}",
              file=sys.stderr)
        json.dump(doc, open(ref_path, "w"), indent=1)  # after each kernel, so partial runs keep their work


if __name__ == "__main__":
    main(sys.argv[1], set(sys.argv[2:]))
