"""Record NVBit warp paths (exact per-warp SASS paths and lane addresses) of
the timeline kernels, for the simulator side of timeline_compare.py.

Each (kernel, block) runs in its own process with the warp_path tool
injected and WP_LAUNCH=0 (the first launch of that kernel: timeline.py's
warm-up launch, whose path is the same as the timed ones), then
research/flash_attention/trace_to_rsim.py converts it to .rpath.gz.

    WARP_PATH_SO=... TRACE_TO_RSIM=... python3 record_paths.py build/timeline.cubin outdir [kernel:block ...]
"""
import gzip
import os
import shutil
import subprocess
import sys
import tempfile

from timeline import DEFAULT


def main(cub, outdir, cfgs):
    so, conv = os.environ["WARP_PATH_SO"], os.environ["TRACE_TO_RSIM"]
    os.makedirs(outdir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for cfg in cfgs:
        name, block = cfg.split(":")
        work = tempfile.mkdtemp(prefix=f"wp-{name}-{block}-", dir=os.environ.get("TMPDIR"))
        env = dict(os.environ, CUDA_INJECTION64_PATH=so, WP_KERNEL=name, WP_LAUNCH="0",
                   WP_CTAS="0", WP_OUT=work)
        lock = ["flock", os.environ["GPU_LOCK"]] if os.environ.get("GPU_LOCK") else []
        tmpj = tempfile.mktemp(suffix=".json")
        p = subprocess.run(lock + [sys.executable, os.path.join(here, "timeline.py"), cub, tmpj, cfg],
                           env=env, capture_output=True, text=True)
        if not any(f.endswith(".wp") for f in os.listdir(work)):
            print(f"{cfg}: recording failed: {(p.stderr or p.stdout)[-400:]}", flush=True)
            continue
        dst = os.path.join(outdir, f"{name}_{block}.rpath")
        q = subprocess.run([sys.executable, conv, work, dst, "--regs", "64"], capture_output=True, text=True)
        shutil.rmtree(work, ignore_errors=True)
        if q.returncode != 0 or not os.path.exists(dst):
            print(f"{cfg}: conversion failed: {q.stderr[-400:]}", flush=True)
            continue
        with open(dst, "rb") as f, gzip.open(dst + ".gz", "wb", compresslevel=9) as g:
            shutil.copyfileobj(f, g)
        os.remove(dst)
        print(f"{cfg}: {os.path.getsize(dst + '.gz') / 1024:.0f} KB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or DEFAULT)
