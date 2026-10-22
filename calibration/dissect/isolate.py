"""Run a sweep once per kernel in a child process and merge the JSON outputs.

A patched kernel that traps (illegal instruction/address) leaves a sticky
error that poisons the whole CUDA context, so each kernel gets a process.

    python3 isolate.py <script.py> <cubin> <sass> <out.json> [kernels]
"""
import json
import os
import subprocess
import sys
import tempfile

from sasslib import parse

script, cub, sass, out = sys.argv[1:5]
only = sys.argv[5].split(",") if len(sys.argv) > 5 else None
res = {}
for fn in parse(open(sass).read()):
    if only and fn not in only:
        continue
    tmp = tempfile.mktemp(suffix=".json")
    # GPU 3 is shared: take the lock per kernel, not for the whole sweep.
    lock = ["flock", os.environ["GPU_LOCK"]] if os.environ.get("GPU_LOCK") else []
    p = subprocess.run(lock + [sys.executable, script, cub, sass, tmp, fn], capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0 or not os.path.exists(tmp):
        print(f"{fn}: failed: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode}", flush=True)
        res[fn] = {"error": p.stderr[-400:]}
    else:
        res.update(json.load(open(tmp)))
        os.unlink(tmp)
    sys.stdout.flush()
    json.dump(res, open(out, "w"), indent=1)
