"""Validate the long-context fast path (rackscale.extrap) against full cycle-level runs."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
os.environ.setdefault("RACKSCALE_EXTRAP_CACHE", os.path.join(os.path.dirname(__file__), "results", "kernel_cache.json"))
from rackscale import extrap

SYSTEM = "hgx_b300_8"
CASES = [
    {"type": "attention", "batch": 1, "heads_q": 32, "heads_kv": 8, "sq": 16384, "skv": 16384, "d": 128, "causal": True},
    {"type": "attention", "batch": 1, "heads_q": 8, "heads_kv": 1, "sq": 16384, "skv": 8192, "d": 128, "causal": False},
    {"type": "gemm", "m": 262144, "n": 6144, "k": 4096},
    {"type": "ssd", "stage": "chunk_scan", "batch": 1, "seqlen": 65536, "heads": 32, "headdim": 128, "d_state": 128, "groups": 16, "chunk": 64},
    {"type": "ssd", "stage": "chunk_state", "batch": 1, "seqlen": 65536, "heads": 128, "headdim": 64, "d_state": 128, "groups": 8, "chunk": 256},
    {"type": "row_norm", "rows": 65536, "cols": 4096},
]
out = os.path.join(os.path.dirname(__file__), "results", "extrap_check.json")
res = json.load(open(out)) if os.path.exists(out) else []
seen = {json.dumps(r["spec"], sort_keys=True) for r in res}
for c in CASES:
    if json.dumps(c, sort_keys=True) in seen:
        continue
    t = time.time()
    r = extrap.check(SYSTEM, c)
    r["wall_s"] = time.time() - t
    print(f"{c['type']:10s} err {100*r['error']:+.2f}%  est {r['extrapolated_us']:.1f} sim {r['simulated_us']:.1f} us  ({r['wall_s']:.0f}s)", flush=True)
    res.append(r)
    json.dump(res, open(out, "w"), indent=1)
