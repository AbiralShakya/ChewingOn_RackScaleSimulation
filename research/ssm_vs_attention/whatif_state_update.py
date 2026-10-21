#!/usr/bin/env python3
"""What-if: Mamba-2 70B decode on GB200 NVL72 with a state update that moves 8
headdim rows (4 KB of state) per CTA instead of the Triton kernel's 4 (2 KB).

    python research/ssm_vs_attention/whatif_state_update.py
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sweep  # noqa: E402

if __name__ == "__main__":
    pts = [dict(phase="decode", model="mamba2-70b-ssu8", tp=tp, batch=b, context=c)
           for tp in (4, 8) for b in (64, 256, 512) for c in (1024, 131072)]
    os.chdir(sweep.ROOT)
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(sweep.run_point, pts))
    for r in res:
        print(r["tp"], r["batch"], r["context"], round(r.get("step_us", 0)), round(r.get("tokens_per_s_per_gpu", 0)), flush=True)
    with open(os.path.join(HERE, "results", "sweep_whatif_ssu8.json"), "w") as f:
        json.dump(dict(points=res), f, indent=1)
