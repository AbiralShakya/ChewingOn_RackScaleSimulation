"""Run FA4 forward (bf16, b=4 h=16 s=4096 d=128, non-causal: the recorded
b300_fa4_s4096 configuration) with clock64 probes on every pipeline event
and write the per-warp event lists of CTAs 0..7 to a JSON file.

    python3 run_fa4_probe.py out.json [--b 4 --h 16 --s 4096]
"""
import argparse
import json
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--b", type=int, default=4)
ap.add_argument("--h", type=int, default=16)
ap.add_argument("--s", type=int, default=4096)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--off", action="store_true", help="no probes (overhead check)")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
NB = 8 * 32 * (1 << 15) * 16
buf = torch.zeros(NB // 8, dtype=torch.int64, device="cuda")
if not a.off:
    os.environ["FA4_PROBE_ADDR"] = str(buf.data_ptr())
import fa4_probe  # noqa: E402

classes = fa4_probe.install() if not a.off else {}
sys.argv = [sys.argv[0]]  # FA4's CuTe DSL parses sys.argv
from flash_attn.cute import flash_attn_func  # noqa: E402

q, k, v = (torch.randn(a.b, a.s, a.h, a.d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
flash_attn_func(q, k, v)  # JIT + warm-up
torch.cuda.synchronize()
times = []
for rep in range(a.reps):
    buf.zero_()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    flash_attn_func(q, k, v)
    en.record()
    torch.cuda.synchronize()
    times.append(st.elapsed_time(en) * 1e3)
h = buf.view(8, 32, 1 << 15, 2).cpu()
warps = {}
for cta in range(8):
    for w in range(32):
        r = h[cta, w]
        nz = r[:, 0] != 0
        if not nz.any():
            continue
        ev = sorted((int(c), int(t)) for c, t in r[nz].tolist())
        warps[f"{cta}:{w}"] = ev
json.dump({"times_us": times, "min_us": min(times), "classes": classes, "config": vars(a), "warps": warps}, open(a.out, "w"))
print("kernel us min", round(min(times), 1), "median", round(sorted(times)[len(times) // 2], 1), "warps recorded", len(warps),
      "events", sum(len(v) for v in warps.values()))
