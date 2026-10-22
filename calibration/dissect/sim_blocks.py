"""Simulated cycles between the two clock reads of each microbenchmark
kernel (chain.cu, issue.cu, pair.cu), for comparison with the hardware
numbers. Uses `rsim timeline` (one CTA, warp 0 on SM 0).

    python3 sim_blocks.py listing.sass [--block 32] [--calibration c.json] [--only f1,f2]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from sasslib import parse  # noqa: E402


def sim_cycles(rsim, sass, fn, block, calib, regs=64, extra=()):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    cmd = [rsim, "timeline", sass, "--kernel", fn, "--block", str(block), "--gpu", "b300",
           "--regs", str(regs), "-o", path] + (["--calibration", calib] if calib else []) + list(extra)
    subprocess.run(cmd, check=True, capture_output=True)
    d = json.load(open(path))
    os.unlink(path)
    return d["warps"][0]["warps"]


def block_cycles(warps, pcs):
    """Per warp: issue(second clock) - issue(first clock)."""
    out = []
    for w in warps:
        iss = w["iss"]
        first = {}
        for c, pc in zip(iss[0::2], iss[1::2]):
            first.setdefault(pc, c)
        if pcs[0] in first and pcs[-1] in first:
            out.append(first[pcs[-1]] - first[pcs[0]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sass")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--calibration")
    ap.add_argument("--rsim", default="target/release/rsim")
    ap.add_argument("--only")
    a = ap.parse_args()
    funcs = parse(open(a.sass).read())
    for fn, insts in funcs.items():
        if a.only and fn not in a.only.split(","):
            continue
        pcs = [i.pc for i in insts if "CLOCK" in " ".join(i.operands)]
        clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
        body = insts[clk[0] + 1:clk[-1]]
        ops = {}
        for i in body:
            ops[i.base] = ops.get(i.base, 0) + 1
        main_op = max(ops, key=ops.get)
        n = ops[main_op]
        cyc = block_cycles(sim_cycles(a.rsim, a.sass, fn, a.block, a.calibration), pcs)
        print(json.dumps({"fn": fn, "op": main_op, "n": n, "sim_cycles": max(cyc), "per_op": max(cyc) / n}))


if __name__ == "__main__":
    main()
