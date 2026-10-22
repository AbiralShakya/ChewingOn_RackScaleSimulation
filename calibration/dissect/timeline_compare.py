"""Compare per-warp clock64 probe timelines: hardware (timeline.py) vs rsim.

    python3 timeline_compare.py timeline_hw.json timeline.sass out.json \
        [--rsim target/release/rsim] [--calibration calibration.json] [--extra "--lrr"]

For each (kernel, block) the simulator runs the same SASS (`rsim timeline`)
and reports the issue cycle of every instruction of every warp on SM 0. The
probe instructions (CS2R/S2UR of SR_CLOCKLO) are matched to the hardware
probes in program order. Both timelines are taken relative to the earliest
probe 0 of any warp. Reported per configuration:

  mae         mean |sim - hw| over every (warp, probe), cycles
  seg_err     sum |sim segment - hw segment| / sum hw segment, where a
              segment is the time between consecutive probes of one warp
  span_err    (sim - hw) / hw for the last probe of the slowest warp
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from sasslib import parse  # noqa: E402


SKIP = 0


def probe_pcs(insts):
    return [i.pc for i in insts if "CLOCK" in " ".join(i.operands)]


def sim_timeline(rsim, sass, kernel, block, calib, extra):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    cmd = [rsim, "timeline", sass, "--kernel", kernel, "--block", str(block), "--gpu", "b300",
           "--regs", "64", "-o", path] + (["--calibration", calib] if calib else []) + extra
    subprocess.run(cmd, check=True, capture_output=True)
    d = json.load(open(path))
    os.unlink(path)
    return d["warps"][0]["warps"]


def compare(hw, sass_funcs, rsim, sass, calib, extra):
    out = {}
    for cfg, h in hw.items():
        k, block = h["kernel"], h["block"]
        pcs = probe_pcs(sass_funcs[k])
        np_ = h["nprobe"]
        if len(pcs) != np_:
            print(f"{cfg}: {len(pcs)} probe instructions in SASS, {np_} probes recorded; skipped")
            continue
        warps = sim_timeline(rsim, sass, k, block, calib, extra)
        sim = {}
        for w in warps:
            iss = w["iss"]
            first = {}
            for c, pc in zip(iss[0::2], iss[1::2]):
                first.setdefault(pc, c)
            sim[w["widx"]] = [first.get(pc) for pc in pcs]
        if any(None in v for v in sim.values()) or len(sim) != len(h["warps"]):
            print(f"{cfg}: probes missing in the simulated timeline; skipped")
            continue
        s0 = min(v[0] for v in sim.values())
        S = [[x - s0 for x in sim[w]] for w in range(len(h["warps"]))]
        H = h["warps"]
        abs_err = [abs(S[w][p] - H[w][p]) for w in range(len(H)) for p in range(np_)]
        seg_e = seg_h = 0
        per_seg = [[0, 0] for _ in range(np_ - 1)]
        for w in range(len(H)):
            for p in range(1 + SKIP, np_):
                ds, dh = S[w][p] - S[w][p - 1], H[w][p] - H[w][p - 1]
                seg_e += abs(ds - dh)
                seg_h += dh
                per_seg[p - 1][0] += ds
                per_seg[p - 1][1] += dh
        span_h = max(r[-1] for r in H)
        span_s = max(r[-1] for r in S)
        out[cfg] = {
            "kernel": k, "block": block, "warps": len(H),
            "mae": sum(abs_err) / len(abs_err),
            "seg_err": seg_e / max(seg_h, 1),
            "span_hw": span_h, "span_sim": span_s, "span_err": (span_s - span_h) / span_h,
            # Mean duration of each segment over warps (sim, hw).
            "segments": [[s / len(H), hw_ / len(H)] for s, hw_ in per_seg],
            "hw": H, "sim": S,
        }
        print(f"{cfg:18} warps {len(H):3}  span hw {span_h:7.0f} sim {span_s:7.0f} "
              f"({100 * out[cfg]['span_err']:+5.1f}%)  probe MAE {out[cfg]['mae']:7.1f} cyc  "
              f"segment err {100 * out[cfg]['seg_err']:5.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hw")
    ap.add_argument("sass")
    ap.add_argument("out")
    ap.add_argument("--rsim", default="target/release/rsim")
    ap.add_argument("--calibration")
    ap.add_argument("--extra", default="")
    ap.add_argument("--skip-first", action="store_true",
                    help="leave the first segment out of the segment error (its global loads use "
                         "synthetic addresses in a static-SASS replay)")
    ap.add_argument("--kernels", help="comma-separated kernel names to compare (default: all)")
    a = ap.parse_args()
    global SKIP
    SKIP = 1 if a.skip_first else 0
    hw = json.load(open(a.hw))
    if a.kernels:
        hw = {k: v for k, v in hw.items() if v["kernel"] in a.kernels.split(",")}
    funcs = parse(open(a.sass).read())
    res = compare(hw, funcs, a.rsim, a.sass, a.calibration, a.extra.split())
    tot_seg = sum(r["seg_err"] for r in res.values()) / max(len(res), 1)
    tot_span = sum(abs(r["span_err"]) for r in res.values()) / max(len(res), 1)
    print(f"mean segment error {100 * tot_seg:.1f}%, mean |span error| {100 * tot_span:.1f}% over {len(res)} configs")
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
