"""Dispatch occupancy and pipe initiation interval per opcode (kernels/pair.cu).

For p_<X> (X interleaved with an independent FMUL) and a_<X> (X alone):
patch X's stall to s in STALLS (filler FMUL stall 1, no waits, no barriers
written) and time the block with W warps per SMSP (block = 128 * W). Report
cycles per X for the slowest warp. One warp per SMSP isolates per-warp
effects (dispatch occupancy: does X block its own warp beyond its stall
count?); more warps show the pipe's initiation interval shared by the SMSP.

    python3 pair_sweep.py build/pair.cubin build/pair.sass out.json
"""
import json
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse

STALLS = [1, 2, 3, 4, 6, 8, 11]
WARPS = [1, 2, 4, 8]  # per SMSP


def main(cub, sass, out, only=None):
    image = open(cub, "rb").read()
    funcs = parse(open(sass).read())
    dev = Device(0)
    buf = {}

    def setup():
        buf["inp"] = dev.alloc(4 * 256)
        buf["o"] = dev.alloc(4 * 8192)
        dev.h2d(buf["inp"], struct.pack("<256I", *([0x3f000000 + 1023 * i for i in range(256)])))
    setup()
    dev.on_reset.append(setup)
    res = {}
    for fn, insts in funcs.items():
        if only and fn not in only:
            continue
        clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
        body = insts[clk[0] + 1:clk[-1]]
        ops = {}
        for i in body:
            ops[i.base] = ops.get(i.base, 0) + 1
        xop = max((k for k in ops if k != "FMUL" or fn.endswith("ffma") is False), key=ops.get) \
            if fn.startswith("a_") else max((k for k in ops if k != "FMUL"), key=ops.get)
        xs = [i for i in body if i.base == xop]
        fill = [i for i in body if i.base == "FMUL"] if fn.startswith("p_") else []
        r = {"opcode": xop, "sass": sorted({i.opcode for i in xs}), "n": len(xs),
             "orig_stalls": sorted({i.ctrl.stall for i in xs}), "body_ops": ops, "runs": {}}

        def run(img, warps):
            m = dev.load(img)
            k = m.fn(fn)
            best = None
            for _ in range(3):
                dev.memset(buf["o"], 4 * 8192)
                k.launch(1, 128 * warps, [("p", buf["o"]), ("p", buf["inp"]), ("i", 0)])
                tt = u32s(dev.d2h(buf["o"] + 4 * 6144, 4 * 2 * 4 * warps))
                # CTA span: max(end) - min(start) over warps. A per-warp
                # window misses the time other warps ran before it read
                # its first clock (greedy scheduling).
                cyc = (max(tt[1::2]) - min(tt[0::2])) & 0xFFFFFFFF
                best = cyc if best is None else min(best, cyc)
            m.unload()
            return best

        # Only waits on barriers that X itself sets are dropped: X's address
        # operands may come from scoreboarded instructions (LDC, S2R).
        own = 0
        for i in xs:
            if i.ctrl.wbar != 7:
                own |= 1 << i.ctrl.wbar
        for s in [None] + STALLS:
            c = Cubin(image)
            if s is not None:
                for i in xs:
                    c.set_ctrl(fn, i.pc, stall=s, yld=1, wait=i.ctrl.wait & ~own, wbar=7, rbar=7)
                for i in fill:
                    c.set_ctrl(fn, i.pc, stall=1, yld=1, wait=i.ctrl.wait & ~own)
            for w in WARPS:
                try:
                    cyc = run(c.data, w)
                    r["runs"][f"{'orig' if s is None else s}:{w}"] = cyc / len(xs)
                except RuntimeError as e:
                    print(fn, s, w, e, flush=True)
                    raise
        res[fn] = r
        line = " ".join(f"{k}={v:.1f}" if v else f"{k}=ERR" for k, v in r["runs"].items())
        print(f"{fn:12} {xop:6} n={len(xs)} {line}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4], only=sys.argv[4].split(",") if len(sys.argv) > 4 else None)
