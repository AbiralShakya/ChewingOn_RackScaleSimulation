"""Register-file read bandwidth per SM sub-partition (kernels/issue_mw.cu).

W warps per SMSP (block = 128 W, W = 1, 2, 4, 8) each run 256 independent
instructions with registers rewritten to a chosen pattern (stall 1, no
waits). Aggregate cycles per instruction per SMSP = slowest warp's block
time / (256 W). Patterns vary how many register-file reads each instruction
needs (3 sources, 2, reuse-cached) and how they fall into banks.

    python3 rfport.py build/issue_mw.cubin build/issue_mw.sass out.json [kernels]
"""
import json
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from rfpatterns import patterns
from sasslib import parse

WARPS = [1, 2, 4, 8]
REGCOUNT = 64


def main(cub, sass, out, only=None):
    image = open(cub, "rb").read()
    funcs = parse(open(sass).read())
    dev = Device(0)
    inp = dev.alloc(4 * 256)
    o = dev.alloc(4 * 8192)
    dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))
    res = {}
    for fn, nsrc in (("m_ffma", 3), ("m_fmul", 2), ("m_imad", 3), ("m_lop3", 3), ("m_hfma2", 3), ("m_fadd", 2)):
        if only and fn not in only:
            continue
        insts = funcs[fn]
        clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
        body = insts[clk[0] + 1:clk[-1]]
        op = max({i.base for i in body}, key=lambda b: sum(1 for i in body if i.base == b))
        tg = [i for i in body if i.base == op]
        n = len(tg)
        r = {"opcode": op, "n": n}
        for name, pat in patterns(nsrc).items():
            c = Cubin(image)
            c.set_regcount(fn, REGCOUNT)
            for k, i in enumerate(tg):
                rd, src, reuse = pat(k)
                c.set_ctrl(fn, i.pc, stall=1, yld=1, wait=0, wbar=7, rbar=7, reuse=reuse)
                c.set_reg(fn, i.pc, "rd", rd)
                for slot, x in zip(("ra", "rb", "rc"), src):
                    c.set_reg(fn, i.pc, slot, x)
            m = dev.load(c.data)
            k_ = m.fn(fn)
            row = {}
            for w in WARPS:
                best = None
                for _ in range(5):
                    dev.memset(o, 4 * 8192)
                    k_.launch(1, 128 * w, [("p", o), ("p", inp), ("i", 0)])
                    tt = u32s(dev.d2h(o + 4 * 6144, 4 * 2 * 4 * w))
                    # CTA span: max(end) - min(start) over warps. A per-warp
                    # window misses the time other warps ran before it read
                    # its first clock (greedy scheduling).
                    cyc = (max(tt[1::2]) - min(tt[0::2])) & 0xFFFFFFFF
                    best = cyc if best is None else min(best, cyc)
                row[w] = best / (n * w)
            m.unload()
            r[name] = row
            print(f"{fn:8} {name:22} " + "  ".join(f"W{w}:{v:.2f}" for w, v in row.items()), flush=True)
        res[fn] = r
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4], only=sys.argv[4].split(",") if len(sys.argv) > 4 else None)
