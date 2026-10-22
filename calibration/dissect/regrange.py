"""Where does the extra issue cycle of independent ALU streams come from?

One warp, 256 FMULs/FFMAs (kernels/issue_mw.cu), stall s = 1..4, no waits.
Variants change one thing at a time:
  * register index range (R20.. vs R140..) and allocation size (64 / 255),
  * the destination pattern: one fixed register, two alternating registers
    of the same bank, of different banks, or 16 rotating registers,
  * the sources: fixed with/without reuse, or rotating.
"""
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse

cub, sass = sys.argv[1:3]
image = open(cub, "rb").read()
funcs = parse(open(sass).read())
dev = Device(0)
inp = dev.alloc(4 * 256)
o = dev.alloc(4 * 8192)
dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))


def run(fn, tg, regcount, s, regs_of, reuse_of=lambda k: 0):
    c = Cubin(image)
    c.set_regcount(fn, regcount)
    for k, i in enumerate(tg):
        c.set_ctrl(fn, i.pc, stall=s, yld=1, wait=0, wbar=7, rbar=7, reuse=reuse_of(k))
        for slot, r in regs_of(k).items():
            c.set_reg(fn, i.pc, slot, r)
    m = dev.load(c.data)
    kk = m.fn(fn)
    best = None
    for _ in range(7):
        dev.memset(o, 4 * 8192)
        kk.launch(1, 32, [("p", o), ("p", inp), ("i", 0)])
        cyc = u32s(dev.d2h(o + 4 * 4096, 4))[0]
        best = cyc if best is None else min(best, cyc)
    m.unload()
    return best / len(tg)


for fn in ("m_fmul", "m_ffma"):
    insts = funcs[fn]
    clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
    tg = [i for i in insts[clk[0] + 1:clk[-1]] if i.base in ("FMUL", "FFMA")]
    variants = {}
    for base, rc in ((20, 64), (20, 255), (140, 255)):
        variants[f"rotating, R{base}+, alloc {rc}"] = (rc, lambda k, b=base: {
            "rd": b + 24 + k % 16, "ra": b + 2 * (k % 4), "rb": b + 9 + 2 * (k % 4), "rc": b + 16 + 2 * (k % 4)})
    variants["dst fixed R44, srcs rotating"] = (64, lambda k: {"rd": 44, "ra": 20 + 2 * (k % 4), "rb": 29 + 2 * (k % 4), "rc": 36 + 2 * (k % 4)})
    variants["dst R44/R46 (same bank)"] = (64, lambda k: {"rd": 44 + 2 * (k % 2), "ra": 20 + 2 * (k % 4), "rb": 29 + 2 * (k % 4), "rc": 36 + 2 * (k % 4)})
    variants["dst R44/R45 (alternate bank)"] = (64, lambda k: {"rd": 44 + (k % 2), "ra": 20 + 2 * (k % 4), "rb": 29 + 2 * (k % 4), "rc": 36 + 2 * (k % 4)})
    variants["srcs fixed R20,R29,R38 no reuse"] = (64, lambda k: {"rd": 44 + k % 16, "ra": 20, "rb": 29, "rc": 38})
    variants["all fixed (chain-like regs)"] = (64, lambda k: {"rd": 44, "ra": 20, "rb": 29, "rc": 38})
    variants["dst = src a (R44 <- R44)"] = (64, lambda k: {"rd": 44, "ra": 44, "rb": 29, "rc": 38})
    for name, (rc, f) in variants.items():
        row = [run(fn, tg, rc, s, f) for s in (1, 2, 3, 4)]
        print(f"{fn} {name:36} " + "  ".join(f"s{s}:{v:.2f}" for s, v in zip((1, 2, 3, 4), row)), flush=True)
    row = [run(fn, tg, 64, s, variants["srcs fixed R20,R29,R38 no reuse"][1], lambda k: 7) for s in (1, 2, 3, 4)]
    print(f"{fn} {'srcs fixed, reuse a,b,c':36} " + "  ".join(f"s{s}:{v:.2f}" for s, v in zip((1, 2, 3, 4), row)), flush=True)
