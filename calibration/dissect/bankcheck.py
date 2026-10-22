"""Register-bank check on clean single- and multi-warp FMUL streams
(kernels/ifetch.cu f_256: load-free clock window). Sources are rewritten to
fixed bank patterns, stall 1, no reuse; cycles/instruction for 1 warp and
for the CTA span with 8 warps per SMSP (block 1024)."""
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse

cub, sass = sys.argv[1:3]
image = open(cub, "rb").read()
fn = "f_256"
insts = parse(open(sass).read())[fn]
clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
tg = [i for i in insts[clk[0] + 1:clk[-1]] if i.base == "FMUL"]
dev = Device(0)
inp = dev.alloc(4 * 256)
o = dev.alloc(4 * 16384)
dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))
P = {
    "orig regs": None,
    "a,b different banks": lambda k: (40 + k % 8, 20 + 2 * (k % 4), 29 + 2 * (k % 4)),
    "a,b same bank": lambda k: (40 + k % 8, 20 + 2 * (k % 4), 28 + 2 * (k % 4)),
    "a,b same bank, a reused": lambda k: (40 + k % 8, 20, 28 + 2 * (k % 4)),
    "a == b": lambda k: (40 + k % 8, 20 + 2 * (k % 4), 20 + 2 * (k % 4)),
}
for name, f in P.items():
    for reuse in ((0, 1) if name == "a,b same bank, a reused" else (0,)):
        c = Cubin(image)
        c.set_regcount(fn, 64)
        for k, i in enumerate(tg):
            c.set_ctrl(fn, i.pc, stall=1, yld=1, reuse=reuse)
            if f:
                rd, ra, rb = f(k)
                c.set_reg(fn, i.pc, "rd", rd)
                c.set_reg(fn, i.pc, "ra", ra)
                c.set_reg(fn, i.pc, "rb", rb)
        m = dev.load(c.data)
        kk = m.fn(fn)
        row = []
        for blk in (32, 1024):
            best = None
            for _ in range(6):
                dev.memset(o, 4 * 16384)
                kk.launch(1, blk, [("p", o), ("p", inp), ("i", 0)])
                tt = u32s(dev.d2h(o + 4 * 6144, 4 * 2 * (blk // 32)))
                span = (max(tt[1::2]) - min(tt[0::2])) & 0xFFFFFFFF
                best = span if best is None else min(best, span)
            per = best / len(tg) / (1 if blk == 32 else 8)
            row.append(per)
        m.unload()
        print(f"{name:26} reuse={reuse}  1 warp {row[0]:.2f}   8 warps/SMSP {row[1]:.2f} per inst per SMSP", flush=True)
