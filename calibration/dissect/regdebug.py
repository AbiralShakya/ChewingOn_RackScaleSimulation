"""Why are some random-register FMUL blocks ~2x slower? Reload stability and
bisection between a fast and a slow register map from regfile.json."""
import json
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse

cub, sass, rj = sys.argv[1:4]
image = open(cub, "rb").read()
funcs = parse(open(sass).read())
d = json.load(open(rj))
dev = Device(0)
inp = dev.alloc(4 * 256)
o = dev.alloc(4 * 64)
dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))

fn = sys.argv[4] if len(sys.argv) > 4 else "s_fmul"
r = d[fn]
nsrc = r["nsrc"]
insts = funcs[fn]
clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
body = insts[clk[0] + 1:clk[-1]]
tg = [i for i in body if i.base == r["opcode"]]
n = len(tg)


def build(regs, stall=1):
    c = Cubin(image)
    c.set_regcount(fn, 255)
    for i, rg in zip(tg, regs):
        c.set_ctrl(fn, i.pc, stall=stall, yld=1, wait=0, wbar=7, rbar=7, reuse=rg[1 + nsrc])
        c.set_reg(fn, i.pc, "rd", rg[0])
        for slot, x in zip(("ra", "rb", "rc"), rg[1:1 + nsrc]):
            c.set_reg(fn, i.pc, slot, x)
    return c.data


def t(img, reps=3):
    m = dev.load(img)
    k = m.fn(fn)
    out = []
    for _ in range(reps):
        dev.memset(o, 4 * 64)
        k.launch(1, 32, [("p", o), ("p", inp), ("i", 0)])
        out.append(u32s(dev.d2h(o, 4 * 33))[32] / n)
    m.unload()
    return min(out)


tr = [x for x in r["trials"] if x["mode"] == 0]
fast = min(tr, key=lambda x: x["cycles"])
slow = max(tr, key=lambda x: x["cycles"])
print("recorded fast %.2f slow %.2f" % (fast["cycles"] / n, slow["cycles"] / n))
print("reload fast:", [round(t(build(fast["regs"])), 2) for _ in range(4)])
print("reload slow:", [round(t(build(slow["regs"])), 2) for _ in range(4)])
# Bisection: first k instructions from slow, rest from fast.
for k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256):
    regs = slow["regs"][:k] + fast["regs"][k:]
    print("slow prefix", k, round(t(build(regs)), 2))
# Which register *values* matter: slow map with every source replaced by R128.
regs = [[x[0]] + [128] * nsrc + [0] for x in slow["regs"]]
print("slow dsts, all srcs R128", round(t(build(regs)), 2))
regs = [[200 + (k % 48)] + x[1:] for k, x in enumerate(slow["regs"])]
print("slow srcs, rotating dsts", round(t(build(regs)), 2))
regs = [[x[0]] + [128 + (k % 8) + 9 * j for j in range(nsrc)] + [0] for k, x in enumerate(slow["regs"])]
print("slow dsts, rotating srcs", round(t(build(regs)), 2))
regs = [[x[0]] + x[1:] for x in fast["regs"]]
regs = [[slow["regs"][k][0]] + fast["regs"][k][1:] for k in range(n)]
print("fast srcs, slow dsts", round(t(build(regs)), 2))
regs = [[fast["regs"][k][0]] + slow["regs"][k][1:] for k in range(n)]
print("slow srcs, fast dsts", round(t(build(regs)), 2))
