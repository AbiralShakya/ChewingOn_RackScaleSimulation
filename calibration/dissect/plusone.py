"""Which cubin patch adds one cycle per instruction to a single warp?
f_256 of kernels/ifetch.cu (256 FMULs), stall s; variants change one field."""
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
orig_rc = Cubin(image).regcount(fn)
dev = Device(0)
inp = dev.alloc(4 * 256)
o = dev.alloc(4 * 8192)
dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))
K = {i.pc: k for k, i in enumerate(tg)}
print("original regcount", orig_rc, "example", tg[0].text, tg[0].ctrl)


def run(c):
    m = dev.load(c.data)
    k = m.fn(fn)
    v = []
    for _ in range(6):
        dev.memset(o, 4 * 8192)
        k.launch(1, 32, [("p", o), ("p", inp), ("i", 0)])
        v.append(u32s(dev.d2h(o + 4 * 4096, 4))[0] / len(tg))
    m.unload()
    return min(v[1:])


V = {
    "stall only": lambda c, i: None,
    "wait=0 wbar=7 rbar=7": lambda c, i: c.set_ctrl(fn, i.pc, wait=0, wbar=7, rbar=7),
    "reuse=0": lambda c, i: c.set_ctrl(fn, i.pc, reuse=0),
    "reuse a": lambda c, i: c.set_ctrl(fn, i.pc, reuse=1),
    "rd rewritten same": lambda c, i: c.set_reg(fn, i.pc, "rd", c.get_bits(fn, i.pc, 16, 8)),
    "rotating R20+ (regrange)": lambda c, i: (c.set_regcount(fn, 64), c.set_reg(fn, i.pc, "rd", 44 + K[i.pc] % 16), c.set_reg(fn, i.pc, "ra", 20 + 2 * (K[i.pc] % 4)), c.set_reg(fn, i.pc, "rb", 29 + 2 * (K[i.pc] % 4))),
    "rotating rd only": lambda c, i: (c.set_regcount(fn, 64), c.set_reg(fn, i.pc, "rd", 44 + K[i.pc] % 16)),
    "rotating ra only": lambda c, i: (c.set_regcount(fn, 64), c.set_reg(fn, i.pc, "ra", 20 + 2 * (K[i.pc] % 4))),
    "rotating rb only": lambda c, i: (c.set_regcount(fn, 64), c.set_reg(fn, i.pc, "rb", 29 + 2 * (K[i.pc] % 4))),
    "fixed ra R20 rb R29": lambda c, i: (c.set_regcount(fn, 64), c.set_reg(fn, i.pc, "ra", 20), c.set_reg(fn, i.pc, "rb", 29)),
    "fixed ra R20 rb R29 reuse ab": lambda c, i: (c.set_regcount(fn, 64), c.set_reg(fn, i.pc, "ra", 20), c.set_reg(fn, i.pc, "rb", 29), c.set_ctrl(fn, i.pc, reuse=3)),
}
for name, f in V.items():
    if len(sys.argv) > 3 and sys.argv[3] not in name:
        continue
    row = []
    for s in (1, 2, 4):
        c = Cubin(image)
        for i in tg:
            c.set_ctrl(fn, i.pc, stall=s, yld=1)
            f(c, i)
            pass
        row.append(run(c))
    print(f"{name:26} " + "  ".join(f"s{s}:{v:.2f}" for s, v in zip((1, 2, 4), row)), flush=True)
