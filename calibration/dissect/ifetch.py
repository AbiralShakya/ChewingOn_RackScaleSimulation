"""Instruction-fetch cost vs code size (kernels/ifetch.cu): stall s of every
FMUL patched to 1, 2, 4, 8; one warp (block 32) and four warps on each
SMSP (block 512). Prints cycles per instruction; the first launch of each
patched module is reported separately (cold instruction caches)."""
import json
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse

cub, sass, out = sys.argv[1:4]
image = open(cub, "rb").read()
funcs = parse(open(sass).read())
dev = Device(0)
inp = dev.alloc(4 * 256)
o = dev.alloc(4 * 8192)
dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))
res = {}
for fn in sorted(funcs, key=lambda f: int(f.split("_")[1])):
    insts = funcs[fn]
    clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
    tg = [i for i in insts[clk[0] + 1:clk[-1]] if i.base == "FMUL"]
    for s in (1, 2, 4, 8):
        c = Cubin(image)
        for i in tg:
            c.set_ctrl(fn, i.pc, stall=s, yld=1)
        m = dev.load(c.data)
        k = m.fn(fn)
        row = {}
        for blk in (32, 512):
            v = []
            for _ in range(6):
                dev.memset(o, 4 * 8192)
                k.launch(1, blk, [("p", o), ("p", inp), ("i", 0)])
                v.append(max(u32s(dev.d2h(o + 4 * 4096, 4 * (blk // 32)))) / len(tg))
            row[blk] = {"cold": v[0], "warm": min(v[1:])}
        m.unload()
        res[f"{fn}:s{s}"] = {"n": len(tg), "stall": s, **{str(b): r for b, r in row.items()}}
        print(f"{fn:8} n={len(tg):6} s={s}  1 warp: warm {row[32]['warm']:.2f} cold {row[32]['cold']:.2f}"
              f"   16 warps: warm {row[512]['warm']:.2f} (per warp)", flush=True)
json.dump(res, open(out, "w"), indent=1)
