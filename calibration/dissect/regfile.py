"""Register-file read conflicts: random register assignments + regression.

For FMUL (2 register sources) and FFMA (3) from kernels/issue.cu, each trial
gives every one of the 256 block instructions random source registers
(R128..R199) and a random destination (R200..R247), stall s, no yield, no
reuse (or random reuse flags), and times one warp. The per-trial register
maps are saved with the cycle counts; regfile_fit.py regresses cycles on
candidate conflict features (same register / same bank mod 2,4,8 within an
instruction and against the previous one or two instructions).

    python3 regfile.py build/issue.cubin build/issue.sass out.json [trials]
"""
import json
import random
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse


def main(cub, sass, out, trials=300):
    trials = int(trials)
    image = open(cub, "rb").read()
    funcs = parse(open(sass).read())
    dev = Device(0)
    inp = dev.alloc(4 * 256)
    o = dev.alloc(4 * 64)
    dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))
    rng = random.Random(11)
    res = {}
    for fn, nsrc in (("s_fmul", 2), ("s_ffma", 3), ("s_imad", 3), ("s_fadd", 2)):
        insts = funcs[fn]
        clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
        body = insts[clk[0] + 1:clk[-1]]
        op = max({i.base for i in body}, key=lambda b: sum(1 for i in body if i.base == b))
        tg = [i for i in body if i.base == op]
        rows = []
        for t in range(trials):
            mode = t % 4  # 0: random, 1: low-entropy (few regs), 2: random + reuse, 3: stall 2
            stall = 2 if mode == 3 else 1
            pool = list(range(128, 200)) if mode != 1 else rng.sample(range(128, 200), 6)
            regs = []
            c = Cubin(image)
            c.set_regcount(fn, 255)
            for k, i in enumerate(tg):
                src = [rng.choice(pool) for _ in range(nsrc)]
                rd = rng.randrange(200, 248)
                reuse = rng.randrange(0, 1 << nsrc) if mode == 2 else 0
                c.set_ctrl(fn, i.pc, stall=stall, yld=1, wait=0, wbar=7, rbar=7, reuse=reuse)
                c.set_reg(fn, i.pc, "rd", rd)
                for slot, r in zip(("ra", "rb", "rc"), src):
                    c.set_reg(fn, i.pc, slot, r)
                regs.append([rd] + src + [reuse])
            rows.append({"mode": mode, "stall": stall, "image": bytes(c.data), "regs": regs})
        # GPU 3 is shared with other users' jobs, which can slow a whole
        # block down ~2x for milliseconds at a time: time every trial in
        # several rounds spread over the whole run and keep the minimum.
        for rnd in range(4):
            for row in rows:
                m = dev.load(row["image"])
                k = m.fn(fn)
                for _ in range(2):
                    dev.memset(o, 4 * 64)
                    k.launch(1, 32, [("p", o), ("p", inp), ("i", 0)])
                    cyc = u32s(dev.d2h(o, 4 * 33))[32]
                    row["cycles"] = min(row.get("cycles", cyc), cyc)
                m.unload()
        for row in rows:
            del row["image"]
        res[fn] = {"opcode": op, "nsrc": nsrc, "n": len(tg), "trials": rows}
        cs = [r["cycles"] / len(tg) for r in rows]
        print(fn, op, f"cycles/inst min {min(cs):.2f} max {max(cs):.2f}", flush=True)
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
