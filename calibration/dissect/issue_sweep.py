"""Issue rate, register-bank conflicts and operand reuse for one warp.

kernels/issue.cu puts 256 instructions of one opcode between two clock reads.
This tool rewrites their registers and control bits in the cubin:

  indep   every instruction independent (fresh destination, read-only
          sources), stall s = 1 or 2, no scoreboard waits: cycles/instruction
          is the per-warp issue interval of the opcode's pipe.
  banks   FFMA/IMAD/... with Ra, Rb, Rc placed at bank offsets (db, dc) mod 8
          from each other: the extra cycles reveal the bank count and the
          conflict cost.
  reuse   the conflicting configuration again with the reuse flag set on the
          operand that repeats, to see if the reuse cache removes the conflict.

    python3 issue_sweep.py build/issue.cubin build/issue.sass out.json
"""
import json
import struct
import subprocess
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse

THREE_SRC = {"FFMA", "IMAD", "LOP3", "SHF", "HFMA2", "FMUL", "FADD", "IADD3"}


def body(insts):
    clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
    return insts[clk[0] + 1:clk[-1]]


def main(cub, sass, out, only=None):
    image = open(cub, "rb").read()
    funcs = parse(open(sass).read())
    dev = Device(0)
    inp = dev.alloc(4 * 256)
    o = dev.alloc(4 * 64)
    dev.h2d(inp, struct.pack("<256I", *([0x3f000000 + i for i in range(256)])))
    res = {}

    def run(img, fn, reps=3):
        m = dev.load(img)
        k = m.fn(fn)
        best = None
        for _ in range(reps):
            dev.memset(o, 4 * 64)
            k.launch(1, 32, [("p", o), ("p", inp), ("i", 0)])
            c = u32s(dev.d2h(o, 4 * 33))[32]
            best = c if best is None else min(best, c)
        m.unload()
        return best

    for fn, insts in funcs.items():
        if only and fn not in only:
            continue
        b = body(insts)
        ops = {}
        for i in b:
            ops[i.base] = ops.get(i.base, 0) + 1
        main_op = max(ops, key=ops.get)
        tg = [i for i in b if i.base == main_op]
        n = len(tg)
        r = {"opcode": main_op, "sass": sorted({i.opcode for i in tg}), "n": n,
             "other": {k: v for k, v in ops.items() if k != main_op},
             "orig_cycles": run(image, fn)}
        # Which register fields does this instruction have? (operands that are R regs)
        ex = tg[0]
        slots = ["rd"] + [s for s, o in zip(["ra", "rb", "rc"], ex.operands[1:]) if o.startswith("R")]

        def patched(stall, regs_of, reuse_of=None, yld=None):
            c = Cubin(image)
            c.set_regcount(fn, 255)
            for k, i in enumerate(tg):
                c.set_ctrl(fn, i.pc, stall=stall, wait=0, wbar=7, rbar=7,
                           reuse=reuse_of(k) if reuse_of else 0)
                if yld is not None:
                    c.set_ctrl(fn, i.pc, yld=yld)
                for slot, reg in regs_of(k).items():
                    if slot in slots:
                        c.set_reg(fn, i.pc, slot, reg)
            return c

        def indep(k):
            return {"rd": 200 + (k % 48), "ra": 128 + (k % 8), "rb": 137 + (k % 8),
                    "rc": 146 + (k % 8)}

        r["indep"] = {}
        for s in range(0, 16):
            for y in (0, 1):
                c = patched(s, indep, yld=y)
                r["indep"][f"s{s}y{y}"] = run(c.data, fn) / n
        if r == 0:
            pass
        # Bank sweep: Ra at bank base, Rb at +db, Rc at +dc (mod 8).
        if main_op in THREE_SRC and len(slots) >= 3:
            grid = {}
            for db in range(8):
                for dc in range(8 if len(slots) == 4 else 1):
                    def regs(k, db=db, dc=dc):
                        j, base = k % 3, k % 8
                        ra = 128 + 32 * j + base
                        return {"rd": 200 + (k % 48), "ra": ra, "rb": ra + 8 + db, "rc": ra + 16 + dc}
                    c = patched(1, regs)
                    grid[f"{db},{dc}"] = run(c.data, fn) / n
            r["banks"] = grid

            # Same-bank Ra/Rb (db=0): Ra repeated across consecutive instructions
            # with and without its reuse flag.
            def rep(k):
                return {"rd": 200 + (k % 48), "ra": 128, "rb": 136 + 8 * (k % 4), "rc": 147 + 8 * (k % 4)}
            r["reuse"] = {
                "same_bank_a_b_noreuse": run(patched(1, rep).data, fn) / n,
                "same_bank_a_b_reuse_a": run(patched(1, rep, lambda k: 1).data, fn) / n,
                "same_bank_a_b_reuse_b": run(patched(1, rep, lambda k: 2).data, fn) / n,
            }
            if len(slots) == 4:
                def rep3(k):
                    return {"rd": 200 + (k % 48), "ra": 128, "rb": 136, "rc": 144 + 8 * (k % 4)}
                r["reuse"]["three_same_bank_noreuse"] = run(patched(1, rep3).data, fn) / n
                r["reuse"]["three_same_bank_reuse_ab"] = run(patched(1, rep3, lambda k: 3).data, fn) / n
        res[fn] = r
        print(json.dumps({fn: r}), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4], only=sys.argv[4].split(",") if len(sys.argv) > 4 else None)
