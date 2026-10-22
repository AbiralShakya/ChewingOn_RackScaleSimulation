"""True pipeline latency per opcode from stall-count patching.

For each kernel in kernels/chain.cu: find the dependent-chain instructions
(between the two clock reads), set their stall count to s = 1..15 with no
scoreboard wait, run one warp, and compare the result with the unpatched
kernel. Fixed-latency pipes have no interlock, so a stall below the latency
reads a stale register and the answer changes. Also records cycles per chain
instruction at each s, which shows whether the stall count is honoured.

    python3 stall_sweep.py build/chain.cubin build/chain.sass out.json
"""
import json
import struct
import sys

from cubin import Cubin
from cudrv import Device, u32s
from sasslib import parse

BRACKET = ("CS2R", "S2UR", "S2R")


def chain_pcs(insts):
    """Instructions between the two clock reads, minus the clock reads."""
    clk = [k for k, i in enumerate(insts) if i.base in BRACKET and "CLOCK" in " ".join(i.operands)]
    a, b = clk[0], clk[-1]
    return insts[a + 1:b], insts[a], insts[b]


def main(cub, sass, out, only=None):
    image = open(cub, "rb").read()
    funcs = parse(open(sass).read())
    dev = Device(0)
    buf = {}
    host = []
    import random
    random.seed(1)
    for i in range(256):
        # Floats near 1 so chains stay finite; ints arbitrary.
        host.append(struct.unpack("<I", struct.pack("<f", 0.5 + random.random() * 0.49))[0]
                    if i < 128 else random.getrandbits(32))

    def setup():
        buf["inp"] = dev.alloc(4 * 256)
        buf["o"] = dev.alloc(4 * 64)
        dev.h2d(buf["inp"], struct.pack("<256I", *host))
    setup()
    dev.on_reset.append(setup)
    res = {}
    for fn, insts in funcs.items():
        if only and fn != only:
            continue
        body, c0, c1 = chain_pcs(insts)
        ops = {}
        for i in body:
            ops[i.base] = ops.get(i.base, 0) + 1
        def run(img):
            m = dev.load(img)
            k = m.fn(fn)
            o = buf["o"]
            dev.memset(o, 4 * 64)
            k.launch(1, 32, [("p", o), ("p", buf["inp"]), ("i", 0)])
            r = u32s(dev.d2h(o, 4 * 33))
            m.unload()
            return r[:32], r[32]

        try:
            ref, ref_cyc = run(image)
        except RuntimeError as e:
            print(fn, "reference run failed", e)
            dev.reset()
            continue
        # Every opcode that makes up the chain body (>= N/2 copies) is swept
        # on its own; the other body instructions keep ptxas's stalls.
        for main_op in [k for k, v in ops.items() if v >= 32]:
            targets = [i for i in body if i.base == main_op]
            sweep = []
            # Legal encodings only: 1..11 without yield, 12..15 with it
            # (stall 0 and 12..15 without yield are never emitted by ptxas
            # and trap or stall the warp; see issue_sweep).
            for s in range(1, 16):
                c = Cubin(image)
                for i in targets:
                    c.set_ctrl(fn, i.pc, stall=s, wbar=7, yld=1 if s <= 11 else 0)
                vals, cyc = run(c.data)
                sweep.append({"stall": s, "correct": vals == ref, "cycles": cyc,
                              "cycles_per_op": cyc / len(targets)})
            lat = next((x["stall"] for x in sweep if x["correct"] and x["stall"] <= 11 and all(
                y["correct"] for y in sweep if y["stall"] >= x["stall"])), None)
            key = fn if main_op == max(ops, key=ops.get) else f"{fn}:{main_op}"
            res[key] = {"opcode": main_op, "sass": sorted({i.opcode for i in targets}),
                        "n_chain": len(targets), "body_ops": ops,
                        "orig_stalls": sorted({i.ctrl.stall for i in targets}),
                        "orig_scoreboarded": any(i.ctrl.wbar != 7 for i in targets),
                        "ref_cycles": ref_cyc, "latency": lat, "sweep": sweep}
            print(f"{key:18} {main_op:8} n={len(targets):3} orig stall {res[key]['orig_stalls']} "
                  f"sb={res[key]['orig_scoreboarded']} ref {ref_cyc/len(targets):.2f} c/op  "
                  f"=> latency {lat}   " +
                  " ".join(f"{x['stall']}:{'ok' if x['correct'] else '--'}/{x.get('cycles_per_op', -1):.1f}" for x in sweep),
                  flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
