"""Replay the dissect microbenchmarks in rsim and compare with hardware.

Every hardware sweep (stall_sweep, pair_sweep, issue_sweep, rfport) patched
control bits or registers in a cubin; here the same patches are applied to
the SASS listing (hex words for control bits, operand text for registers),
the patched kernel is simulated (`rsim timeline`, one CTA) and the cycles
between its two clock reads are compared with what the GPU measured.

    python3 harness.py --sass DIR --hw DIR [--calibration c.json] [--suite pair,issue,rfport,stall]

Prints one line per configuration and a summary (mean |error| per suite);
--json writes all rows.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubin import CTRL_FIELDS, REG_FIELDS  # noqa: E402
from sasslib import parse, patch_listing  # noqa: E402

RSIM = "target/release/rsim"


def set_bits(w, lo, n, v):
    m = ((1 << n) - 1) << lo
    return (w & ~m) | ((v << lo) & m)


def word(i):
    return i.lo | (i.hi << 64)


def ctrl_word(i, **kw):
    w = word(i)
    for k, v in kw.items():
        w = set_bits(w, *CTRL_FIELDS[k], v)
    return w


REG_RE = re.compile(r"^(-|\|)?R(\d+)")


def rewrite_text(text, regs):
    """regs: {slot: n} with slot in rd, ra, rb, rc -> operand index 0..3."""
    g = re.match(r"^(@!?U?P[T0-9]\s+)?(\S+)\s+(.*)$", text)
    pre, op, rest = g.group(1) or "", g.group(2), g.group(3)
    ops = [o.strip() for o in rest.split(",")]
    for slot, n in regs.items():
        k = {"rd": 0, "ra": 1, "rb": 2, "rc": 3}[slot]
        if k < len(ops) and REG_RE.match(ops[k]):
            ops[k] = REG_RE.sub(lambda m: (m.group(1) or "") + f"R{n}", ops[k], count=1)
    return f"{pre}{op} {', '.join(ops)}"


def apply(text, fn, insts, patches):
    """patches: {pc: {"ctrl": {...}, "regs": {...}}}"""
    words = {}
    texts = {}
    by_pc = {i.pc: i for i in insts}
    for pc, p in patches.items():
        i = by_pc[pc]
        w = word(i)
        for k, v in p.get("ctrl", {}).items():
            w = set_bits(w, *CTRL_FIELDS[k], v)
        for k, v in p.get("regs", {}).items():
            w = set_bits(w, *REG_FIELDS[k], v)
        words[pc] = (w & (2**64 - 1), w >> 64)
        if p.get("regs"):
            texts[pc] = rewrite_text(i.text, p["regs"])
    out = patch_listing(text, fn, words)
    if texts:
        lines = []
        cur = None
        for line in out.splitlines():
            m = re.search(r"Function\s*:\s*(\S+)", line)
            if m:
                cur = m.group(1)
            m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
            if cur == fn and m and int(m.group(1), 16) in texts:
                line = line.replace(m.group(2), texts[int(m.group(1), 16)], 1)
            lines.append(line)
        out = "\n".join(lines) + "\n"
    return out


def sim_cycles(text, fn, block, calib, regs=64, extra=()):
    """Cycles between the first and last clock read of the slowest warp."""
    with tempfile.NamedTemporaryFile("w", suffix=".sass", delete=False) as f:
        f.write(text)
        src = f.name
    out = src + ".json"
    cmd = [RSIM, "timeline", src, "--kernel", fn, "--block", str(block), "--gpu", "b300",
           "--regs", str(regs), "--warm", "1", "-o", out] + (["--calibration", calib] if calib else []) + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(src)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-300:])
    d = json.load(open(out))
    os.unlink(out)
    funcs = parse(text)
    pcs = [i.pc for i in funcs[fn] if "CLOCK" in " ".join(i.operands)]
    # CTA span, as the hardware sweeps measure it: last warp's second clock
    # read minus the first warp's first clock read.
    t0, t1 = None, None
    for tl in d["warps"]:
        for w in tl["warps"]:
            first = {}
            iss = w["iss"]
            for c, pc in zip(iss[0::2], iss[1::2]):
                first.setdefault(pc, c)
            if pcs[0] in first and pcs[-1] in first:
                t0 = first[pcs[0]] if t0 is None else min(t0, first[pcs[0]])
                t1 = first[pcs[-1]] if t1 is None else max(t1, first[pcs[-1]])
    return (t1 - t0) if t0 is not None else 0


def body(insts):
    clk = [k for k, i in enumerate(insts) if "CLOCK" in " ".join(i.operands)]
    return insts[clk[0] + 1:clk[-1]]


def suite_pair(sass_dir, hw_dir, calib):
    text = open(os.path.join(sass_dir, "pair.sass")).read()
    funcs = parse(text)
    hw = json.load(open(os.path.join(hw_dir, "pair_sweep.json")))
    rows = []
    for fn, r in hw.items():
        if "runs" not in r or fn not in funcs:
            continue
        b = body(funcs[fn])
        xs = [i for i in b if i.base == r["opcode"]]
        fill = [i for i in b if i.base == "FMUL"] if fn.startswith("p_") else []
        own = 0
        for i in xs:
            if i.ctrl.wbar != 7:
                own |= 1 << i.ctrl.wbar
        for key, hv in r["runs"].items():
            if hv is None:
                continue
            s, w = key.split(":")
            patches = {}
            if s != "orig":
                for i in xs:
                    patches[i.pc] = {"ctrl": dict(stall=int(s), yld=1, wait=i.ctrl.wait & ~own, wbar=7, rbar=7)}
                for i in fill:
                    patches[i.pc] = {"ctrl": dict(stall=1, yld=1, wait=i.ctrl.wait & ~own)}
            t = apply(text, fn, funcs[fn], patches)
            sv = sim_cycles(t, fn, 128 * int(w), calib) / len(xs)
            rows.append({"suite": "pair", "cfg": f"{fn} s={s} W={w}", "hw": hv, "sim": sv})
    return rows


def suite_issue(sass_dir, hw_dir, calib):
    text = open(os.path.join(sass_dir, "issue.sass")).read()
    funcs = parse(text)
    hw = json.load(open(os.path.join(hw_dir, "issue_sweep.json")))
    rows = []
    for fn, r in hw.items():
        if fn not in funcs:
            continue
        b = body(funcs[fn])
        tg = [i for i in b if i.base == r["opcode"]]
        ex = tg[0]
        slots = ["rd"] + [s for s, o in zip(["ra", "rb", "rc"], ex.operands[1:]) if o.startswith("R")]
        n = len(tg)
        for s in range(1, 12):
            patches = {}
            for k, i in enumerate(tg):
                regs = {"rd": 200 + (k % 48), "ra": 128 + (k % 8), "rb": 137 + (k % 8), "rc": 146 + (k % 8)}
                patches[i.pc] = {"ctrl": dict(stall=s, wait=0, wbar=7, rbar=7, reuse=0, yld=1),
                                 "regs": {k2: v for k2, v in regs.items() if k2 in slots}}
            t = apply(text, fn, funcs[fn], patches)
            sv = sim_cycles(t, fn, 32, calib, regs=255) / n
            rows.append({"suite": "issue", "cfg": f"{fn} s={s}", "hw": r["indep"][f"s{s}y1"], "sim": sv})
    return rows


def suite_rfport(sass_dir, hw_dir, calib):
    text = open(os.path.join(sass_dir, "issue_mw.sass")).read()
    funcs = parse(text)
    hw = json.load(open(os.path.join(hw_dir, "rfport.json")))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rfpatterns import patterns
    rows = []
    for fn, r in hw.items():
        nsrc = 2 if fn in ("m_fmul", "m_fadd") else 3
        tg = [i for i in body(funcs[fn]) if i.base == r["opcode"]]
        for name, pat in patterns(nsrc).items():
            if name not in r:
                continue
            patches = {}
            for k, i in enumerate(tg):
                rd, src, reuse = pat(k)
                regs = {"rd": rd}
                regs.update(dict(zip(("ra", "rb", "rc"), src)))
                patches[i.pc] = {"ctrl": dict(stall=1, yld=1, wait=0, wbar=7, rbar=7, reuse=reuse), "regs": regs}
            t = apply(text, fn, funcs[fn], patches)
            for w, hv in r[name].items():
                sv = sim_cycles(t, fn, 128 * int(w), calib) / (len(tg) * int(w))
                rows.append({"suite": "rfport", "cfg": f"{fn} {name} W={w}", "hw": hv, "sim": sv})
    return rows


def suite_stall(sass_dir, hw_dir, calib):
    text = open(os.path.join(sass_dir, "chain.sass")).read()
    funcs = parse(text)
    hw = json.load(open(os.path.join(hw_dir, "stall_sweep.json")))
    rows = []
    for key, r in hw.items():
        if "sweep" not in r:
            continue
        fn = key.split(":")[0]
        tg = [i for i in body(funcs[fn]) if i.base == r["opcode"]]
        # Reference (unpatched) and the legal stall values.
        rows.append({"suite": "stall", "cfg": f"{key} orig", "hw": r["ref_cycles"] / r["n_chain"],
                     "sim": sim_cycles(text, fn, 32, calib) / r["n_chain"]})
        for x in r["sweep"]:
            s = x["stall"]
            if s not in (1, 2, 4, 8, 11) or not x.get("cycles"):
                continue
            patches = {i.pc: {"ctrl": dict(stall=s, wbar=7, yld=1)} for i in tg}
            t = apply(text, fn, funcs[fn], patches)
            rows.append({"suite": "stall", "cfg": f"{key} s={s}", "hw": x["cycles_per_op"],
                         "sim": sim_cycles(t, fn, 32, calib) / r["n_chain"]})
    return rows


def main():
    global RSIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--hw", required=True)
    ap.add_argument("--calibration")
    ap.add_argument("--rsim", default=RSIM)
    ap.add_argument("--suite", default="stall,issue,pair,rfport")
    ap.add_argument("--json")
    ap.add_argument("-q", action="store_true")
    a = ap.parse_args()
    RSIM = a.rsim
    rows = []
    for s in a.suite.split(","):
        f = {"pair": suite_pair, "issue": suite_issue, "rfport": suite_rfport, "stall": suite_stall}[s]
        try:
            got = f(a.sass, a.hw, a.calibration)
        except FileNotFoundError as e:
            print(f"{s}: skipped ({e})")
            continue
        for r in got:
            r["err"] = (r["sim"] - r["hw"]) / r["hw"]
            if not a.q:
                print(f"{r['suite']:7} {r['cfg']:40} hw {r['hw']:7.2f}  sim {r['sim']:7.2f}  {100 * r['err']:+6.1f}%")
        rows += got
    for s in a.suite.split(","):
        e = [abs(r["err"]) for r in rows if r["suite"] == s]
        if e:
            e.sort()
            print(f"== {s}: {len(e)} configs, mean |err| {100 * sum(e) / len(e):.1f}%, "
                  f"median {100 * e[len(e) // 2]:.1f}%, within 5%: {sum(x <= 0.05 for x in e)}/{len(e)}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
