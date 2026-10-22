"""Recover the compiler's fixed-latency table from the stall counts in SASS.

ptxas guarantees every fixed-latency dependence with stall counts alone (no
scoreboard), so for a producer P and a consumer C of its result the sum of
stall counts from P up to (not including) C is at least P's latency as ptxas
models it. The tightest such distance over a large corpus is that latency.
Run on cuobjdump -sass output:

    python3 mine_latency.py out.json listing1.sass listing2.sass ...
"""
import collections
import json
import re
import sys

from sasslib import files, regs_in, width_of

NOT_DST_FIRST = {
    "STG", "STS", "STL", "ST", "RED", "REDG", "SUST", "STSM", "STTM", "BAR", "BRA", "BRX",
    "JMP", "JMX", "CALL", "RET", "EXIT", "NOP", "BSSY", "BSYNC", "BREAK", "WARPSYNC", "YIELD",
    "KILL", "DEPBAR", "LDGDEPBAR", "MEMBAR", "FENCE", "ERRBAR", "CCTL", "UTMASTG", "UTMAPF",
    "UTMALDG", "UBLKCP", "UTCHMMA", "UTCIMMA", "UTCQMMA", "UTCOMMA", "UTCMMA", "UTCBAR", "UTCCP",
    "WARPGROUP", "NANOSLEEP", "LDGSTS", "ARRIVES", "UTMAREDG", "UBLKRED", "STAS", "SYNCS",
    "UCGABAR_ARV", "UCGABAR_WAIT", "ELECT", "ACQBULK", "ENDCOLLECTIVE", "BRA.U", "STG.E",
}
CONTROL = {"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "BREAK",
           "WARPSYNC", "KILL", "BPT"}


def reg_set(kind, n, w):
    if kind in ("R", "UR"):
        return [(kind, n + k) for k in range(w)]
    return [(kind, n)]


def dsts_srcs(i):
    ops = i.operands
    base = i.base
    dsts, srcs = [], []
    if not ops:
        pass
    elif base in NOT_DST_FIRST:
        for o in ops:
            for r in regs_in(o):
                srcs += reg_set(*r)
    else:
        first = regs_in(ops[0])
        w = width_of(i.opcode)
        for k, n, ww in first:
            dsts += reg_set(k, n, max(w, ww))
        rest = ops[1:]
        # Trailing predicate destinations (ISETP P0, P1 / IADD3 R0, P0, ...).
        while rest and re.fullmatch(r"!?U?P[T0-9]", rest[0]) and base.lstrip("U") in (
            "ISETP", "FSETP", "DSETP", "HSETP2", "PSETP", "PLOP3", "IADD3", "LEA", "VOTE",
            "R2P", "IADD", "FSET", "HSET2", "LOP3", "IMAD", "SHF", "FCHK", "DSETP",
        ) and not rest[0].startswith("!"):
            for k, n, ww in regs_in(rest[0]):
                dsts.append((k, n))
            rest = rest[1:]
        for o in rest:
            for k, n, ww in regs_in(o):
                srcs += reg_set(k, n, ww)
    if i.guard:
        for k, n, ww in regs_in(i.guard):
            srcs.append((k, n))
    return dsts, srcs


def key_of(i):
    """Producer key: base opcode plus the modifiers that change the datapath."""
    mods = [m for m in i.opcode.split(".")[1:] if m in (
        "WIDE", "HI", "X", "64", "128", "F32", "BF16", "F16", "TF32", "E4M3", "SAT", "FTZ", "RZ",
        "RN", "PACK_AB", "U32", "S32", "F64", "EX2", "RSQ", "RCP", "SIN", "COS", "LG2", "SQRT",
        "TANH", "MOV", "BFLY", "IDX", "UP", "DOWN", "IADD", "MIX", "AND", "OR", "XOR",
    )]
    return ".".join([i.base] + mods)


def mine(paths):
    dist = collections.defaultdict(list)       # (producer, consumer) -> [distance]
    prod = collections.defaultdict(list)       # producer -> [distance]
    var_lat = collections.Counter()            # producers that set a write barrier
    fixed = collections.Counter()
    nfunc = 0
    for path, funcs in files(paths):
        for fn, insts in funcs.items():
            nfunc += 1
            targets = set()
            for i in insts:
                if i.base in ("BRA", "BRX", "JMP", "BSSY", "CALL") and i.operands:
                    m = re.search(r"0x([0-9a-f]+)", i.operands[-1])
                    if m:
                        targets.add(int(m.group(1), 16))
            ds = [dsts_srcs(i) for i in insts]
            for a, ia in enumerate(insts):
                if ia.ctrl.wbar != 7:
                    var_lat[key_of(ia)] += 1
                    continue
                d, _ = ds[a]
                d = [r for r in d if r[0] in ("R", "P", "UR", "UP")]
                if not d or ia.base in CONTROL:
                    continue
                fixed[key_of(ia)] += 1
                acc = 0
                for b in range(a + 1, min(a + 40, len(insts))):
                    ib = insts[b]
                    acc += insts[b - 1].ctrl.stall
                    if ib.pc in targets:
                        break
                    db, sb = ds[b]
                    hit = [r for r in d if r in sb]
                    if hit:
                        # Uniform and vector producers read through different
                        # paths; key on the consumer's opcode too.
                        dist[(key_of(ia), key_of(ib))].append(acc)
                        prod[key_of(ia)].append(acc)
                        break
                    if any(r in db for r in d) or ib.base in CONTROL:
                        break
    return nfunc, dist, prod, var_lat, fixed


def summarize(v):
    m = min(v)
    c = collections.Counter(v)
    return {"min": m, "n": len(v), "n_at_min": c[m], "mode": c.most_common(1)[0][0]}


if __name__ == "__main__":
    out, paths = sys.argv[1], sys.argv[2:]
    nfunc, dist, prod, var_lat, fixed = mine(paths)
    res = {
        "functions": nfunc,
        "producer": {k: summarize(v) for k, v in sorted(prod.items()) if len(v) >= 3},
        "pair": {f"{a} -> {b}": summarize(v) for (a, b), v in sorted(dist.items()) if len(v) >= 3},
        "scoreboarded": dict(var_lat.most_common()),
        "fixed_count": dict(fixed.most_common()),
    }
    json.dump(res, open(out, "w"), indent=1)
    print(f"{nfunc} functions")
    for k, s in sorted(res["producer"].items(), key=lambda x: -x[1]["n"])[:60]:
        print(f"  {k:28} min {s['min']:3} ({s['n_at_min']}/{s['n']}) mode {s['mode']}")
