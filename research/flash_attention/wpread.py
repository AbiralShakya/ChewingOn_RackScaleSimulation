"""Reader for the NVBit warp_path tool's output (.wp, .sass, .json)."""
import json
import re
import struct
from collections import defaultdict

REC = struct.Struct("<IHBBIIIIQQQQ")  # 56 bytes, see nvbit_warp_path/common.h
WP_BASIC, WP_MBAR, WP_WAIT, WP_MEM, WP_TMA, WP_UREG, WP_BAR = range(7)


def read_listing(path):
    """[(offset, opcode, operands, full_text, lo, hi)] in instruction-index order."""
    insts = []
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*;?\s*/\* (0x[0-9a-f]+) \*/", lines[i])
        if m:
            hi = int(re.search(r"0x[0-9a-f]+", lines[i + 1]).group(0), 16)
            text = m.group(2).strip()
            body = re.sub(r"^@!?U?P\w+\s+", "", text)
            op = body.split()[0]
            insts.append({"off": int(m.group(1), 16), "op": op, "text": text, "lo": int(m.group(3), 16), "hi": hi})
            i += 2
        else:
            i += 1
    return insts


def read_wp(path):
    """Yield dicts per record, in file order."""
    data = open(path, "rb").read()
    p = 0
    n = len(data)
    while p + REC.size <= n:
        cta, warp, kind, aux, idx, active, predon, _pad, a, b, c, clk = REC.unpack_from(data, p)
        p += REC.size
        r = {"cta": cta, "warp": warp, "kind": kind, "aux": aux, "idx": idx, "active": active,
             "predon": predon, "a": a, "b": b, "c": c, "clk": clk}
        if kind == WP_MEM:
            r["addrs"] = struct.unpack_from("<32Q", data, p)
            p += 256
        elif kind == WP_TMA:
            nruns = b
            r["runs"] = [struct.unpack_from("<QQ", data, p + 16 * i) for i in range(nruns)]
            p += 16 * nruns
            r["handle"] = data[p:p + 128]
            p += 128
        yield r


def per_warp(path):
    w = defaultdict(list)
    for r in read_wp(path):
        w[(r["cta"], r["warp"])].append(r)
    return w
