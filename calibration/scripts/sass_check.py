#!/usr/bin/env python3
"""Tie ALU measurements to the SASS that actually ran.

ptxas is free to fuse, fold or vectorise the PTX a bench writes (two dependent
adds become one IADD3, two maxes one FMNMX3, a pair of rcps vanishes). A
latency per *PTX op* is therefore not a latency per *SASS instruction*. For
every timed kernel this script counts, statically, the target PTX instruction in
the kernel's PTX and the target SASS opcode in its SASS. Both listings share the
same loop structure (NVVM unrolls before PTX is emitted), so the ratio is the
number of SASS instructions per PTX op, and the per-SASS figures follow.

Usage: sass_check.py build/bench_alu build/bench_alu.ptx results/raw/alu.json
Rewrites the JSON in place, adding a "sass" block to every op.
"""
import collections
import json
import re
import subprocess
import sys

# Ops timed as a chain of two instructions; merge.py subtracts the partner.
# OpSIN: ptxas emits FMUL.RZ (range reduction) ahead of every MUFU.SIN.
COMPOSITE = {"OpRCP": "OpFADD", "OpSHFL": "OpIADD", "OpSIN": "OpFMUL"}


def split_functions(text, header_re):
    funcs, cur, name = {}, [], None
    for line in text.splitlines():
        m = header_re.search(line)
        if m:
            if name:
                funcs[name] = cur
            name, cur = m.group(1), []
        elif name:
            cur.append(line)
    if name:
        funcs[name] = cur
    return funcs


def demangle(names):
    out = subprocess.run(["cu++filt"], input="\n".join(names), capture_output=True, text=True).stdout
    return dict(zip(names, out.splitlines()))


SASS_INST = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[0-9T]+\s+)?([A-Z][A-Z0-9_.]*)")


def sass_hist(lines):
    h = collections.Counter()
    for l in lines:
        m = SASS_INST.search(l)
        if m:
            h[m.group(1)] += 1
    return h


def ptx_token(asm):
    """The instruction a PTX template is about: the last one that is not a move."""
    insts = [i.strip() for i in re.split(r"[;{}]", asm) if i.strip()]
    insts = [i for i in insts if not i.startswith((".reg", "mov."))]
    return insts[-1].split()[0]


def ptx_count(lines, token):
    n = 0
    for l in lines:
        for t in re.split(r"[;{}]", l):
            t = t.strip()
            if t.startswith("@"):
                t = t.split(None, 1)[1] if " " in t else t
            if t.split(None, 1)[0:1] == [token]:
                n += 1
    return n


def main(binary, ptx_path, json_path):
    sass_text = subprocess.run(["cuobjdump", "-sass", binary], capture_output=True, text=True, check=True).stdout
    sass_funcs = split_functions(sass_text, re.compile(r"Function : (\S+)"))
    ptx_funcs = split_functions(open(ptx_path).read(), re.compile(r"^\.(?:visible|weak)\s+\.entry\s+(\S+?)\("))
    dm = demangle(list(set(sass_funcs) | set(ptx_funcs)))

    def find(funcs, pretty):
        # "lat_kernel<OpFADD>" -> demangled "void lat_kernel<OpFADD>(OpFADD::T*, int, unsigned long*)"
        for k, v in funcs.items():
            if pretty + "(" in dm.get(k, k):
                return v
        return None

    data = json.load(open(json_path))
    for name, op in data["ops"].items():
        token = ptx_token(op["ptx"])
        prefixes = op["sass_expected"].split(",")
        block = {"ptx_token": token}
        for kind in ("lat", "thr"):
            s_lines = find(sass_funcs, op[f"{kind}_kernel"])
            p_lines = find(ptx_funcs, op[f"{kind}_kernel"])
            if s_lines is None or p_lines is None:
                block[f"{kind}_error"] = "kernel not found"
                continue
            h = sass_hist(s_lines)
            n_sass = sum(c for o, c in h.items() if o.startswith(tuple(prefixes)) and ".MOV" not in o)
            n_ptx = ptx_count(p_lines, token)
            block[f"{kind}_sass_count"] = n_sass
            block[f"{kind}_ptx_count"] = n_ptx
            block[f"{kind}_sass_per_ptx"] = (n_sass / n_ptx) if n_ptx else None
            block[f"{kind}_opcodes_top"] = dict(h.most_common(8))
        r_lat = block.get("lat_sass_per_ptx")
        r_thr = block.get("thr_sass_per_ptx")
        if r_lat:
            block["latency_cycles"] = op["latency_cycles_per_ptx_op"] / r_lat
        if r_thr:
            block["throughput_sass_inst_per_clk_per_sm"] = op["throughput_ptx_ops_per_clk_per_sm"] * r_thr
        if name in COMPOSITE:
            block["composite_with"] = COMPOSITE[name]
        op["sass"] = block
        print(f"{name:10s} {token:22s} lat x{r_lat if r_lat is None else round(r_lat, 3)} thr x{r_thr if r_thr is None else round(r_thr, 3)}"
              f"  top={list(block.get('lat_opcodes_top', {}).items())[:3]}", file=sys.stderr)
    json.dump(data, open(json_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
