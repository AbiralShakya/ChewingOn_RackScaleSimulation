"""Minimal SASS listing parser for sm_90/sm_100/sm_103 (cuobjdump -sass / nvdisasm -hex).

Each instruction keeps both 64-bit words so that encoding fields can be checked
directly. Control bits (Volta+ layout, bits 105..125 of the 128-bit word, i.e.
bits 41..61 of the high word):

    [41:44] stall  [45] yield  [46:48] write barrier  [49:51] read barrier
    [52:57] wait mask  [58:61] operand reuse (a, b, c, d)
"""
import re
import sys

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\*\s*(0x[0-9a-f]{16})\s*\*/")
HI = re.compile(r"^\s*/\*\s*(0x[0-9a-f]{16})\s*\*/\s*$")
FUNC = re.compile(r"Function\s*:\s*(\S+)")


class Ctrl:
    __slots__ = ("stall", "yld", "wbar", "rbar", "wait", "reuse")

    def __init__(self, hi):
        b = hi >> 41
        self.stall = b & 0xF
        self.yld = (b >> 4) & 1
        self.wbar = (b >> 5) & 7
        self.rbar = (b >> 8) & 7
        self.wait = (b >> 11) & 0x3F
        self.reuse = (b >> 17) & 0xF

    def __repr__(self):
        w = "-" if self.wbar == 7 else self.wbar
        r = "-" if self.rbar == 7 else self.rbar
        return f"S{self.stall:02d} Y{self.yld} W{w} R{r} wait{self.wait:06b} reuse{self.reuse:04b}"


class Inst:
    __slots__ = ("func", "pc", "text", "guard", "opcode", "operands", "lo", "hi", "ctrl")

    def __init__(self, func, pc, text, lo):
        self.func, self.pc, self.text, self.lo = func, pc, text, lo
        self.hi = None
        self.ctrl = None
        t = text
        self.guard = None
        m = re.match(r"^(@!?U?P[T0-9])\s+(.*)$", t)
        if m:
            self.guard, t = m.group(1), m.group(2)
        parts = t.split(None, 1)
        self.opcode = parts[0]
        self.operands = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []

    @property
    def base(self):
        return self.opcode.split(".")[0]

    def set_hi(self, hi):
        self.hi = hi
        self.ctrl = Ctrl(hi)


def parse(text):
    funcs = {}
    cur = None
    last = None
    for line in text.splitlines():
        m = FUNC.search(line)
        if m:
            cur = m.group(1)
            funcs.setdefault(cur, [])
            last = None
            continue
        if cur is None:
            continue
        m = LINE.search(line)
        if m:
            pc = int(m.group(1), 16)
            last = Inst(cur, pc, m.group(2), int(m.group(3), 16))
            # nvdisasm -hex style: both words on one line.
            rest = line[m.end():]
            m2 = re.search(r"/\*\s*(0x[0-9a-f]{16})\s*\*/", rest)
            if m2:
                last.set_hi(int(m2.group(1), 16))
            funcs[cur].append(last)
            continue
        m = HI.match(line)
        if m and last is not None and last.hi is None:
            last.set_hi(int(m.group(1), 16))
    return {k: [i for i in v if i.hi is not None] for k, v in funcs.items()}


REG = re.compile(r"(?<![A-Z_\[])(U?R|U?P)(\d+|Z|T)(\.(64|128|reuse|H0|H1|B\d|X\d))*")


def width_of(opcode):
    mods = opcode.split(".")[1:]
    if "128" in mods:
        return 4
    if "64" in mods or "WIDE" in mods:
        return 2
    return 1


def regs_in(op):
    """Registers named in one operand: [(kind, n, width_hint)]."""
    out = []
    for m in REG.finditer(op):
        kind, n = m.group(1), m.group(2)
        if n in ("Z", "T"):
            continue
        w = 1
        if ".64" in m.group(0):
            w = 2
        elif ".128" in m.group(0):
            w = 4
        out.append((kind, int(n), w))
    return out


def files(paths):
    for p in paths:
        with open(p, errors="replace") as f:
            yield p, parse(f.read())


if __name__ == "__main__":
    for p, funcs in files(sys.argv[1:]):
        for fn, insts in funcs.items():
            print(fn)
            for i in insts:
                print(f"  {i.pc:05x} {i.ctrl!r:48} {i.guard or '':5} {i.opcode} {', '.join(i.operands)}")


def patch_listing(text, kernel, words):
    """Return the listing `text` with the instruction words of `kernel`
    replaced: words = {pc: (lo, hi)} (128-bit encoding halves). Only the hex
    comments change; the simulator re-decodes control bits from them, and
    register operands are rewritten in the mnemonic text as well when the
    caller passes `text_of = {pc: new_text}` (see sim_patched)."""
    out = []
    cur = None
    pending_hi = None
    for line in text.splitlines():
        m = FUNC.search(line)
        if m:
            cur = m.group(1)
        if cur == kernel:
            m = LINE.search(line)
            if m:
                pc = int(m.group(1), 16)
                if pc in words:
                    lo, hi = words[pc]
                    line = line.replace(m.group(3), f"0x{lo:016x}")
                    pending_hi = hi
                else:
                    pending_hi = None
                out.append(line)
                continue
            m = HI.match(line)
            if m and pending_hi is not None:
                line = line.replace(m.group(1), f"0x{pending_hi:016x}")
                pending_hi = None
        out.append(line)
    return "\n".join(out) + "\n"
