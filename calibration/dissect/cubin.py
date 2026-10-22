"""Read and patch instructions in a CUDA cubin (ELF64, Volta+ 128-bit encoding).

    c = Cubin(open("k.cubin", "rb").read())
    c.set_ctrl("_Z3fooPf", 0x120, stall=4)
    c.set_bits("_Z3fooPf", 0x120, 24, 8, 30)   # Ra := R30
    open("k2.cubin", "wb").write(c.data)

Instruction bit i is bit i of the little-endian 128-bit word at the PC's offset
in `.text.<kernel>`. Field positions used here (verified by re-disassembling
patched cubins, see check_fields.py):

    Rd [16:24)  Ra [24:32)  Rb [32:40)  Rc [64:72)
    stall [105:109)  yield [109]  wbar [110:113)  rbar [113:116)
    wait [116:122)  reuse [122:126)
"""
import struct

CTRL_FIELDS = {
    "stall": (105, 4),
    "yld": (109, 1),
    "wbar": (110, 3),
    "rbar": (113, 3),
    "wait": (116, 6),
    "reuse": (122, 4),
}
REG_FIELDS = {"rd": (16, 8), "ra": (24, 8), "rb": (32, 8), "rc": (64, 8)}


class Cubin:
    def __init__(self, data):
        self.data = bytearray(data)
        d = self.data
        assert d[:4] == b"\x7fELF" and d[4] == 2, "not an ELF64 cubin"
        (self.shoff,) = struct.unpack_from("<Q", d, 0x28)
        self.shentsize, self.shnum, self.shstrndx = struct.unpack_from("<HHH", d, 0x3A)
        self.sections = []
        for i in range(self.shnum):
            off = self.shoff + i * self.shentsize
            name, typ, flags, addr, offset, size, link, info = struct.unpack_from(
                "<IIQQQQII", d, off)
            self.sections.append(dict(idx=i, hdr=off, name_off=name, type=typ,
                                      offset=offset, size=size, info=info))
        strtab = self.sections[self.shstrndx]
        for s in self.sections:
            a = strtab["offset"] + s["name_off"]
            b = d.index(b"\0", a)
            s["name"] = d[a:b].decode()
        self.by_name = {s["name"]: s for s in self.sections}

    def text(self, kernel):
        return self.by_name[".text." + kernel]

    def kernels(self):
        return [s["name"][6:] for s in self.sections if s["name"].startswith(".text.")]

    def word(self, kernel, pc):
        s = self.text(kernel)
        assert pc % 16 == 0 and pc < s["size"]
        lo, hi = struct.unpack_from("<QQ", self.data, s["offset"] + pc)
        return lo | (hi << 64)

    def put(self, kernel, pc, w):
        s = self.text(kernel)
        struct.pack_into("<QQ", self.data, s["offset"] + pc, w & (2**64 - 1), w >> 64)

    def get_bits(self, kernel, pc, lo, n):
        return (self.word(kernel, pc) >> lo) & ((1 << n) - 1)

    def set_bits(self, kernel, pc, lo, n, v):
        w = self.word(kernel, pc)
        m = ((1 << n) - 1) << lo
        self.put(kernel, pc, (w & ~m) | ((v << lo) & m))

    def set_ctrl(self, kernel, pc, **kw):
        for k, v in kw.items():
            self.set_bits(kernel, pc, *CTRL_FIELDS[k], v)

    def get_ctrl(self, kernel, pc):
        return {k: self.get_bits(kernel, pc, *f) for k, f in CTRL_FIELDS.items()}

    def set_reg(self, kernel, pc, slot, r):
        self.set_bits(kernel, pc, *REG_FIELDS[slot], r)

    def regcount(self, kernel):
        return self.text(kernel)["info"] >> 24

    def symbol_index(self, name):
        st = self.by_name[".symtab"]
        strtab = self.sections[struct.unpack_from("<I", self.data, st["hdr"] + 40)[0]]
        for k in range(st["size"] // 24):
            off = st["offset"] + 24 * k
            (nm,) = struct.unpack_from("<I", self.data, off)
            a = strtab["offset"] + nm
            if self.data[a:self.data.index(b"\0", a)].decode() == name:
                return k
        raise KeyError(name)

    def set_regcount(self, kernel, n):
        """Register count lives in two places: the top byte of the text
        section's sh_info and EIATTR_REGCOUNT (0x2f) in .nv.info."""
        s = self.text(kernel)
        info = (s["info"] & 0x00FFFFFF) | (n << 24)
        struct.pack_into("<I", self.data, s["hdr"] + 44, info)
        s["info"] = info
        sym = self.symbol_index(kernel)
        for name in (".nv.info", ".nv.info." + kernel):
            sec = self.by_name.get(name)
            if not sec:
                continue
            p, end = sec["offset"], sec["offset"] + sec["size"]
            while p < end:
                fmt, attr, size = self.data[p], self.data[p + 1], struct.unpack_from("<H", self.data, p + 2)[0]
                if fmt != 0x04:  # only EIFMT_SVAL carries a size field
                    p += 4
                    continue
                if attr == 0x2F and size == 8:
                    fsym, _ = struct.unpack_from("<II", self.data, p + 4)
                    if fsym == sym:
                        struct.pack_into("<I", self.data, p + 8, n)
                p += 4 + size
