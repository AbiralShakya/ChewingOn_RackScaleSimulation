"""Rewrite an .rpath so every recorded mbarrier wait uses try_wait.parity
semantics (kind 6, arg = parity) instead of an absolute phase count (kind 1).

trace_to_rsim.py turns each successful wait into the phase count it needs,
assuming a warp never skips more than one phase between two waits on one
barrier. FA4's epilogue warp breaks that (it waits on an O-stage barrier only
every few tiles), so it ran ahead in the replay. The parity is recovered
from the count: parity = (need + 1) % 2.

    python3 rpath_parity.py in.rpath.gz out.rpath.gz [warp,warp...]
"""
import gzip
import struct
import sys

src, dst = sys.argv[1:3]
only = {int(x) for x in sys.argv[3].split(',')} if len(sys.argv) > 3 else None
d = bytearray(gzip.open(src).read() if src.endswith(".gz") else open(src, "rb").read())
n = struct.unpack_from("<I", d, 8)[0]
p = 12 + n
(nw,) = struct.unpack_from("<I", d, p)
p += 4
changed = 0
for _ in range(nw):
    c, w, nops = struct.unpack_from("<III", d, p)
    p += 12
    for _ in range(nops):
        p += 16
        kind, mid, arg = struct.unpack_from("<BII", d, p)
        if kind == 1 and (only is None or w in only):
            struct.pack_into("<BII", d, p, 6, mid, (arg + 1) % 2)
            changed += 1
        p += 15
        run_bytes, naddr = struct.unpack_from("<IH", d, p)
        p += 6 + 8 * naddr
# Trailing per-CTA TMA source table (optional) is copied unchanged.
(gzip.open if dst.endswith(".gz") else open)(dst, "wb").write(bytes(d))
print(f"{changed} waits rewritten")
