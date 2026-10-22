"""Dump the mbarrier operations of selected warps of an .rpath recording.
    python3 dump_rpath.py file.rpath.gz CTA WARP [WARP ...]"""
import gzip
import json
import struct
import sys

KIND = {1: "wait", 2: "arrive", 3: "txdone", 4: "commit", 5: "named", 6: "waitpar"}
path, cta = sys.argv[1], int(sys.argv[2])
want = {int(x) for x in sys.argv[3:]}
d = gzip.open(path).read() if path.endswith(".gz") else open(path, "rb").read()
n = struct.unpack_from("<I", d, 8)[0]
hdr = json.loads(d[12:12 + n])
p = 12 + n
(nw,) = struct.unpack_from("<I", d, p)
p += 4
for _ in range(nw):
    c, w, nops = struct.unpack_from("<III", d, p)
    p += 12
    out = []
    for _ in range(nops):
        idx, active, payload = struct.unpack_from("<IIQ", d, p)
        p += 16
        kind, mid, arg, mn, ranks = struct.unpack_from("<BIIIH", d, p)
        p += 15
        run_bytes, naddr = struct.unpack_from("<IH", d, p)
        p += 6 + 8 * naddr
        if kind and c == cta and w in want:
            out.append((idx, KIND.get(kind, kind), hex(mid), arg, mn))
    if out:
        print(f"cta {c} warp {w}: {len(out)} mbar ops")
        for o in out[:80]:
            print("  ", o)
