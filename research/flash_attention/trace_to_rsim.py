#!/usr/bin/env python3
"""Convert an NVBit warp_path trace into the simulator's path format (.rpath).

  python trace_to_rsim.py <trace_dir> <out.rpath> [--primary 0] [--regs 128]

What the conversion does (see crates/rsim-sass/src/path.rs for the format):

* Failed mbarrier waits are removed together with the spin loop that follows
  them (the branch, sleep and retries up to the next wait on the same barrier),
  so the simulator re-times the producer/consumer pipeline instead of replaying
  the spins the hardware happened to do. A failed wait followed by unrelated
  work is a non-blocking poll and is kept as a plain instruction. Each
  successful wait keeps its phase parity; the simulator applies try_wait.parity
  semantics through an absolute phase count derived from the parity sequence.
* mbarrier.init (SYNCS.EXCH.64) values give each barrier's arrival count.
  A warp-wide SYNCS.ARRIVE contributes one arrival per active thread ; the expect-tx form adds its transaction bytes.
* TMA copies complete their byte count on the barrier named in the copy's
  parameters; `.2CTA` copies complete on the leader CTA of the pair.
* tcgen05.commit (UTCBAR) arrives when the CTA's tensor work drains; the
  multicast form arrives on every CTA in its mask.
* UMMA FLOPs come from the instruction descriptor (M, N) and the operand type (K).
* CTAs that were not recorded replay the primary CTA's path. Their memory
  addresses are shifted by per-instruction strides inferred from recorded CTAs
  that differ in one grid coordinate, and, when the trace recorded every CTA's
  TMA copies (WP_TMA_ALL=1), their TMA sources are the real ones.
"""
import argparse
import glob
import json
import os
import struct
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpread import WP_BAR, WP_BASIC, WP_MBAR, WP_MEM, WP_TMA, WP_UREG, WP_WAIT, read_listing, read_wp  # noqa: E402

K_NONE, K_WAIT, K_ARRIVE, K_TXDONE, K_COMMIT, K_NAMED, K_WAIT_PARITY = 0, 1, 2, 3, 4, 5, 6


def popcount(x):
    return bin(x & 0xFFFFFFFF).count("1")


def umma_flops(op, idesc):
    """2*M*N*K from a tcgen05 instruction descriptor (PTX ISA, instruction descriptor table)."""
    n = ((idesc >> 17) & 0x3F) << 3
    m = ((idesc >> 24) & 0x1F) << 4
    if "QMMA" in op or "F8" in op:
        k = 32
    elif "OMMA" in op or "F4" in op:
        k = 64
    elif "TF32" in op:
        k = 8
    else:
        k = 16  # f16 / bf16
    return 2 * m * n * k, (m, n, k)


class Conv:
    def __init__(self, trace_dir, primary=None):
        self.dir = trace_dir
        base = glob.glob(os.path.join(trace_dir, "*.json"))[0][:-5]
        self.meta = json.load(open(base + ".json"))
        self.sass_text = open(base + ".sass").read()
        self.L = read_listing(base + ".sass")
        self.cluster = max(1, self.meta["cluster"][0] * self.meta["cluster"][1] * self.meta["cluster"][2])
        self.warps = defaultdict(list)
        self.tma_all = defaultdict(list)  # (cta, warp) -> [runs] for every CTA (address-only)
        sampled = set(c for c in self.meta["ctas"] if c != 0xFFFFFFFF)
        for r in read_wp(base + ".wp"):
            if r["cta"] in sampled:
                self.warps[(r["cta"], r["warp"])].append(r)
            if r["kind"] == WP_TMA:
                self.tma_all[(r["cta"], r["warp"])].append(r)
        self.ctas = sorted(set(k[0] for k in self.warps))
        self.primary = primary if primary is not None else self.ctas[0]
        self.stats = Counter()
        self.mma_shapes = Counter()

    def rank(self, cta):
        return cta % self.cluster

    def ranks_of(self, addr_rank):
        return 0 if self.cluster == 1 else (1 << addr_rank)

    def barrier_counts(self):
        counts = {}
        for (cta, _w), recs in self.warps.items():
            if cta != self.primary:
                continue
            for r in recs:
                if r["kind"] == WP_MBAR and self.L[r["idx"]]["op"].startswith("SYNCS.EXCH") and r["predon"]:
                    low = r["c"] & 0x1FFFFF
                    counts[r["a"] & 0xFFFFFF] = max(1, (0x200000 - low) >> 1)
        return counts

    SPIN_OPS = ("BRA", "NANOSLEEP", "YIELD", "WARPSYNC")

    def spin_window(self, recs, i):
        """recs[i] is a failed wait on barrier B. If the warp next waits on B again before doing
        anything that interacts with other warps or memory, it is spinning: return the index of
        that next wait. Otherwise (a non-blocking poll) return None."""
        addr = recs[i]["a"]
        for j in range(i + 1, min(len(recs), i + 4096)):
            r = recs[j]
            if r["kind"] == WP_WAIT and r["a"] == addr and r["predon"]:
                return j
            if r["kind"] in (WP_MBAR, WP_TMA, WP_UREG, WP_BAR, WP_MEM) or (r["kind"] == WP_WAIT and r["predon"]):
                return None
        return None

    def convert_warp(self, cta, recs):
        """Records of one warp -> list of op tuples. Spin loops on mbarriers (a failed wait, the
        branch and sleep around it, the failed retries) are removed; the simulator times the wait."""
        ops = []
        need_last = {}  # barrier id -> last needed phase count
        own = self.rank(cta)
        drop = set()
        i = 0
        while i < len(recs):
            r = recs[i]
            if r["kind"] == WP_WAIT and r["predon"] and not (r["b"] >> r["aux"]) & 1:
                j = self.spin_window(recs, i)
                if j is not None:
                    self.stats["failed_waits"] += 1
                    drop.add(i)
                    for k in range(i + 1, j):
                        if self.L[recs[k]["idx"]]["op"].split(".")[0] in self.SPIN_OPS:
                            drop.add(k)
                    i = j
                    continue
                self.stats["polls_kept"] += 1
            i += 1
        for n, r in enumerate(recs):
            if n in drop:
                self.stats["spin_ops_dropped"] += 1
                continue
            op = self.L[r["idx"]]["op"]
            kind = r["kind"]
            mk, mid, marg, mn, mranks = K_NONE, 0, 0, 0, 0
            payload, run_bytes, addrs = 0, 0, []
            active = r["active"]
            if kind == WP_WAIT and r["predon"] and (r["b"] >> r["aux"]) & 1:
                bid = r["a"] & 0xFFFFFF
                parity = (r["c"] >> 31) & 1 if r["c"] != 0xFFFFFFFF else 0
                # The phase this wait needs, as an absolute count. Successive waits by one
                # warp alternate parity; an unchanged parity means the warp skipped a phase
                # (FA4's lazy rescaling skips some), so the count advances by two. Checked
                # against the hardware clock: every repeat had exactly two commits between.
                if bid not in need_last:
                    need = 0 if parity == 1 else 1
                else:
                    last, last_p = need_last[bid]
                    need = last + (1 if parity != last_p else 2)
                need_last[bid] = (need, parity)
                mk, mid, marg = K_WAIT, bid, need
                self.stats["waits"] += 1
            elif kind == WP_MBAR and r["predon"]:
                bid, arank = r["a"] & 0xFFFFFF, (r["a"] >> 24) & 0xF
                if op.startswith("SYNCS.ARRIVE"):
                    tx = 0 if ".A1T0" in op or r["b"] == 0xFFFFFFFF else r["b"]
                    if tx <= 64:
                        # A few bytes expected from an st.async to the peer CTA (STAS), which
                        # lands with the same DSMEM latency as the remote arrive itself.
                        tx = 0
                    # One arrival per active thread (an elected single-thread arrive counts
                    # once); the barrier init counts confirm this for every FA4 barrier.
                    n = popcount(r["predon"])
                    mk, mid, marg, mn, mranks = K_ARRIVE, bid, tx, n, self.ranks_of(arank)
                    self.stats["arrives"] += 1
                elif op.startswith("UTCBAR"):
                    if "MULTICAST" in op and r["c"] != 0xFFFFFFFF:
                        mranks = r["c"] & 0xFFFF
                    else:
                        mranks = self.ranks_of(arank)
                    mk, mid = K_COMMIT, bid
                    self.stats["commits"] += 1
            elif kind == WP_TMA and r["predon"]:
                handle = struct.unpack("<16Q", r["handle"])
                baddr = (handle[0] >> 32) & 0xFFFFFFFF
                arank = (baddr >> 24) & 0xF if ".2CTA" in op else own
                bytes_ = r["a"]
                runs = r["runs"]
                payload = bytes_
                if runs:
                    lens = Counter(n for _a, n in runs)
                    run_bytes = lens.most_common(1)[0][0]
                    addrs = [a for a, _n in runs]
                if not op.startswith("UTMASTG"):
                    mk, mid, marg, mranks = K_TXDONE, baddr & 0xFFFFFF, bytes_, self.ranks_of(arank)
                self.stats["tma"] += 1
            elif kind == WP_UREG and "MMA" in op and r["predon"]:
                fl, shape = umma_flops(op, r["a"])
                payload = fl
                self.mma_shapes[(op, shape)] += 1
            elif kind == WP_BAR:
                mk, mid = K_NAMED, r["b"] & 0xF
            elif kind == WP_MEM and "MMA" in op:
                pass  # tcgen05 MMA recorded as a memory op (descriptor operands); FLOPs from the opcode
            elif kind == WP_MEM:
                active = r["predon"]
                addrs = [r["addrs"][l] for l in range(32) if (active >> l) & 1]
            ops.append((r["idx"], active, payload, mk, mid, marg, mn, mranks, run_bytes, addrs))
        return ops

    def strides(self, paths):
        """Per instruction index: byte stride per unit of blockIdx.{x,y,z}."""
        g = self.meta["grid"]

        def coords(c):
            return (c % g[0], (c // g[0]) % g[1], c // (g[0] * g[1]))

        pc = coords(self.primary)
        cand = defaultdict(lambda: [Counter(), Counter(), Counter()])
        for c in self.ctas:
            if c == self.primary:
                continue
            cc = coords(c)
            diff = [i for i in range(3) if cc[i] != pc[i]]
            if len(diff) != 1:
                continue
            d = diff[0]
            delta = cc[d] - pc[d]
            for (cta, w), ops in paths.items():
                if cta != self.primary or (c, w) not in paths:
                    continue
                other = paths[(c, w)]
                occ_p, occ_o = defaultdict(list), defaultdict(list)
                for o in ops:
                    if o[9]:
                        occ_p[o[0]].append(o[9][0])
                for o in other:
                    if o[9]:
                        occ_o[o[0]].append(o[9][0])
                for idx, la in occ_p.items():
                    lb = occ_o.get(idx, [])
                    for a, b in zip(la, lb):
                        cand[idx][d][(b - a) // delta] += 1
        out = {}
        for idx, per_d in cand.items():
            s = [0, 0, 0]
            for d in range(3):
                if per_d[d]:
                    v, n = per_d[d].most_common(1)[0]
                    if n >= 0.5 * sum(per_d[d].values()):
                        s[d] = v
            if any(s):
                out[str(idx)] = s
        return out

    def write(self, out, regs=None, flops=0.0, nbytes=0.0):
        paths = {}
        for (cta, w), recs in sorted(self.warps.items()):
            paths[(cta, w)] = self.convert_warp(cta, recs)
        smem = self.meta["smem_static"] + self.meta["smem_dynamic"]
        header = {
            "kernel": self.meta["kernel"],
            "grid": self.meta["grid"],
            "block": self.meta["block"],
            "cluster": self.meta["cluster"],
            "regs": regs or self.meta.get("regs") or 0,
            "smem": smem,
            "primary": self.primary,
            "barriers": {str(k): v for k, v in self.barrier_counts().items()},
            "strides": self.strides(paths),
            "sass": self.sass_text,
            "flops": flops,
            "bytes": nbytes,
        }
        hb = json.dumps(header).encode()
        with open(out, "wb") as f:
            f.write(b"RSIMPTH1")
            f.write(struct.pack("<I", len(hb)))
            f.write(hb)
            f.write(struct.pack("<I", len(paths)))
            for (cta, w), ops in paths.items():
                f.write(struct.pack("<III", cta, w, len(ops)))
                for idx, active, payload, mk, mid, marg, mn, mranks, rb, addrs in ops:
                    f.write(struct.pack("<IIQBIIIHIH", idx, active & 0xFFFFFFFF, payload, mk, mid, marg & 0xFFFFFFFF,
                                        mn, mranks, rb, len(addrs)))
                    if addrs:
                        f.write(struct.pack(f"<{len(addrs)}Q", *addrs))
            # Real TMA sources of every CTA (address-only), when recorded.
            tma = {k: v for k, v in self.tma_all.items() if k[0] not in self.ctas}
            f.write(struct.pack("<I", len(tma)))
            for (cta, w), recs in sorted(tma.items()):
                f.write(struct.pack("<III", cta, w, len(recs)))
                for r in recs:
                    addrs = [a for a, _n in r["runs"]]
                    f.write(struct.pack("<H", len(addrs)))
                    if addrs:
                        f.write(struct.pack(f"<{len(addrs)}Q", *addrs))
        return paths, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("out")
    ap.add_argument("--primary", type=int)
    ap.add_argument("--regs", type=int, help="registers/thread (NVBit reports 0 for JIT CuTe kernels)")
    ap.add_argument("--flops", type=float, default=0.0)
    ap.add_argument("--bytes", type=float, default=0.0)
    a = ap.parse_args()
    c = Conv(a.trace_dir, a.primary)
    paths, header = c.write(a.out, a.regs, a.flops, a.bytes)
    lens = {f"{k[0]}:{k[1]}": len(v) for k, v in paths.items() if k[0] == c.primary}
    summary = {"out": a.out, "primary": c.primary, "ctas": c.ctas, "cluster": c.cluster,
               "barriers": len(header["barriers"]), "strided_insts": len(header["strides"]),
               "stats": dict(c.stats), "mma_shapes": {f"{k[0]} {k[1]}": v for k, v in c.mma_shapes.items()},
               "ops_per_warp_primary": lens, "tma_all_ctas": len(set(k[0] for k in c.tma_all))}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
