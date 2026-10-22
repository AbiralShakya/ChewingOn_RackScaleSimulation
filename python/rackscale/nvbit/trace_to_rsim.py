#!/usr/bin/env python3
"""Convert an NVBit warp_path recording into the simulator's path format (.rpath).

  python -m rackscale.nvbit.trace_to_rsim <trace_dir> <out.rpath> [--primary 0] [--regs 128]

(`research/flash_attention/trace_to_rsim.py` runs the same thing without
installing the package.)

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
import re
import struct
import sys
from collections import Counter, defaultdict

try:
    from .wpread import WP_BAR, WP_BASIC, WP_MBAR, WP_MBARV, WP_MEM, WP_TMA, WP_UREG, WP_WAIT, read_listing, read_wp
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wpread import WP_BAR, WP_BASIC, WP_MBAR, WP_MBARV, WP_MEM, WP_TMA, WP_UREG, WP_WAIT, read_listing, read_wp  # noqa: E402,E501

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
    def __init__(self, trace_dir, primary=None, parity_waits=False, detect_completed_init=None, clock_waits=None):
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
        # Wait semantics: absolute phase counts (validated on FA4), or the hardware's
        # try_wait.parity rule, which needs no assumption about skipped phases.
        self.parity_waits = parity_waits
        # Barriers initialised in the completed state (cuBLAS) are detected from the clock;
        # off by default with phase counts, which the FlashAttention study validated without it.
        self.detect_completed_init = parity_waits if detect_completed_init is None else detect_completed_init
        # Same-parity repeat waits resolved from the recorded arrivals' clocks (else: two phases).
        self.clock_waits = self.detect_completed_init if clock_waits is None else clock_waits
        self.mma_shapes = Counter()
        self.mma_by_idx = {}  # instruction index -> (M, N, K) from its descriptor

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
            if r["kind"] in (WP_MBAR, WP_MBARV, WP_TMA, WP_UREG, WP_BAR, WP_MEM) or \
                    (r["kind"] == WP_WAIT and r["predon"]):
                return None
        return None

    def arrival_events(self, cta):
        """Barrier id -> sorted SM clocks of the arrivals that CTA `cta`'s barrier received from
        any recorded CTA of its cluster (arrives and tcgen05 commits, one entry per arrival).

        B300 SM clocks agree to within a few thousand cycles across SMs, so these can be
        compared with a wait's clock to tell how many phases completed between two waits.
        """
        if not hasattr(self, "_arrivals"):
            self._arrivals = {}
        if cta in self._arrivals:
            return self._arrivals[cta]
        out = defaultdict(list)
        base, rank = cta - self.rank(cta), self.rank(cta)
        for (c, _w), recs in self.warps.items():
            if c - self.rank(c) != base:
                continue  # another cluster
            for r in recs:
                if r["kind"] not in (WP_MBAR, WP_MBARV) or not r["predon"]:
                    continue
                op = self.L[r["idx"]]["op"]
                if op.startswith("SYNCS.ARRIVE") and ".A0T" not in op:
                    if r["kind"] == WP_MBARV:
                        lanes = [(r["addrs"][l] >> 24) & 0xF for l in range(32) if (r["predon"] >> l) & 1]
                    else:
                        lanes = [(r["a"] >> 24) & 0xF] * popcount(r["predon"])
                    hits = sum(1 for lr in lanes if (lr if self.cluster > 1 else rank) == rank)
                    if hits:
                        out[r["a"] & 0xFFFFFF].extend([r["clk"]] * hits)
                elif op.startswith("UTCBAR"):
                    if "MULTICAST" in op and r["c"] != 0xFFFFFFFF:
                        mine = (r["c"] >> rank) & 1
                    else:
                        mine = c == cta
                    if mine:
                        out[r["a"] & 0xFFFFFF].append(r["clk"])
        for v in out.values():
            v.sort()
        self._arrivals[cta] = out
        return out

    def phases_between(self, cta, bid, t0, t1):
        """Phases of barrier `bid` of CTA `cta` completed between clocks t0 and t1, from its
        recorded arrivals; None when the barrier has no recorded arrivals (TMA-only)."""
        ev = self.arrival_events(cta).get(bid)
        if not ev:
            return None
        import bisect
        n = bisect.bisect_right(ev, t1) - bisect.bisect_right(ev, t0)
        return n / max(1, self.barrier_counts_cached().get(bid, 1))

    def barrier_counts_cached(self):
        if not hasattr(self, "_counts"):
            self._counts = self.barrier_counts()
        return self._counts

    def first_signal(self, cta):
        """Barrier id -> SM clock of the first op in CTA `cta` that can complete one of its phases."""
        if not hasattr(self, "_first_signal"):
            self._first_signal = {}
        if cta not in self._first_signal:
            out = {}
            for (c, _w), recs in self.warps.items():
                if c != cta:
                    continue
                for r in recs:
                    op = self.L[r["idx"]]["op"]
                    if r["kind"] in (WP_MBAR, WP_MBARV) and not op.startswith("SYNCS.ARRIVE"):
                        continue  # mbarrier.init (SYNCS.EXCH) and the like complete nothing
                    if r["kind"] in (WP_MBAR, WP_MBARV, WP_TMA) and r["predon"]:
                        if r["kind"] == WP_TMA:
                            bid = (struct.unpack("<16Q", r["handle"])[0] >> 32) & 0xFFFFFF
                        else:
                            bid = r["a"] & 0xFFFFFF
                        out[bid] = min(out.get(bid, r["clk"]), r["clk"])
                    elif r["kind"] == WP_UREG and r["predon"] and self.L[r["idx"]]["op"].startswith("UTCBAR"):
                        bid = r["a"] & 0xFFFFFF
                        out[bid] = min(out.get(bid, r["clk"]), r["clk"])
            self._first_signal[cta] = out
        return self._first_signal[cta]

    def completed_init(self, cta):
        """Barriers of CTA `cta` that the kernel initialised in the completed state.

        The earliest wait on such a barrier (across the CTA's warps) passed on its
        first try, before anything in the CTA could have completed a phase of it.
        cuBLAS's Blackwell GEMMs initialise their pipeline barriers this way.
        """
        if not self.detect_completed_init:
            return set()
        if not hasattr(self, "_completed_init"):
            self._completed_init = {}
        if cta not in self._completed_init:
            first_wait = {}  # bid -> (clk, passed at first try)
            for (c, _w), recs in self.warps.items():
                if c != cta:
                    continue
                tried = set()
                for r in recs:
                    if r["kind"] != WP_WAIT or not r["predon"]:
                        continue
                    bid = r["a"] & 0xFFFFFF
                    ok = (r["b"] >> r["aux"]) & 1
                    if bid not in tried:
                        tried.add(bid)
                        if bid not in first_wait or r["clk"] < first_wait[bid][0]:
                            first_wait[bid] = (r["clk"], bool(ok))
            sig = self.first_signal(cta)
            self._completed_init[cta] = {b for b, (clk, ok) in first_wait.items()
                                          if ok and (b not in sig or clk < sig[b])}
            self.stats["completed_init_barriers"] += len(self._completed_init[cta])
        return self._completed_init[cta]

    def convert_warp(self, cta, recs):
        """Records of one warp -> list of op tuples. Spin loops on mbarriers (a failed wait, the
        branch and sleep around it, the failed retries) are removed; the simulator times the wait."""
        ops = []
        need_last = {}  # barrier id -> (last needed phase count, its parity, its clock)
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
            if kind == WP_UREG and op.startswith("UTMA"):
                continue  # the CTA mask of the multicast copy that follows, not an instruction
            mk, mid, marg, mn, mranks = K_NONE, 0, 0, 0, 0
            payload, run_bytes, addrs = 0, 0, []
            active = r["active"]
            if kind == WP_WAIT and r["predon"] and (r["b"] >> r["aux"]) & 1:
                bid = r["a"] & 0xFFFFFF
                # 0xFFFFFFFF is also the tool's "no register operand" value, but a phase register
                # can hold it (~0 after a flip): it is real whenever the instruction names one.
                has_reg = re.search(r"\]\s*,\s*R\d+", self.L[r["idx"]]["text"]) is not None
                parity = (r["c"] >> 31) & 1 if (r["c"] != 0xFFFFFFFF or has_reg) else 0
                # The phase this wait needs, as an absolute count. Successive waits by one
                # warp alternate parity; an unchanged parity means the warp skipped a phase
                # (FA4's lazy rescaling skips some), so the count advances by two. Checked
                # against the hardware clock: every repeat had exactly two commits between.
                if self.parity_waits:
                    # try_wait.parity(p) passes once the barrier's current phase has parity != p.
                    # A barrier the kernel initialised in the completed state starts one phase
                    # ahead of the simulator's count, which flips the parity of every wait on it.
                    if bid in self.completed_init(cta):
                        parity ^= 1
                    mk, mid, marg = K_WAIT_PARITY, bid, parity
                else:
                    if bid not in need_last:
                        need = 0 if parity == 1 else 1
                        # A barrier initialised in the completed state is one phase ahead of the
                        # simulator's count: the phase a wait passes on is one fewer.
                        # (hardware count: the first C >= 1 whose parity differs from the wait's).
                        if bid in self.completed_init(cta):
                            need = 0 if parity == 0 else 1
                    else:
                        last, last_p, last_clk = need_last[bid]
                        if parity != last_p:
                            need = last + 1
                        else:
                            # The same parity again: either no phase completed since the last wait
                            # (it passes at once) or two did (FA4's lazy rescaling skips one). The
                            # recorded arrivals between the two waits decide; without any, two.
                            # (Off for plain phase counts, which stay as the FlashAttention study
                            # validated them.)
                            between = self.phases_between(cta, bid, last_clk, r["clk"]) \
                                if self.clock_waits else None
                            need = last + (0 if between is not None and between < 1 else 2)
                            self.stats["same_parity_waits_" + ("0" if need == last else "2")] += 1
                    need_last[bid] = (need, parity, r["clk"])
                    mk, mid, marg = K_WAIT, bid, need
                self.stats["waits"] += 1
            elif kind in (WP_MBAR, WP_MBARV) and r["predon"]:
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
                    ranks = self.ranks_of(arank)
                    if kind == WP_MBARV:
                        # Per-lane targets: lane i usually arrives on CTA i of the cluster.
                        # One op signals every targeted rank, with its share of the lanes.
                        lane_ranks = {(r["addrs"][lane] >> 24) & 0xF for lane in range(32) if (r["predon"] >> lane) & 1}
                        ranks = 0
                        for rk in lane_ranks:
                            ranks |= self.ranks_of(rk)
                        n = max(1, n // len(lane_ranks))
                        self.stats["multi_rank_arrives"] += len(lane_ranks) > 1
                    if ".A0TX" in op and tx == 0:
                        # Expect-tx only, of a few bytes the simulator does not track: nothing to
                        # do (an Arrive op always counts at least one arrival).
                        self.stats["expect_tx_only_dropped"] += 1
                    elif ".A0T1" in op:
                        # complete-tx (mbarrier.complete_tx): the bytes land without an arrival
                        nbytes = 0 if r["b"] == 0xFFFFFFFF else r["b"]
                        mk, mid, marg, mranks = K_TXDONE, bid, nbytes, ranks
                        self.stats["complete_tx"] += 1
                    else:
                        if ".A0TX" in op:
                            n = 0  # expect-tx only: decoded as one arrival, an approximation
                            self.stats["expect_tx_only_as_arrive"] += 1
                        mk, mid, marg, mn, mranks = K_ARRIVE, bid, tx, n, ranks
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
                mask = None
                if "MULTICAST" in op and n > 0 and recs[n - 1]["kind"] == WP_UREG and recs[n - 1]["idx"] == r["idx"]:
                    # Destination CTAs (cluster ranks): the high half of the register on sm_100/103.
                    v = recs[n - 1]["a"]
                    mask = (v >> 16) & 0xFFFF or v & 0xFFFF
                bytes_ = r["a"]
                runs = r["runs"]
                payload = bytes_
                if runs:
                    lens = Counter(n for _a, n in runs)
                    run_bytes = lens.most_common(1)[0][0]
                    addrs = [a for a, _n in runs]
                if not op.startswith("UTMASTG"):
                    mranks = self.ranks_of(arank)
                    if mask is not None and self.cluster > 1:
                        # The box lands in every destination CTA and completes its bytes on that
                        # CTA's barrier (on the pair leader's for a 2-CTA copy).
                        mranks = 0
                        for rk in range(16):
                            if (mask >> rk) & 1:
                                mranks |= 1 << ((rk & ~1) if ".2CTA" in op else rk)
                        self.stats["tma_multicast"] += 1
                    mk, mid, marg, mranks = K_TXDONE, baddr & 0xFFFFFF, bytes_, mranks
                self.stats["tma"] += 1
            elif kind == WP_UREG and "MMA" in op and r["predon"]:
                fl, shape = umma_flops(op, r["a"])
                payload = fl
                self.mma_shapes[(op, shape)] += 1
                self.mma_by_idx.setdefault(r["idx"], shape)
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
        # Cluster launch control: the primary CTA's tiles = its UGETNEXTWORKID queries (the last
        # one finds no work, the CTA's own tile makes up for it). Only ~grid/tiles CTAs run.
        clc = sum(1 for (c, _w), recs in self.warps.items() if c == self.primary
                  for r in recs if r["predon"] and self.L[r["idx"]]["op"].startswith("UGETNEXTWORKID"))
        launch_ctas = None
        g = self.meta["grid"]
        if clc > 1:
            launch_ctas = -(-(g[0] * g[1] * g[2]) // clc)
            launch_ctas = -(-launch_ctas // self.cluster) * self.cluster
            self.stats["clc_tiles_per_cta"] = clc
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
        if launch_ctas is not None:
            header["launch_ctas"] = launch_ctas
        if self.mma_by_idx:
            header["mma"] = {str(k): list(v) for k, v in self.mma_by_idx.items()}
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


def convert(trace_dir, out, primary=None, regs=None, flops=0.0, nbytes=0.0, parity_waits=False,
            detect_completed_init=None, clock_waits=None):
    """Write `out` (.rpath) from the recording in `trace_dir`; return a summary dict."""
    c = Conv(trace_dir, primary, parity_waits, detect_completed_init, clock_waits)
    paths, header = c.write(out, regs, flops, nbytes)
    lens = {f"{k[0]}:{k[1]}": len(v) for k, v in paths.items() if k[0] == c.primary}
    return {"out": out, "primary": c.primary, "ctas": c.ctas, "cluster": c.cluster,
            "barriers": len(header["barriers"]), "strided_insts": len(header["strides"]),
            "stats": dict(c.stats), "mma_shapes": {f"{k[0]} {k[1]}": v for k, v in c.mma_shapes.items()},
            "ops_per_warp_primary": lens, "tma_all_ctas": len(set(k[0] for k in c.tma_all))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("out")
    ap.add_argument("--primary", type=int)
    ap.add_argument("--regs", type=int, help="registers/thread (NVBit reports 0 for JIT CuTe kernels)")
    ap.add_argument("--flops", type=float, default=0.0)
    ap.add_argument("--bytes", type=float, default=0.0)
    ap.add_argument("--parity-waits", action="store_true", help="hardware try_wait.parity semantics for waits")
    a = ap.parse_args()
    print(json.dumps(convert(a.trace_dir, a.out, a.primary, a.regs, a.flops, a.bytes, a.parity_waits), indent=1))


if __name__ == "__main__":
    main()
