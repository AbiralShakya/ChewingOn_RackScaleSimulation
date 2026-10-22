"""Fit register-file conflict features to the regfile.py trials.

Each trial is one 256-instruction block with random registers; its cycle
count is regressed (least squares, numpy) on per-trial sums of candidate
conflict features. Features with negligible coefficients are dropped by the
reader; the point is to find the few that explain the variance.

    python3 regfile_fit.py regfile.json
"""
import json
import sys

import numpy as np


def feats(regs, nsrc, lags=(1, 2, 3)):
    f = {}

    def add(k, v=1):
        f[k] = f.get(k, 0) + v

    for k, r in enumerate(regs):
        rd, src, reuse = r[0], r[1:1 + nsrc], r[1 + nsrc]
        # Operands actually read from the register file this instruction.
        read = []
        for j, s in enumerate(src):
            if k > 0:
                prev = regs[k - 1]
                if prev[1 + nsrc] >> j & 1 and prev[1 + j] == s:
                    continue  # reuse-cache hit: same slot, same register
            read.append(s)
        for i in range(len(read)):
            for j in range(i + 1, len(read)):
                if read[i] == read[j]:
                    add("intra_same_reg")
                for B in (2, 4, 8):
                    if read[i] % B == read[j] % B and read[i] != read[j]:
                        add(f"intra_bank{B}")
        for L in lags:
            if k - L < 0:
                continue
            p = regs[k - L]
            psrc = p[1:1 + nsrc]
            for s in read:
                if s in psrc:
                    add(f"lag{L}_same_reg")
                for B in (2, 4, 8):
                    if any(s % B == q % B and s != q for q in psrc):
                        add(f"lag{L}_bank{B}")
            for B in (2, 4):
                if any(s % B == p[0] % B for s in read):
                    add(f"lag{L}_dst_bank{B}")
        add("reads", len(read))
    return f


def main(path):
    d = json.load(open(path))
    for fn, r in d.items():
        nsrc, n = r["nsrc"], r["n"]
        for modes, label in (((0, 1), "stall 1, no reuse"), ((0, 1, 2), "stall 1, with reuse"), ((3,), "stall 2")):
            rows = [t for t in r["trials"] if t["mode"] in modes]
            F = [feats(t["regs"], nsrc) for t in rows]
            keys = sorted({k for x in F for k in x})
            X = np.array([[x.get(k, 0) for k in keys] + [n] for x in F], float)
            y = np.array([t["cycles"] for t in rows], float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coef
            r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            print(f"{fn} [{label}] trials {len(rows)}  R^2 {r2:.3f}  rms {np.sqrt(((y - pred) ** 2).mean()):.1f} cyc  "
                  f"cycles/inst {y.min() / n:.2f}..{y.max() / n:.2f}")
            for k, c in sorted(zip(keys + ["base/inst"], coef), key=lambda x: -abs(x[1])):
                if abs(c) > 0.05:
                    print(f"     {k:20} {c:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
