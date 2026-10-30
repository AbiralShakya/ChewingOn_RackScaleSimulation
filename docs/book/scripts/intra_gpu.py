"""Chapter 15: parallelism inside one GPU, rung by rung, on the calibrated B300.

Each experiment changes one tiling choice of the Blackwell tcgen05 GEMM and
holds the rest at the defaults `rsim kernel gemm` picks on its own:

    depth     shared-memory pipeline stages (bytes in flight per SM)
    waves     M around the point where the tiles stop fitting in one wave
    tile      output tile width N (operand reuse per byte staged)
    cluster   B multicast across a 2-CTA cluster, or not
    splitk    a decode-shaped GEMM with its K loop split across CTAs
    reserve   SMs held back for communication kernels

Every row prints the time at the full 2032 MHz clock (what the SMs do) and
at the power-capped clock (what the board allows), so the two effects stay
apart.

    cargo build --release -p rsim-cli
    python docs/book/scripts/intra_gpu.py              # every experiment
    python docs/book/scripts/intra_gpu.py waves depth  # some of them
"""

import concurrent.futures as cf
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RSIM = os.path.join(ROOT, "target", "release", "rsim")
SMS, MHZ = 148, 2032.4
PEAK = SMS * MHZ * 1e6 * 8192  # BF16 FLOP/s at the full clock


def run(m, n, k, *flags):
    cmd = [RSIM, "kernel", "--gpu", "b300", "gemm", str(m), str(n), str(k), *map(str, flags)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    t = out.stdout + out.stderr
    us = float(re.search(r"time\s+([\d.]+) us", t).group(1))
    x = re.search(r"\(x([\d.]+) on time\)", t)
    stretch = float(x.group(1)) if x else 1.0
    per_sm = int(re.search(r"(\d+) CTA/SM", t).group(1))
    ctas = int(re.search(r"(\d+) CTAs", t).group(1))
    rd, wr = map(float, re.search(r"dram\s+([\d.]+) MB read, ([\d.]+) MB written", t).groups())
    name = re.search(r"^(sm100_gemm\S+)", t, re.M).group(1)
    return dict(us_full=us / stretch, us_cap=us, ctas=ctas, per_sm=per_sm,
                dram_mb=rd + wr, name=name, flops=2.0 * m * n * k)


def table(title, rows, label):
    print(f"\n## {title}\n")
    print(f"| {label} | CTAs | CTA/SM | µs @ 2032 MHz | % of peak | µs power-capped | DRAM MB |")
    print("|---|---|---|---|---|---|---|")
    for lab, r in rows:
        eff = 100 * r["flops"] / (r["us_full"] * 1e-6) / PEAK
        print(f"| {lab} | {r['ctas']} | {r['per_sm']} | {r['us_full']:.1f} | {eff:.1f} | "
              f"{r['us_cap']:.1f} | {r['dram_mb']:.0f} |")


EXPERIMENTS = {
    "depth": ("Pipeline depth, 4096³", "stages",
              [(s, (4096, 4096, 4096, "--stages", s)) for s in (1, 2, 3, 4)]),
    "waves": ("Waves: M × 4096 × 4096, 128 × 256 tiles", "M (tiles)",
              [(f"{m} ({m // 128 * 16})", (m, 4096, 4096, "--tile-n", 256, "--stages", 2))
               for m in (2048, 2304, 2432, 2560, 4608, 4736)]),
    "tile": ("Tile width, 4096³", "tile N",
             [(tn, (4096, 4096, 4096, "--tile-n", tn)) for tn in (64, 128, 256)]),
    "cluster": ("Cluster multicast of B, 4096³", "cluster",
                [(c, (4096, 4096, 4096, "--cluster", c)) for c in (1, 2)]),
    "splitk": ("Split-K, decode shape 16 × 8192 × 8192", "split",
               [(s, (16, 8192, 8192, "--split-k", s)) for s in (1, 2, 4, 8, 16)]),
    "reserve": ("SMs held back for NCCL, 4096³", "SMs reserved",
                [(r, (4096, 4096, 4096, "--reserve-sms", r)) for r in (0, 4, 8, 16, 24, 32)]),
}


def main():
    names = sys.argv[1:] or list(EXPERIMENTS)
    jobs = [(e, lab, args) for e in names for lab, args in EXPERIMENTS[e][2]]
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(lambda j: run(*j[2]), jobs))
    for e in names:
        title, label, _ = EXPERIMENTS[e]
        table(title, [(lab, r) for (ee, lab, _), r in zip(jobs, res) if ee == e], label)


if __name__ == "__main__":
    main()
