"""Chapter 1: GEMM time vs arithmetic intensity on the calibrated B300 and A100.

Sweeps M for an M x N x K GEMM (N = K = 8192 by default), the shape of one
weight matrix applied to M tokens. Small M is decode, large M is prefill or
training. Prints simulated time, achieved TFLOP/s and GB/s, the arithmetic
intensity of the ideal byte count, and the roofline bound for comparison.

    cargo build --release -p rsim-cli
    python docs/book/scripts/roofline_sweep.py                 # B300, BF16
    python docs/book/scripts/roofline_sweep.py --gpu a100      # A100 80GB PCIe, BF16
    python docs/book/scripts/roofline_sweep.py --dtype fp4     # B300, FP4 weights and activations
"""

import argparse
import concurrent.futures as cf
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RSIM = os.path.join(ROOT, "target", "release", "rsim")

# Measured: SM count x clock x tensor FLOP/clk/SM, and HBM read bandwidth.
# B300: tcgen05 8192 BF16 / 43691 FP4 (K=96) FLOP/clk/SM, 2032 MHz, 6.82 TB/s.
# A100 80GB PCIe: mma.sync 2029 BF16 FLOP/clk/SM, 1395 MHz, 1.70 TB/s.
GPUS = {
    "b300": dict(args=["--gpu", "b300"], sms=148, mhz=2032.4, hbm=6818e9,
                 flop_clk={"bf16": 8192, "fp8": 16384, "fp4": 43691}),
    "a100": dict(args=["--gpu", "a100_pcie", "--calibration",
                       os.path.join(ROOT, "calibration/results/a100-80gb-pcie-sm80/calibration.json")],
                 sms=108, mhz=1394.7, hbm=1698e9, flop_clk={"bf16": 2029}),
}
BYTES = {"bf16": 2, "fp8": 1, "fp4": 0.5}


def run(gpu, dtype, m, n, k):
    cmd = [RSIM, "kernel", *GPUS[gpu]["args"], "--dtype", dtype, "gemm", str(m), str(n), str(k)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    text = out.stdout + out.stderr
    us = float(re.search(r"time\s+([\d.]+) us", text).group(1))
    clk = re.search(r"sustained SM clock (\d+) MHz", text)
    return us, (float(clk.group(1)) if clk else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="b300", choices=GPUS)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--ms", default="1,16,64,128,256,512,1024,2048,4096,8192")
    a = ap.parse_args()
    g = GPUS[a.gpu]
    peak = g["sms"] * g["mhz"] * 1e6 * g["flop_clk"][a.dtype]
    ridge = peak / g["hbm"]
    ms = [int(x) for x in a.ms.split(",")]
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(lambda m: run(a.gpu, a.dtype, m, a.n, a.k), ms))
    print(f"{a.gpu} {a.dtype}: peak {peak / 1e12:.0f} TFLOP/s, HBM {g['hbm'] / 1e12:.2f} TB/s, "
          f"ridge {ridge:.0f} FLOP/byte")
    print(f"{'M':>6} {'AI':>7} {'roofline_us':>11} {'sim_us':>9} {'TFLOP/s':>8} {'%peak':>6} {'GB/s':>7} {'clock':>6}")
    eb = BYTES[a.dtype]
    for m, (us, clk) in zip(ms, res):
        flops = 2.0 * m * a.n * a.k
        nbytes = eb * (m * a.k + a.k * a.n) + 2 * m * a.n  # A, B in dtype; C written in BF16
        ai = flops / nbytes
        bound = max(flops / peak, nbytes / g["hbm"]) * 1e6
        print(f"{m:>6} {ai:>7.1f} {bound:>11.1f} {us:>9.1f} {flops / us / 1e6:>8.0f} "
              f"{100 * flops / us / 1e6 / (peak / 1e12):>6.1f} {nbytes / us / 1e3:>7.0f} {clk or '':>6}")


if __name__ == "__main__":
    main()
