"""Build the example viewer pages in docs/viz/ (see docs/VIZ.md).

    cargo build --release -p rsim-cli
    python examples/viz_gallery.py                 # all pages
    python examples/viz_gallery.py deepseek        # one page: deepseek | mamba | flash

Pages:

* ``deepseek_v3_nvl72.html``: DeepSeek-V3 decode (4 layers, EP72, batch 64 per GPU, 4k context) on a
  GB200 NVL72, single stream and two-micro-batch overlap. Rack fabric, all 72 GPUs, GPU 0's 148 SMs,
  warps of SM 0, SASS hot spots of every kernel on GPU 0.
* ``mamba_vs_attention_b300.html``: Llama-3-8B vs Mamba-2-8B decode on one calibrated B300, plus the
  attention decode kernel alone. Mamba-2's ``selective_state_update`` runs the real Triton SASS.
* ``flash_attention_b300.html``: FlashAttention-2 vs FlashAttention-4 forward (b4 h16 s4096 d128) on
  B300, replayed from NVBit recordings of the real kernels (the committed
  ``research/flash_attention/rpath/*.rpath.gz``, or raw ``.rpath`` files from ``--traces DIR``).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import rackscale as rs  # noqa: E402
from rackscale import lower, models  # noqa: E402

RSIM = os.path.join(ROOT, "target", "release", "rsim")
OUT = os.path.join(ROOT, "docs", "viz")
TMP = tempfile.mkdtemp(prefix="rsviz_")
B300 = os.path.join(ROOT, "research", "ssm_vs_attention", "configs", "b300_calibrated.yaml")
B300_CAL = os.path.join(ROOT, "calibration", "results", "b300-sxm6-ac-sm103", "calibration.json")
FA_CAL = os.path.join(ROOT, "research", "flash_attention", "calibration", "b300-sxm6-ac-sm103.json")


def rsim(*args):
    cmd = [RSIM, *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def workload(name, wl):
    p = os.path.join(TMP, name + ".json")
    with open(p, "w") as f:
        json.dump(wl, f)
    return p


def part(name):
    return os.path.join(TMP, name + ".rsviz.json")


def deepseek():
    par = rs.Parallel(ep=72, comm_sms=20)
    runs = []
    for overlap in (False, True):
        tag = "two-micro-batch overlap" if overlap else "single stream"
        wl = lower.decode_step(models.DEEPSEEK_V3, par, 64, 4096, overlap=overlap, layers=4)
        name = f"ds_{int(overlap)}"
        rsim("viz", "run", "--system", "configs/gb200_nvl72.yaml", "--workload", workload(name, wl), "--fast",
             "--detail-ranks", "0", "--warp-sms", "0", "--max-warp-events", "60000",
             "--name", f"DeepSeek-V3 decode, NVL72 EP72, {tag}", "-o", part(name))
        runs.append(part(name))
    rsim("viz", "merge", *runs, "--title", "DeepSeek-V3 decode on GB200 NVL72", "-o",
         os.path.join(OUT, "deepseek_v3_nvl72.html"))


def mamba():
    runs = []
    for m in (models.LLAMA3_8B, models.MAMBA2_8B):
        wl = lower.decode_step(m, rs.Parallel(tp=1), 64, 16384, layers=2)
        rsim("viz", "run", "--system", B300, "--workload", workload(m.name, wl), "--fast", "--launch-gap-ns", "554",
             "--warp-sms", "0", "--max-warp-events", "60000",
             "--name", f"{m.name} decode, 2 layers, batch 64, 16k context, B300", "-o", part(m.name))
        runs.append(part(m.name))
    # The attention kernel alone at the same shape (the Mamba-2 run above already carries
    # selective_state_update, which runs the real Triton SASS, with full detail).
    rsim("viz", "kernel", "--gpu", "b300", "--calibration", B300_CAL, "attention", 64, 32, 8, 1, 16384, 128,
         "--name", "Attention decode, batch 64, 16k context", "--max-warp-events", "120000", "-o", part("attn"))
    runs.append(part("attn"))
    rsim("viz", "merge", *runs, "--title", "Mamba-2 vs attention decode on B300", "-o",
         os.path.join(OUT, "mamba_vs_attention_b300.html"))


def flash(traces):
    runs = []
    for impl, regs in (("fa2", 255), ("fa4", 128)):
        p = os.path.join(traces, f"b300_{impl}_s4096.rpath")
        if not os.path.exists(p):
            p += ".gz"
        if not os.path.exists(p):
            print(f"skip flash: {p} not found (see research/flash_attention/README.md)")
            return
        rsim("viz", "path", p, "--gpu", "b300", "--calibration", FA_CAL, "--regs", regs, "--warp-sms", "0,1",
             "--max-warp-events", "400000",
             "--name", f"FlashAttention-{impl[-1]} forward, b4 h16 s4096 d128, B300 (NVBit replay)", "-o", part(impl))
        runs.append(part(impl))
    rsim("viz", "merge", *runs, "--title", "FlashAttention-2 vs FlashAttention-4 on B300", "-o",
         os.path.join(OUT, "flash_attention_b300.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="*", default=["deepseek", "mamba", "flash"])
    ap.add_argument("--traces", default=os.path.join(ROOT, "research", "flash_attention", "rpath"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for p in a.pages:
        {"deepseek": deepseek, "mamba": mamba, "flash": lambda: flash(a.traces)}[p]()


if __name__ == "__main__":
    main()
