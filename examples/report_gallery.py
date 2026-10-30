"""Build the example run reports in docs/reports/ and their figures in docs/img/report_*/.

    maturin develop --release
    python examples/report_gallery.py                  # all reports
    python examples/report_gallery.py deepseek         # one: deepseek | llama | train

Reports (open the HTML in a browser; the SVGs drop straight into an article):

* ``deepseek_v3_nvl72.html``: DeepSeek-V3 decode (first 4 layers, EP72, batch 64 per GPU, 4k context)
  on GB200 NVL72 in BF16, and on GB300 NVL72 in BF16 and with FP4 GEMMs.
* ``llama3_70b_generations.html``: Llama-3-70B decode (2 layers, TP8, batch 32, 4k context) on
  HGX H100, B200 and B300.
* ``llama3_70b_train.html``: one Llama-3-70B training step (2 layers, TP8, micro-batch 1, 4k) on
  HGX H100 and B200.

Every run is fast mode (each distinct kernel simulated once at cycle level, then reused).
"""

import argparse
import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import rackscale as rs  # noqa: E402
from rackscale import lower, models, report  # noqa: E402

OUT = os.path.join(ROOT, "docs", "reports")
IMG = os.path.join(ROOT, "docs", "img")


def with_dtype(wl: dict, dtype: str) -> dict:
    """Run every GEMM of the workload in `dtype` (weights and activations)."""
    wl = copy.deepcopy(wl)
    for stream in wl["spmd"]["program"]["streams"]:
        for op in stream:
            k = op.get("kernel", {})
            if k.get("type") in ("gemm", "grouped_gemm"):
                k["dtype"] = dtype
    wl["name"] += f" [{dtype}]"
    return wl


def build(name, runs, systems, labels, title, note):
    reps = []
    for (sysname, wl), lab in zip(runs, labels):
        print(f"  {lab:<28}", end="", flush=True)
        rep = rs.simulate(sysname, wl, fast=True)
        print(f"{rep.makespan_us:9.1f} us")
        reps.append(rep)
    path = report.write(reps, os.path.join(OUT, name + ".html"), system=systems, labels=labels, title=title, note=note)
    figs = report.figures(reps, os.path.join(IMG, "report_" + name), system=systems, labels=labels)
    print(f"  -> {os.path.relpath(path, ROOT)} and {len(figs)} figures in docs/img/report_{name}/")


def deepseek():
    base = lower.decode_step(models.DEEPSEEK_V3, rs.Parallel(ep=72), 64, 4096, layers=4)
    runs = [("gb200_nvl72", with_dtype(base, "bf16")), ("gb300_nvl72", with_dtype(base, "bf16")),
            ("gb300_nvl72", with_dtype(base, "fp4"))]
    build("deepseek_v3_nvl72", runs, [r[0] for r in runs], ["GB200 NVL72, BF16", "GB300 NVL72, BF16", "GB300 NVL72, FP4 GEMMs"],
          "DeepSeek-V3 decode on an NVL72 rack: GB200 vs GB300, BF16 vs FP4",
          "First 4 of 61 layers (3 dense, 1 MoE), expert parallel over all 72 GPUs, 64 sequences per GPU at 4k "
          "context, one token each. Memory is checked for the whole model.\n\n"
          "Decode GEMMs are weight-bandwidth bound, so FP4 helps through bytes moved, not through the 3x higher "
          "FP4 tensor peak; attention and the all-to-alls do not change.")


def llama():
    wl = lower.decode_step(models.LLAMA3_70B, rs.Parallel(tp=8), 32, 4096, layers=2)
    systems = ["hgx_h100_8", "hgx_b200_8", "hgx_b300_8"]
    build("llama3_70b_generations", [(s, wl) for s in systems], systems, ["HGX H100", "HGX B200", "HGX B300"],
          "Llama-3-70B decode, TP8: three GPU generations",
          "2 of 80 layers, tensor parallel over 8 GPUs, 32 sequences at 4k context. Same workload, same "
          "kernel library per generation (HGMMA on Hopper, tcgen05 on Blackwell).")


def train():
    wl = lower.train_step(models.LLAMA3_70B, rs.Parallel(tp=8), 1, 4096, layers=2)
    systems = ["hgx_h100_8", "hgx_b200_8"]
    build("llama3_70b_train", [(s, wl) for s in systems], systems, ["HGX H100", "HGX B200"],
          "Llama-3-70B training step, TP8: H100 vs B200",
          "2 of 80 layers, forward and backward, tensor parallel over 8 GPUs, micro-batch 1 at 4k sequence length.")


GALLERY = {"deepseek": deepseek, "llama": llama, "train": train}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", choices=[[], *GALLERY], help="reports to build (default: all)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for w in a.which or GALLERY:
        print(w)
        GALLERY[w]()
