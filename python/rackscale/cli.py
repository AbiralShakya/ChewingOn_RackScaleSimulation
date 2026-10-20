"""``rackscale`` command line: lower a model or a captured trace and simulate it.

    rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 2 --trace out/ds.json
    rackscale train llama3-70b --system hgx_h100_8 --tp 8 --micro-batch 1 --seqlen 4096 --layers 2
    rackscale capture vllm_trace.json.gz --system gb200_nvl72 --world 8
"""

from __future__ import annotations

import argparse
import json
import sys

from . import capture, lower, models, simulate
from .lower import Parallel


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--system", default="gb200_nvl72", help="preset name or system YAML path")
    p.add_argument("--trace", help="write a Perfetto trace (open in ui.perfetto.dev)")
    p.add_argument("--report", help="write the full JSON report")
    p.add_argument("--dump-workload", help="write the lowered workload JSON and continue")
    p.add_argument("--launch-gap-ns", type=float)
    p.add_argument("--fast", action="store_true", help="simulate each distinct kernel once and reuse it across ranks/layers")


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", choices=sorted(models.ALL))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--comm-sms", type=int, default=20)
    p.add_argument("--layers", type=int, help="simulate only the first N layers")


def _par(a) -> Parallel:
    return Parallel(tp=a.tp, ep=a.ep, dp=a.dp, comm_sms=a.comm_sms)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rackscale", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decode", help="one decode iteration")
    _model_args(d)
    d.add_argument("--batch", type=int, default=64, help="sequences per rank")
    d.add_argument("--context", type=int, default=4096)
    d.add_argument("--overlap", action="store_true", help="two-micro-batch compute/comm overlap (DeepSeek style)")
    _common(d)

    pf = sub.add_parser("prefill", help="one prefill pass")
    _model_args(pf)
    pf.add_argument("--batch", type=int, default=1)
    pf.add_argument("--seqlen", type=int, default=4096)
    _common(pf)

    t = sub.add_parser("train", help="forward+backward of one micro-batch")
    _model_args(t)
    t.add_argument("--micro-batch", type=int, default=1)
    t.add_argument("--seqlen", type=int, default=4096)
    _common(t)

    c = sub.add_parser("capture", help="simulate a torch-profiler trace (vLLM, SGLang, training)")
    c.add_argument("path")
    c.add_argument("--world", type=int, default=1)
    c.add_argument("--group", default="world", help="communicator group for captured NCCL ops")
    c.add_argument("--mode", choices=["resimulate", "replay"], default="resimulate")
    c.add_argument("--max-kernels", type=int)
    _common(c)

    w = sub.add_parser("run", help="simulate a workload JSON/YAML file")
    w.add_argument("workload")
    _common(w)

    a = ap.parse_args(argv)
    if a.cmd == "decode":
        wl = lower.decode_step(models.ALL[a.model], _par(a), a.batch, a.context, overlap=a.overlap, layers=a.layers)
    elif a.cmd == "prefill":
        wl = lower.prefill(models.ALL[a.model], _par(a), a.batch, a.seqlen, layers=a.layers)
    elif a.cmd == "train":
        wl = lower.train_step(models.ALL[a.model], _par(a), a.micro_batch, a.seqlen, layers=a.layers)
    elif a.cmd == "capture":
        wl = capture.from_file(a.path, mode=a.mode, world=a.world, collective_group=a.group, max_kernels=a.max_kernels)
    else:
        wl = a.workload

    if a.dump_workload and isinstance(wl, dict):
        with open(a.dump_workload, "w") as f:
            json.dump(wl, f, indent=1)
    rep = simulate(a.system, wl, trace=a.trace, launch_gap_ns=a.launch_gap_ns, fast=a.fast)
    print(rep.summary())
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep.raw, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
