"""``rackscale`` command line: lower a model or a captured trace and simulate it.

    rackscale demo --open                  # 30 seconds: H100 vs B200 on Llama-3-70B, then the viewer
    rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 2 --trace out/ds.json
    rackscale train llama3-70b --system hgx_h100_8 --tp 8 --micro-batch 1 --seqlen 4096 --layers 2
    rackscale train llama3-70b --system hgx_b200_8 --pp 8 --micro-batches 16 --schedule zb --fast
    rackscale capture vllm_trace.json.gz --system gb200_nvl72 --world 8
    rackscale capture -o cap/ --region forward:4:2 -- python my_engine.py ...   # any CUDA program, under NVBit
    rackscale replay cap/ --system gb300_nvl72 --tp 8 --dp 9
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import webbrowser

from . import _term, capture, lower, models, record, replay, simulate
from .lower import Parallel

#: Commands of the Rust ``rsim`` CLI, forwarded as-is (``rackscale kernel ...`` == ``rsim kernel ...``).
RSIM_COMMANDS = {
    "presets": "list built-in GPU, switch and system presets",
    "system": "resolve a system YAML (or preset) and print its topology",
    "kernel": "simulate one synthetic kernel on one GPU (gemm, attention, ssm_update, ...)",
    "sass": "simulate one kernel from a cuobjdump/nvdisasm listing",
    "path": "simulate a kernel recorded on hardware with NVBit",
    "timeline": "per-warp issue timeline of one kernel (compare with clock64 probes)",
    "validate": "replay measured reference kernels and report error vs Nsight Compute",
    "net": "collective sweeps, NCCL cost model, validation against published numbers",
    "viz": "write the zoomable rack -> GPU -> SM -> warp -> SASS viewer page",
}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--system", default="gb200_nvl72", help="preset name or system YAML path")
    p.add_argument("--trace", help="write a Perfetto trace (open in ui.perfetto.dev)")
    p.add_argument("--report", help="write the full JSON report")
    p.add_argument("--dump-workload", help="write the lowered workload JSON and continue")
    p.add_argument("--launch-gap-ns", type=float)
    p.add_argument("--fast", action="store_true", help="simulate each distinct kernel once and reuse it across ranks/layers")
    p.add_argument("--per-sm", action="store_true", help="add one Perfetto track per SM with a span per CTA (exact mode)")
    p.add_argument("--viz", metavar="PAGE.html", help="also write the zoomable viewer page (rack down to SASS) for rank 0")
    p.add_argument("--open", action="store_true", help="open the --viz page (or Perfetto for --trace) in a browser")


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", choices=sorted(models.ALL), metavar="MODEL",
                   help="one of: " + ", ".join(sorted(models.ALL)) + " (see `rackscale models`)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1, help="pipeline stages (train)")
    p.add_argument("--comm-sms", type=int, default=20)
    p.add_argument("--layers", type=int, help="simulate only the first N layers (memory is still checked for the full model)")


def _par(a) -> Parallel:
    return Parallel(tp=a.tp, ep=a.ep, dp=a.dp, comm_sms=a.comm_sms, pp=a.pp)


def main(argv=None) -> int:
    if argv is None and hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `rackscale ... | head` exits quietly
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in RSIM_COMMANDS:
        return _rsim(argv)

    from . import __version__
    epilog = "simulator-core commands (same as the `rsim` binary; `rackscale CMD --help` for each):\n" + "\n".join(
        f"  {k:<10} {v}" for k, v in RSIM_COMMANDS.items()) + "\n\ndocs: https://abiralshakya.github.io/ChewingOn_RackScaleSimulation/"
    ap = argparse.ArgumentParser(prog="rackscale", description=__doc__, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"rackscale {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

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
    t.add_argument("--micro-batches", type=int, default=1, help="micro-batches per step (pipeline)")
    t.add_argument("--schedule", choices=["gpipe", "1f1b", "zb"], default="1f1b",
                   help="pipeline schedule: GPipe, 1F1B, or ZB-H1 zero-bubble")
    t.add_argument("--optimizer", action="store_true", help="end the step with an Adam update and DP all-gather")
    _common(t)

    c = sub.add_parser("capture", help="capture any CUDA program under the NVBit tool (`-- cmd ...`), "
                                       "or simulate a torch-profiler trace (PATH)")
    c.add_argument("path", nargs="?", help="torch-profiler trace (when no `-- cmd` is given)")
    c.add_argument("-o", "--out", default="capture", help="capture directory (with `-- cmd`)")
    c.add_argument("--region", default="marked",
                   help="marked (rsim_capture.region() in the program) | all | forward:SKIP:COUNT[:CLASS_REGEX]")
    c.add_argument("--key", choices=["config", "func"], default="config",
                   help="what makes two launches the same kernel (one recording each)")
    c.add_argument("--passes", default="graph,paths,timing",
                   help="graph (launch graph), paths (warp paths), timing (native nsys run for exact times)")
    c.add_argument("--no-modules", action="store_true", help="do not mark module calls (cleaner host timeline)")
    c.add_argument("--max-paths", type=int, default=1024)
    c.add_argument("--max-mb", type=int, default=2048, help="drop recordings larger than this")
    c.add_argument("--filter", default="", help="only record kernels whose mangled name contains this")
    c.add_argument("--gpu-lock", help="flock this file around each pass")
    c.add_argument("--build-only", action="store_true", help="rebuild capture.json from existing passes")
    c.add_argument("--world", type=int, default=1)
    c.add_argument("--group", default="world", help="communicator group for captured NCCL ops")
    c.add_argument("--mode", choices=["resimulate", "replay"], default="resimulate")
    c.add_argument("--max-kernels", type=int)
    _common(c)

    r = sub.add_parser("replay", help="simulate a capture on a system with a parallelism plan")
    r.add_argument("capture")
    r.add_argument("--plan", help="plan YAML (tp, ep, pp, dp, microbatches, collectives)")
    for f in ("tp", "ep", "pp", "dp", "microbatches"):
        r.add_argument(f"--{f}", type=int)
    r.add_argument("--topk", type=float, help="experts per token (all-to-all bytes)")
    r.add_argument("--mode", choices=["paths", "measured"], default="paths",
                   help="re-simulate recorded kernels, or keep every kernel's measured duration")
    r.add_argument("--host", action="store_true", help="replay the captured host issue time of every launch")
    r.add_argument("--steps", help="engine steps to replay, A:B (0-based, B exclusive)")
    r.add_argument("--kernels", action="store_true", help="also compare every recorded kernel alone to its measurement")
    _common(r)

    w = sub.add_parser("run", help="simulate a workload JSON/YAML file")
    w.add_argument("workload")
    _common(w)

    _extra_commands(sub)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    if a.cmd == "capture" and not (cmd or a.build_only or a.path):
        c.error("give a torch-profiler trace, or `-- cmd ...` to capture a program")
    try:
        return _run(a, cmd)
    except (ValueError, OSError, ImportError) as e:
        # Bad layouts, configs and workloads: one line, no traceback.
        print(_term.style(f"rackscale {a.cmd}: error: {e}", "red", stream=sys.stderr), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _rsim(argv, prog: str = "rackscale") -> int:
    from . import _need_native
    return _need_native().rsim_main([str(x) for x in argv], prog)


def _extra_commands(sub) -> None:
    dm = sub.add_parser("demo", help="a 30-second tour: Llama-3-70B decode on HGX H100 vs B200, then the viewer")
    dm.add_argument("-o", "--out", default="rackscale_demo.html", help="viewer page to write")
    dm.add_argument("--open", action="store_true", help="open the page in a browser")
    sub.add_parser("models", help="list the built-in model architectures")
    v = sub.add_parser("view", help="open a viewer page (.html) or a Perfetto trace (.json) in a browser")
    v.add_argument("file")


def _open(path: str) -> None:
    if path.endswith(".html"):
        url = "file://" + os.path.abspath(path)
        print(f"opening {url}")
        webbrowser.open(url)
        return
    print(f"Perfetto traces open in https://ui.perfetto.dev: click 'Open trace file' and pick {os.path.abspath(path)}")
    webbrowser.open("https://ui.perfetto.dev")


def _models() -> int:
    print(f"{'model':<20} {'layers':>6} {'hidden':>7} {'heads':>9} {'active':>8} {'total':>8}  notes")
    for name in sorted(models.ALL):
        m = models.ALL[name]
        total = m.hidden * m.vocab + (sum(m.attn_params() + m.ffn_params(i) for i in range(m.layers))
                                      if m.blocks is None else m.active_params(lm_head=False))
        notes = []
        if m.moe:
            notes.append(f"MoE {m.moe.experts}x top-{m.moe.topk}")
        if m.mla:
            notes.append("MLA")
        if m.mamba:
            notes.append("Mamba-2" + (" hybrid" if m.blocks and "A" in m.blocks else ""))
        if m.kv_heads != m.heads and not m.mla:
            notes.append(f"GQA {m.heads}/{m.kv_heads}")
        heads = f"{m.heads}/{m.kv_heads}"
        print(f"{name:<20} {m.layers:>6} {m.hidden:>7} {heads:>9} {m.active_params() / 1e9:>7.1f}B "
              f"{total / 1e9:>7.1f}B  {', '.join(notes)}")
    return 0


def _demo(a) -> int:
    from . import viz
    m = models.LLAMA3_70B
    par = Parallel(tp=8)
    print(_term.style("rackscale demo", "bold") + ": one decode step of Llama-3-70B (2 of 80 layers), TP8, "
          "32 sequences at 4k context, on two 8-GPU systems.\n")
    wl = lower.decode_step(m, par, batch=32, context=4096, layers=2)
    rows = []
    for system in ("hgx_h100_8", "hgx_b200_8"):
        with _term.progress(f"simulating every SM of {system} (fast mode)"):
            rep = simulate(system, wl, fast=True)
        rows.append((system, rep))
    print()
    print(f"{'system':<12} {'step':>10} {'GPU idle':>9} {'tensor idle':>12}   top kernel")
    for system, rep in rows:
        b = rep.bubbles
        top = rep.kernels(1)[0]["name"] if rep.kernels(1) else "-"
        print(f"{system:<12} {rep.makespan_us:>7.1f} us {100 * b['gpu_idle_fraction']:>8.1f}% "
              f"{100 * b['tensor_idle_fraction']:>11.1f}%   {top}")
    h, b = rows[0][1].makespan_us, rows[1][1].makespan_us
    print(f"\nB200 is {h / b:.2f}x faster per step here. Full anatomy of the B200 run:\n")
    print(_term.colorize_summary(rows[1][1].summary()))
    with _term.progress(f"writing the viewer page {a.out} (rack -> GPU -> SMs -> warps -> SASS)"):
        viz("hgx_b200_8", wl, a.out, name="Llama-3-70B decode, TP8 on HGX B200", warp_sms=[0])
    print(f"\nviewer: {os.path.abspath(a.out)}  (no server needed; `rackscale view {a.out}`)")
    print("next:   rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --layers 4 --fast --viz ds.html --open")
    if a.open:
        _open(a.out)
    return 0


def _run(a, cmd) -> int:
    if a.cmd == "demo":
        return _demo(a)
    if a.cmd == "models":
        return _models()
    if a.cmd == "view":
        if not os.path.exists(a.file):
            raise OSError(f"no such file: {a.file}")
        _open(a.file)
        return 0
    if a.cmd == "capture" and (cmd or a.build_only):
        if a.build_only:
            record.build(a.out, key=a.key)
        else:
            record.run(cmd, a.out, region=a.region, passes=a.passes.split(","), modules=not a.no_modules,
                       key=a.key, max_paths=a.max_paths, max_mb=a.max_mb, kernel_filter=a.filter,
                       gpu_lock=a.gpu_lock)
        return 0
    if a.cmd == "replay":
        return _replay(a)
    if a.cmd == "decode":
        wl = lower.decode_step(models.ALL[a.model], _par(a), a.batch, a.context, overlap=a.overlap, layers=a.layers)
    elif a.cmd == "prefill":
        wl = lower.prefill(models.ALL[a.model], _par(a), a.batch, a.seqlen, layers=a.layers)
    elif a.cmd == "train":
        wl = lower.train_step(models.ALL[a.model], _par(a), a.micro_batch, a.seqlen, layers=a.layers,
                              micro_batches=a.micro_batches, schedule=a.schedule, optimizer=a.optimizer)
    elif a.cmd == "capture":
        wl = capture.from_file(a.path, mode=a.mode, world=a.world, collective_group=a.group, max_kernels=a.max_kernels)
    else:
        wl = a.workload

    if a.dump_workload and isinstance(wl, dict):
        with open(a.dump_workload, "w") as f:
            json.dump(wl, f, indent=1)
    rep = _simulate(a, wl)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep.raw, f, indent=1)
    return 0


def _simulate(a, wl):
    """Simulate with a progress line, print the summary, write/open --trace and --viz."""
    from . import viz
    cfg = {"gpu": {"per_sm_counters": True}} if a.per_sm else None
    what = f"{a.model} {a.cmd}" if getattr(a, "model", None) else a.cmd
    mode = "fast mode" if a.fast else "exact, every SM of every GPU"
    with _term.progress(f"simulating {what} on {a.system} ({mode})"):
        rep = simulate(a.system, wl, trace=a.trace, launch_gap_ns=a.launch_gap_ns, fast=a.fast, config=cfg)
    print(_term.colorize_summary(rep.summary()))
    if a.trace:
        print(f"trace  -> {a.trace} (open in https://ui.perfetto.dev)")
    if a.viz:
        with _term.progress(f"writing viewer page {a.viz}"):
            viz(a.system, wl, a.viz, fast=a.fast, launch_gap_ns=a.launch_gap_ns)
        print(f"viewer -> {a.viz} (open in any browser, or `rackscale view {a.viz}`)")
    if a.open:
        if a.viz or a.trace:
            _open(a.viz or a.trace)
        else:
            print("--open needs --viz PAGE.html or --trace FILE.json", file=sys.stderr)
    return rep


def _replay(a) -> int:
    plan = dict(replay.load_plan(a.plan).__dict__) if a.plan else {}
    for f in ("tp", "ep", "pp", "dp", "microbatches", "topk"):
        if getattr(a, f) is not None:
            plan[f] = getattr(a, f)
    plan = replay.Plan(plan)
    steps = None
    if a.steps:
        lo, _, hi = a.steps.partition(":")
        steps = (int(lo or 0), int(hi) if hi else None)
    cap = record.load(a.capture)
    wl = replay.to_workload(cap, plan, mode=a.mode, host=a.host, steps=steps)
    if a.dump_workload:
        with open(a.dump_workload, "w") as f:
            json.dump(wl, f, indent=1)
    rep = _simulate(a, wl)
    m = replay.measured(cap, steps=steps)
    print(f"captured on {cap['device'].get('name')}: GPU span {m['gpu_span_us']:.1f} us, host span "
          f"{m['host_span_us']:.1f} us, kernel busy {m['kernel_busy_us']:.1f} us; simulated makespan "
          f"{rep.makespan_us:.1f} us on {plan.world} GPU(s)")
    if a.kernels:
        print(f"{'kernel':<10} {'name':<56} {'n':>5} {'meas_us':>9} {'sim_us':>9} {'err%':>7}")
        for row in replay.kernel_table(cap):
            if "error" in row:
                print(f"{row['kernel']:<10} {row['name'][:56]:<56} error: {row['error']}")
                continue
            print(f"{row['kernel']:<10} {row['name'][:56]:<56} {row['count']:>5} {row['measured_us'] or 0:>9.2f} "
                  f"{row['sim_us']:>9.2f} {row['error_pct'] or 0:>+7.1f}")
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep.raw, f, indent=1)
    return 0


def rsim_entry() -> None:
    """The ``rsim`` console script installed by pip (the Rust CLI, run in-process)."""
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(_rsim(sys.argv[1:], "rsim"))


if __name__ == "__main__":
    sys.exit(main())
