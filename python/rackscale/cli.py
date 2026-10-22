"""``rackscale`` command line: lower a model or a captured trace and simulate it.

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
import sys

from . import capture, lower, models, record, replay, simulate
from .lower import Parallel


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--system", default="gb200_nvl72", help="preset name or system YAML path")
    p.add_argument("--trace", help="write a Perfetto trace (open in ui.perfetto.dev)")
    p.add_argument("--report", help="write the full JSON report")
    p.add_argument("--dump-workload", help="write the lowered workload JSON and continue")
    p.add_argument("--launch-gap-ns", type=float)
    p.add_argument("--fast", action="store_true", help="simulate each distinct kernel once and reuse it across ranks/layers")
    p.add_argument("--per-sm", action="store_true", help="add one Perfetto track per SM with a span per CTA (exact mode)")


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", choices=sorted(models.ALL))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1, help="pipeline stages (train)")
    p.add_argument("--comm-sms", type=int, default=20)
    p.add_argument("--layers", type=int, help="simulate only the first N layers (memory is still checked for the full model)")


def _par(a) -> Parallel:
    return Parallel(tp=a.tp, ep=a.ep, dp=a.dp, comm_sms=a.comm_sms, pp=a.pp)


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

    sw = sub.add_parser("sweep", help="simulate a grid of models x systems x phases x layouts (rackscale.sweep)")
    sw.add_argument("--study", choices=["ladder", "attention", "moe", "context", "infra"],
                    help="a named grid from research/model_sweep (otherwise build one from the flags below)")
    sw.add_argument("--quick", action="store_true", help="the study's smoke-test grid")
    sw.add_argument("--models", help="comma-separated model presets")
    sw.add_argument("--systems", default="b300", help="comma-separated: a100, h100, b300, gb300")
    sw.add_argument("--phases", default="decode", help="comma-separated: decode, prefill, train")
    sw.add_argument("--batches", default="1,16,64,256")
    sw.add_argument("--contexts", default="8192", help="decode context lengths")
    sw.add_argument("--seqlen", type=int, default=8192, help="prefill prompt / training sequence length")
    sw.add_argument("--launch", default="graph", help="graph | eager | megakernel | gap in ns")
    sw.add_argument("--jobs", type=int, default=1)
    sw.add_argument("-o", "--out", help="write the points as JSON")

    pl = sub.add_parser("plan", help="best layout (and decode batch) for a model on a system, with the bottleneck")
    pl.add_argument("model", choices=sorted(models.ALL))
    pl.add_argument("--system", default="gb300", help="a100, h100, b300, gb300")
    pl.add_argument("--phase", choices=["decode", "prefill", "train"], default="decode")
    pl.add_argument("--context", type=int, default=8192)
    pl.add_argument("--seqlen", type=int, default=8192)
    pl.add_argument("--batches", default="1,8,32,128,256")
    pl.add_argument("--min-tps-user", type=float, default=0.0, help="decode: tokens/s each user must see")
    pl.add_argument("--max-ttft-ms", type=float, default=float("inf"), help="prefill: time-to-first-token budget")
    pl.add_argument("--launch", default="graph")
    pl.add_argument("--jobs", type=int, default=1)
    pl.add_argument("--json", action="store_true", help="machine-readable output (for scripts and agents)")

    w = sub.add_parser("run", help="simulate a workload JSON/YAML file")
    w.add_argument("workload")
    _common(w)

    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    try:
        return _run(a, cmd)
    except (ValueError, OSError) as e:
        # Bad layouts, configs and workloads: one line, no traceback.
        print(f"rackscale {a.cmd}: error: {e}", file=sys.stderr)
        return 2


def _run(a, cmd) -> int:
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
    if a.cmd == "sweep":
        return _sweep(a)
    if a.cmd == "plan":
        return _plan(a)
    if a.cmd == "decode":
        wl = lower.decode_step(models.ALL[a.model], _par(a), a.batch, a.context, overlap=a.overlap, layers=a.layers)
    elif a.cmd == "prefill":
        wl = lower.prefill(models.ALL[a.model], _par(a), a.batch, a.seqlen, layers=a.layers)
    elif a.cmd == "train":
        wl = lower.train_step(models.ALL[a.model], _par(a), a.micro_batch, a.seqlen, layers=a.layers,
                              micro_batches=a.micro_batches, schedule=a.schedule, optimizer=a.optimizer)
    elif a.cmd == "capture":
        if not a.path:
            ap.error("capture: give a torch-profiler trace, or `-- cmd ...` to capture a program")
        wl = capture.from_file(a.path, mode=a.mode, world=a.world, collective_group=a.group, max_kernels=a.max_kernels)
    else:
        wl = a.workload

    if a.dump_workload and isinstance(wl, dict):
        with open(a.dump_workload, "w") as f:
            json.dump(wl, f, indent=1)
    cfg = {"gpu": {"per_sm_counters": True}} if a.per_sm else None
    rep = simulate(a.system, wl, trace=a.trace, launch_gap_ns=a.launch_gap_ns, fast=a.fast, config=cfg)
    print(rep.summary())
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep.raw, f, indent=1)
    return 0


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
    cfg = {"gpu": {"per_sm_counters": True}} if a.per_sm else None
    rep = simulate(a.system, wl, trace=a.trace, launch_gap_ns=a.launch_gap_ns, fast=a.fast, config=cfg)
    print(rep.summary())
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



def _ints(s: str) -> list:
    return [int(x) for x in s.split(",") if x]


def _point_line(r: dict) -> str:
    from .sweep import _plan_str
    if "step_us" not in r:
        why = r.get("error") or f"does not fit ({r.get('memory_total_gb', 0):.0f} of {r.get('hbm_gb', 0):.0f} GB)"
        return f"{r['model']:18s} {r['system']:6s} {r['phase']:7s} {_plan_str(r['plan']):10s} b{r['batch']:<4d} {why}"
    bd = r["breakdown_us"]
    tot = sum(bd.values()) or 1.0
    parts = " ".join(f"{k} {100 * v / tot:3.0f}%" for k, v in bd.items() if v / tot >= 0.02)
    rate = f"{r['tok_s_gpu']:9.0f} tok/s/GPU"
    if "tok_s_user" in r:
        rate += f" {r['tok_s_user']:6.1f} tok/s/user"
    if "ttft_ms" in r:
        rate += f" TTFT {r['ttft_ms']:7.1f} ms"
    return (f"{r['model']:18s} {r['system']:6s} {r['phase']:7s} {_plan_str(r['plan']):10s} b{r['batch']:<4d} "
            f"{r['step_us']:10.1f} us {rate}  [{r['dominant']}] {parts}")


def _sweep(a) -> int:
    from . import sweep as sw
    if a.study:
        pts = sw.study_points(a.study, quick=a.quick)
    else:
        if not a.models:
            raise ValueError("give --study or --models")
        pts = []
        for mo in a.models.split(","):
            if mo not in models.ALL:
                raise ValueError(f"unknown model {mo!r}")
            for sy in a.systems.split(","):
                if sy not in sw.SYSTEMS:
                    raise ValueError(f"unknown system {sy!r} (known: {', '.join(sw.SYSTEMS)})")
                for ph in a.phases.split(","):
                    if ph == "train":
                        for p in sw.train_plans(mo, sy):
                            pts.append(sw.Point(mo, sy, "train", p, seqlen=a.seqlen, micro_batches=max(4, 2 * p["pp"]),
                                                launch=a.launch))
                        continue
                    for p in sw.inference_plans(mo, sy):
                        if ph == "decode":
                            pts += [sw.Point(mo, sy, "decode", p, batch=b, context=c, launch=a.launch)
                                    for c in _ints(a.contexts) for b in _ints(a.batches)]
                        else:
                            pts.append(sw.Point(mo, sy, "prefill", p, seqlen=a.seqlen, launch=a.launch))
    res = sw.run(pts, jobs=a.jobs)
    for r in res:
        print(_point_line(r))
    if a.out:
        sw.save(res, a.out, study=a.study or "custom")
        print("wrote", a.out)
    return 0


def _plan(a) -> int:
    import json
    from . import sweep as sw
    if a.system not in sw.SYSTEMS:
        raise ValueError(f"unknown system {a.system!r} (known: {', '.join(sw.SYSTEMS)})")
    out = sw.best_plan(a.model, a.system, a.phase, context=a.context, seqlen=a.seqlen, batches=_ints(a.batches),
                       min_tps_user=a.min_tps_user, max_ttft_ms=a.max_ttft_ms, launch=a.launch, jobs=a.jobs)
    if a.json:
        print(json.dumps(sw.plan_report(out, a.system), indent=1))
        return 0
    for r in out["points"]:
        print(_point_line(r))
    b = out["best"]
    print()
    if b is None:
        print("no layout meets the target")
        return 1
    print("best: " + _point_line(b))
    if sw.SYSTEMS[a.system].caveat:
        print("note: " + sw.SYSTEMS[a.system].caveat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
