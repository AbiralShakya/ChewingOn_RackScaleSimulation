"""Chapter 5: pipeline-parallel bubbles, simulated.

Llama-3-70B training, pipeline parallel over the 8 GPUs of an HGX B200
(10 layers per stage), sequence length 4096, one sequence per micro-batch.
A stage's forward and backward (about twice the forward GEMM and attention
work) are first simulated once at cycle level on one GPU. The pipeline runs
then use those two durations as fixed-length kernels on 140 of the 148 SMs,
which keeps each schedule to seconds of simulation. Activations (forward)
and activation gradients (backward) move between neighbouring stages as
point-to-point transfers on their own streams over the simulated NVLink
fabric, so compute and communication overlap the way Megatron's batched
isend/irecv do.

Two schedules:
  gpipe  all forwards, then all backwards
  1f1b   (PipeDream-Flush) p - r - 1 warm-up forwards on stage r, then
         alternate one forward and one backward, then drain

The textbook bubble fraction for both is (p - 1) / (m + p - 1). 1F1B does not
shrink it; it bounds the activations a stage holds to p micro-batches
instead of m.

    maturin develop --release
    python docs/book/scripts/pipeline_bubbles.py
"""

import argparse

import rackscale as rs
from rackscale import lower, models

TENSOR_KERNELS = ("gemm", "grouped_gemm", "attention")


def stage_ops(m, layers, seqlen):
    mm = m.with_layers(layers)
    fwd = lower._layers(mm, rs.Parallel(), 1, seqlen, seqlen, True, "fwd.")
    bwd = []
    for op in lower._layers(mm, rs.Parallel(), 1, seqlen, seqlen, True, "bwd."):
        bwd.append(op)
        if op["op"] == "kernel" and op["kernel"]["type"] in TENSOR_KERNELS:
            bwd.append(op)  # dgrad and wgrad
    return fwd, bwd


def order(schedule, p, r, m):
    if schedule == "gpipe":
        return [("F", i) for i in range(m)] + [("B", i) for i in range(m)]
    warm = min(m, p - r - 1)
    seq = [("F", i) for i in range(warm)]
    f, b = warm, 0
    while b < m:
        if f < m:
            seq.append(("F", f))
            f += 1
        seq.append(("B", b))
        b += 1
    return seq


def workload(schedule, p, m, fwd, bwd, act_bytes):
    ranks = []
    for r in range(p):
        comp, fsend, frecv, bsend, brecv = [], [], [], [], []
        for kind, i in order(schedule, p, r, m):
            if kind == "F":
                if r > 0:
                    comp.append({"op": "wait", "event": f"fin{i}"})
                comp += fwd
                if r < p - 1:
                    comp.append({"op": "record", "event": f"fout{i}"})
            else:
                if r < p - 1:
                    comp.append({"op": "wait", "event": f"bin{i}"})
                comp += bwd
                if r > 0:
                    comp.append({"op": "record", "event": f"bout{i}"})
        for i in range(m):
            if r < p - 1:
                fsend += [{"op": "wait", "event": f"fout{i}"}, {"op": "send", "bytes": act_bytes, "peer": r + 1}]
                brecv += [{"op": "recv", "bytes": act_bytes, "peer": r + 1}, {"op": "record", "event": f"bin{i}"}]
            if r > 0:
                frecv += [{"op": "recv", "bytes": act_bytes, "peer": r - 1}, {"op": "record", "event": f"fin{i}"}]
                bsend += [{"op": "wait", "event": f"bout{i}"}, {"op": "send", "bytes": act_bytes, "peer": r - 1}]
        ranks.append({"streams": [s for s in (comp, fsend, frecv, bsend, brecv) if s]})
    return {"name": f"pp{p} {schedule} m{m}", "ranks": ranks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="hgx_b200_8")
    ap.add_argument("--pp", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--microbatches", default="1,2,4,8,16,32")
    a = ap.parse_args()
    m = models.LLAMA3_70B
    fwd_ops, bwd_ops = stage_ops(m, m.layers // a.pp, a.seqlen)
    act = a.seqlen * m.hidden * 2
    t_f, t_b = (rs.simulate(a.system, {"name": "stage", "spmd": {"world": 1, "program": {"streams": [ops]}}},
                            fast=True, launch_gap_ns=0).makespan_us for ops in (fwd_ops, bwd_ops))
    fwd = [{"op": "kernel", "name": "stage_fwd", "kernel": {"type": "fixed", "ns": t_f * 1e3, "sms": 140}}]
    bwd = [{"op": "kernel", "name": "stage_bwd", "kernel": {"type": "fixed", "ns": t_b * 1e3, "sms": 140}}]
    t_mb = t_f + t_b
    print(f"{m.name}, PP={a.pp} on {a.system}, {m.layers // a.pp} layers/stage, seq {a.seqlen}, "
          f"activation {act / 1e6:.0f} MB per hop")
    print(f"one micro-batch on one stage: forward {t_f:.0f} us + backward {t_b:.0f} us = {t_mb:.0f} us")
    print(f"{'m':>4} {'schedule':<6} {'step_us':>9} {'ideal_us':>9} {'bubble':>7} {'(p-1)/(m+p-1)':>14}")
    for mbs in [int(x) for x in a.microbatches.split(",")]:
        for sched in ("gpipe", "1f1b"):
            rep = rs.simulate(a.system, workload(sched, a.pp, mbs, fwd, bwd, act), fast=True, launch_gap_ns=0)
            ideal = mbs * t_mb
            print(f"{mbs:>4} {sched:<6} {rep.makespan_us:>9.0f} {ideal:>9.0f} {1 - ideal / rep.makespan_us:>7.1%} "
                  f"{(a.pp - 1) / (mbs + a.pp - 1):>14.1%}")


if __name__ == "__main__":
    main()
