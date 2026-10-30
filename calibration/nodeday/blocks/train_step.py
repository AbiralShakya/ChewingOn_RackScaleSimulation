"""Measure one training step of chapter 11's cases on real GPUs.

    torchrun --nproc-per-node 8 train_step.py --plan fsdp --layers 2 --out fsdp_L2.json
    torchrun --nproc-per-node 1 train_step.py --plan single --layers 8 --out single_L8.json

A step is forward + backward of a stack of Llama-3-8B layers on one
micro-batch of 4096 tokens in BF16 (pipeline plans: 8 micro-batches over 2
stages), exactly what ``rackscale.lower.train_step`` lowers: no embedding,
LM head, loss or optimizer. The step ends when the last rank's last
collective is done. Reported: per-step wall time (max over ranks) over
``--steps`` steps after ``--warmup``, plus a torch.profiler trace of two
steps on rank 0 for the per-kernel comparison.

Plans: single, ddp, fsdp, hsdp, tp, tp_sp, pp_gpipe, pp_1f1b, pp_zb.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from llama import LLAMA3_8B, Cfg, Stack, tp_plan  # noqa: E402

CFG = LLAMA3_8B

MB, SEQ, MICRO = 1, 4096, 8
CPU = not torch.cuda.is_available()  # smoke test: gloo on CPU with a tiny config (smoke.py)
DEV_TYPE = "cpu" if CPU else "cuda"


def sync():
    if not CPU:
        torch.cuda.synchronize()


def pp_order(schedule: str, pp: int, stage: int, mbs: int):
    """Per-stage op order, copied from rackscale.lower._pp_order (smoke.py checks they agree)."""
    if schedule == "gpipe":
        seq = [("F", i) for i in range(mbs)]
        for i in range(mbs):
            seq += [("B", i), ("W", i)]
        return seq
    warm = min(mbs, pp - stage - 1)
    lag = pp - stage - 1 if schedule == "zb" else 0
    seq = [("F", i) for i in range(warm)]
    pending = []
    f = warm
    for b in range(mbs):
        if f < mbs:
            seq.append(("F", f))
            f += 1
        seq.append(("B", b))
        pending.append(b)
        while len(pending) > lag:
            seq.append(("W", pending.pop(0)))
        if f >= mbs and pending and lag:
            seq.append(("W", pending.pop(0)))
    seq += [("W", i) for i in pending]
    return seq


def build(plan: str, layers: int, dev: torch.device):
    """Model, the step function and a description of how it was parallelised."""
    torch.manual_seed(0)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    c = CFG
    info = dict(plan=plan, world=world, layers=layers)

    if plan.startswith("pp_"):
        return build_pp(plan[3:], layers, dev, info)

    dt = torch.float32 if CPU else torch.bfloat16
    model = Stack(c, layers).to(device=dev, dtype=dt)
    seq_local = SEQ
    if plan == "single":
        pass
    elif plan == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        # One bucket per layer, as the simulator all-reduces each layer's gradients when its backward ends.
        layer_mb = sum(p.numel() for p in model.layers[0].parameters()) * 2 / 2**20
        model = DDP(model, device_ids=None if CPU else [dev.index], bucket_cap_mb=layer_mb, gradient_as_bucket_view=True)
        info["bucket_mb"] = round(layer_mb, 1)
    elif plan in ("fsdp", "hsdp"):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        mesh = (init_device_mesh(DEV_TYPE, (world,)) if plan == "fsdp"
                else init_device_mesh(DEV_TYPE, (world // 4, 4), mesh_dim_names=("replicate", "shard")))
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh)
        info["mesh"] = list(mesh.shape)
    elif plan in ("tp", "tp_sp"):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import parallelize_module
        mesh = init_device_mesh(DEV_TYPE, (world,))
        for layer in model.layers:
            parallelize_module(layer, mesh, tp_plan(plan == "tp_sp"))
        if plan == "tp_sp":
            seq_local = SEQ // world
    else:
        raise SystemExit(f"unknown plan {plan}")

    torch.manual_seed(1 + rank)
    x = torch.randn(MB, seq_local, c.hidden, device=dev, dtype=dt, requires_grad=True)
    g = torch.randn_like(x)

    def step():
        out = model(x)
        out.backward(g)

    return step, info


def build_pp(schedule: str, layers: int, dev, info):
    """Two-stage pipeline, each stage a contiguous slice of layers, ops in the simulator's order.

    B is the input-gradient pass and W the weight-gradient pass, split with
    PyTorch's own zero-bubble helpers (torch.distributed.pipelining._backward).
    Activations and gradients go over separate communicators, so a pending
    gradient send never waits behind an activation send.
    """
    world, rank = dist.get_world_size(), dist.get_rank()
    assert world == 2, "pipeline plans run on 2 GPUs"
    c = CFG
    per = layers // world
    torch.manual_seed(rank)
    stage = Stack(c, per).to(device=dev, dtype=torch.float32 if CPU else torch.bfloat16)
    params = list(stage.parameters())
    fwd_pg = dist.new_group([0, 1])
    bwd_pg = dist.new_group([0, 1])
    order = pp_order(schedule, world, rank, MICRO)
    try:
        from torch.distributed.pipelining._backward import stage_backward_input, stage_backward_weight
        split = True
    except ImportError:
        split = False
    info.update(schedule=schedule, micro_batches=MICRO, order=["%s%d" % o for o in order], zb_split=split)
    shape = (MB, SEQ, c.hidden)
    dt = torch.float32 if CPU else torch.bfloat16
    first, last = rank == 0, rank == world - 1

    def step():
        xs, outs, gouts, pgroups, sends = {}, {}, {}, {}, []
        for op, i in order:
            if op == "F":
                if first:
                    x = torch.randn(shape, device=dev, dtype=dt, requires_grad=True)
                else:
                    x = torch.empty(shape, device=dev, dtype=dt)
                    dist.irecv(x, src=rank - 1, group=fwd_pg).wait()
                    x.requires_grad_(True)
                y = stage(x)
                xs[i], outs[i] = x, y
                if not last:
                    y_ = y.detach()
                    sends.append((dist.isend(y_, dst=rank + 1, group=fwd_pg), y_))
            elif op == "B":
                if last:
                    gy = torch.ones(shape, device=dev, dtype=dt)
                else:
                    gy = torch.empty(shape, device=dev, dtype=dt)
                    dist.irecv(gy, src=rank + 1, group=bwd_pg).wait()
                gouts[i] = gy
                if split:
                    dx, pgroups[i] = stage_backward_input([outs[i]], [gy], [xs[i]], iter(params))
                    dx = dx[0]
                else:
                    torch.autograd.backward(outs[i], gy)
                    dx = xs[i].grad
                if not first:
                    dx_ = dx.detach()
                    sends.append((dist.isend(dx_, dst=rank - 1, group=bwd_pg), dx_))
            elif op == "W":
                if split:
                    stage_backward_weight(iter(params), pgroups.pop(i))
                outs.pop(i), xs.pop(i), gouts.pop(i)
        for w, _ in sends:
            w.wait()

    return step, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", default="", help="write a torch.profiler trace of 2 steps (rank 0)")
    a = ap.parse_args()

    global CFG, SEQ
    if CPU:
        CFG, SEQ = Cfg(hidden=512, heads=16, kv_heads=8, head_dim=32, ffn=512), 64
    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        local = int(os.environ["LOCAL_RANK"])
        if CPU:
            dist.init_process_group("gloo")
        else:
            torch.cuda.set_device(local)
            dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    dev = torch.device("cpu") if CPU else torch.device("cuda", torch.cuda.current_device())
    rank = dist.get_rank() if distributed else 0

    step, info = build(a.plan, a.layers, dev)

    def timed() -> float:
        if distributed:
            dist.barrier()
        sync()
        t0 = time.perf_counter()
        step()
        sync()
        t = time.perf_counter() - t0
        if distributed:  # the step ends with the slowest rank
            tt = torch.tensor([t], device=dev)
            dist.all_reduce(tt, op=dist.ReduceOp.MAX)
            t = tt.item()
        return t

    for _ in range(a.warmup):
        timed()
    times = [timed() * 1e6 for _ in range(a.steps)]

    if a.trace and rank == 0:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                step()
            sync()
        prof.export_chrome_trace(a.trace)
        kern = {}
        for e in prof.key_averages():
            us = getattr(e, "self_device_time_total", None)
            if us is None:
                us = getattr(e, "self_cuda_time_total", 0)
            if us > 0:
                kern[e.key] = dict(n=e.count // 2, us=us / 2)
        info["kernels"] = dict(sorted(kern.items(), key=lambda kv: -kv[1]["us"])[:40])
    elif a.trace and distributed:
        for _ in range(2):  # keep collectives matched with rank 0's profiled steps
            step()
        sync()

    if rank == 0:
        res = dict(info, step_us=statistics.median(times), min_us=min(times), max_us=max(times),
                   stdev_us=statistics.pstdev(times), steps_us=times, warmup=a.warmup,
                   gpu="cpu" if CPU else torch.cuda.get_device_name(), torch=torch.__version__,
                   nccl=".".join(map(str, torch.cuda.nccl.version())) if distributed and not CPU else None,
                   sdpa_backends=dict(cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
                                      flash=torch.backends.cuda.flash_sdp_enabled()))
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"{a.plan:<9} L{a.layers}  step {res['step_us']:10.1f} us  (min {res['min_us']:.1f}, "
              f"sd {res['stdev_us']:.1f})", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
