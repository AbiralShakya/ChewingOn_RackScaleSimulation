"""Chapter 16: stragglers, link faults and hash collisions, against their closed forms.

    maturin develop --release
    python docs/book/scripts/tail_latency.py              # everything (about 10 minutes on 4 cores)
    python docs/book/scripts/tail_latency.py stragglers   # or: sync, slow, nvlink, ib

stragglers  a bulk-synchronous loop on GB300 NVL72: 16 x (GEMM, 1 MiB all-reduce over N
            ranks), each GEMM slowed by 1 + 0.05 |Z|, against sigma * E[max_N |Z|]
sync        the same 16 GEMMs on 72 ranks, synchronising every 1, 4 or 16 GEMMs
slow        one GPU 10% slow, the rest exact
nvlink      a 64 MiB all-reduce on 72 GPUs (NVLS and ring) with one GPU-NVSwitch link
            dead, at half rate, or down for 50 us
ib          a 32 MiB all-to-all on 64 H800s over a rail-optimised InfiniBand fat tree
            (PXN off, so cross-rail traffic crosses the spines), adaptive routing vs ECMP,
            with one leaf-spine link dead, at half rate, or down for 200 us
"""

import concurrent.futures as cf
import os
import sys

import rackscale as rs
from rackscale import tail

GEMM = {"type": "gemm", "m": 2048, "n": 4096, "k": 4096}


def bsp(group, phases=16, per=1):
    body = [{"op": "kernel", "kernel": GEMM} for _ in range(per)]
    body.append({"op": "collective", "kind": "all_reduce", "bytes": 1 << 20, "group": group, "sms": 16})
    return {"name": "bsp", "spmd": {"world": 72, "program": {"streams": [
        [{"op": "loop", "times": phases // per, "body": body}]]}}}


def finish(wl, var=None, net=None, system="gb300_nvl72", rank=None):
    cfg = {}
    if var:
        cfg["variability"] = var
    if net:
        cfg["net"] = net
    r = rs.simulate(system, wl, fast=True, config=cfg)
    if rank is None:
        return r.makespan_us
    return r.raw["ranks"][rank]["finish_us"]


SEEDS = (1, 2, 3)


def gemm_us():
    """One GEMM's span on a rank, at the power-capped clock (what jitter scales)."""
    wl = {"name": "g", "spmd": {"world": 72, "program": {"streams": [[
        {"op": "loop", "times": 16, "body": [{"op": "kernel", "kernel": GEMM}]}]]}}}
    r = rs.simulate("gb300_nvl72", wl, fast=True)
    return r.raw["ranks"][0]["kernel_us"] / 16


def pmap(f, xs):
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        return list(ex.map(f, xs))


def stragglers():
    sigma, phases = 0.05, 16
    tg = gemm_us()
    ns = [2, 4, 8, 18, 36, 72]
    vs = [None] + [{"kernel_sigma": sigma, "seed": s} for s in SEEDS]
    jobs = [(n, v) for n in ns for v in vs]
    res = pmap(lambda j: finish(bsp(f"block:{j[0]}"), j[1], rank=0), jobs)
    print(f"\n## Stragglers: 16 x (GEMM {tg:.1f} us + 1 MiB all-reduce), kernel_sigma = {sigma}, "
          f"mean of {len(SEEDS)} seeds\n")
    print("| ranks N | step, no jitter (µs) | step, jitter (µs) | added (µs) | added / compute | "
          "closed form σ·E[max_N \\|Z\\|] | one GPU alone σ·E\\|Z\\| |")
    print("|---|---|---|---|---|---|---|")
    k = len(vs)
    for i, n in enumerate(ns):
        b, v = res[k * i], sum(res[k * i + 1:k * i + k]) / len(SEEDS)
        print(f"| {n} | {b:.1f} | {v:.1f} | {v - b:.1f} | {100 * (v - b) / (phases * tg):.1f}% | "
              f"{100 * tail.straggler_overhead(n, sigma):.1f}% | {100 * tail.mean_overhead(sigma):.1f}% |")


def sync():
    sigma, phases = 0.05, 16
    tg = gemm_us()
    pers = [1, 4, 16]
    vs = [None] + [{"kernel_sigma": sigma, "seed": s} for s in SEEDS]
    jobs = [(p, v) for p in pers for v in vs]
    res = pmap(lambda j: finish(bsp("world", phases, j[0]), j[1], rank=0), jobs)
    k = len(vs)
    res = [x for i in range(len(pers)) for x in (res[k * i], sum(res[k * i + 1:k * i + k]) / len(SEEDS))]
    print(f"\n## Synchronise less often: 16 GEMMs on 72 ranks, kernel_sigma = {sigma}, mean of {len(SEEDS)} seeds\n")
    print("| GEMMs between all-reduces | step, no jitter (µs) | step, jitter (µs) | added / compute | closed form |")
    print("|---|---|---|---|---|")
    for i, p in enumerate(pers):
        b, v = res[2 * i], res[2 * i + 1]
        cf_ = tail.straggler_overhead(72, sigma, sync_every=p, trials=2000)
        print(f"| {p} | {b:.1f} | {v:.1f} | {100 * (v - b) / (phases * tg):.1f}% | {100 * cf_:.1f}% |")


def slow():
    tg = gemm_us()
    jobs = [None, {"slow_ranks": [[5, 1.10]]}, {"gpu_sigma": 0.02, "seed": 3}]
    res = pmap(lambda v: finish(bsp("world"), v, rank=0), jobs)
    print("\n## One slow GPU, 72 ranks synchronising every GEMM\n")
    print("| case | step (µs) | vs exact | compute share |")
    print("|---|---|---|---|")
    for lab, r in zip(["all exact", "rank 5 is 10% slow", "every GPU 1 + 0.02 \\|Z\\| (fixed per GPU)"], res):
        print(f"| {lab} | {r:.1f} | {100 * (r / res[0] - 1):+.2f}% | {100 * 16 * tg / res[0]:.0f}% |")
    print(f"\nclosed form, fixed per-GPU spread: 0.02 · E[max_72 |Z|] = {100 * 0.02 * tail.expected_max_half_normal(72):.1f}% of compute")


def nvlink():
    def wl(algo):
        return {"name": "ar", "spmd": {"world": 72, "program": {"streams": [[
            {"op": "collective", "kind": "all_reduce", "bytes": 64 << 20, "group": "world", "algo": algo, "sms": 20}]]}}}
    cases = [("healthy", []),
             ("gpu0-nvsw0.0 dead (never trained)", [{"a": "gpu0", "b": "nvsw0.0"}]),
             ("gpu0-nvsw0.0 at half rate", [{"a": "gpu0", "b": "nvsw0.0", "rate_scale": 0.5}]),
             ("gpu0-nvsw0.0 down 50 µs from t = 20 µs", [{"a": "gpu0", "b": "nvsw0.0", "start_us": 20, "duration_us": 50}])]
    jobs = [(a, c) for a in ("nvls", "ring") for c in cases]
    res = pmap(lambda j: finish(wl(j[0]), net={"faults": j[1][1]}), jobs)
    print("\n## One NVLink of GPU 0 (of 18), 64 MiB all-reduce on 72 GPUs\n")
    print("| link | NVLS (µs) | vs healthy | ring (µs) | vs healthy |")
    print("|---|---|---|---|---|")
    n = len(cases)
    for i, (lab, _) in enumerate(cases):
        a, b = res[i], res[n + i]
        print(f"| {lab} | {a:.1f} | {100 * (a / res[0] - 1):+.1f}% | {b:.1f} | {100 * (b / res[n] - 1):+.1f}% |")
    print(f"\nclosed forms: dead plane {tail.dead_plane_slowdown(18):.3f}x, half-rate plane "
          f"{tail.degraded_plane_slowdown(18, 0.5):.1f}x on the bandwidth term")


def ib():
    wl = {"name": "a2a", "spmd": {"world": 64, "program": {"streams": [[
        {"op": "collective", "kind": "all_to_all", "bytes": 32 << 20, "group": "world", "sms": 20}]]}}}
    cases = [("healthy", []),
             ("leaf0-spine0 dead", [{"a": "leaf0", "b": "spine0"}]),
             ("leaf0-spine0 at half rate", [{"a": "leaf0", "b": "spine0", "rate_scale": 0.5}]),
             ("leaf0-spine0 down 200 µs from t = 50 µs", [{"a": "leaf0", "b": "spine0", "start_us": 50, "duration_us": 200}])]
    jobs = [(r, c) for r in ("adaptive", "ecmp") for c in cases]
    res = pmap(lambda j: finish(wl, system="h800_8node_ib", net={
        "fidelity": "packet", "packet": {"routing": j[0]}, "nccl": {"pxn": False}, "faults": j[1][1]}), jobs)
    print("\n## One leaf-spine link (of 64), 32 MiB all-to-all on 64 H800s, packet level\n")
    print("| link | adaptive (µs) | vs healthy | ECMP (µs) | vs healthy adaptive |")
    print("|---|---|---|---|---|")
    n = len(cases)
    for i, (lab, _) in enumerate(cases):
        a, b = res[i], res[n + i]
        print(f"| {lab} | {a:.1f} | {100 * (a / res[0] - 1):+.1f}% | {b:.1f} | {100 * (b / res[0] - 1):+.1f}% |")


EXPERIMENTS = {"stragglers": stragglers, "sync": sync, "slow": slow, "nvlink": nvlink, "ib": ib}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(EXPERIMENTS):
        EXPERIMENTS[name]()
