"""Generic capture: events.jsonl -> capture.json -> workload with a parallelism plan."""

import json
import os

import pytest

import rackscale as rs
from rackscale import record, replay

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")

GEMM = "_Z4gemmPKfS0_Pf"
NORM = "_Z7rmsnormPKfPf"


def _write_events(d, pid=4242):
    """A two-layer step on two streams, as the NVBit tool and the site package would log it."""
    p = os.path.join(d, "graph", f"p{pid}")
    os.makedirs(p)
    ev = [
        {"e": "D", "name": "NVIDIA B300 SXM6 AC", "cc": [10, 3], "sms": 148, "clock_khz": 2032000, "pid": pid,
         "mode": "graph"},
        # launches before the region are ignored
        {"e": "F", "id": 0, "m": GEMM, "n": "void gemm(const float*, const float*, float*)", "r": 168, "ss": 0},
        {"e": "L", "i": 0, "f": 0, "s": "0x0", "g": [148, 1, 1], "b": [256, 1, 1], "c": [1, 1, 1], "sm": 0, "h": 1},
        {"e": "M", "k": "region_begin", "p": "", "i": 1, "h": 0},
        {"e": "M", "k": "step", "p": "LlamaForCausalLM", "i": 1, "h": 5},
        {"e": "F", "id": 1, "m": NORM, "n": "void rmsnorm(const float*, float*)", "r": 32, "ss": 0},
    ]
    seq, t = 1, 1000
    for layer in range(2):
        ev.append({"e": "M", "k": "push", "p": f"model.layers.{layer}\tLlamaDecoderLayer", "i": seq, "h": t})
        ev.append({"e": "L", "i": seq, "f": 1, "s": "0x0", "g": [64, 1, 1], "b": [128, 1, 1], "c": [1, 1, 1],
                   "sm": 0, "h": t})
        ev.append({"e": "T", "i": seq, "a": t, "b": t + 2000})
        seq, t = seq + 1, t + 3000
        ev.append({"e": "M", "k": "push", "p": f"model.layers.{layer}.mlp.down_proj\tRowParallelLinear", "i": seq,
                   "h": t})
        ev.append({"e": "L", "i": seq, "f": 0, "s": "0x0", "g": [148, 1, 1], "b": [256, 1, 1], "c": [1, 1, 1],
                   "sm": 0, "h": t})
        ev.append({"e": "T", "i": seq, "a": t, "b": t + 10000})
        ev.append({"e": "M", "k": "pop", "p": "64x2048\tbfloat16\t262144", "i": seq + 1, "h": t + 1})
        ev.append({"e": "M", "k": "pop", "p": "64x2048\tbfloat16\t262144", "i": seq + 1, "h": t + 2})
        seq, t = seq + 1, t + 12000
    # a side stream that waits on the main stream, then a copy back
    ev += [
        {"e": "R", "ev": "0xe1", "s": "0x0", "h": t},
        {"e": "W", "ev": "0xe1", "s": "0x77", "h": t + 1},
        {"e": "L", "i": seq, "f": 1, "s": "0x77", "g": [8, 1, 1], "b": [128, 1, 1], "c": [1, 1, 1], "sm": 0,
         "h": t + 2},
        {"e": "T", "i": seq, "a": t + 500, "b": t + 1500},
        {"e": "W", "ev": "0xdead", "s": "0x77", "h": t + 3},  # recorded before the region: dropped
        {"e": "C", "i": seq + 1, "k": "DtoH", "n": 256, "s": "0x0", "h": t + 4},
        {"e": "T", "i": seq + 1, "a": t + 2000, "b": t + 4000},
        {"e": "M", "k": "region_end", "p": "", "i": seq + 2, "h": t + 5000},
    ]
    with open(os.path.join(p, "events.jsonl"), "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")


def test_build_joins_launches_into_kernels(tmp_path):
    _write_events(str(tmp_path))
    cap = record.load(record.build(str(tmp_path), convert=False))
    ks = cap["kernels"]
    # 2 GEMMs, 2 norms, and a norm with a smaller grid on the side stream: a distinct configuration
    assert sorted(k["count"] for k in ks.values()) == [1, 2, 2]
    gemm = next(k for k in ks.values() if k["mangled"] == GEMM)
    assert gemm["regs"] == 168 and gemm["measured_ns"]["median"] == 10000
    ops = cap["processes"][0]["regions"][0]
    assert [o["op"] for o in ops].count("wait") == 1  # the pre-region event is already satisfied
    assert cap["processes"][0]["streams"] == 2


def test_measured_span(tmp_path):
    _write_events(str(tmp_path))
    m = replay.measured(record.build(str(tmp_path), convert=False))
    assert m["kernel_busy_us"] == pytest.approx((2 * 2000 + 2 * 10000 + 1000) / 1e3)
    assert m["gpu_span_us"] > m["kernel_busy_us"] - 1


def test_plan_inserts_tp_allreduce_after_row_parallel(tmp_path):
    _write_events(str(tmp_path))
    cap = record.load(record.build(str(tmp_path), convert=False))
    wl = replay.to_workload(cap, {"tp": 4, "dp": 2}, mode="measured")
    assert wl["spmd"]["world"] == 8
    main = wl["spmd"]["program"]["streams"][0]
    colls = [o for o in main if o["op"] == "collective"]
    assert len(colls) == 2 and all(c["kind"] == "all_reduce" and c["group"] == "block:4" for c in colls)
    assert all(c["bytes"] == 262144 for c in colls)
    # every all-reduce follows the row-parallel GEMM
    for i, o in enumerate(main):
        if o["op"] == "collective":
            assert main[i - 1]["kernel"]["ns"] == 10000


def test_pipeline_split_by_layer(tmp_path):
    _write_events(str(tmp_path))
    cap = record.load(record.build(str(tmp_path), convert=False))
    wl = replay.to_workload(cap, {"pp": 2}, mode="measured")
    r0, r1 = wl["ranks"]
    k0 = [o for o in r0["streams"][0] if o["op"] == "kernel"]
    k1 = [o for o in r1["streams"][0] if o["op"] == "kernel"]
    assert len(k0) == 2 and len(k1) == 2  # one layer (norm + GEMM) per stage
    assert r0["streams"][0][-1]["op"] == "send" and r1["streams"][0][0]["op"] == "recv"


def test_steps_and_short_names():
    assert replay.short_name("void at::native::vectorized_elementwise_kernel<8, Foo<int> >(int, Foo<int>)") \
        == "vectorized_elementwise_kernel<8, Foo<int> >"
    assert replay.short_name("nvjet_sm103_tst_128x256") == "nvjet_sm103_tst_128x256"


def test_shard_overrides():
    cfg = {"hidden_size": 8192, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672,
           "vocab_size": 128256}
    o = replay.shard_overrides(cfg, tp=16)
    assert o == {"num_attention_heads": 4, "num_key_value_heads": 1, "intermediate_size": 1792,
                 "vocab_size": 8016, "head_dim": 128}


@native
def test_replay_simulates_with_host_timeline(tmp_path):
    _write_events(str(tmp_path))
    path = record.build(str(tmp_path), convert=False)
    free = replay.replay(path, "hgx_b300_8", mode="measured", launch_gap_ns=0)
    hosted = replay.replay(path, "hgx_b300_8", mode="measured", host=True, launch_gap_ns=0)
    busy = replay.measured(path)["kernel_busy_us"]
    assert free.makespan_us >= busy - 1.0
    assert hosted.makespan_us > free.makespan_us  # the host issues launches later than the GPU could run them
    tp = replay.replay(path, "hgx_b300_8", {"tp": 8}, mode="measured", launch_gap_ns=0)
    assert tp.raw["collectives"] and tp.makespan_us > free.makespan_us
