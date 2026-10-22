"""Front-end tests: lowering and capture produce valid workloads that simulate."""

import json

import pytest

import rackscale as rs
from rackscale import capture, lower, models

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")


def _ops(wl):
    return [o for s in wl["spmd"]["program"]["streams"] for o in s]


def test_decode_has_moe_all_to_all():
    wl = lower.decode_step(models.DEEPSEEK_V3, rs.Parallel(ep=8), batch=16, context=1024, layers=4)
    kinds = [o.get("kind") for o in _ops(wl) if o["op"] == "collective"]
    assert kinds.count("all_to_all") == 2  # dispatch + combine for the one MoE layer
    assert wl["spmd"]["world"] == 8 and wl["model_flops"] > 0


def test_tp_allreduce_and_overlap_streams():
    wl = lower.decode_step(models.LLAMA3_70B, rs.Parallel(tp=8), batch=8, context=2048, layers=1, overlap=True)
    assert len(wl["spmd"]["program"]["streams"]) == 2
    assert any(o.get("kind") == "all_reduce" for o in _ops(wl))


def test_train_step_comm_stream_waits_on_events():
    wl = lower.train_step(models.LLAMA3_8B, rs.Parallel(dp=8), micro_batch=1, seqlen=512, layers=2)
    compute, comm = wl["spmd"]["program"]["streams"]
    recs = {o["event"] for o in compute if o["op"] == "record"}
    waits = {o["event"] for o in comm if o["op"] == "wait"}
    assert recs == waits and len(recs) == 2


def _fake_kineto():
    ev = [
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 5, "args": {"External id": 1, "Input Dims": [[256, 4096], [4096, 4096]]}},
        {"ph": "X", "cat": "kernel", "name": "sm90_xmma_gemm_bf16", "ts": 1, "dur": 20, "args": {"External id": 1, "stream": 7, "grid": [132, 1, 1]}},
        {"ph": "X", "cat": "kernel", "name": "triton_poi_fused_silu", "ts": 30, "dur": 4, "args": {"stream": 7, "grid": [64, 1, 1]}},
        {"ph": "X", "cat": "cpu_op", "name": "record_param_comms", "ts": 40, "dur": 1, "args": {"External id": 2, "In msg nelems": 1 << 20, "dtype": "BFloat16"}},
        {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "ts": 41, "dur": 30, "args": {"External id": 2, "stream": 20}},
    ]
    return {"traceEvents": ev}


def test_capture_lowers_gemm_nccl_and_fixed():
    wl = capture.from_torch_profiler(_fake_kineto(), world=8)
    ops = _ops(wl)
    assert [o["kernel"]["type"] for o in ops if o["op"] == "kernel"] == ["gemm", "fixed"]
    coll = [o for o in ops if o["op"] == "collective"]
    assert coll[0]["kind"] == "all_reduce" and coll[0]["bytes"] == 2 << 20
    assert ops[0]["kernel"]["m"] == 256


def test_capture_replay_keeps_durations():
    wl = capture.from_torch_profiler(_fake_kineto(), mode="replay")
    k = [o for o in _ops(wl) if o["op"] == "kernel"]
    assert k[0]["kernel"] == {"type": "fixed", "ns": 20000.0, "sms": 132}


@native
def test_simulate_captured_trace(tmp_path):
    rep = rs.simulate("hgx_h100_8", capture.from_torch_profiler(_fake_kineto(), world=8))
    assert rep.makespan_us > 0 and rep["collectives"]


@native
def test_fast_mode_matches_exact():
    wl = lower.decode_step(models.LLAMA3_8B, rs.Parallel(tp=2), batch=8, context=512, layers=1)
    exact = rs.simulate("hgx_h100_8", wl).makespan_us
    fast = rs.simulate("hgx_h100_8", wl, fast=True).makespan_us
    assert abs(fast - exact) / exact < 0.10, (fast, exact)


@native
def test_kernel_api():
    r = rs.kernel("h100_sxm", "gemm", 2048, 2048, 2048)
    assert 300 < r["tflops"] < 989


# --- parallelism checks, memory, strict workloads, pipeline parallelism ------

def test_parallel_check_rejects_bad_layouts():
    with pytest.raises(lower.ParallelError, match="does not divide"):
        lower.decode_step(models.LLAMA3_70B, rs.Parallel(tp=3), batch=8, context=1024, layers=1)
    with pytest.raises(lower.ParallelError, match="dense"):
        lower.decode_step(models.LLAMA3_8B, rs.Parallel(ep=8), batch=8, context=1024, layers=1)
    with pytest.raises(lower.ParallelError, match="more stages"):
        lower.train_step(models.LLAMA3_8B, rs.Parallel(pp=64), 1, 512)
    # DeepSeek-V3's 256 experts over 72 GPUs is a real deployment (redundant experts).
    lower.decode_step(models.DEEPSEEK_V3, rs.Parallel(ep=72), batch=8, context=1024, layers=1)


def test_memory_counts_full_model():
    mem = lower.memory(models.LLAMA3_70B, rs.Parallel(tp=8), phase="decode", batch=32, context=4096)
    # 70.6B params over 8 GPUs in BF16 ~ 17.6 GB; KV: 80 layers x 1 KV head x 128 x 2 (K,V) x 2 B per token.
    assert 17e9 < mem["weights"] < 18.5e9
    assert mem["kv_cache"] == pytest.approx(32 * 4096 * 80 * 2 * 128 * 2)
    wl = lower.decode_step(models.LLAMA3_70B, rs.Parallel(tp=8), batch=32, context=4096, layers=1)
    assert wl["memory"] == mem  # not shrunk by layers=1
    tr = lower.memory(models.LLAMA3_8B, rs.Parallel(dp=8), phase="train", batch=1, seqlen=4096)
    assert tr["optimizer"] == pytest.approx(6 * tr["weights"] / 8)  # 12 B/param vs 2 B/param, over dp


@native
def test_report_warns_when_model_does_not_fit():
    wl = lower.decode_step(models.DEEPSEEK_V3, rs.Parallel(ep=8), batch=8, context=1024, layers=1)
    rep = rs.simulate("hgx_h100_8", wl, fast=True)
    assert rep.memory["total_gb"] > rep.memory["capacity_gb"]
    assert rep.warnings and "would not fit" in rep.warnings[0]
    ok = rs.simulate("hgx_h100_8", lower.decode_step(models.LLAMA3_8B, rs.Parallel(tp=8), 8, 1024, layers=1), fast=True)
    assert ok.warnings == [] and ok.memory["total_gb"] < 80


@native
def test_unknown_workload_key_is_an_error():
    wl = {"name": "t", "spmd": {"world": 1, "program": {"streams": [[
        {"op": "kernel", "kernel": {"type": "gemm", "m": 64, "n": 64, "k": 64, "dtpye": "fp8"}}]]}}}
    with pytest.raises(ValueError, match="did you mean `dtype`"):
        rs.simulate("hgx_h100_8", wl)


@native
def test_unknown_system_key_is_an_error():
    bad = "preset: hgx_h100_8\ngpus:\n  - preset: h100_sxm\n    count: 8\n    overrides: {sm_clok_mhz: 1000}\n"
    with pytest.raises(ValueError, match="sm_clock_mhz"):
        rs.simulate(bad, lower.decode_step(models.LLAMA3_8B, rs.Parallel(), 1, 128, layers=1))


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b", "zb"])
def test_pipeline_orders_are_complete(schedule):
    for stage in range(4):
        order = lower._pp_order(schedule, 4, stage, 6)
        for kind in "FBW":
            assert sorted(i for k, i in order if k == kind) == list(range(6))
        pos = {(k, i): n for n, (k, i) in enumerate(order)}
        assert all(pos[("F", i)] < pos[("B", i)] < pos[("W", i)] for i in range(6))
    # 1F1B bounds in-flight micro-batches on stage 0 to pp.
    o = lower._pp_order("1f1b", 4, 0, 6)
    live = peak = 0
    for k, _ in o:
        live += (k == "F") - (k == "B")
        peak = max(peak, live)
    assert peak == 4


def test_pipeline_workload_wiring():
    wl = lower.train_step(models.LLAMA3_8B, rs.Parallel(pp=2, dp=2), 1, 512, layers=4, micro_batches=3,
                          optimizer=True)
    assert len(wl["ranks"]) == 4
    ops = lambda r: [o for s in wl["ranks"][r]["streams"] for o in s]
    # stage 0 (ranks 0, 1) sends to stage 1 (ranks 2, 3), 3 activations each way.
    sends0 = [o for o in ops(0) if o["op"] == "send"]
    assert [o["peer"] for o in sends0] == [2] * 3
    assert sum(o["op"] == "recv" for o in ops(3)) == 3
    rs_groups = {tuple(o["group"]) for o in ops(2) if o.get("kind") == "reduce_scatter"}
    assert rs_groups == {(2, 3)}
    assert any(o.get("name") == "adam_step" for o in ops(0))


@native
def test_pipeline_bubble_and_zero_bubble():
    def run(schedule, m):
        wl = lower.train_step(models.LLAMA3_8B, rs.Parallel(pp=4), 1, 1024, layers=8, micro_batches=m,
                              schedule=schedule)
        return rs.simulate("hgx_h100_8", wl, fast=True, launch_gap_ns=0).makespan_us
    one = run("1f1b", 1)   # one micro-batch: every stage in turn, (p-1)/p bubble
    eight = run("1f1b", 8)
    # m=8, p=4: ideal 8/11 of (m+p-1) slots -> step ~ (8+3)/4 of the m=1 time.
    assert 2.4 < eight / one < 3.0, eight / one
    assert run("zb", 8) < eight
