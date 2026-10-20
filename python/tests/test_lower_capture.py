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
