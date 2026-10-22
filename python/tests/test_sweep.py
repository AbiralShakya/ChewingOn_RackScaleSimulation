"""rackscale.sweep: plans, lowering, bottleneck attribution and the plan CLI."""

import json

import pytest

import rackscale as rs
from rackscale import lower, models
from rackscale import sweep as sw
from rackscale.cli import main as cli

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")


def test_inference_plans_respect_heads_and_experts():
    dense = sw.inference_plans("llama3.2-3b", "gb300")  # 24 heads: no TP16
    assert {p["tp"] for p in dense} == {1, 2, 4, 8}
    moe = sw.inference_plans("mixtral-8x7b", "gb300")
    assert all(p["ep"] <= 8 for p in moe) and any(p["ep"] == 8 for p in moe)


def test_train_plans_use_every_gpu():
    for p in sw.train_plans("llama3-70b", "gb300"):
        assert p["tp"] * p["pp"] * p["dp"] == 72


def test_mixed_step_rows_and_attention():
    wl = lower.mixed_step(models.LLAMA3_8B, rs.Parallel(), decode_batch=16, context=2048, chunk=512, layers=1)
    ks = [o["kernel"] for o in wl["spmd"]["program"]["streams"][0] if o["op"] == "kernel"]
    att = [k for k in ks if k["type"] == "attention"]
    assert sorted(k["sq"] for k in att) == [1, 512]
    assert any(k["type"] == "gemm" and k["m"] == 528 for k in ks)


def test_fp8_rewrites_gemms_only():
    wl = sw.workload(sw.Point("llama3-8b", "b300", "decode", batch=4, context=128, dtype="fp8"))
    ks = [o["kernel"] for o in wl["spmd"]["program"]["streams"][0] if o["op"] == "kernel"]
    assert all(k["dtype"] == "fp8" for k in ks if k["type"] == "gemm")
    assert wl["memory"]["weights"] < 9e9  # one byte per parameter


def test_too_big_is_recorded_not_simulated():
    r = sw.run_point(sw.Point("llama3-405b", "a100", "decode", sw.plan(tp=8), batch=1, context=1024))
    assert r["fits"] is False and "step_us" not in r


@native
def test_breakdown_sums_to_step_and_cache_is_exact():
    pt = sw.Point("llama3.2-1b", "b300", "decode", batch=1, context=256)
    a = sw.run_point(pt, cache=False)
    b = sw.run_point(pt, cache=True)
    c = sw.run_point(pt, cache=True)  # served from the process-wide memo
    assert abs(sum(a["breakdown_us"].values()) - a["step_us"]) < 1e-6 * a["step_us"] + 1e-3
    assert a["step_us"] == b["step_us"] == c["step_us"]
    assert a["dominant"] in a["breakdown_us"]
    assert a["tok_s_user"] == pytest.approx(1e6 / a["step_us"])


@native
def test_plan_cli_json(capsys):
    assert cli(["plan", "llama3.2-1b", "--system", "b300", "--context", "256", "--batches", "1,4", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["best"]["phase"] == "decode" and out["best"]["batch"] == 4
    assert {p["plan"]["tp"] for p in out["points"]} == {1, 2, 4, 8}


def test_moe_small_batch_runs_only_visited_experts():
    # 8 tokens x top-8 over 128 experts: 64 experts get one token each, none get zero.
    wl = lower.decode_step(models.QWEN3_30B_A3B, rs.Parallel(), batch=8, context=256, layers=1)
    gg = [o["kernel"] for o in wl["spmd"]["program"]["streams"][0] if o["op"] == "kernel" and o["kernel"]["type"] == "grouped_gemm"]
    assert gg and all(k["ms"] == [1] * 64 for k in gg)


def test_moe_tp_shards_experts():
    m = models.MIXTRAL_8X22B
    one = lower.memory(m, rs.Parallel(), phase="decode", batch=1, context=1)["weights"]
    tp8 = lower.memory(m, rs.Parallel(tp=8), phase="decode", batch=1, context=1)["weights"]
    assert tp8 < one / 7
    wl = lower.decode_step(m, rs.Parallel(tp=8), batch=8, context=256, layers=1)
    ops = wl["spmd"]["program"]["streams"][0]
    gg = [o["kernel"] for o in ops if o["op"] == "kernel" and o["kernel"]["type"] == "grouped_gemm"]
    assert gg[1]["k"] == m.moe.expert_ffn // 8
    assert sum(1 for o in ops if o.get("kind") == "all_reduce") == 2
