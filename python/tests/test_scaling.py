"""rackscale.scaling and the ZeRO / FSDP / HSDP / sequence-parallel / Ulysses lowerings."""

import pytest

import rackscale as rs
from rackscale import lower, models
from rackscale import scaling as sc

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")

M = models.LLAMA3_8B.with_layers(2)


def _hw(**kw) -> sc.Hardware:
    base = dict(name="test", gpu="test", gpus=72, domain=72, flops={"bf16": 2.0e15}, hbm_bw=8.0e12, hbm_bytes=288e9,
                nvlink_bw=900e9, nvls=True, ib_bw=0.0, alpha=0.0, alpha_p2p=0.0)
    base.update(kw)
    return sc.Hardware(**base)


def _colls(wl, rank=0):
    streams = wl["spmd"]["program"]["streams"] if "spmd" in wl else wl["ranks"][rank]["streams"]
    return [o for s in streams for o in s if o["op"] == "collective"]


def test_collective_closed_forms():
    hw = _hw(nvls=False)
    b = 1e9
    ar = sc.collective("all_reduce", b, 8, hw)[0]
    rs_ = sc.collective("reduce_scatter", b, 8, hw)[0]
    ag = sc.collective("all_gather", b, 8, hw)[0]
    # A ring all-reduce is a reduce-scatter then an all-gather: 2 (n-1)/n B / W.
    assert ar == pytest.approx(rs_ + ag) == pytest.approx(2 * 7 / 8 * b / 900e9)
    # NVLS halves it at large n: B / W.
    assert sc.collective("all_reduce", b, 8, _hw())[0] == pytest.approx(b / 900e9)
    # Past the NVLink domain the NIC sets the pace.
    ib = _hw(domain=8, ib_bw=50e9, nvls=False)
    assert sc.collective("all_gather", b, 16, ib)[0] == pytest.approx(15 / 16 * b / 50e9)
    assert sc.collective("all_reduce", b, 1, hw)[0] == 0.0


def test_kernel_roofline():
    hw = _hw()
    big = dict(type="gemm", m=8192, n=8192, k=8192, dtype="bf16")
    assert sc.kernel_time(big, hw) == pytest.approx(2 * 8192**3 / 2e15)  # compute bound
    skinny = dict(type="gemm", m=1, n=8192, k=8192, dtype="bf16")
    assert sc.kernel_time(skinny, hw) == pytest.approx((8192 + 8192 * 8192 + 8192) * 2 / 8e12, rel=1e-3)
    # Causal attention over a square block does (S + 1) / 2S of the full work.
    f_full, _ = sc.kernel_work(dict(type="attention", batch=1, heads_q=8, heads_kv=8, sq=1024, skv=1024, d=128))
    f_causal, _ = sc.kernel_work(dict(type="attention", batch=1, heads_q=8, heads_kv=8, sq=1024, skv=1024, d=128,
                                      causal=True))
    assert f_causal / f_full == pytest.approx(1025 / 2048)


def test_default_parallel_name_unchanged():
    assert repr(lower.Parallel(tp=2, dp=4)) == "Parallel(tp=2, ep=1, dp=4, comm_sms=20, pp=1)"
    assert repr(lower.Parallel(dp=4, zero=3, shard=2)).endswith("zero=3, shard=2)")


def test_zero_stages_communication():
    ddp = _colls(lower.train_step(M, lower.Parallel(dp=8, zero=0), 1, 1024))
    z2 = _colls(lower.train_step(M, lower.Parallel(dp=8, zero=2), 1, 1024))
    z3 = _colls(lower.train_step(M, lower.Parallel(dp=8, zero=3), 1, 1024))
    assert {c["kind"] for c in ddp} == {"all_reduce"}
    assert {c["kind"] for c in z2} == {"reduce_scatter"}
    # FSDP: every layer gathered for forward and again for backward, then scattered.
    assert sum(c["kind"] == "all_gather" for c in z3) == 2 * 2 and sum(c["kind"] == "reduce_scatter" for c in z3) == 2
    # Same gradient bytes at every stage; FSDP adds 2x the parameter bytes in gathers.
    g = sum(c["bytes"] for c in ddp)
    assert sum(c["bytes"] for c in z2) == g
    assert sum(c["bytes"] for c in z3 if c["kind"] == "all_gather") == 2 * g


def test_zero_memory():
    mem = {z: lower.memory(models.LLAMA3_8B, lower.Parallel(dp=8, zero=z), phase="train", batch=1, seqlen=4096)
           for z in range(4)}
    assert mem[0]["optimizer"] == 8 * mem[1]["optimizer"]
    assert mem[1]["grads"] == 8 * mem[2]["grads"]
    assert mem[3]["weights"] < mem[2]["weights"] / 4


def test_hsdp_groups():
    p = lower.Parallel(tp=2, dp=4, zero=3, shard=2)
    wl = lower.train_step(M, p, 1, 1024)
    cs = _colls(wl, rank=3)  # d = 1, t = 1
    ag = next(c for c in cs if c["kind"] == "all_gather")
    ar = next(c for c in cs if c["name"].startswith("grad_ar"))
    assert ag["group"] == [1, 3] and ar["group"] == [3, 7]
    # Without TP the groups are expressible as block / stride.
    named = _colls(lower.train_step(M, lower.Parallel(dp=8, zero=3, shard=4), 1, 1024))
    assert {c["group"] for c in named if c["op"] == "collective"} == {"block:4", "stride:4"}
    with pytest.raises(lower.ParallelError):
        lower.check(M, lower.Parallel(dp=8, zero=2, shard=4))


def test_fsdp_with_tp_keeps_tp_groups():
    # TP collectives stay "block:tp"; the FSDP gathers go over the ranks with the same TP index.
    wl = lower.train_step(M, lower.Parallel(tp=8, dp=2, zero=3), 1, 1024)
    groups = {(c["kind"], c["group"]) for c in _colls(wl)}
    assert ("all_reduce", "block:8") in groups and ("all_gather", "stride:8") in groups
    assert all(isinstance(g, str) for _, g in groups)
    ddp = lower.train_step(M, lower.Parallel(tp=2, dp=4, zero=0), 1, 1024)
    assert {c["group"] for c in _colls(ddp) if c["kind"] == "all_reduce"} == {"block:2", "stride:2"}


def test_sequence_parallel_lowering():
    wl = lower.prefill(M, lower.Parallel(tp=4, sp=True), 1, 4096)
    ops = wl["spmd"]["program"]["streams"][0]
    kinds = [o["kind"] for o in ops if o["op"] == "collective"]
    assert kinds.count("all_gather") == kinds.count("reduce_scatter") == 4 and "all_reduce" not in kinds
    norms = [o["kernel"] for o in ops if o["op"] == "kernel" and o["kernel"]["type"] == "row_norm"]
    assert all(k["rows"] == 1024 for k in norms)


def test_ulysses_lowering():
    wl = lower.prefill_cp(M, 8, 32768, kind="ulysses")
    ops = wl["spmd"]["program"]["streams"][0]
    assert wl["spmd"]["world"] == 8
    assert sum(o["op"] == "collective" and o["kind"] == "all_to_all" for o in ops) == 4
    att = next(o["kernel"] for o in ops if o["op"] == "kernel" and o["kernel"]["type"] == "attention")
    assert (att["heads_q"], att["heads_kv"], att["sq"]) == (4, 1, 32768)


def test_replay_matches_pipeline_formula():
    # Uniform stages and free transfers: the replay must give (m + p - 1) (t_F + t_B).
    hw = _hw(alpha_p2p=0.0, nvlink_bw=1e30)
    p, m = 4, 8
    wl = lower.train_step(models.LLAMA3_8B.with_layers(4), lower.Parallel(pp=p), 1, 1024, micro_batches=m,
                          schedule="gpipe")
    r = sc.replay(wl, hw)
    one = lower.train_step(models.LLAMA3_8B.with_layers(1), lower.Parallel(), 1, 1024, micro_batches=1)
    stage = sc.replay(one, hw).step
    assert r.step == pytest.approx((m + p - 1) * stage, rel=0.02)


def test_estimate_decode_critical_batch_and_world():
    hw = _hw()
    est = sc.estimate("llama3-70b", hw, "decode", sc.Plan(tp=8), batch=64, context=8192)
    assert est["critical"]["batch"] == pytest.approx(2e15 * 2 / (2 * 8e12))
    assert est["world"] == 8 and est["replicas"] == 9
    ds = sc.estimate("deepseek-v3", hw, "decode", sc.Plan(ep=72), batch=64, context=4096)
    assert ds["world"] == 72
    assert any(t["name"].startswith("EP") for t in ds["terms"])


def test_estimate_train_terms():
    hw = _hw()
    est = sc.estimate("llama3-70b", hw, "train", sc.Plan(tp=8, pp=4, dp=2, schedule="1f1b"), micro_batch=1,
                      seqlen=4096, micro_batches=16)
    names = [t["name"] for t in est["terms"]]
    assert "pipeline bubble" in names and "TP activations" in names
    bubble = next(t for t in est["terms"] if t["name"] == "pipeline bubble")
    assert bubble["fraction"] == pytest.approx(3 / (16 + 3), rel=1e-6)
    zb = sc.estimate("llama3-70b", hw, "train", sc.Plan(tp=8, pp=4, dp=2, schedule="zb"), micro_batch=1,
                     seqlen=4096, micro_batches=16)
    assert zb["step_us"] < est["step_us"]
    fsdp = sc.estimate("llama3-70b", hw, "train", sc.Plan(tp=8, dp=9, zero=3), micro_batch=1, seqlen=4096)
    assert fsdp["memory_gb"]["weights"] < est["memory_gb"]["weights"]


@native
def test_check_runs_against_simulator():
    r = sc.check("llama3-8b", "gb300", "decode", sc.Plan(tp=2), layers=1, batch=8, context=1024)
    assert r["replay_us"] > 0 and r["sim_us"] > 0
    # The simulator can only be slower than the ideal machine.
    assert r["err_vs_replay"] > -0.05
