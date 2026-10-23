"""GTA / GLA attention, linear-attention mixers, context-parallel prefill and the long-context fast path."""

import pytest

import rackscale as rs
from rackscale import extrap, lower, models

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")


def _kernels(wl, rank=0):
    prog = wl["spmd"]["program"] if "spmd" in wl else wl["ranks"][rank]
    return [o for s in prog["streams"] for o in s if o.get("op") == "kernel"]


def test_cache_per_token_follows_the_paper():
    m70 = models.ALL
    # GQA-8 2048, GTA-8 8*128 + 64, MLA 512 + 64, GLA-2 2*256 + 64 elements per token per layer.
    assert m70["llama3-70b"].kv_elems_per_token_layer() == 2048
    assert m70["llama3-70b-gta8"].kv_elems_per_token_layer() == 1088
    assert m70["llama3-70b-mla"].kv_elems_per_token_layer() == 576
    assert m70["llama3-70b-gla2"].kv_elems_per_token_layer() == 576
    # Under TP8, MLA's single latent is replicated; GLA keeps one of its two latents.
    assert m70["llama3-70b-mla"].kv_elems_per_token_layer(8) == 576
    assert m70["llama3-70b-gla2"].kv_elems_per_token_layer(8) == 320


def test_gla_and_gta_lower_to_tied_attention():
    for name, d, hkv in (("llama3-70b-gla2", 320, 1), ("llama3-70b-gta8", 128, 1)):
        wl = lower.decode_step(models.ALL[name], rs.Parallel(tp=8), 4, 1024, layers=1)
        att = [o["kernel"] for o in _kernels(wl) if o["kernel"]["type"] == "attention"]
        assert len(att) == 1 and att[0]["kv_tied"] and att[0]["d"] == d and att[0]["heads_kv"] == hkv
    with pytest.raises(lower.ParallelError):
        lower.check(models.replace(models.ALL["llama3-70b-gla2"], gla=models.GLA(latent_heads=3)), rs.Parallel(tp=2))


def test_linear_attention_uses_the_ssm_kernels():
    m = models.ALL["gdn-8b"]
    dec = [o["kernel"] for o in _kernels(lower.decode_step(m, rs.Parallel(), 2, 128, layers=1))]
    su = [k for k in dec if k["type"] == "ssm_state_update"]
    assert su and su[0]["d_state"] == 128 and su[0]["headdim"] == 128 and su[0]["heads"] == 32
    pre = [o["name"] for o in _kernels(lower.prefill(m, rs.Parallel(), 1, 1024, layers=1))]
    assert "delta_kkt" in pre and "lin_chunk_scan" in pre
    gla = [o["name"] for o in _kernels(lower.prefill(models.ALL["gla-8b"], rs.Parallel(), 1, 1024, layers=1))]
    assert "delta_kkt" not in gla
    # A constant-size state: no KV cache growth for a pure linear-attention stack.
    assert m.cache_bytes_per_seq(1 << 20) == m.cache_bytes_per_seq(1024)


def test_prefill_cp_ring_and_state_chain():
    ring = lower.prefill_cp(models.ALL["llama3-8b"], 4, 4 * 1024, layers=1)
    r1 = ring["ranks"][1]["streams"]
    assert sum(o["op"] == "send" for o in r1[1]) == 3 and sum(o["op"] == "recv" for o in r1[2]) == 3
    assert sum(o.get("name") == "l0.ring_step" for o in r1[0]) == 3
    ssm = lower.prefill_cp(models.ALL["mamba2-8b"], 4, 4 * 1024, layers=1)
    first, last = ssm["ranks"][0]["streams"], ssm["ranks"][-1]["streams"]
    assert not first[2] and not last[1]  # rank 0 receives nothing, the last rank sends nothing
    assert any(o.get("name") == "l0.state_correct" for o in last[0])


def test_extrap_rules_are_selected_by_size():
    small = {"type": "attention", "batch": 1, "heads_q": 32, "heads_kv": 8, "sq": 4096, "skv": 4096, "d": 128}
    big = dict(small, sq=1 << 20, skv=1 << 20)
    assert not extrap.is_long(small) and extrap.is_long(big)
    assert extrap.is_long({"type": "gemm", "m": 1 << 18, "n": 4096, "k": 4096})
    assert not extrap.is_long({"type": "gemm", "m": 64, "n": 4096, "k": 4096})


@native
def test_prefill_cp_simulates_and_state_chain_pipelines():
    wl = lower.prefill_cp(models.ALL["mamba2-8b"], 4, 4 * 512, layers=2)
    rep = rs.simulate("hgx_b300_8", wl, fast=True).raw
    busy = [r["gpu_busy_us"] for r in rep["ranks"]]
    assert len(busy) == 4 and rep["makespan_us"] > max(busy) * 0.9
