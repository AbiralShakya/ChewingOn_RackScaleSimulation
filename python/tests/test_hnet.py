"""rackscale.hnet: byte-level hierarchical models (H-Net) and their baselines."""

import pytest

from rackscale import hnet
from rackscale.lower import Parallel


def test_published_sizes():
    # H-Net paper Table 1: 1-stage XL 1.3B, 2-stage XL 1.6B parameters.
    assert hnet.HNET_1STAGE_XL.total_params() / 1e9 == pytest.approx(1.3, abs=0.06)
    assert hnet.HNET_2STAGE_XL.total_params() / 1e9 == pytest.approx(1.6, abs=0.06)
    # Table 2: about 0.7 GFLOP per byte at XL scale, close to the BPE Transformer.
    for h in hnet.ALL.values():
        assert 0.6e9 < hnet.flops_per_byte(h, 8192) < 0.8e9


def test_prefill_shapes_follow_compression():
    h = hnet.HNET_2STAGE_XL
    wl = hnet.prefill(h, Parallel(), 1, 8192)
    ops = wl["spmd"]["program"]["streams"][0]
    att = [o["kernel"] for o in ops if o["op"] == "kernel" and o["kernel"]["type"] == "attention"]
    main = [k for k in att if k["sq"] == round(round(8192 / 3.0) / 2.3)]
    assert len(main) == 27  # one per main-network layer
    ssd = [o for o in ops if o["op"] == "kernel" and o["kernel"]["type"] == "ssd"]
    assert {o["kernel"]["seqlen"] for o in ssd} == {8192, round(8192 / 3.0)}


def test_decode_runs_main_network_only_when_routers_fire():
    h = hnet.HNET_1STAGE_XL
    counts = hnet.step_counts(h, 64, 8, seed=1)
    assert len(counts) == 8 and all(0 <= c[0] <= 64 for c in counts)
    wl = hnet.decode_steps(h, Parallel(), 64, 8192, counts=counts)
    assert set(wl) == {"name", "spmd"}  # a plain workload the simulator accepts
    ops = wl["spmd"]["program"]["streams"][0]
    main_qkv = [o for o in ops if o.get("name", "").endswith("main.qkv_proj")]
    assert sum(1 for c in counts if c[-1]) * 24 == len(main_qkv)
    assert hnet.decode_steps(h, Parallel(), 64, 8192, steps=8, seed=1) == wl


def test_mean_field_and_even_counts_keep_the_mean():
    h = hnet.HNET_2STAGE_XL
    mf = hnet.step_counts(h, 64, 30, mean_field=True)
    # 64 / 3 and 64 / 3 / 2.3 sequences per step on average, in whole sequences.
    assert sum(c[0] for c in mf) == 640 and sum(c[1] for c in mf) == int(64 / 3 / 2.3 * 30)
    assert all(c[0] in (21, 22) and c[1] in (9, 10) for c in mf)
    drawn = hnet.step_counts(h, 64, 30, seed=3)
    even = hnet.even_counts(drawn)
    for j in range(2):
        assert sum(c[j] for c in even) == sum(c[j] for c in drawn)
        assert max(c[j] for c in even) - min(c[j] for c in even) <= 1
    # A step where no router fires skips the inner stages entirely.
    wl = hnet.decode_steps(h, Parallel(), 4, 1024, counts=[[0, 0], [1, 0]])
    names = [o.get("name", "") for o in wl["spmd"]["program"]["streams"][0]]
    assert not any(n.startswith("t0.s1") or ".main." in n for n in names)


def test_tp_lowering():
    wl = hnet.prefill(hnet.HNET_1STAGE_XL, Parallel(tp=2), 1, 4096, layers=2)
    assert any(o["op"] == "collective" for o in wl["spmd"]["program"]["streams"][0])


def test_decode_traffic_closed_form():
    # Book chapter 13.2: bytes one decode step moves at batch 64, 8 KiB of context.
    bpe = hnet.decode_traffic(hnet.BPE_XL, 64, 8192)
    assert bpe["weights"] / 1e9 == pytest.approx(2.39, abs=0.01)
    assert bpe["KV"] / 1e9 == pytest.approx(23.16, abs=0.01)
    assert bpe["per_byte"] == pytest.approx(bpe["step"] / hnet.BYTES_PER_BPE_TOKEN)
    one = hnet.decode_traffic(hnet.HNET_1STAGE_XL, 64, 8192)
    two = hnet.decode_traffic(hnet.HNET_2STAGE_XL, 64, 8192)
    assert one["per_byte"] / bpe["per_byte"] == pytest.approx(1.32, abs=0.005)
    assert two["per_byte"] / bpe["per_byte"] == pytest.approx(1.22, abs=0.005)
    assert one["outer attention KV"] == 0 and two["outer attention KV"] > 0
    # S/R^2 positions per byte against S/4.45^2: about 0.90x BPE's KV per byte.
    assert one["main KV"] / (bpe["KV"] / hnet.BYTES_PER_BPE_TOKEN) == pytest.approx((4.45 / 4.7) ** 2)
    # The re-read weights amortise over the batch, so the ratio falls with it.
    ratios = [hnet.decode_traffic(hnet.HNET_1STAGE_XL, b, 8192)["per_byte"]
              / hnet.decode_traffic(hnet.BPE_XL, b, 8192)["per_byte"] for b in (16, 64, 256)]
    assert ratios == sorted(ratios, reverse=True)
