import pytest

import rackscale as rs
from rackscale import recsys
from rackscale.lower import ParallelError

SMALL = recsys.DLRM("tiny", tables=8, rows=1_000_000, dim=64, pooling=8, dense_in=64, bottom=(64, 64),
                    top=(128, 1))


def test_balance_closed_forms():
    m = recsys.DLRM_2TB
    b = recsys.balance(m, 64, 4096, hbm_gbps=6600, net_gbps=630)
    assert b["lookup_bytes"] == 4096 * 256 * 32 * 128 * 2
    assert b["a2a_bytes"] == pytest.approx(4096 * 256 * 128 * 2 * 63 / 64)
    assert b["lookup_bytes"] / b["a2a_bytes"] == pytest.approx(32 * 64 / 63)  # L times more through HBM
    ib = recsys.balance(m, 64, 4096, 6600, 100, domain=8)
    assert ib["a2a_us"] > 5 * b["a2a_us"]
    assert m.embedding_bytes() / 1e12 == pytest.approx(2.62, abs=0.01)


def test_step_shards_tables_and_trains():
    wl = recsys.step(SMALL, 4, 128)
    ops = wl["spmd"]["program"]["streams"][0]
    look = next(o for o in ops if o.get("name") == "embedding_lookup")["kernel"]
    assert look["bags"] == 128 * 4 * 2 and look["table_rows"] == 2 * 1_000_000
    upd = next(o for o in ops if o.get("name") == "embedding_update")["kernel"]
    assert upd["update"]
    a2a = [o for o in ops if o.get("kind") == "all_to_all"]
    assert len(a2a) == 2 and a2a[0]["bytes"] == 128 * 8 * 64 * 2
    assert not [o for o in recsys.step(SMALL, 4, 128, train=False)["spmd"]["program"]["streams"][0]
                if o.get("name") == "embedding_update"]
    with pytest.raises(ParallelError):
        recsys.step(SMALL, 3, 128)


def test_step_simulates_and_the_gather_is_memory_bound():
    r = rs.simulate("hgx_b300_8", recsys.step(SMALL, 8, 512), fast=True)
    k = next(k for k in r.kernels(top=50) if k["name"] == "embedding_bag")
    assert k["tensor_util"] == 0 and k["gbps"] > 1000
