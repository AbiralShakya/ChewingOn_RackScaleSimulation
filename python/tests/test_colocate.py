"""rackscale.colocate: several models placed on one rack."""

import pytest

import rackscale as rs
from rackscale import colocate as co
from rackscale.scaling import Plan

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")


def _tenants():
    return [co.Tenant("moe", "qwen3-30b-a3b", "decode", Plan(tp=4), gpus=range(0, 4), batch=32, context=4096, steps=2),
            co.Tenant("pp", "llama3-8b", "train", Plan(pp=2), gpus=[6, 7], micro_batches=4, seqlen=2048, steps=1),
            co.Tenant("small", "llama3.2-1b", "decode", Plan(), gpus=[4], batch=16, context=2048)]


def test_build_places_groups_peers_and_events():
    wl = co.build(_tenants(), 8, layers=2, flows=[co.Flow("kv", src=[4], dst=[0, 1], bytes=1 << 20, times=3)])
    ranks = wl["ranks"]
    assert len(ranks) == 8
    body = ranks[1]["streams"][0][0]["body"]
    colls = [o for o in body if o["op"] == "collective"]
    assert colls[0]["group"] == [0, 1, 2, 3] and colls[0]["name"].startswith("moe.")
    # Pipeline stage 0 (GPU 6) talks to stage 1 on GPU 7, not to rank 1.
    p2p = [o for s in ranks[6]["streams"] for o in s if o["op"] in ("send", "recv")]
    assert p2p and all(o["peer"] == 7 for o in p2p)
    events = {o["event"] for s in ranks[6]["streams"] for o in s if o["op"] in ("record", "wait")}
    assert events and all(e.startswith("pp.") for e in events)
    # The flow: GPU 4 sends to 0 and 1, each receives once per repetition.
    sends = [s[0] for s in ranks[4]["streams"] if s[0].get("op") == "loop" and s[0]["body"][-1]["op"] == "send"]
    assert [s["body"][-1]["peer"] for s in sends] == [0, 1] and sends[0]["times"] == 3
    assert "small.weights" in wl["memory_per_rank"][4]
    assert ranks[5]["streams"] == [[{"op": "delay", "ns": 0}]]


def test_paced_flow_issues_on_a_clock():
    f = co.Flow("kv", src=[0], dst=[1], bytes=1 << 20, times=3, every_ns=1000.0, start_ns=500.0)
    prog = f.program("send", 1)
    assert [o["at_ns"] for o in prog if o["op"] == "host"] == [500.0, 1500.0, 2500.0]
    assert sum(o["op"] == "send" for o in prog) == 3
    with pytest.raises(ValueError, match="unique"):
        co.build([co.Tenant("kv", "llama3.2-1b", "decode", Plan(), gpus=[1])], 2, layers=1, flows=[f])


def test_build_rejects_wrong_gpu_count():
    with pytest.raises(ValueError, match="needs 4 GPUs"):
        co.build([co.Tenant("x", "llama3-8b", "decode", Plan(tp=4), gpus=[0, 1])], 8, layers=1)


def test_shared_gpu_gets_both_programs():
    ts = [co.Tenant("a", "llama3.2-1b", "decode", Plan(), gpus=[0]),
          co.Tenant("b", "llama3.2-1b", "decode", Plan(), gpus=[0])]
    wl = co.build(ts, 2, layers=1)
    assert len(wl["ranks"][0]["streams"]) == 2


@native
def test_disjoint_tenants_do_not_interfere_and_shared_ones_do():
    a = co.Tenant("a", "llama3.2-1b", "decode", Plan(tp=2), gpus=[0, 1], batch=8, context=1024, steps=2)
    b = co.Tenant("b", "llama3.2-1b", "decode", Plan(tp=2), gpus=[2, 3], batch=8, context=1024, steps=2)
    res = co.run([a, b], "gb300", layers=1)
    assert all(abs(t["slowdown"]) < 0.02 for t in res["tenants"])
    b2 = co.Tenant("b", "llama3.2-1b", "decode", Plan(tp=2), gpus=[0, 1], batch=8, context=1024, steps=2)
    res = co.run([a, b2], "gb300", layers=1)
    assert all(t["slowdown"] > 0.2 for t in res["tenants"])


@native
def test_tenant_time_excludes_other_streams_on_its_gpus():
    # A transfer that outlasts the tenant on the same GPU must not count as the tenant's time.
    t = co.Tenant("t", "llama3.2-1b", "decode", Plan(), gpus=[1], batch=8, context=1024, steps=2)
    f = co.Flow("bg", src=[0], dst=[1], bytes=1 << 20, times=40, every_ns=40_000.0)
    res = co.run([t], "gb300", layers=1, flows=[f])
    row, flow = res["tenants"][0], res["flows"][0]
    assert flow["span_us"] > 2 * row["step_us"] * row["steps"]
    assert abs(row["slowdown"]) < 0.1
    assert flow["delivered_gbps"] == pytest.approx(flow["offered_gbps"], rel=0.05)
    # Two different models on one GPU each keep their own finish.
    small = co.Tenant("s", "llama3.2-1b", "decode", Plan(), gpus=[0], batch=8, context=1024, steps=1)
    big = co.Tenant("b", "llama3-8b", "decode", Plan(), gpus=[0], batch=8, context=1024, steps=2)
    res = co.run([small, big], "gb300", layers=1)
    a, b = res["tenants"]
    assert a["step_us"] * a["steps"] < b["step_us"] * b["steps"]
