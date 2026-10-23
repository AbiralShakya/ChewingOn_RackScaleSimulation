"""The Hazy megakernel schedule, written as an all-CTA path, replays with inter-SM sync."""

import json

import pytest

import rackscale as rs
from rackscale import megakernel as mk

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")
TINY = mk.LlamaShape("tiny", 512, 2, 8, 2, 64, 1024, 4096)


def test_llama1b_schedule_matches_the_dag():
    s = mk.hazy_llama_schedule(mk.LLAMA_3_2_1B, 148, context=40)
    per_layer = {mk.QKV: 148, mk.PARTIAL: 8, mk.OPROJ: 128, mk.UPGATE: 148, mk.DOWN: 148}
    for op, n in per_layer.items():
        assert sum(t.op == op for t in s.tasks) == 16 * n
    assert sum(t.op == mk.LMHEAD for t in s.tasks) == 148
    # every weight byte of the model is streamed once
    m = mk.LLAMA_3_2_1B
    layer = m.hidden * ((m.heads + 2 * m.kv_heads) * m.head_dim + m.hidden + 3 * m.ffn) * 2
    kv = 16 * m.kv_heads * 2 * 40 * m.head_dim * 2
    assert sum(t.weight_bytes for t in s.tasks) == 16 * layer + kv + m.vocab * m.hidden * 2
    # each counter is waited for exactly as often as it is added to
    adds = {}
    for t in s.tasks:
        for c in t.adds:
            adds[c] = adds.get(c, 0) + 1
    for t in s.tasks:
        for c, need in t.waits:
            assert adds[c] == need
    # round-robin in DAG order
    assert [t.sm for t in s.tasks[:300]] == [i % 148 for i in range(300)]


@native
def test_replay_orders_tasks_through_the_counters(tmp_path):
    s = mk.hazy_llama_schedule(TINY, 8, context=16)
    mk.write_rpath(s, str(tmp_path / "mk.rpath"))
    wl = {"name": "mk", "spmd": {"world": 1, "program": {"streams": [[
        {"op": "kernel", "name": "mk", "kernel": {"type": "path", "path": str(tmp_path / "mk.rpath")}}]]}}}
    rs.viz("hgx_b300_8", wl, str(tmp_path / "b.json"), detail_ranks=[0], warp_sms=[0])
    run = json.loads((tmp_path / "b.json").read_text())["runs"][0]
    k = run["kernels"][0]
    tasks = mk.tasks_from_probes(s, run["details"][k["d"]]["probes"], 2032)
    rows = [tasks[i:i + 6] for i in range(0, len(tasks), 6)]
    assert len(rows) == len(s.tasks)
    done = {}
    for (sm, op, layer, t0, t1, t2), t in zip(rows, sorted(s.tasks, key=lambda t: (t.sm, t.slot))):
        assert t0 <= t1 <= t2
        done[(t.sm, t.slot)] = t2
    # a task's inputs are ready only after every task it waits on has finished
    for t in s.tasks:
        ready = rows[sorted(s.tasks, key=lambda x: (x.sm, x.slot)).index(t)][4]
        for c, _ in t.waits:
            for p in s.tasks:
                if c in p.adds:
                    assert done[(p.sm, p.slot)] <= ready + 1e-6
