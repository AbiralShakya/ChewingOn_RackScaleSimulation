"""The rack-scale answer layer: answer blocks, the new tools, the learn layer."""

import pytest

from rackscale import api, learn, scaling, tools, toolkit

DECODE = {"model": "llama3-8b", "phase": "decode", "batch": 32, "context": 4096, "parallel": {"tp": 8}}


def test_every_preset_resolves():
    # list_hardware resolves every system preset; one broken preset broke the tool
    h = api.hardware()
    assert "gb200_nvl72_2rack_ib" in h["systems"]
    assert h["systems"]["gb200_nvl72_2rack_ib"]["num_gpus"] == 144


def test_run_answer_block():
    r = api.run("hgx_b300_8", DECODE)
    a = r["answer"]
    assert a["quantity"] == "step time" and a["unit"] == "us"
    assert a["band"]["lo"] < a["value"] < a["band"]["hi"]
    # the roofline replay of the same program is a lower bound
    assert 0 < a["closed_form"]["value"] < a["value"] and a["sim_over_closed_form"] > 1
    assert a["bottleneck"] in toolkit.WHY and a["why"]
    assert r["bottleneck"]["bottleneck"] == a["bottleneck"]
    tb = toolkit.textbook(api._rec(r["run_id"]))
    split = tb["gap_split_us"]
    gap = r["step_time_us"]["value"] - tb["step_us"]
    assert sum(split.values()) == pytest.approx(gap, rel=0.02, abs=2)


def test_hardware_overrides_reach_closed_forms():
    base = scaling.hardware("hgx_b300_8")
    fast = scaling.hardware("hgx_b300_8", overrides={"hbm_bw_gbps": "x2"})
    assert fast.hbm_bw == pytest.approx(2 * base.hbm_bw)


def test_variability_is_part_of_the_run():
    wl = toolkit._bsp(8, "world", 4, 1, {"type": "gemm", "m": 512, "n": 1024, "k": 1024})
    a = api.run("hgx_b300_8", wl)
    b = api.run("hgx_b300_8", wl, variability={"kernel_sigma": 0.2, "seed": 1})
    assert a["run_id"] != b["run_id"]
    assert b["step_time_us"]["value"] > a["step_time_us"]["value"]


def test_gemm_tiling_split_k():
    r = toolkit.gemm_tiling("hgx_b300_8", 16, 8192, 8192, vary={"split_k": [1, 2]})
    rows = {x["variant"]: x for x in r["rows"]}
    # 64 tiles on 148 SMs: splitting K fills more SMs and wins
    assert rows["split_k=2"]["ctas"] == 2 * rows["split_k=1"]["ctas"]
    assert rows["split_k=2"]["time_us"]["value"] < rows["split_k=1"]["time_us"]["value"]
    assert r["best"] == "split_k=2"
    assert r["answer"]["closed_form"]["value"] > 0


def test_tail_closed_forms():
    r = toolkit.tail_latency(72, 0.05, plane_scale=0.5, flows=448, links=8, groups=16)
    assert r["stragglers"]["overhead_closed_form"] == pytest.approx(0.132, abs=0.002)
    assert r["link"]["nvls_striped_slowdown_closed_form"] == 2.0
    assert r["ecmp"]["busiest_over_mean"] == pytest.approx(1.34, abs=0.03)


def test_tokens_or_bytes_closed_form():
    r = toolkit.tokens_or_bytes()
    d = r["decode_hbm_bytes_per_text_byte"]
    assert d["ratio"] > 0 and r["prefill_flops_per_byte"]["ratio"] > 0
    assert r["answer"]["bottleneck"] == "kernels:hbm"


@pytest.mark.parametrize("topic", sorted(learn.CONCEPTS))
def test_learn_topics(topic):
    r = learn.learn(topic)
    assert r["formula"] and r["idea"] and r["chapter"].startswith("docs/book/")
    assert r["see_it_in_the_simulator"]["tool"] in {t["name"] for t in tools.TOOLS}


def test_learn_params_and_suggestions():
    assert learn.learn("pipeline_bubble", stages=8, micro_batches=32)["worked_example"]["gpipe_1f1b_bubble"] == \
        pytest.approx(7 / 39, abs=1e-4)
    err = tools.call("learn", {"topic": "bubbles"})
    assert "pipeline_bubble" in err["error"]


def test_scaling_math_answer_has_expected_range():
    r = tools.call("scaling_math", {"model": "llama3-8b", "phase": "train", "plan": {"tp": 8}, "micro_batch": 1})
    a = r["answer"]
    rng = a["expected_simulated_us"]
    assert rng["lo"] < rng["median"] <= rng["hi"]
    assert rng["lo"] > a["value"]  # training textbooks are lower bounds
    assert a["bottleneck"] == r["dominant"]


def test_new_tools_registered():
    names = {t["name"] for t in tools.TOOLS}
    for n in ("gemm_tiling", "tail_latency", "power_thermal", "tokens_or_bytes", "predict_gpu", "html_report", "learn"):
        assert n in names
    for t in tools.specs():
        assert len(t["description"]) > 60, t["name"]
