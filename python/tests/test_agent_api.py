"""The agent API (rackscale.api / tools / mcp_server) on small, fast workloads."""

import json

import pytest

from rackscale import accuracy, api, tools

SPEC = {"model": "llama3-8b", "phase": "decode", "batch": 8, "context": 1024, "parallel": {"tp": 8}, "layers": 1}


@pytest.fixture(scope="module")
def base():
    return api.run("hgx_h100_8", SPEC, label="base")


def test_run_summary(base):
    assert base["schema"] == api.SCHEMA
    st = base["step_time_us"]
    assert st["lo"] < st["value"] < st["hi"]
    assert base["tokens_per_step"] == 8  # tp=8 over 8 GPUs: one replica of batch 8
    assert base["top_kernels"] and base["provenance"]["mode"] == "fast"
    # H100 has no measured calibration: the band says so.
    assert any(t["part"] == "kernels:uncalibrated" for t in base["accuracy"]["terms"])
    json.dumps(base)  # JSON-able


def test_run_is_cached(base):
    again = api.run("hgx_h100_8", SPEC)
    assert again["run_id"] == base["run_id"] and again["cached"]


def test_bottlenecks(base):
    b = api.bottlenecks(base["run_id"])
    assert b["items"] and b["summary"].startswith("Step")
    top = b["items"][0]
    assert top["bound"] in {"hbm", "tensor", "memory_latency", "tensor_pipeline", "sync", "smem_mio", "alu", "tail",
                            "latency", "pipeline_handoff"}
    # M=8 against a 128-row tile is flagged.
    assert any("tile_padding" in it["flags"] for it in b["items"])
    for lv in b["suggested_what_ifs"]:
        assert set(lv["what_if"]) <= {"workload", "hardware", "system", "mode", "launch_gap_ns"}


def test_overrides_scale_and_apply():
    r = api.resolve("hgx_h100_8", {"hbm_bw_gbps": "x2", "nvlink_gbps_per_dir": 50})
    base = api.resolve("hgx_h100_8")
    assert r["gpus"][0]["hbm"]["bw_gbps"] == 2 * base["gpus"][0]["hbm"]["bw_gbps"]
    assert r["spec"]["topology"]["link"]["gbps_per_dir"] == 50
    assert r["spec"]["topology"]["switch"]["gbps_per_port_per_dir"] == 50
    with pytest.raises(ValueError):
        api.resolve("hgx_h100_8", {"no_such_knob": 1})


def test_a100_gets_measured_calibration():
    r = api.resolve("hgx_h100_8")
    assert r["calibrations"] == []
    sys = {"name": "A100 x1", "gpus": [{"preset": "a100_sxm", "count": 1}],
           "topology": {"kind": "full_mesh", "links_per_pair": 1, "link": {"gbps_per_dir": 25.0, "latency_ns": 20.0}}}
    a = api.resolve(sys)
    assert a["calibrations"] and a["calibrations"][0]["file"].endswith("a100-sxm4-80gb-sm80/calibration.json")


def test_what_if_and_compare(base):
    w = api.what_if(base["run_id"], {"hardware": {"hbm_bw_gbps": "x2"}})
    c = w["comparison"]
    assert c["difference"]["change"] == "hardware"
    assert w["result"]["step_time_us"]["value"] < base["step_time_us"]["value"]
    assert c["kernels"] and "verdict" in c
    w2 = api.what_if(base["run_id"], {"workload": {"dtype": "fp8"}})
    # Kernels renamed by the dtype change still line up by op.
    assert any(r["kernel_a"] and r["kernel_b"] and r["kernel_a"] != r["kernel_b"] for r in w2["comparison"]["kernels"])


def test_sweep(base):
    s = api.sweep(base["run_id"], "workload.batch", [4, 16], objective="tokens_per_s")
    assert [r["value"] for r in s["rows"]] == [4, 16]
    assert s["best"]["value"] == 16


def test_kernel_drilldown(base):
    k = api.kernel(base["run_id"], "lm_head", top_pcs=5)
    assert k["role_anatomy"] and k["top_pcs"] and len(k["top_pcs"]) <= 5
    assert k["time_us"]["lo"] < k["time_us"]["value"]
    assert k["diagnosis"]
    k2 = api.kernel(system="hgx_b300_8", spec={"type": "gemm", "m": 256, "n": 1024, "k": 1024})
    assert k2["accuracy_class"]["class"] == "synthetic" and k2["accuracy_class"]["calibrated_gpu"]


def test_diff_band_correlation():
    hw = accuracy.diff_band(100, 20, 90, 20, "hardware")
    sysd = accuracy.diff_band(100, 20, 90, 20, "system")
    assert hw["band_us"] < sysd["band_us"]
    assert hw["significant"] and not sysd["significant"]


def test_tools_registry(base):
    names = {t["name"] for t in tools.specs()}
    assert {"simulate", "find_bottlenecks", "inspect_kernel", "what_if", "sweep", "compare_runs"} <= names
    for t in tools.specs():
        assert t["input_schema"]["type"] == "object"
    assert "error" in tools.call("simulate", {"system": "nope", "workload": SPEC})
    out = json.loads(tools.call_json("list_runs", {}))
    assert any(r["run_id"] == base["run_id"] for r in out["runs"])


def test_simulate_colocated_tool():
    out = json.loads(tools.call_json("simulate_colocated", {
        "tenants": [{"name": "a", "model": "llama3.2-1b", "phase": "decode", "plan": {"tp": 2}, "gpus": "0-1",
                     "batch": 8, "context": 1024, "steps": 2},
                    {"name": "b", "model": "llama3.2-1b", "phase": "decode", "gpus": [2], "batch": 8,
                     "context": 1024, "steps": 2}],
        "flows": [{"name": "kv", "src": [3], "dst": [2], "bytes": 1 << 20, "times": 2, "every_ns": 50000}],
        "layers": 1}))
    assert [t["name"] for t in out["tenants"]] == ["a", "b"] and out["flows"][0]["delivered_gbps"] > 0
    assert "flow kv" in out["summary"]



def test_scaling_ladder_tool():
    out = json.loads(tools.call_json("scaling_ladder", {"model": "llama3-70b", "phase": "decode", "gpus": [8, 72],
                                                        "min_tps_user": 30}))
    assert [r["gpus"] for r in out["rows"]] == [8, 72]
    assert all(r["fits"] and r["tok_s_user"] >= 30 for r in out["rows"])
    assert "error" in tools.call("scaling_ladder", {"model": "llama3-70b", "phase": "prefill"})


def test_mcp_server_builds():
    pytest.importorskip("mcp")
    from rackscale import mcp_server
    assert mcp_server.build() is not None
