"""rs.viz writes a viewer page / bundle with rack-to-SASS capture."""

import json

import pytest

import rackscale as rs
from rackscale import lower, models

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")


@native
def test_viz_bundle_and_page(tmp_path):
    wl = lower.decode_step(models.LLAMA3_8B, rs.Parallel(tp=2), batch=8, context=1024, layers=1)
    wl["spmd"]["world"] = 2
    rep = rs.viz("hgx_h100_8", wl, str(tmp_path / "b.json"), name="llama tp2")
    assert rep.makespan_us > 0
    b = json.loads((tmp_path / "b.json").read_text())
    assert b["format"] == "rsviz/1" and len(b["runs"]) == 1
    run = b["runs"][0]
    assert len(run["ranks"]) == 2
    det = [d for d in run["details"] if d["pc"]]
    assert det, "rank 0 kernels carry per-PC stats"
    assert any(d["warps"] for d in run["details"])
    rs.viz("hgx_h100_8", wl, str(tmp_path / "p.html"))
    page = (tmp_path / "p.html").read_text()
    assert 'id="rsviz-data">gz:' in page
