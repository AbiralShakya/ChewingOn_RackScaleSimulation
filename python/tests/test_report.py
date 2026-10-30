"""Run reports: the HTML page, the standalone SVG figures and the command-line paths that write them."""

import json
import xml.dom.minidom

import pytest

import rackscale as rs
from rackscale import cli, lower, models, report

pytestmark = pytest.mark.skipif(rs._native is None, reason="native extension not built")


@pytest.fixture(scope="module")
def runs():
    wl = lower.decode_step(models.LLAMA3_8B, rs.Parallel(tp=2), 8, 1024, layers=1)
    return [rs.simulate(s, wl, fast=True) for s in ("hgx_h100_8", "hgx_b200_8")]


def test_analyze_classifies_every_kernel(runs):
    a = report.analyze(runs[0], "hgx_h100_8")
    assert a["label"] == "HGX H100 8-GPU"
    assert a["kernels"] and 0 < sum(k["share"] for k in a["kernels"]) <= 1 + 1e-6  # of all GPU time, idle included
    assert {k["bound"] for k in a["kernels"]} <= {"tensor", "hbm", "latency"}
    assert abs(sum(a["split"].values()) - 1) < 1e-3


def test_single_page(runs):
    html = report.page(runs[1], "hgx_b200_8", title="one run")
    assert html.startswith("<!doctype html>") and "<h1>one run</h1>" in html
    assert "Kernels on the roofline" in html and "Side by side" not in html


def test_compare_page_and_labels(runs):
    html = report.page(runs, ["hgx_h100_8", "hgx_b200_8"], labels=["Hopper", "Blackwell"], note="a\n\nb")
    assert "Side by side" in html and "Hopper" in html and "Blackwell" in html
    with pytest.raises(ValueError):
        report.page(runs, ["hgx_h100_8"])


def test_figures_are_valid_svg(runs, tmp_path):
    paths = report.figures(runs, str(tmp_path), ["hgx_h100_8", "hgx_b200_8"], labels=["H100", "B200 (TP2)"])
    names = sorted(p.split("/")[-1] for p in paths)
    assert "compare_step.svg" in names and "roofline_1_b200_tp2.svg" in names
    for p in paths:
        xml.dom.minidom.parse(p)


def test_cli_html_compare_and_report_command(tmp_path, capsys):
    html, js = tmp_path / "r.html", tmp_path / "r.json"
    code = cli.main(["decode", "llama3-8b", "--system", "hgx_h100_8,hgx_b200_8", "--tp", "2", "--batch", "8",
                     "--context", "1024", "--layers", "1", "--fast", "--html", str(html), "--report", str(js),
                     "--figures", str(tmp_path / "figs")])
    assert code == 0
    assert "speedup" in capsys.readouterr().out
    assert "Side by side" in html.read_text()
    data = json.loads(js.read_text())
    assert isinstance(data, list) and len(data) == 2 and data[0]["gpu_spec"]
    assert (tmp_path / "figs" / "compare_step.svg").exists()
    out = tmp_path / "again.html"
    assert cli.main(["report", str(js), "-o", str(out), "--title", "again"]) == 0
    assert "<h1>again</h1>" in out.read_text()
