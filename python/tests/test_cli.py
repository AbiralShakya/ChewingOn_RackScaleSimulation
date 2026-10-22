"""The command line: forwarding to the Rust core, new commands, errors and the summary."""

import pytest

import rackscale as rs
from rackscale import cli

native = pytest.mark.skipif(rs._native is None, reason="native extension not built")


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"rackscale {rs.__version__}"


def test_models_lists_every_model(capsys):
    code, out, _ = run(capsys, "models")
    assert code == 0
    for name in rs.models.ALL:
        assert name in out


@native
def test_rsim_commands_are_forwarded():
    # The Rust CLI prints straight to the process's stdout, so only the exit codes are checked here.
    assert cli.main(["presets"]) == 0
    assert cli.main(["system", "hgx_b200_8"]) == 0
    assert cli.main(["kernel", "--gpu", "nope", "gemm", "1", "1", "1"]) == 1
    assert cli.main(["system", "missing.yaml"]) == 1


def test_bad_layout_is_one_line_error(capsys):
    code, _, err = run(capsys, "decode", "llama3-70b", "--system", "hgx_h100_8", "--tp", "3")
    assert code == 2
    assert "does not divide" in err and "Traceback" not in err


def test_capture_without_input_is_usage_error(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["capture"])
    assert e.value.code == 2
    assert "torch-profiler trace" in capsys.readouterr().err


@native
def test_decode_writes_viewer_page(tmp_path, capsys):
    page = tmp_path / "v.html"
    code, out, _ = run(capsys, "decode", "llama3-8b", "--system", "hgx_h100_8", "--tp", "8", "--layers", "1",
                       "--fast", "--viz", str(page))
    assert code == 0
    assert "makespan" in out and page.exists() and page.stat().st_size > 10_000


def test_collectives_are_grouped():
    rows = [{"kind": "AllReduce", "algo": "Ring", "bytes": 1 << 20, "ranks": 8, "last_post_ps": 0,
             "done_ps": 10_000_000, "busbw_gbps": 100.0}] * 3
    g = rs._group_collectives(rows)
    assert len(g) == 1 and g[0]["n"] == 3 and g[0]["mean_us"] == pytest.approx(10.0)
    assert rs._bytes(1 << 20) == "1.0 MiB"
