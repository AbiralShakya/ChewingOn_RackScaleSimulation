# Contributing to rackscale

Thanks for helping. rackscale aims to be a simulator people can trust, so the bar for a change is
simple to state: it is tested, it is deterministic, and every number it adds says where it came
from.

## Set up

```bash
git clone https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation rackscale && cd rackscale
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest && maturin develop --release   # Python package + Rust core, editable
cargo build --release -p rsim-cli                          # optional: the standalone `rsim` binary
```

Python edits apply immediately; re-run `maturin develop --release` after changing Rust.

## Before opening a pull request

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
maturin develop --release && pytest -q python/tests
```

CI runs the same commands plus a validation smoke test against measured Nsight Compute cycles
(`rsim validate`, the fast reference kernels on B300 and A100). If you touched the docs, preview
the site with `pip install mkdocs-material && mkdocs serve`.

## Where things live

| path | what |
|---|---|
| `crates/rsim-core` | discrete-event engine, time, ids |
| `crates/rsim-hw` | GPU, switch and system specs, presets, calibration loader, YAML |
| `crates/rsim-sass` | SASS parser, opcode classes, control bits |
| `crates/rsim-gpu` | the cycle-level SM model, memory hierarchy, tensor cores, synthetic kernels |
| `crates/rsim-net` | NCCL-faithful collectives over NVLink, NVSwitch and InfiniBand |
| `crates/rsim-rack` | the rack: ranks, streams, events, PDES, reports, traces |
| `crates/rsim-cli` | the `rsim` command line (also a library, run in-process by the Python package) |
| `crates/rsim-py` | Python bindings (`rackscale._native`) |
| `python/rackscale` | lowering, models, capture and replay, CLI, agent API |
| `calibration/` | microbenchmarks, measured results, reference kernels |
| `docs/` | the documentation site (`mkdocs.yml`) and the guide (`docs/book/`) |

The [architecture overview](https://abiralshakya.github.io/ChewingOn_RackScaleSimulation/ARCHITECTURE/) explains how the pieces fit.

## Guidelines

- **Every hardware number has a provenance tag** (`spec`, `ubench`, `est` or `calibrated`) in
  the preset `notes`. A PR that changes a number should say where the new value comes from.
- **Determinism is a feature.** Do not iterate a `HashMap` where the order can affect timing.
  Break ties by id, and keep rank-parallel code free of shared mutable state, except for
  memoization that is computed once.
- **Hot paths** (`sm.rs`, `mem.rs`, `engine.rs`) should not allocate per cycle. Profile with
  `cargo run --release --example bench_gemm`.
- **Speed work must not move results.** `crates/rsim-gpu/tests/golden.rs` and
  `crates/rsim-rack/tests/golden.rs` pin cycles and counter digests. A change meant to alter
  timing updates them in the same commit and says why (`RSIM_GOLDEN_PRINT=1` prints the new
  values). See "Simulator speed" in `docs/ARCHITECTURE.md`.
- **Accuracy claims need a measurement.** A change that is meant to make the simulator more
  accurate should show the before and after error against a reference (`rsim validate`, or a
  capture replayed with `--kernels`).
- **User-facing errors are one line and say how to fix the input.** Raise `ValueError` in
  Python (the CLI prints it without a traceback) and `anyhow` errors with context in Rust.

## Common changes

- **A new instruction**: add the opcode to `rsim-sass/src/isa.rs::classify`, set its timing in
  `rsim-gpu/src/pipes.rs`, and add a parse test.
- **New hardware**: add a preset in `rsim-hw/src/presets.rs` with a datasheet test, plus a
  `configs/` example.
- **A new model architecture**: add it to `python/rackscale/models.py` and `models.ALL`; the
  lowering in `lower.py` derives every kernel shape from the dataclass. Add a test in
  `python/tests/test_lower_capture.py`.
- **A new CLI command**: Python commands go in `python/rackscale/cli.py`; simulator-core commands
  in `crates/rsim-cli` (they appear under `rackscale` automatically once listed in
  `RSIM_COMMANDS`). Regenerate the reference with `python docs/reference/gen_cli.py`.

## Reporting problems

Use the issue templates: a bug, an **accuracy report** (the simulator disagrees with hardware;
include the measurement), a feature request, or a request for new hardware. Security issues go
to the address in [SECURITY.md](https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/blob/main/SECURITY.md), not to a public issue.

## Releases

Versions follow [semantic versioning](https://semver.org) from the workspace version in
`Cargo.toml`, which the Python package also reports. To release, update `CHANGELOG.md`, bump
`[workspace.package] version`, and push a `vX.Y.Z` tag: the release workflow builds wheels for
Linux and macOS, `rsim` binaries, and publishes to PyPI.

By contributing you agree that your contributions are licensed under the Apache License 2.0.
