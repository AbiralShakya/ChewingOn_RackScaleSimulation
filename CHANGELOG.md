# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org);
until 1.0, a minor version may change simulated timings, and the release notes say which.

## [Unreleased]

### Added
- Every data-parallel sharding in the lowering: `Parallel(zero=0|1|2|3)` (DDP, ZeRO-1/2, FSDP with
  one-layer prefetch), `shard=` for HSDP, `sp=True` for Megatron sequence parallelism, and
  `prefill_cp(kind="ulysses")`. Default layouts lower exactly as before.
- `rackscale.scaling`: closed-form step time, MFU, throughput, memory and critical batch sizes for
  any DP / ZeRO / FSDP / HSDP / TP / SP / CP / PP / EP plan; a roofline replay of any lowered
  workload; and `check()`, which runs a plan through the formulas, the replay and the simulator
  and splits the difference into kernel efficiency, power-capped clock, collectives and overlap.
  Agent tools `scaling_math` and `check_scaling_math`. Book chapter 11 and
  `research/parallelism/check_math.py` (22 cases on GB300 NVL72).
- One install gives every command: the `rsim` command line now runs in-process from the Python
  package, so `pip install rackscale` provides `rackscale`, `rsim` and `rackscale-mcp`.
  `rackscale presets|system|kernel|sass|path|timeline|validate|net|viz` forward to it.
- `rackscale demo` (a 30-second tour ending in the viewer), `rackscale models`, `rackscale view`,
  `--viz PAGE.html` and `--open` on every simulate command, and `--version`.
- Progress line with elapsed time on long runs, colour on terminals (`NO_COLOR` respected), and
  collectives grouped by kind and size in the summary.
- Documentation site (MkDocs Material) with a quickstart, four tutorials, an examples gallery,
  and command-line and Python references; release workflow with prebuilt wheels; issue and PR
  templates; `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.

### Fixed
- `rsim run`, `rsim system` and `rsim viz run` accept a system preset name as well as a YAML path.
- File errors name the file; unknown GPU presets list the valid ones.
- Piping output to `head` no longer panics.
- `rackscale capture` with neither a trace nor a command prints usage instead of a traceback.
- `rackscale-mcp` without the MCP SDK says how to install it.

## [0.1.0] - 2026-10-22

First public version: cycle-level SM model for Ampere, Hopper and Blackwell (HMMA, HGMMA,
`tcgen05`, TMA, mbarriers); presets for A100, H100, H200, B200, GB200, B300 and GB300 with HGX,
NVL72 and InfiniBand systems; B300 and A100 calibration from microbenchmarks; NCCL-faithful
network model with transfer and packet fidelity; lowering of dense, GQA, MLA, MoE and Mamba-2
models for decode, prefill, training (with pipeline schedules) and async RL; capture of any CUDA
program with NVBit and replay on a rack; the rack-to-SASS viewer; fast mode and exact-mode rank
folding; the JSON agent API and MCP server; the guide in `docs/book/`.

[Unreleased]: https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/tree/main
