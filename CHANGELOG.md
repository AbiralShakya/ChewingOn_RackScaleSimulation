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
- `rackscale.colocate`: several models on one rack at once. Each tenant (model, phase, parallel
  plan, GPUs) is lowered on its own and placed on its GPUs of one shared workload, partitioned or
  sharing GPUs MPS-style, with optional `Flow`s between tenants (KV caches from a prefill pool,
  weights from a trainer) issued on a clock. Reports each tenant's step together and alone, and
  each flow's delivered rate. Agent tool `simulate_colocated`; book chapter 12;
  `research/colocation/run.py`.
- `rackscale.scaling.ladder`: the best closed-form plan at every GPU count from one GPU to the
  rack (training at a fixed global batch, decode under a per-user speed floor), with
  `best_train`, `best_decode`, `train_plans` and `decode_plans`. Agent tool `scaling_ladder`;
  table in book chapter 11. `scaling.replay(..., spans=True)` returns every op's span for
  timelines.
- `rackscale.hnet`: H-Net (byte-level, dynamic chunking) 1-stage and 2-stage XL with the BPE and
  byte-level Transformer baselines: parameters and FLOPs per byte (within 7% of the paper),
  prefill, and byte-by-byte decode whose main-network batch is drawn from the routers
  (`step_counts`, `even_counts`, `decode_steps(counts=...)`). Book chapter 13 and
  `research/bytes/run.py`, which also splits every simulated step by part of the network from
  the per-op trace.
- Book chapter 14, "Why cycle level, and how far to trust a simulated rack": what only the
  cycle-level model sees, which kernels run as real SASS, and how per-kernel error carries to a
  node and a rack.
- Rank reports carry `stream_finish_us`, when each stream finished its last op, so jobs that
  share a GPU are timed by their own streams.
- `host` ops are documented in the workload format.
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

### Changed (simulated timings)
- Each network transfer's starting NVLink plane is seeded from its communicator's GPUs instead of
  the order in which the workload created communicators, so a job's timing no longer depends on
  unrelated jobs elsewhere in the system (a TP4 group's finish used to move by 2% when another TP4
  group was added on other GPUs). Single-job results move as the seed changes: in our checks
  makespans by −0.5% to +1.6%, transfer-mode all-to-alls by −6% to +12% and one GB200 32 MiB
  all-gather by +23%; packet-mode all-to-alls and the median validation point did not move.
  `docs/NETWORK.md` has the new validation numbers.

### Fixed
- `scaling.estimate` for inference counts every DP group's batch (`tokens_per_step` was one
  group's), and with PP it now gives the step a token takes through all stages and the tokens of
  the pp batches in flight; inference plan labels leave out training-only terms (schedule, ZeRO).
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
