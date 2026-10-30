# Python API

```python
import rackscale as rs
from rackscale import lower, models
```

The package is a thin layer over the Rust core (`rackscale._native`): it builds workloads, runs
them and presents the results. Everything crosses into Rust as JSON or YAML, so a workload you
build in Python can be saved, edited by hand and run with `rackscale run` or `rsim run`.

## Simulate

### `rs.simulate(system, workload, *, fast=False, trace=None, launch_gap_ns=None, config=None) -> Report`

Run `workload` on `system`.

| argument | accepts |
|---|---|
| `system` | a preset name (`"gb200_nvl72"`, `"hgx_b200_8"`, see `rackscale presets`), a path to a system YAML, or YAML text |
| `workload` | a dict from `rackscale.lower` / `rackscale.capture` / `rackscale.replay`, JSON text, or a path to a JSON/YAML file ([format](../WORKLOAD_FORMAT.md)) |
| `fast` | simulate each distinct kernel once on a cold GPU and reuse the result across ranks and layers; collectives and dependencies stay exact |
| `trace` | write a Chrome/Perfetto trace to this path |
| `launch_gap_ns` | override the host launch gap (e.g. `0` or `300` for CUDA graphs) |
| `config` | extra `RunConfig` fields, e.g. `{"gpu": {"per_sm_counters": True}}` |

### `rs.viz(system, workload, out, *, fast=True, detail_ranks=(0,), warp_sms=(0,), name=None, ...) -> Report`

Like `simulate`, and also writes the [viewer](../VIZ.md): `out` ending in `.html` is a
self-contained page, `.json` an `rsviz/1` bundle. `detail_ranks` get per-SM and per-kernel
detail; `warp_sms` get per-warp, per-cycle timelines.

### `rs.kernel(gpu, kind, *dims, dtype="bf16") -> dict`

One synthetic kernel alone on one GPU: `rs.kernel("b200", "gemm", 8192, 8192, 8192)`. Kinds and
dimensions are those of [`rackscale kernel`](cli.md).

### `rs.presets() -> dict`

The built-in GPU, switch and system presets with their resolved specs.

## `Report`

| member | what |
|---|---|
| `makespan_us` | step time: first launch to last completion on any GPU |
| `bubbles` | `gpu_idle_fraction`, `tensor_idle_fraction`, and `sched`, the scheduler-cycle anatomy (fractions by stall reason) |
| `kernels(top=20)` | per-kernel rows: `name`, `count`, `total_us`, `tflops`, `gbps`, `tensor_util`, ... |
| `memory` | the fullest GPU's `total_gb`, `capacity_gb` and `by_category_gb` (weights, KV cache, activations, optimizer) |
| `rep["power"]` | GPUs with a power model (B300, GB300) only: summed board power over time, `bin_us` and `gpus_w` (W per bin), with `avg_w`, `peak_w`, `min_w`, `energy_j` and `max_step_w_per_us`; each rank in `rep["ranks"]` has `energy_j` and `avg_power_w`. GPU boards only (book chapter 17) |
| `warnings` | reasons the result may not be meaningful, e.g. the model does not fit |
| `summary()` | the text report the CLI prints |
| `raw`, `rep["key"]` | the full JSON report, including per-rank stats in `rep["ranks"]` and every collective in `rep["collectives"]` |

## Closed forms

The book's closed forms are importable, so a simulated number can sit next to its formula.

| module | what |
|---|---|
| `rackscale.tail` | stragglers and faults (chapter 16): `expected_max_half_normal(n)`, `straggler_overhead(n, sigma, sync_every=1)`, `mean_overhead`, `dead_plane_slowdown(planes)`, `degraded_plane_slowdown(planes, scale)`, `ecmp_max_load(flows, links, groups)` |
| `rackscale.power` | power and cooling (chapter 17): `with_power_cap(system, cap_w)` (system text for `simulate`), `cap_tradeoff`, `rack_trace(rep)`, `smoothing(watts, bin_us, floor_fraction)`, `square_wave_buffer_j`, `thermal_power_limit`, `junction_temp_c`, `distribution(rack_w, volts)`, `coolant_flow(heat_w, rise_k)`, `racks_per_cdu`, `tokens_per_joule` |

## Run reports and figures

```python
from rackscale import report

reps = [rs.simulate(s, wl, fast=True) for s in ("hgx_h100_8", "hgx_b200_8")]
report.write(reps, "compare.html", system=["hgx_h100_8", "hgx_b200_8"], title="H100 vs B200",
             note="Free text under the title.")
report.figures(reps, "figs/", system=["hgx_h100_8", "hgx_b200_8"], labels=["H100", "B200"])
```

| function | what |
|---|---|
| `report.write(runs, path, system=None, *, title=None, note=None, labels=None)` | one self-contained HTML page for one run or several side by side. `system` (one name or one per run) gives the GPU spec for the roofline; `labels` name the runs (default: the system) |
| `report.page(...)` | the same page as a string |
| `report.figures(runs, out_dir, system=None, labels=None)` | each figure as a standalone SVG: `compare_step`, `time_split`, `roofline`, `top_kernels`, `stalls` (suffixed by run when there are several) |
| `report.analyze(rep, system=None, label=None)` | the numbers behind the page: per-kernel `share`, `ai` (FLOP/byte), `bound` (tensor, hbm or latency), `roof_frac`, and the time `split` |

## Build workloads from a model

```python
par = rs.Parallel(tp=8, ep=1, dp=1, pp=1, comm_sms=20)
```

`Parallel` describes the layout. Ranks are laid out TP innermost, then DP, then PP, so `world()`
is `tp * dp * pp`. `ep` is the DeepSeek inference layout (data-parallel attention, experts
sharded over `ep` ranks) and does not combine with the others. A layout that does not divide the
model raises `ValueError` with the reason and the fix.

| function | builds |
|---|---|
| `lower.decode_step(model, par, batch, context, *, overlap=False, layers=None)` | one decode iteration: every rank generates one token for `batch` sequences. `overlap=True` is DeepSeek-style two-micro-batch overlap |
| `lower.prefill(model, par, batch, seqlen, *, layers=None)` | one prefill pass |
| `lower.train_step(model, par, micro_batch, seqlen, *, micro_batches=1, schedule="1f1b", optimizer=False, layers=None)` | one training step; `schedule` is `gpipe`, `1f1b` or `zb` (ZB-H1 zero bubble) |
| `lower.async_rl(model, *, rollout_ranks, trainer_ranks, rollout_batch, context, decode_steps, train_micro_batch, seqlen, layers=None)` | rollout GPUs decode while trainer GPUs train, then the weight push |
| `lower.memory(model, par, *, phase, batch, context, seqlen)` | bytes the fullest GPU needs, by category, for the whole model |

`layers=N` keeps the first N layers to make runs short; memory is still checked for the whole
model. `models.ALL` maps names to architectures (`models.LLAMA3_70B`, `models.DEEPSEEK_V3`,
`models.MAMBA2_8B`, ...); `rackscale models` prints them. A `models.Model` is a frozen dataclass,
so `dataclasses.replace(models.LLAMA3_8B, hidden=5120)` gives you a new architecture.

## Captures

| function | what |
|---|---|
| `capture.from_file(path, *, mode="resimulate", world=1, collective_group="world")` | a PyTorch-profiler trace as a workload |
| `record.load(path)` | read a `rackscale capture` directory |
| `replay.to_workload(cap, plan=None, *, mode="paths", host=False, steps=None)` | the capture as a workload over `plan.world` ranks; edit it or pass it to `simulate` |
| `replay.replay(cap, system, plan=None, *, fast=True, ...)` | the capture simulated on `system`; returns a `Report` |

See [Capture and replay](../CAPTURE.md).

## Agent API

`rackscale.api` is the same simulator with JSON in and JSON out, error bands and provenance on
every number, and a session cache of runs. `rackscale.tools` exposes it as tool definitions for
LLM tool use, and `rackscale-mcp` serves it over MCP.

| function | what |
|---|---|
| `api.run(system, spec, *, mode="fast", overrides=None)` | simulate a workload spec; returns a `run_id`, step time with its band, top kernels |
| `api.bottlenecks(run_id)` | each kernel classed (tensor, HBM, latency, ...) with suggested what-ifs |
| `api.kernel(run_id, name)` | per-warp-role stall anatomy and the SASS instructions warps wait on |
| `api.what_if(run_id, change)` | re-run with one change; verdict on whether it beats the band |
| `api.sweep(run_id, key, values)` | one re-run per value |
| `api.compare(run_a, run_b)` | step, throughput, per-kernel and stall deltas |

See [Agent API and MCP](../AGENT_API.md) and [tutorial 4](../tutorials/agent.md).
