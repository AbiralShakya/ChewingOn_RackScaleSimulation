"""The agent API as a tool registry: name, description, JSON schema, function.

One registry serves every way an agent can reach the simulator:

* ``rackscale-mcp`` (``rackscale.mcp_server``) exposes it over the Model
  Context Protocol, for Claude Code, Claude Desktop or any MCP client;
* ``specs()`` gives the tool definitions in the shape LLM tool-use APIs take
  (``name``, ``description``, ``input_schema``), and ``call(name, args)``
  runs one, for agents written directly against an API
  (``examples/agent/perf_agent.py``).

Every tool returns a JSON-able dict (``schema: rackscale-api/v1``).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from . import api

INSTRUCTIONS = """\
rackscale simulates GPU racks cycle by cycle, from SASS instructions up to NVLink collectives.
A typical performance-engineering loop:
1. list_hardware / list_models to see systems, knobs and models.
2. simulate(system, workload) -> run_id, step time with an error band, top kernels.
3. find_bottlenecks(run_id) -> each kernel classed (hbm / tensor / memory-latency / ...) with
   ready-to-run what-ifs.
4. inspect_kernel(run_id, name) -> per-warp-role stalls and the SASS instructions warps wait on.
5. what_if(run_id, change) or sweep(run_id, key, values) -> re-simulate and compare, with a
   verdict on whether the difference beats the error band.
6. best_layout(model, system, phase) -> the best TP / EP / PP layout and batch under a latency
   target, and what bounds each step (compute, HBM, latency, communication, launch gaps).
7. scaling_math(model, system, phase, plan) -> the closed-form step for any DP / ZeRO / FSDP / HSDP /
   TP / SP / CP / PP / EP plan, term by term; check_scaling_math runs the same plan through a roofline
   replay and the simulator and says where and why reality departs from the formulas.
   scaling_ladder(model, system, phase) -> the best plan at every GPU count from 1 GPU to the rack.
8. simulate_colocated(tenants) -> several models on one rack at once (partitioned or sharing GPUs),
   each tenant's slowdown from the others.
Bands come from measured validation (accuracy tool). Differences between runs are more
reliable than absolute times, especially on GPUs without measured calibration.
Runs are cached by content, so repeating a call is free. Fast mode takes seconds to a minute;
exact mode and long-context kernels can take minutes.
"""

_SYSTEM = {"description": "preset name (e.g. gb200_nvl72, hgx_b200_8, hgx_b300_8, hgx_h100_8), a system YAML path, "
                          "or an inline system dict", "type": ["string", "object"]}
_WORKLOAD = {
    "type": "object",
    "description": "workload spec. Model: {model, phase: decode|prefill|train, batch, context (decode), seqlen "
                   "(prefill/train), micro_batch (train), parallel: {tp, ep, dp, comm_sms}, layers, overlap, dtype: "
                   "bf16|fp8|fp4}. Byte-level H-Net (hnet-1stage-xl, hnet-2stage-xl; baselines bpe-transformer-xl, "
                   "byte-transformer-xl): {model, phase: prefill|decode, batch, seqlen or context in bytes, steps "
                   "(decode byte steps), routing: sampled|mean_field}. One kernel: {kernel: {type: gemm, m, n, k, "
                   "dtype} | {type: attention, batch, heads_q, heads_kv, sq, skv, d, causal} | {type: path, path: "
                   "<.rpath>}}. Capture: {capture: <dir>, plan: {tp, ep, pp, dp}}. Raw: {workload: {...}}.",
}
_OVERRIDES = {"type": "object", "description": "hardware knobs {name or path: value}; 'x1.5' scales the current value. "
                                               "See list_hardware().knobs"}
_RUN_ID = {"type": "string", "description": "run id returned by simulate / what_if"}
_PLAN = {"type": "object", "description": "parallel plan: {tp, sp (bool, Megatron sequence parallel), cp, cp_kind: "
                                          "ring|ulysses, pp, schedule: gpipe|1f1b|interleaved|zb, virtual, dp, zero: "
                                          "0 (DDP)|1|2|3 (FSDP), shard (HSDP group size), ep}. world = tp*cp*pp*dp."}


def _tool(name: str, fn: Callable, description: str, properties: Dict[str, Any], required: List[str]) -> dict:
    return {"name": name, "fn": fn, "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": required}}


TOOLS: List[dict] = [
    _tool("list_hardware", lambda name=None: api.hardware(name),
          "GPU and system presets with peaks and calibration status, plus the tunable hardware knobs. Give `name` "
          "for one preset's full resolved spec.",
          {"name": {"type": "string", "description": "optional GPU or system preset"}}, []),
    _tool("list_models", lambda: api.models_list(),
          "Model architectures the simulator can lower (Llama 3, DeepSeek-V3, Qwen3 MoE, Mixtral, Mamba-2 and "
          "hybrids, and byte-level H-Net with its BPE and byte Transformer baselines), with sizes and phases.", {}, []),
    _tool("describe_workload", lambda spec: api.workload(spec),
          "Build a workload from a spec without simulating it: kernel and collective counts, world size, tokens "
          "per step.", {"spec": _WORKLOAD}, ["spec"]),
    _tool("simulate",
          lambda system, workload, mode="fast", overrides=None, launch_gap_ns=None, label=None: api.run(
              system, workload, mode=mode, overrides=overrides, launch_gap_ns=launch_gap_ns, label=label),
          "Simulate a workload on a system. Returns run_id, step time with an error band, tokens/s (model specs), "
          "MFU, idle fraction, scheduler-stall anatomy, top kernels, collectives, warnings and provenance.",
          {"system": _SYSTEM, "workload": _WORKLOAD,
           "mode": {"type": "string", "enum": ["fast", "exact"],
                    "description": "fast: each distinct kernel simulated once and reused (~1% from exact); exact: "
                                   "full co-simulation"},
           "overrides": _OVERRIDES,
           "launch_gap_ns": {"type": "number", "description": "kernel launch gap; ~300 models CUDA-graph replay"},
           "label": {"type": "string"}},
          ["system", "workload"]),
    _tool("find_bottlenecks", lambda run_id, top=8: api.bottlenecks(run_id, top),
          "Where a run's time goes: every kernel classed as hbm-, tensor-, memory-latency-, tensor-pipeline-, "
          "sync- or alu-bound from achieved rates and stalls, with wave and tile-padding flags, a diagnosis, and "
          "what-ifs ready to pass to what_if.",
          {"run_id": _RUN_ID, "top": {"type": "integer"}}, ["run_id"]),
    _tool("inspect_kernel",
          lambda run_id=None, name=None, system=None, spec=None, overrides=None, top_pcs=15: api.kernel(
              run_id, name, system=system, spec=spec, overrides=overrides, top_pcs=top_pcs),
          "Drill into one kernel on a cold GPU (as Nsight Compute times it): per-warp-role stall anatomy "
          "(producer / MMA / softmax / epilogue), the SASS instructions warps wait on most, waves, tile padding "
          "and a diagnosis. Name a kernel of a run (run_id + name, matching kernel or op name), or give system + "
          "kernel spec.",
          {"run_id": _RUN_ID, "name": {"type": "string", "description": "kernel or op name substring, e.g. lm_head"},
           "system": _SYSTEM, "spec": {"type": "object", "description": "kernel spec, e.g. {type: gemm, m, n, k}"},
           "overrides": _OVERRIDES, "top_pcs": {"type": "integer"}}, []),
    _tool("what_if", lambda run_id, change, label=None: api.what_if(run_id, change, label=label),
          "Re-run a run with one change and compare: step time, tokens/s, per-op deltas, stall-anatomy deltas, and "
          "whether the difference beats the error band. change: {workload: {batch|dtype|parallel|overlap|layers...}, "
          "hardware: {knob: value}, system: preset, mode, launch_gap_ns}.",
          {"run_id": _RUN_ID, "change": {"type": "object"}, "label": {"type": "string"}}, ["run_id", "change"]),
    _tool("sweep", lambda run_id, key, values, objective="step_time": api.sweep(run_id, key, values, objective=objective),
          "Re-run once per value of one parameter and tabulate. key: workload.<field> (workload.batch, "
          "workload.parallel.tp), hardware.<knob>, system, mode or launch_gap_ns.",
          {"run_id": _RUN_ID, "key": {"type": "string"}, "values": {"type": "array"},
           "objective": {"type": "string", "enum": ["step_time", "tokens_per_s"]}}, ["run_id", "key", "values"]),
    _tool("best_layout",
          lambda model, system, phase="decode", context=8192, seqlen=8192, batches=None, min_tps_user=0.0,
          max_ttft_ms=None, launch="graph": _best_layout(model, system, phase, context, seqlen, batches,
                                                         min_tps_user, max_ttft_ms, launch),
          "Search every parallel layout (TP, and EP with data-parallel attention for MoE; TP x PP x DP for "
          "training) and decode batch size that fits in HBM, and return the best one under a per-user speed or "
          "TTFT target, the decode Pareto front, and each point's step split into compute / hbm / latency / comm "
          "/ launch. Systems: a100, h100, b300, gb300 (NVL72). Takes a minute or more on a cold cache.",
          {"model": {"type": "string", "description": "a list_models name, e.g. llama3-70b, deepseek-v3"},
           "system": {"type": "string", "enum": ["a100", "h100", "b300", "gb300"]},
           "phase": {"type": "string", "enum": ["decode", "prefill", "train"]},
           "context": {"type": "integer", "description": "decode: cached tokens per sequence"},
           "seqlen": {"type": "integer", "description": "prefill / train: tokens per sequence"},
           "batches": {"type": "array", "items": {"type": "integer"},
                       "description": "decode batch sizes to try (default 1, 8, 32, 128, 256)"},
           "min_tps_user": {"type": "number", "description": "decode: minimum tokens/s per user"},
           "max_ttft_ms": {"type": "number", "description": "prefill: maximum time to first token"},
           "launch": {"type": "string", "enum": ["eager", "graph", "megakernel"]}},
          ["model", "system"]),
    _tool("scaling_math",
          lambda model, system="gb300", phase="train", plan=None, **kw: _scaling_math(model, system, phase, plan, kw),
          "Closed-form ('How to Scale Your Model') step time, MFU, throughput, memory and critical batch sizes for "
          "any parallel plan, in milliseconds: every term (matmul FLOPs, weight and KV reads, TP / EP / CP / PP "
          "communication, pipeline bubble, DP / ZeRO / FSDP / HSDP traffic, Adam) with its formula and whether it "
          "is exposed or overlapped. Use it to rank layouts before simulating, or to explain a simulated result.",
          {"model": {"type": "string", "description": "a list_models name"},
           "system": {"type": "string", "description": "gb300 (NVL72), b300, h100, a100, or a system preset"},
           "phase": {"type": "string", "enum": ["train", "prefill", "decode"]},
           "plan": _PLAN, "micro_batch": {"type": "integer"}, "seqlen": {"type": "integer"},
           "micro_batches": {"type": "integer"}, "batch": {"type": "integer"}, "context": {"type": "integer"},
           "dtype": {"type": "string", "enum": ["bf16", "fp8", "fp4"]}},
          ["model"]),
    _tool("check_scaling_math",
          lambda model, system="gb300", phase="train", plan=None, layers=2, **kw: _check_math(
              model, system, phase, plan, layers, kw),
          "Run one plan three ways on a model cut to `layers` layers: the closed-form estimate, a roofline replay "
          "of the exact lowered program (ideal kernels, closed-form collectives, perfect overlap) and the "
          "cycle-level simulator. Returns signed errors and splits sim - replay into kernel efficiency, the "
          "power-capped clock, collective efficiency and lost overlap; plus every collective's simulated vs "
          "closed-form time. Takes 10 s to a few minutes.",
          {"model": {"type": "string"}, "system": {"type": "string"},
           "phase": {"type": "string", "enum": ["train", "prefill", "decode"]}, "plan": _PLAN,
           "layers": {"type": "integer"}, "micro_batch": {"type": "integer"}, "seqlen": {"type": "integer"},
           "micro_batches": {"type": "integer"}, "batch": {"type": "integer"}, "context": {"type": "integer"},
           "optimizer": {"type": "boolean"}},
          ["model"]),
    _tool("scaling_ladder",
          lambda model, system="gb300", phase="train", gpus=None, **kw: _ladder(model, system, phase, gpus, kw),
          "From one GPU to the whole rack in closed form: at each GPU count, the best plan (train: the most tokens/s "
          "per GPU at a fixed global batch over every TP / SP / CP / PP / DP / ZeRO / HSDP plan that fits in HBM; "
          "decode: the most tokens/s per GPU over TP / PP / EP layouts and batches with a per-user speed floor), "
          "with its step time, MFU, memory, dominant term, every term in ms and the scaling efficiency against the "
          "smallest count that fits. Takes well under a second.",
          {"model": {"type": "string", "description": "a list_models name"},
           "system": {"type": "string", "description": "gb300 (NVL72), b300, h100, a100, or a system preset"},
           "phase": {"type": "string", "enum": ["train", "decode"]},
           "gpus": {"type": "array", "items": {"type": "integer"},
                    "description": "GPU counts (default 1, 2, 4, 8, 16, 36, 72; counts above the system are skipped)"},
           "seqlen": {"type": "integer", "description": "train: tokens per sequence (default 8192)"},
           "global_batch": {"type": "integer", "description": "train: sequences per step (default 256)"},
           "context": {"type": "integer", "description": "decode: context length (default 8192)"},
           "min_tps_user": {"type": "number", "description": "decode: minimum tokens/s per user"},
           "dtype": {"type": "string", "enum": ["bf16", "fp8", "fp4"]}},
          ["model"]),
    _tool("simulate_colocated",
          lambda tenants, system="gb300", layers=2, flows=None, alone=True, mode="fast": _colocated(
              tenants, system, layers, flows, alone, mode),
          "Run several models on one rack at the same time: each tenant has its own model, phase, parallel plan and "
          "GPUs (disjoint GPUs = a partitioned rack sharing the NVLink switches; overlapping GPUs = MPS-style "
          "sharing), plus optional flows between tenants (KV caches from a prefill pool to a decode pool, weights "
          "from a trainer to samplers). Returns each tenant's step time together and alone, the slowdown, and each "
          "flow's delivered rate. "
          "Exact mode makes tenants on the same GPU share SMs, L2 and HBM cycle by cycle.",
          {"tenants": {"type": "array", "items": {"type": "object"},
                       "description": "[{name, model, phase: decode|prefill|train, plan: {...}, gpus: [0, 1, ...] or "
                                      "'0-63', batch, context, seqlen, micro_batches, steps}]"},
           "system": {"type": "string", "description": "gb300 (NVL72) or another system key"},
           "layers": {"type": "integer", "description": "layers per tenant model (fewer = faster)"},
           "flows": {"type": "array", "items": {"type": "object"},
                     "description": "[{name, src: gpus, dst: gpus, bytes, times, every_ns, sms, start_ns}]: "
                                    "send/recv from each src GPU to the dst GPU at the same position, one transfer "
                                    "issued every every_ns (offered rate bytes/every_ns per pair)"},
           "alone": {"type": "boolean", "description": "also simulate each tenant alone to measure interference"},
           "mode": {"type": "string", "enum": ["fast", "exact"]}},
          ["tenants"]),
    _tool("compare_runs", lambda run_a, run_b: api.compare(run_a, run_b),
          "Compare two runs: step time and throughput with a significance verdict, per-op and stall deltas.",
          {"run_a": _RUN_ID, "run_b": _RUN_ID}, ["run_a", "run_b"]),
    _tool("list_runs", lambda: api.runs(), "Runs simulated in this session.", {}, []),
    _tool("accuracy", lambda: api.accuracy_table(),
          "How the error bands are derived: per kernel class, typical and worst measured error with sources.", {}, []),
]


def _best_layout(model, system, phase, context, seqlen, batches, min_tps_user, max_ttft_ms, launch) -> dict:
    from . import sweep
    if system not in sweep.SYSTEMS:
        raise ValueError(f"unknown system {system!r} (known: {', '.join(sweep.SYSTEMS)})")
    out = sweep.best_plan(model, system, phase, context=context, seqlen=seqlen,
                          batches=tuple(batches or (1, 8, 32, 128, 256)), min_tps_user=min_tps_user,
                          max_ttft_ms=float("inf") if max_ttft_ms is None else max_ttft_ms, launch=launch)
    return dict(sweep.plan_report(out, system), schema="rackscale-api/v1")


def _plan(plan) -> "Any":
    from .scaling import Plan
    return Plan(**(plan or {}))


def _scaling_math(model, system, phase, plan, kw) -> dict:
    from . import scaling
    est = scaling.estimate(model, scaling.hardware(system), phase, _plan(plan), **kw)
    est["explain"] = scaling.explain(est)
    return dict(est, schema="rackscale-api/v1")


def _check_math(model, system, phase, plan, layers, kw) -> dict:
    from . import scaling
    return dict(scaling.check(model, system, phase, _plan(plan), layers=layers, **kw), schema="rackscale-api/v1")


def _ladder(model, system, phase, gpus, kw) -> dict:
    from . import scaling
    if gpus:
        kw["gpus"] = tuple(gpus)
    return dict(scaling.ladder(model, system, phase, **kw), schema="rackscale-api/v1")


def _gpus(g) -> list:
    if isinstance(g, str):
        out = []
        for part in g.split(","):
            lo, _, hi = part.partition("-")
            out += list(range(int(lo), int(hi or lo) + 1))
        return out
    return list(g)


def _colocated(tenants, system, layers, flows, alone, mode) -> dict:
    from . import colocate as co
    ts = []
    for t in tenants:
        t = dict(t)
        t["plan"] = _plan(t.get("plan"))
        t["gpus"] = _gpus(t.get("gpus", []))
        ts.append(co.Tenant(**t))
    fl = [co.Flow(**{**f, "src": _gpus(f["src"]), "dst": _gpus(f["dst"])}) for f in (flows or [])]
    res = co.run(ts, system, layers=layers, alone=alone, fast=mode != "exact", flows=fl)
    res.pop("report", None)
    res["summary"] = co.summary(res)
    return dict(res, schema="rackscale-api/v1")


_BY_NAME = {t["name"]: t for t in TOOLS}


def specs() -> List[dict]:
    """Tool definitions for LLM tool-use APIs."""
    return [{k: t[k] for k in ("name", "description", "input_schema")} for t in TOOLS]


def call(name: str, args: Dict[str, Any]) -> dict:
    """Run one tool. Errors come back as ``{"error": ...}`` so an agent can correct itself."""
    if name not in _BY_NAME:
        return {"error": f"unknown tool {name!r}"}
    try:
        return _BY_NAME[name]["fn"](**(args or {}))
    except (ValueError, KeyError, TypeError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def call_json(name: str, args: Dict[str, Any], limit: int = 60_000) -> str:
    """``call`` serialised for a model's context, trimmed to ``limit`` characters."""
    text = json.dumps(call(name, args), default=str)
    return text if len(text) <= limit else text[:limit] + '..."(truncated)"'
