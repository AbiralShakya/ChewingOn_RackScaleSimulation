"""Stable JSON API over the simulator, for scripts, notebooks and agents.

Every function takes JSON-able arguments and returns a JSON-able dict tagged
``"schema": "rackscale-api/v1"``. The MCP server (``rackscale.mcp_server``)
exposes these same functions as tools, so a person and an agent get identical
answers.

    >>> from rackscale import api
    >>> r = api.run("hgx_b200_8", {"model": "deepseek-v3", "phase": "decode", "batch": 32,
    ...                            "context": 4096, "parallel": {"ep": 8}, "layers": 4})
    >>> api.bottlenecks(r["run_id"])["items"][0]["diagnosis"]
    >>> api.what_if(r["run_id"], {"workload": {"dtype": "fp8"}})

Concepts:

* **system**: a preset name (``gb200_nvl72``), a YAML path, YAML text or a dict.
  ``overrides`` change hardware knobs by name (``hardware()["knobs"]``) or by
  path (``gpu.hbm.bw_gbps``); a string value ``"x1.5"`` scales the current one.
* **workload spec**: a dict that names a model and a phase, a capture, a single
  kernel or a raw workload (see ``workload()``). Runs built from a spec can be
  changed field by field in ``what_if``/``sweep``.
* **run**: simulated once and kept in the session under a ``run_id``, a hash of
  everything that went in, so asking twice costs nothing.
* **bands**: every time carries ``lo``/``hi`` from ``rackscale.accuracy``, and
  every run carries provenance (git commit, calibration files and hashes, mode).
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Union

from . import accuracy, analysis, lower, models

SCHEMA = "rackscale-api/v1"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

#: Measured calibrations attached automatically (``calibrated="auto"``). B300 and
#: GB300 carry theirs built in.
CALIBRATIONS = {
    "a100_sxm": "calibration/results/a100-sxm4-80gb-sm80/calibration.json",
    "a100_pcie": "calibration/results/a100-80gb-pcie-sm80/calibration.json",
}
BUILTIN_CALIBRATED = {"b300", "gb300"}

#: Friendly knob names -> path. ``gpu.*`` paths apply to every GPU group;
#: ``link.*`` change the scale-up fabric (links and switch ports together).
KNOBS = {
    "sm_clock_mhz": ("gpu.sm_clock_mhz", "SM clock, MHz"),
    "sm_count": ("gpu.sm_count", "SMs per GPU"),
    "hbm_bw_gbps": ("gpu.hbm.bw_gbps", "HBM bandwidth, GB/s"),
    "hbm_latency_ns": ("gpu.hbm.latency_ns", "idle HBM latency, ns"),
    "hbm_capacity_gb": ("gpu.hbm.capacity_gb", "HBM capacity, GB"),
    "l2_size_bytes": ("gpu.l2.size_bytes", "L2 capacity, bytes"),
    "l2_hit_latency": ("gpu.l2.hit_latency", "L2 hit latency, cycles"),
    "l2_bytes_per_cycle": ("gpu.l2_bytes_per_cycle", "aggregate L2 bandwidth, bytes/cycle"),
    "smem_bytes_per_cycle": ("gpu.pipes.l1_bytes_per_cycle", "shared memory / L1 bandwidth per SM, bytes/cycle"),
    "tensor_bf16_flops_per_cycle_per_sm": ("gpu.tensor.flops_per_cycle_per_sm.bf16", "dense BF16 tensor rate per SM"),
    "tensor_fp8_flops_per_cycle_per_sm": ("gpu.tensor.flops_per_cycle_per_sm.fp8", "dense FP8 tensor rate per SM"),
    "tensor_fp4_flops_per_cycle_per_sm": ("gpu.tensor.flops_per_cycle_per_sm.fp4", "dense FP4 tensor rate per SM"),
    "launch_overhead_ns": ("gpu.launch_overhead_ns", "back-to-back kernel launch gap, ns"),
    "power_cap": ("gpu.power", "board power model; null removes the cap"),
    "nvlink_gbps_per_dir": ("link.gbps_per_dir", "scale-up link bandwidth per direction, GB/s (per link)"),
    "nvlink_latency_ns": ("link.latency_ns", "scale-up link latency, ns"),
    "nic_gbps_per_dir": ("nic.gbps_per_dir", "scale-out NIC bandwidth per direction, GB/s"),
}

PHASES = ("decode", "prefill", "train")
_runs: Dict[str, dict] = {}


def _native():
    from . import _need_native
    return _need_native()


def _out(**kw) -> dict:
    return {"schema": SCHEMA, **kw}


# ---------------------------------------------------------------------------
# hardware and systems


def _system_text(system: Union[str, dict, os.PathLike]) -> str:
    if isinstance(system, dict):
        return json.dumps(system)
    from . import _system_yaml
    return _system_yaml(system)


def _get(d: Any, path: List[str]) -> Any:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _set(d: dict, path: List[str], value: Any) -> None:
    for p in path[:-1]:
        d = d.setdefault(p, {})
    d[path[-1]] = value


def _scaled(value: Any, current: Any, what: str) -> Any:
    """``"x1.5"`` scales the current value; anything else is taken as is."""
    if isinstance(value, str) and re.fullmatch(r"[x*]\s*[0-9.]+(e-?\d+)?", value.strip()):
        if not isinstance(current, (int, float)):
            raise ValueError(f"cannot scale {what}: current value is {current!r}")
        out = current * float(value.strip()[1:])
        return round(out) if isinstance(current, int) else out
    return value


def _fabric_links(topo: dict) -> List[dict]:
    if topo.get("kind") == "multi_node":
        return _fabric_links(topo["intra"])
    return [topo]


def resolve(system: Union[str, dict, os.PathLike], overrides: Optional[dict] = None,
            calibrated: Union[bool, str] = "auto") -> dict:
    """System + overrides -> ``{"yaml", "spec", "gpus", "calibrations", ...}`` (internal shape)."""
    nat = _native()
    base = json.loads(nat.resolve_system(_system_text(system)))
    spec = copy.deepcopy(base["spec"])
    cals = []
    for i, g in enumerate(spec["gpus"]):
        preset = g.get("preset") or ""
        if g.get("calibration"):
            g["calibration"] = _repo_path(g["calibration"])
        elif calibrated and preset in CALIBRATIONS and os.path.exists(os.path.join(REPO, CALIBRATIONS[preset])):
            g["calibration"] = os.path.join(REPO, CALIBRATIONS[preset])
        if g.get("calibration"):
            cals.append({"group": i, "preset": preset, "file": os.path.relpath(g["calibration"], REPO),
                         "sha256": _sha(g["calibration"])})
        elif preset in BUILTIN_CALIBRATED:
            cals.append({"group": i, "preset": preset, "file": "built in (calibration/results/b300-sxm6-ac-sm103)",
                         "sha256": None})
    applied = {}
    for key, value in (overrides or {}).items():
        path = KNOBS.get(key, (key, ""))[0]
        head, *rest = path.split(".")
        if head == "gpu":
            for i, g in enumerate(spec["gpus"]):
                cur = _get(base["gpus"][i], rest)
                v = _scaled(value, cur, key)
                if rest == ["calibration"]:
                    g["calibration"] = _repo_path(v)
                    cals.append({"group": i, "preset": g.get("preset") or "", "file": os.path.relpath(g["calibration"], REPO),
                                 "sha256": _sha(g["calibration"])})
                    applied[f"{key}[group {i}]"] = {"from": None, "to": v}
                    continue
                o = g.setdefault("overrides", {})
                _set(o, rest, v)
                applied[f"{key}[group {i}]"] = {"from": cur, "to": v}
        elif head in ("link", "nic"):
            topo = spec["topology"]
            targets = [topo] if head == "nic" else _fabric_links(topo)
            for t in targets:
                node = t.get(head if head == "nic" else "link")
                if node is None:
                    raise ValueError(f"{key}: this topology has no {head}")
                cur = _get(node, rest)
                v = _scaled(value, cur, key)
                _set(node, rest, v)
                if head == "link" and rest == ["gbps_per_dir"] and "switch" in t:
                    t["switch"]["gbps_per_port_per_dir"] = _scaled(value, t["switch"]["gbps_per_port_per_dir"], key)
                applied[key] = {"from": cur, "to": v}
        elif head == "system":
            cur = _get(spec, rest)
            _set(spec, rest, _scaled(value, cur, key))
            applied[key] = {"from": cur, "to": _get(spec, rest)}
        else:
            raise ValueError(f"unknown knob {key!r}: use one of {sorted(KNOBS)} or a gpu./link./nic./system. path")
    text = json.dumps(spec)
    full = json.loads(nat.resolve_system(text))
    return {"yaml": text, "spec": spec, "gpus": full["gpus"], "num_gpus": full["num_gpus"],
            "switches": full["switches"], "links": full["links"], "calibrations": cals, "overrides": applied}


def _repo_path(p: str) -> str:
    """Absolute path; relative paths that do not exist here resolve against the repository."""
    if os.path.isabs(p) or os.path.exists(p):
        return os.path.abspath(p)
    return os.path.join(REPO, p)


def _sha(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


def _gpu_summary(g: dict) -> dict:
    tf = g["tensor"]["flops_per_cycle_per_sm"]
    clk = g["sm_clock_mhz"]
    peak = {k: v * g["sm_count"] * clk * 1e6 / 1e12 for k, v in tf.items() if k in ("bf16", "fp8", "fp4", "tf32")}
    return {"name": g["name"], "arch": g["arch"], "sm_count": g["sm_count"], "sm_clock_mhz": round(clk, 1),
            "hbm_gbps": g["hbm"]["bw_gbps"], "hbm_gb": g["hbm"]["capacity_gb"], "l2_mb": round(g["l2"]["size_bytes"] / 2**20, 1),
            "peak_dense_tflops": {k: round(v, 1) for k, v in peak.items()},
            "nvlink_gbps_per_dir": g["nvlink"]["gbps_per_port_per_dir"] * g["nvlink"]["ports"],
            "power_model": g.get("power") is not None,
            "calibrated": bool(g.get("opcode_timing")) or any("calibration" in n.lower() and "measured" in n.lower()
                                                               for n in g.get("notes", []))}


def hardware(name: Optional[str] = None) -> dict:
    """Presets, the tunable knobs and how well each GPU is calibrated.

    With ``name`` (a GPU or system preset), the resolved spec in full.
    """
    from . import presets
    p = presets()
    if name:
        if name in p["gpus"]:
            return _out(gpu=name, spec=p["gpus"][name], summary=_gpu_summary(p["gpus"][name]))
        r = resolve(name)
        return _out(system=name, spec=r["spec"], gpus=[_gpu_summary(g) for g in r["gpus"]], num_gpus=r["num_gpus"],
                    switches=r["switches"], calibrations=r["calibrations"])
    gpus = {}
    for k, g in p["gpus"].items():
        s = _gpu_summary(g)
        s["calibrated"] = k in BUILTIN_CALIBRATED or k in CALIBRATIONS
        s["calibration"] = ("built in" if k in BUILTIN_CALIBRATED else CALIBRATIONS.get(k, "none (datasheet preset)"))
        gpus[k] = s
    systems = {}
    for s in p["systems"]:
        r = resolve(s)
        systems[s] = {"name": r["spec"]["name"], "num_gpus": r["num_gpus"], "gpu": r["gpus"][0]["name"],
                      "switches": r["switches"], "topology": r["spec"]["topology"]["kind"]}
    return _out(gpus=gpus, systems=systems,
                knobs={k: {"path": v[0], "meaning": v[1]} for k, v in KNOBS.items()},
                override_syntax="{knob or path: value}; a string 'x1.5' scales the current value; paths start with "
                                "gpu. (every GPU group), link. (scale-up links + switch ports), nic. or system.")


def models_list() -> dict:
    """Model configs the lowering knows, with sizes."""
    out = {}
    for k, m in models.ALL.items():
        out[k] = {"layers": m.layers, "hidden": m.hidden, "heads": m.heads, "kv_heads": m.kv_heads,
                  "active_params_b": round(m.active_params() / 1e9, 2),
                  "moe": None if not m.moe else {"experts": m.moe.experts, "top_k": m.moe.topk},
                  "mla": bool(m.mla), "mamba": bool(m.mamba), "blocks": m.block_list()[:64]}
    return _out(models=out, phases=list(PHASES))


# ---------------------------------------------------------------------------
# workloads


def _with_dtype(wl: dict, dtype: str) -> dict:
    for ops in _all_streams(wl):
        for op in _walk(ops):
            k = op.get("kernel", {})
            if k.get("type") in ("gemm", "grouped_gemm"):
                k["dtype"] = dtype
    return wl


def _all_streams(wl: dict):
    if wl.get("spmd"):
        yield from wl["spmd"]["program"]["streams"]
    for r in wl.get("ranks", []):
        yield from r["streams"]


def _walk(ops: List[dict]):
    for op in ops:
        if op.get("op") == "loop":
            yield from _walk(op.get("body", []))
        else:
            yield op


def build_workload(spec: Union[dict, str]) -> dict:
    """Workload spec -> simulator workload (``docs/WORKLOAD_FORMAT.md``).

    Spec forms:

    * model: ``{"model": "deepseek-v3", "phase": "decode"|"prefill"|"train", "batch": 32,
      "context": 4096 (decode), "seqlen": 4096 (prefill/train), "micro_batch": 1 (train),
      "parallel": {"tp":1,"ep":1,"dp":1,"comm_sms":20}, "layers": 4, "overlap": false,
      "dtype": "bf16"|"fp8"|"fp4"}``
    * one kernel: ``{"kernel": {"type": "gemm", "m": 4096, "n": 4096, "k": 4096}}``
    * capture: ``{"capture": "<dir>", "plan": {"tp": 8}, "mode": "paths", "steps": [0, 1]}``
    * raw: ``{"workload": {...}}`` or a path to a workload JSON/YAML.
    """
    if isinstance(spec, str):
        if os.path.exists(spec):
            with open(spec) as f:
                text = f.read()
            spec = {"workload": json.loads(text) if text.lstrip().startswith("{") else text}
        else:
            spec = json.loads(spec)
    if "workload" in spec:
        wl = spec["workload"]
        if isinstance(wl, str):
            return {"name": "workload", "_text": wl}
        return copy.deepcopy(wl)
    if "kernel" in spec:
        k = spec["kernel"]
        name = spec.get("name") or k.get("type", "kernel")
        return {"name": name, "spmd": {"world": 1, "program": {"streams": [[{"op": "kernel", "name": name, "kernel": k}]]}}}
    if "capture" in spec:
        from . import record, replay
        cap = record.load(spec["capture"])
        steps = tuple(spec["steps"]) if spec.get("steps") else None
        return replay.to_workload(cap, spec.get("plan"), mode=spec.get("mode", "paths"), host=spec.get("host", False),
                                  steps=steps)
    if "model" not in spec:
        raise ValueError("workload spec needs one of: model, kernel, capture, workload")
    if spec["model"] not in models.ALL:
        raise ValueError(f"unknown model {spec['model']!r}; known: {sorted(models.ALL)}")
    m = models.ALL[spec["model"]]
    par = lower.Parallel(**{k: int(v) for k, v in (spec.get("parallel") or {}).items()})
    phase = spec.get("phase", "decode")
    layers = spec.get("layers")
    if phase == "decode":
        wl = lower.decode_step(m, par, int(spec.get("batch", 64)), int(spec.get("context", 4096)),
                               overlap=bool(spec.get("overlap", False)), layers=layers)
    elif phase == "prefill":
        wl = lower.prefill(m, par, int(spec.get("batch", 1)), int(spec.get("seqlen", 4096)), layers=layers)
    elif phase == "train":
        wl = lower.train_step(m, par, int(spec.get("micro_batch", 1)), int(spec.get("seqlen", 4096)), layers=layers)
    else:
        raise ValueError(f"phase must be one of {PHASES}")
    if spec.get("dtype") and spec["dtype"] != "bf16":
        wl = _with_dtype(wl, spec["dtype"])
        wl["name"] += f" {spec['dtype']}"
    return wl


def tokens_per_step(spec: Optional[dict]) -> Optional[int]:
    """Tokens a model-spec step produces (decode) or processes (prefill/train), rack-wide."""
    if not spec or "model" not in spec:
        return None
    par = lower.Parallel(**{k: int(v) for k, v in (spec.get("parallel") or {}).items()})
    replicas = max(1, par.world() // max(1, par.tp))
    phase = spec.get("phase", "decode")
    if phase == "decode":
        return int(spec.get("batch", 64)) * replicas
    if phase == "prefill":
        return int(spec.get("batch", 1)) * int(spec.get("seqlen", 4096)) * replicas
    return int(spec.get("micro_batch", 1)) * int(spec.get("seqlen", 4096)) * replicas


def workload(spec: Union[dict, str]) -> dict:
    """Build a workload from a spec and describe it (ops, kernels, collectives) without simulating."""
    wl = build_workload(spec)
    n_k = n_c = 0
    kinds: Dict[str, int] = {}
    for ops in _all_streams(wl):
        for op in _walk(ops):
            if op.get("op") == "kernel":
                n_k += 1
                t = op["kernel"].get("type", "?")
                kinds[t] = kinds.get(t, 0) + 1
            elif op.get("op") == "collective":
                n_c += 1
    world = wl["spmd"]["world"] if wl.get("spmd") else len(wl.get("ranks", []))
    return _out(name=wl.get("name"), world=world, kernel_ops=n_k, collective_ops=n_c, kernel_types=kinds,
                model_flops=wl.get("model_flops"), tokens_per_step=tokens_per_step(spec if isinstance(spec, dict) else None))


# ---------------------------------------------------------------------------
# runs


def _git() -> dict:
    try:
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return {"commit": sha or None, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _kernel_index(sys_yaml: str, wl: dict) -> Dict[str, dict]:
    """Report kernel name -> {spec, ops, kind}, by building each distinct kernel once."""
    nat = _native()
    idx: Dict[str, dict] = {}
    seen: Dict[str, dict] = {}
    lib = wl.get("kernels", {})
    for ops in _all_streams(wl):
        for op in _walk(ops):
            if op.get("op") != "kernel":
                continue
            k = op["kernel"]
            if k.get("type") == "ref":
                k = lib.get(k.get("name"), k)
            key = json.dumps(k, sort_keys=True)
            if key not in seen:
                info = None
                if k.get("type") != "fixed":
                    try:
                        info = json.loads(nat.kernel_info(sys_yaml, json.dumps(k), REPO))
                    except ValueError:
                        pass
                seen[key] = info or {"name": op.get("name") or k.get("type", "?")}
            info = seen[key]
            name = info["name"]
            e = idx.setdefault(name, {"spec": k, "kind": k.get("type"), "ops": [], "info": info})
            if op.get("name") and len(e["ops"]) < 8 and op["name"] not in e["ops"]:
                e["ops"].append(op["name"])
    return idx


def _summarize(rec: dict) -> dict:
    rep = rec["report"]
    sysr = rec["system"]
    world = rep["world"]
    span = world * rep["makespan_us"] or 1.0
    calibrated = [bool(c) for c in _group_calibration(sysr)]
    parts: Dict[tuple, float] = {}
    for k in rep["kernels"]:
        kind = rec["kernels"].get(k["name"], {}).get("kind")
        cls = accuracy.kernel_class(kind, all(calibrated))
        parts[cls] = parts.get(cls, 0.0) + k["total_us"] / span
    coll = sum(r["collective_us"] for r in rep["ranks"]) / span
    idle = rep["bubbles"]["gpu_idle_fraction"]
    acc = accuracy.run_band(parts, coll, idle, rec["mode"] == "fast")
    step = rep["makespan_us"]
    tok = tokens_per_step(rec.get("spec"))
    out = _out(
        run_id=rec["run_id"], label=rec.get("label"), cached=rec.get("cached", False),
        system={"name": rep["system"], "num_gpus": sysr["num_gpus"], "gpus": sorted({g["name"] for g in sysr["gpus"]}),
                "overrides": sysr["overrides"]},
        workload={"name": rep["workload"], "world": world, "spec": rec.get("spec")},
        mode=rec["mode"],
        step_time_us=accuracy.band(step, acc["typical_pct"]),
        mfu=rep.get("mfu"),
        gpu_idle_fraction=idle,
        tensor_idle_fraction=rep["bubbles"]["tensor_idle_fraction"],
        sched_anatomy=dict(sorted(rep["bubbles"]["sched"].items(), key=lambda kv: -kv[1])[:8]),
        top_kernels=[{"name": k["name"], "share_of_gpu_time": round(k["total_us"] / span, 4), "count": k["count"],
                      "mean_us": round(k["mean_us"], 2), "tflops": round(k["tflops"], 1), "gbps": round(k["gbps"]),
                      "tensor_util": round(k["tensor_util"], 3), "ops": rec["kernels"].get(k["name"], {}).get("ops", [])[:3]}
                     for k in rep["kernels"][:10]],
        collectives=[{"kind": c["kind"], "algo": c["algo"], "bytes": c["bytes"], "ranks": c["ranks"],
                      "busbw_gbps": round(c["busbw_gbps"], 1)} for c in rep["collectives"][:6]],
        accuracy=acc,
        warnings=list(rep.get("warnings", [])) + _caveats(rec, acc),
        provenance=rec["provenance"],
    )
    if tok:
        out["tokens_per_step"] = tok
        out["tokens_per_s"] = accuracy.band(tok / step * 1e6, acc["typical_pct"])
        if rec["spec"].get("phase", "decode") == "decode":
            out["us_per_output_token_per_seq"] = round(step, 2)
    return out


def _caveats(rec: dict, acc: dict) -> List[str]:
    out = []
    spec = rec.get("spec") or {}
    if spec.get("layers") and spec.get("model") in models.ALL and spec["layers"] < models.ALL[spec["model"]].layers:
        out.append(f"only {spec['layers']} of {models.ALL[spec['model']].layers} layers simulated: step time is for "
                   "those layers (scale per layer, mind dense vs MoE layers)")
    if any(t["part"] == "kernels:uncalibrated" and t["share"] > 0.2 for t in acc["terms"]):
        out.append("most kernel time is on a GPU without measured calibration: treat absolute times as estimates and "
                   "rely on comparisons between runs")
    if rec["mode"] == "fast":
        out.append("fast mode ignores cross-kernel L2 reuse and CTA interleaving of concurrent kernels (about +1% on "
                   "the DeepSeek demos); use mode='exact' to confirm a close call")
    return out


def _group_calibration(sysr: dict) -> List[Optional[str]]:
    by_group = {c["group"]: c["file"] for c in sysr["calibrations"]}
    return [by_group.get(i) for i in range(len(sysr["spec"]["gpus"]))]


def _run_id(sys_yaml: str, wl_text: str, cfg: dict) -> str:
    h = hashlib.sha256()
    for part in (sys_yaml, wl_text, json.dumps(cfg, sort_keys=True), _git().get("commit") or ""):
        h.update(part.encode())
        h.update(b"\0")
    return "run_" + h.hexdigest()[:12]


def run(system: Union[str, dict], workload: Union[dict, str], *, mode: str = "fast",
        overrides: Optional[dict] = None, calibrated: Union[bool, str] = "auto",
        launch_gap_ns: Optional[float] = None, label: Optional[str] = None, trace: Optional[str] = None) -> dict:
    """Simulate ``workload`` (a spec, see ``build_workload``) on ``system`` and keep the result.

    ``mode``: ``fast`` (each distinct kernel simulated once, reused; ~1% from exact) or
    ``exact`` (full co-simulation of every rank). Returns the summary; the full
    report stays in the session (``report(run_id)``).
    """
    from . import simulate
    if mode not in ("fast", "exact"):
        raise ValueError("mode must be 'fast' or 'exact'")
    raw = not isinstance(workload, dict) or "spmd" in workload or "ranks" in workload or "workload" in workload
    spec = None if raw else workload
    sysr = resolve(system, overrides, calibrated)
    wl = build_workload(workload if isinstance(workload, str) or not raw or "workload" in workload
                        else {"workload": workload})
    wl_text = wl.pop("_text", None) or json.dumps(wl)
    cfg = {"memoize_kernels": mode == "fast", "launch_gap_ns": launch_gap_ns}
    rid = _run_id(sysr["yaml"], wl_text, cfg)
    if rid in _runs and not trace:
        rec = _runs[rid]
        rec["cached"] = True
        if label:
            rec["label"] = label
        return _summarize(rec)
    cwd = os.getcwd()
    try:
        os.chdir(REPO)  # workload paths (recordings, SASS) are repository-relative
        rep = simulate(sysr["yaml"], wl_text, fast=mode == "fast", launch_gap_ns=launch_gap_ns, trace=trace)
    finally:
        os.chdir(cwd)
    from . import _native as nat
    rec = {
        "run_id": rid, "label": label, "spec": spec, "system_input": system,
        "overrides": dict(overrides or {}), "calibrated": calibrated, "mode": mode, "launch_gap_ns": launch_gap_ns,
        "system": sysr, "workload": wl if wl_text.lstrip().startswith("{") else None, "report": rep.raw,
        "kernels": _kernel_index(sysr["yaml"], wl) if wl.get("spmd") or wl.get("ranks") else {},
        "provenance": {
            "rackscale": getattr(nat, "__version__", None), "git": _git(), "mode": mode,
            "config": {k: v for k, v in cfg.items() if v is not None},
            "system_sha256": hashlib.sha256(sysr["yaml"].encode()).hexdigest()[:16],
            "workload_sha256": hashlib.sha256(wl_text.encode()).hexdigest()[:16],
            "calibrations": sysr["calibrations"], "overrides": sysr["overrides"],
            "simulated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "wallclock_s": round(rep.raw["sim"]["wallclock_s"], 2),
            "warp_instructions": rep.raw["sim"]["warp_instructions"],
        },
    }
    if trace:
        rec["provenance"]["trace"] = os.path.abspath(trace)
    _runs[rid] = rec
    return _summarize(rec)


def _rec(run_id: str) -> dict:
    if run_id not in _runs:
        raise KeyError(f"unknown run {run_id!r}; runs in this session: {sorted(_runs)}")
    return _runs[run_id]


def runs() -> dict:
    """Runs kept in this session."""
    return _out(runs=[{"run_id": r["run_id"], "label": r.get("label"), "system": r["report"]["system"],
                       "workload": r["report"]["workload"], "mode": r["mode"], "step_time_us": r["report"]["makespan_us"],
                       "overrides": r["overrides"]} for r in _runs.values()])


def summary(run_id: str) -> dict:
    return _summarize(_rec(run_id))


def report(run_id: str) -> dict:
    """The full simulator report of a run (large)."""
    return _out(run_id=run_id, report=_rec(run_id)["report"])


# ---------------------------------------------------------------------------
# analysis


def bottlenecks(run_id: str, top: int = 8) -> dict:
    """Where the step's time goes and why: each kernel classed as tensor-, HBM-, latency-
    or pipeline-bound from its achieved rates and stall anatomy, with concrete what-ifs."""
    rec = _rec(run_id)
    return _out(run_id=run_id, **analysis.bottlenecks(rec, top))


def kernel(run_id: Optional[str] = None, name: Optional[str] = None, *, system: Union[str, dict, None] = None,
           spec: Optional[dict] = None, overrides: Optional[dict] = None, rank: int = 0, top_pcs: int = 15) -> dict:
    """Drill into one kernel: per-warp-role stall anatomy and the SASS instructions warps wait on.

    Either name a kernel of a run (``run_id`` + ``name``, a substring of the kernel or op
    name) or give ``system`` + a kernel ``spec`` (``{"type": "gemm", ...}``, a ``.rpath``, SASS).
    The kernel runs alone on a cold GPU, as Nsight Compute times it.
    """
    if run_id:
        rec = _rec(run_id)
        sysr = resolve(rec["system_input"], {**rec["overrides"], **(overrides or {})}, rec["calibrated"])
        if spec is None:
            if not name:
                raise ValueError("name a kernel of the run")
            hits = [(n, e) for n, e in rec["kernels"].items() if name in n or any(name in o for o in e["ops"])]
            if not hits:
                raise KeyError(f"no kernel matching {name!r}; kernels: {list(rec['kernels'])[:20]}")
            hits.sort(key=lambda h: -next((k["total_us"] for k in rec["report"]["kernels"] if k["name"] == h[0]), 0))
            spec = hits[0][1]["spec"]
    else:
        if system is None or spec is None:
            raise ValueError("give run_id + name, or system + spec")
        sysr = resolve(system, overrides)
    if spec.get("type") in ("fixed", "ref"):
        raise ValueError("this kernel has a fixed (measured) duration and no SASS to inspect")
    d = json.loads(_native().kernel_detail(sysr["yaml"], json.dumps(spec), REPO, rank, top_pcs))
    g = sysr["gpus"][rank]
    cal = bool(_group_calibration(sysr)[0]) or bool(g.get("opcode_timing"))
    cls = accuracy.kernel_class(spec.get("type"), cal)
    d["time_us"] = accuracy.band(d["time_us"], accuracy.CLASSES[cls]["typical_pct"])
    d["accuracy_class"] = {"class": cls[0], "calibrated_gpu": cls[1], **accuracy.CLASSES[cls]}
    d["spec"] = spec
    d.update(analysis.kernel_diagnosis(d, g, spec))
    return _out(**d)


# ---------------------------------------------------------------------------
# what-ifs


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _change_kind(change: dict) -> str:
    if "system" in change:
        return "system"
    if "workload" in change:
        return "workload"
    if "hardware" in change:
        return "hardware"
    return "run"


def what_if(run_id: str, change: dict, *, label: Optional[str] = None) -> dict:
    """Re-run ``run_id`` with one change and say whether it beats the error bar.

    ``change`` keys (any combination):

    * ``"workload"``: fields merged into the workload spec (``{"batch": 64}``,
      ``{"parallel": {"ep": 16}}``, ``{"dtype": "fp8"}``, ``{"overlap": true}``)
    * ``"hardware"``: knob overrides added to the run's (``{"hbm_bw_gbps": "x1.5"}``)
    * ``"system"``: another system (preset, YAML or dict)
    * ``"mode"`` / ``"launch_gap_ns"``: run settings
    """
    base = _rec(run_id)
    spec = base["spec"]
    if "workload" in change:
        if spec is None:
            raise ValueError("this run was given a raw workload; only hardware/system/run changes apply")
        spec = _merge(spec, change["workload"])
    new = run(change.get("system", base["system_input"]), spec if spec is not None else base["workload"],
              mode=change.get("mode", base["mode"]), overrides={**base["overrides"], **change.get("hardware", {})},
              calibrated=base["calibrated"], launch_gap_ns=change.get("launch_gap_ns", base["launch_gap_ns"]),
              label=label or json.dumps(change, sort_keys=True))
    cmp = compare(run_id, new["run_id"], change=_change_kind(change))
    return _out(change=change, base=_summarize(base)["step_time_us"], result=new, comparison=cmp)


def compare(run_a: str, run_b: str, *, change: Optional[str] = None) -> dict:
    """Step time, throughput, per-kernel and stall-anatomy deltas between two runs, with a
    significance verdict against the combined error band."""
    a, b = _rec(run_a), _rec(run_b)
    sa, sb = _summarize(a), _summarize(b)
    kind = change or _infer_change(a, b)
    d = accuracy.diff_band(sa["step_time_us"]["value"], sa["step_time_us"]["band_pct"],
                           sb["step_time_us"]["value"], sb["step_time_us"]["band_pct"], kind)
    ka, kb = _by_op(a), _by_op(b)
    rows = []
    for n in set(ka) | set(kb):
        ra, rb = ka.get(n, {}), kb.get(n, {})
        ta, tb = ra.get("us", 0.0) / a["report"]["world"], rb.get("us", 0.0) / b["report"]["world"]
        rows.append({"op": n, "kernel_a": ra.get("kernel"), "kernel_b": rb.get("kernel"), "a_us_per_gpu": round(ta, 2),
                     "b_us_per_gpu": round(tb, 2), "delta_us": round(tb - ta, 2)})
    rows.sort(key=lambda r: -abs(r["delta_us"]))
    anat = {k: round(b["report"]["bubbles"]["sched"].get(k, 0) - a["report"]["bubbles"]["sched"].get(k, 0), 4)
            for k in set(a["report"]["bubbles"]["sched"]) | set(b["report"]["bubbles"]["sched"])}
    out = _out(a=run_a, b=run_b, step_time_us={"a": sa["step_time_us"], "b": sb["step_time_us"],
                                               "delta_pct": round(100 * d["delta_us"] / sa["step_time_us"]["value"], 2)},
               difference=d, kernels=rows[:12], sched_anatomy_delta=dict(sorted(anat.items(), key=lambda kv: -abs(kv[1]))[:8]),
               gpu_idle_delta=round(b["report"]["bubbles"]["gpu_idle_fraction"] - a["report"]["bubbles"]["gpu_idle_fraction"], 4))
    verdict = "faster" if d["delta_us"] < 0 else "slower"
    out["verdict"] = (f"b is {abs(out['step_time_us']['delta_pct']):.1f}% {verdict} per step"
                      + ("" if d["significant"] else " (within the error band: not a reliable difference)"))
    if "tokens_per_s" in sa and "tokens_per_s" in sb:
        ta, tb = sa["tokens_per_s"]["value"], sb["tokens_per_s"]["value"]
        # Token counts are exact, so the throughput ratio carries only the step-time
        # uncertainty: the relative band of the step-time difference.
        rel = d["band_us"] / sa["step_time_us"]["value"]
        sig = abs(math.log(tb / ta)) > rel
        out["tokens_per_s"] = {"a": round(ta), "b": round(tb), "delta_pct": round(100 * (tb - ta) / ta, 2),
                               "significant": sig}
        out["verdict"] += (f"; throughput {'up' if tb > ta else 'down'} {abs(100 * (tb - ta) / ta):.1f}%"
                           + ("" if sig else " (within the error band)"))
    return out


def _by_op(rec: dict) -> Dict[str, dict]:
    """Kernel time keyed by the workload ops that launched it, so runs whose kernels are
    renamed by a change (dtype, tile choice) still line up."""
    out: Dict[str, dict] = {}
    for k in rec["report"]["kernels"]:
        ops = rec["kernels"].get(k["name"], {}).get("ops") or [k["name"]]
        key = "+".join(ops[:3])
        e = out.setdefault(key, {"us": 0.0, "kernel": k["name"]})
        e["us"] += k["total_us"]
    return out


def _infer_change(a: dict, b: dict) -> str:
    if a["system_input"] != b["system_input"]:
        return "system"
    if a["spec"] != b["spec"] or a.get("workload") != b.get("workload"):
        return "workload"
    if a["overrides"] != b["overrides"]:
        return "hardware"
    return "run"


def sweep(run_id: str, key: str, values: List[Any], *, objective: str = "step_time") -> dict:
    """Re-run ``run_id`` once per value of ``key`` and tabulate.

    ``key``: ``workload.<field>`` (``workload.batch``, ``workload.parallel.ep``),
    ``hardware.<knob or path>`` (``hardware.hbm_bw_gbps``), ``system``, ``mode``,
    ``launch_gap_ns``. ``objective``: ``step_time`` (lower is better) or
    ``tokens_per_s`` (higher is better, model specs only).
    """
    head, _, rest = key.partition(".")
    rows = []
    for v in values:
        if head == "workload":
            ch: Dict[str, Any] = {}
            _set(ch, rest.split("."), v)
            change = {"workload": ch}
        elif head == "hardware":
            change = {"hardware": {rest: v}}
        elif head in ("system", "mode", "launch_gap_ns"):
            change = {head: v}
        else:
            raise ValueError("key must start with workload., hardware., or be system/mode/launch_gap_ns")
        w = what_if(run_id, change, label=f"{key}={v}")
        r = w["result"]
        c = w["comparison"]
        rows.append({"value": v, "run_id": r["run_id"], "step_time_us": r["step_time_us"],
                     "tokens_per_s": r.get("tokens_per_s"), "mfu": r["mfu"], "gpu_idle_fraction": r["gpu_idle_fraction"],
                     "top_kernel": r["top_kernels"][0]["name"] if r["top_kernels"] else None,
                     "step_delta_pct_vs_base": c["step_time_us"]["delta_pct"],
                     "tokens_per_s_delta_pct_vs_base": (c.get("tokens_per_s") or {}).get("delta_pct"),
                     "significant": (c["tokens_per_s"]["significant"] if objective == "tokens_per_s" and c.get("tokens_per_s")
                                     else c["difference"]["significant"])})
    if objective == "tokens_per_s" and all(r["tokens_per_s"] for r in rows):
        best = max(rows, key=lambda r: r["tokens_per_s"]["value"])
    else:
        best = min(rows, key=lambda r: r["step_time_us"]["value"])
    return _out(base=run_id, key=key, objective=objective, rows=rows, best={"value": best["value"], "run_id": best["run_id"]})


def accuracy_table() -> dict:
    """How the error bands are derived, with the evidence behind each number."""
    return _out(**accuracy.table())
