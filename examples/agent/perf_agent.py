"""A performance-engineering agent on top of the rackscale tools.

It takes a workload, finds what bounds it, proposes changes, checks each one in the
simulator, and keeps a change only when it beats the error band and meets a latency
limit. Two drivers share the same tools (``rackscale.tools``):

* ``--policy scripted`` (default): a deterministic loop over the simulator's own
  suggested what-ifs. Needs nothing but the simulator.
* ``--policy claude``: Claude drives the tools itself (Anthropic API, manual tool-use
  loop). Needs ``pip install anthropic`` and an API key or ``ant auth login``.

    python examples/agent/perf_agent.py                                   # DeepSeek-V3 decode, HGX B200
    python examples/agent/perf_agent.py --system gb200_nvl72 --ep 72 --batch 64
    python examples/agent/perf_agent.py --policy claude --goal "maximise tokens/s with step time under 2 ms"

The scripted run writes its log (every run, verdict and band) to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from rackscale import tools


def t(tool, **args):
    out = tools.call(tool, args)
    if "error" in out:
        raise RuntimeError(f"{tool}: {out['error']}")
    return out


def fmt_band(b) -> str:
    return f"{b['value']:.0f} (±{b['band_pct']:.0f}%)"


# ---------------------------------------------------------------------------
# scripted policy


def adoptable(lever: dict) -> bool:
    """Software can change the workload and the run, not the hardware."""
    return "hardware" not in lever["what_if"] and "system" not in lever["what_if"]


def scripted(a) -> dict:
    log = {"goal": "max tokens/s", "slo_us": a.slo_us, "rounds": []}
    spec = {"model": a.model, "phase": "decode", "batch": a.batch, "context": a.context,
            "parallel": {"ep": a.ep, "tp": a.tp}, "layers": a.layers}
    base = t("simulate", system=a.system, workload=spec, label="baseline")
    print(f"## Baseline: {base['workload']['name']} on {base['system']['name']}")
    print(f"step {fmt_band(base['step_time_us'])} us, {fmt_band(base['tokens_per_s'])} tokens/s, "
          f"MFU {100 * (base['mfu'] or 0):.1f}%")
    slo = a.slo_us or base["step_time_us"]["value"] * a.slo_factor
    log["slo_us"] = slo
    print(f"latency limit: step <= {slo:.0f} us\n")
    for rnd in range(a.rounds):
        tried = set()
        b = t("find_bottlenecks", run_id=base["run_id"], top=6)
        print(f"### Round {rnd + 1}: {b['summary']}\n")
        rec = {"base": base["run_id"], "summary": b["summary"], "tried": [], "kept": None}
        best = None
        for lv in b["suggested_what_ifs"]:
            key = json.dumps(lv["what_if"], sort_keys=True)
            if not adoptable(lv) or key in tried:
                continue
            tried.add(key)
            w = t("what_if", run_id=base["run_id"], change=lv["what_if"])
            c, r = w["comparison"], w["result"]
            tps = c.get("tokens_per_s") or {}
            ok = r["step_time_us"]["value"] <= slo
            gain = tps.get("delta_pct", -c["step_time_us"]["delta_pct"])
            sig = tps.get("significant", c["difference"]["significant"])
            verdict = "keep" if ok and sig and gain > 0 else ("over latency limit" if not ok else
                                                              "not better" if gain <= 0 else "within error band")
            print(f"- {key}: {c['verdict']}; step {r['step_time_us']['value']:.0f} us -> {verdict}")
            print(f"    why tried: {lv['why']}")
            rec["tried"].append({"what_if": lv["what_if"], "why": lv["why"], "run_id": r["run_id"],
                                 "step_us": r["step_time_us"], "tokens_per_s_delta_pct": gain, "significant": sig,
                                 "verdict": verdict})
            if verdict == "keep" and (best is None or gain > best[0]):
                best = (gain, lv, r)
        if best is None:
            print("\nNo adoptable change beats the error band within the latency limit; stopping.\n")
            log["rounds"].append(rec)
            break
        gain, lv, r = best
        print(f"\n**Kept** {json.dumps(lv['what_if'])}: tokens/s +{gain:.1f}%\n")
        rec["kept"] = {"what_if": lv["what_if"], "run_id": r["run_id"], "gain_pct": gain}
        log["rounds"].append(rec)
        base = r
    # Where the final configuration's time goes, down to the warp roles and SASS.
    b = t("find_bottlenecks", run_id=base["run_id"], top=3)
    top = b["items"][0]
    k = t("inspect_kernel", run_id=base["run_id"], name=top["kernel"], top_pcs=5)
    print(f"### Final: {base['workload']['spec']}")
    print(f"step {fmt_band(base['step_time_us'])} us, {fmt_band(base['tokens_per_s'])} tokens/s")
    print(f"top kernel {top['kernel']}: {k['diagnosis']}")
    for role in k["role_anatomy"][:4]:
        print(f"  role {role['role']:<14} {role['dominant_wait'] or 'issuing':<16} {100 * role['fraction']:.0f}%")
    # What hardware would buy, for the hardware engineers in the room.
    sens = []
    for lv in b["suggested_what_ifs"] + b["step_levers"]:
        if "hardware" in lv["what_if"]:
            w = t("what_if", run_id=base["run_id"], change=lv["what_if"])
            sens.append({"what_if": lv["what_if"], "verdict": w["comparison"]["verdict"]})
            print(f"  hardware sensitivity {json.dumps(lv['what_if']['hardware'])}: {w['comparison']['verdict']}")
    log["final"] = {"run_id": base["run_id"], "spec": base["workload"]["spec"], "step_time_us": base["step_time_us"],
                    "tokens_per_s": base["tokens_per_s"], "top_kernel": k["diagnosis"], "hardware_sensitivity": sens,
                    "provenance": base["provenance"]}
    return log


# ---------------------------------------------------------------------------
# Claude policy

SYSTEM = """You are a GPU performance engineer with a cycle-level rack simulator as your instrument.
Work like an engineer: measure a baseline, find what bounds it, form a hypothesis, test it with the
simulator, and keep only changes whose improvement beats the error band the tools report. Prefer
changes the operator can make (batch, parallelism, precision, overlap, CUDA graphs) and report hardware
sensitivities separately. Drill into the dominant kernel's warp roles before you explain why it is slow.
Finish with a short report: the baseline, each change you tried with its verdict, the final
configuration, the evidence (run ids), and the error bars."""


def claude(a) -> dict:
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic()
    goal = a.goal or (f"Workload: {a.model} decode, batch {a.batch} per rank, context {a.context}, EP {a.ep}, TP {a.tp}, "
                      f"first {a.layers} layers, on {a.system}. Goal: maximise tokens/s while keeping the step time "
                      f"within {a.slo_factor}x of the baseline.")
    messages = [{"role": "user", "content": tools.INSTRUCTIONS + "\n\n" + goal}]
    extra = {} if a.no_fallback else {"betas": ["server-side-fallback-2026-07-01"], "extra_body": {"fallbacks": "default"}}
    calls = []
    for _ in range(a.max_turns):
        resp = client.beta.messages.create(model=a.claude_model, max_tokens=16000, system=SYSTEM, tools=tools.specs(),
                                           output_config={"effort": "high"}, messages=messages, **extra)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"request declined: {resp.stop_details}")
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                print(b.text)
        if resp.stop_reason != "tool_use":
            break
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            t0 = time.time()
            text = tools.call_json(b.name, b.input)
            print(f"  [tool] {b.name}({json.dumps(b.input)[:140]}) {time.time() - t0:.1f}s")
            calls.append({"tool": b.name, "input": b.input})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": text,
                            "is_error": text.startswith('{"error"')})
        messages.append({"role": "user", "content": results})
    return {"goal": goal, "tool_calls": calls}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["scripted", "claude"], default="scripted")
    ap.add_argument("--system", default="hgx_b200_8")
    ap.add_argument("--model", default="deepseek-v3")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--slo-us", type=float, help="step-time limit (default: --slo-factor x baseline)")
    ap.add_argument("--slo-factor", type=float, default=1.5)
    ap.add_argument("--goal", help="free-form goal for --policy claude")
    ap.add_argument("--claude-model", default="claude-opus-5-5")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--no-fallback", action="store_true", help="do not opt into server-side refusal fallback")
    ap.add_argument("--out", default="perf_agent_log.json")
    a = ap.parse_args(argv)
    log = scripted(a) if a.policy == "scripted" else claude(a)
    with open(a.out, "w") as f:
        json.dump(log, f, indent=1, default=str)
    print(f"\nlog: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
