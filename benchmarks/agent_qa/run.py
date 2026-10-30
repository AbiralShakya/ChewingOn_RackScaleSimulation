"""Ask Claude every question with the rackscale tools and without them, and grade.

Uses the Claude Code CLI headless (``claude -p``), so it runs wherever Claude
Code is signed in, with no API key in the environment:

* ``tools``: only the rackscale MCP server (``rackscale-mcp``); no shell, no
  files, no web, so every number comes from the model or the tools;
* ``none``: no tools at all; the model answers from what it knows.

Both conditions get the same system prompt (the tools condition adds one sentence saying the
tools exist and should ground the answer) and run in an empty directory.

    python benchmarks/agent_qa/run.py --model sonnet [--conditions tools none] [--jobs 3] [ids...]
    python benchmarks/agent_qa/run.py --grade-only --model sonnet

Answers land in results/<model>/<condition>/<id>.json; the graded table in
results/<model>/summary.md and summary.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from questions import HERE, QUESTIONS, ROOT, schema_text  # noqa: E402

SYSTEM = (
    "You are an expert ML systems and GPU performance engineer. Answer the question about GPU, node or rack-scale "
    "performance as accurately as you can. Give a short explanation of the mechanism, then end your reply with one "
    "JSON object in a ```json fenced block with exactly the keys asked for. Numbers must be plain numbers (no units, "
    "no ranges). If you are unsure, still give your best single estimate."
)

# The session's own identity must not leak into the child runs.
_ENV_DROP = ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID", "CLAUDE_CODE_TEE_SDK_STDOUT",
             "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_CHILD_SESSION",
             "CLAUDE_CODE_PROJECTS_SESSION", "CLAUDE_CODE_COORDINATOR_EXTRA_TOOLS", "CLAUDE_CODE_TERMINAL_MCP_TOOLS")


def _mcp_config(path: str, tools: bool) -> str:
    exe = shutil.which("rackscale-mcp") or os.path.join(ROOT, ".venv", "bin", "rackscale-mcp")
    cfg = {"mcpServers": {"rackscale": {"command": exe, "args": []}}} if tools else {"mcpServers": {}}
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


TOOLS_NOTE = (
    " You have the rackscale tools: a cycle-level GPU and rack simulator calibrated on real hardware, with the "
    "closed forms beside its numbers. Use them to ground your answer rather than answering from memory."
)


def ask(q: dict, condition: str, model: str, timeout: int) -> dict:
    prompt = f"{q['prompt']}\n\nEnd with a ```json block: {schema_text(q)}"
    with tempfile.TemporaryDirectory(prefix="agentqa-") as cwd:
        system = SYSTEM + (TOOLS_NOTE if condition == "tools" else "")
        cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--system-prompt", system,
               "--tools", "", "--strict-mcp-config", "--mcp-config",
               _mcp_config(os.path.join(cwd, "mcp.json"), condition == "tools")]
        if condition == "tools":
            cmd += ["--allowedTools", "mcp__rackscale"]
        env = {k: v for k, v in os.environ.items() if k not in _ENV_DROP}
        # Claude Code gives an MCP tool call 60 s by default; plan searches and full-model
        # simulations take minutes.
        env["MCP_TOOL_TIMEOUT"] = str(timeout * 1000)
        t = time.time()
        try:
            p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
            out = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"result": p.stdout, "is_error": True,
                                                                                 "stderr": p.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            out = {"result": "", "is_error": True, "error": f"timeout after {timeout}s"}
    keep = ("result", "is_error", "num_turns", "total_cost_usd", "duration_ms", "error", "stderr", "permission_denials")
    rec = {k: out.get(k) for k in keep if k in out}
    rec.update(id=q["id"], condition=condition, model=model, wall_s=round(time.time() - t, 1),
               answer=parse(rec.get("result") or ""))
    return rec


def parse(text: str) -> dict:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{[^{}]*\})", text, re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return {}


# ---------------------------------------------------------------------------
# grading


def _num(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(\.\d+)?([eE]-?\d+)?", x.replace(",", ""))
        return float(m.group(0)) if m else None
    return None


def grade_field(spec: dict, got, ref) -> dict:
    if spec["type"] == "choice":
        g = got
        if isinstance(ref, (int, float)):
            g = _num(got)
            ok = g is not None and float(g) == float(ref)
        else:
            ok = str(got).strip().lower().replace(" ", "_") == str(ref).lower()
        return {"got": got, "ref": ref, "score": 1.0 if ok else 0.0}
    g = _num(got)
    if g is None:
        return {"got": got, "ref": ref, "score": 0.0, "rel_err": None}
    rel = (g - ref) / ref if ref else float("inf")
    tol = spec["tol"]
    score = 1.0 if abs(rel) <= tol else 0.5 if abs(rel) <= 2 * tol else 0.0
    return {"got": g, "ref": ref, "rel_err": round(rel, 4), "score": score}


def grade(model: str) -> dict:
    with open(os.path.join(HERE, "refs.json")) as f:
        refs = json.load(f)["refs"]
    base = os.path.join(HERE, "results", model)
    rows = []
    for q in QUESTIONS:
        row = {"id": q["id"], "category": q["category"], "truth": q["truth"]}
        for cond in ("tools", "none"):
            path = os.path.join(base, cond, f"{q['id']}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                rec = json.load(f)
            fields = {k: grade_field(s, rec["answer"].get(k), refs[q["id"]][k]) for k, s in q["fields"].items()}
            row[cond] = {"score": round(sum(v["score"] for v in fields.values()) / len(fields), 3), "fields": fields,
                         "turns": rec.get("num_turns"), "cost_usd": rec.get("total_cost_usd"), "wall_s": rec.get("wall_s"),
                         "error": rec.get("is_error")}
        rows.append(row)
    summary = {"model": model, "rows": rows, "totals": {}}
    for cond in ("tools", "none"):
        got = [r[cond] for r in rows if cond in r]
        if not got:
            continue
        tot = {"questions": len(got), "score": round(sum(g["score"] for g in got), 2),
               "cost_usd": round(sum(g["cost_usd"] or 0 for g in got), 3)}
        for truth in ("measured", "simulated", "closed_form"):
            sel = [r[cond]["score"] for r in rows if cond in r and r["truth"] == truth]
            tot[truth] = f"{sum(sel):g} / {len(sel)}"
        summary["totals"][cond] = tot
    with open(os.path.join(base, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(base, "summary.md"), "w") as f:
        f.write(markdown(summary))
    return summary


def _fmt(fields: dict) -> str:
    out = []
    for k, v in fields.items():
        if v.get("rel_err") is not None:
            out.append(f"{k} {v['got']:.4g} ({100 * v['rel_err']:+.0f}%)")
        else:
            out.append(f"{k} {v['got']}" + ("" if v["score"] else " ✗"))
    return "; ".join(out)


def markdown(s: dict) -> str:
    lines = [f"# Rack-scale questions: {s['model']} with and without rackscale\n",
             "| | with tools | without tools |", "|---|---|---|"]
    t = s["totals"]
    for key in ("score", "measured", "simulated", "closed_form", "cost_usd"):
        lines.append(f"| {key} | {t.get('tools', {}).get(key, '')} | {t.get('none', {}).get(key, '')} |")
    lines += ["", "| question | truth | reference | with tools | score | without tools | score |", "|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        ref = "; ".join(f"{k} {v['ref']:.4g}" if isinstance(v["ref"], float) else f"{k} {v['ref']}"
                        for k, v in (r.get("tools") or r.get("none") or {"fields": {}})["fields"].items())
        a, b = r.get("tools"), r.get("none")
        lines.append(f"| {r['id']} | {r['truth']} | {ref} | {_fmt(a['fields']) if a else ''} | {a['score'] if a else ''} "
                     f"| {_fmt(b['fields']) if b else ''} | {b['score'] if b else ''} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--conditions", nargs="+", default=["tools", "none"])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--redo", action="store_true", help="re-ask questions that already have an answer")
    a = ap.parse_args()
    if not a.grade_only:
        jobs = []
        for q in QUESTIONS:
            if a.ids and q["id"] not in a.ids:
                continue
            for cond in a.conditions:
                path = os.path.join(HERE, "results", a.model, cond, f"{q['id']}.json")
                if a.redo or not os.path.exists(path):
                    jobs.append((q, cond, path))

        def go(job):
            q, cond, path = job
            rec = ask(q, cond, a.model, a.timeout)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(rec, f, indent=1)
            print(f"{q['id']:<18} {cond:<5} {rec['wall_s']:7.1f}s turns {rec.get('num_turns')}  {rec['answer']}",
                  flush=True)

        with cf.ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(go, jobs))
    s = grade(a.model)
    print(markdown(s))


if __name__ == "__main__":
    main()
