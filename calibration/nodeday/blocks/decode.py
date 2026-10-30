"""Measure decode steps on a real serving engine (vLLM, dummy weights).

    python decode.py --case "decode TP8 b64 (70B)" --layers full --out tp8_70b_full.json
    python decode.py --all --outdir results/decode

Each case starts ``vllm serve`` with random ("dummy") weights in the case's
layout, sends ``concurrency`` requests at once (random prompts of the case's
context, ``--output`` tokens each, EOS ignored) with ``vllm bench serve``, and
reports the median inter-token latency: with every sequence decoding, one
inter-token gap is one decode step of the whole batch. The full-depth runs are
the comparison; 2- and 4-layer runs match chapter 11's truncated cases but are
dominated by the engine's per-step CPU work, which the simulator does not
model, and are kept to measure exactly that overhead.

Llama configs are shipped in ``../hf`` so no gated download is needed; the
tokenizer comes from ``$NODEDAY_LLAMA_TOKENIZER`` (only prompt length matters).
Extra server flags go in ``$VLLM_EXTRA_ARGS``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from cases import DECODE  # noqa: E402

TOKENIZER = os.environ.get("NODEDAY_LLAMA_TOKENIZER", "NousResearch/Meta-Llama-3.1-8B")
PORT = int(os.environ.get("NODEDAY_PORT", "8711"))


def model_args(case: dict, layers) -> list:
    hf = case["hf"]
    local = os.path.join(HERE, "..", "hf", hf)
    args = [os.path.abspath(local)] if os.path.isdir(local) else [hf]
    if os.path.isdir(local):
        args += ["--tokenizer", TOKENIZER]
    over = {}
    if layers != "full":
        over["num_hidden_layers"] = int(layers)
    if "DeepSeek" in hf:
        over["num_nextn_predict_layers"] = 0  # no MTP head: the simulator decodes one token per step
    if over:
        args += ["--hf-overrides", json.dumps(over)]
    return args


def serve_cmd(case: dict, layers, max_len: int) -> list:
    cmd = ["vllm", "serve"] + model_args(case, layers) + [
        "--load-format", "dummy", "--port", str(PORT), "--max-model-len", str(max_len),
        "--max-num-seqs", str(case["concurrency"] // case["dp"]), "--gpu-memory-utilization", "0.90",
        "--tensor-parallel-size", str(case["tp"])]
    if case["dp"] > 1:
        cmd += ["--data-parallel-size", str(case["dp"])]
    if case["ep"]:
        cmd += ["--enable-expert-parallel"]
    return cmd + shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))


def wait_ready(proc, timeout: float) -> bool:
    t_end = time.time() + timeout
    while time.time() < t_end:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def run(case: dict, layers, out: str, output_len: int, startup: float) -> dict:
    ctx = case["spec"]["context"]
    log = os.path.splitext(out)[0] + ".server.log"
    cmd = serve_cmd(case, layers, ctx + output_len + 64)
    res = dict(case=case["name"], layers=layers, server_cmd=cmd, context=ctx, output_len=output_len,
               concurrency=case["concurrency"])
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not wait_ready(proc, startup):
            res["error"] = f"server did not come up (see {os.path.basename(log)})"
            return res
        model = model_args(case, layers)[0]
        bench = ["vllm", "bench", "serve", "--backend", "vllm", "--port", str(PORT), "--model", model,
                 "--dataset-name", "random", "--random-input-len", str(ctx), "--random-output-len", str(output_len),
                 "--num-prompts", str(case["concurrency"]), "--max-concurrency", str(case["concurrency"]),
                 "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl", "--metric-percentiles", "10,50,90,99",
                 "--save-result", "--result-dir", os.path.dirname(os.path.abspath(out)),
                 "--result-filename", os.path.basename(out) + ".bench.json"]
        if os.path.isdir(os.path.join(HERE, "..", "hf", case["hf"])):
            bench += ["--tokenizer", TOKENIZER]
        res["bench_cmd"] = bench
        p = subprocess.run(bench, capture_output=True, text=True)
        res["bench_tail"] = (p.stdout + p.stderr)[-3000:]
        bj = os.path.join(os.path.dirname(os.path.abspath(out)), os.path.basename(out) + ".bench.json")
        if os.path.exists(bj):
            b = json.load(open(bj))
            for k in ("median_itl_ms", "p10_itl_ms", "p90_itl_ms", "p99_itl_ms", "median_tpot_ms", "mean_tpot_ms",
                      "median_ttft_ms", "output_throughput", "completed"):
                if k in b:
                    res[k] = b[k]
            if "median_itl_ms" in res:
                res["step_us"] = res["median_itl_ms"] * 1e3
        else:
            res["error"] = "bench produced no result file"
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(5)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--layers", default="full")
    ap.add_argument("--all", action="store_true", help="every case at every depth in cases.DECODE")
    ap.add_argument("--gpus", type=int, default=8, help="GPUs on this node; cases needing more are skipped")
    ap.add_argument("--skip", default="", help="comma-separated GPU keys from cases.DECODE['skip'] (e.g. h100)")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--startup", type=float, default=1800.0, help="seconds to wait for the server")
    ap.add_argument("--out", default="")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    todo = []
    for c in DECODE:
        if a.all or c["name"] == a.case:
            if a.skip and set(a.skip.split(",")) & set(c.get("skip", ())):
                continue
            if c["tp"] * c["dp"] > a.gpus:
                continue
            for layers in (c["layers"] if a.all else [a.layers]):
                todo.append((c, layers))
    if not todo:
        sys.exit(f"no case named {a.case!r}")
    os.makedirs(a.outdir, exist_ok=True)
    for c, layers in todo:
        slug = "".join(ch if ch.isalnum() else "_" for ch in c["name"]).strip("_").lower()
        out = a.out or os.path.join(a.outdir, f"{slug}_L{layers}.json")
        if os.path.exists(out) and "step_us" in json.load(open(out)):
            print(f"  {c['name']} L{layers}: done already")
            continue
        r = run(c, layers, out, a.output_len, a.startup)
        with open(out, "w") as f:
            json.dump(r, f, indent=1)
        print(f"  {c['name']:<34} L{layers:<4} " +
              (f"step {r['step_us']:9.1f} us (median ITL)" if "step_us" in r else r.get("error", "?")), flush=True)


if __name__ == "__main__":
    main()
