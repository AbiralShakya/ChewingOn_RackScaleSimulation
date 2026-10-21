#!/usr/bin/env python3
"""FlashAttention across generations: hardware vs simulator, then simulator-only scaling.

  python research/flash_attention/run_study.py validate   # traces -> .rpath -> rsim, compared with ncu
  python research/flash_attention/run_study.py whatif     # one hardware number at a time, on the recorded kernels
  python research/flash_attention/run_study.py scale      # context-parallel ring attention at 72 GPUs
  python research/flash_attention/run_study.py figures    # results/study.json -> figures/*.png
  python research/flash_attention/run_study.py all

Hardware inputs (collected on the GPUs by remote/*.sh, copied into results/<gpu>/):
  timing*.jsonl                  fa_bench.py timing sweeps
  ncu_<impl>_*.json              Nsight Compute ground truth (cycles, instructions, stall reasons)
NVBit recordings: raw warp_path output is large and not in git (FA_TRACES, default
research/flash_attention/traces/); the converted recordings the study replays are committed as
rpath/<gpu>_<impl>_<shape>.rpath.gz, so validate / whatif / scale run without the raw traces.
Outputs: results/study.json (everything the write-up shows), results/sim/*.json, figures/*.png.
Each simulation result is cached in results/sim/; delete a file to re-run it.
"""
import copy
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(HERE, "results")
SIM = os.path.join(RES, "sim")
FIG = os.path.join(HERE, "figures")
TRACES = os.environ.get("FA_TRACES", os.path.join(HERE, "traces"))
RSIM = os.path.join(ROOT, "target", "release", "rsim")
CALDIR = os.path.join(HERE, "calibration")
JOBS = int(os.environ.get("FA_JOBS", "6"))

SIM_STALLS = ["selected", "not_selected", "wait", "long_scoreboard", "short_scoreboard", "math_pipe_throttle",
              "mio_throttle", "lg_throttle", "barrier", "async_wait", "tma_wait", "tensor_wait", "tensor_throttle",
              "branch_resolving", "membar", "drain", "idle"]

# One entry per validated kernel: the recording, the ncu ground truth, the simulator setup.
GPUS = {
    "a100": {"name": "A100 80GB PCIe (sm_80)", "preset": "a100_pcie", "cal": "a100-80gb-pcie-sm80.json",
             "res": "a100_NVIDIA_A100_80GB_PCIe", "trace": "a100_NVIDIA_A100_80GB_PCIe"},
    "a100sxm": {"name": "A100 SXM4 80GB (sm_80)", "preset": "a100_sxm", "cal": "a100-sxm4-80gb-sm80.json",
                "res": "a100_NVIDIA_A100-SXM4-80GB", "trace": "a100_NVIDIA_A100-SXM4-80GB"},
    "b300": {"name": "B300 SXM6 (sm_103)", "preset": "b300", "cal": "b300-sxm6-ac-sm103.json",
             "res": "b300", "trace": "b300"},
}
VALIDATE = [
    # (gpu, impl, shape tag, ncu kernel substring, regs)
    ("a100", "fa2", "s4096", "flash_fwd_kernel", 255),
    ("a100", "fa4", "s4096", "FlashAttentionForwardSm80", 255),
    ("b300", "fa2", "s4096", "flash_fwd_kernel", 255),
    ("b300", "fa4", "s4096", "FlashAttentionForwardSm100", 128),
]
# Ring-attention step kernels (Llama-3-70B heads, one chunk of queries x one chunk of keys),
# validated the same way before the rack-scale runs use them.
VALIDATE_CP = [(g, impl, f"cp_c{c}_kv{kv}", sub, regs)
               for g, impl, sub, regs in (("b300", "fa4", "FlashAttentionForwardSm100", 128),
                                          ("a100sxm", "fa2", "flash_fwd_kernel", 255))
               for c in (1024, 2048, 4096) for kv in (8, 64)]


def sh(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(" ".join(cmd) + "\n" + p.stderr[-2000:])
    return p.stdout


def trace_dir(gpu, impl, shape):
    return os.path.join(TRACES, f"{GPUS[gpu]['trace']}_{impl}_{shape}")


def rpath_of(gpu, impl, shape, regs):
    d = trace_dir(gpu, impl, shape)
    out = d + ".rpath"
    # Committed recordings: research/flash_attention/rpath/<name>.rpath.gz (rsim reads gzip).
    gz = os.path.join(HERE, "rpath", os.path.basename(d) + ".rpath.gz")
    if not os.path.exists(out) and os.path.exists(gz):
        return gz
    if not os.path.exists(out):
        if not os.path.isdir(d):
            return None
        sh([sys.executable, os.path.join(HERE, "trace_to_rsim.py"), d, out, "--regs", str(regs)])
    return out


def simulate(key, rpath, gpu, regs=None, cal_path=None):
    """Run `rsim path` once and cache the summary under results/sim/<key>.json."""
    os.makedirs(SIM, exist_ok=True)
    cache = os.path.join(SIM, key + ".json")
    if os.path.exists(cache):
        return json.load(open(cache))
    cmd = [RSIM, "path", rpath, "--gpu", GPUS[gpu]["preset"], "--calibration",
           cal_path or os.path.join(CALDIR, GPUS[gpu]["cal"]), "--json"]
    if regs:
        cmd += ["--regs", str(regs)]
    d = json.loads(sh(cmd))
    st = d["stats"]
    issued = max(1, st["insts_issued"])
    out = {
        "cycles": d["end_cycle"] - d["start_cycle"],
        "insts_issued": st["insts_issued"],
        "tensor_busy_cycles": st["tensor_busy_cycles"],
        "tensor_flops": st["tensor_flops"],
        "dram_read_bytes": st["dram_read_bytes"],
        "l2_hit_rate": st["l2_hits"] / max(1, st["l2_hits"] + st["l2_misses"]),
        "warp_stalls_per_issue": {k: v / issued for k, v in zip(SIM_STALLS, st["warp"]) if k != "idle"},
        "sched_fraction": {k: v / max(1, sum(st["sched"])) for k, v in zip(SIM_STALLS, st["sched"])},
        "class_mix": st["class_mix"],
        "unknown_opcodes": st.get("unknown_opcodes", {}),
        # Per warp index within the CTA (= per role for warp-specialised kernels):
        # warp-cycles by state and instructions issued.
        "warp_by_index": [dict(zip(SIM_STALLS, w)) for w in st.get("warp_by_index", [])],
        "issued_by_index": st.get("issued_by_index", []),
    }
    json.dump(out, open(cache, "w"), indent=1)
    return out


def ncu_of(gpu, impl, shape, substr):
    p = os.path.join(RES, GPUS[gpu]["res"], f"ncu_{impl}_{shape}.json")
    if not os.path.exists(p):
        return None
    ks = [k for k in json.load(open(p))["kernels"] if substr in k["kernel"]]
    return ks[0] if ks else None


def shares(d, skip=("selected", "not_selected", "drain", "idle")):
    """Stall reasons as fractions of all stalled warp-cycles (selected/idle excluded)."""
    tot = sum(v for k, v in d.items() if k not in skip and isinstance(v, (int, float)))
    return {k: v / tot for k, v in d.items() if k not in skip and isinstance(v, (int, float)) and tot > 0}


def validate(entries=None):
    rows = []
    jobs = []
    for gpu, impl, shape, sub, regs in (entries or VALIDATE):
        rp = rpath_of(gpu, impl, shape, regs)
        if rp:
            jobs.append((gpu, impl, shape, sub, regs, rp))
    with ThreadPoolExecutor(JOBS) as ex:
        sims = list(ex.map(lambda j: simulate(f"{j[0]}_{j[1]}_{j[2]}", j[5], j[0], j[4]), jobs))
    for (gpu, impl, shape, sub, regs, _rp), s in zip(jobs, sims):
        n = ncu_of(gpu, impl, shape, sub)
        row = {"gpu": gpu, "gpu_name": GPUS[gpu]["name"], "impl": impl, "shape": shape, "sim": s}
        if n:
            nc = n["ncu"]
            row["ncu"] = {
                "kernel": n["kernel"], "cycles": nc["sm__cycles_elapsed.max"], "us": nc["gpu__time_duration.sum"] / 1e3,
                "insts": nc["smsp__inst_executed.sum"], "dram_read_bytes": nc.get("dram__bytes_read.sum"),
                "regs": nc.get("launch__registers_per_thread"), "stalls_per_issue": n["stalls_per_issue"],
            }
            row["cycle_error"] = (s["cycles"] - nc["sm__cycles_elapsed.max"]) / nc["sm__cycles_elapsed.max"]
            row["inst_error"] = (s["insts_issued"] - nc["smsp__inst_executed.sum"]) / nc["smsp__inst_executed.sum"]
        rows.append(row)
        e = row.get("cycle_error")
        print(f"{gpu:5} {impl:4} {shape:6} sim {s['cycles']:>10} ncu "
              f"{row.get('ncu', {}).get('cycles', float('nan')):>10.0f}  err {100 * e if e is not None else float('nan'):+.1f}%")
    return rows


# What-ifs: change one number in the calibration and re-simulate the recorded kernel.
def cal_variant(gpu, name, fn):
    base = json.load(open(os.path.join(CALDIR, GPUS[gpu]["cal"])))
    doc = copy.deepcopy(base)
    fn(doc["model"])
    os.makedirs(os.path.join(SIM, "cal"), exist_ok=True)
    p = os.path.join(SIM, "cal", f"{gpu}_{name}.json")
    json.dump(doc, open(p, "w"))
    return p


def set_sass(m, op, ii=None, lat=None):
    for k in list(m["sass"]):
        if k == op or k.startswith(op + "@"):
            if ii is not None:
                m["sass"][k]["issue_interval_cycles_per_smsp"] = ii
            if lat is not None:
                m["sass"][k]["latency_cycles"] = lat


def scale_tensor(m, f):
    for k, v in m["tensor"].items():
        if k.startswith("tcgen05") and isinstance(v, dict) and "flop_per_clk_per_sm_allsm" in v:
            v["flop_per_clk_per_sm_allsm"] *= f


def whatif():
    out = []
    variants = {
        "b300": [
            ("baseline", lambda m: None),
            # B200-like exp unit: MUFU.EX2 at the same 16/clk/SM as the other MUFU functions.
            ("ex2_half_rate", lambda m: set_sass(m, "MUFU.EX2", ii=8.0, lat=18.0)),
            ("ex2_double_rate", lambda m: set_sass(m, "MUFU.EX2", ii=2.0)),
            ("tensor_1.5x", lambda m: scale_tensor(m, 1.5)),
            ("tensor_2x", lambda m: scale_tensor(m, 2.0)),
            # With twice the tensor rate, which unit binds next?
            ("tensor_2x+ex2_2x", lambda m: (scale_tensor(m, 2.0), set_sass(m, "MUFU.EX2", ii=2.0))),
            ("tensor_2x+fma2_2x", lambda m: (scale_tensor(m, 2.0), set_sass(m, "FFMA2", ii=1.1),
                                             set_sass(m, "FADD2", ii=1.0), set_sass(m, "FMUL2", ii=1.0))),
            ("tensor_2x+ex2_2x+fma2_2x", lambda m: (scale_tensor(m, 2.0), set_sass(m, "MUFU.EX2", ii=2.0),
                                                    set_sass(m, "FFMA2", ii=1.1), set_sass(m, "FADD2", ii=1.0),
                                                    set_sass(m, "FMUL2", ii=1.0))),
            ("fma2_half_rate", lambda m: (set_sass(m, "FFMA2", ii=4.4), set_sass(m, "FADD2", ii=4.0),
                                          set_sass(m, "FMUL2", ii=4.0))),
            ("hbm_half", lambda m: m["memory"]["bandwidth"].update(
                hbm_read_GBps=m["memory"]["bandwidth"]["hbm_read_GBps"] / 2,
                hbm_peak_GBps_from_attrs=m["memory"]["bandwidth"]["hbm_peak_GBps_from_attrs"] / 2)),
        ],
        "a100": [
            ("baseline", lambda m: None),
            ("no_hmma_fp32_coupling", lambda m: (m.pop("hmma_coissue", None),
                                                 m["tensor"].pop("mma_sync_fp32_pipe_cycles", None))),
            ("ex2_double_rate", lambda m: set_sass(m, "MUFU.EX2", ii=4.0)),
        ],
    }
    jobs = []
    for gpu, impl, shape, _sub, regs in VALIDATE:
        if gpu not in variants or (gpu, impl) == ("a100", "fa4"):
            continue
        rp = rpath_of(gpu, impl, shape, regs)
        if not rp:
            continue
        for name, fn in variants[gpu]:
            # FA2 on B300 runs on mma.sync, so only the exp/FMA/HBM knobs apply to it.
            if (gpu, impl) == ("b300", "fa2") and name.startswith("tensor"):
                continue
            cp = cal_variant(gpu, name, fn)
            jobs.append((gpu, impl, shape, regs, rp, name, cp))
    with ThreadPoolExecutor(JOBS) as ex:
        sims = list(ex.map(lambda j: simulate(f"whatif_{j[0]}_{j[1]}_{j[2]}_{j[5]}", j[4], j[0], j[3], j[6]), jobs))
    base = {}
    for j, s in zip(jobs, sims):
        if j[5] == "baseline":
            base[(j[0], j[1])] = s["cycles"]
    for j, s in zip(jobs, sims):
        b = base.get((j[0], j[1]))
        out.append({"gpu": j[0], "impl": j[1], "variant": j[5], "cycles": s["cycles"],
                    "speedup_vs_baseline": b / s["cycles"] if b else None,
                    "sched_fraction": s["sched_fraction"]})
        print(f"{j[0]:5} {j[1]:4} {j[5]:24} {s['cycles']:>10}  x{(b / s['cycles']) if b else 0:.3f}")
    return out


# Context-parallel (ring) attention, simulator only. Each rank owns S/N tokens; each of the N
# ring steps runs one attention kernel on (its queries x one KV chunk) while the next KV chunk
# moves to the neighbour over the fabric (double buffered).
LLAMA70 = {"hq": 64, "hkv": 8, "d": 128}


def ring_workload(n, chunk, hkv, rpath, regs):
    kv_bytes = 2 * chunk * hkv * LLAMA70["d"] * 2  # K and V, bf16
    kern = {"op": "kernel", "name": f"attn_c{chunk}", "kernel": {"type": "path", "path": rpath, "regs": regs}}
    compute = [kern, {"op": "record", "event": "kd"},
               {"op": "loop", "times": n - 1, "body": [{"op": "wait", "event": "rx"}, kern,
                                                       {"op": "record", "event": "kd"}]}]
    # Send and receive on their own streams (a send and a recv on one stream would deadlock
    # the ring). A rank forwards chunk i only after it has received it.
    send = {"op": "send", "bytes": kv_bytes, "peer": 1, "relative": True, "sms": 2}
    recv = {"op": "recv", "bytes": kv_bytes, "peer": -1, "relative": True, "sms": 2}
    sends = [send, {"op": "loop", "times": n - 2, "body": [{"op": "wait", "event": "fwd"}, send]}]
    recvs = [{"op": "loop", "times": n - 1, "body": [recv, {"op": "record", "event": "rx"},
                                                     {"op": "record", "event": "fwd"}]}]
    flops = 4.0 * chunk * chunk * LLAMA70["hq"] * LLAMA70["d"] * n * n
    return {"name": f"ring attention N={n} chunk={chunk} hkv={hkv}", "model_flops": flops,
            "spmd": {"world": n, "program": {"streams": [compute, sends, recvs]}}}


SYSTEMS = {
    "gb300_nvl72": ("b300", os.path.join(HERE, "configs", "gb300_nvl72.yaml"), "fa4", 128),
    "b300_9x8_ib": ("b300", os.path.join(HERE, "configs", "b300_9x8_ib.yaml"), "fa4", 128),
    "a100_9x8_ib": ("a100sxm", os.path.join(HERE, "configs", "a100_9x8_ib.yaml"), "fa2", 255),
}


def scale():
    rows = []
    os.makedirs(os.path.join(SIM, "ring"), exist_ok=True)
    jobs = []
    for sysname, (gpu, cfg, impl, regs) in SYSTEMS.items():
        for chunk in (1024, 2048):
            for hkv in (8, 64):
                rp = rpath_of(gpu, impl, f"cp_c{chunk}_kv{hkv}", regs)
                if not rp:
                    continue
                for n in (8, 72):
                    key = f"ring_{sysname}_{impl}_c{chunk}_kv{hkv}_n{n}"
                    # 72-way rings for every system; 8-way ones only where already simulated.
                    if n == 72 or os.path.exists(os.path.join(SIM, "ring", key + ".json")):
                        jobs.append((sysname, cfg, gpu, impl, chunk, hkv, n, rp, regs))

    def run(j):
        sysname, cfg, gpu, impl, chunk, hkv, n, rp, regs = j
        key = f"ring_{sysname}_{impl}_c{chunk}_kv{hkv}_n{n}"
        cache = os.path.join(SIM, "ring", key + ".json")
        if not os.path.exists(cache):
            wl = os.path.join(SIM, "ring", key + ".workload.json")
            json.dump(ring_workload(n, chunk, hkv, rp, regs), open(wl, "w"))
            sh([RSIM, "run", "--system", cfg, "--workload", wl, "--fast", "--report", cache])
        return j, json.load(open(cache))

    with ThreadPoolExecutor(max(1, JOBS // 2)) as ex:
        for j, rep in ex.map(run, jobs):
            sysname, _cfg, gpu, impl, chunk, hkv, n, _rp, _regs = j
            mk = rep["makespan_us"]
            kern = [k for k in rep["kernels"] if "flash" in k["name"].lower() or "attn" in k["name"].lower()]
            k_us = kern[0]["mean_us"] if kern else None
            colls = rep.get("collectives", [])
            # Slowest KV-chunk transfer to a ring neighbour: the ring moves at its slowest hop
            # (an InfiniBand hop between nodes; NVLink everywhere on NVL72).
            durs = sorted((c["done_ps"] - c["last_post_ps"]) / 1e6 for c in colls if c.get("kind") == "SendRecv")
            send_us = durs[-1] if durs else None
            # The same kernel alone on an idle GPU (the validated replay), for comparison: during
            # the ring it shares SMs, L2 and HBM with the transfers.
            alone = os.path.join(SIM, f"{gpu}_{impl}_cp_c{chunk}_kv{hkv}.json")
            mhz = json.load(open(os.path.join(CALDIR, GPUS[gpu]["cal"])))["model"]["sm_clock_mhz_measured"]
            k_alone = json.load(open(alone))["cycles"] / mhz if os.path.exists(alone) else None
            rows.append({"system": sysname, "gpu": gpu, "impl": impl, "chunk": chunk, "hkv": hkv, "n": n,
                         "kernel_alone_us": k_alone,
                         "eff_alone": (n * k_alone / mk) if k_alone else None,
                         "context": chunk * n, "makespan_us": mk, "kernel_us": k_us,
                         "kv_bytes": 2 * chunk * hkv * LLAMA70["d"] * 2, "sendrecv_us": send_us,
                         # Ring efficiency: time the GPU spends in attention kernels / step time.
                         "eff": (n * k_us / mk) if k_us else None,
                         "tflops_per_gpu": 4.0 * chunk * chunk * LLAMA70["hq"] * LLAMA70["d"] * n / mk / 1e6,
                         "bubbles": rep.get("bubbles")})
            print(f"{sysname:12} {impl} N={n:2} chunk={chunk:5} hkv={hkv:2} -> {mk:9.1f} us")
    return rows


def load_timing():
    out = {}
    for d in glob.glob(os.path.join(RES, "*")):
        for f in glob.glob(os.path.join(d, "timing*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.setdefault(os.path.basename(d), []).append(dict(r, file=os.path.basename(f)))
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    study_p = os.path.join(RES, "study.json")
    study = json.load(open(study_p)) if os.path.exists(study_p) else {}
    if what in ("validate", "all"):
        study["validation"] = validate()
    if what in ("validate_cp", "all"):
        study["validation_cp"] = validate(VALIDATE_CP)
    if what in ("whatif", "all"):
        study["whatif"] = whatif()
    if what in ("scale", "all"):
        study["scale"] = scale()
    study["timing"] = load_timing()
    json.dump(study, open(study_p, "w"), indent=1)
    if what in ("figures", "all"):
        from figures import make_figures
        make_figures(study, FIG)


if __name__ == "__main__":
    main()
