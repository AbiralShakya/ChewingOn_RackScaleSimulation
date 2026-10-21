#!/usr/bin/env python3
"""Fold the per-bench raw JSON in <outdir>/raw/ into <outdir>/calibration.json.

The output has two parts:
  "model": the numbers the simulator loads, keyed the way the SM model looks
           them up (SASS opcode -> latency/throughput, memory levels, tensor
           core instruction shapes, TMA, launch).
  "raw":   every bench's full output, for provenance.

Usage: merge.py results/<gpu>-sm<cc>
"""
import datetime
import glob
import json
import os
import subprocess
import sys

SCHEMA = "rackscale-calibration/v1"


def load_raw(outdir):
    raw = {}
    for p in sorted(glob.glob(os.path.join(outdir, "raw", "*.json"))):
        try:
            raw[os.path.basename(p)[:-5]] = json.load(open(p))
        except json.JSONDecodeError as e:
            print(f"skipping {p}: {e}", file=sys.stderr)
    return raw


def sass_opcode(op):
    """The SASS opcode that implements the op: the most frequent matching opcode in the latency kernel."""
    s = op.get("sass", {})
    prefixes = tuple(op["sass_expected"].split(","))
    top = s.get("lat_opcodes_top", {})
    best = [o for o in sorted(top, key=top.get, reverse=True) if o.startswith(prefixes) and ".MOV" not in o]
    return best[0] if best else op["sass_expected"].split(",")[0]


def model_alu(alu):
    ops = alu["ops"]
    out = {}
    for name, op in ops.items():
        s = op.get("sass", {})
        lat = s.get("latency_cycles", op["latency_cycles_per_ptx_op"])
        thr = s.get("throughput_sass_inst_per_clk_per_sm", op["throughput_ptx_ops_per_clk_per_sm"])
        partner = s.get("composite_with")
        note = None
        if partner and partner in ops:
            p = ops[partner]
            plat = p.get("sass", {}).get("latency_cycles", p["latency_cycles_per_ptx_op"])
            lat = lat - plat
            note = f"timed as {name}+{partner}; {partner} latency {plat:.2f} subtracted"
        entry = {
            "bench_op": name,
            "ptx": op["ptx"],
            "latency_cycles": round(lat, 2),
            "issue_per_clk_per_sm": round(thr / 32.0, 4),       # warp instructions
            "thread_ops_per_clk_per_sm": round(thr, 2),
            "issue_interval_cycles_per_smsp": round(4 * 32.0 / thr, 3) if thr > 0 else None,
        }
        if note:
            entry["note"] = note
        out.setdefault(sass_opcode(op), entry)
        out[sass_opcode(op) + "@" + name] = entry
    return out


def model_mem(mem):
    m = {"l2_bytes": mem["l2_bytes"], "smem_per_sm_bytes": mem["smem_per_sm_bytes"],
         "smem_per_block_optin_bytes": mem["smem_per_block_optin_bytes"]}
    sweep = mem["latency_sweep"]
    l1 = [r["ld_ca_cycles"] for r in sweep if r["footprint_bytes"] <= 64 << 10]
    l2 = [r["ld_cg_cycles"] for r in sweep if (1 << 20) <= r["footprint_bytes"] <= mem["l2_bytes"] // 4]
    l2f = [r["ld_cg_first_pass_cycles"] for r in sweep if (1 << 20) <= r["footprint_bytes"] <= mem["l2_bytes"] // 4]
    dram = [r["ld_cg_cycles"] for r in sweep if not r["warmed"]]
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    m["latency_cycles"] = {
        "l1_hit": med(l1),
        "l2_hit_repeat": med(l2),
        "l2_hit_first_touch": med(l2f),
        "dram": med(dram),
        "smem": mem["smem"]["ld_shared_latency_cycles"],
        "dsmem_remote": mem["smem"].get("dsmem_remote_latency_cycles"),
    }
    d = mem.get("l2_hit_latency_distribution", {})
    m["l2_partitions"] = {k: d.get(k) for k in ("near_mean_cycles", "far_mean_cycles", "far_fraction", "split_threshold_cycles")}
    bw = mem["bandwidth"]
    m["bandwidth"] = {
        "hbm_read_GBps": bw["hbm_read_GBps"], "hbm_write_GBps": bw["hbm_write_GBps"],
        "hbm_copy_GBps": bw["hbm_copy_GBps_rd_plus_wr"], "hbm_peak_GBps_from_attrs": bw["hbm_peak_GBps_from_attrs"],
        "l2_read_bytes_per_clk_per_sm": max(v for k, v in bw.items() if k.startswith("l2_read_bytes_per_clk_per_sm")),
        "l1_read_bytes_per_clk_per_sm": bw["l1_read_bytes_per_clk_per_sm"],
        "smem_bytes_per_clk_per_sm": mem["smem"]["ld_shared_v4_bytes_per_clk_per_sm"],
        "hbm_read_vs_sms": bw["hbm_read_vs_sms"],
    }
    m["smem_bank_conflicts"] = {str(r["stride_words"]): r["cycles_per_warp_load_per_sm"] for r in mem["smem"]["bank_conflict_sweep"]}
    return m


def model_tc(raw):
    out = {}
    for k, v in raw.items():
        if k.startswith("tc_"):
            out.update(v.get("tc", {}))
    return out


def model_ampere(raw):
    """sm_80+ async copy, warp MMA sweep and L2 partition map (bench_cpasync,
    bench_mma_sync, bench_l2part, bench_overlap)."""
    m = {}
    if "cpasync" in raw:
        c = raw["cpasync"]
        lat = c["latency"]
        m["async_copy"] = {
            "latency_cycles": {lvl: {k: v["hop_cycles"] for k, v in lat[lvl].items() if isinstance(v, dict)}
                               for lvl in ("l2", "dram")},
            "lds_hop_cycles": lat["lds_hop_cycles"],
            "bandwidth_GBps": {lvl: {k: v["best_gbps"] for k, v in c["bandwidth"][lvl].items() if isinstance(v, dict)}
                               for lvl in ("l2", "dram")},
        }
    if "mma_sync" in raw:
        ms = raw["mma_sync"]
        m["warp_mma"] = {
            k: {"shape": v["shape"], "latency_cycles": v["latency_cycles"],
                "flop_per_clk_per_sm": v["throughput_flops_per_clk_per_sm"],
                "issue_interval_cycles_per_smsp": v["issue_interval_cycles_per_smsp"],
                "saturating_warps": v["best_warps"], "saturating_ilp": v["best_ilp"]}
            for k, v in ms["mma"].items()}
        ld = ms["ldmatrix"]
        m["ldmatrix"] = {"latency_cycles_x1": ld["latency_cycles_x1"],
                         **{k: v["throughput_bytes_per_clk_per_sm"] for k, v in ld.items() if k.startswith("ldmatrix_x")}}
    if "l2part" in raw:
        l2 = raw["l2part"]
        m["l2_partition_map"] = {
            "near_cycles": l2["cluster_near_cycles"], "far_cycles": l2["cluster_far_cycles"],
            "sm_side": {str(r["smid"]): r["side"] for r in l2["sm"]},
            "best_single_address_bit": l2["best_single_address_bit"],
            "best_single_bit_accuracy": l2["best_single_bit_accuracy"],
        }
    if "overlap" in raw:
        o = raw["overlap"]
        # Per other-kind: cycles per warp-iteration per SMSP for (hmma, other) mixes,
        # plus the overlap factor (1 = hidden behind HMMA, 0 = serialized).
        m["hmma_coissue"] = {
            "hmma4_alone_cycles": o["hmma4_alone_cycles"], "hmma8_alone_cycles": o["hmma8_alone_cycles"],
            **{k: {f"h{p['hmma']}_o{p['other']}": {"cycles": p["cycles"], "overlap": p["overlap"]} for p in v["points"]}
               for k, v in o["mix"].items()},
        }
        m["warp_collectives_cycles"] = o["collectives"]
        # FP32-pipe cycles each HMMA takes from FFMA/FMUL issued in the same
        # warps: (t_mix - t_fp32_alone) / n_hmma on points where the FP32 stream
        # is the longer one, median across FFMA and FMUL.
        k = sorted((p["cycles"] - p["cycles_other_alone"]) / p["hmma"]
                   for kind in ("ffma", "fmul") for p in o["mix"][kind]["points"]
                   if p["cycles_other_alone"] >= p["cycles_hmma_alone"])
        if k:
            m["mma_sync_fp32_pipe_cycles"] = k[len(k) // 2]
    return m


def _power_runs(log, trace, legacy_label=None):
    """Runs of one power sweep: BEGIN/END windows in `log` over the nvidia-smi
    `trace`. Old FP4-only logs (cutlass_fp4.log) have no label on BEGIN/END;
    `legacy_label` names them."""
    ts = lambda t: datetime.datetime.strptime(t.strip(), "%Y/%m/%d %H:%M:%S.%f")
    samples = []
    for row in open(trace):
        f = [x.strip() for x in row.split(",")]
        if len(f) < 8:
            continue
        num = lambda x: float(x.split()[0])
        samples.append({"t": ts(f[0]), "mhz": num(f[1]), "w": num(f[2]), "cap_w": num(f[3]), "temp_c": num(f[4]),
                        "reasons": f[5], "util": num(f[6]), "max_mhz": num(f[7])})
    runs, cur, verified = [], None, {}
    for line in open(log):
        p = line.split()
        if legacy_label and p[:1] in (["BEGIN"], ["END"]):
            p = [p[0], legacy_label] + p[1:]
        if p[:1] == ["VERIFY"]:
            verified[p[1]] = "Passed" in line
        elif legacy_label and "Disposition: Passed" in line:
            verified[legacy_label] = True
        elif p[:1] == ["BEGIN"]:
            cur = {"label": p[1], "size": int(p[2]), "t0": ts(" ".join(p[3:5])), "gflops": []}
        elif p[:1] == ["END"] and cur:
            cur["t1"] = ts(" ".join(p[3:5]))
            runs.append(cur)
            cur = None
        elif cur and "GFLOPS" in line:
            cur["gflops"].append(float(line.split(":")[1]))
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    cap = samples[0]["cap_w"] if samples else 0
    out = []
    for r in runs:
        # GEMM steady state inside the window: host setup and on-GPU data init run
        # far below the cap, so keep samples above half the cap.
        win = [s for s in samples if r["t0"] <= s["t"] <= r["t1"] and s["util"] >= 50 and s["w"] >= 0.5 * cap]
        out.append({
            "gemm": r["label"], "verified": verified.get(r["label"]),
            "m": r["size"], "n": r["size"], "k": r["size"],
            "tflops": max(r["gflops"]) / 1e3 if r["gflops"] else None,
            "busy_samples": len(win),
            "sm_mhz_median": med([s["mhz"] for s in win]),
            "sm_mhz_min": min((s["mhz"] for s in win), default=None),
            "board_w_median": med([s["w"] for s in win]),
            "board_w_max": max((s["w"] for s in win), default=None),
            "utilization_pct_median": med([s["util"] for s in win]),
            "power_capped_fraction": (sum("0x0000000000000004" in s["reasons"] for s in win) / len(win)) if win else None,
        })
    meta = {"power_cap_w": cap or None, "max_sm_mhz": samples[0]["max_mhz"] if samples else None,
            "idle_w": med([s["w"] for s in samples if s["util"] == 0])}
    return out, meta


def model_power(outdir):
    """Clock and power while CUTLASS GEMMs of each dtype run (scripts/cutlass_power.sh).
    Runs from the older FP4-only sweep (cutlass_fp4.*) fill in any GEMM the
    current sweep has not recorded."""
    raw = os.path.join(outdir, "raw")
    sweeps = [("cutlass_power", None), ("cutlass_fp4", "nvfp4_ultra_k96")]
    runs, meta = [], {}
    for stem, legacy in sweeps:
        log, trace = os.path.join(raw, stem + ".log"), os.path.join(raw, stem + ".clk.csv")
        if not (os.path.exists(log) and os.path.exists(trace)):
            continue
        r, m = _power_runs(log, trace, legacy)
        have = {(x["gemm"], x["m"]) for x in runs}
        runs += [x for x in r if (x["gemm"], x["m"]) not in have and x["busy_samples"]]
        for k, v in m.items():
            meta.setdefault(k, v)
    if not runs:
        return None
    return {
        "source": "CUTLASS examples 70 (fp16, fp8), 72b (nvfp4 K=64), 89 (sm103 ultra nvfp4 K=96)",
        **meta,
        "sample_interval_ms": 100,
        "runs": runs,
    }


def model_window(raw, outdir):
    """Fixed kernel window per launch shape: bench_window's configs with the
    `sm__cycles_elapsed.avg` ncu measured for them (ncu_reference.py; run.sh
    merges again after it)."""
    ref = os.path.join(outdir, "ncu_reference.json")
    if "window" not in raw or not os.path.exists(ref):
        return []
    ncu = {k["label"]: k.get("ncu", {}) for k in json.load(open(ref))["kernels"]}
    pts = []
    for name, c in raw["window"]["window"].items():
        cyc = ncu.get(f"window_{name}", {}).get("sm__cycles_elapsed.avg")
        if isinstance(cyc, (int, float)):
            pts.append({"config": name, "threads": c["threads"], "smem_bytes": c["smem_dynamic"],
                        "cluster": c["cluster"], "tmem_prologue": c["fa4_prologue"], "cycles": cyc})
    return pts


def main(outdir):
    raw = load_raw(outdir)
    if "alu" not in raw:
        sys.exit("no alu.json: run the benches first")
    dev = raw["alu"]["device"]
    model = {"device": dev, "sm_clock_mhz_measured": raw["alu"]["sm_clock_mhz_measured"]}
    model["sass"] = model_alu(raw["alu"])
    if "mem" in raw:
        model["memory"] = model_mem(raw["mem"])
    model["tensor"] = model_tc(raw)
    if "tma" in raw and "bulk_g2s_latency" in raw["tma"]:
        model["tma"] = {"bulk_g2s_latency": raw["tma"]["bulk_g2s_latency"], "bulk_g2s_bandwidth": raw["tma"]["bulk_g2s_bandwidth"]}
    amp = model_ampere(raw)
    if "mma_sync_fp32_pipe_cycles" in amp:
        model["tensor"]["mma_sync_fp32_pipe_cycles"] = amp.pop("mma_sync_fp32_pipe_cycles")
    model.update(amp)
    if "misc" in raw:
        model["launch"] = raw["misc"]["launch"]
        model["syncthreads_cycles"] = raw["misc"]["syncthreads_cycles"]
        model["atomics"] = raw["misc"]["atomics"]
    window = model_window(raw, outdir)
    if window:
        model.setdefault("launch", {})["kernel_window"] = window
    power = model_power(outdir)
    if power:
        model["power"] = power
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0]
    except Exception:
        smi = None
    doc = {
        "schema": SCHEMA,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "driver": smi,
        "missing_benches": sorted({"alu", "mem", "misc", "tma", "window"} - set(raw)),
        "model": model,
        "raw": raw,
    }
    path = os.path.join(outdir, "calibration.json")
    json.dump(doc, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])
