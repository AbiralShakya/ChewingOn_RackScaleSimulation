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
    bench_mma_sync, bench_l2part)."""
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
    return m


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
    model.update(model_ampere(raw))
    if "misc" in raw:
        model["launch"] = raw["misc"]["launch"]
        model["syncthreads_cycles"] = raw["misc"]["syncthreads_cycles"]
        model["atomics"] = raw["misc"]["atomics"]
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0]
    except Exception:
        smi = None
    doc = {
        "schema": SCHEMA,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "driver": smi,
        "missing_benches": sorted({"alu", "mem", "misc", "tma"} - set(raw)),
        "model": model,
        "raw": raw,
    }
    path = os.path.join(outdir, "calibration.json")
    json.dump(doc, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])
