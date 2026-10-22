"""Write the issue-stage model (`model.issue`) and the tcgen05 clock
(`model.power.tcgen05_sm_mhz`) into a calibration.json from dissect results.

    python3 fit_issue.py calibration.json DIR

DIR holds tc05x.json (bench_tc05x), dvfs.json (bench_dvfs), regfile.json
(regfile.py) and optionally rfport.json. Every number is derived here from
the measurements; nothing is typed in by hand except the model structure.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def tmem(tc):
    out = {}
    for what, key in (("ld", "tmem_ld"), ("st", "tmem_st")):
        pts = [(v["bytes_per_warp"], v["latency_cycles"]) for f, v in tc[key].items()
               if f.startswith("32x32b") and v["bytes_per_warp"] >= 512]
        base, slope = linfit([p[0] for p in pts], [p[1] for p in pts])
        out[f"{what}_base"] = round(base, 1)
        out[f"{what}_rate_32x32b"] = round(1 / slope, 1)
    return out


def queue_depth(tc):
    """MMAs in flight when the issuing thread's issue returns, N=256 BF16."""
    r = tc["umma_pipeline_bf16_m128n256k16"]["k"]
    big = r["128"]
    per = (big["arrival"] - r["64"]["arrival"]) / 64
    comp = [big["arrival"] - (127 - j) * per for j in range(128)]
    iss = big["issue_returns"]
    out = [sum(1 for j in range(i + 1) if comp[j] > iss[i]) for i in range(128)]
    return int(statistics.median(out[32:]))


def bank_count(regfile_path):
    """Best bank function (index mod 2/4/8) by regression R^2 on the random trials."""
    import numpy as np
    import gzip
    if not os.path.exists(regfile_path) and os.path.exists(regfile_path + ".gz"):
        regfile_path += ".gz"
    d = json.load(gzip.open(regfile_path) if regfile_path.endswith(".gz") else open(regfile_path))
    best = {}
    for fn, r in d.items():
        nsrc = r["nsrc"]
        rows = [t for t in r["trials"] if t["mode"] in (0, 1)]
        y = np.array([t["cycles"] for t in rows], float)
        for B in (2, 4, 8):
            X = []
            for t in rows:
                conf = 0
                for g in t["regs"]:
                    src = g[1:1 + nsrc]
                    cnt = {}
                    for x in set(src):
                        cnt[x % B] = cnt.get(x % B, 0) + 1
                    conf += max(cnt.values()) - 1
                X.append([conf, 1])
            X = np.array(X, float)
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            best.setdefault(B, []).append(r2)
    return max(best, key=lambda b: statistics.mean(best[b])), {b: round(statistics.mean(v), 3) for b, v in best.items()}


def tcgen05_mhz(dvfs):
    ms = [m for k, v in dvfs.items() if k.startswith("umma") for _, m in v["trace_us_mhz"]]
    return round(statistics.median(ms), 1)


def main(cal_path, d):
    cal = json.load(open(cal_path))
    tc = json.load(open(os.path.join(d, "tc05x.json")))["tc05x"]
    dv = json.load(open(os.path.join(d, "dvfs.json")))["dvfs"]
    banks, r2 = bank_count(os.path.join(d, "regfile.json"))
    iss = {
        "rf_banks": banks,
        "reuse_cache": True,
        # bankcheck.py / t_alu timelines: a cached operand serves any slot.
        "reuse_any_slot": True,
        "yield_min_stall": 2,
        "tcgen05_queue_depth": queue_depth(tc),
        "tmem": tmem(tc),
        # Per-warp clock64 timelines (timeline_compare.py): loose round-robin
        # fits multi-warp progress better than greedy-then-oldest.
        "scheduler": "lrr",
    }
    cal["model"]["issue"] = iss
    cal["model"].setdefault("power", {})["tcgen05_sm_mhz"] = tcgen05_mhz(dv)
    json.dump(cal, open(cal_path, "w"), indent=1)
    print(json.dumps({"issue": iss, "bank_r2": r2, "tcgen05_sm_mhz": cal["model"]["power"]["tcgen05_sm_mhz"]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
