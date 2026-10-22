"""FA4 per-warp / per-role timelines: hardware clock64 probes on every
pipeline event (run_fa4_probe.py) vs the simulator replaying the NVBit
recording of the same kernel (`rsim timeline <rpath>`).

Events are matched per warp by progress, not one by one (the probes add a
few instructions and the recording drops failed try_waits): for each warp,
t(q) is the time at which it reached fraction q of its synchronisation
events (hardware: probe stamps; simulator: issue of SYNCS*/UTCBAR
instructions), relative to its CTA's first event. Reported per warp and per
role: span (first to last event) and the mean |t_sim(q) - t_hw(q)| over
q = 0.05 .. 1, as a fraction of the hardware span.

    python3 fa4_compare.py fa4_probe.json fa4.rpath.gz out.json --calibration c.json [--extra ...]
"""
import argparse
import gzip
import json
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sasslib import parse  # noqa: E402

ROLES = {**{w: "softmax0" for w in range(0, 4)}, **{w: "softmax1" for w in range(4, 8)},
         **{w: "correction" for w in range(8, 12)}, 12: "mma", 13: "epilogue", 14: "load", 15: "empty"}
QS = [i / 20 for i in range(1, 21)]


def rpath_sass(path):
    d = gzip.open(path).read() if path.endswith(".gz") else open(path, "rb").read()
    n = struct.unpack_from("<I", d, 8)[0]
    h = json.loads(d[12:12 + n])
    return h["sass"], h


def progress(ts):
    if len(ts) < 2:
        return None
    out = []
    for q in QS:
        out.append(ts[min(len(ts) - 1, max(0, round(q * (len(ts) - 1))))])
    return out


def sim_events(rpath, calib, extra):
    sass, hdr = rpath_sass(rpath)
    funcs = parse(sass)
    insts = next(iter(funcs.values()))
    sync_pcs = {i.pc for i in insts if i.base in ("SYNCS", "UTCBAR")}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    cmd = ["target/release/rsim", "timeline", rpath, "--gpu", "b300", "--regs", "128", "--sms", "0,1,2,3,4,5,6,7", "-o", out]
    cmd += (["--calibration", calib] if calib else []) + extra
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(p.stderr[-800:])
    d = json.load(open(out))
    os.unlink(out)
    ev = {}
    for tl in d["warps"]:
        for w in tl["warps"]:
            iss = w["iss"]
            ev[(w["cta"], w["widx"])] = [c for c, pc in zip(iss[0::2], iss[1::2]) if pc in sync_pcs]
    return ev, p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hw")
    ap.add_argument("rpath")
    ap.add_argument("out")
    ap.add_argument("--calibration")
    ap.add_argument("--extra", default="")
    ap.add_argument("--ctas", default="0,1,2,3")
    ap.add_argument("--hw-total", type=float,
                    help="ncu cycles of the unprobed kernel: hardware times are scaled by "
                         "hw_total / (probed CTA span) to take the probes' own cost out")
    a = ap.parse_args()
    hw = json.load(open(a.hw))
    sim, msg = sim_events(a.rpath, a.calibration, a.extra.split())
    ctas = [int(x) for x in a.ctas.split(",")]
    rows = []
    for cta in ctas:
        H = {int(k.split(":")[1]): [c for c, t in v] for k, v in hw["warps"].items() if int(k.split(":")[0]) == cta}
        S = {w: v for (c, w), v in sim.items() if c == cta and v}
        if not H or not S:
            continue
        h0 = min(v[0] for v in H.values())
        if a.hw_total:
            span = max(v[-1] for v in H.values()) - h0
            f = a.hw_total / span
            H = {w: [h0 + (x - h0) * f for x in v] for w, v in H.items()}
        s0 = min(v[0] for v in S.values())
        for w in sorted(set(H) & set(S)):
            ph = progress([x - h0 for x in H[w]])
            ps = progress([x - s0 for x in S[w]])
            if not ph or not ps:
                continue
            span_h, span_s = ph[-1] - ph[0], ps[-1] - ps[0]
            err = sum(abs(x - y) for x, y in zip(ps, ph)) / len(QS) / max(1, ph[-1])
            # Shape only: both curves normalised to their own end time, so a
            # uniform rate difference (e.g. the probes' own cost) drops out.
            shape = sum(abs(x / ps[-1] - y / ph[-1]) for x, y in zip(ps, ph)) / len(QS)
            rows.append({"cta": cta, "warp": w, "role": ROLES.get(w, "?"), "events_hw": len(H[w]),
                         "events_sim": len(S[w]), "end_hw": ph[-1], "end_sim": ps[-1],
                         "span_err": (ps[-1] - ph[-1]) / max(1, ph[-1]), "progress_err": err, "shape_err": shape,
                         "hw": ph, "sim": ps})
    by = {}
    for r in rows:
        by.setdefault(r["role"], []).append(r)
    print(msg)
    print(f"{'role':12} {'warps':>5} {'events hw/sim':>15} {'end hw':>9} {'end sim':>9} {'end err':>8} {'progress err':>13} {'shape err':>10}")
    summ = {}
    for role, rs in sorted(by.items()):
        n = len(rs)
        eh = sum(r["end_hw"] for r in rs) / n
        es = sum(r["end_sim"] for r in rs) / n
        pe = sum(r["progress_err"] for r in rs) / n
        se = sum(r["shape_err"] for r in rs) / n
        summ[role] = {"end_hw": eh, "end_sim": es, "end_err": (es - eh) / eh, "progress_err": pe, "shape_err": se}
        print(f"{role:12} {n:5} {sum(r['events_hw'] for r in rs) // n:>7}/{sum(r['events_sim'] for r in rs) // n:<7} "
              f"{eh:9.0f} {es:9.0f} {100 * (es - eh) / eh:+7.1f}% {100 * pe:12.1f}% {100 * se:9.1f}%")
    allp = sum(r["progress_err"] for r in rows) / max(1, len(rows))
    alls = sum(r["shape_err"] for r in rows) / max(1, len(rows))
    print(f"all warps: mean progress error {100 * allp:.1f}% of the warp's span, shape error {100 * alls:.1f}%")
    json.dump({"rows": rows, "roles": summ, "mean_progress_err": allp, "mean_shape_err": alls}, open(a.out, "w"))


if __name__ == "__main__":
    main()
