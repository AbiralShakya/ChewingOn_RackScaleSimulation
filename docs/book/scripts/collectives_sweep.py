"""Chapter 4: collective time vs message size on the fabric presets.

Each run is one collective on every rank of a group, alone on an idle system,
so the time is pure communication: no compute to overlap. The fabrics are
presets (NVLink5 18 x 50 GB/s per direction per GPU, NVSwitch, 400G IB at
50 GB/s per NIC), not calibrated against a real multi-GPU system.

    maturin develop --release
    python docs/book/scripts/collectives_sweep.py
    python docs/book/scripts/collectives_sweep.py --kinds all_to_all --systems gb200_nvl72:72
"""

import argparse

import rackscale as rs

SYSTEMS = ["hgx_h100_8:8", "hgx_b200_8:8", "gb200_nvl72:8", "gb200_nvl72:72", "h100_2node_ib:16"]


def one(system, world, kind, nbytes, algo):
    op = {"op": "collective", "kind": kind, "bytes": int(nbytes), "group": "world", "sms": 32}
    if algo:
        op["algo"] = algo
    wl = {"name": f"{kind} {nbytes}", "spmd": {"world": world, "program": {"streams": [[op]]}}}
    r = rs.simulate(system, wl)
    c = r["collectives"][0]
    return r.makespan_us, c["busbw_gbps"], c["algo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--kinds", default="all_reduce,all_gather,all_to_all")
    ap.add_argument("--sizes", default="4096,65536,1048576,16777216,268435456")
    ap.add_argument("--algo", default="", help="force ring / nvls / direct")
    a = ap.parse_args()
    print(f"{'system':<18} {'ranks':>5} {'collective':<11} {'bytes/rank':>11} {'algo':<6} {'time_us':>9} {'busbw GB/s':>10}")
    for s in a.systems.split(","):
        name, world = s.split(":")
        for kind in a.kinds.split(","):
            for n in a.sizes.split(","):
                us, bw, algo = one(name, int(world), kind, int(n), a.algo)
                print(f"{name:<18} {world:>5} {kind:<11} {int(n):>11} {algo:<6} {us:>9.1f} {bw:>10.0f}")


if __name__ == "__main__":
    main()
