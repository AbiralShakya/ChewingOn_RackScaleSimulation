"""Record per-warp clock64 probe timelines on hardware (kernels/timeline.cu).

    python3 timeline.py build/timeline.cubin out.json [kernel:block ...]

For each (kernel, block) one CTA runs alone on the GPU. The kernel is
launched once to warm the instruction caches, then REPS times; per warp and
probe we keep the median over repetitions of (probe - min over all warps of
probe 0), i.e. times relative to the first warp's start on that SM.
"""
import json
import statistics
import struct
import sys

from cudrv import Device, u32s

NP = 64
REPS = 9
DEFAULT = [f"{k}:{b}" for k in ("t_alu", "t_softmax", "t_smem", "t_global", "t_hmma")
           for b in (32, 128, 256, 512, 1024)]


def main(cub, out, cfgs):
    dev = Device(0)
    m = dev.load(open(cub, "rb").read())
    words = NP * 4096 + 64 + 65536 + 64
    o = dev.alloc(4 * words)
    inp = dev.alloc(4 * 8192)
    import random
    random.seed(7)
    # A random permutation of 0..4095 (chases), then floats in [0.5, 1).
    perm = list(range(4096))
    random.shuffle(perm)
    vals = perm + [struct.unpack("<I", struct.pack("<f", 0.5 + random.random() / 2))[0] for _ in range(4096)]
    dev.h2d(inp, struct.pack("<8192I", *vals))
    res = {}
    for cfg in cfgs:
        name, block = cfg.split(":")
        block = int(block)
        k = m.fn(name)
        nw = block // 32
        runs = []
        for rep in range(REPS + 1):
            dev.memset(o, 4 * words)
            k.launch(1, block, [("p", o), ("p", inp), ("i", 0)])
            w = u32s(dev.d2h(o, 4 * NP * nw))
            if rep == 0:
                continue
            rows = [w[i * NP:(i + 1) * NP] for i in range(nw)]
            runs.append(rows)
        # Number of probes: the leading run of non-zero words (probe 0 can be
        # any value; later probes are larger). Determine from warp 0.
        r0 = runs[0][0]
        nprobe = next(p for p in range(1, NP - 1) if r0[p] == 0)
        med = []
        for wi in range(nw):
            row = []
            for p in range(nprobe):
                row.append(statistics.median(
                    (r[wi][p] - min(x[0] for x in r)) & 0xFFFFFFFF for r in runs))
            med.append(row)
        spread = statistics.median(
            max(r[wi][nprobe - 1] - r[wi][0] for wi in range(nw)) for r in runs)
        res[cfg] = {"kernel": name, "block": block, "nprobe": nprobe, "warps": med,
                    "smid": runs[0][0][NP - 1], "span_median": spread,
                    "rep_spans": [max(r[wi][nprobe - 1] for wi in range(nw)) - min(x[0] for x in r) for r in runs]}
        print(cfg, "probes", nprobe, "span", res[cfg]["rep_spans"], flush=True)
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or DEFAULT)
