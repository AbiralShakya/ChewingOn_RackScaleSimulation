"""Study A: one prefill mixer block on one B300, 1k to 1M tokens.

FlashAttention-4-style GQA attention, MLA, Mamba-2 (SSD), scalar-gated linear attention and
Gated DeltaNet, all at d_model 4096 (the 8B-class models in rackscale.models), batch 1. Each
mixer block is lowered exactly as in a full model (projections, conv, scan or attention, norm,
output projection) and every kernel runs on the calibrated B300 model; kernels past 8k x 8k
attention or 32k tokens use the long-context fast path (rackscale.extrap).

  python research/dao_lab/prefill_mixers.py
"""
import time

from common import GPU_SYSTEM, load, save
from rackscale import extrap, lower, models

MIXERS = {  # label: (model, block letter)
    "GQA-8 (FA4)": ("llama3-8b", "A"),
    "MLA": ("llama3-8b-mla", "A"),
    "Mamba-2 (SSD)": ("mamba2-8b", "M"),
    "Linear attn (scalar gate)": ("gla-8b", "L"),
    "Gated DeltaNet": ("gdn-8b", "L"),
}
SEQLENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def block_ops(m, letter, s):
    p = lower.Parallel()
    if letter == "A":
        return lower._attention_ops(m, p, 1, s, s, True, "")
    if letter == "M":
        return lower._mamba2_ops(m, p, 1, s, "")
    return lower._linear_ops(m, p, 1, s, "")


def main():
    rows = load("prefill_mixers.json")
    done = {(r["mixer"], r["seqlen"]) for r in rows}
    for s in SEQLENS:
        for label, (name, letter) in MIXERS.items():
            if (label, s) in done:
                continue
            m = models.ALL[name]
            t0 = time.time()
            parts, rules = {}, {}
            for op in block_ops(m, letter, s):
                if op.get("op") != "kernel":
                    continue
                us, how = extrap.estimate(GPU_SYSTEM, op["kernel"])
                parts[op["name"]] = parts.get(op["name"], 0.0) + us
                rules[op["name"]] = how["rule"]
            total = sum(parts.values())
            r = dict(mixer=label, model=name, seqlen=s, block_us=total, parts_us=parts, rules=rules,
                     us_per_token=total / s, wall_s=time.time() - t0)
            print(f"s={s:<8d} {label:26s} {total / 1e3:10.3f} ms  ({r['wall_s']:.0f}s)", flush=True)
            rows.append(r)
            save("prefill_mixers.json", rows)


if __name__ == "__main__":
    main()
