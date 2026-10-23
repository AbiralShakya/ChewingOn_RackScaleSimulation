"""Study C: one long prompt, prefilled with context parallelism on one GPU, an HGX node and NVL72.

Softmax attention runs ring attention (K/V chunks circulate while the next block computes);
Mamba-2 and linear attention pass a fixed-size recurrent state along the ranks. Every kernel is
cycle-level on the calibrated B300 model (long ones through rackscale.extrap); every send/recv
crosses the simulated NVLink fabric; ranks are co-simulated, so pipelining across layers and
waiting on neighbours show up in the makespan.

  python research/dao_lab/cp_prefill.py
"""
import sys
import time

from common import load, save
from rackscale import extrap, lower, models, simulate

MODELS = ["llama3-8b", "llama3-8b-gta8", "mamba2-8b", "gla-8b", "gdn-8b", "gdn-hybrid-8b"]
SYSTEMS = {1: "hgx_b300_8", 8: "hgx_b300_8", 72: "gb300_nvl72"}
# Total prompt tokens: 72 x 2k, 72 x 8k, 72 x 16k.
SEQLENS = [147456, 589824, 1179648]
PEAK_BF16 = 2.46e15  # dense BF16 FLOP/s of one B300 at the measured max clock


def main(cps=(72, 8, 1)):
    rows = load("cp_prefill.json")
    done = {(r["model"], r["cp"], r["seqlen"]) for r in rows}
    for s in SEQLENS:
        for cp in cps:
            for name in MODELS:
                if (name, cp, s) in done:
                    continue
                m = models.ALL[name]
                system = SYSTEMS[cp]
                t0 = time.time()
                wl = lower.prefill_cp(m, cp, s)
                wl, log = extrap.freeze(system, wl)
                t1 = time.time()
                rep = simulate(system, wl, fast=True).raw
                mk = rep["makespan_us"]
                r0 = rep["ranks"][0]
                rl = rep["ranks"][-1]
                r = dict(model=name, cp=cp, seqlen=s, tokens_per_rank=s // cp, system=system,
                         makespan_us=mk, ttft_ms=mk / 1e3, mfu=wl["model_flops"] / (mk * 1e-6 * cp * PEAK_BF16),
                         rank0_busy_us=r0["gpu_busy_us"], last_busy_us=rl["gpu_busy_us"],
                         collective_us=rep["ranks"][0].get("collective_us"),
                         kernels=[{k: v for k, v in kk.items() if k in ("name", "count", "total_us")}
                                  for kk in rep["kernels"][:12]],
                         extrapolated=[{k: l[k] for k in ("op", "us", "rule")} for l in log],
                         freeze_s=t1 - t0, sim_s=time.time() - t1)
                print(f"S={s:<8d} cp{cp:<3d} {name:16s} TTFT {mk / 1e3:9.1f} ms  MFU {100 * r['mfu']:5.1f}%  "
                      f"rank0 busy {r0['gpu_busy_us'] / mk:5.1%}  ({r['freeze_s']:.0f}+{r['sim_s']:.0f}s)", flush=True)
                rows.append(r)
                save("cp_prefill.json", rows)


if __name__ == "__main__":
    main(tuple(int(x) for x in sys.argv[1:]) or (72, 8, 1))
