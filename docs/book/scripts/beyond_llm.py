"""Chapter 18: vision encoders, diffusion transformers, VLMs and recommenders.

    maturin develop --release
    python docs/book/scripts/beyond_llm.py               # everything (about 15 minutes on 4 cores)
    python docs/book/scripts/beyond_llm.py vit dit       # or: video, vlm, recsys

vit     ViT-L/14 and ViT-H/14 (one B300, batch 64) and ViT-22B (HGX B300, TP8), 224 px: images per
        second, and where the time goes against the closed-form FLOPs
dit     DiT-XL/2 at 256 and 512 px, 50 steps with classifier-free guidance, batch 8, one B300
video   one denoising step of a 14B video DiT (Wan2.1-14B shape) over 81 frames of 720p
        (75,600 latent tokens, 512 text tokens), Ulysses over 8 B300s and over 40 GB300s
vlm     Llama-3-8B with a ViT-L/14 at 336 px: prefill of 8 requests with 1 image (576 tokens)
        or 5 tiles (2880), and 256 text tokens, on one B300
recsys  a 2.6 TB DLRM-style model, 4096 samples per GPU: a training step on 16, 32 and 64
        GPUs of a GB300 NVL72 and on 8 HGX B300 nodes over 800G InfiniBand
"""

import concurrent.futures as cf
import copy
import os
import sys

import rackscale as rs
from rackscale import api, extrap, lower, models, recsys, vision

B300_PEAK = 148 * 8192 * 2032e6  # dense BF16 FLOP/s at the full clock


def pmap(f, xs):
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        return list(ex.map(f, xs))


def run(system, wl):
    wl, log = extrap.freeze(system, wl)
    return rs.simulate(system, wl, fast=True), log


def split(r, world):
    """Per-GPU kernel time by class (attention, GEMM, other), in µs."""
    out = {"attention": 0.0, "gemm": 0.0, "other": 0.0}
    for k in r.kernels(top=1000):
        n = k["name"]
        c = "attention" if "fmha" in n else "gemm" if "gemm" in n else "other"
        out[c] += k["total_us"] / world
    return out


def vit():
    cases = [(models.VIT_L_14, rs.Parallel(), 64), (models.VIT_H_14, rs.Parallel(), 64),
             (models.VIT_22B, rs.Parallel(tp=8), 64)]
    res = pmap(lambda c: run("hgx_b300_8", vision.vit_forward(c[0], c[1], batch=c[2]))[0], cases)
    print("\n## ViT encoders at 224 px (257 tokens per image), batch 64\n")
    print("| model | GPUs | params | GFLOP per image | step (ms) | images/s | TFLOP/s per GPU | % of peak "
          "| attention / GEMM / other | attention FLOPs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (m, p, b), r in zip(cases, res):
        w = p.world()
        f = vision.encoder_flops(m, 1, 257, 588, 256)
        att = 4.0 * m.layers * m.heads * m.head_dim * 257 * 257
        s = split(r, w)
        tot = sum(s.values())
        print(f"| {m.name} | {w} | {m.active_params(lm_head=False) / 1e9:.2f} B | {f / 1e9:.0f} | "
              f"{r.makespan_us / 1e3:.2f} | {b / r.makespan_us * 1e6:.0f} | {b * f / r.makespan_us / 1e6 / w:.0f} | "
              f"{100 * b * f / (r.makespan_us * 1e-6) / w / B300_PEAK:.0f}% | "
              f"{100 * s['attention'] / tot:.0f} / {100 * s['gemm'] / tot:.0f} / {100 * s['other'] / tot:.0f}% | "
              f"{100 * att / f:.0f}% |")


def dit():
    cases = [(32, "256 px"), (64, "512 px")]
    m = models.DIT_XL_2
    res = pmap(lambda c: run("hgx_b300_8", vision.dit_sample(m, batch=8, latent=c[0], steps=50))[0], cases)
    print("\n## DiT-XL/2, 50 steps with classifier-free guidance (16 sequences per step), batch 8, one B300\n")
    print("| image | latent tokens | TFLOP per image | time for 8 images (ms) | images/s | % of peak "
          "| attention share of kernel time | attention share of FLOPs |")
    print("|---|---|---|---|---|---|---|---|")
    for (lat, lab), r in zip(cases, res):
        n = vision.patches(lat, 2)
        f = vision.dit_flops(m, 1, n, steps=50)
        att = 50 * 2 * 4.0 * m.layers * m.heads * m.head_dim * n * n
        s = split(r, 1)
        peak = 100 * 8 * f / (r.makespan_us * 1e-6) / B300_PEAK
        print(f"| {lab} | {n} | {f / 1e12:.1f} | {r.makespan_us / 1e3:.0f} | {8 / r.makespan_us * 1e6:.1f} | "
              f"{peak:.0f}% | {100 * s['attention'] / sum(s.values()):.0f}% | {100 * att / f:.0f}% |")


def video():
    m = models.VIDEO_DIT_14B
    kw = dict(batch=1, latent=(90, 160), frames=21, patch=2, steps=1, text_tokens=512, latent_channels=16,
              learn_sigma=False)
    cases = [("hgx_b300_8", 8), ("gb300_nvl72", 40)]
    res = pmap(lambda c: run(c[0], vision.dit_sample(m, cp=c[1], **kw)), cases)
    n = vision.patches((90, 160), 2, 21)
    f = vision.dit_flops(m, 1, n, steps=1, text_tokens=512)
    att = 2 * 4.0 * m.layers * m.heads * m.head_dim * n * n
    print(f"\n## One denoising step of a 14B video DiT: {n} latent tokens, CFG, 512 text tokens\n")
    print(f"closed form: {f / 1e15:.1f} PFLOP per step, {100 * att / f:.0f}% of it self-attention\n")
    print("| system | GPUs (Ulysses) | step (s) | 50 steps (s) | % of peak | self-attention share of the step "
          "| all-to-all share |")
    print("|---|---|---|---|---|---|---|")
    for (system, cp), (r, log) in zip(cases, res):
        # The long self-attention kernels run as extrapolated fixed ops, once per block.
        att_us = m.layers * sum(x["us"] for x in log if x["op"].endswith(".attention"))
        peak = 100 * f / (r.makespan_us * 1e-6) / cp / B300_PEAK
        coll = 100 * r.raw["ranks"][0]["collective_us"] / r.makespan_us
        print(f"| {system} | {cp} | {r.makespan_us / 1e6:.2f} | {50 * r.makespan_us / 1e6:.0f} | {peak:.0f}% | "
              f"{100 * att_us / r.makespan_us:.0f}% | {coll:.0f}% |")


def vlm():
    llm, enc = models.LLAMA3_8B, models.VIT_L_14
    cases = [1, 5]
    jobs = []
    for im in cases:
        jobs += [vision.vlm_prefill(llm, enc, batch=8, images=im, image=336, text=256),
                 vision.vit_forward(enc, batch=8 * im, image=336, cls_token=False),
                 lower.prefill(llm, rs.Parallel(), 8, im * 576 + 256)]
    res = pmap(lambda wl: run("hgx_b300_8", wl)[0].makespan_us, jobs)
    print("\n## VLM prefill: Llama-3-8B + ViT-L/14 at 336 px, 8 requests, 256 text tokens each, one B300\n")
    print("| images per request | image tokens | prefill (ms) | ViT alone (ms) | LLM alone (ms) | ViT share "
          "| ViT share of FLOPs |")
    print("|---|---|---|---|---|---|---|")
    for i, im in enumerate(cases):
        tot, v, lm = res[3 * i:3 * i + 3]
        fs = vision.vlm_flops(llm, enc, 8, images=im, image=336, text=256)
        print(f"| {im} | {im * 576} | {tot / 1e3:.1f} | {v / 1e3:.1f} | {lm / 1e3:.1f} | {100 * v / tot:.0f}% | "
              f"{100 * fs['vit'] / sum(fs.values()):.0f}% |")


def ib_b300(nodes=8):
    """8 HGX B300 nodes, one 800G NIC per GPU, rail-optimised fat tree (the H800 cluster's layout)."""
    s = copy.deepcopy(api.resolve("h800_8node_ib")["spec"])
    s["name"] = f"{nodes}x HGX B300, 800G IB"
    s["gpus"] = [{"count": 8 * nodes, "preset": "b300"}]
    s["topology"]["intra"] = api.resolve("hgx_b300_8")["spec"]["topology"]
    s["topology"]["nic"]["gbps_per_dir"] = 100.0
    s["topology"]["leaf"]["gbps_per_port_per_dir"] = 100.0
    return api.resolve(s)["yaml"]


def recsys_():
    m = recsys.DLRM_2TB
    ib = ib_b300()
    gb = "GB300 NVL72"
    cases = [("gb300_nvl72", 16, gb), ("gb300_nvl72", 32, gb), ("gb300_nvl72", 64, gb),
             (ib, 64, "8 x HGX B300, 800G IB")]
    res = pmap(lambda c: run(c[0], recsys.step(m, c[1], 4096)), cases)
    print(f"\n## {m.name}: {m.tables} tables x {m.rows / 1e6:.0f}M rows x {m.dim} fp16 "
          f"({m.embedding_bytes() / 1e12:.1f} TB), {m.pooling} lookups per table, 4096 samples per GPU\n")
    print("| system | GPUs | tables per GPU (GB) | step (ms) | gather | update | all-to-all, each | "
          "MLPs and the rest | samples/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (system, w, lab), (r, log) in zip(cases, res):
        t = {x["op"]: x["us"] for x in log}
        a2a = [(c["done_ps"] - c["first_post_ps"]) / 1e6 for c in r.raw["collectives"] if c["kind"] == "AllToAll"]
        rest = r.makespan_us - t["embedding_lookup"] - t["embedding_update"] - sum(a2a)
        print(f"| {lab} | {w} | {m.embedding_bytes() / w / 1e9:.0f} | {r.makespan_us / 1e3:.2f} | "
              f"{t['embedding_lookup'] / 1e3:.2f} | {t['embedding_update'] / 1e3:.2f} | {a2a[0] / 1e3:.2f} | "
              f"{rest / 1e3:.2f} | {w * 4096 / r.makespan_us * 1e6 / 1e6:.1f} M |")
    nv = recsys.balance(m, 64, 4096, 6600, 630)
    ibf = recsys.balance(m, 64, 4096, 6600, 100, domain=8)
    print(f"\nclosed forms per GPU: gather {nv['lookup_bytes'] / 1e9:.2f} GB, "
          f"{nv['lookup_us'] / 1e3:.2f} ms at 6.6 TB/s; "
          f"all-to-all {nv['a2a_bytes'] / 1e6:.0f} MB at 630 GB/s (NVLink) = {nv['a2a_us'] / 1e3:.2f} ms, "
          f"{ibf['a2a_bytes'] / 1e6:.0f} MB through a 100 GB/s NIC = {ibf['a2a_us'] / 1e3:.2f} ms")
    print(f"all-to-all / gather: {nv['a2a_over_lookup']:.2f} on NVL72, {ibf['a2a_over_lookup']:.2f} over IB")


EXPERIMENTS = {"vit": vit, "dit": dit, "video": video, "vlm": vlm, "recsys": recsys_}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(EXPERIMENTS):
        EXPERIMENTS[name]()
