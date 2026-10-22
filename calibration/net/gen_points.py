#!/usr/bin/env python3
"""Regenerate calibration/net/published.json.

Two kinds of reference points:

* ``measured``: numbers published by NVIDIA, DeepSeek and the NCCL maintainers,
  entered by hand below with the source and the sentence they come from.
* ``nccl_model``: NCCL 2.32.3's own tuning-model prediction for the
  algorithm/protocol NCCL would pick (``rsim net model``), across systems,
  collectives and sizes. This is what NCCL itself believes; it is a useful
  dense target for latency/protocol behaviour but it is not a measurement.

Run from the repo root after ``cargo build --release``:

    python3 calibration/net/gen_points.py > calibration/net/published.json
"""
import json
import re
import subprocess

RSIM = "./target/release/rsim"

TRTLLM = (
    "https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/"
    "blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md"
)
DEEPEP = "https://github.com/deepseek-ai/DeepEP/blob/567632d/README.md"
DEEPEP_V2 = "https://github.com/deepseek-ai/DeepEP/blob/3e490fb/README.md"


def measured():
    pts = []
    # nccl-tests issue: the NCCL maintainer's expected peak for H100 NVLS.
    pts.append(dict(
        id="h100_8.allreduce.nvls_peak", system="hgx_h100_8", kind="all_reduce",
        bytes=1 << 30, metric="busbw_gbps", value=480.0, class_="measured", role="fit",
        source="https://github.com/NVIDIA/nccl-tests/issues/212",
        quote="On 8 GPUs, you should reach a peak busBW of ~480GB/s (H100, NVLS)"))
    # NVIDIA AI cluster runtime validation threshold for GB200 NVL72 all-reduce.
    pts.append(dict(
        id="nvl72.allreduce.nvls_floor", system="gb200_nvl72", kind="all_reduce", ranks=8,
        bytes=1 << 30, metric="busbw_gbps", value=450.0, class_="bound", role="holdout",
        source="https://github.com/NVIDIA/aicr/issues/2784",
        quote="nccl-all-reduce-bw-nvls >= 500 (validator gate threshold x 0.9), "
              "sized for a 2-node GB200 pair (8 GPUs)"))
    # TRT-LLM one-sided NVLink all-to-all on GB200 NVL72, DeepSeek-V3 profile, BF16.
    # bytes per rank = batch x min(ep, 8) x 14336; the reported bandwidth is
    # logical (includes the local share), i.e. our `gbps` metric with factor 1.
    tbl = {
        8: {8: (17.6, 31.4), 128: (31.7, 38.1), 512: (89.5, 91.7), 2048: (311.8, 322.6)},
        16: {8: (18.9, 31.8), 128: (34.6, 39.9), 512: (97.1, 98.2), 2048: (343.3, 348.4)},
        32: {8: (19.4, 32.2), 128: (37.8, 41.1), 512: (109.6, 102.5), 2048: (373.7, 369.5)},
        64: {8: (20.4, 33.3), 128: (41.6, 44.5), 512: (115.7, 106.7), 2048: (399.2, 372.1)},
    }
    for ep, rows in tbl.items():
        for bs, (disp, comb) in rows.items():
            fit = ep == 8 and bs in (8, 2048)
            pts.append(dict(
                id=f"nvl72.a2a.trtllm.ep{ep}.bs{bs}", system="gb200_nvl72", kind="all_to_all",
                ranks=ep, bytes=bs * min(ep, 8) * 14336, metric="time_us", value=disp,
                class_="measured", role="fit" if fit else "holdout", source=TRTLLM,
                quote=f"ep_size={ep}, batch {bs}: dispatch {disp} us (combine {comb} us)"))
    # DeepEP normal kernels on H800 (8 NVLink x 20 GB/s ~ 160 GB/s, CX7 400G).
    # 4096 tokens x 7168 x FP8, top-8: ~5.25 ranks per token inside EP8 -> ~154 MB
    # per rank. The intranode number is NVLink bytes / time.
    pts.append(dict(
        id="h800_8.a2a.deepep.ep8", system="hgx_h800_8", kind="all_to_all",
        bytes=154 << 20, metric="gbps", bytes_factor=7 / 8, value=153.0,
        class_="measured", role="holdout", source=DEEPEP,
        quote="Intranode | 8 | 153 GB/s (NVLink) | 8 | 158 GB/s (NVLink)"))
    pts.append(dict(
        id="h800_2n.a2a.deepep.ep16", system="h800_2node_ib", kind="all_to_all",
        bytes=154 << 20, metric="gbps", bytes_factor=8 / 16, value=43.0,
        class_="measured", role="holdout", source=DEEPEP,
        quote="Internode | 16 | 43 GB/s (RDMA) | 16 | 43 GB/s (RDMA)"))
    # DeepEP V2 on B200 (SM100), EP8 over NVLink, 8K tokens, logical bandwidth.
    pts.append(dict(
        id="b200_8.a2a.deepep_v2.ep8", system="hgx_b200_8", kind="all_to_all",
        bytes=308 << 20, metric="gbps", value=726.0,
        class_="measured", role="holdout", source=DEEPEP_V2,
        quote="SM100 | N/A | EP 8 | 726 GB/s (NVLink) | 740 GB/s (NVLink) | 64 (Max perf)"))
    return pts


SIZES = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
LINE = re.compile(r"^\s*(\d+)([KMG]?)\s+(\w+)/(\w+) ([\d.]+)us")


def nccl_model(system, coll, sizes, fit_every=2):
    out = subprocess.run(
        [RSIM, "net", "model", "--system", system, "--coll", coll, "-b", sizes[0], "-e", sizes[1],
         "-f", sizes[2]], check=True, capture_output=True, text=True).stdout
    pts = []
    for i, line in enumerate(l for l in out.splitlines() if LINE.match(l)):
        n, unit, algo, proto, us = LINE.match(line).groups()
        b = int(n) * SIZES.get(unit, 1)
        pts.append(dict(
            id=f"{system}.{coll}.{n}{unit}.nccl", system=system, kind=coll, bytes=b,
            metric="time_us", value=float(us), class_="nccl_model",
            role="fit" if i % fit_every == 0 else "holdout",
            source="NCCL 2.32.3 tuning model (src/graph/tuning.cc), `rsim net model`",
            quote=f"NCCL picks {algo}/{proto}: {us} us"))
    return pts


def main():
    pts = measured()
    for sys_ in ["hgx_h100_8", "hgx_b200_8", "gb200_nvl72", "h100_8node_ib", "hgx_b200_2node_ib"]:
        pts += nccl_model(sys_, "all_reduce", ["8K", "512M", "16"])
        pts += nccl_model(sys_, "all_gather", ["128K", "512M", "16"])
        pts += nccl_model(sys_, "reduce_scatter", ["128K", "512M", "16"])
    for p in pts:
        p["class"] = p.pop("class_")
    print(json.dumps({"points": pts}, indent=1))


if __name__ == "__main__":
    main()
