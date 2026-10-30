"""Rack-scale questions for an LLM, with reference answers and how to grade them.

Each question says where its reference answer comes from (``truth``):

* ``measured``: a measurement on real hardware (B300 cuBLAS / FlashAttention
  timings from research/ssm_vs_attention/results/validation_b300.json). The
  fairest test: the simulator can be wrong here too.
* ``simulated``: the cycle-level simulator, computed by ``make_refs.py``. The
  tool is graded against its own instrument, so these test whether an LLM can
  *get* the number (and whether a model without the tool lands near it), not
  whether the simulator is right; its own error band is in ``accuracy``.
* ``closed_form``: a formula with no free parameters (pipeline bubble, KV bytes,
  balls into bins). A model without tools can derive these.

``fields`` say what the answer must contain: a number with a relative
tolerance (full credit within ``tol``, half within ``2 tol``) or a choice.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MEASURED = os.path.join(ROOT, "research", "ssm_vs_attention", "results", "validation_b300.json")


def _measured(case: str) -> float:
    with open(MEASURED) as f:
        return json.load(f)["cases"][case]["measured_us"]


# ---------------------------------------------------------------------------
# reference functions (simulated / closed form); each returns {field: value, "source": ...}


def ref_plan_llama70b():
    from rackscale import tools
    r = tools.call("best_layout", {"model": "llama3-70b", "system": "b300", "phase": "decode", "context": 8192,
                                   "min_tps_user": 30})["best"]
    return {"tp": r["plan"]["tp"], "batch": r["batch"], "tokens_per_s_per_gpu": r["tok_s_gpu"],
            "source": "best_layout (cycle-level, fast mode): every TP and batch that fits"}


def ref_idle_llama8b():
    from rackscale import api
    r = api.run("hgx_b300_8", {"model": "llama3-8b", "phase": "decode", "batch": 32, "context": 4096,
                               "parallel": {"tp": 8}})
    from rackscale import analysis
    split = analysis.bottlenecks(api._rec(r["run_id"]), 1)["time_split"]
    bn = r["answer"]["bottleneck"]
    dominant = {"idle": "idle_gaps", "collectives": "communication"}.get(bn, "hbm" if bn == "kernels:hbm" else
                                                                        "compute" if bn == "kernels:tensor" else
                                                                        "latency")
    return {"dominant": dominant, "idle_fraction": split["idle_without_collectives"],
            "source": "simulate on hgx_b300_8, default (eager) launch gaps; idle = neither kernels nor collectives"}


def ref_gap_train():
    return {"ratio": 9141 / 4685, "largest_cause": "power_capped_clock",
            "source": "docs/book/11-every-parallelism.md 11.5, '1 GPU' row: textbook 4685 us, sim 9141 us; power "
                      "clock +2620 us of the 4456 us gap"}


def ref_predict_hbm():
    from rackscale import toolkit
    r = toolkit.predict_gpu("hgx_b300_8", {"model": "llama3-70b", "phase": "decode", "batch": 64, "context": 8192,
                                           "parallel": {"tp": 8}}, {"hbm_bw_gbps": "x1.5"})
    return {"speedup": r["speedup"], "source": "predict_gpu: simulate base and hbm_bw_gbps x1.5"}


def ref_predict_tensor():
    from rackscale import toolkit
    r = toolkit.predict_gpu("hgx_b200_8", {"model": "deepseek-v3", "phase": "decode", "batch": 128,
                                           "parallel": {"ep": 8}},
                            {"tensor_bf16_flops_per_cycle_per_sm": "x2", "tensor_fp8_flops_per_cycle_per_sm": "x2"})
    return {"speedup": r["speedup"], "source": "predict_gpu: simulate base and 2x tensor rate"}


def ref_power_cap():
    from rackscale import api, toolkit
    r = api.run("hgx_b300_8", {"model": "llama3-70b", "phase": "prefill", "batch": 1, "seqlen": 8192,
                               "parallel": {"tp": 8}})
    p = toolkit.power_thermal(r["run_id"], caps_w=[800])
    return {"slowdown_pct": p["caps"][0]["delta_pct"], "source": "power_thermal: re-simulated at an 800 W board cap"}


def ref_stragglers():
    return {"overhead_pct": 11.2, "source": "docs/book/16 stragglers table, N = 72, simulated (closed form 13.2%)"}


def ref_nvls_fault():
    return {"nvls_slowdown": 261.6 / 174.4, "ring_slowdown": 567.1 / 566.6,
            "source": "docs/book/16 NVLink table, simulated (closed form: 2x on the bandwidth term for NVLS)"}


def ref_allreduce():
    return {"time_us": 174.4, "source": "docs/book/16 NVLink table, healthy NVLS, simulated"}


def ref_ecmp():
    from rackscale import tail
    return {"busiest_over_mean": tail.ecmp_max_load(448, 8, 16, trials=20000), "source": "tail.ecmp_max_load Monte Carlo"}


def ref_bubble():
    return {"bubble_fraction": 7 / 39, "source": "(p - 1) / (m + p - 1)"}


def ref_kv():
    from rackscale import models
    m = models.ALL["llama3-70b"]
    return {"kv_bytes_per_token": m.attn_layers() * m.kv_elems_per_token_layer() * 2,
            "source": "layers x 2 x kv_heads x head_dim x 2 bytes"}


def ref_splitk():
    from rackscale import toolkit
    r = toolkit.gemm_tiling("hgx_b300_8", 16, 8192, 8192, vary={"split_k": [1, 2, 4, 8]})
    best = min(r["rows"], key=lambda x: x["time_us"]["value"])
    return {"best_split": int(best["variant"].split("=")[1]), "time_us": best["time_us"]["value"],
            "source": "gemm_tiling, split_k 1 / 2 / 4 / 8"}


def ref_hnet():
    from rackscale import toolkit
    r = toolkit.tokens_or_bytes(batch=64, context=8192, system="hgx_b300_8", simulate=True)
    return {"ratio": r["simulated"]["ratio"], "source": "tokens_or_bytes(simulate=True)"}


def ref_waves():
    import math
    tiles = math.ceil(4864 / 128) * math.ceil(4096 / 256)
    w = tiles / 148
    return {"waves": math.ceil(w), "efficiency": w / math.ceil(w), "source": "ceil(M/128) ceil(N/256) / 148 SMs"}


QUESTIONS = [
    # ---- inside one GPU, measured on a real B300
    dict(id="gemm_m1", category="kernel", truth="measured",
         prompt="On one NVIDIA B300 SXM6 GPU, a BF16 GEMM with M=1, N=28672, K=4096 (a decode-time MLP gate/up "
                "projection for one token) runs through cuBLAS inside a CUDA graph. How long does one such GEMM take, "
                "in microseconds?",
         fields={"time_us": {"type": "number", "tol": 0.15}},
         ref=lambda: {"time_us": _measured("gemm_mlp_gate_up_m1"), "source": "measured, CUDA-graph timing"}),
    dict(id="gemm_m256", category="kernel", truth="measured",
         prompt="On one NVIDIA B300 SXM6 GPU, how long does a BF16 cuBLAS GEMM with M=256, N=18560, K=4096 take "
                "(CUDA-graph timing), in microseconds?",
         fields={"time_us": {"type": "number", "tol": 0.15}},
         ref=lambda: {"time_us": _measured("gemm_mamba_in_proj_m256"), "source": "measured, CUDA-graph timing"}),
    dict(id="gemm_m1024", category="kernel", truth="measured",
         prompt="On one NVIDIA B300 SXM6 GPU, how long does a BF16 cuBLAS GEMM with M=1024, N=18560, K=4096 take "
                "(CUDA-graph timing), in microseconds?",
         fields={"time_us": {"type": "number", "tol": 0.15}},
         ref=lambda: {"time_us": _measured("gemm_mamba_in_proj_m1024"), "source": "measured, CUDA-graph timing"}),
    dict(id="attn_decode", category="kernel", truth="measured",
         prompt="On one NVIDIA B300 SXM6 GPU, decode attention (one new query token per sequence) for a batch of 64 "
                "sequences, 32 query heads, 8 KV heads, head dim 128, 16384 cached tokens per sequence, BF16 KV cache, "
                "with a FlashAttention split-KV kernel plus its combine kernel. How long does it take, in "
                "microseconds?",
         fields={"time_us": {"type": "number", "tol": 0.15}},
         ref=lambda: {"time_us": _measured("attn_decode_b64_ctx16384"), "source": "measured, CUDA-graph timing"}),
    dict(id="splitk", category="kernel", truth="simulated",
         prompt="A decode-shaped BF16 GEMM, M=16, N=8192, K=8192, on one B300 (148 SMs) with 128x128 output tiles. "
                "Which split-K factor among 1, 2, 4 and 8 is fastest, and how long does the GEMM take with it, in "
                "microseconds?",
         fields={"best_split": {"type": "choice", "options": [1, 2, 4, 8]},
                 "time_us": {"type": "number", "tol": 0.15}},
         ref=ref_splitk),
    dict(id="waves", category="kernel", truth="closed_form",
         prompt="A BF16 GEMM M=4864, N=4096, K=4096 on a 148-SM B300 uses 128x256 output tiles, one CTA per SM. How "
                "many waves does it run, and what fraction of the last-wave-rounded SM time does useful work "
                "(waves / ceil(waves))?",
         fields={"waves": {"type": "choice", "options": [1, 2, 3, 4, 5, 6]},
                 "efficiency": {"type": "number", "tol": 0.03}},
         ref=ref_waves),
    # ---- a node / a rack
    dict(id="plan_llama70b", category="plan", truth="simulated",
         prompt="Serve Llama-3-70B (BF16) decode on one HGX B300 8-GPU node at 8192 tokens of context per sequence. "
                "Each model replica uses TP in {1, 2, 4, 8} (replicas fill the node) and a decode batch per replica in "
                "{1, 8, 32, 128, 256}. Every user must get at least 30 output tokens/s. Which TP and batch give the "
                "most output tokens/s per GPU, and how many tokens/s per GPU is that?",
         fields={"tp": {"type": "choice", "options": [1, 2, 4, 8]},
                 "batch": {"type": "choice", "options": [1, 8, 32, 128, 256]},
                 "tokens_per_s_per_gpu": {"type": "number", "tol": 0.25}},
         ref=ref_plan_llama70b),
    dict(id="explain_idle", category="explain", truth="simulated",
         prompt="Llama-3-8B decode on an HGX B300 node, TP=8, batch 32, 4096 tokens of context, launched eagerly "
                "(no CUDA graphs). What takes the largest share of GPU time: compute-bound kernels, HBM-bound kernels, "
                "latency-bound kernels, communication (time inside all-reduces), or idle gaps (no kernel and no "
                "collective running)? What fraction of GPU time is such idle gaps?",
         fields={"dominant": {"type": "choice", "options": ["compute", "hbm", "latency", "communication", "idle_gaps"]},
                 "idle_fraction": {"type": "number", "tol": 0.2}},
         ref=ref_idle_llama8b),
    dict(id="explain_gap_train", category="explain", truth="simulated",
         prompt="A roofline estimate of a Llama-3-8B training step on one B300 (one 4096-token sequence, BF16; every "
                "matmul at the datasheet dense BF16 peak at the 2032 MHz boost clock, perfect overlap) gives T. How "
                "many times T does the real step take, and what is the largest single reason for the gap: "
                "power_capped_clock, kernel_inefficiency, memory_bandwidth, launch_overhead or communication?",
         fields={"ratio": {"type": "number", "tol": 0.2},
                 "largest_cause": {"type": "choice", "options": ["power_capped_clock", "kernel_inefficiency",
                                                                 "memory_bandwidth", "launch_overhead",
                                                                 "communication"]}},
         ref=ref_gap_train),
    dict(id="predict_hbm", category="predict", truth="simulated",
         prompt="Llama-3-70B decode, TP=8 on an HGX B300 node, batch 64, 8192 tokens of context (CUDA-graph-free "
                "default launches). A hypothetical B300 with 1.5x the HBM bandwidth and everything else equal: what is "
                "the speedup of the decode step (old time / new time)?",
         fields={"speedup": {"type": "number", "tol": 0.05}},
         ref=ref_predict_hbm),
    dict(id="predict_tensor", category="predict", truth="simulated",
         prompt="DeepSeek-V3 decode on an HGX B200 node with expert parallelism EP=8 (attention data parallel), batch "
                "128 per GPU, BF16. A hypothetical B200 with twice the tensor-core FLOP rate (BF16 and FP8) and "
                "everything else equal: what is the step speedup?",
         fields={"speedup": {"type": "number", "tol": 0.05}},
         ref=ref_predict_tensor),
    dict(id="power_cap", category="predict", truth="simulated",
         prompt="Llama-3-70B prefill of one 8192-token prompt with TP=8 on an HGX B300 node, whose GPUs normally have "
                "a 1100 W board power limit. If every GPU is capped at 800 W instead, by what percent does the "
                "prefill step get slower?",
         fields={"slowdown_pct": {"type": "number", "tol": 0.3}},
         ref=ref_power_cap),
    dict(id="hnet", category="predict", truth="simulated",
         prompt="Decode on one B300 with batch 64 and 8192 bytes of text context per sequence: H-Net 1-stage XL "
                "(byte-level, dynamic chunking, ~1.6B-class) against a BPE Transformer XL of similar size (Llama-3 "
                "tokenizer, ~4.45 bytes/token). What is the ratio of generated text bytes per second, H-Net / BPE?",
         fields={"ratio": {"type": "number", "tol": 0.2}},
         ref=ref_hnet),
    # ---- the fabric and the tail
    dict(id="allreduce_nvl72", category="network", truth="simulated",
         prompt="A 64 MiB BF16 all-reduce across all 72 GPUs of a GB300 NVL72 rack with NCCL's NVLS (NVLink SHARP) "
                "algorithm. How long does it take, in microseconds?",
         fields={"time_us": {"type": "number", "tol": 0.15}},
         ref=ref_allreduce),
    dict(id="nvls_fault", category="network", truth="simulated",
         prompt="On a GB300 NVL72 rack, one of GPU 0's 18 NVLink links to the NVSwitches runs at half rate. By what "
                "factor does a 64 MiB all-reduce over all 72 GPUs slow down (faulted / healthy time) with the NVLS "
                "algorithm, and with the ring algorithm?",
         fields={"nvls_slowdown": {"type": "number", "tol": 0.1}, "ring_slowdown": {"type": "number", "tol": 0.05}},
         ref=ref_nvls_fault),
    dict(id="stragglers", category="tail", truth="simulated",
         prompt="72 GPUs run 16 phases of (one 2048x4096x4096 BF16 GEMM, then a 1 MiB all-reduce over all 72). Every "
                "GEMM's time is independently multiplied by 1 + 0.05|Z|, Z standard normal. By how many percent of the "
                "jitter-free compute time does the step grow?",
         fields={"overhead_pct": {"type": "number", "tol": 0.2}},
         ref=ref_stragglers),
    dict(id="ecmp", category="tail", truth="closed_form",
         prompt="448 equal flows are hashed uniformly at random onto 8 equal-cost links, independently in each of 16 "
                "leaf switches. The busiest link of all 16 x 8 sets the finish time. What is the expected load of that "
                "busiest link divided by the mean load (448 / 8)?",
         fields={"busiest_over_mean": {"type": "number", "tol": 0.05}},
         ref=ref_ecmp),
    dict(id="bubble", category="parallelism", truth="closed_form",
         prompt="A 1F1B pipeline with 8 stages and 32 micro-batches per step: what fraction of each stage's step "
                "time is pipeline bubble?",
         fields={"bubble_fraction": {"type": "number", "tol": 0.03}},
         ref=ref_bubble),
    dict(id="kv_bytes", category="parallelism", truth="closed_form",
         prompt="How many bytes of BF16 KV cache does Llama-3-70B store per token (all layers)?",
         fields={"kv_bytes_per_token": {"type": "number", "tol": 0.02}},
         ref=ref_kv),
]


def schema_text(q: dict) -> str:
    parts = []
    for k, f in q["fields"].items():
        parts.append(f'"{k}": one of {f["options"]}' if f["type"] == "choice" else f'"{k}": <number>')
    return "{" + ", ".join(parts) + "}"
