"""Concepts of rack-scale computing, each with its formula and a worked number.

``learn()`` lists the concepts; ``learn("roofline")`` explains one: the idea in
a few sentences, the formula, numbers worked out on a real preset (computed
now, from the same closed forms and hardware the simulator uses), the book
chapter that derives it, and the tool call that shows it in the simulator.
Parameters change the worked example: ``learn("pipeline_bubble", stages=8,
micro_batches=32)``.

It serves people reading along and agents that need the mechanism before they
answer, so every number here is computed, never typed in.
"""

from __future__ import annotations

import difflib
import math
from typing import Any, Callable, Dict, List

BOOK = "docs/book/"


def _hw(system: str = "gb300"):
    from . import scaling
    return scaling.hardware(system)


def _g(x: float, unit: str = "") -> str:
    for scale, suf in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= scale:
            return f"{x / scale:.3g} {suf}{unit}"
    return f"{x:.3g} {unit}".strip()


# ---------------------------------------------------------------------------
# each concept: (title, idea, formula, chapter, see, numbers(**params) -> dict)


def _roofline(system="gb300", dtype="bf16", m=16, n=8192, k=8192):
    from . import scaling
    hw = _hw(system)
    ridge = hw.peak(dtype) / hw.hbm_bw
    f, b = scaling.kernel_work({"type": "gemm", "m": m, "n": n, "k": k, "dtype": dtype})
    big_f, big_b = scaling.kernel_work({"type": "gemm", "m": 8192, "n": n, "k": k, "dtype": dtype})
    return {"gpu": hw.gpu, "peak_flops": _g(hw.peak(dtype), "FLOP/s"), "hbm_bw": _g(hw.hbm_bw, "B/s"),
            "ridge_flop_per_byte": round(ridge, 1),
            f"gemm_{m}x{n}x{k}": {"flop_per_byte": round(f / b, 1), "bound": "HBM" if f / b < ridge else "compute",
                                  "roofline_us": round(scaling.kernel_time({"type": "gemm", "m": m, "n": n, "k": k,
                                                                           "dtype": dtype}, hw) * 1e6, 2)},
            f"gemm_8192x{n}x{k}": {"flop_per_byte": round(big_f / big_b, 1),
                                   "bound": "HBM" if big_f / big_b < ridge else "compute"}}


def _decode_batch(system="gb300", model="llama3-70b", dtype="bf16"):
    from . import models
    hw = _hw(system)
    m = models.ALL[model]
    eb = {"bf16": 2.0, "fp8": 1.0, "fp4": 0.5}[dtype]
    ridge = hw.peak(dtype) / hw.hbm_bw
    # a decode GEMM of b rows against W (k x n): 2 b k n FLOPs over k n eb bytes -> 2 b / eb FLOP/B
    b_star = ridge * eb / 2
    p = m.active_params(lm_head=False)
    return {"model": model, "dtype": dtype, "weights_gb": round(p * eb / 1e9, 1),
            "flop_per_byte_at_batch_b": f"2 b / {eb:g} = {2 / eb:g} b",
            "batch_to_reach_ridge": round(b_star),
            "weight_read_us_per_step_on_one_gpu": round(p * eb / hw.hbm_bw * 1e6),
            "max_tokens_per_s_per_user_one_gpu": round(hw.hbm_bw / (p * eb), 1)}


def _waves(system="gb300", m=4608, n=4096, tile_m=128, tile_n=256, ctas_per_sm=1):
    hw_sms = {"gb300": 148, "b300": 148, "h100": 132, "a100": 108}.get(system)
    if hw_sms is None:
        from . import api
        hw_sms = api.resolve(system)["gpus"][0]["sm_count"]
    tiles = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    slots = hw_sms * ctas_per_sm
    waves = tiles / slots
    return {"sms": hw_sms, "tiles": tiles, "waves": round(waves, 3), "rounded_up": math.ceil(waves),
            "efficiency": round(waves / math.ceil(waves), 3),
            "lesson": f"{tiles} tiles on {slots} slots: the last wave runs {tiles - (math.ceil(waves) - 1) * slots} "
                      f"tiles while {math.ceil(waves) * slots - tiles} slots idle"}


def _littles_law(system="gb300"):
    from . import api
    hw = _hw(system)
    g = api.resolve({"gb300": "gb300_nvl72", "b300": "hgx_b300_8", "h100": "hgx_h100_8", "a100": "hgx_a100_8"}
                    .get(system, system))["gpus"][0]
    lat = g["hbm"]["latency_ns"] * 1e-9
    per_sm = hw.hbm_bw / g["sm_count"] * lat
    return {"hbm_bw": _g(hw.hbm_bw, "B/s"), "hbm_latency_ns": g["hbm"]["latency_ns"], "sms": g["sm_count"],
            "bytes_in_flight_gpu": _g(hw.hbm_bw * lat, "B"), "bytes_in_flight_per_sm": round(per_sm),
            "lesson": "a kernel that keeps fewer bytes in flight per SM than this cannot reach the HBM roof, however "
                      "much bandwidth the GPU has: it is latency bound"}


def _split_k(system="gb300", m=16, n=8192, k=8192, tile=128):
    sms = 148 if system in ("gb300", "b300") else 132
    tiles = math.ceil(m / tile) * math.ceil(n / tile)
    rows = []
    for s in (1, 2, 4, 8):
        ctas = tiles * s
        rows.append({"split": s, "ctas": ctas, "sm_fill": round(min(1.0, ctas / sms), 3),
                     "extra_fp32_partials_mb": round((2 * s * m * n * 4 if s > 1 else 0) / 1e6, 2)})
    return {"tiles": tiles, "sms": sms, "rows": rows,
            "lesson": "a decode GEMM has few output tiles; splitting K gives every SM work at the price of writing and "
                      "re-reading fp32 partials"}


def _power_cap(cap_w=800.0, ref_cap_w=1100.0, idle_w=261.0):
    from . import power
    t = power.cap_tradeoff(cap_w, ref_cap_w, idle_w)
    return {"cap_w": cap_w, "ref_cap_w": ref_cap_w, "idle_w": idle_w, "clock_ratio": round(t["clock"], 3),
            "time_ratio": round(t["time"], 3), "energy_ratio": round(t["energy"], 3),
            "lesson": f"{100 * (1 - cap_w / ref_cap_w):.0f}% less power costs only {100 * (t['time'] - 1):.0f}% more "
                      f"time for a compute-bound kernel, and saves {100 * (1 - t['energy']):.0f}% energy"}


def _all_reduce(system="gb300", mib=64, n=72):
    from . import scaling
    hw = _hw(system)
    b = mib * 2 ** 20
    ring = hw.alpha + 2 * (n - 1) / n * b / hw.link_bw(n)
    nvls = hw.alpha + b / hw.link_bw(n)
    small = scaling.collective("all_reduce", 2 ** 20, n, hw)[0]
    return {"bytes": _g(b, "B"), "ranks": n, "link_bw_per_gpu": _g(hw.link_bw(n), "B/s"),
            "ring_us": round(ring * 1e6, 1), "nvls_us": round(nvls * 1e6, 1), "nvls_available": hw.nvls,
            "1_MiB_us": round(small * 1e6, 1), "alpha_us": hw.alpha * 1e6,
            "lesson": "big all-reduces are bandwidth terms (NVLS halves them); small ones are all alpha"}


def _all_to_all(system="gb300", model="deepseek-v3", tokens_per_gpu=64, ep=8):
    from . import models, scaling
    hw = _hw(system)
    m = models.ALL[model]
    moe = m.moe
    per_tok = m.hidden * moe.topk * moe.dispatch_bytes
    b = tokens_per_gpu * per_tok
    t = scaling.collective("all_to_all", b, ep, hw)[0]
    return {"model": model, "topk": moe.topk, "hidden": m.hidden, "dispatch_bytes_per_token": _g(per_tok, "B"),
            "bytes_per_gpu": _g(b, "B"), "ep": ep, "dispatch_us": round(t * 1e6, 2),
            "tokens_per_expert_per_step": round(tokens_per_gpu * ep * moe.topk / moe.experts, 2),
            "lesson": "each token goes to top-k experts and comes back: two all-to-alls per MoE layer; tokens per "
                      "expert, not batch, sets how well the expert GEMMs use the GPU"}


def _tensor_parallel(system="gb300", model="llama3-70b", tp=8, seqlen=8192):
    from . import scaling
    hw = _hw(system)
    e = scaling.estimate(model, hw, "prefill", scaling.Plan(tp=tp), batch=1, seqlen=seqlen)
    terms = {t["name"]: round(t["us"], 1) for t in e["terms"]}
    return {"model": model, "tp": tp, "seqlen": seqlen, "step_us": round(e["step_us"], 1), "terms_us": terms,
            "dominant": e["dominant"],
            "lesson": "TP splits every matmul across GPUs and pays two all-reduces of the activations per layer; the "
                      "communication grows with tokens and hidden size, the compute per GPU shrinks with tp"}


def _pipeline_bubble(stages=4, micro_batches=8):
    p, m = stages, micro_batches
    return {"stages": p, "micro_batches": m, "gpipe_1f1b_bubble": round((p - 1) / (m + p - 1), 4),
            "lesson": "the first stage waits for nothing and the last for everything: p - 1 slots of m + p - 1 are "
                      "idle; more micro-batches, interleaving (virtual stages) or zero-bubble schedules shrink it"}


def _data_parallel(system="gb300", zero=0, n=72):
    from . import scaling
    hw = _hw(system)
    return {"zero": zero, "ranks": n, "critical_tokens_per_gpu": round(scaling.dp_critical_tokens(hw, zero, n)),
            "lesson": "the backward pass does 4PT FLOPs per GPU while DP moves ~2P-4P bytes: above this many tokens "
                      "per GPU per step the gradient traffic hides behind compute"}


def _kv_cache(model="llama3-70b", context=8192, hbm_gb=288.0):
    from . import models
    m = models.ALL[model]
    per_tok = m.attn_layers() * m.kv_elems_per_token_layer() * m.bytes_per_param()
    weights = m.total_params() * m.bytes_per_param()
    free = hbm_gb * 1e9 - weights
    return {"model": model, "kv_bytes_per_token": _g(per_tok, "B"), "kv_per_sequence": _g(per_tok * context, "B"),
            "weights": _g(weights, "B"), "sequences_that_fit_on_one_gpu": max(0, int(free // (per_tok * context))),
            "lesson": "every decode step reads the whole KV cache of every sequence, so long contexts make decode "
                      "memory bound on the cache, not the weights; GQA, MLA and FP8 KV shrink it"}


def _stragglers(n=72, sigma=0.05):
    from . import tail
    return {"ranks": n, "sigma": sigma, "expected_max_abs_z": round(tail.expected_max_half_normal(n), 3),
            "overhead": round(tail.straggler_overhead(n, sigma), 4), "one_gpu": round(tail.mean_overhead(sigma), 4),
            "lesson": "a synchronous step waits for the slowest rank; the expected maximum of N draws grows like "
                      "sqrt(2 ln N), so the same jitter costs 3x more on 72 GPUs than on one"}


def _link_faults(planes=18, scale=0.5):
    from . import tail
    return {"planes": planes, "dead_plane": round(tail.dead_plane_slowdown(planes), 4),
            "half_rate_plane": tail.degraded_plane_slowdown(planes, scale),
            "lesson": "a collective striped in lockstep over all planes runs at the slowest plane's pace: one dead "
                      "link costs P/(P-1), one slow link costs its whole slowdown"}


def _ecmp(flows=448, links=8, groups=16):
    from . import tail
    return {"flows": flows, "links": links, "groups": groups,
            "busiest_over_mean": round(tail.ecmp_max_load(flows, links, groups, trials=1000), 3),
            "lesson": "static hashing puts flows on links at random; the busiest link sets the finish time"}


def _tokens_vs_bytes():
    from . import hnet
    return {"bytes_per_bpe_token": hnet.BYTES_PER_BPE_TOKEN,
            "hnet_bytes_per_main_step": {"1-stage": 4.7, "2-stage": 6.9},
            "lesson": "a byte model's decoder steps 4.45x as often per byte of text as a BPE model, so its outer "
                      "layers must be cheap; H-Net's main network steps only at chunk boundaries"}


def _expert_parallel(model="deepseek-v3", batch_per_gpu=64, ep=72):
    from . import models
    m = models.ALL[model].moe
    return {"experts": m.experts, "topk": m.topk, "ep": ep, "experts_per_gpu": m.experts / ep,
            "tokens_per_expert": round(batch_per_gpu * ep * m.topk / m.experts, 1),
            "lesson": "with EP every GPU holds a few experts and receives the tokens routed to them from all GPUs: "
                      "rows per expert GEMM = batch x EP x top-k / experts, so EP turns tiny expert GEMMs into "
                      "efficient ones"}


def _critical_batch_train(system="gb300"):
    hw = _hw(system)
    return {"ridge_flop_per_byte": round(hw.peak() / hw.hbm_bw, 1),
            "lesson": "training matmuls at micro-batch x seqlen tokens are compute bound far above the ridge; what "
                      "limits large-scale training is communication (DP / FSDP traffic, TP all-reduces, pipeline "
                      "bubbles), which the parallel plan trades against memory"}


CONCEPTS: Dict[str, dict] = {
    "roofline": dict(
        title="The roofline: compute- or memory-bound",
        idea="A kernel takes at least its FLOPs over the peak FLOP rate and at least its bytes over the memory "
             "bandwidth. Which one is larger depends on arithmetic intensity (FLOPs per byte) against the GPU's "
             "ridge point (peak / bandwidth).",
        formula="t >= max(F / C, B / W); compute bound when F / B > C / W",
        chapter=BOOK + "01-rooflines.md", fn=_roofline,
        see={"tool": "inspect_kernel", "args": {"system": "hgx_b300_8", "spec": {"type": "gemm", "m": 16, "n": 8192,
                                                                                  "k": 8192}}}),
    "decode_batch": dict(
        title="Why decode is memory bound, and the batch that fixes it",
        idea="Generating one token per sequence reads every weight once per step. At batch b each weight byte "
             "serves 2b/bytes-per-param FLOPs, so small batches leave the tensor cores idle; per-user speed is "
             "capped by HBM bandwidth over model bytes.",
        formula="intensity = 2 b / bytes_per_param; b* = ridge x bytes_per_param / 2; tokens/s/user <= W / model bytes",
        chapter=BOOK + "06-inference.md", fn=_decode_batch,
        see={"tool": "sweep", "args": {"key": "workload.batch", "values": [1, 8, 64, 256]}}),
    "waves": dict(
        title="Waves and wave quantisation",
        idea="A GPU runs a grid of CTAs in waves of (SMs x CTAs per SM). A grid one tile past a full wave needs a "
             "second wave that mostly idles.",
        formula="waves = CTAs / (SMs x CTAs per SM); efficiency = waves / ceil(waves)",
        chapter=BOOK + "15-parallelism-inside-one-gpu.md", fn=_waves,
        see={"tool": "gemm_tiling", "args": {"m": 4608, "n": 4096, "k": 4096, "base": {"tile_n": 256}}}),
    "littles_law": dict(
        title="Little's law: bytes in flight",
        idea="To sustain bandwidth W through a memory with latency L, W x L bytes must be in flight. Each SM must "
             "keep its share outstanding, which is why kernels pipeline loads several stages deep.",
        formula="bytes in flight = W x L; per SM = W / SMs x L",
        chapter=BOOK + "15-parallelism-inside-one-gpu.md", fn=_littles_law,
        see={"tool": "gemm_tiling", "args": {"vary": {"stages": [1, 2, 3, 4]}}}),
    "split_k": dict(
        title="Split-K for skinny GEMMs",
        idea="A decode GEMM (few rows) has too few output tiles to fill the GPU. Splitting the K loop over several "
             "CTAs multiplies the CTAs and adds a reduction of fp32 partials.",
        formula="CTAs = tiles x split; extra bytes = 2 x split x M x N x 4",
        chapter=BOOK + "15-parallelism-inside-one-gpu.md", fn=_split_k,
        see={"tool": "gemm_tiling", "args": {"m": 16, "n": 8192, "k": 8192, "vary": {"split_k": [1, 2, 4, 8]}}}),
    "warp_specialization": dict(
        title="Warp specialisation",
        idea="Blackwell kernels give warps roles: a TMA producer loads tiles, one warp issues tensor-core MMAs, "
             "others do softmax or the epilogue, handing buffers over through mbarriers. Where the time goes is a "
             "per-role question: FlashAttention-4 on B300 has softmax warps 0-7, correction 8-11, epilogue 12, MMA "
             "13 and load 14.",
        formula="kernel time = the slowest role's critical path; the others park on mbarriers (async_wait)",
        chapter=BOOK + "03-kernels-sass-cycles.md", fn=lambda: {},
        see={"tool": "inspect_kernel", "args": {"system": "hgx_b300_8", "spec": {"type": "attention", "batch": 1,
                                                                                   "heads_q": 32, "heads_kv": 8,
                                                                                   "sq": 4096, "skv": 4096, "d": 128,
                                                                                   "causal": True}}}),
    "power_cap": dict(
        title="The power cap moves the roof",
        idea="Dynamic power grows roughly with the cube of the clock. A kernel that reaches the board power cap "
             "runs at the clock the cap allows, so the effective peak is a property of the workload, not the chip "
             "(B300 BF16 GEMMs drop from 2032 to ~1300-1455 MHz).",
        formula="P = P_idle + D (f / f_max)^3; at the cap, f ~ (cap - idle)^(1/3)",
        chapter=BOOK + "17-power-cooling-and-the-clock.md", fn=_power_cap,
        see={"tool": "power_thermal", "args": {"caps_w": [800, 950]}}),
    "all_reduce": dict(
        title="All-reduce: ring vs in-switch (NVLS)",
        idea="A ring all-reduce sends each byte around twice (reduce-scatter then all-gather). NVSwitch with "
             "SHARP-style reduction (NVLS) sends each byte in once and reads the result once.",
        formula="ring: alpha + 2 (n-1)/n B / W; NVLS: alpha + B / W",
        chapter=BOOK + "04-networks-collectives.md", fn=_all_reduce,
        see={"tool": "simulate", "args": {"system": "gb300_nvl72", "workload": {"workload": {"name": "ar", "spmd": {
            "world": 72, "program": {"streams": [[{"op": "collective", "kind": "all_reduce", "bytes": 67108864,
                                                   "group": "world"}]]}}}}}}),
    "all_to_all": dict(
        title="All-to-all for mixture-of-experts",
        idea="Expert parallelism sends each token's activations to the GPUs holding its top-k experts and back.",
        formula="alpha + (n-1)/n B / W per direction, B = tokens x hidden x top-k x bytes",
        chapter=BOOK + "05-parallelism.md", fn=_all_to_all,
        see={"tool": "simulate", "args": {"system": "hgx_b200_8", "workload": {"model": "deepseek-v3",
                                                                              "phase": "decode", "batch": 128,
                                                                              "parallel": {"ep": 8}}}}),
    "expert_parallel": dict(
        title="Expert parallelism and tokens per expert",
        idea="Spreading experts over more GPUs gathers more tokens onto each expert, turning many tiny GEMMs into "
             "fewer efficient ones, at the cost of all-to-all traffic.",
        formula="rows per expert = batch per GPU x EP x top-k / experts",
        chapter=BOOK + "11-every-parallelism.md", fn=_expert_parallel,
        see={"tool": "best_layout", "args": {"model": "deepseek-v3", "system": "gb300", "phase": "decode"}}),
    "tensor_parallel": dict(
        title="Tensor parallelism",
        idea="TP shards each weight matrix over GPUs; every layer then all-reduces its activations twice (or "
             "reduce-scatters and all-gathers with sequence parallelism).",
        formula="compute / tp + 2 x layers x all_reduce(tokens x hidden x 2 B)",
        chapter=BOOK + "11-every-parallelism.md", fn=_tensor_parallel,
        see={"tool": "scaling_math", "args": {"model": "llama3-70b", "phase": "prefill", "plan": {"tp": 8}}}),
    "pipeline_bubble": dict(
        title="Pipeline bubbles",
        idea="Pipeline parallelism splits layers into stages; filling and draining the pipeline idles stages.",
        formula="bubble = (p - 1) / (m + p - 1) for GPipe and 1F1B; zero-bubble schedules split the backward pass",
        chapter=BOOK + "07-training.md", fn=_pipeline_bubble,
        see={"tool": "check_scaling_math", "args": {"model": "llama3-8b", "plan": {"pp": 2, "schedule": "1f1b"},
                                                    "micro_batches": 8}}),
    "data_parallel": dict(
        title="Data parallelism, ZeRO and FSDP",
        idea="Replicas exchange gradients each step; ZeRO shards optimizer state, gradients and (FSDP) weights at "
             "the price of more gathers. Enough tokens per GPU hide the traffic behind the backward pass.",
        formula="hidden when tokens per GPU >= C / W (DDP ring, FSDP) or C / 2W (ZeRO-1/2)",
        chapter=BOOK + "11-every-parallelism.md", fn=_data_parallel,
        see={"tool": "scaling_ladder", "args": {"model": "llama3-8b", "phase": "train"}}),
    "kv_cache": dict(
        title="The KV cache",
        idea="Attention keeps keys and values of every past token; decode reads all of it each step. It sets how "
             "many sequences fit and, at long context, the step time.",
        formula="bytes / token = layers x 2 x kv_heads x head_dim x bytes (MLA: the latent instead)",
        chapter=BOOK + "06-inference.md", fn=_kv_cache,
        see={"tool": "sweep", "args": {"key": "workload.context", "values": [2048, 8192, 32768]}}),
    "critical_batch_train": dict(
        title="Training is compute bound; the plan is about communication",
        idea="Training matmuls run thousands of tokens per weight read, far above the ridge. The parallel plan "
             "decides what is communicated and how much of it hides behind compute.",
        formula="matmul time = 6 N T / (world x C); exposed communication decides the rest",
        chapter=BOOK + "07-training.md", fn=_critical_batch_train,
        see={"tool": "scaling_ladder", "args": {"model": "llama3-70b", "phase": "train"}}),
    "stragglers": dict(
        title="Stragglers: the maximum of N",
        idea="Bulk-synchronous steps wait for the slowest rank at every collective. Independent jitter costs the "
             "expected maximum, which grows with the number of ranks.",
        formula="overhead = sigma x E[max_N |Z|] ~ sigma sqrt(2 ln N); one GPU: sigma x 0.798",
        chapter=BOOK + "16-stragglers-flaps-tail-latency.md", fn=_stragglers,
        see={"tool": "tail_latency", "args": {"ranks": 72, "sigma": 0.05, "simulate": True}}),
    "link_faults": dict(
        title="Link faults and degraded planes",
        idea="NVLS stripes a collective across all of a GPU's links in lockstep, so one slow link slows the whole "
             "collective; a ring is barely affected.",
        formula="dead plane: P / (P - 1); plane at rate s: max(1, 1 / s) on the bandwidth term",
        chapter=BOOK + "16-stragglers-flaps-tail-latency.md", fn=_link_faults,
        see={"tool": "tail_latency", "args": {"ranks": 72, "plane_scale": 0.5, "simulate": True}}),
    "ecmp": dict(
        title="ECMP hash collisions",
        idea="Static multipath hashing places flows on equal-cost links at random; collisions overload some links.",
        formula="busiest link / mean = E[max of balls in bins] / (flows / links)",
        chapter=BOOK + "16-stragglers-flaps-tail-latency.md", fn=_ecmp,
        see={"tool": "tail_latency", "args": {"flows": 448, "links": 8, "groups": 16}}),
    "tokens_vs_bytes": dict(
        title="Tokens or bytes (H-Net)",
        idea="A byte-level model sees ~4.45x more positions than a BPE model for the same text. H-Net's router "
             "compresses bytes into chunks so the big main network steps rarely; the small outer layers step on "
             "every byte.",
        formula="main-network steps per byte = 1 / prod(R_s); decode reads outer weights every byte",
        chapter=BOOK + "13-tokens-or-bytes.md", fn=_tokens_vs_bytes,
        see={"tool": "tokens_or_bytes", "args": {"simulate": True}}),
    "colocation": dict(
        title="Many models on one rack",
        idea="A rack can be partitioned (disjoint GPUs sharing switches) or shared (MPS-style, models on the same "
             "GPUs competing for SMs, L2 and HBM). Interference shows up as slowdown against running alone, and "
             "flows between tenants (KV caches, weights) compete for links.",
        formula="slowdown = step together / step alone",
        chapter=BOOK + "12-many-models-one-rack.md", fn=lambda: {},
        see={"tool": "simulate_colocated", "args": {"tenants": [
            {"name": "chat", "model": "llama3-8b", "phase": "decode", "plan": {"tp": 4}, "gpus": "0-3", "batch": 64},
            {"name": "rl", "model": "llama3-8b", "phase": "train", "plan": {"tp": 4}, "gpus": "4-7"}]}}),
}


def topics() -> List[dict]:
    return [{"topic": k, "title": v["title"]} for k, v in CONCEPTS.items()]


def learn(topic: str = None, **params: Any) -> dict:
    """Explain ``topic`` with its formula and a worked example (see the module docstring)."""
    if not topic:
        return {"topics": topics(), "usage": "learn(topic, **params) for the idea, formula, worked numbers, chapter "
                                             "and the tool call that shows it in the simulator"}
    key = topic.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in CONCEPTS:
        near = difflib.get_close_matches(key, list(CONCEPTS), n=3, cutoff=0.3)
        hits = [k for k, v in CONCEPTS.items() if key.replace("_", " ") in (v["title"] + " " + v["idea"]).lower()]
        raise KeyError(f"unknown topic {topic!r}; did you mean {near or hits[:3]}? topics: {list(CONCEPTS)}")
    c = CONCEPTS[key]
    fn: Callable = c["fn"]
    try:
        numbers = fn(**params)
    except TypeError as e:
        raise ValueError(f"{key}: {e}") from None
    return {"topic": key, "title": c["title"], "idea": c["idea"], "formula": c["formula"],
            "worked_example": numbers, "chapter": c["chapter"], "see_it_in_the_simulator": c["see"]}
