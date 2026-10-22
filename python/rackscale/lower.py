"""Lower a model + parallelism + phase into a simulator workload.

The output is the JSON workload format consumed by ``rsim-rack`` (see
``docs/WORKLOAD_FORMAT.md``): an SPMD program with CUDA-like streams of
kernels and collectives. Every GEMM / attention / norm becomes a synthetic
SASS kernel (TMA + HGMMA on Hopper, TMA + tcgen05 on Blackwell) that the
cycle-level SM model executes.

Layouts supported:

* dense / MoE with tensor parallelism (``tp``) for attention and FFN;
* MoE with expert parallelism (``ep``) and data-parallel attention, the
  DeepSeek-V3 inference layout (``Parallel(ep=72)``: every rank owns its own
  batch for attention, experts are sharded, tokens move by all-to-all);
* data parallelism (``dp``) for training with per-layer gradient
  reduce-scatter overlapped on a second stream;
* two-micro-batch overlap (``overlap=True``): two streams, each carrying half
  the batch, so one micro-batch's all-to-all hides behind the other's compute;
* pipeline parallelism (``pp``) for training: contiguous layer slices per
  stage, activations and their gradients sent point to point between
  neighbouring stages, with GPipe, 1F1B or zero-bubble (ZB-H1) schedules over
  ``micro_batches`` micro-batches (:func:`train_step`).

Every workload also carries ``memory``: the bytes each GPU needs for weights,
KV cache, activations, gradients and optimizer state, computed for the *full*
model even when only a few ``layers`` are simulated. The simulator reports it
against the GPU's HBM and warns when a configuration would not fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import Model


@dataclass(frozen=True)
class Parallel:
    tp: int = 1
    ep: int = 1
    dp: int = 1
    #: SMs given to communication kernels (NCCL channels / DeepEP SMs).
    comm_sms: int = 20
    #: Pipeline stages (training). Ranks are laid out tp-innermost, then dp,
    #: then pp: rank = (stage * dp + d) * tp + t.
    pp: int = 1

    def world(self) -> int:
        return max(self.tp * self.dp * self.pp, self.ep, 1)


class ParallelError(ValueError):
    """A parallel layout that the model cannot be split into."""


def check(m: Model, p: Parallel) -> None:
    """Reject layouts that do not divide the model: heads over TP, experts
    over EP, layers over PP. Raises :class:`ParallelError` with the fix."""
    errs = []
    for name, v in (("tp", p.tp), ("ep", p.ep), ("dp", p.dp), ("pp", p.pp)):
        if v < 1:
            errs.append(f"{name}={v} must be >= 1")
    if errs:
        raise ParallelError("; ".join(errs))
    if p.tp > 1:
        if m.heads % p.tp:
            errs.append(f"tp={p.tp} does not divide {m.name}'s {m.heads} attention heads")
        if not m.mla and m.kv_heads % p.tp and p.tp % m.kv_heads:
            errs.append(f"tp={p.tp} and {m.kv_heads} KV heads: one must divide the other")
        if m.mamba is not None and m.mamba.nheads % p.tp:
            errs.append(f"tp={p.tp} does not divide {m.mamba.nheads} Mamba-2 heads")
    if p.ep > 1:
        if not m.moe:
            errs.append(f"ep={p.ep} needs an MoE model; {m.name} is dense (use tp or dp)")
        elif p.ep > m.moe.experts:
            errs.append(f"ep={p.ep} is more ranks than {m.name}'s {m.moe.experts} experts")
        # An EP degree that does not divide the experts (DeepSeek-V3's 256 over
        # 72 GPUs) is how it is deployed, with redundant experts filling the
        # remainder; the lowering gives each rank experts // ep of them.
        if p.tp > 1 or p.dp > 1 or p.pp > 1:
            errs.append("ep is the DeepSeek inference layout (data-parallel attention, sharded experts) and does "
                        "not combine with tp/dp/pp here")
    n_layers = len(m.block_list()) if m.blocks is not None else m.layers
    if p.pp > n_layers:
        errs.append(f"pp={p.pp} is more stages than {m.name}'s {n_layers} layers")
    if errs:
        raise ParallelError("; ".join(errs))


def _k(name: str, **kernel) -> dict:
    return {"op": "kernel", "name": name, "kernel": kernel}


def _gemm(name: str, m: int, n: int, k: int, dtype: str = "bf16") -> dict:
    return _k(name, type="gemm", m=max(1, int(m)), n=max(1, int(n)), k=max(1, int(k)), dtype=dtype)


def _coll(kind: str, nbytes: float, group: str, sms: int, name: str = "") -> dict:
    d = {"op": "collective", "kind": kind, "bytes": int(max(1, nbytes)), "group": group, "sms": sms}
    if name:
        d["name"] = name
    return d


def _attention_ops(m: Model, p: Parallel, batch: int, sq: int, skv: int, causal: bool, tag: str) -> List[dict]:
    """Attention block for `batch` sequences of `sq` new tokens each."""
    t = batch * sq
    h = m.hidden
    tp = p.tp
    ops = [_k(f"{tag}input_norm", type="row_norm", rows=t, cols=h)]
    if m.mla:
        a = m.mla
        heads = max(1, m.heads // tp)
        ops += [
            _gemm(f"{tag}q_a_proj", t, a.q_lora_rank, h),
            _gemm(f"{tag}q_b_proj", t, heads * (a.nope_dim + a.rope_dim), a.q_lora_rank),
            _gemm(f"{tag}kv_a_proj", t, a.kv_lora_rank + a.rope_dim, h),
        ]
        if sq == 1:
            # Absorbed MLA decode (FlashMLA-style): MQA over the latent cache.
            ops.append(_k(f"{tag}mla_decode", type="attention", batch=batch, heads_q=heads, heads_kv=1, sq=1, skv=skv,
                          d=a.kv_lora_rank + a.rope_dim))
        else:
            ops.append(_gemm(f"{tag}kv_b_proj", t, heads * (a.nope_dim + a.v_head_dim), a.kv_lora_rank))
            ops.append(_k(f"{tag}mla_prefill", type="attention", batch=batch, heads_q=heads, heads_kv=heads, sq=sq, skv=skv,
                          d=a.nope_dim + a.rope_dim, causal=causal))
        ops.append(_gemm(f"{tag}o_proj", t, h, heads * a.v_head_dim))
    else:
        hq = max(1, m.heads // tp)
        hkv = max(1, m.kv_heads // tp)
        ops += [
            _gemm(f"{tag}qkv_proj", t, (hq + 2 * hkv) * m.head_dim, h),
            _k(f"{tag}attention", type="attention", batch=batch, heads_q=hq, heads_kv=hkv, sq=sq, skv=skv, d=m.head_dim,
               causal=causal),
        ]
        if sq == 1:
            # Split-KV decode (FA2/FA3/FA4) ends with a combine kernel over the
            # fp32 partial outputs and log-sum-exps.
            ops.append(_k(f"{tag}attn_combine", type="elementwise", n=batch * hq * m.head_dim * 4, inputs=1, elem_bytes=4.0,
                          flops_per_elem=4))
        ops.append(_gemm(f"{tag}o_proj", t, h, hq * m.head_dim))
    if tp > 1:
        ops.append(_coll("all_reduce", t * h * 2, f"block:{tp}", p.comm_sms, f"{tag}attn_allreduce"))
    ops.append(_k(f"{tag}residual", type="elementwise", n=t * h, inputs=2, flops_per_elem=1))
    return ops


def _ffn_ops(m: Model, p: Parallel, layer: int, t: int, tag: str) -> List[dict]:
    h = m.hidden
    ops = [_k(f"{tag}post_attn_norm", type="row_norm", rows=t, cols=h)]
    moe = m.moe
    if moe and layer >= moe.first_dense_layers:
        ep = p.ep
        local = max(1, moe.experts // ep)
        # Tokens arriving at this rank's experts (balanced routing).
        recv = t * moe.topk
        # Spread the tokens as evenly as routing allows. With fewer tokens than
        # local experts (small decode batches over many experts) only that many
        # experts run, and the others' weights are never read.
        ms = [n for n in (recv // local + (1 if i < recv % local else 0) for i in range(local)) if n > 0]
        ops += [
            _gemm(f"{tag}router", t, moe.experts, h),
            _k(f"{tag}topk", type="elementwise", n=t * moe.experts, inputs=1, flops_per_elem=4),
        ]
        if ep > 1:
            ops.append(_coll("all_to_all", t * moe.topk * h * moe.dispatch_bytes, f"block:{ep}", p.comm_sms, f"{tag}moe_dispatch"))
        # Without EP, tensor parallelism splits every expert's FFN over the TP
        # ranks (vLLM / SGLang FusedMoE) and all-reduces the output.
        efn = max(1, moe.expert_ffn // p.tp)
        ops += [
            _k(f"{tag}experts_gate_up", type="grouped_gemm", ms=ms, n=2 * efn, k=h),
            _k(f"{tag}experts_act", type="elementwise", n=max(1, recv * efn), inputs=2, flops_per_elem=4),
            _k(f"{tag}experts_down", type="grouped_gemm", ms=ms, n=h, k=efn),
        ]
        if moe.shared_experts:
            sf = max(1, moe.shared_experts * moe.expert_ffn // p.tp)
            ops += [_gemm(f"{tag}shared_gate_up", t, 2 * sf, h), _gemm(f"{tag}shared_down", t, h, sf)]
        if ep > 1:
            ops.append(_coll("all_to_all", t * moe.topk * h * 2, f"block:{ep}", p.comm_sms, f"{tag}moe_combine"))
        elif p.tp > 1:
            ops.append(_coll("all_reduce", t * h * 2, f"block:{p.tp}", p.comm_sms, f"{tag}moe_allreduce"))
    else:
        ffn = (moe.dense_ffn if moe and moe.dense_ffn else m.ffn) // p.tp
        ops += [
            _gemm(f"{tag}gate_up_proj", t, 2 * ffn, h),
            _k(f"{tag}act_mul", type="elementwise", n=t * ffn, inputs=2, flops_per_elem=4),
            _gemm(f"{tag}down_proj", t, h, ffn),
        ]
        if p.tp > 1:
            ops.append(_coll("all_reduce", t * h * 2, f"block:{p.tp}", p.comm_sms, f"{tag}mlp_allreduce"))
    ops.append(_k(f"{tag}residual2", type="elementwise", n=t * h, inputs=2, flops_per_elem=1))
    return ops


def _mamba2_ops(m: Model, p: Parallel, batch: int, sq: int, tag: str) -> List[dict]:
    """Mamba-2 mixer block (state-spaces/mamba ``Mamba2``), heads split over TP.

    Decode (``sq == 1``) is ``Mamba2.step``: in_proj, ``causal_conv1d_update``,
    ``selective_state_update``, gated RMSNorm, out_proj. Prefill is
    ``Mamba2.forward``: in_proj, ``causal_conv1d_fn``, the five Triton kernels of
    ``mamba_chunk_scan_combined`` (SSD), gated RMSNorm, out_proj.
    """
    mb = m.mamba
    tp = p.tp
    t = batch * sq
    h = m.hidden
    heads = max(1, mb.nheads // tp)
    groups = max(1, mb.ngroups // tp)
    d_in = heads * mb.headdim
    conv_dim = d_in + 2 * groups * mb.d_state
    ops = [_k(f"{tag}input_norm", type="row_norm", rows=t, cols=h),
           _gemm(f"{tag}in_proj", t, 2 * d_in + 2 * groups * mb.d_state + heads, h)]
    ssm = dict(batch=batch, heads=heads, headdim=mb.headdim, d_state=mb.d_state, groups=groups)
    ops.append(_k(f"{tag}conv1d", type="conv1d", batch=batch, dim=conv_dim, seqlen=sq, width=mb.d_conv))
    if sq == 1:
        extra = {"block_m": mb.state_block_m} if mb.state_block_m else {}
        ops.append(_k(f"{tag}ssm_state_update", type="ssm_state_update", **ssm, **extra))
    else:
        for stage in ("chunk_cumsum", "chunk_state", "state_passing", "bmm_chunk", "chunk_scan"):
            ops.append(_k(f"{tag}ssd_{stage}", type="ssd", stage=stage, seqlen=sq, chunk=mb.chunk, **ssm))
    ops += [_k(f"{tag}gated_norm", type="row_norm", rows=t, cols=d_in),
            _gemm(f"{tag}out_proj", t, h, d_in)]
    if tp > 1:
        ops.append(_coll("all_reduce", t * h * 2, f"block:{tp}", p.comm_sms, f"{tag}mamba_allreduce"))
    ops.append(_k(f"{tag}residual", type="elementwise", n=t * h, inputs=2, flops_per_elem=1))
    return ops


def _mlp_ops(m: Model, p: Parallel, t: int, tag: str) -> List[dict]:
    """Standalone MLP block (hybrid stacks): pre-norm, gate/up, act, down, residual."""
    h = m.hidden
    ffn = m.ffn // p.tp
    ops = [_k(f"{tag}mlp_norm", type="row_norm", rows=t, cols=h),
           _gemm(f"{tag}gate_up_proj", t, 2 * ffn, h),
           _k(f"{tag}act_mul", type="elementwise", n=t * ffn, inputs=2, flops_per_elem=4),
           _gemm(f"{tag}down_proj", t, h, ffn)]
    if p.tp > 1:
        ops.append(_coll("all_reduce", t * h * 2, f"block:{p.tp}", p.comm_sms, f"{tag}mlp_allreduce"))
    ops.append(_k(f"{tag}mlp_residual", type="elementwise", n=t * h, inputs=2, flops_per_elem=1))
    return ops


def _layers(m: Model, p: Parallel, batch: int, sq: int, skv: int, causal: bool, tag: str = "") -> List[dict]:
    ops: List[dict] = []
    t = batch * sq
    if m.blocks is not None:
        for b in m.blocks:
            if b == "M":
                ops += _mamba2_ops(m, p, batch, sq, tag)
            elif b == "A":
                ops += _attention_ops(m, p, batch, sq, skv, causal, tag)
            elif b == "F":
                ops += _mlp_ops(m, p, t, tag)
            else:
                raise ValueError(f"unknown block {b!r} in {m.name}")
        return ops
    for layer in range(m.layers):
        ops += _attention_ops(m, p, batch, sq, skv, causal, tag)
        ops += _ffn_ops(m, p, layer, t, tag)
    return ops


def _model_flops(m: Model, tokens: int, skv: int, train: bool, lm_head: bool) -> float:
    """Model FLOPs of the lowered work: matmuls over the active parameters,
    attention score/value FLOPs at mean KV length ``skv``, the SSD scan, and the
    LM head only when the workload contains one (prefill and training lower no
    LM head, so counting it inflated their MFU)."""
    dense = 2.0 * m.active_params(lm_head=lm_head) * tokens
    attn = 4.0 * m.attn_layers() * m.heads * m.head_dim * skv * tokens
    ssm = 0.0
    if m.mamba is not None:
        mb = m.mamba
        n_m = m.block_list().count("M")
        # Per token and head: state update + output contraction (decode), or
        # chunk state + inter-chunk output + intra-chunk (C B^T) x (prefill).
        per_head = 4.0 * mb.headdim * mb.d_state
        if skv > 1:
            per_head += 2.0 * mb.headdim * (mb.chunk / 2) + 2.0 * mb.d_state * (mb.chunk / 2) / max(1, mb.nheads // mb.ngroups)
        ssm = n_m * mb.nheads * per_head * tokens
    return (3.0 if train else 1.0) * (dense + attn + ssm)


def _weights_per_gpu(m: Model, p: Parallel) -> float:
    """Parameters held by the fullest GPU: TP shards attention, dense FFNs,
    embedding and LM head; EP shards routed experts (attention and shared
    experts are replicated, the DeepSeek inference layout); PP gives each
    stage a contiguous slice of layers, the last stage keeping the LM head."""
    tp, ep, pp = p.tp, p.ep, p.pp
    if m.blocks is not None:
        per = {"M": m.mamba_params() if m.mamba else 0, "A": m.attn_params(), "F": m.mlp_params()}
        layer_params = [per[b] / tp for b in m.blocks]
    else:
        layer_params = []
        for l in range(m.layers):
            attn = m.attn_params() / tp
            if m.moe and l >= m.moe.first_dense_layers:
                mo = m.moe
                routed = 3 * m.hidden * mo.expert_ffn * mo.experts / (ep * tp)
                shared = 3 * m.hidden * mo.expert_ffn * mo.shared_experts / tp
                layer_params.append(attn + routed + shared + m.hidden * mo.experts)
            else:
                layer_params.append(attn + m.ffn_params(l) / tp)
    per_stage = -(-len(layer_params) // pp)
    stages = [sum(layer_params[i:i + per_stage]) for i in range(0, len(layer_params), per_stage)]
    emb = m.vocab * m.hidden / tp
    stages[0] += emb
    stages[-1] += emb  # LM head (untied)
    return max(stages)


def _kv_bytes_per_token(m: Model, p: Parallel, elem_bytes: float = 2.0) -> float:
    """Inference cache per token on one GPU, all layers (Mamba-2 state is per
    sequence and handled separately)."""
    if m.mla:
        # The compressed latent is not head-sharded: every TP rank keeps it.
        per_layer = (m.mla.kv_lora_rank + m.mla.rope_dim) * elem_bytes
    else:
        per_layer = 2 * max(1, m.kv_heads // p.tp) * m.head_dim * elem_bytes
    return per_layer * m.attn_layers() / p.pp


def memory(m: Model, p: Parallel, *, phase: str, batch: int = 1, context: int = 0, seqlen: int = 0,
           micro_batches: int = 1, schedule: str = "1f1b") -> Dict[str, float]:
    """Bytes the fullest GPU needs, by category, for the whole model.

    First-order and stated so it can be checked: weights in the model dtype;
    KV cache for ``batch`` sequences of ``context`` tokens (``batch`` is per
    rank, as in :func:`decode_step`); Mamba-2 conv and SSM state; for
    training, BF16 gradients and FP32 Adam state (master weights, m, v) both
    sharded over ``dp`` (ZeRO-2, which is what :func:`train_step` lowers),
    and activations of about ``34 * tokens * hidden`` bytes per layer
    divided by ``tp`` (Korthikanti et al. 2022 with FlashAttention and
    sequence parallelism, no recomputation), times the micro-batches a stage
    holds at once (``pp`` for 1F1B, all of them for GPipe).
    """
    w = _weights_per_gpu(m, p)
    out = {"weights": w * m.bytes_per_param()}
    if phase in ("decode", "prefill"):
        ctx = context if phase == "decode" else seqlen
        out["kv_cache"] = batch * ctx * _kv_bytes_per_token(m, p)
        if m.mamba is not None:
            mb = m.mamba
            n_m = m.block_list().count("M")
            state = (mb.nheads * mb.headdim * mb.d_state * 4 + mb.conv_dim * (mb.d_conv - 1) * 2) / p.tp
            out["ssm_state"] = batch * n_m * state / p.pp
        if phase == "prefill":
            t = batch * seqlen
            ffn = max(m.ffn, m.moe.expert_ffn * m.moe.topk if m.moe else 0)
            out["activations"] = t * (4 * m.hidden + 2 * ffn / p.tp) * 2.0
    elif phase == "train":
        out["grads"] = 2.0 * w / p.dp
        out["optimizer"] = 12.0 * w / p.dp
        layers = len(m.block_list()) if m.blocks is not None else m.layers
        in_flight = micro_batches if schedule == "gpipe" else min(micro_batches, p.pp)
        per_layer = 34.0 * batch * seqlen * m.hidden / p.tp
        out["activations"] = per_layer * (-(-layers // p.pp)) * in_flight
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return out


def decode_step(m: Model, p: Parallel, batch: int, context: int, *, overlap: bool = False,
                layers: Optional[int] = None, lm_head: bool = True) -> dict:
    """One decode iteration: every rank generates one token for `batch` sequences."""
    check(m, p)
    mem = memory(m, p, phase="decode", batch=batch, context=context)
    if layers is not None:
        m = m.with_layers(layers)
    world = p.world()
    streams = []
    halves = [batch // 2, batch - batch // 2] if overlap and batch > 1 else [batch]
    for i, b in enumerate(halves):
        tag = f"mb{i}." if len(halves) > 1 else ""
        ops = _layers(m, p, b, 1, context, False, tag)
        if lm_head:
            ops += [_k(f"{tag}final_norm", type="row_norm", rows=b, cols=m.hidden),
                    _gemm(f"{tag}lm_head", b, m.vocab // max(1, p.tp), m.hidden)]
        streams.append(ops)
    return {
        "name": f"{m.name} decode b{batch} ctx{context} {p}",
        "spmd": {"world": world, "program": {"streams": streams}},
        "model_flops": _model_flops(m, batch * world // max(1, p.tp), context, False, lm_head),
        "memory": mem,
    }


def prefill(m: Model, p: Parallel, batch: int, seqlen: int, *, layers: Optional[int] = None) -> dict:
    check(m, p)
    mem = memory(m, p, phase="prefill", batch=batch, seqlen=seqlen)
    if layers is not None:
        m = m.with_layers(layers)
    world = p.world()
    ops = _layers(m, p, batch, seqlen, seqlen, True)
    return {
        "name": f"{m.name} prefill b{batch} s{seqlen} {p}",
        "spmd": {"world": world, "program": {"streams": [ops]}},
        "model_flops": _model_flops(m, batch * seqlen * world // max(1, p.tp), seqlen // 2, False, lm_head=False),
        "memory": mem,
    }


def mixed_step(m: Model, p: Parallel, decode_batch: int, context: int, chunk: int, *, prefix: int = 0,
               layers: Optional[int] = None) -> dict:
    """One continuous-batching iteration with chunked prefill (vLLM / SGLang
    default): ``decode_batch`` sequences generate a token while one prompt
    contributes a ``chunk``-token slice that starts at position ``prefix``.

    The GEMMs see ``decode_batch + chunk`` rows; attention runs as two
    kernels, the decode one over ``context`` cached tokens and a causal one
    for the chunk over its ``prefix + chunk`` keys. GQA/MHA attention only.
    """
    check(m, p)
    if m.mla or m.blocks is not None:
        raise ValueError(f"mixed_step lowers GQA/MHA transformers; {m.name} has MLA or a block stack")
    mem = memory(m, p, phase="decode", batch=decode_batch + 1, context=max(context, prefix + chunk))
    if layers is not None:
        m = m.with_layers(layers)
    t = decode_batch + chunk
    h, tp = m.hidden, p.tp
    hq, hkv = max(1, m.heads // tp), max(1, m.kv_heads // tp)
    ops: List[dict] = []
    for layer in range(m.layers):
        ops += [_k("input_norm", type="row_norm", rows=t, cols=h),
                _gemm("qkv_proj", t, (hq + 2 * hkv) * m.head_dim, h)]
        if decode_batch:
            ops += [_k("attention", type="attention", batch=decode_batch, heads_q=hq, heads_kv=hkv, sq=1, skv=context,
                       d=m.head_dim),
                    _k("attn_combine", type="elementwise", n=decode_batch * hq * m.head_dim * 4, inputs=1,
                       elem_bytes=4.0, flops_per_elem=4)]
        if chunk:
            ops.append(_k("chunk_attention", type="attention", batch=1, heads_q=hq, heads_kv=hkv, sq=chunk,
                          skv=prefix + chunk, d=m.head_dim, causal=True))
        ops.append(_gemm("o_proj", t, h, hq * m.head_dim))
        if tp > 1:
            ops.append(_coll("all_reduce", t * h * 2, f"block:{tp}", p.comm_sms, "attn_allreduce"))
        ops.append(_k("residual", type="elementwise", n=t * h, inputs=2, flops_per_elem=1))
        ops += _ffn_ops(m, p, layer, t, "")
    # Only sequences that finish a token (decode rows, plus the prompt when
    # this is its last chunk) need logits; count the decode rows.
    rows = max(1, decode_batch)
    ops += [_k("final_norm", type="row_norm", rows=rows, cols=h), _gemm("lm_head", rows, m.vocab // max(1, tp), h)]
    flops = _model_flops(m, decode_batch, context, False, True) + _model_flops(m, chunk, prefix + chunk // 2, False, False)
    return {
        "name": f"{m.name} mixed d{decode_batch} ctx{context} + chunk{chunk}@{prefix} {p}",
        "spmd": {"world": p.world(), "program": {"streams": [ops]}},
        "model_flops": flops * p.world() / max(1, tp),
        "memory": mem,
    }


TENSOR_KERNELS = ("gemm", "grouped_gemm", "attention")


def _layer_ops(m: Model, p: Parallel, idx: int, batch: int, seqlen: int, tag: str) -> List[dict]:
    """Forward ops of layer ``idx`` (a block of a block stack)."""
    t = batch * seqlen
    if m.blocks is not None:
        b = m.blocks[idx]
        if b == "M":
            return _mamba2_ops(m, p, batch, seqlen, tag)
        if b == "A":
            return _attention_ops(m, p, batch, seqlen, seqlen, True, tag)
        return _mlp_ops(m, p, t, tag)
    return _attention_ops(m, p, batch, seqlen, seqlen, True, tag) + _ffn_ops(m, p, idx, t, tag)


def _layer_params(m: Model, p: Parallel, idx: int) -> float:
    """Parameters of layer ``idx`` on one GPU (all experts of an MoE layer)."""
    if m.blocks is not None:
        per = {"M": m.mamba_params() if m.mamba else 0, "A": m.attn_params(), "F": m.mlp_params()}
        return per[m.blocks[idx]] / max(1, p.tp)
    return (m.attn_params() + m.ffn_params(idx)) / max(1, p.tp)


def _backward(fwd: List[dict]) -> tuple:
    """Backward of a layer's forward ops, split the way zero-bubble schedules
    split it: ``B`` (input gradients: dgrad GEMMs, attention backward, which
    needs two passes for dQ and dK/dV, every non-tensor kernel and the
    collectives) and ``W`` (weight-gradient GEMMs). Each backward GEMM has
    the forward GEMM's FLOPs."""
    b, w = [], []
    for op in fwd:
        b.append(op)
        if op["op"] == "kernel":
            ty = op["kernel"]["type"]
            if ty == "attention":
                b.append(op)
            elif ty in ("gemm", "grouped_gemm"):
                w.append(op)
    return b, w


def _pp_order(schedule: str, pp: int, stage: int, mbs: int) -> List[tuple]:
    """Per-stage op order: ("F", i), ("B", i), ("W", i)."""
    if schedule == "gpipe":
        seq = [("F", i) for i in range(mbs)]
        for i in range(mbs):
            seq += [("B", i), ("W", i)]
        return seq
    if schedule not in ("1f1b", "zb"):
        raise ValueError(f"unknown pipeline schedule {schedule!r} (gpipe, 1f1b, zb)")
    warm = min(mbs, pp - stage - 1)
    # ZB-H1 (Qi et al. 2023): weight gradients wait `lag` backward passes, so
    # in the drain, where 1F1B idles waiting for the next stage's gradient,
    # the stage computes deferred W passes instead.
    lag = pp - stage - 1 if schedule == "zb" else 0
    seq = [("F", i) for i in range(warm)]
    pending: List[int] = []
    f = warm
    for b in range(mbs):
        if f < mbs:
            seq.append(("F", f))
            f += 1
        seq.append(("B", b))
        pending.append(b)
        while len(pending) > lag:
            seq.append(("W", pending.pop(0)))
        if f >= mbs and pending and lag:
            # Drain: one deferred W after each remaining B.
            seq.append(("W", pending.pop(0)))
    seq += [("W", i) for i in pending]
    return seq


def train_step(m: Model, p: Parallel, micro_batch: int, seqlen: int, *, layers: Optional[int] = None,
               grad_bucket_layers: int = 1, micro_batches: int = 1, schedule: str = "1f1b",
               optimizer: bool = False) -> dict:
    """One training step.

    Without pipeline parallelism and with one micro-batch, this is forward +
    backward of that micro-batch with the DP gradient reduce-scatter
    overlapped on a communication stream (FSDP/ZeRO-2 style).

    With ``p.pp > 1`` or ``micro_batches > 1`` the step is a pipeline over
    ``micro_batches`` micro-batches with the given ``schedule`` (``gpipe``,
    ``1f1b`` or ``zb`` for ZB-H1 zero-bubble), each stage owning a contiguous
    slice of layers and sending activations / activation gradients to its
    neighbours on their own streams. Gradients of the last micro-batch are
    reduce-scattered over DP per layer as they are produced; with
    ``optimizer=True`` the step ends with an Adam update of each rank's
    shard and an all-gather of the new BF16 weights.
    """
    check(m, p)
    mem = memory(m, p, phase="train", batch=micro_batch, seqlen=seqlen, micro_batches=micro_batches,
                 schedule=schedule)
    if layers is not None:
        m = m.with_layers(layers)
    if p.pp > 1 or micro_batches > 1 or optimizer:
        wl = _train_pipeline(m, p, micro_batch, seqlen, micro_batches, schedule, grad_bucket_layers, optimizer)
        wl["memory"] = mem
        return wl
    world = p.world()
    t = micro_batch * seqlen
    fwd = _layers(m, p, micro_batch, seqlen, seqlen, True, "fwd.")
    compute: List[dict] = list(fwd)
    comm: List[dict] = []
    params_per_layer = (m.attn_params() + m.ffn_active_params(m.layers - 1)) / max(1, p.tp)
    for layer in reversed(range(m.layers)):
        # Backward ~= 2x forward GEMM work (dgrad + wgrad) plus attention bwd.
        lay = _attention_ops(m, p, micro_batch, seqlen, seqlen, True, f"bwd{layer}.") + _ffn_ops(m, p, layer, t, f"bwd{layer}.")
        for op in lay:
            compute.append(op)
            if op["op"] == "kernel" and op["kernel"]["type"] in TENSOR_KERNELS:
                compute.append(op)  # second pass: weight-gradient / dK,dV
        if p.dp > 1 and layer % grad_bucket_layers == 0:
            ev = f"grad{layer}"
            compute.append({"op": "record", "event": ev})
            comm.append({"op": "wait", "event": ev})
            comm.append(_coll("reduce_scatter", params_per_layer * grad_bucket_layers * 2, f"stride:{p.tp}" if p.tp > 1 else "world",
                              p.comm_sms, f"grad_rs{layer}"))
    streams = [compute] + ([comm] if comm else [])
    return {
        "name": f"{m.name} train mb{micro_batch} s{seqlen} {p}",
        "spmd": {"world": world, "program": {"streams": streams}},
        "model_flops": _model_flops(m, t * world // max(1, p.tp), seqlen // 2, True, lm_head=False),
        "memory": mem,
    }


def _train_pipeline(m: Model, p: Parallel, micro_batch: int, seqlen: int, mbs: int, schedule: str,
                    grad_bucket_layers: int, optimizer: bool) -> dict:
    tp, dp, pp = p.tp, p.dp, p.pp
    world = tp * dp * pp
    n_layers = len(m.blocks) if m.blocks is not None else m.layers
    per_stage = -(-n_layers // pp)
    act_bytes = micro_batch * seqlen * m.hidden * 2 // tp
    # Zero-bubble schedules send the input gradient as soon as B is done.
    split_w = schedule == "zb"
    ranks = []
    for r in range(world):
        stage, d, t = r // (tp * dp), (r // tp) % dp, r % tp
        mine = list(range(stage * per_stage, min(n_layers, (stage + 1) * per_stage)))
        dp_group = [(stage * dp + dd) * tp + t for dd in range(dp)]
        nxt, prv = r + tp * dp, r - tp * dp
        first, last = stage == 0, stage == pp - 1
        fwd = [op for l in mine for op in _layer_ops(m, p, l, micro_batch, seqlen, f"L{l}.")]
        bwd = {}
        for l in reversed(mine):
            bwd[l] = _backward(_layer_ops(m, p, l, micro_batch, seqlen, f"L{l}.bwd."))
        comp: List[dict] = []
        grads: List[dict] = []
        fsend: List[dict] = []
        frecv: List[dict] = []
        bsend: List[dict] = []
        brecv: List[dict] = []
        bucket = 0.0
        for kind, i in _pp_order(schedule, pp, stage, mbs):
            if kind == "F":
                if not first:
                    comp.append({"op": "wait", "event": f"fin{i}"})
                comp += fwd
                if not last:
                    comp.append({"op": "record", "event": f"fout{i}"})
            elif kind == "B":
                if not last:
                    comp.append({"op": "wait", "event": f"bin{i}"})
                for l in reversed(mine):
                    comp += bwd[l][0]
                if not first and split_w:
                    comp.append({"op": "record", "event": f"bout{i}"})
            else:
                for l in reversed(mine):
                    comp += bwd[l][1]
                    if dp > 1 and i == mbs - 1:
                        bucket += _layer_params(m, p, l)
                        if (l - mine[0]) % grad_bucket_layers == 0:
                            ev = f"grad{l}"
                            comp.append({"op": "record", "event": ev})
                            grads += [{"op": "wait", "event": ev},
                                      _coll("reduce_scatter", bucket * 2, dp_group, p.comm_sms, f"grad_rs{l}")]
                            bucket = 0.0
                if not first and not split_w:
                    # GPipe and 1F1B run each backward as one pass, so the
                    # activation gradient leaves after the weight gradients.
                    comp.append({"op": "record", "event": f"bout{i}"})
        for i in range(mbs):
            if not last:
                fsend += [{"op": "wait", "event": f"fout{i}"}, {"op": "send", "bytes": act_bytes, "peer": nxt}]
                brecv += [{"op": "recv", "bytes": act_bytes, "peer": nxt}, {"op": "record", "event": f"bin{i}"}]
            if not first:
                frecv += [{"op": "recv", "bytes": act_bytes, "peer": prv}, {"op": "record", "event": f"fin{i}"}]
                bsend += [{"op": "wait", "event": f"bout{i}"}, {"op": "send", "bytes": act_bytes, "peer": prv}]
        if optimizer:
            shard = sum(_layer_params(m, p, l) for l in mine) / dp
            if grads:
                grads.append({"op": "record", "event": "grads_done"})
                comp.append({"op": "wait", "event": "grads_done"})
            # Adam on this rank's shard: FP32 master weight, m, v and the
            # gradient in, three FP32 tensors and the BF16 weight out.
            comp.append(_k("adam_step", type="elementwise", n=max(1, int(shard)), inputs=4, elem_bytes=4.0,
                           flops_per_elem=12))
            if dp > 1:
                comp.append(_coll("all_gather", shard * dp * 2, dp_group, p.comm_sms, "param_allgather"))
        ranks.append({"streams": [x for x in (comp, grads, fsend, frecv, bsend, brecv) if x]})
    tokens = micro_batch * seqlen * mbs * dp
    return {
        "name": f"{m.name} train pp{pp} {schedule} m{mbs} mb{micro_batch} s{seqlen} {p}",
        "ranks": ranks,
        "model_flops": _model_flops(m, tokens, seqlen // 2, True, lm_head=False),
    }


def async_rl(m: Model, *, rollout_ranks: int, trainer_ranks: int, rollout_batch: int, context: int, decode_steps: int,
             train_micro_batch: int, seqlen: int, layers: Optional[int] = None) -> dict:
    """Asynchronous RL: rollout GPUs decode while trainer GPUs train on the
    previous batch; the trainer then pushes new weights to every rollout GPU
    (point-to-point), which picks them up after its current decode burst.

    Rollout ranks come first. Heterogeneous systems (e.g. H100 samplers + B200
    trainers) are expressed in the *system* YAML; this program is the same.
    """
    if layers is not None:
        m = m.with_layers(layers)
    p_r = Parallel()
    p_t = Parallel(dp=trainer_ranks)
    weight_bytes = 2 * (m.attn_params() + m.ffn_active_params(m.layers - 1)) * m.layers
    roll = decode_step(m, p_r, rollout_batch, context, lm_head=True)["spmd"]["program"]["streams"][0]
    train = train_step(m, p_t, train_micro_batch, seqlen)["spmd"]["program"]["streams"]
    ranks = []
    for r in range(rollout_ranks):
        ops = [{"op": "loop", "times": decode_steps, "body": roll}]
        src = rollout_ranks + (r % trainer_ranks)
        ops.append({"op": "recv", "bytes": int(weight_bytes), "peer": src, "sms": 8})
        ranks.append({"streams": [ops]})
    trainer_group = list(range(rollout_ranks, rollout_ranks + trainer_ranks))
    for i in range(trainer_ranks):
        compute = [_rewrite_groups(op, trainer_group) for op in train[0]]
        for r in range(i, rollout_ranks, trainer_ranks):
            compute.append({"op": "send", "bytes": int(weight_bytes), "peer": r, "sms": 8})
        streams = [compute] + ([[_rewrite_groups(op, trainer_group) for op in train[1]]] if len(train) > 1 else [])
        ranks.append({"streams": streams})
    return {"name": f"async-rl {m.name} {rollout_ranks} samplers + {trainer_ranks} trainers", "ranks": ranks}


def _rewrite_groups(op: dict, group: List[int]) -> dict:
    if op.get("op") == "collective":
        op = dict(op)
        op["group"] = group
    return op
