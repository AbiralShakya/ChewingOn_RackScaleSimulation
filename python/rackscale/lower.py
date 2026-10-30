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


@dataclass(frozen=True, repr=False)
class Parallel:
    tp: int = 1
    ep: int = 1
    dp: int = 1
    #: SMs given to communication kernels (NCCL channels / DeepEP SMs).
    comm_sms: int = 20
    #: Pipeline stages (training). Ranks are laid out tp-innermost, then dp,
    #: then pp: rank = (stage * dp + d) * tp + t.
    pp: int = 1
    #: Data-parallel sharding of training state (Rajbhandari et al. 2020):
    #: 0 = DDP (gradient all-reduce, everything replicated), 1 = optimizer
    #: state sharded, 2 = optimizer state and gradients sharded (gradient
    #: reduce-scatter; the default), 3 = parameters sharded too (FSDP: every
    #: layer's weights are all-gathered before its forward and again before
    #: its backward).
    zero: int = 2
    #: Hybrid sharding (HSDP): ZeRO-3 inside groups of ``shard`` DP ranks,
    #: plain replication across the ``dp // shard`` groups (gradients are
    #: reduce-scattered in the group, then all-reduced across groups).
    #: 0 = shard over all of ``dp``.
    shard: int = 0
    #: Megatron sequence parallelism: with ``tp > 1``, norms, residuals and
    #: dropout run on ``1/tp`` of the tokens, and each TP all-reduce becomes a
    #: reduce-scatter followed by an all-gather before the next block.
    sp: bool = False

    def world(self) -> int:
        return max(self.tp * self.dp * self.pp, self.ep, 1)

    def __repr__(self) -> str:
        # The fields added after v0.1 are shown only when set, so workload
        # names (and the result caches keyed on them) stay as they were.
        extra = "".join(f", {k}={v}" for k, v, d in (("zero", self.zero, 2), ("shard", self.shard, 0),
                                                      ("sp", self.sp, False)) if v != d)
        return (f"Parallel(tp={self.tp}, ep={self.ep}, dp={self.dp}, comm_sms={self.comm_sms}, "
                f"pp={self.pp}{extra})")

    def shard_group(self) -> int:
        """DP ranks that share one copy of the sharded state."""
        return self.shard if self.shard else self.dp


class ParallelError(ValueError):
    """A parallel layout that the model cannot be split into."""


def check(m: Model, p: Parallel) -> None:
    """Reject layouts that do not divide the model: heads over TP, experts
    over EP, layers over PP. Raises :class:`ParallelError` with the fix."""
    errs = []
    for name, v in (("tp", p.tp), ("ep", p.ep), ("dp", p.dp), ("pp", p.pp)):
        if v < 1:
            errs.append(f"{name}={v} must be >= 1")
    if p.zero not in (0, 1, 2, 3):
        errs.append(f"zero={p.zero} must be 0 (DDP), 1, 2 or 3 (FSDP)")
    if p.shard and (p.shard < 1 or p.dp % p.shard):
        errs.append(f"shard={p.shard} must divide dp={p.dp} (HSDP shard groups)")
    if p.shard and p.shard != p.dp and p.zero != 3:
        errs.append("shard (HSDP) shards parameters inside each group, so it needs zero=3")
    if errs:
        raise ParallelError("; ".join(errs))
    if p.tp > 1:
        if m.heads % p.tp:
            errs.append(f"tp={p.tp} does not divide {m.name}'s {m.heads} attention heads")
        if m.gla is not None and m.gla.latent_heads % p.tp and p.tp % m.gla.latent_heads:
            errs.append(f"tp={p.tp} and {m.gla.latent_heads} GLA latent heads: one must divide the other")
        if m.linear is not None and (m.linear.heads_v % p.tp or m.linear.heads_qk % p.tp):
            errs.append(f"tp={p.tp} does not divide the linear-attention heads")
        if not m.mla and not m.gla and m.kv_heads % p.tp and p.tp % m.kv_heads:
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


def _sp(p: Parallel) -> bool:
    return bool(p.sp) and p.tp > 1


def _rows(t: int, p: Parallel) -> int:
    """Tokens a rank's norms and residuals see: all of them, or ``1/tp``
    under sequence parallelism."""
    return max(1, t // p.tp) if _sp(p) else t


def _tp_enter(t: int, h: int, p: Parallel, tag: str, what: str) -> List[dict]:
    """Sequence parallelism gathers the sequence shards before a block's first GEMM."""
    if not _sp(p):
        return []
    return [_coll("all_gather", t * h * 2, f"block:{p.tp}", p.comm_sms, f"{tag}{what}_allgather")]


def _tp_exit(t: int, h: int, p: Parallel, tag: str, what: str) -> List[dict]:
    """The partial sums of a row-parallel GEMM: one all-reduce, or a
    reduce-scatter onto the sequence shards under sequence parallelism."""
    if p.tp <= 1:
        return []
    if _sp(p):
        return [_coll("reduce_scatter", t * h * 2, f"block:{p.tp}", p.comm_sms, f"{tag}{what}_reducescatter")]
    return [_coll("all_reduce", t * h * 2, f"block:{p.tp}", p.comm_sms, f"{tag}{what}_allreduce")]


def _attention_ops(m: Model, p: Parallel, batch: int, sq: int, skv: int, causal: bool, tag: str) -> List[dict]:
    """Attention block for `batch` sequences of `sq` new tokens each."""
    t = batch * sq
    h = m.hidden
    tp = p.tp
    ops = [_k(f"{tag}input_norm", type="row_norm", rows=_rows(t, p), cols=h)] + _tp_enter(t, h, p, tag, "attn")
    if m.gla:
        a = m.gla
        heads = max(1, m.heads // tp)
        lat = max(1, a.latent_heads // tp)
        ops += [
            _gemm(f"{tag}q_proj", t, heads * (a.nope_dim + a.rope_dim), h),
            _gemm(f"{tag}kv_a_proj", t, lat * a.latent_dim + a.rope_dim, h),
        ]
        if sq == 1:
            # Absorbed decode: each latent is an MQA head for its query heads.
            ops.append(_k(f"{tag}gla_decode", type="attention", batch=batch, heads_q=heads, heads_kv=lat, sq=1,
                          skv=skv, d=a.latent_dim + a.rope_dim, kv_tied=True))
            ops.append(_k(f"{tag}attn_combine", type="elementwise", n=batch * heads * a.latent_dim * 4, inputs=1,
                          elem_bytes=4.0, flops_per_elem=4))
        else:
            ops.append(_gemm(f"{tag}kv_b_proj", t, heads * (a.nope_dim + a.v_head_dim), a.latent_dim))
            ops.append(_k(f"{tag}gla_prefill", type="attention", batch=batch, heads_q=heads, heads_kv=heads, sq=sq,
                          skv=skv, d=a.nope_dim + a.rope_dim, causal=causal))
        ops.append(_gemm(f"{tag}o_proj", t, h, heads * a.v_head_dim))
    elif m.mla:
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
        if m.gta:
            # Tied K/V state per group plus one shared half-width RoPE key.
            proj = (hq + hkv) * m.head_dim + m.head_dim // 2
            attn = _k(f"{tag}attention", type="attention", batch=batch, heads_q=hq, heads_kv=hkv, sq=sq, skv=skv,
                      d=m.head_dim, causal=causal, kv_tied=True)
        else:
            proj = (hq + 2 * hkv) * m.head_dim
            attn = _k(f"{tag}attention", type="attention", batch=batch, heads_q=hq, heads_kv=hkv, sq=sq, skv=skv,
                      d=m.head_dim, causal=causal)
        ops += [_gemm(f"{tag}qkv_proj", t, proj, h), attn]
        if sq == 1:
            # Split-KV decode (FA2/FA3/FA4) ends with a combine kernel over the
            # fp32 partial outputs and log-sum-exps.
            ops.append(_k(f"{tag}attn_combine", type="elementwise", n=batch * hq * m.head_dim * 4, inputs=1, elem_bytes=4.0,
                          flops_per_elem=4))
        ops.append(_gemm(f"{tag}o_proj", t, h, hq * m.head_dim))
    ops += _tp_exit(t, h, p, tag, "attn")
    ops.append(_k(f"{tag}residual", type="elementwise", n=_rows(t, p) * h, inputs=2, flops_per_elem=1))
    return ops


def _ffn_ops(m: Model, p: Parallel, layer: int, t: int, tag: str) -> List[dict]:
    h = m.hidden
    ops = [_k(f"{tag}post_attn_norm", type="row_norm", rows=_rows(t, p), cols=h)]
    moe = m.moe
    if p.ep <= 1:
        ops += _tp_enter(t, h, p, tag, "moe" if moe and layer >= moe.first_dense_layers else "mlp")
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
        else:
            ops += _tp_exit(t, h, p, tag, "moe")
    else:
        ffn = (moe.dense_ffn if moe and moe.dense_ffn else m.ffn) // p.tp
        ops += [
            _gemm(f"{tag}gate_up_proj", t, 2 * ffn, h),
            _k(f"{tag}act_mul", type="elementwise", n=t * ffn, inputs=2, flops_per_elem=4),
            _gemm(f"{tag}down_proj", t, h, ffn),
        ]
        ops += _tp_exit(t, h, p, tag, "mlp")
    ops.append(_k(f"{tag}residual2", type="elementwise", n=_rows(t, p) * h, inputs=2, flops_per_elem=1))
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
    ops = [_k(f"{tag}input_norm", type="row_norm", rows=_rows(t, p), cols=h), *_tp_enter(t, h, p, tag, "mamba"),
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
    ops += _tp_exit(t, h, p, tag, "mamba")
    ops.append(_k(f"{tag}residual", type="elementwise", n=_rows(t, p) * h, inputs=2, flops_per_elem=1))
    return ops


def _linear_ops(m: Model, p: Parallel, batch: int, sq: int, tag: str) -> List[dict]:
    """Linear-attention mixer block (:class:`~rackscale.models.LinearAttn`),
    heads split over TP. The recurrence runs on the Mamba-2 kernels with
    ``d_state = d_k``, ``headdim = d_v`` and one B/C (K/Q) group per QK head:
    ``selective_state_update`` for decode, the SSD chunked scan for prefill.
    Gated DeltaNet adds a per-head ``K K^T`` and a WY-transform pass per chunk.
    """
    la = m.linear
    tp = p.tp
    t = batch * sq
    h = m.hidden
    hv = la.heads_v // tp
    hqk = la.heads_qk // tp
    conv_dim = 2 * hqk * la.d_k + hv * la.d_v
    ops = [_k(f"{tag}input_norm", type="row_norm", rows=_rows(t, p), cols=h), *_tp_enter(t, h, p, tag, "linear"),
           _gemm(f"{tag}in_proj", t, conv_dim + hv * la.d_v + hv * (2 if la.delta else 1), h),
           _k(f"{tag}conv1d", type="conv1d", batch=batch, dim=conv_dim, seqlen=sq, width=la.d_conv)]
    ssm = dict(batch=batch, heads=hv, headdim=la.d_v, d_state=la.d_k, groups=hqk)
    if sq == 1:
        ops.append(_k(f"{tag}state_update", type="ssm_state_update", **ssm))
    else:
        stages = ["chunk_cumsum", "chunk_state", "state_passing", "bmm_chunk", "chunk_scan"]
        if la.delta:
            # Per-head K K^T (the delta rule's beta-weighted Gram matrix) and
            # the WY pass that forms W = A(beta K) and U = A(beta V).
            ops.append(_k(f"{tag}delta_kkt", type="ssd", stage="bmm_chunk", seqlen=sq, chunk=la.chunk,
                          **{**ssm, "groups": hv}))
            ops.append(_k(f"{tag}delta_wy", type="ssd", stage="chunk_scan", seqlen=sq, chunk=la.chunk, **ssm))
        for stage in stages:
            ops.append(_k(f"{tag}lin_{stage}", type="ssd", stage=stage, seqlen=sq, chunk=la.chunk, **ssm))
    ops += [_k(f"{tag}gated_norm", type="row_norm", rows=t, cols=hv * la.d_v),
            _gemm(f"{tag}out_proj", t, h, hv * la.d_v)]
    ops += _tp_exit(t, h, p, tag, "linear")
    ops.append(_k(f"{tag}residual", type="elementwise", n=_rows(t, p) * h, inputs=2, flops_per_elem=1))
    return ops


def _mlp_ops(m: Model, p: Parallel, t: int, tag: str) -> List[dict]:
    """Standalone MLP block (hybrid stacks): pre-norm, gate/up, act, down, residual."""
    h = m.hidden
    ffn = m.ffn // p.tp
    ops = [_k(f"{tag}mlp_norm", type="row_norm", rows=_rows(t, p), cols=h), *_tp_enter(t, h, p, tag, "mlp"),
           _gemm(f"{tag}gate_up_proj", t, 2 * ffn, h),
           _k(f"{tag}act_mul", type="elementwise", n=t * ffn, inputs=2, flops_per_elem=4),
           _gemm(f"{tag}down_proj", t, h, ffn)]
    ops += _tp_exit(t, h, p, tag, "mlp")
    ops.append(_k(f"{tag}mlp_residual", type="elementwise", n=_rows(t, p) * h, inputs=2, flops_per_elem=1))
    return ops


def _layers(m: Model, p: Parallel, batch: int, sq: int, skv: int, causal: bool, tag: str = "") -> List[dict]:
    ops: List[dict] = []
    t = batch * sq
    if m.blocks is not None:
        for b in m.blocks:
            if b == "M":
                ops += _mamba2_ops(m, p, batch, sq, tag)
            elif b == "L":
                ops += _linear_ops(m, p, batch, sq, tag)
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
    if m.linear is not None:
        la = m.linear
        per_head = 4.0 * la.d_k * la.d_v
        if skv > 1:
            per_head += 2.0 * (la.d_k + la.d_v) * (la.chunk / 2)
        ssm += m.block_list().count("L") * la.heads_v * per_head * tokens
    return (3.0 if train else 1.0) * (dense + attn + ssm)


def _weights_per_gpu(m: Model, p: Parallel) -> float:
    """Parameters held by the fullest GPU: TP shards attention, dense FFNs,
    embedding and LM head; EP shards routed experts (attention and shared
    experts are replicated, the DeepSeek inference layout); PP gives each
    stage a contiguous slice of layers, the last stage keeping the LM head."""
    tp, ep, pp = p.tp, p.ep, p.pp
    if m.blocks is not None:
        per = m._block_params()
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
    # MLA's compressed latent is not head-sharded: every TP rank keeps it.
    per_layer = m.kv_elems_per_token_layer(p.tp) * elem_bytes
    return per_layer * m.attn_layers() / p.pp


def memory(m: Model, p: Parallel, *, phase: str, batch: int = 1, context: int = 0, seqlen: int = 0,
           micro_batches: int = 1, schedule: str = "1f1b") -> Dict[str, float]:
    """Bytes the fullest GPU needs, by category, for the whole model.

    First-order and stated so it can be checked: weights in the model dtype;
    KV cache for ``batch`` sequences of ``context`` tokens (``batch`` is per
    rank, as in :func:`decode_step`); Mamba-2 conv and SSM state; for
    training, BF16 gradients and FP32 Adam state (master weights, m, v),
    sharded over the DP shard group as ``p.zero`` says (ZeRO-1 shards the
    optimizer state, ZeRO-2 the gradients too, ZeRO-3 the weights as well),
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
        if m.linear is not None:
            la = m.linear
            n_l = m.block_list().count("L")
            state = (la.state_elems() * 4 + la.conv_dim * (la.d_conv - 1) * 2) / p.tp
            out["ssm_state"] = out.get("ssm_state", 0.0) + batch * n_l * state / p.pp
        if phase == "prefill":
            t = batch * seqlen
            ffn = max(m.ffn, m.moe.expert_ffn * m.moe.topk if m.moe else 0)
            out["activations"] = t * (4 * m.hidden + 2 * ffn / p.tp) * 2.0
    elif phase == "train":
        s = p.shard_group()
        if p.zero == 3:
            # The shard, plus the two layers FSDP holds gathered at once
            # (the running one and the prefetched one).
            n_l = len(m.block_list()) if m.blocks is not None else m.layers
            out["weights"] = w * m.bytes_per_param() * (1.0 / s + 2.0 / n_l)
        out["grads"] = 2.0 * w / (s if p.zero >= 2 else 1)
        out["optimizer"] = 12.0 * w / (s if p.zero >= 1 else 1)
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


def prefill_cp(m: Model, cp: int, seqlen: int, *, layers: Optional[int] = None, comm_sms: int = 2,
               kind: str = "ring") -> dict:
    """Context-parallel prefill of one ``seqlen``-token sequence over ``cp``
    ranks, ``seqlen / cp`` tokens each (no TP; one model replica per rank).

    * Attention blocks run **ring attention**: each rank computes its causal
      diagonal block, then ``cp - 1`` steps of its queries against a K/V chunk
      received from its left neighbour, forwarding each chunk to the right
      while the next step computes. Chunks are zig-zag assigned, so every
      off-diagonal step does half a chunk's work (``skv = T / 2``).
    * Mamba-2 and linear-attention blocks run the local chunked scan from a
      zero state, then pass the ``heads x d_v x d_k`` FP32 state along the
      ranks: rank ``r`` waits for rank ``r - 1``'s final state, folds in its
      own (``S_out = decay * S_in + S_local``; a ``d_k x d_k`` matmul per head
      for the delta rule), sends it on, and adds ``C_t S_in`` to its outputs.
      Successive layers pipeline across ranks. The ``d_conv - 1``-token conv
      halo is not exchanged (a few KB).

    With ``kind="ulysses"`` (DeepSpeed-Ulysses, Jacobs et al. 2023) attention
    blocks instead all-to-all Q, K and V from sequence shards to head shards,
    run causal attention over the whole sequence on ``heads / cp`` heads, and
    all-to-all the output back. KV heads fewer than ``cp`` are replicated.
    """
    if kind == "ulysses":
        return _prefill_ulysses(m, cp, seqlen, layers=layers)
    if kind != "ring":
        raise ValueError(f"unknown context-parallel kind {kind!r} (ring, ulysses)")
    if seqlen % cp:
        raise ParallelError(f"seqlen={seqlen} is not a multiple of cp={cp}")
    if m.moe or m.mla or m.gla:
        raise ValueError("prefill_cp lowers dense GQA/GTA attention, Mamba-2 and linear-attention stacks")
    if layers is not None:
        m = m.with_layers(layers)
    p = Parallel()
    t = seqlen // cp
    blocks = m.block_list()
    ranks = []
    for r in range(cp):
        comp: List[dict] = []
        tx: List[dict] = []
        rx: List[dict] = []
        for li, b in enumerate(blocks):
            tag = f"l{li}."
            if b == "F":
                comp += _mlp_ops(m, p, t, tag)
            elif b == "A":
                hq, hkv, d = m.heads, m.kv_heads, m.head_dim
                proj = (hq + hkv) * d + d // 2 if m.gta else (hq + 2 * hkv) * d
                kv = t * hkv * d * 2 * (1 if m.gta else 2)
                ev = f"{tag}kv"
                comp += [_k(f"{tag}input_norm", type="row_norm", rows=t, cols=m.hidden),
                         _gemm(f"{tag}qkv_proj", t, proj, m.hidden),
                         {"op": "record", "event": f"{ev}_mine"},
                         _k(f"{tag}ring_diag", type="attention", batch=1, heads_q=hq, heads_kv=hkv, sq=t, skv=t,
                            d=d, causal=True, **({"kv_tied": True} if m.gta else {}))]
                tx += [{"op": "wait", "event": f"{ev}_mine"}]
                for step in range(1, cp):
                    tx.append({"op": "send", "bytes": kv, "peer": (r + 1) % cp, "sms": comm_sms})
                    rx.append({"op": "recv", "bytes": kv, "peer": (r - 1) % cp, "sms": comm_sms})
                    rx.append({"op": "record", "event": f"{ev}_rx"})
                    if step < cp - 1:
                        tx.append({"op": "wait", "event": f"{ev}_rx"})
                    comp += [{"op": "wait", "event": f"{ev}_rx"},
                             _k(f"{tag}ring_step", type="attention", batch=1, heads_q=hq, heads_kv=hkv, sq=t,
                                skv=t // 2, d=d, causal=False, **({"kv_tied": True} if m.gta else {})),
                             _k(f"{tag}ring_merge", type="elementwise", n=t * hq * d, inputs=2, elem_bytes=4.0,
                                flops_per_elem=4)]
                comp += [_gemm(f"{tag}o_proj", t, m.hidden, hq * d),
                         _k(f"{tag}residual", type="elementwise", n=t * m.hidden, inputs=2, flops_per_elem=1)]
            elif b in "ML":
                ops = _mamba2_ops(m, p, 1, t, tag) if b == "M" else _linear_ops(m, p, 1, t, tag)
                cut = next(i for i, o in enumerate(ops) if o.get("name", "").endswith("gated_norm"))
                comp += ops[:cut]
                if b == "M":
                    mb = m.mamba
                    heads, dv, dk, delta = mb.nheads, mb.headdim, mb.d_state, False
                else:
                    la = m.linear
                    heads, dv, dk, delta = la.heads_v, la.d_v, la.d_k, la.delta
                state = heads * dv * dk * 4
                ev = f"{tag}state"
                if r > 0:
                    rx += [{"op": "recv", "bytes": state, "peer": r - 1, "sms": comm_sms},
                           {"op": "record", "event": f"{ev}_in"}]
                    comp.append({"op": "wait", "event": f"{ev}_in"})
                    if delta:
                        comp.append(_gemm(f"{tag}state_fold", heads * dk, dv, dk))
                    else:
                        comp.append(_k(f"{tag}state_fold", type="elementwise", n=heads * dv * dk, inputs=2,
                                       elem_bytes=4.0, flops_per_elem=2))
                if r < cp - 1:
                    comp.append({"op": "record", "event": f"{ev}_out"})
                    tx += [{"op": "wait", "event": f"{ev}_out"},
                           {"op": "send", "bytes": state, "peer": r + 1, "sms": comm_sms}]
                if r > 0:
                    # y_t += C_t (decay_t * S_in): T x d_k by d_k x d_v per head.
                    comp.append(_gemm(f"{tag}state_correct", t * heads, dv, dk))
                comp += ops[cut:]
            else:
                raise ValueError(f"unknown block {b!r} in {m.name}")
        ranks.append({"streams": [comp, tx, rx]})
    return {
        "name": f"{m.name} prefill s{seqlen} cp{cp}",
        "ranks": ranks,
        "model_flops": _model_flops(m, seqlen, seqlen // 2, False, lm_head=False),
    }


def _prefill_ulysses(m: Model, cp: int, seqlen: int, *, layers: Optional[int] = None) -> dict:
    if seqlen % cp:
        raise ParallelError(f"seqlen={seqlen} is not a multiple of cp={cp}")
    if m.moe or m.mla or m.gla or m.gta or "M" in m.block_list() or "L" in m.block_list():
        raise ValueError("Ulysses prefill lowers dense GQA attention + MLP stacks")
    if m.heads % cp:
        raise ParallelError(f"Ulysses needs cp={cp} to divide {m.name}'s {m.heads} query heads")
    if m.kv_heads % cp and cp % m.kv_heads:
        raise ParallelError(f"cp={cp} and {m.kv_heads} KV heads: one must divide the other")
    if layers is not None:
        m = m.with_layers(layers)
    p = Parallel()
    t = seqlen // cp
    hq, hkv, d = m.heads, m.kv_heads, m.head_dim
    hq_l, hkv_l = hq // cp, max(1, hkv // cp)
    ops: List[dict] = []
    for li, b in enumerate(m.block_list()):
        tag = f"l{li}."
        if b == "F":
            ops += _mlp_ops(m, p, t, tag)
            continue
        # Each rank sends (cp - 1)/cp of its t x (H + 2 H_kv) x d shard; the
        # ``bytes`` of an all-to-all are the rank's whole buffer.
        ops += [_k(f"{tag}input_norm", type="row_norm", rows=t, cols=m.hidden),
                _gemm(f"{tag}qkv_proj", t, (hq + 2 * hkv) * d, m.hidden),
                _coll("all_to_all", t * (hq + 2 * hkv_l * cp) * d * 2, "world", 16, f"{tag}ulysses_qkv_a2a"),
                _k(f"{tag}attention", type="attention", batch=1, heads_q=hq_l, heads_kv=hkv_l, sq=seqlen, skv=seqlen,
                   d=d, causal=True),
                _coll("all_to_all", t * hq * d * 2, "world", 16, f"{tag}ulysses_out_a2a"),
                _gemm(f"{tag}o_proj", t, m.hidden, hq * d),
                _k(f"{tag}residual", type="elementwise", n=t * m.hidden, inputs=2, flops_per_elem=1)]
    return {
        "name": f"{m.name} prefill s{seqlen} cp{cp} ulysses",
        "spmd": {"world": cp, "program": {"streams": [ops]}},
        "model_flops": _model_flops(m, seqlen, seqlen // 2, False, lm_head=False),
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
    if m.mla or m.gla or m.blocks is not None:
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
        if b == "L":
            return _linear_ops(m, p, batch, seqlen, tag)
        if b == "A":
            return _attention_ops(m, p, batch, seqlen, seqlen, True, tag)
        return _mlp_ops(m, p, t, tag)
    return _attention_ops(m, p, batch, seqlen, seqlen, True, tag) + _ffn_ops(m, p, idx, t, tag)


def _layer_params(m: Model, p: Parallel, idx: int) -> float:
    """Parameters of layer ``idx`` on one GPU (all experts of an MoE layer)."""
    if m.blocks is not None:
        return m._block_params()[m.blocks[idx]] / max(1, p.tp)
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
    if p.zero == 3 and (p.pp > 1 or micro_batches > 1):
        raise ParallelError("zero=3 (FSDP) is lowered for one micro-batch without pipeline stages; "
                            "use zero<=2 with pp, or one micro-batch")
    if p.zero in (0, 3):
        wl = _train_sharded(m, p, micro_batch, seqlen, optimizer)
        wl["memory"] = mem
        return wl
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


def _dp_groups(p: Parallel, rank: int) -> tuple:
    """(shard group, replica group) of ``rank`` for data-parallel ranks laid
    out as rank = d * tp + t (no pipeline stages). The shard group holds
    ``shard_group()`` consecutive DP ranks; the replica group links the ranks
    with the same position in every shard group."""
    tp, s = p.tp, p.shard_group()
    d, t = rank // tp, rank % tp
    k = d // s
    shard = [(k * s + j) * tp + t for j in range(s)]
    repl = [(kk * s + d % s) * tp + t for kk in range(p.dp // s)]
    return shard, repl


def _train_sharded(m: Model, p: Parallel, micro_batch: int, seqlen: int, optimizer: bool) -> dict:
    """One training step under DDP (``zero=0``) or FSDP / ZeRO-3 and HSDP
    (``zero=3``), one micro-batch, no pipeline stages.

    * DDP: each layer's gradients are all-reduced over DP on a communication
      stream as soon as its backward finishes.
    * FSDP: a gather stream all-gathers layer ``l``'s BF16 weights before its
      forward (issued when layer ``l - 1``'s forward starts, FSDP's one-layer
      prefetch) and again before its backward; a reduce stream
      reduce-scatters each layer's gradients when they are ready. Under HSDP
      the gather and scatter run inside the shard group and the gradient
      shard is then all-reduced across the replica groups.
    """
    tp, dp = p.tp, p.dp
    world = tp * dp
    n_layers = len(m.blocks) if m.blocks is not None else m.layers
    s = p.shard_group()
    fwd = [_layer_ops(m, p, l, micro_batch, seqlen, f"L{l}.") for l in range(n_layers)]
    bwd = [sum(_backward(_layer_ops(m, p, l, micro_batch, seqlen, f"L{l}.bwd.")), []) for l in range(n_layers)]
    lp = [_layer_params(m, p, l) for l in range(n_layers)]

    def program(rank: int) -> List[List[dict]]:
        if dp > 1:
            shard, repl = _dp_groups(p, rank)
        comp: List[dict] = []
        gather: List[dict] = []
        reduce: List[dict] = []
        fsdp = p.zero == 3 and dp > 1
        for l in range(n_layers):
            if fsdp:
                if l:
                    gather.append({"op": "wait", "event": f"fstart{l - 1}"})
                gather += [_coll("all_gather", lp[l] * 2, shard, p.comm_sms, f"param_ag_fwd{l}"),
                           {"op": "record", "event": f"agf{l}"}]
                comp += [{"op": "wait", "event": f"agf{l}"}, {"op": "record", "event": f"fstart{l}"}]
            comp += fwd[l]
        for l in reversed(range(n_layers)):
            if fsdp:
                if l < n_layers - 1:
                    gather.append({"op": "wait", "event": f"bstart{l + 1}"})
                gather += [_coll("all_gather", lp[l] * 2, shard, p.comm_sms, f"param_ag_bwd{l}"),
                           {"op": "record", "event": f"agb{l}"}]
                comp += [{"op": "wait", "event": f"agb{l}"}, {"op": "record", "event": f"bstart{l}"}]
            comp += bwd[l]
            if dp > 1:
                comp.append({"op": "record", "event": f"grad{l}"})
                reduce.append({"op": "wait", "event": f"grad{l}"})
                if fsdp:
                    reduce.append(_coll("reduce_scatter", lp[l] * 2, shard, p.comm_sms, f"grad_rs{l}"))
                    if len(repl) > 1:
                        reduce.append(_coll("all_reduce", lp[l] * 2 / s, repl, p.comm_sms, f"grad_ar{l}"))
                else:
                    reduce.append(_coll("all_reduce", lp[l] * 2, shard, p.comm_sms, f"grad_ar{l}"))
        if optimizer:
            if reduce:
                reduce.append({"op": "record", "event": "grads_done"})
                comp.append({"op": "wait", "event": "grads_done"})
            n = sum(lp) / (s if p.zero == 3 else 1)
            comp.append(_k("adam_step", type="elementwise", n=max(1, int(n)), inputs=4, elem_bytes=4.0,
                           flops_per_elem=12))
        return [x for x in (comp, gather, reduce) if x]

    name = f"{m.name} train mb{micro_batch} s{seqlen} {p}"
    flops = _model_flops(m, micro_batch * seqlen * dp, seqlen // 2, True, lm_head=False)
    if p.shard and p.shard != dp and tp > 1:
        # HSDP shard groups are not expressible as block/stride groups when
        # TP is innermost: give every rank its own program.
        return {"name": name, "ranks": [{"streams": program(r)} for r in range(world)], "model_flops": flops}
    streams = program(0)
    if dp > 1:
        named = {tuple(_dp_groups(p, 0)[0]): f"block:{s}" if tp == 1 else f"stride:{tp}",
                 tuple(_dp_groups(p, 0)[1]): f"stride:{s * tp}"}
        streams = [[_rewrite_groups(op, named[tuple(op["group"])]) if op.get("op") == "collective" else op
                    for op in st] for st in streams]
    return {"name": name, "spmd": {"world": world, "program": {"streams": streams}}, "model_flops": flops}


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
                            kind = "all_reduce" if p.zero == 0 else "reduce_scatter"
                            grads += [{"op": "wait", "event": ev},
                                      _coll(kind, bucket * 2, dp_group, p.comm_sms, f"grad_rs{l}")]
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
            shard = sum(_layer_params(m, p, l) for l in mine) / (dp if p.zero else 1)
            if grads:
                grads.append({"op": "record", "event": "grads_done"})
                comp.append({"op": "wait", "event": "grads_done"})
            # Adam on this rank's shard: FP32 master weight, m, v and the
            # gradient in, three FP32 tensors and the BF16 weight out.
            comp.append(_k("adam_step", type="elementwise", n=max(1, int(shard)), inputs=4, elem_bytes=4.0,
                           flops_per_elem=12))
            if dp > 1 and p.zero:
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
