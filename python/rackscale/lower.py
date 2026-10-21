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
  the batch, so one micro-batch's all-to-all hides behind the other's compute.
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

    def world(self) -> int:
        return max(self.tp * self.dp, self.ep, 1)


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
        per_expert = max(0, recv // local)
        ops += [
            _gemm(f"{tag}router", t, moe.experts, h),
            _k(f"{tag}topk", type="elementwise", n=t * moe.experts, inputs=1, flops_per_elem=4),
        ]
        if ep > 1:
            ops.append(_coll("all_to_all", t * moe.topk * h * moe.dispatch_bytes, f"block:{ep}", p.comm_sms, f"{tag}moe_dispatch"))
        ops += [
            _k(f"{tag}experts_gate_up", type="grouped_gemm", ms=[per_expert] * local, n=2 * moe.expert_ffn, k=h),
            _k(f"{tag}experts_act", type="elementwise", n=max(1, recv * moe.expert_ffn), inputs=2, flops_per_elem=4),
            _k(f"{tag}experts_down", type="grouped_gemm", ms=[per_expert] * local, n=h, k=moe.expert_ffn),
        ]
        if moe.shared_experts:
            sf = moe.shared_experts * moe.expert_ffn
            ops += [_gemm(f"{tag}shared_gate_up", t, 2 * sf, h), _gemm(f"{tag}shared_down", t, h, sf)]
        if ep > 1:
            ops.append(_coll("all_to_all", t * moe.topk * h * 2, f"block:{ep}", p.comm_sms, f"{tag}moe_combine"))
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


def decode_step(m: Model, p: Parallel, batch: int, context: int, *, overlap: bool = False,
                layers: Optional[int] = None, lm_head: bool = True) -> dict:
    """One decode iteration: every rank generates one token for `batch` sequences."""
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
    }


def prefill(m: Model, p: Parallel, batch: int, seqlen: int, *, layers: Optional[int] = None) -> dict:
    if layers is not None:
        m = m.with_layers(layers)
    world = p.world()
    ops = _layers(m, p, batch, seqlen, seqlen, True)
    return {
        "name": f"{m.name} prefill b{batch} s{seqlen} {p}",
        "spmd": {"world": world, "program": {"streams": [ops]}},
        "model_flops": _model_flops(m, batch * seqlen * world // max(1, p.tp), seqlen // 2, False, lm_head=False),
    }


def train_step(m: Model, p: Parallel, micro_batch: int, seqlen: int, *, layers: Optional[int] = None,
               grad_bucket_layers: int = 1) -> dict:
    """Forward + backward of one micro-batch with DP gradient reduce-scatter
    overlapped on a communication stream (FSDP/ZeRO-2 style)."""
    if layers is not None:
        m = m.with_layers(layers)
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
            if op["op"] == "kernel" and op["kernel"]["type"] in ("gemm", "grouped_gemm", "attention"):
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
