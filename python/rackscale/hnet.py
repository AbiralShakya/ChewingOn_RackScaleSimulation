"""Byte-level and hierarchical models: H-Net and the baselines it is compared with.

H-Net (Hwang, Wang and Gu, "Dynamic Chunking for End-to-End Hierarchical
Sequence Modeling", 2025; Cartesia's "From tokens to bytes") reads raw UTF-8
bytes. Each stage of the hierarchy is

    encoder (Mamba-2 layers at width D_s, on every position of the stage)
    -> router: p_t = (1 - cos(W_q x_t, W_k x_{t-1})) / 2, boundary where p_t >= 0.5
    -> downsample: keep the boundary positions (about 1 in R_s)
    -> [the next stage, or the main network: a Transformer at width D_S]
    -> dechunk: EMA smoothing z_t = p_t z^_t + (1 - p_t) z_{t-1}, repeat each chunk over its bytes
    -> residual: + Linear(x_t)
    -> decoder (Mamba-2 layers at width D_s)

so the main network sees ``bytes / prod(R_s)`` positions. Widths only grow
inward; the paper pads ``D_s`` to ``D_{s+1}`` with learned vectors, which we
lower as a projection GEMM (an upper bound on its cost).

At decode time the encoders and decoders step on every byte; the main
network steps only when the router fires, which for a batch is a random
subset of the sequences each step. :func:`decode_steps` lowers a run of
byte steps with the main network's batch drawn per step, so the simulator
sees the uneven steps a real server would.

Published shapes (paper Table 1): 1-stage XL ``M4 + T24 + M4`` at widths
1024 / 2048, 4.7 bytes per chunk; 2-stage XL ``M4 + T1M4 + T27 + M4T1 + M4``
at 1024 / 1536 / 2048, 6.9 bytes per main-network position. Head counts and
FFN widths are not in the paper text; we use 128-wide heads and an 8/3
SwiGLU FFN (stated, not sourced). Cartesia (part 1) puts the outer stages at
about 43% of H-Net's FLOPs per byte; with these shapes they are 19% (1-stage)
and 39% (2-stage).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from . import lower
from .lower import Parallel, _gemm, _k
from .models import Mamba2, Model

BYTES_PER_BPE_TOKEN = 4.45  # Llama-3 tokenizer on English web text (Cartesia part 1)


@dataclass(frozen=True)
class Stage:
    """One outer stage: ``enc``/``dec`` block strings (``M`` = Mamba-2,
    ``A`` = attention + ``F`` = MLP pairs as in :class:`~rackscale.models.Model`),
    its width, and the compression its router targets (positions in per
    position out)."""

    width: int
    enc: str = "MMMM"
    dec: str = "MMMM"
    ratio: float = 4.7
    #: Sliding window of the attention blocks in this stage (0 = global).
    window: int = 0


@dataclass(frozen=True)
class HNet:
    name: str
    stages: Tuple[Stage, ...]
    main: Model
    vocab: int = 256

    @property
    def compression(self) -> float:
        out = 1.0
        for s in self.stages:
            out *= s.ratio
        return out

    def stage_model(self, i: int, blocks: str) -> Model:
        """A :class:`Model` that lowers stage ``i``'s ``blocks`` at its width."""
        s = self.stages[i]
        d = s.width
        heads = max(1, d // 128)
        return Model(f"{self.name}.s{i}", hidden=d, layers=len(blocks), heads=heads, kv_heads=heads, head_dim=128,
                     ffn=_ffn(d), vocab=self.vocab, blocks=blocks,
                     mamba=Mamba2(d_inner=2 * d, headdim=64, d_state=128, ngroups=1, d_conv=4, chunk=256))

    def params(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, s in enumerate(self.stages):
            m = self.stage_model(i, s.enc + s.dec)
            nxt = self.stages[i + 1].width if i + 1 < len(self.stages) else self.main.hidden
            # router W_q, W_k; residual Linear; width projections in and out
            extra = 2 * s.width * s.width + s.width * s.width + 2 * s.width * nxt
            out[f"stage{i}"] = m.active_params(lm_head=False) + extra
        out["main"] = self.main.active_params(lm_head=False)
        out["embed_head"] = 2 * self.vocab * self.stages[0].width
        return out

    def total_params(self) -> float:
        return sum(self.params().values())


def _ffn(d: int) -> int:
    return int(round(8 * d / 3 / 256)) * 256


def _transformer(name: str, d: int, layers: int, vocab: int) -> Model:
    h = d // 128
    return Model(name, hidden=d, layers=layers, heads=h, kv_heads=h, head_dim=128, ffn=_ffn(d), vocab=vocab)


HNET_1STAGE_XL = HNet("hnet-1stage-xl", (Stage(1024, ratio=4.7),), _transformer("hnet-1stage-xl.main", 2048, 24, 256))
HNET_2STAGE_XL = HNet("hnet-2stage-xl", (Stage(1024, ratio=3.0), Stage(1536, enc="AFMMMM", dec="MMMMAF", ratio=2.3)),
                      _transformer("hnet-2stage-xl.main", 2048, 27, 256))
#: The token-level baseline at the same main-network shape (the paper's
#: isotropic Transformer), reading BPE tokens of ~4.45 bytes.
BPE_XL = _transformer("bpe-transformer-xl", 2048, 24, 128256)
#: The same Transformer reading raw bytes with no chunking.
BYTE_XL = _transformer("byte-transformer-xl", 2048, 24, 256)

ALL: Dict[str, HNet] = {m.name: m for m in (HNET_1STAGE_XL, HNET_2STAGE_XL)}
BASELINES: Dict[str, Model] = {m.name: m for m in (BPE_XL, BYTE_XL)}


# ---------------------------------------------------------------------------
# lowering


def _blocks_ops(m: Model, p: Parallel, batch: int, sq: int, skv: int, causal: bool, tag: str,
                window: int = 0) -> List[dict]:
    if window and sq > 1:
        skv = min(skv, window)
    return lower._layers(m, p, batch, sq, skv, causal, tag)


def _router_ops(d: int, t: int, tag: str) -> List[dict]:
    """q/k projections, cosine similarity + threshold, and the gather that
    keeps the boundary positions."""
    return [_gemm(f"{tag}router_qk", t, 2 * d, d),
            _k(f"{tag}router_cos", type="elementwise", n=t * d, inputs=2, flops_per_elem=4),
            _k(f"{tag}downsample", type="elementwise", n=t * d, inputs=1, flops_per_elem=1)]


def _dechunk_ops(d: int, t_in: int, t_out: int, tag: str) -> List[dict]:
    """EMA smoothing over the chunks, repeat to bytes, residual Linear."""
    return [_k(f"{tag}dechunk_ema", type="elementwise", n=t_in * d, inputs=2, flops_per_elem=3),
            _k(f"{tag}upsample", type="elementwise", n=t_out * d, inputs=1, flops_per_elem=1),
            _gemm(f"{tag}residual_proj", t_out, d, d),
            _k(f"{tag}residual_add", type="elementwise", n=t_out * d, inputs=2, flops_per_elem=1)]


def forward_ops(h: HNet, p: Parallel, batch: int, nbytes: int, *, causal: bool = True) -> List[dict]:
    """Prefill / training forward over ``batch`` sequences of ``nbytes`` bytes."""
    ops: List[dict] = [_k("embed", type="elementwise", n=batch * nbytes * h.stages[0].width, inputs=1)]
    lens = [nbytes]
    for s in h.stages:
        lens.append(max(1, int(round(lens[-1] / s.ratio))))
    for i, s in enumerate(h.stages):
        m = h.stage_model(i, s.enc)
        ops += _blocks_ops(m, p, batch, lens[i], lens[i], causal, f"s{i}.enc.", s.window)
        ops += _router_ops(s.width, batch * lens[i], f"s{i}.")
        nxt = h.stages[i + 1].width if i + 1 < len(h.stages) else h.main.hidden
        if nxt != s.width:
            ops.append(_gemm(f"s{i}.widen", batch * lens[i + 1], nxt, s.width))
    ops += lower._layers(h.main, p, batch, lens[-1], lens[-1], causal, "main.")
    for i in reversed(range(len(h.stages))):
        s = h.stages[i]
        nxt = h.stages[i + 1].width if i + 1 < len(h.stages) else h.main.hidden
        if nxt != s.width:
            ops.append(_gemm(f"s{i}.narrow", batch * lens[i + 1], s.width, nxt))
        ops += _dechunk_ops(s.width, batch * lens[i + 1], batch * lens[i], f"s{i}.")
        ops += _blocks_ops(h.stage_model(i, s.dec), p, batch, lens[i], lens[i], causal, f"s{i}.dec.", s.window)
    return ops


def flops_per_byte(h: HNet, nbytes: int) -> float:
    """Model FLOPs per byte of one forward pass (matmuls, attention, SSD)."""
    total = 0.0
    lens = [float(nbytes)]
    for s in h.stages:
        lens.append(lens[-1] / s.ratio)
    for i, s in enumerate(h.stages):
        m = h.stage_model(i, s.enc + s.dec)
        skv = min(lens[i], s.window) if s.window else lens[i]
        total += lower._model_flops(m, int(lens[i]), int(skv // 2), False, lm_head=False)
        nxt = h.stages[i + 1].width if i + 1 < len(h.stages) else h.main.hidden
        total += 2 * lens[i] * (3 * s.width * s.width) + 2 * 2 * lens[i + 1] * s.width * nxt
    total += lower._model_flops(h.main, int(lens[-1]), int(lens[-1] // 2), False, lm_head=False)
    total += 2 * nbytes * h.vocab * h.stages[0].width
    return total / nbytes


def prefill(h: HNet, p: Parallel, batch: int, nbytes: int, *, layers: Optional[int] = None) -> dict:
    """Prefill of ``batch`` prompts of ``nbytes`` bytes. ``layers`` cuts the
    main network (the outer stages are always whole)."""
    if layers is not None:
        h = replace(h, main=h.main.with_layers(layers))
    ops = forward_ops(h, p, batch, nbytes)
    return {"name": f"{h.name} prefill b{batch} {nbytes}B {p}",
            "spmd": {"world": p.world(), "program": {"streams": [ops]}},
            "model_flops": flops_per_byte(h, nbytes) * batch * nbytes * p.world() / max(1, p.tp)}


def _step_ops(h: HNet, p: Parallel, batch: int, context: int, fired: Sequence[int], tag: str) -> List[dict]:
    """One byte of decode for ``batch`` sequences; ``fired[i]`` sequences
    reach stage ``i + 1`` (the last entry is the main network's batch)."""
    ops: List[dict] = []
    rows = [batch] + list(fired)
    ctx = [context]
    for s in h.stages:
        ctx.append(max(1, int(ctx[-1] / s.ratio)))
    for i, s in enumerate(h.stages):
        ops += _blocks_ops(h.stage_model(i, s.enc), p, rows[i], 1, min(ctx[i], s.window or ctx[i]), False,
                           f"{tag}s{i}.enc.")
        ops += _router_ops(s.width, rows[i], f"{tag}s{i}.")
        if rows[i + 1] == 0:
            break
    if rows[-1]:
        ops += lower._layers(h.main, p, rows[-1], 1, ctx[-1], False, f"{tag}main.")
    for i in reversed(range(len(h.stages))):
        s = h.stages[i]
        if rows[i] == 0:
            continue
        ops += _dechunk_ops(s.width, rows[i], rows[i], f"{tag}s{i}.")
        ops += _blocks_ops(h.stage_model(i, s.dec), p, rows[i], 1, min(ctx[i], s.window or ctx[i]), False,
                           f"{tag}s{i}.dec.")
    ops.append(_gemm(f"{tag}lm_head", batch, h.vocab, h.stages[0].width))
    return ops


def fired_counts(h: HNet, batch: int, steps: int, seed: int = 0) -> List[List[int]]:
    """Per step, how many of ``batch`` sequences reach each inner stage when
    each stage's router fires independently with probability ``1 / ratio``."""
    rng = random.Random(seed)
    out = []
    for _ in range(steps):
        alive = batch
        row = []
        for s in h.stages:
            alive = sum(rng.random() < 1.0 / s.ratio for _ in range(alive))
            row.append(alive)
        out.append(row)
    return out


def _nested(rows: List[List[int]]) -> List[List[int]]:
    # A sequence reaches stage i + 1 only through stage i.
    for row in rows:
        for j in range(1, len(row)):
            row[j] = min(row[j], row[j - 1])
    return rows


def _spread(totals: Sequence[float], steps: int) -> List[List[int]]:
    """Whole sequences per step whose running sums track ``totals / steps``,
    so the average over the steps is exact."""
    return _nested([[int((i + 1) * t / steps) - int(i * t / steps) for t in totals] for i in range(steps)])


def step_counts(h: HNet, batch: int, steps: int, *, seed: int = 0, mean_field: bool = False) -> List[List[int]]:
    """Per decode step, how many of ``batch`` sequences run each stage's inner
    network: drawn from the routers (:func:`fired_counts`), or with
    ``mean_field`` the expected batch, the assumption an analytic model
    makes (whole sequences per step, averaging exactly to the expectation)."""
    if not mean_field:
        return fired_counts(h, batch, steps, seed)
    exp, alive = [], float(batch)
    for s in h.stages:
        alive /= s.ratio
        exp.append(alive * steps)
    return _spread(exp, steps)


def even_counts(counts: List[List[int]]) -> List[List[int]]:
    """The same inner-network runs per stage as ``counts``, spread evenly
    over the steps: a router draw's mean without its step-to-step
    variability."""
    return _spread([sum(c[j] for c in counts) for j in range(len(counts[0]))], len(counts))


def decode_steps(h: HNet, p: Parallel, batch: int, context: int, steps: int = 16, *, seed: int = 0,
                 layers: Optional[int] = None, mean_field: bool = False,
                 counts: Optional[List[List[int]]] = None) -> dict:
    """``steps`` byte-decode steps of ``batch`` sequences at ``context``
    bytes, each step's inner batches from ``counts`` (default
    :func:`step_counts`)."""
    if layers is not None:
        h = replace(h, main=h.main.with_layers(layers))
    counts = counts if counts is not None else step_counts(h, batch, steps, seed=seed, mean_field=mean_field)
    ops: List[dict] = []
    for i, c in enumerate(counts):
        ops += _step_ops(h, p, batch, context, c, f"t{i}.")
    return {"name": f"{h.name} decode b{batch} ctx{context}B x{len(counts)}{' mean' if mean_field else ''} {p}",
            "spmd": {"world": p.world(), "program": {"streams": [ops]}}}


def decode_baseline(m: Model, p: Parallel, batch: int, context: int, *, layers: Optional[int] = None) -> dict:
    """One token-level decode step of a baseline model (BPE or bytes)."""
    return lower.decode_step(m, p, batch, context, layers=layers)
