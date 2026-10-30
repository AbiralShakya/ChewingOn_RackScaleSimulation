"""Recommendation models: DLRM-style embedding tables, all-to-all and small MLPs.

Chapter 18 of the book. A DLRM step does almost no arithmetic. Each sample
looks up ``pooling`` rows in each of ``tables`` embedding tables and sums
them (an embedding bag), a bottom MLP embeds the dense features, a pairwise
dot-product interaction mixes the ``tables + 1`` vectors, and a top MLP
scores the result. The tables hold most of the parameters (terabytes), so
they are **model parallel**: each rank owns ``tables / world`` whole tables
(TorchRec's table-wise sharding) and looks them up for the *global* batch,
then an all-to-all sends every sample's pooled rows to the rank that owns
the sample, where the MLPs run data parallel. Training sends the gradients
back the same way and updates the gathered rows in place (the optimizer is
fused into the embedding backward, as in FBGEMM).

Two closed forms set the step, per rank and per step:

    HBM bytes of the lookup   = b · T · L · D · e          (b samples per rank)
    bytes sent by all-to-all  = b · T · D · e_c · (N - G)/N  (G ranks per fast domain)

so the gather moves ``L`` times more bytes through HBM than the all-to-all
moves over the network, and which one is slower is decided by
``L · BW_net / BW_HBM``. :func:`balance` evaluates both.

Not modelled: row-wise or column-wise sharding, caches of hot rows (pass a
smaller ``hot_rows`` to stand for one), index sorting and deduplication,
the input pipeline, and quantised tables beyond their element size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .lower import ParallelError, _coll, _gemm, _k


@dataclass(frozen=True)
class DLRM:
    name: str
    tables: int
    rows: int
    dim: int
    pooling: int
    dense_in: int
    bottom: Tuple[int, ...]
    top: Tuple[int, ...]
    #: Bytes per table element (fp16 rows) and per element sent by the all-to-all.
    table_bytes: float = 2.0
    comm_bytes: float = 2.0

    def embedding_bytes(self) -> float:
        return self.tables * self.rows * self.dim * self.table_bytes

    def interaction_out(self) -> int:
        """Width of the top MLP's input: every pair of the T + 1 vectors, plus the dense vector."""
        f = self.tables + 1
        return f * (f - 1) // 2 + self.dim

    def _mlp(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """(bottom, top) layers as (in, out) widths."""
        bot = list(zip((self.dense_in,) + self.bottom[:-1], self.bottom))
        top = list(zip((self.interaction_out(),) + self.top[:-1], self.top))
        return bot, top

    def mlp_params(self) -> int:
        bot, top = self._mlp()
        return sum(i * o for i, o in bot + top)

    def dense_flops(self, samples: int) -> float:
        """Forward FLOPs of the MLPs and the interaction for `samples` samples."""
        f = self.tables + 1
        return 2.0 * samples * (self.mlp_params() + f * f * self.dim)


#: An illustrative large multi-hot model: 256 tables of 40M rows x 128 fp16
#: (2.6 TB of embeddings, more than eight B300s hold), 32 lookups per table
#: per sample, MLPs in the DLRM style.
DLRM_2TB = DLRM("dlrm-2.6tb", tables=256, rows=40_000_000, dim=128, pooling=32, dense_in=512,
                bottom=(512, 256, 128), top=(1024, 1024, 512, 256, 1))

MODELS = {m.name: m for m in [DLRM_2TB]}


def balance(m: DLRM, world: int, batch: int, hbm_gbps: float, net_gbps: float, domain: int = 1) -> Dict[str, float]:
    """Closed forms of one forward per rank: the bytes and time of the
    embedding gather at `hbm_gbps`, and of the part of the all-to-all that
    leaves the rank's `domain` (the ranks sharing a faster fabric: 8 for an
    HGX node behind its NICs, 1 inside an NVL72) at `net_gbps` per rank."""
    lookup = batch * m.tables * m.pooling * m.dim * m.table_bytes
    a2a = batch * m.tables * m.dim * m.comm_bytes * (world - domain) / world
    t_l, t_a = lookup / (hbm_gbps * 1e9), a2a / (net_gbps * 1e9)
    return {"lookup_bytes": lookup, "a2a_bytes": a2a, "lookup_us": t_l * 1e6, "a2a_us": t_a * 1e6,
            "a2a_over_lookup": t_a / t_l if t_l else float("inf"),
            "embedding_gb_per_rank": m.embedding_bytes() / world / 1e9}


def _mlp_ops(layers: List[Tuple[int, int]], b: int, tag: str) -> List[dict]:
    ops = []
    for i, (k, n) in enumerate(layers):
        ops += [_gemm(f"{tag}{i}", b, n, k),
                _k(f"{tag}{i}_relu", type="elementwise", n=b * n, inputs=1, flops_per_elem=1)]
    return ops


def _mlp_backward(layers: List[Tuple[int, int]], b: int, tag: str) -> List[dict]:
    """dgrad (b x k from b x n) and wgrad (k x n from b) per layer, last layer first."""
    ops = []
    for i, (k, n) in reversed(list(enumerate(layers))):
        ops += [_k(f"{tag}{i}_relu_bwd", type="elementwise", n=b * n, inputs=2, flops_per_elem=1),
                _gemm(f"{tag}{i}_dgrad", b, k, n), _gemm(f"{tag}{i}_wgrad", k, n, b)]
    return ops


def step(m: DLRM, world: int, batch: int, *, train: bool = True, hot_rows: Optional[int] = None,
         comm_sms: int = 16) -> dict:
    """One step on `world` ranks with `batch` samples per rank (global batch
    `world * batch`): table-wise sharded embeddings, data-parallel MLPs.
    `hot_rows` replaces each rank's table footprint by a hot set of that many
    rows (a cache that catches every lookup)."""
    if m.tables % world:
        raise ParallelError(f"table-wise sharding needs world={world} to divide {m.name}'s {m.tables} tables")
    tl = m.tables // world
    bags = batch * world * tl
    rows = hot_rows if hot_rows else tl * m.rows
    bot, top = m._mlp()
    f = m.tables + 1
    a2a = batch * m.tables * m.dim * m.comm_bytes
    emb = dict(type="embedding_bag", bags=bags, pooling=m.pooling, dim=m.dim, elem_bytes=m.table_bytes,
               table_rows=rows)
    ops = [_k("embedding_lookup", **emb),
           _coll("all_to_all", a2a, "world", comm_sms, "embedding_a2a")]
    ops += _mlp_ops(bot, batch, "bottom")
    ops += [_gemm("interaction", batch * f, f, m.dim),
            _k("interaction_gather", type="elementwise", n=batch * m.interaction_out(), inputs=1, flops_per_elem=1)]
    ops += _mlp_ops(top, batch, "top")
    if train:
        ops += _mlp_backward(top, batch, "top")
        ops += [_gemm("interaction_dgrad", batch * f, m.dim, f), _gemm("interaction_dgrad2", batch * f, m.dim, f)]
        ops += _mlp_backward(bot, batch, "bottom")
        ops += [_coll("all_to_all", a2a, "world", comm_sms, "embedding_grad_a2a"),
                _k("embedding_update", **dict(emb, update=True))]
        if world > 1:
            ops.append(_coll("all_reduce", m.mlp_params() * 2, "world", comm_sms, "mlp_grad_allreduce"))
    lookups = batch * world * m.tables * m.pooling * m.dim
    return {
        "name": f"{m.name} {'train' if train else 'infer'} b{batch} x{world}",
        "spmd": {"world": world, "program": {"streams": [ops]}},
        "model_flops": (3.0 if train else 1.0) * m.dense_flops(batch * world) + lookups * (3 if train else 1),
    }


__all__ = ["DLRM", "DLRM_2TB", "MODELS", "balance", "step"]
