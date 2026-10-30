"""A plain Llama-3 decoder layer in PyTorch, written to match what the simulator lowers.

rackscale.lower lowers each layer as: RMSNorm, fused QKV GEMM (here three
GEMMs, as Megatron/torchtitan and DTensor TP need), RoPE, causal attention,
O projection, residual, RMSNorm, SwiGLU (gate and up GEMMs, the product, down
GEMM), residual. Nothing outside the layers (embedding, LM head, loss) is in a
training step, so nothing outside is built here.

Heads are inferred from the local weight shapes, so the same module runs
unsharded and under DTensor tensor parallelism (column-parallel Q/K/V hold
heads / tp heads each).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    hidden: int = 4096
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    ffn: int = 14336
    eps: float = 1e-5
    rope_theta: float = 500000.0


LLAMA3_8B = Cfg()


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary embedding on (B, S, H, D), positions 0..S-1."""
    s, d = x.shape[1], x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = torch.outer(torch.arange(s, device=x.device, dtype=torch.float32), inv)
    cos, sin = ang.cos()[None, :, None, :].to(x.dtype), ang.sin()[None, :, None, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.wq = nn.Linear(c.hidden, c.heads * c.head_dim, bias=False)
        self.wk = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.wv = nn.Linear(c.hidden, c.kv_heads * c.head_dim, bias=False)
        self.wo = nn.Linear(c.heads * c.head_dim, c.hidden, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        d = self.c.head_dim
        q = rope(self.wq(x).view(b, s, -1, d), self.c.rope_theta)
        k = rope(self.wk(x).view(b, s, -1, d), self.c.rope_theta)
        v = self.wv(x).view(b, s, -1, d)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                           enable_gqa=True)
        return self.wo(o.transpose(1, 2).reshape(b, s, -1))


class FFN(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.w1 = nn.Linear(c.hidden, c.ffn, bias=False)  # gate
        self.w3 = nn.Linear(c.hidden, c.ffn, bias=False)  # up
        self.w2 = nn.Linear(c.ffn, c.hidden, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.attention_norm = RMSNorm(c.hidden, c.eps)
        self.attention = Attention(c)
        self.ffn_norm = RMSNorm(c.hidden, c.eps)
        self.feed_forward = FFN(c)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class Stack(nn.Module):
    def __init__(self, c: Cfg, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(Layer(c) for _ in range(layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def tp_plan(sp: bool) -> dict:
    """Megatron-style DTensor plan for one Layer (torchtitan's, minus embedding and head)."""
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                   SequenceParallel)
    if not sp:
        return {
            "attention.wq": ColwiseParallel(), "attention.wk": ColwiseParallel(), "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(), "feed_forward.w3": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
        }
    return {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
        "attention.wq": ColwiseParallel(), "attention.wk": ColwiseParallel(), "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
        "feed_forward.w1": ColwiseParallel(), "feed_forward.w3": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    }
