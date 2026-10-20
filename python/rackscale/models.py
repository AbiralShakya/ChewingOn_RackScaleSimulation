"""Model architecture descriptions (enough to derive every kernel shape)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass(frozen=True)
class MoE:
    experts: int
    topk: int
    expert_ffn: int
    shared_experts: int = 0
    #: Leading layers that use a dense FFN instead of MoE.
    first_dense_layers: int = 0
    dense_ffn: int = 0
    #: Bytes per element for dispatch (DeepSeek dispatches FP8, combines BF16).
    dispatch_bytes: float = 2.0


@dataclass(frozen=True)
class MLA:
    """DeepSeek multi-head latent attention."""
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    rope_dim: int = 64
    nope_dim: int = 128
    v_head_dim: int = 128


@dataclass(frozen=True)
class Model:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    moe: Optional[MoE] = None
    mla: Optional[MLA] = None
    dtype: str = "bf16"
    extra: dict = field(default_factory=dict)

    def with_layers(self, n: int) -> "Model":
        return replace(self, layers=n)

    # --- parameter / FLOP accounting -------------------------------------
    def attn_params(self) -> int:
        h = self.hidden
        if self.mla:
            m = self.mla
            return (h * m.q_lora_rank + m.q_lora_rank * self.heads * (m.nope_dim + m.rope_dim)
                    + h * (m.kv_lora_rank + m.rope_dim) + m.kv_lora_rank * self.heads * (m.nope_dim + m.v_head_dim)
                    + self.heads * m.v_head_dim * h)
        return h * (self.heads + 2 * self.kv_heads) * self.head_dim + self.heads * self.head_dim * h

    def ffn_active_params(self, layer: int) -> int:
        h = self.hidden
        if self.moe and layer >= self.moe.first_dense_layers:
            m = self.moe
            return 3 * h * m.expert_ffn * (m.topk + m.shared_experts) + h * m.experts
        ffn = self.moe.dense_ffn if self.moe and self.moe.dense_ffn else self.ffn
        return 3 * h * ffn

    def active_params(self) -> int:
        return sum(self.attn_params() + self.ffn_active_params(l) for l in range(self.layers)) + self.hidden * self.vocab


LLAMA3_8B = Model("llama3-8b", hidden=4096, layers=32, heads=32, kv_heads=8, head_dim=128, ffn=14336, vocab=128256)
LLAMA3_70B = Model("llama3-70b", hidden=8192, layers=80, heads=64, kv_heads=8, head_dim=128, ffn=28672, vocab=128256)
LLAMA3_405B = Model("llama3-405b", hidden=16384, layers=126, heads=128, kv_heads=8, head_dim=128, ffn=53248, vocab=128256)
MIXTRAL_8X22B = Model("mixtral-8x22b", hidden=6144, layers=56, heads=48, kv_heads=8, head_dim=128, ffn=16384, vocab=32768,
                      moe=MoE(experts=8, topk=2, expert_ffn=16384))
QWEN3_235B_A22B = Model("qwen3-235b-a22b", hidden=4096, layers=94, heads=64, kv_heads=4, head_dim=128, ffn=12288, vocab=151936,
                        moe=MoE(experts=128, topk=8, expert_ffn=1536))
DEEPSEEK_V3 = Model("deepseek-v3", hidden=7168, layers=61, heads=128, kv_heads=128, head_dim=128, ffn=18432, vocab=129280,
                    moe=MoE(experts=256, topk=8, expert_ffn=2048, shared_experts=1, first_dense_layers=3, dense_ffn=18432,
                            dispatch_bytes=1.0),
                    mla=MLA())

ALL = {m.name: m for m in [LLAMA3_8B, LLAMA3_70B, LLAMA3_405B, MIXTRAL_8X22B, QWEN3_235B_A22B, DEEPSEEK_V3]}
