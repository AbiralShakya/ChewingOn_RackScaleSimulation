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
class Mamba2:
    """Mamba-2 (SSD) mixer, as in state-spaces/mamba ``Mamba2``."""
    d_inner: int
    headdim: int = 64
    d_state: int = 128
    ngroups: int = 1
    d_conv: int = 4
    chunk: int = 256
    #: What-if: headdim rows per CTA in the decode state update (None = the
    #: kernel's own choice, 4 rows at d_state 128).
    state_block_m: Optional[int] = None

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.ngroups * self.d_state

    @property
    def in_proj_out(self) -> int:
        # z, x, B, C, dt
        return 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads


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
    mamba: Optional[Mamba2] = None
    #: Heterogeneous stacks (Mamba, hybrids): one letter per block, ``M`` =
    #: Mamba-2 mixer, ``A`` = attention, ``F`` = MLP, each with its own
    #: pre-norm and residual. ``None`` = the usual ``layers`` x (attention + FFN).
    blocks: Optional[str] = None

    def with_layers(self, n: int) -> "Model":
        """Keep the first ``n`` layers. For a block stack, a layer is one block."""
        if self.blocks is not None:
            return replace(self, layers=n, blocks=self.blocks[:n])
        return replace(self, layers=n)

    def block_list(self) -> str:
        return self.blocks if self.blocks is not None else "AF" * self.layers

    # --- parameter / FLOP accounting -------------------------------------
    def mamba_params(self) -> int:
        mb = self.mamba
        return self.hidden * mb.in_proj_out + mb.conv_dim * mb.d_conv + mb.d_inner * self.hidden + 3 * mb.nheads + mb.d_inner

    def mlp_params(self) -> int:
        return 3 * self.hidden * self.ffn

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

    def active_params(self, lm_head: bool = True) -> int:
        """Parameters touched per token (embedding lookups excluded, LM head optional)."""
        head = self.hidden * self.vocab if lm_head else 0
        if self.blocks is not None:
            per = {"M": self.mamba_params() if self.mamba else 0, "A": self.attn_params(), "F": self.mlp_params()}
            return sum(per[b] for b in self.blocks) + head
        return sum(self.attn_params() + self.ffn_active_params(l) for l in range(self.layers)) + head

    def bytes_per_param(self) -> float:
        return {"fp8": 1.0, "fp4": 0.5, "nvfp4": 0.5625, "fp32": 4.0}.get(self.dtype, 2.0)

    def ffn_params(self, layer: int) -> int:
        """All FFN parameters of a layer (every expert, the router)."""
        h = self.hidden
        if self.moe and layer >= self.moe.first_dense_layers:
            m = self.moe
            return 3 * h * m.expert_ffn * (m.experts + m.shared_experts) + h * m.experts
        ffn = self.moe.dense_ffn if self.moe and self.moe.dense_ffn else self.ffn
        return 3 * h * ffn

    def total_params(self) -> int:
        """Every parameter, embedding and (untied) LM head included."""
        emb = 2 * self.vocab * self.hidden
        if self.blocks is not None:
            per = {"M": self.mamba_params() if self.mamba else 0, "A": self.attn_params(), "F": self.mlp_params()}
            return sum(per[b] for b in self.blocks) + emb
        return sum(self.attn_params() + self.ffn_params(l) for l in range(self.layers)) + emb

    def attn_layers(self) -> int:
        return self.block_list().count("A")

    def cache_bytes_per_seq(self, context: int, elem_bytes: float = 2.0) -> float:
        """Inference cache of one sequence: the KV cache of the attention
        layers (grows with ``context``) plus the Mamba-2 conv and SSM states
        (constant). MLA and MoE layouts are not handled here."""
        kv = 2.0 * self.attn_layers() * self.kv_heads * self.head_dim * context * elem_bytes
        ssm = 0.0
        if self.mamba is not None:
            mb = self.mamba
            per = mb.nheads * mb.headdim * mb.d_state + mb.conv_dim * (mb.d_conv - 1)
            ssm = self.block_list().count("M") * per * elem_bytes
        return kv + ssm


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

# 8B-class models after "An Empirical Study of Mamba-based Language Models"
# (Waleffe et al., 2024): d_model 4096; Mamba-2 with expand 2, headdim 64,
# d_state 128, 8 groups; the hybrid has 24 Mamba-2, 4 attention and 28 MLP
# blocks with the attention blocks spread evenly. The transformer baseline is
# the Llama-3-8B shape (GQA 32/8). All three use Llama 3's vocabulary so the LM
# heads are identical, and the pure Mamba-2 stack has 64 blocks so its
# non-embedding parameters match the transformer's (7.0B; the hybrid is 7.7B).
_M2_8B = Mamba2(d_inner=8192, headdim=64, d_state=128, ngroups=8, d_conv=4, chunk=256)
MAMBA2_8B = Model("mamba2-8b", hidden=4096, layers=64, heads=32, kv_heads=8, head_dim=128, ffn=14336, vocab=128256,
                  mamba=_M2_8B, blocks="M" * 64)


def _hybrid_blocks(n_mamba: int, n_attn: int, n_mlp: int) -> str:
    """Mixers (Mamba-2 or attention) each followed by an MLP while MLPs remain;
    the attention mixers sit at even spacing among the mixers."""
    mixers = n_mamba + n_attn
    attn_at = {round((i + 0.5) * mixers / n_attn) for i in range(n_attn)}
    out = []
    for i in range(mixers):
        out.append("A" if i in attn_at else "M")
        if i < n_mlp:
            out.append("F")
    return "".join(out) + "F" * max(0, n_mlp - mixers)


MAMBA2_HYBRID_8B = Model("mamba2-hybrid-8b", hidden=4096, layers=56, heads=32, kv_heads=8, head_dim=128, ffn=14336,
                         vocab=128256, mamba=_M2_8B, blocks=_hybrid_blocks(24, 4, 28))

# 70B class at Llama-3-70B's width (d_model 8192): Mamba-2 with d_inner 16384
# (256 heads of 64), and block counts chosen so the non-embedding parameters
# match Llama-3-70B's 68.5B. The hybrid keeps the 8B hybrid's 24:4:28 mix.
_M2_70B = Mamba2(d_inner=16384, headdim=64, d_state=128, ngroups=8, d_conv=4, chunk=256)
MAMBA2_70B = Model("mamba2-70b", hidden=8192, layers=163, heads=64, kv_heads=8, head_dim=128, ffn=28672, vocab=128256,
                   mamba=_M2_70B, blocks="M" * 163)
# What-if: the same model with a decode state update that moves 8 headdim rows
# (4 KB of state) per CTA instead of 4 (2 KB).
MAMBA2_70B_SSU8 = replace(MAMBA2_70B, name="mamba2-70b-ssu8", mamba=replace(_M2_70B, state_block_m=8))
MAMBA2_HYBRID_70B = Model("mamba2-hybrid-70b", hidden=8192, layers=126, heads=64, kv_heads=8, head_dim=128, ffn=28672,
                          vocab=128256, mamba=_M2_70B, blocks=_hybrid_blocks(54, 9, 63))

ALL = {m.name: m for m in [LLAMA3_8B, LLAMA3_70B, LLAMA3_405B, MIXTRAL_8X22B, QWEN3_235B_A22B, DEEPSEEK_V3,
                           MAMBA2_8B, MAMBA2_HYBRID_8B, MAMBA2_70B, MAMBA2_HYBRID_70B, MAMBA2_70B_SSU8]}
