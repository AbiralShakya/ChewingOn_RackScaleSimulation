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
class GLA:
    """Grouped Latent Attention (Zadouri, Strauss and Dao, "Hardware-Efficient
    Attention for Fast Decoding", 2025): MLA with ``latent_heads`` latents of
    ``latent_dim`` each, every latent serving ``heads / latent_heads`` query
    heads. Unlike MLA's single latent, the latents shard over tensor
    parallelism, so a TP rank caches and reads only its own."""
    latent_heads: int = 2
    latent_dim: int = 256
    rope_dim: int = 64
    nope_dim: int = 128
    v_head_dim: int = 128


@dataclass(frozen=True)
class LinearAttn:
    """Chunked linear-attention mixer with a matrix state per value head.

    ``delta=False`` is the scalar-gated form (GLA with a per-head scalar gate,
    RetNet, Lightning Attention), which is exactly Mamba-2's SSD with
    ``d_state = d_k`` and ``headdim = d_v``, and is lowered onto the SSD
    kernels. ``delta=True`` is Gated DeltaNet (Yang, Kautz and Hatamizadeh,
    2024; Qwen3-Next, Kimi Linear), lowered as SSD plus the per-chunk
    ``K K^T`` and WY-transform passes; the delta rule's sequential
    ``d_k x d_k`` correction between chunks is not modelled."""
    heads_qk: int
    heads_v: int
    d_k: int = 128
    d_v: int = 128
    d_conv: int = 4
    chunk: int = 64
    delta: bool = False

    def state_elems(self) -> int:
        return self.heads_v * self.d_k * self.d_v

    @property
    def conv_dim(self) -> int:
        return 2 * self.heads_qk * self.d_k + self.heads_v * self.d_v

    @property
    def in_proj_out(self) -> int:
        # q, k, v, output gate, decay (and beta for the delta rule)
        return self.conv_dim + self.heads_v * self.d_v + self.heads_v * (2 if self.delta else 1)


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
    #: Grouped-Tied Attention (same paper as GLA): each of the ``kv_heads``
    #: groups caches one tied state used as V and as the non-RoPE half of K,
    #: plus a single RoPE key of ``head_dim / 2`` shared by all groups.
    gta: bool = False
    gla: Optional[GLA] = None
    #: Linear-attention mixer for ``L`` blocks.
    linear: Optional[LinearAttn] = None
    #: Gated MLP (SwiGLU: gate, up, down) or the plain two-matrix MLP of
    #: ViT and DiT (fc1, GELU, fc2).
    gated: bool = True

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

    def linear_params(self) -> int:
        la = self.linear
        return (self.hidden * la.in_proj_out + la.conv_dim * la.d_conv + la.heads_v * la.d_v * self.hidden
                + 2 * la.heads_v)

    def _mlp_mats(self) -> int:
        return 3 if self.gated else 2

    def mlp_params(self) -> int:
        return self._mlp_mats() * self.hidden * self.ffn

    def attn_params(self) -> int:
        h = self.hidden
        if self.gla:
            g = self.gla
            return (h * self.heads * (g.nope_dim + g.rope_dim) + h * (g.latent_heads * g.latent_dim + g.rope_dim)
                    + g.latent_dim * self.heads * (g.nope_dim + g.v_head_dim) + self.heads * g.v_head_dim * h)
        if self.gta:
            d = self.head_dim
            return h * self.heads * d + h * self.kv_heads * d + h * (d // 2) + self.heads * d * h
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
        return self._mlp_mats() * h * ffn

    def active_params(self, lm_head: bool = True) -> int:
        """Parameters touched per token (embedding lookups excluded, LM head optional)."""
        head = self.hidden * self.vocab if lm_head else 0
        if self.blocks is not None:
            return sum(self._block_params()[b] for b in self.blocks) + head
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
        return self._mlp_mats() * h * ffn

    def total_params(self) -> int:
        """Every parameter, embedding and (untied) LM head included."""
        emb = 2 * self.vocab * self.hidden
        if self.blocks is not None:
            return sum(self._block_params()[b] for b in self.blocks) + emb
        return sum(self.attn_params() + self.ffn_params(l) for l in range(self.layers)) + emb

    def _block_params(self) -> dict:
        return {"M": self.mamba_params() if self.mamba else 0, "A": self.attn_params(), "F": self.mlp_params(),
                "L": self.linear_params() if self.linear else 0}

    def attn_layers(self) -> int:
        return self.block_list().count("A")

    def kv_elems_per_token_layer(self, tp: int = 1) -> int:
        """Cache elements one attention layer keeps per token on one of ``tp``
        tensor-parallel ranks. MLA's single latent is replicated on every
        rank; GLA's latents and GQA/GTA's KV heads shard until one is left."""
        if self.mla:
            return self.mla.kv_lora_rank + self.mla.rope_dim
        if self.gla:
            g = self.gla
            return max(1, g.latent_heads // tp) * g.latent_dim + g.rope_dim
        kv = max(1, self.kv_heads // tp)
        if self.gta:
            return kv * self.head_dim + self.head_dim // 2
        return 2 * kv * self.head_dim

    def cache_bytes_per_seq(self, context: int, elem_bytes: float = 2.0) -> float:
        """Inference cache of one sequence: the KV cache of the attention
        layers (grows with ``context``) plus the Mamba-2 conv and SSM states
        (constant). Attention layers use :meth:`kv_elems_per_token_layer` (MLA, GLA, GTA, GQA)."""
        kv = float(self.attn_layers() * self.kv_elems_per_token_layer() * context * elem_bytes)
        ssm = 0.0
        if self.mamba is not None:
            mb = self.mamba
            per = mb.nheads * mb.headdim * mb.d_state + mb.conv_dim * (mb.d_conv - 1)
            ssm = self.block_list().count("M") * per * elem_bytes
        if self.linear is not None:
            la = self.linear
            per = la.state_elems() + la.conv_dim * (la.d_conv - 1)
            ssm += self.block_list().count("L") * per * elem_bytes
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

# Size ladder (published configs): Llama 3.2 1B/3B (tied embeddings in the
# release; the LM head is still one 128256-wide GEMM per token), Qwen2.5-32B.
LLAMA32_1B = Model("llama3.2-1b", hidden=2048, layers=16, heads=32, kv_heads=8, head_dim=64, ffn=8192, vocab=128256)
LLAMA32_3B = Model("llama3.2-3b", hidden=3072, layers=28, heads=24, kv_heads=8, head_dim=128, ffn=8192, vocab=128256)
QWEN25_32B = Model("qwen2.5-32b", hidden=5120, layers=64, heads=40, kv_heads=8, head_dim=128, ffn=27648, vocab=152064)

# Smaller MoEs: Mixtral-8x7B (few large experts) and Qwen3-30B-A3B (many
# small experts, 3.3B active).
MIXTRAL_8X7B = Model("mixtral-8x7b", hidden=4096, layers=32, heads=32, kv_heads=8, head_dim=128, ffn=14336, vocab=32000,
                     moe=MoE(experts=8, topk=2, expert_ffn=14336))
QWEN3_30B_A3B = Model("qwen3-30b-a3b", hidden=2048, layers=48, heads=32, kv_heads=4, head_dim=128, ffn=6144,
                      vocab=151936, moe=MoE(experts=128, topk=8, expert_ffn=768))

# Attention what-ifs on the Llama-3-8B body: only the attention changes.
# MHA keeps 32 KV heads (Llama-2 style, 4x the GQA cache), MQA keeps one, and
# MLA swaps in DeepSeek's latent attention at this width (32 heads, 512-wide
# latent + 64 RoPE dims: 576 cache elements per token per layer, vs 2048 for
# GQA-8 and 8192 for MHA). Parameter counts stay within 10% of the original.
LLAMA3_8B_MHA = replace(LLAMA3_8B, name="llama3-8b-mha", kv_heads=32)
LLAMA3_8B_MQA = replace(LLAMA3_8B, name="llama3-8b-mqa", kv_heads=1)
LLAMA3_8B_MLA = replace(LLAMA3_8B, name="llama3-8b-mla", kv_heads=32, mla=MLA())

# Decode-attention variants at Llama-3-70B's shape (64 query heads of 128,
# everything but the attention block unchanged), after "Hardware-Efficient
# Attention for Fast Decoding" (Zadouri, Strauss and Dao, 2025): MQA, GTA with
# 8 tied groups, MLA with DeepSeek's 512 + 64 latent, and GLA with two 256-wide
# latents (half of MLA's KV cache per TP rank once TP >= 2).
LLAMA3_70B_MQA = replace(LLAMA3_70B, name="llama3-70b-mqa", kv_heads=1)
LLAMA3_70B_GTA = replace(LLAMA3_70B, name="llama3-70b-gta8", gta=True)
LLAMA3_70B_MLA = replace(LLAMA3_70B, name="llama3-70b-mla", mla=MLA())
LLAMA3_70B_GLA = replace(LLAMA3_70B, name="llama3-70b-gla2", gla=GLA(latent_heads=2, latent_dim=256))
LLAMA3_8B_GTA = replace(LLAMA3_8B, name="llama3-8b-gta8", gta=True)
LLAMA3_8B_GLA = replace(LLAMA3_8B, name="llama3-8b-gla2", gla=GLA(latent_heads=2, latent_dim=256))

# Linear-attention models at Llama-3-8B's width and depth (32 mixer + 32 MLP
# blocks). The mixer follows Qwen3-Next's Gated DeltaNet layout scaled to
# d_model 4096: 16 QK heads and 32 V heads of 128, short conv 4, chunk 64.
# ``gla-8b`` is the scalar-gated variant of the same mixer (no delta rule);
# ``gdn-hybrid-8b`` keeps one softmax-attention block in four (3:1, as in
# Qwen3-Next and Kimi Linear).
_GDN_8B = LinearAttn(heads_qk=16, heads_v=32, d_k=128, d_v=128, d_conv=4, chunk=64, delta=True)
GDN_8B = Model("gdn-8b", hidden=4096, layers=64, heads=32, kv_heads=8, head_dim=128, ffn=14336, vocab=128256,
               linear=_GDN_8B, blocks="LF" * 32)
GLA_LINEAR_8B = replace(GDN_8B, name="gla-8b", linear=replace(_GDN_8B, delta=False))
GDN_HYBRID_8B = replace(GDN_8B, name="gdn-hybrid-8b", blocks="LFLFLFAF" * 8)
_GDN_70B = LinearAttn(heads_qk=32, heads_v=64, d_k=128, d_v=128, d_conv=4, chunk=64, delta=True)
GDN_HYBRID_70B = Model("gdn-hybrid-70b", hidden=8192, layers=160, heads=64, kv_heads=8, head_dim=128, ffn=28672,
                       vocab=128256, linear=_GDN_70B, blocks="LFLFLFAF" * 20)

# Vision and diffusion transformers (chapter 18; `rackscale.vision`). Plain
# GELU MLPs and full multi-head attention. Encoders, so kept out of ALL.
VIT_L_14 = Model("vit-l/14", hidden=1024, layers=24, heads=16, kv_heads=16, head_dim=64, ffn=4096, vocab=0,
                 gated=False)
VIT_H_14 = Model("vit-h/14", hidden=1280, layers=32, heads=16, kv_heads=16, head_dim=80, ffn=5120, vocab=0,
                 gated=False)
VIT_22B = Model("vit-22b", hidden=6144, layers=48, heads=48, kv_heads=48, head_dim=128, ffn=24576, vocab=0,
                gated=False)
DIT_XL_2 = Model("dit-xl/2", hidden=1152, layers=28, heads=16, kv_heads=16, head_dim=72, ffn=4608, vocab=0,
                 gated=False)
#: A video DiT with the shape of Wan2.1-14B (dim 5120, 40 blocks, 40 heads,
#: FFN 13824, cross-attention to 512 text tokens).
VIDEO_DIT_14B = Model("video-dit-14b", hidden=5120, layers=40, heads=40, kv_heads=40, head_dim=128, ffn=13824,
                      vocab=0, gated=False)
VISION = {m.name: m for m in [VIT_L_14, VIT_H_14, VIT_22B, DIT_XL_2, VIDEO_DIT_14B]}

ALL = {m.name: m for m in [LLAMA32_1B, LLAMA32_3B, LLAMA3_8B, QWEN25_32B, LLAMA3_70B, LLAMA3_405B,
                           MIXTRAL_8X7B, MIXTRAL_8X22B, QWEN3_30B_A3B, QWEN3_235B_A22B, DEEPSEEK_V3,
                           LLAMA3_8B_MHA, LLAMA3_8B_MQA, LLAMA3_8B_MLA,
                           MAMBA2_8B, MAMBA2_HYBRID_8B, MAMBA2_70B, MAMBA2_HYBRID_70B, MAMBA2_70B_SSU8,
                           LLAMA3_70B_MQA, LLAMA3_70B_GTA, LLAMA3_70B_MLA, LLAMA3_70B_GLA, LLAMA3_8B_GTA, LLAMA3_8B_GLA,
                           GDN_8B, GLA_LINEAR_8B, GDN_HYBRID_8B, GDN_HYBRID_70B]}
