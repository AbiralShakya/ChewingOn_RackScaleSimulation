"""Vision, diffusion and multimodal workloads: ViT, DiT and VLM prefill.

Chapter 18 of the book. Each lowering reuses the transformer blocks of
:mod:`rackscale.lower` and changes what these models change:

* **ViT** (image encoder): a patch-embedding GEMM, then non-causal attention
  over the patches and a plain GELU MLP. One forward per image.
* **DiT** (diffusion transformer): the same blocks on latent patches, with
  adaLN modulation from the timestep and class (or pooled text), optional
  cross-attention to text tokens, run once per denoising step, on twice the
  batch with classifier-free guidance. A long video sequence shards over
  ``cp`` ranks with Ulysses: two all-to-alls per attention turn sequence
  shards into head shards and back.
* **VLM prefill**: a ViT encodes the images, a two-layer projector maps its
  patch tokens into the language model, and the LLM prefills the image and
  text tokens together (causal). Decode is :func:`rackscale.lower.decode_step`
  with the image tokens in the context.

Not lowered: the VAE that turns DiT latents into pixels, image
preprocessing, and the text encoder of a text-to-image model (run once per
prompt). The closed forms count FLOPs as two per multiply-add.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from .lower import Parallel, ParallelError, _attention_ops, _coll, _ffn_ops, _gemm, _k, _layers, check
from .models import Model


def patches(image: Union[int, Tuple[int, int]], patch: int, frames: int = 1) -> int:
    """Patch tokens of `frames` images (or latents) of `image` x `image`, or
    `(height, width)`, cut into `patch` x `patch` squares."""
    h, w = (image, image) if isinstance(image, int) else image
    return frames * (h // patch) * (w // patch)


def encoder_flops(m: Model, images: int, seq: int, patch_in: int = 0, patch_tokens: int = 0) -> float:
    """Forward FLOPs of a non-causal encoder over `images` sequences of `seq`
    tokens: 2 x params x tokens for the matmuls, plus 4 x layers x heads x
    head_dim x seq per token for attention (every query sees every key, so
    twice a causal pass), plus the patch-embedding GEMM."""
    tokens = images * seq
    dense = 2.0 * m.active_params(lm_head=False) * tokens
    attn = 4.0 * m.attn_layers() * m.heads * m.head_dim * seq * tokens
    return dense + attn + 2.0 * images * patch_tokens * m.hidden * patch_in


def vit_ops(m: Model, p: Parallel, images: int, *, image: int = 224, patch: int = 14, channels: int = 3,
            cls_token: bool = True, tag: str = "") -> List[dict]:
    """A ViT forward over `images` images: patch embedding, position add,
    `m.layers` non-causal blocks, final norm."""
    n = patches(image, patch)
    seq = n + (1 if cls_token else 0)
    h = m.hidden
    return ([_gemm(f"{tag}patch_embed", images * n, h, channels * patch * patch),
             _k(f"{tag}pos_embed", type="elementwise", n=images * seq * h, inputs=2, flops_per_elem=1)]
            + _layers(m, p, images, seq, seq, False, tag)
            + [_k(f"{tag}final_norm", type="row_norm", rows=images * seq, cols=h)])


def vit_forward(m: Model, p: Parallel = Parallel(), batch: int = 1, *, image: int = 224, patch: int = 14,
                channels: int = 3, cls_token: bool = True, layers: Optional[int] = None) -> dict:
    """Encode `batch` images on every data-parallel replica (`p.tp` ranks each)."""
    check(m, p)
    if layers is not None:
        m = m.with_layers(layers)
    n = patches(image, patch)
    seq = n + (1 if cls_token else 0)
    ops = vit_ops(m, p, batch, image=image, patch=patch, channels=channels, cls_token=cls_token)
    replicas = p.world() // max(1, p.tp)
    return {
        "name": f"{m.name} encode b{batch} {image}px/{patch} {p}",
        "spmd": {"world": p.world(), "program": {"streams": [ops]}},
        "model_flops": replicas * encoder_flops(m, batch, seq, channels * patch * patch, n),
    }


def dit_flops(m: Model, batch: int, seq: int, *, steps: int = 1, cfg: bool = True, text_tokens: int = 0) -> float:
    """FLOPs of `steps` denoising steps: the encoder FLOPs of the latent
    sequence, plus adaLN (h -> 6h per block, once per sample) and, with text,
    cross-attention (q and o over the latent tokens, k and v over the text
    tokens, and 4 x heads x head_dim x text per latent token). Classifier-free
    guidance doubles the batch."""
    b = batch * (2 if cfg else 1)
    h, L = m.hidden, m.layers
    f = encoder_flops(m, b, seq) + 2.0 * L * b * 6 * h * h
    if text_tokens:
        f += L * (2.0 * 2 * h * h * b * seq + 2.0 * 2 * h * h * b * text_tokens
                  + 4.0 * m.heads * m.head_dim * text_tokens * b * seq)
    return steps * f


def _ulysses_attention(m: Model, cp: int, batch: int, seq: int, tag: str) -> List[dict]:
    """Self-attention with the sequence sharded over `cp` ranks: each rank
    projects its tokens, an all-to-all turns sequence shards into head
    shards, attention runs over the whole sequence on heads / cp heads, and
    a second all-to-all turns it back. The ``bytes`` of an all-to-all are
    the rank's whole buffer, (cp - 1) / cp of which leaves the rank."""
    t = batch * seq // cp
    h, hq, hkv, d = m.hidden, m.heads, m.kv_heads, m.head_dim
    return [_k(f"{tag}input_norm", type="row_norm", rows=t, cols=h),
            _gemm(f"{tag}qkv_proj", t, (hq + 2 * hkv) * d, h),
            _coll("all_to_all", t * (hq + 2 * hkv) * d * 2, "world", 16, f"{tag}ulysses_qkv_a2a"),
            _k(f"{tag}attention", type="attention", batch=batch, heads_q=hq // cp, heads_kv=max(1, hkv // cp),
               sq=seq, skv=seq, d=d, causal=False),
            _coll("all_to_all", t * hq * d * 2, "world", 16, f"{tag}ulysses_out_a2a"),
            _gemm(f"{tag}o_proj", t, h, hq * d),
            _k(f"{tag}residual", type="elementwise", n=t * h, inputs=2, flops_per_elem=1)]


def _cross_attention(m: Model, p: Parallel, batch: int, sq: int, text_tokens: int, tag: str) -> List[dict]:
    """Cross-attention from `sq` latent tokens per sample (this rank's) to
    `text_tokens` text tokens, heads split over TP. The text K/V are
    projected every step here; caching them across steps would remove the
    kv_proj GEMM."""
    t = batch * sq
    h, tp = m.hidden, max(1, p.tp)
    heads = m.heads // tp
    ops = [_k(f"{tag}cross_norm", type="row_norm", rows=t, cols=h),
           _gemm(f"{tag}cross_q_proj", t, heads * m.head_dim, h),
           _gemm(f"{tag}cross_kv_proj", batch * text_tokens, 2 * heads * m.head_dim, h),
           _k(f"{tag}cross_attention", type="attention", batch=batch, heads_q=heads, heads_kv=heads, sq=sq,
              skv=text_tokens, d=m.head_dim, causal=False),
           _gemm(f"{tag}cross_o_proj", t, h, heads * m.head_dim)]
    if tp > 1:
        ops.append(_coll("all_reduce", t * h * 2, f"block:{tp}", p.comm_sms, f"{tag}cross_allreduce"))
    ops.append(_k(f"{tag}cross_residual", type="elementwise", n=t * h, inputs=2, flops_per_elem=1))
    return ops


def dit_step_ops(m: Model, p: Parallel, batch: int, seq: int, *, text_tokens: int = 0, cp: int = 1,
                 out_dim: int = 32, tag: str = "") -> List[dict]:
    """One denoising step of a DiT on `batch` samples (already doubled for
    guidance) of `seq` latent tokens each: per block, adaLN modulation, the
    self-attention block, optional cross-attention, the MLP block, with
    scale/shift and gate kernels around each, and the final projection to
    `out_dim` values per token (patch^2 x latent channels, x2 when the model
    also predicts the noise variance)."""
    h = m.hidden
    t = batch * seq // cp
    ops: List[dict] = [_k(f"{tag}t_embed", type="elementwise", n=batch * h, inputs=1, flops_per_elem=8)]
    for layer in range(m.layers):
        lt = f"{tag}l{layer}."
        ops.append(_gemm(f"{lt}adaln", batch, 6 * h, h))
        ops.append(_k(f"{lt}modulate_attn", type="elementwise", n=t * h, inputs=3, flops_per_elem=2))
        if cp > 1:
            ops += _ulysses_attention(m, cp, batch, seq, lt)
        else:
            ops += _attention_ops(m, p, batch, seq, seq, False, lt)
        if text_tokens:
            ops += _cross_attention(m, p, batch, seq // cp, text_tokens, lt)
        ops.append(_k(f"{lt}modulate_mlp", type="elementwise", n=t * h, inputs=3, flops_per_elem=2))
        ops += _ffn_ops(m, p if cp == 1 else Parallel(), layer, t, lt)
    ops.append(_gemm(f"{tag}final_proj", t, out_dim, h))
    return ops


def dit_sample(m: Model, p: Parallel = Parallel(), batch: int = 1, *, latent: Union[int, Tuple[int, int]] = 32,
               patch: int = 2,
               frames: int = 1, steps: int = 50, cfg: bool = True, text_tokens: int = 0, cp: int = 1,
               latent_channels: int = 4, learn_sigma: bool = True, layers: Optional[int] = None) -> dict:
    """Generate `batch` images (or videos of `frames` latent frames) with a
    DiT: `steps` denoising steps over a `latent` x `latent` (or `(h, w)`)
    latent cut into `patch` x `patch` patches. With `cp` > 1 the sequence shards over `cp`
    ranks with Ulysses (no TP); otherwise `p` lays the model out."""
    check(m, p)
    if cp > 1:
        if p.world() > 1:
            raise ParallelError("cp shards one sample's sequence over cp ranks; use it without tp/dp")
        if m.heads % cp:
            raise ParallelError(f"Ulysses needs cp={cp} to divide {m.name}'s {m.heads} heads")
    if layers is not None:
        m = m.with_layers(layers)
    seq = patches(latent, patch, frames)
    if seq % cp:
        raise ParallelError(f"{seq} latent tokens do not split over cp={cp}")
    b = batch * (2 if cfg else 1)
    out_dim = patch * patch * latent_channels * (2 if learn_sigma else 1)
    step = dit_step_ops(m, p, b, seq, text_tokens=text_tokens, cp=cp, out_dim=out_dim)
    world = cp if cp > 1 else p.world()
    replicas = 1 if cp > 1 else p.world() // max(1, p.tp)
    return {
        "name": f"{m.name} sample b{batch} {seq} tokens x{steps} steps{' cfg' if cfg else ''}"
                + (f" cp{cp}" if cp > 1 else f" {p}"),
        "spmd": {"world": world, "program": {"streams": [[{"op": "loop", "times": steps, "body": step}]]}},
        "model_flops": replicas * dit_flops(m, batch, seq, steps=steps, cfg=cfg, text_tokens=text_tokens),
    }


def vlm_prefill(llm: Model, vit: Model, p: Parallel = Parallel(), batch: int = 1, *, images: int = 1,
                image: int = 336, patch: int = 14, text: int = 256, layers: Optional[int] = None) -> dict:
    """Prefill `batch` requests of `images` images and `text` text tokens:
    the ViT (patch tokens only, no class token, as LLaVA-style models feed
    the language model), a GELU projector vit.hidden -> llm.hidden ->
    llm.hidden, then causal LLM prefill over images x patches + text tokens.
    Both models use the same tensor-parallel layout, as vLLM runs them.
    :func:`vlm_flops` splits the closed-form FLOPs between the three."""
    check(llm, p)
    check(vit, p)
    if layers is not None:
        llm = llm.with_layers(layers)
    n = patches(image, patch)
    seq = images * n + text
    tp = max(1, p.tp)
    img_tokens = batch * images * n
    ops = vit_ops(vit, p, batch * images, image=image, patch=patch, cls_token=False, tag="vit.")
    ops += [_gemm("proj.fc1", img_tokens, llm.hidden, vit.hidden),
            _k("proj.gelu", type="elementwise", n=img_tokens * llm.hidden, inputs=1, flops_per_elem=8),
            _gemm("proj.fc2", img_tokens, llm.hidden, llm.hidden)]
    ops += _layers(llm, p, batch, seq, seq, True, "llm.")
    replicas = p.world() // tp
    f = vlm_flops(llm, vit, batch, images=images, image=image, patch=patch, text=text)
    return {
        "name": f"{llm.name}+{vit.name} prefill b{batch} {images}x{n}+{text} tokens {p}",
        "spmd": {"world": p.world(), "program": {"streams": [ops]}},
        "model_flops": replicas * sum(f.values()),
    }


def vlm_flops(llm: Model, vit: Model, batch: int = 1, *, images: int = 1, image: int = 336, patch: int = 14,
              text: int = 256) -> dict:
    """Closed-form FLOPs of :func:`vlm_prefill`, split into the ViT, the
    projector and the LLM (causal: a token attends to half the sequence on
    average)."""
    n = patches(image, patch)
    seq = images * n + text
    img_tokens = batch * images * n
    return {"vit": encoder_flops(vit, batch * images, n, 3 * patch * patch, n),
            "projector": 2.0 * img_tokens * llm.hidden * (vit.hidden + llm.hidden),
            "llm": 2.0 * llm.active_params(lm_head=False) * batch * seq
            + 4.0 * llm.attn_layers() * llm.heads * llm.head_dim * (seq / 2) * batch * seq}


__all__ = ["patches", "encoder_flops", "vit_ops", "vit_forward", "dit_flops", "dit_step_ops", "dit_sample",
           "vlm_prefill", "vlm_flops"]
