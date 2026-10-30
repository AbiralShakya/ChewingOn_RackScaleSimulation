import pytest

import rackscale as rs
from rackscale import models, vision
from rackscale.lower import ParallelError


def kernels(wl):
    ops = wl["spmd"]["program"]["streams"][0]
    if ops and ops[0].get("op") == "loop":
        ops = ops[0]["body"]
    return [o for o in ops if o["op"] == "kernel"]


def test_vit_is_non_causal_with_a_plain_mlp():
    m = models.VIT_L_14
    assert abs(m.active_params(lm_head=False) / 1e6 - 302) < 2  # ViT-L: 304M with biases
    ks = kernels(vision.vit_forward(m, batch=2, layers=1))
    attn = [k for k in ks if k["kernel"]["type"] == "attention"]
    assert attn and not attn[0]["kernel"]["causal"] and attn[0]["kernel"]["sq"] == 257
    names = [k["name"] for k in ks]
    assert "fc1" in names and "gelu" in names and "gate_up_proj" not in names
    assert ks[0]["name"] == "patch_embed" and ks[0]["kernel"]["k"] == 3 * 14 * 14


def test_encoder_flops_count_full_attention():
    m = models.VIT_L_14
    f = vision.encoder_flops(m, 1, 257)
    assert f == pytest.approx(2 * m.active_params(lm_head=False) * 257 + 4 * 24 * 16 * 64 * 257 * 257)


def test_dit_sample_loops_steps_and_doubles_the_batch_for_guidance():
    wl = vision.dit_sample(models.DIT_XL_2, batch=4, latent=32, steps=50, layers=2)
    loop = wl["spmd"]["program"]["streams"][0][0]
    assert loop["op"] == "loop" and loop["times"] == 50
    attn = [o for o in loop["body"] if o.get("kernel", {}).get("type") == "attention"]
    assert attn[0]["kernel"]["batch"] == 8 and attn[0]["kernel"]["sq"] == 256
    assert any(o["name"].endswith("adaln") for o in loop["body"] if o["op"] == "kernel")


def test_video_dit_ulysses_and_cross_attention():
    m = models.VIDEO_DIT_14B
    assert vision.patches((90, 160), 2, 21) == 75600
    wl = vision.dit_sample(m, latent=(90, 160), frames=21, steps=1, text_tokens=512, cp=8, layers=1,
                           latent_channels=16, learn_sigma=False)
    assert wl["spmd"]["world"] == 8
    body = wl["spmd"]["program"]["streams"][0][0]["body"]
    a2a = [o for o in body if o["op"] == "collective"]
    assert [o["kind"] for o in a2a] == ["all_to_all", "all_to_all"]
    self_attn = next(o for o in body if o.get("name") == "l0.attention")["kernel"]
    assert (self_attn["heads_q"], self_attn["sq"], self_attn["skv"]) == (5, 75600, 75600)
    cross = next(o for o in body if o.get("name") == "l0.cross_attention")["kernel"]
    assert (cross["sq"], cross["skv"]) == (75600 // 8, 512)
    with pytest.raises(ParallelError):
        vision.dit_sample(m, cp=16)  # 40 heads do not split over 16 ranks


def test_vlm_prefill_feeds_image_tokens_to_the_llm():
    wl = vision.vlm_prefill(models.LLAMA3_8B, models.VIT_L_14, batch=2, images=2, image=336, text=100, layers=1)
    ks = kernels(wl)
    llm_attn = next(k for k in ks if k["name"] == "llm.attention")["kernel"]
    assert llm_attn["causal"] and llm_attn["sq"] == 2 * 576 + 100
    vit_attn = next(k for k in ks if k["name"] == "vit.attention")["kernel"]
    assert not vit_attn["causal"] and vit_attn["batch"] == 4 and vit_attn["sq"] == 576
    f = vision.vlm_flops(models.LLAMA3_8B, models.VIT_L_14, batch=2, images=2, image=336, text=100)
    assert set(f) == {"vit", "projector", "llm"} and f["llm"] > 10 * f["vit"]


def test_vit_block_simulates():
    r = rs.simulate("hgx_b300_8", vision.vit_forward(models.VIT_L_14, batch=4, layers=1), fast=True)
    assert r.makespan_us > 0
