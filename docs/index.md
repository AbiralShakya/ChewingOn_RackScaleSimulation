---
hide:
  - navigation
---

<div class="hero" markdown>

# rackscale

<p class="tagline">Cycle-level simulation of a whole GPU rack, from one SASS instruction up to
NVLink collectives. Every SM of every GPU, warp schedulers, scoreboards, tensor-core queues,
TMA and mbarriers, coupled to an NCCL-faithful model of NVLink, NVSwitch and InfiniBand.</p>

[Get started](getting-started.md){ .md-button .md-button--primary }
[Examples gallery](examples.md){ .md-button }
[GitHub](https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation){ .md-button }

</div>

```bash
pip install git+https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation
rackscale demo --open
```

No GPU is needed. The demo simulates one decode step of Llama-3-70B on HGX H100 and HGX B200
in about three seconds, prints where every microsecond went, and opens the viewer.

![Per-SM CTA timeline of a simulated DeepSeek-V3 decode step on HGX B200](img/deepseek_v3_ep8_b200_per_sm.png)

<div class="grid cards" markdown>

-   :material-chip:{ .lg .middle } **Cycle level, rack scale**

    ---

    Every SM of every GPU in a 72-GPU GB200 NVL72, from SASS or synthetic kernels, with
    Nsight-style stall reasons. Blackwell `tcgen05`/UMMA, TMEM and TMA are modelled.

    [:octicons-arrow-right-24: One kernel at cycle level](tutorials/single-kernel.md)

-   :material-lan:{ .lg .middle } **Real fabrics**

    ---

    NCCL's algorithms and protocols (Ring, Tree, NVLS, CollNet) on packet- or transfer-level
    NVLink5, NVSwitch and InfiniBand fat-trees, validated against published numbers.

    [:octicons-arrow-right-24: A model on an NVL72 rack](tutorials/rack-run.md)

-   :material-record-rec:{ .lg .middle } **Any engine, captured once**

    ---

    Run vLLM, SGLang, a training loop or a megakernel once under `rackscale capture` on one GPU,
    then replay exactly what the GPU did on any rack with any parallelism plan.

    [:octicons-arrow-right-24: Capture and replay](tutorials/capture-replay.md)

-   :material-robot:{ .lg .middle } **A tool for agents**

    ---

    JSON API and an MCP server: an agent can simulate, find the bottleneck, drill into a
    kernel's warp roles and test what-ifs, with an error band on every number.

    [:octicons-arrow-right-24: Drive it from an agent](tutorials/agent.md)

-   :material-magnify-scan:{ .lg .middle } **See every bubble**

    ---

    One self-contained HTML page zooms from the rack's links through a GPU's SMs and a kernel's
    warp roles down to every warp, cycle by cycle, and the SASS listing.

    [:octicons-arrow-right-24: The viewer](VIZ.md)

-   :material-scale-balance:{ .lg .middle } **Measured, not guessed**

    ---

    Hardware numbers come from microbenchmarks on a B300 (sm_103) and A100s (sm_80), and every
    reference kernel is checked against Nsight Compute cycles in CI.

    [:octicons-arrow-right-24: Calibration and accuracy](CALIBRATION.md)

</div>

## Who it is for

- **Inference engineers** who want to know where a decode step's microseconds go before they
  rent the rack: weight streaming, MLA attention, all-to-all, or launch gaps.
- **ML researchers** who want to try a parallelism layout, a model shape or an async-RL
  schedule on hardware they do not have.
- **Hardware engineers** who want to change one number (L2 latency, NVLink bandwidth, UMMA
  rate) and see what it does to a real model end to end.

## How far to trust it

Single-kernel accuracy is measured against Nsight Compute cycles at the same clock on the GPUs we
have: on B300, 12 of 12 reference kernels are within 7%, and FlashAttention kernels average 3.3%
error. Larger systems (HGX boards, NVL72, InfiniBand clusters) are simulation only, built from
those measurements and published specifications. The docs say which inputs are
measured and which are presets (the agent API attaches that provenance to every run). Chapter 9 of the guide,
[How far to trust the numbers](book/09-validation.md), has the full table.
