# FlashAttention across generations

Study 1 of [RESEARCH.md](../../RESEARCH.md). It runs Dao-AILab's FlashAttention kernels on real
A100 (sm_80) and B300 (sm_103) GPUs, replays the same kernels in the simulator from their
recorded SASS, and checks the simulator against Nsight Compute. Then it uses the simulator
alone for what-ifs and for context-parallel attention on 72 GPUs.

Results and figures are in the write-up (`results/study.json` holds every number it shows).

## What is new in the simulator for this study

The kernels that matter here, FA3 and FA4, are *warp specialised*: a load warp, an MMA warp,
softmax warpgroups, a correction warpgroup and an epilogue warp run different code paths
through one binary and hand work to each other through mbarriers and named barriers. A static
SASS listing plus a branch profile cannot reproduce that, so this study adds an exact path:

* **`warp_path`** ([`capture/nvbit/`](../../capture/nvbit/), shared with `rackscale capture`): an NVBit 1.8 tool that records, for a few sampled CTAs, every
  instruction each warp executed, with what a timing model cannot recover statically: lane
  addresses, the mbarrier behind every `SYNCS` op, expect-tx byte counts, try_wait results
  and parities, TMA source boxes (address runs and bytes) for every CTA, tcgen05 instruction
  descriptors (UMMA shapes), and the registers that name barriers. It also writes the
  function's SASS with its 128-bit encodings, so the simulator decodes the real control bits.
* **`trace_to_rsim.py`** (now `python/rackscale/nvbit/trace_to_rsim.py`; the script here runs it): turns a recording into an `.rpath` file. It removes mbarrier spin
  loops (a failed try_wait, the sleep and branch around it, the retries), keeps non-blocking
  polls, converts each successful wait into an absolute phase count (a repeated parity means
  the warp skipped a phase; FA4's lazy rescaling does), and derives each barrier's arrival
  count from its `mbarrier.init`.
* **Simulator (`crates/`)**:
  * `rsim-sass/src/path.rs`: the `.rpath` reader and per-warp replay. CTAs that were not
    recorded replay a recorded CTA with the same cluster rank, with addresses shifted by
    per-instruction grid strides, and with their own TMA sources.
  * mbarriers with transaction counts (`expect_tx` adds, TMA completion subtracts).
  * 2-CTA clusters: cross-SM mbarrier events with the measured DSMEM latency, and
    `cta_group::2` UMMAs that run on both SMs' tensor cores.
  * Named barriers whose id is in a register, `bar.arrive`, and generation-ordered matching.
  * A separate `mma.sync` rate. B300 runs HMMA at 2037 FLOP/clk/SM, a quarter of tcgen05.
  * Stall attribution per warp index (role) and per mbarrier.

## Reproduce

On the GPU machine (della-tridao GPU 3 for B300, Slurm for A100):

```bash
./sync_della.sh                                   # copy this directory, build the NVBit tool
./sync_della.sh 'bash remote/setup.sh'            # torch, FA2 build, FA4 (CuTe DSL), once
CUDA_VISIBLE_DEVICES=3 bash remote/run_gpu.sh b300 naive sdpa_flash sdpa_cudnn fa2 fa4   # timing + ncu
CUDA_VISIBLE_DEVICES=3 bash remote/trace_b300.sh   # NVBit recordings (validation shape)
bash remote/cp_shapes.sh b300 fa4 FlashAttentionForwardSm100 0,1,2,3   # ring-step shapes
sbatch remote/a100.sbatch && sbatch remote/a100_prof.sbatch && sbatch remote/a100_cp2.sbatch
```

Copy `results/<gpu>/` back into `results/` and the trace directories into `traces/` (not in
git: the recordings are 0.1 to 0.5 GB each), then:

```bash
cargo build --release -p rsim-cli
python research/flash_attention/run_study.py all       # validate, what-ifs, rack-scale runs, figures
```

## Files

| path | what |
|---|---|
| `remote/fa_bench.py` | one implementation, timed back to back (naive, SDPA flash/cuDNN, FA2, FA4) |
| `remote/ncu_attn.py` | Nsight Compute ground truth for the profiled call |
| `remote/trace.sh`, `remote/*.sh`, `remote/*.sbatch` | NVBit recording and the GPU runs |
| [`capture/nvbit/`](../../capture/nvbit/) | the NVBit tool (moved; also used by `rackscale capture`) |
| `trace_to_rsim.py` | recording to `.rpath` (runs `python/rackscale/nvbit/trace_to_rsim.py`) |
| `run_study.py`, `figures.py` | validation, what-ifs, ring attention, figures |
| `configs/` | GB300 NVL72, 9x HGX B300 over IB, 9x DGX A100 over IB |
| `calibration/` | the B300 and A100 calibration files used |
| `results/` | timing, ncu, simulator outputs, `study.json` |
