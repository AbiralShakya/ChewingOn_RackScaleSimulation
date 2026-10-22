# Install and quickstart

rackscale runs on Linux and macOS (Windows works through WSL). It needs no GPU: the simulator
is a CPU program. A GPU is only needed to *capture* a new workload or to calibrate new hardware.

## Install

=== "pip (prebuilt wheel)"

    ```bash
    pip install rackscale
    ```

    Wheels for Linux (x86-64, aarch64) and macOS (Apple silicon, Intel) are built by the
    release workflow for every tagged version, starting with the first release; until it is
    on PyPI, use the GitHub tab. They contain the whole simulator: the Python
    package, the `rackscale` command and the `rsim` command.

=== "pip from GitHub (latest main)"

    ```bash
    pip install git+https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation
    ```

    This compiles the Rust core, so it needs a Rust toolchain
    (`curl https://sh.rustup.rs -sSf | sh`). It takes about a minute on four cores.

=== "From a clone (development)"

    ```bash
    git clone https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation rackscale && cd rackscale
    python -m venv .venv && . .venv/bin/activate
    pip install maturin && maturin develop --release     # editable: Python edits apply at once
    ```

    Rebuild with `maturin develop --release` after changing Rust code. The standalone Rust
    binary is `cargo build --release -p rsim-cli` (at `target/release/rsim`).

=== "Rust only"

    ```bash
    cargo install --git https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation rsim-cli
    rsim presets
    ```

Check the install:

```console
$ rackscale --version
rackscale 0.1.0
$ rackscale presets
GPUs:
  a100_sxm   A100 SXM4 80GB               108 SMs @ 1410 MHz     312 TFLOPS bf16   2039 GB/s HBM  NVLink3 12x25 GB/s
  a100_pcie  A100 80GB PCIe               108 SMs @ 1410 MHz     312 TFLOPS bf16   1935 GB/s HBM  NVLink3 12x25 GB/s
  h100_sxm   H100 SXM5 80GB               132 SMs @ 1830 MHz     989 TFLOPS bf16   3350 GB/s HBM  NVLink4 18x25 GB/s
  h800       H800 SXM                     132 SMs @ 1830 MHz     989 TFLOPS bf16   3350 GB/s HBM  NVLink4 8x25 GB/s
  h200_sxm   H200 SXM 141GB               132 SMs @ 1830 MHz     989 TFLOPS bf16   4800 GB/s HBM  NVLink4 18x25 GB/s
  b200       B200 (HGX, 1000 W)           148 SMs @ 1856 MHz    2250 TFLOPS bf16   7700 GB/s HBM  NVLink5 18x50 GB/s
  gb200      B200 in GB200 (1200 W)       148 SMs @ 2062 MHz    2500 TFLOPS bf16   8000 GB/s HBM  NVLink5 18x50 GB/s
  b300       NVIDIA B300 SXM6 AC          148 SMs @ 2032 MHz    2464 TFLOPS bf16   7672 GB/s HBM  NVLink5 18x50 GB/s
  gb300      GB300 (NVIDIA B300 SXM6 AC)  148 SMs @ 2032 MHz    2464 TFLOPS bf16   7672 GB/s HBM  NVLink5 18x50 GB/s
switches: nvswitch3, nvswitch_nvl5, ib_ndr_leaf, ib_ndr_sharp
systems:  hgx_h100_8, hgx_b200_8, hgx_b300_8, hgx_h800_8, gb200_nvl72, gb300_nvl72, h100_2node_ib, h100_8node_ib, h100_8node_ib_sharp, h800_2node_ib, h800_4node_ib, h800_8node_ib, hgx_b200_2node_ib, gb200_nvl72_2rack_ib
```

## Five minutes with rackscale

### 1. The demo (30 seconds)

```bash
rackscale demo --open
```

It simulates one decode step of Llama-3-70B (2 of 80 layers, TP8, 32 sequences at 4k context)
on HGX H100 and HGX B200, cycle level on every SM, and prints a comparison:

```
system             step  GPU idle  tensor idle   top kernel
hgx_h100_8     703.2 us     22.4%        82.0%   sm90_gemm_bf16_32x7168x8192_tile128x256
hgx_b200_8     323.9 us     30.1%        78.7%   sm100_gemm_bf16_32x16032x8192_tile128x128
```

then the full anatomy of the B200 run and a viewer page (`rackscale_demo.html`) that zooms from
the rack down to SASS. `--open` opens it in your browser.

### 2. Any model on any rack

```bash
rackscale models                     # the built-in architectures
rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 4 --fast
```

```
== deepseek-v3 decode b64 ctx4096 Parallel(tp=1, ep=72, ...) on GB200 NVL72 (72 GPUs) ==
makespan 1556.5 us   MFU 12.5%
memory (rank 0): 70.8 of 186 GB  [kv_cache 18.4, weights 52.4]
GPU idle 8.4%   tensor-core idle in kernels 62.4%
scheduler-cycle anatomy:
  async_wait            59.95%  ##############################
  tensor_wait           17.51%  #########
  tma_wait              10.57%  #####
  ...
kernel                                                       n   total_us   TFLOPS     GB/s    tc%
sm100_fmha_decode_b64_hq128_hkv1_sq1_skv4096_d576          288    27738.6    802.7     3527   56.2
sm100_gemm_bf16_64x129280x7168_tile128x256                  72    18909.9    903.3     7197   36.1
...
```

`--layers 4` simulates the first four layers (three dense, one MoE) and still checks memory for
the whole model. `--fast` simulates each distinct kernel once and reuses it across ranks and
layers (within about 1% of exact on these runs); drop it to co-simulate every rank.

Add `--viz ds.html --open` for the viewer, `--trace ds.json` for a Perfetto timeline, and
`--report ds.json` for the full JSON report.

### 3. From Python

```python
import rackscale as rs
from rackscale import lower, models

wl = lower.decode_step(models.LLAMA3_70B, rs.Parallel(tp=8), batch=32, context=4096, layers=2)
for system in ["hgx_h100_8", "hgx_b200_8"]:
    rep = rs.simulate(system, wl, fast=True)
    print(system, f"{rep.makespan_us:.1f} us", f"GPU idle {rep.bubbles['gpu_idle_fraction']:.1%}")
print(rep.summary())
```

### 4. One kernel, down to the scheduler cycle

```bash
rackscale kernel --gpu b200 gemm 8192 8192 8192
```

### 5. Your own hardware

```bash
cp configs/gb200_nvl72.yaml my_rack.yaml       # edit: clocks, HBM, NVLink ports, switches
rackscale system my_rack.yaml
rackscale decode llama3-70b --system my_rack.yaml --tp 8 --layers 2 --fast
```

A misspelled key is an error that names the closest valid key, not a silent no-op.

## Where next

- [Tutorials](tutorials/single-kernel.md): one kernel, an NVL72 rack run, capture and replay, agents.
- [Examples gallery](examples.md): every runnable script and viewer page.
- [The guide](book/README.md): GPU performance from first principles, chapter by chapter.
- [Command-line reference](reference/cli.md) and [Python API](reference/python.md).

## Troubleshooting

| symptom | fix |
|---|---|
| `rackscale native extension not built` | You are importing the source tree without building it. Run `maturin develop --release` in the clone (or `pip install .`). |
| `error: can't find Rust compiler` during `pip install` | No wheel for your platform yet; install Rust (`curl https://sh.rustup.rs -sSf \| sh`) and retry. |
| A run takes minutes | Add `--fast`, and use `--layers N` to simulate a few layers. Exact NVL72 runs fold identical ranks automatically. |
| `WARNING: ... would not fit` | The model's weights, KV cache and activations exceed HBM for that layout. Raise `--tp`/`--ep` or lower `--batch`/`--context`. |
| Colours in logs | Set `NO_COLOR=1`. Output is plain automatically when piped. |
