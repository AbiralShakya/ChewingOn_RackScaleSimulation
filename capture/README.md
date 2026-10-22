# capture/: record any CUDA program for rsim

`nvbit/` holds two [NVBit](https://github.com/NVlabs/NVBit) tools that are injected into an
unmodified program with `CUDA_INJECTION64_PATH`. They see every kernel that reaches the driver,
whatever produced it: PyTorch, vLLM, SGLang, Triton, CuTe DSL, CUTLASS, cuBLAS, ThunderKittens,
plain CUDA.

| tool | records | used by |
|---|---|---|
| `rsim_capture.so` | the launch graph of a whole program (every launch with its configuration and stream, event records and stream waits, copies, host and GPU times) in `graph` mode; the per-warp path of every distinct kernel in `paths` mode | `rackscale capture` |
| `warp_path.so` | the per-warp path of one chosen launch (`WP_KERNEL`, `WP_LAUNCH`) | `calibration/scripts/path_reference.py`, `research/flash_attention` |

Both share the instrumentation in `wp_instrument.h`: for a few sampled CTAs, every dynamic SASS
instruction of every warp, with lane addresses, mbarrier operands, TMA source runs and tcgen05
descriptors. `rackscale.nvbit.trace_to_rsim` converts a recording into an `.rpath` that
`rsim` replays at cycle level (see `crates/rsim-sass/src/path.rs`).

Build against an NVBit 1.8+ release:

```bash
NVBIT=/path/to/nvbit_release_x86_64 capture/nvbit/build.sh
# -> $NVBIT/tools/build-rsim/{rsim_capture,warp_path}.so
```

The workflow is documented in [docs/CAPTURE.md](../docs/CAPTURE.md).
