# Capturing real engines

rackscale reads Kineto/Chrome traces from the PyTorch profiler. Every major engine can produce
one. Enable `record_shapes=True` so that GEMM shapes can be recovered.

| Engine | How to profile |
|---|---|
| vLLM | `VLLM_TORCH_PROFILER_DIR=/tmp/prof vllm serve ...`, then `POST /start_profile` and `/stop_profile` around a few requests |
| SGLang | `--enable-profile` or `POST /start_profile` (writes to `SGLANG_TORCH_PROFILER_DIR`) |
| training / custom | `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True)` and `prof.export_chrome_trace(...)` |

Then run:

```python
from rackscale import capture
import rackscale as rs
wl = capture.from_file("rank0.pt.trace.json.gz", world=8, collective_group="block:8")
print(rs.simulate("gb200_nvl72", wl, fast=True).summary())
```

## Lowering rules

- `aten::mm/addmm/bmm/linear/matmul/_scaled_mm` kernels with recoverable `Input Dims` become a
  synthetic `gemm` and are re-simulated on the target GPU. This makes "same engine, different
  hardware" questions answerable.
- `nccl*Kernel_<Kind>_*` kernels become collectives. Their sizes come from `record_param_comms`
  metadata, which recent PyTorch emits. Without that metadata they fall back to a fixed duration.
- Everything else, including custom Triton and CUDA kernels, becomes a `fixed` kernel with its
  measured duration (`time_scale` can rescale it). It occupies as many SMs as its grid.
- `mode="replay"` keeps every kernel as measured, so only launch gaps, overlap and
  communication are re-simulated.

Streams are preserved. One rank's trace is replicated SPMD-style over `world` ranks.

To get cycle-level detail on a specific kernel (for example a FlashInfer or DeepGEMM binary),
dump its SASS (`cuobjdump -sass`) and reference it as a `sass` kernel, or capture it with
NVBit/Accel-Sim tracing and use a `trace` kernel.
