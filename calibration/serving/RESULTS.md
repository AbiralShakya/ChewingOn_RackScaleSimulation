# Published serving benchmarks vs the simulator

Simulator commit `1161306`, run 2026-10-30T18:28:52+00:00. Error = (predicted − measured) / measured on output (or input) tokens/s per GPU: positive means the simulator predicts more throughput than was published. Sources and configs: [published.json](published.json).

| case | engine, hardware | measured tok/s/GPU | predicted | error | how |
|---|---|---:|---:|---:|---|
| [sglang-h100-decode](https://lmsys.org/blog/2025-05-05-large-scale-ep/) | SGLang (PD disaggregation, large-scale EP, two-batch overlap), 9 nodes x 8 H100, InfiniBand (decode side of 12 nodes) | 2,785 | 2,710 | -2.7% | one decode step: 94460 us |
| [sglang-h100-prefill](https://lmsys.org/blog/2025-05-05-large-scale-ep/) | SGLang (PD disaggregation, large-scale EP), 4 nodes x 8 H100, InfiniBand (prefill side) | 7,417 | 4,021 | -45.8% | one prefill step: 4074963 us |
| sglang-gb200-decode | SGLang (PD disaggregation, large-scale EP), GB200 NVL72, 48 decode ranks | 9,087 | — | — | not simulated: decode: does not fit in HBM |
| [trtllm-b200-r1-fp4](https://nvidia.github.io/TensorRT-LLM/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.html) | TensorRT-LLM (in-flight batching, attention DP, CUDA graphs), 8 x B200 (one node) | 4,548 | 8,868 | +95.0% | batch / (t_decode + batch/OSL * t_prefill) = 384 / (36036 + 384/2048 x 38756) us |
| [trtllm-h200-r1](https://nvidia.github.io/TensorRT-LLM/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.html) | TensorRT-LLM (in-flight batching, attention DP, CUDA graphs), 8 x H200 (one node) | 1,436 | 2,401 | +67.2% | batch / (t_decode + batch/OSL * t_prefill) = 128 / (46888 + 128/2048 x 102706) us |
| [deepseek-production-decode](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) (soft) | DeepSeek's own inference system, 18-node H800 decode units (8 GPUs per node) | 1,850 | 1,620 | -12.4% | one decode step: 54317 us |

Primary cases: 4, median |error| 56.5%, signed errors -2.7%, -45.8%, +95.0%, +67.2%.

## Second references

- sglang-h100-decode no overlap: 2,408 tok/s/GPU (-13.6%)
- sglang-h100-prefill against the alternative published figure 6,288: -36.1%
- trtllm-b200-r1-fp4 decode only: 10,656 tok/s/GPU (+134.3%)
- trtllm-b200-r1-fp4 with attention-DP synchronisation (post hoc, same simulated steps): 5,695 tok/s/GPU (+25.2%)
- trtllm-h200-r1 decode only: 2,730 tok/s/GPU (+90.1%)
- trtllm-h200-r1 with attention-DP synchronisation (post hoc, same simulated steps): 1,449 tok/s/GPU (+0.9%)

## What the misses say (read from the simulated breakdowns)

- **SGLang H100 decode, -2.7%.** The one case run exactly as published (PD-disaggregated decode,
  two-batch overlap) lands within 3%. Without two-batch overlap the simulator gives -13.6%, so
  the overlap is doing real work in the published number.
- **SGLang H100 prefill, -45.8%.** Communication is 2.57 s of the simulated 4.07 s step. The
  lowering sends every token to each of its top-8 experts as a plain all-to-all
  (`python/rackscale/lower.py`, `moe_dispatch`). DeepSeek-V3 limits each token to 4 nodes and
  DeepEP sends it once per node over InfiniBand, then fans it out over NVLink; SGLang also overlaps
  the two micro-batches in prefill. Neither is modelled, so the simulator is pessimistic here.
  This is a lowering gap, not a fabric one.
- **TensorRT-LLM, +95% (B200) and +67% (H200).** The decode step alone is close to what the
  hardware can do; the miss is in how prefill shares the iterations. With attention DP, all 8
  ranks wait for any rank that is prefilling, so prefill costs far more than its token share.
  Accounting for that with the same simulated steps gives +0.9% on H200 and +25% on B200. This
  composition was added after seeing the first result, so it is reported as post hoc. The frozen
  errors above stay the headline.
- **SGLang GB200 decode, not simulated.** 1,408 sequences of 2,048 tokens need 203 GB of BF16 MLA
  KV cache per GPU, more than a GB200 holds, so the published run must have used a shorter
  average KV length or an FP8 cache. The post does not say which, so the case is left unscored.
- **DeepSeek production, -12.4% (soft).** A 24-hour average with load swings and an inferred batch;
  shown for scale only.

