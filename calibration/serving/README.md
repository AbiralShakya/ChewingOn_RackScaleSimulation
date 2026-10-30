# Published serving benchmarks as holdouts

Node- and rack-scale serving numbers we cannot measure ourselves, taken from the engines' own
posts and simulated at the published configuration. None of them was used to fit anything.

| file | what |
|---|---|
| [`published.json`](published.json) | each case: source URL, date, engine, hardware, configuration, the stated number, and any arithmetic done on it |
| [`predict.py`](predict.py) | simulates every case and writes `predictions.json` and `RESULTS.md` with the simulator commit |
| [`RESULTS.md`](RESULTS.md) | signed errors per case |
| [`../../configs/serving/`](../../configs/serving/) | the systems the cases ran on (multi-node H100/H800 over 400G InfiniBand, HGX H200) |

```bash
python calibration/serving/predict.py                  # every case, ~1 h on 4 cores
python calibration/serving/predict.py --only <id> ...   # rerun some (only at the same commit)
```

## Rules

- **A case goes in only if its configuration is stated.** Model, GPU count and type, parallelism,
  batch per GPU (or concurrency and max batch), sequence lengths and precision. The vLLM
  "2.2k tok/s/H200" post, for example, states none of these in its text and is left out.
- **Copy, then derive in the open.** `measured` is the published number, converted to tokens/s
  per GPU; the conversion is written in `derived`.
- **No tuning against these.** If a comparison leads to a model change, `RESULTS.md` keeps the
  error at the commit the case was first simulated with, beside the new one.
- **Soft cases are context.** DeepSeek's production statistics are a 24-hour average with unknown
  batch and MTP use. They are simulated and shown, but not counted in the median.

## How a case becomes a simulated step

PD-disaggregated engines (SGLang's large-scale EP posts, DeepSeek) run decode and prefill on
separate GPUs, so one simulated decode or prefill step at the stated batch and KV length gives
tokens/s per GPU directly.

In-flight batching (`trtllm-bench`) runs prefill inside the decode iterations. With ISL/OSL
tokens per request, each decode step of a `batch`-sequence rank also prefills `batch / OSL`
prompts on average, so the time per output token is `t_decode + batch/OSL × t_prefill(1 prompt)`.
Running the two separately is slightly pessimistic (a fused mixed batch shares weight reads), so
the decode-only figure is printed next to it as the optimistic bound.

## What the simulator does not model here

- Expert load imbalance: tokens are routed uniformly, which is what SGLang's "simulated perfect
  EPLB" approximates. Real routing is skewed, which favours the simulator.
- Scheduler, HTTP and sampling overhead on the CPU, and PD KV transfer.
- MTP / speculative decoding (no scored case uses it).
- FP8 attention and FP8 KV (attention runs in BF16).
