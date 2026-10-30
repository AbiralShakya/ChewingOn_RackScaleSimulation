# Rack-scale questions: an LLM with and without rackscale

Nineteen questions a performance engineer gets asked: how long a kernel takes, which layout to
serve a model with, why a step is slow, what a GPU we don't have would do, what a power cap or a
slow link costs. Claude answers each one twice through the Claude Code CLI, headless:

- **with tools**: only the rackscale MCP server (`rackscale-mcp`). No shell, files or web, so every
  number comes from the model or from the tools. The system prompt adds one sentence saying the
  tools exist.
- **without tools**: nothing. The model answers from what it knows.

```bash
maturin develop --release && pip install mcp
python benchmarks/agent_qa/make_refs.py            # reference answers -> refs.json (~10 min)
python benchmarks/agent_qa/run.py --model sonnet   # both conditions, then grade -> results/<model>/summary.md
```

## Where the reference answers come from

Each question names its `truth`, and the three kinds test different things:

| truth | questions | what it tests |
|---|---|---|
| `measured` | 4 | real B300 timings (cuBLAS GEMMs and FlashAttention decode, CUDA-graph timing, from `research/ssm_vs_attention`). The simulator can be wrong here too, so it is the fair test. |
| `simulated` | 11 | the cycle-level simulator (`make_refs.py`, or the book's simulated tables). The tool is graded against its own instrument, so this measures whether an LLM can *get* the simulator's number and whether a model without it lands nearby. It does not show that the simulator is right; its error bands (`accuracy`) say how far to trust it. |
| `closed_form` | 4 | formulas with no free parameters (waves, pipeline bubble, KV bytes, balls into bins). A model without tools can derive them. |

A number scores 1 within its tolerance (15% for kernel times, 5% for speedups, 2-3% for closed
forms, 20-30% for noisier quantities), 0.5 within twice that, else 0. A choice scores 1 or 0. A
question's score is the mean over its fields.

## Results

Claude Sonnet 5.5, 2026-10-30. The full table, with every answer beside its reference, is in
[`results/sonnet/summary.md`](results/sonnet/summary.md).

| | with tools | without tools |
|---|---|---|
| total (of 19) | **16.3** | 13.7 |
| measured hardware (of 4) | 2.5 | **4** |
| simulated (of 11) | **9.8** | 5.7 |
| closed form (of 4) | 4 | 4 |
| cost | $0.96 | $0.34 |

What it shows:

- **Mechanism questions are where the tools pay.** Without tools, Sonnet guessed that 1.5x HBM
  speeds a 70B decode 1.27x (the simulator says 1.05x: the kernels turn latency- and ALU-bound).
  It also called a step "idle" when all-reduces dominated, put the power-cap cost at 14% (9%), and
  a half-rate NVLink at +3% for NVLS (+50%). With the tools it got all four, each with the reason.
- **On measured kernel times the model alone did better.** Sonnet's own estimates of B300 GEMM and
  attention times were all within 9%. Through the tools it reported the simulator's numbers, which
  are 33% fast on the 256 x 18560 x 4096 GEMM and 17% fast on the 1024-row one. That is the
  simulator's known weak spot on mid-sized GEMMs (now in its error bands), and the honest headline
  is that a calibrated simulator does not beat a strong model's priors on single-kernel times.
- **Closed forms are a tie.** Both conditions derive waves, bubbles, KV bytes and ECMP exactly.
- **Two misses with tools come from using them badly.** On the layout question the agent stopped
  waiting for `best_layout` (minutes) and used the closed-form ladder, which over-predicts decode
  throughput, so it picked batch 256. On the NVL72 all-reduce it did not find a direct tool and
  estimated by hand.

Caveats: one model, one sample per question, and most references are the simulator's own numbers,
so "with tools" partly measures whether the model can read the instrument. The measured rows are
the only ones that test the simulator itself.

## How the benchmark improved the tools

The first round (with tools 15.0, without 13.7) exposed problems in the tools themselves. Each one
was fixed, and every with-tools question was asked again for the table above:

- **Timeouts.** Claude Code gives an MCP tool call 60 s. `best_layout` and full-model simulations
  take minutes, so the agent fell back to closed forms and got the plan question wrong. The runner
  now sets `MCP_TOOL_TIMEOUT`, and the docs tell users to do the same.
- **"Idle" counted time inside all-reduces.** The simulator's `gpu_idle_fraction` counts every
  moment without a compute kernel, collectives included. The bottleneck therefore said "idle"
  where communication dominated. Summaries now carry a `time_split` that partitions GPU time, and
  the bottleneck uses it.
- **The launch-gap default was undocumented.** The agent set a 5 µs gap to "model eager launches"
  when the default already is the eager overhead (1.7 µs on B300). The parameter now says so.
- **Link-fault closed forms read as the answer.** `tail_latency` gave 2x (the bandwidth-term bound)
  beside a note of +50% simulated, and the agent reported 2x. The description now says the closed
  form bounds the bandwidth term only, and `simulate=true` with `allreduce_mib` gives the number.
- **Error bars on mid-sized GEMMs were too tight.** The 256 x 18560 x 4096 GEMM is 33% fast in the
  simulator against cuBLAS, outside a flat 12% band. GEMM bands now follow the measured size
  classes (3.9% / 19.1% / 11.6% median |error| for decode / mid / large).

## Files

- `questions.py`: the questions, reference functions, fields and tolerances
- `make_refs.py`: computes `refs.json`
- `run.py`: runs both conditions with `claude -p` and grades; `--grade-only` regrades
- `results/<model>/<condition>/<id>.json`: each answer with its full text, turns, cost and time
- `results/<model>/summary.md`: the graded table
