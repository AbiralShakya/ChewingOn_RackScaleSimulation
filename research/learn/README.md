# NVL72 Scaling Lab

A single-page companion to book chapters 6 and 11 to 13: pick a model, a phase, a plan and a
batch, and the page recomputes every closed form of chapter 11 in the browser, next to the
replay and simulator numbers it is checked against. Eight views:

| view | what it shows | numbers from |
|---|---|---|
| Plan | step time as a sum of terms (compute, weight reads, KV, each collective), MFU, memory, and what bounds it | formulas, live |
| 1 GPU to rack | the best plan on 1, 2, 4, 8, 16, 36 and 72 GPUs, for training and decode | `scaling.ladder` |
| Collectives | ring and NVLS all-reduce step by step, and every collective against its formula from 1 KiB to 1 GiB | `rsim net sweep` |
| Roofline | arithmetic intensity, the ridge at C/β, and the power-capped clock | formulas |
| Pipelines | GPipe, 1F1B and ZB-H1 timelines of the same step | `scaling.replay` |
| Formula vs sim | the 22 parallelism cases of chapter 11: textbook, replay and simulated | `research/parallelism` |
| Many models | jobs sharing an NVL72, on a map of the rack | `research/colocation` |
| Tokens vs bytes | H-Net against BPE and byte Transformers on one B300, simulated and in closed form | `research/bytes` |

The published page is [`docs/viz/scaling-lab.html`](../../docs/viz/scaling-lab.html). To
rebuild it after any of those studies changes:

```bash
python research/learn/net_sweep.py   # data/net_sweep.json (needs the rsim CLI)
python research/learn/export.py      # data/learn.json: model tables, constants, study results
python research/learn/build.py       # dist/scaling-lab.html and docs/viz/scaling-lab.html
python research/learn/verify.py      # the page's JavaScript closed forms against rackscale.scaling
```

`rsmath.js` is a line-for-line port of `rackscale.scaling`; `verify.py` draws 600 random
models, phases and plans and fails on any relative difference above 1e-9. The page has no
dependencies and works offline.
