# Every parallelism: formulas vs simulator

`check_math.py` runs 22 parallel plans on a GB300 NVL72 three ways (closed-form estimate,
roofline replay of the lowered program, cycle-level simulation) and writes signed errors to
[`results/check.md`](results/check.md). The write-up is
[book chapter 11](../../docs/book/11-every-parallelism.md).

```bash
python research/parallelism/check_math.py           # ~40 min on 4 cores, resumable
python research/parallelism/check_math.py --table   # rebuild the table from results/check.jsonl
```

All multi-GPU numbers are simulated on presets (calibrated B300 die, NVLink 5 fabric from
spec). No GPU was used.
