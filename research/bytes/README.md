# Tokens vs bytes: H-Net on one B300

`run.py` compares H-Net (byte-level, dynamic chunking; 1-stage and 2-stage XL) with a BPE
Transformer and a byte-level Transformer of the same main-network shape, per byte of text:

- **flops**: FLOPs per byte against the H-Net paper's Table 2 (signed error);
- **prefill**: 8 prompts of 8 KiB on one B300;
- **decode**: batch 64 at 8 KiB of context, H-Net with its routers sampled per step, the same
  draw spread evenly over the steps, and the expected (mean-field) batch;
- **batch**: decode cost per byte at batch 16, 64 and 256;
- **closed**: the bytes each decode step moves, in closed form (`hnet.decode_traffic`), and the
  time per byte at HBM bandwidth, at the same batches. No simulation; it runs in a second.

Every simulated run is also split by part of the network (outer stages, main-network
attention, matmuls and elementwise ops, LM head, launch gaps) from the simulator's per-op trace.
Results are in [`results/bytes.md`](results/bytes.md); the write-up is
[book chapter 13](../../docs/book/13-tokens-or-bytes.md).

```bash
python research/bytes/run.py                 # all studies, ~40 min on 4 cores
python research/bytes/run.py --only decode   # one study, keeping the others' results
python research/bytes/run.py --only closed   # the closed form only, and rewrite results/bytes.md
```

Simulated on the calibrated B300 preset in fast mode. No GPU was used, and no H-Net was run on
hardware. Head counts and FFN widths are not in the paper; we use 128-wide heads and an 8/3
SwiGLU FFN.
