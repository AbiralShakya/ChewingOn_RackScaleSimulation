# Workload format

A workload is JSON or YAML. You can write it by hand or emit it from `rackscale.lower` or
`rackscale.capture`.

```json
{
  "name": "tp8 mlp",
  "model_flops": 1.2e15,
  "spmd": {"world": 8, "program": {"streams": [
    [ {"op": "kernel", "name": "up", "kernel": {"type": "gemm", "m": 4096, "n": 3584, "k": 8192}},
      {"op": "record", "event": "x"},
      {"op": "collective", "kind": "all_reduce", "bytes": 67108864, "group": "block:8", "sms": 16} ],
    [ {"op": "wait", "event": "x"},
      {"op": "kernel", "kernel": {"type": "fixed", "ns": 5000, "sms": 32}} ]
  ]}},
  "kernels": {"my_attn": {"type": "attention", "batch": 8, "heads_q": 64, "heads_kv": 8, "sq": 1, "skv": 8192, "d": 128}}
}
```

Instead of `spmd` you can give `"ranks": [{"streams": [...]}, ...]`, one entry per rank.

Keys are checked: a key no field takes is an error that names the closest valid key
(`unknown key \`spmd.program.streams[0][0].kernel.dtpye\` (did you mean \`dtype\`?)`), rather
than being dropped and silently running the default. System YAMLs and GPU `overrides` are
checked the same way.

### Memory

`"memory": {"weights": 1.76e10, "kv_cache": 5.4e9}` states the bytes each rank needs, by
category (any names). The report shows the fullest GPU against its HBM capacity and adds a
warning when a rank does not fit. With explicit `ranks`, `"memory_per_rank": [{...}, ...]`
gives one map per rank. `rackscale.lower` fills `memory` for the full model (weights in the
model dtype, KV cache, Mamba-2 state, activations, BF16 gradients and FP32 Adam state sharded
over DP), even when only a few `layers` are simulated; see `lower.memory`.

## Ops

| op | fields |
|---|---|
| `kernel` | `kernel` (see below), optional `name` |
| `collective` | `kind` (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`), `bytes` (per rank), `group`, `algo` (`auto`/`ring`/`nvls`/`direct`), `sms` (default 16) |
| `send` / `recv` | `bytes`, `peer`, `relative` (peer is an offset), `sms` (default 2) |
| `record` / `wait` | `event`; waits match records in order, CUDA-event style, across the rank's streams |
| `delay` | `ns` |
| `host` | `at_ns`: the stream's next op cannot start before this time from the start of the run (a host launch timeline, or a transfer issued on a clock) |
| `loop` | `times`, `body` |

**Groups.** A group is `"world"`, `"block:N"` (my aligned block of N ranks, e.g. TP),
`"stride:N"` (ranks with my `rank % N`, e.g. DP), or an explicit rank list.

## Kernels

| type | fields |
|---|---|
| `gemm` | `m n k dtype` |
| `grouped_gemm` | `ms` (rows per group), `n k dtype` |
| `attention` | `batch heads_q heads_kv sq skv d causal`; `d > 256` selects MLA (shared K/V latent) |
| `elementwise` | `n inputs elem_bytes flops_per_elem` |
| `row_norm` | `rows cols elem_bytes` |
| `sass` | `path kernel grid block regs smem branch_profile` |
| `trace` | `path` (Accel-Sim `.traceg`) |
| `path` | `path` (`.rpath` from the NVBit `warp_path` tool, see `research/flash_attention/`), optional `regs` |
| `fixed` | `ns sms` |
| `ref` | `name` from the `kernels` library |
