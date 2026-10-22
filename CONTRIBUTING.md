# Contributing

Thanks for helping. Before opening a PR:

```bash
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
maturin develop --release && pytest -q python/tests
```

Guidelines:

- **Every hardware number has a provenance tag** (`spec`, `ubench`, `est` or `calibrated`) in
  the preset `notes`. A PR that changes a number should say where the new value comes from.
- **Determinism is a feature.** Do not iterate a `HashMap` where the order can affect timing.
  Break ties by id, and keep rank-parallel code free of shared mutable state, except for
  memoization that is computed once.
- **Hot paths** (`sm.rs`, `mem.rs`, `engine.rs`) should not allocate per cycle. Profile with
  `cargo run --release --example bench_gemm`.
- **Speed work must not move results.** `crates/rsim-gpu/tests/golden.rs` and
  `crates/rsim-rack/tests/golden.rs` pin cycles and counter digests. A change meant to alter
  timing updates them in the same commit and says why (`RSIM_GOLDEN_PRINT=1` prints the new
  values). See "Simulator speed" in `docs/ARCHITECTURE.md`.
- **New instructions**: add the opcode to `rsim-sass/src/isa.rs::classify`, set its timing in
  `rsim-gpu/src/pipes.rs`, and add a parse test.
- **New hardware**: add a preset in `rsim-hw/src/presets.rs` with a datasheet test, plus a
  `configs/` example.
