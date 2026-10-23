## What and why

<!-- What does this change, and what question or problem motivated it? -->

## How it was checked

- [ ] `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace --release`
- [ ] `pytest -q python/tests`
- [ ] Timing changes: golden tests updated in this PR with the reason, or results are bit-identical
- [ ] New or changed hardware numbers carry a provenance tag (`spec`, `ubench`, `est`, `calibrated`)
- [ ] Accuracy changes: before/after error against a measurement

## Results

<!-- For model or accuracy changes: numbers before and after, and the command that produces them. -->
