<!--
Style: technical and structural, not narrative. Cite specific files and symbols.
Delete sections that don't apply (e.g., a typo fix doesn't need a Validation block).
-->

## Summary

<One sentence: what does this PR do, and why?>

## Changes

<Per-file or per-area list. Cite specific files and symbols.

Example:
- `crates/decibri/src/vad.rs`: `SileroVad::process()` now returns `Result<VadResult, DecibriError>` (was `Option<VadResult>`); callers updated in `bindings/python/src/lib.rs:142` and `bindings/node/src/lib.rs:88`.
- `bindings/python/CHANGELOG.md`: new entry under `[Unreleased]`.
-->

## Validation

<List the gates that ran green. Drop the ones that don't apply.

- `cargo build --release --workspace`
- `cargo clippy --workspace -- -D warnings`
- `cargo fmt --all -- --check`
- `cargo test-decibri`
- `cd bindings/python && uv run mypy --strict && uv run pytest`
- `cd bindings/python && uv run maturin develop --uv` (if Rust changes touch the wheel)
- `node tests/test-ci.js`
- `npx vitest run` (if browser sources changed)
- Em-dash sweep: 0 hits across modified files
-->

## Related

<Link to issue, master plan section, or backlog entry. Optional.>
