## Code Quality

- Run `cargo clippy --workspace` before committing Rust changes. Fix all warnings — do not suppress with `#[allow]`.
- Run `cargo test -p decibri` after any Rust changes to verify no regressions.
- Run `node tests/test-capture.js`, `node tests/test-api.js`, `node tests/test-output.js`, `node tests/test-vad-silero.js` after JS or napi binding changes.
- Run `npx vitest run` after browser source changes.
- Rebuild the native addon (`cd npm/decibri && npm run build`) after any Rust changes before running Node.js tests.

## API Compatibility

- The Node.js `Decibri` class API is frozen. Do not change constructor options, event names, method signatures, or error messages without explicit approval.
- Downstream consumers (mcp-listen, voxagent, Wake Word) depend on exact API compatibility. Any change must be tested against all three.
- Error messages are exact string matches. Do not rephrase them.

## Changelog

- Update `CHANGELOG.md` when adding features, fixing bugs, or making breaking changes.
- Add entries under the `## [3.0.0] - Unreleased` section in the appropriate subsection (Added, Changed, Fixed, Removed).
- Use [Keep a Changelog](https://keepachangelog.com) format. One bullet per change, concise.
