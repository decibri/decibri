## Guardrails

**Claude Code must never:**
- Run `git commit`, `git push`, or push to any remote
- Run `npm publish` or `cargo publish`
- Create or push git tags
- Modify files in `.github/workflows/` without explicit approval in the same request

All commits, tags, and registry publishes are performed manually. If a task appears to require any of the above, stop and ask first.

**Claude Code is allowed to:**
- Stage changes with `git add`
- Inspect repo state with `git status`, `git diff`, `git log`
- Run tests, builds, linters, and formatters locally
- Modify any source file outside `.github/workflows/`

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

## CI

- `tests/test-ci.js` is the CI-safe test subset (~38 assertions, no hardware required). Run it with `node tests/test-ci.js`.
- Hardware tests (`test-capture.js`, `test-api.js`, `test-output.js`, `test-vad-silero.js`) require a microphone/speaker and are local-only — they do NOT run in CI.
- Browser tests (`npx vitest run`) mock browser globals and run in Node.js — safe for CI.
- CI runs on every push to main and on PRs. See `.github/workflows/ci.yml`.
- Release dry-run (`.github/workflows/release-dryrun.yml`) is manually triggered to validate multi-platform builds before tagging a release.

## Changelog

- Update `CHANGELOG.md` when adding features, fixing bugs, or making breaking changes.
- Add entries under the `## [3.0.0] - Unreleased` section in the appropriate subsection (Added, Changed, Fixed, Removed).
- Use [Keep a Changelog](https://keepachangelog.com) format. One bullet per change, concise.
