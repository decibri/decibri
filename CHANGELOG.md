# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.3.2] - Unreleased

### Changed

- **`crates/decibri/build.rs` rewrite to use Cargo.toml as primary source.** The previous build script (introduced in 3.3.1 to source the cpal version from a single point of truth) read the workspace `Cargo.lock` via a path traversal that worked in workspace builds but panicked in `cargo publish` verify because the published tarball is flat (`decibri-X.Y.Z/Cargo.toml` and `decibri-X.Y.Z/Cargo.lock` are siblings, not in workspace structure). The traversal landed at `target/Cargo.lock` (nonexistent) and aborted the build. **3.3.1's crates.io publish failed on this defect; 3.3.1 shipped to npm but not to crates.io.** 3.3.2 fixes the build.rs to read `CARGO_MANIFEST_DIR / Cargo.toml` directly, with belt-and-suspenders fallbacks: env-var override (`DECIBRI_CPAL_VERSION`), workspace `Cargo.toml` fallback for `{ workspace = true }` inherit form, hardcoded `"0.17"` constant fallback (with `cargo:warning=`) for unforeseen build contexts. Cargo.toml is unambiguously present in every build context because cargo guarantees `CARGO_MANIFEST_DIR` always points at the manifest's directory.
- **No functional change to user-visible API or error messages.** All 5 message refinements from 3.3.1 (`SampleRateOutOfRange`, `FramesPerBufferOutOfRange`, `AlreadyRunning`, `OrtInitFailed`, `OrtLoadFailed` / `OrtPathInvalid`, `PermissionDenied`) are preserved unchanged.
- **`decibri::CPAL_VERSION` byte-identity preserved across all four cargo-emitted dep forms.** Verified 2026-04-26 by walking through `find_cpal_in_dependencies` + `truncate_to_major_minor` against on-disk Cargo.toml content: Form 1 (`cpal = "0.17"` workspace.dependencies) -> `"0.17"`; Form 2 (`cpal = { version = "0.17", optional = true }` hypothetical inline-table) -> `"0.17"`; Form 3 (`cpal = { workspace = true, optional = true }` source crate) -> workspace fallback -> `"0.17"`; Form 4 (`[dependencies.cpal] version = "0.17"` published normalized) -> `"0.17"`. All four forms produce identical output to the v3.3.1 build.rs's Cargo.lock-resolved truncation because the workspace pin is at major.minor granularity already (`cpal = "0.17"` in `[workspace.dependencies]`).

### Migration notes

- **Direct Rust crate consumers** of decibri: 3.3.1 was never published to crates.io. crates.io's decibri version history is 3.3.0 -> 3.3.2; 3.3.1 is skipped entirely on this registry. **npm consumers** see the standard 3.3.0 -> 3.3.1 -> 3.3.2 progression (3.3.1 shipped successfully on npm; only the cargo publish step in `release.yml` failed). The asymmetry is documented here for archaeological clarity.
- All 3.3.1 message refinements (per `[3.3.1]` entry above) are preserved in 3.3.2. Consumer-side migration from 3.3.0 to 3.3.2 is the same as 3.3.0 to 3.3.1 from the user-visible API perspective.
- **No npm migration required** for 3.3.1 -> 3.3.2. 3.3.2 publishes a new wheel set with the same user-visible behavior; bump-and-rebuild is sufficient.

## [3.3.1] - 2026-04-25

### Changed

- **Audience-neutral error message pass.** Five `DecibriError` `Display` strings refined to remove cross-binding and platform-specific awkwardness. No variant identity, layout, or count change; no API-surface change. Strict patch release.
  - `SampleRateOutOfRange`: `"sampleRate must be between 1000 and 384000"` -> `"sample rate must be between 1000 and 384000"`. The previous camelCase form was Node-API-targeting and matched no other field-name convention in the rest of the Rust crate (snake_case fields throughout, natural-language rustdoc voice).
  - `FramesPerBufferOutOfRange`: `"framesPerBuffer must be between 64 and 65536"` -> `"frames per buffer must be between 64 and 65536"`. Same rationale.
  - `AlreadyRunning`: `"Decibri is already running. Call stop() first."` -> `"audio stream is already running. Call stop() first."`. Hardcoded class name was misleading when raised from `DecibriOutput`; "audio stream" matches the existing `error.rs` vocabulary ("capture stream", "output stream", "audio stream").
  - `OrtInitFailed`: `"Either pass ort_library_path in VadConfig, ..."` -> `"Either pass ort_library_path when constructing the VAD, ..."`. Drops Rust-internal `VadConfig` type reference; phrasing now correct for Node, Python, and direct-Rust crate consumers alike.
  - `OrtLoadFailed` and `OrtPathInvalid`: `"the bundled ORT may be missing from your platform package"` -> `"the bundled ONNX Runtime may be missing from your installation"`. Drops npm-internal "platform package" phrasing; "installation" works for npm platform packages, Python wheels, and direct-Rust crate use.
  - `PermissionDenied`: macOS-specific `"Enable in System Preferences > Security & Privacy."` replaced with attribute-gated per-platform guidance. macOS hint extended to specifically reference `> Microphone`. Windows hint references the modern `Settings > Privacy & Security > Microphone` UX. Linux hint references PulseAudio / PipeWire (the user-facing audio control layer over cpal's ALSA backend).
- Lockstep updates to `bindings/node/src/lib.rs` (one duplicate `AlreadyRunning` message in the napi `start()` pre-running check), `npm/decibri/src/errors.js` (two prefix-match strings in the typed-error shim), `npm/decibri/src/decibri.js` (two thrown messages in client-side validation; line 101's `channels` message is already natural-language and unchanged), `npm/decibri/src/decibri-output.js` (one thrown message), `npm/decibri/src/browser/decibri-browser.js` (two thrown messages in browser-side validation), `tests/test-ci.js`, `tests/test-api.js`, `tests/test-output.js` (15 hardcoded message-substring assertions).

### Migration notes

Error message wording on shipped `DecibriError` variants has historically been stable across the 3.x line. 3.3.1 explicitly refines five messages to remove audience-leak issues identified during P3 Phase 2 planning: Node-API-targeted camelCase parameter names, class-name hardcoding in `AlreadyRunning`, a Rust-internal type reference (`VadConfig`) in `OrtInitFailed`, npm-internal phrasing ("platform package") in `OrtLoadFailed` / `OrtPathInvalid`, and macOS-only platform guidance in `PermissionDenied`. Future releases re-establish wording stability; 3.3.1 is the consolidation point.

- **Direct Rust crate consumers** asserting on `DecibriError::Display` strings should update assertions for the five refined messages. Type-level matching on `DecibriError` variants is unaffected; only string-text assertions need updating.
- **Node consumers** using `e.message.includes(prefix)` or `e.message.startsWith(prefix)` patterns should update for `sampleRate` -> `sample rate`, `framesPerBuffer` -> `frames per buffer`, and the `AlreadyRunning` message text. Type-level matching on `RangeError` / `TypeError` is unaffected.
- **Python consumers**: P3 Phase 2's first wheel (still pre-release) consumes `decibri@3.3.1`, so the messages it sees are the corrected forms from day one. No migration needed (Phase 1 wheel was scaffold-only with no error message surface).

## [3.3.0] - 2026-04-23

Groundwork release for upcoming P3 Python bindings. Adds a stable-ID form
for audio device selection (`DeviceSelector::Id`), fixes a long-standing
direction bug in `DecibriError::DeviceNotFound` when resolving output
devices, exposes both in the Node binding, and extends the reference
documentation with a Cargo feature flag guide plus additional crate-level
rustdoc. No Node.js or browser API break. Direct Rust crate consumers
pattern-matching on `DeviceSelector` or struct-literal-constructing
`DeviceInfo` / `OutputDeviceInfo` need to update for the new
`#[non_exhaustive]` attributes (see Migration notes below).

### Changed

- `DeviceSelector`, `DeviceInfo`, and `OutputDeviceInfo` are now
  `#[non_exhaustive]`. External Rust consumers pattern-matching on
  `DeviceSelector` must add a `_ =>` catch-all arm; consumers
  constructing `DeviceInfo` or `OutputDeviceInfo` via struct literal
  from outside the crate must switch to reading fields off instances
  returned by `enumerate_input_devices` / `enumerate_output_devices`.
  Field names and display strings are unchanged. This future-proofs
  the API: subsequent variant or field additions are source-compatible
  for consumers who include the catch-all.

### Added

- `DeviceSelector::Id(String)` for selecting audio devices by stable
  per-host identifier (WASAPI endpoint ID on Windows, CoreAudio UID on
  macOS, ALSA pcm_id on Linux). Unlike `DeviceSelector::Name`
  (case-insensitive substring) and `DeviceSelector::Index` (positional),
  `Id` survives across enumerations: display names can shift when other
  devices are plugged in but per-host IDs do not.
- `id: String` field on `DeviceInfo` and `OutputDeviceInfo`, populated
  from `cpal::DeviceId`'s `Display` output. Empty string if cpal cannot
  produce a stable ID for a given device (rare; some host backends
  cannot assign IDs to every enumerated device). Obtain the ID from
  these fields and pass to `DeviceSelector::Id`.
- `DecibriError::OutputDeviceNotFound(String)` variant, the output-device
  equivalent of `DeviceNotFound`. See Fixed below for the motivating
  bug.
- Node binding accepts `device: { id: string }` as a third form
  alongside `device: <number>` (index) and `device: <string>` (name
  substring). The JS wrapper passes it through to Rust unchanged; Rust
  resolves via cpal's `DeviceId`.
- `DeviceInfoJs.id` and `OutputDeviceInfoJs.id` fields on the Node
  binding types, mirroring the Rust `DeviceInfo.id` / `OutputDeviceInfo.id`
  additions. Visible in the auto-regenerated `npm/decibri/index.d.ts`
  and in the hand-authored `npm/decibri/src/decibri.d.ts`.
- `npm/decibri/src/errors.js` helper that re-wraps plain `Error`
  instances thrown from the native boundary as `TypeError` or
  `RangeError`, matching the JS wrapper's existing validation error
  classes. Brought in by `decibri.js` and `decibri-output.js`
  constructors to align Rust-originated errors with the JS wrapper's
  error class contract. Triggered only by code paths that reach Rust's
  `to_napi_error` (currently only `device: { id: ... }` selection); all
  other validation paths continue to throw from the JS wrapper directly
  with no behavior change.
- `docs/features.md`: comprehensive Cargo feature reference covering
  ORT distribution mode tradeoffs, execution-provider features,
  binding-author guidance, and feature compatibility constraints.
  Targeted at Rust crate consumers and FFI binding authors; lib.rs
  rustdoc now links to it for deep-dive reference.

### Fixed

- `DecibriError::DeviceNotFound`'s display string hardcoded
  "No audio input device found matching..." regardless of whether
  the lookup was against input or output devices. Direct Rust
  consumers and the new `device: { id: ... }` Node path now receive
  the correct direction via `DecibriError::OutputDeviceNotFound`
  for output-device misses. No change visible through the Node
  binding for existing name- and index-based lookups: those are
  intercepted by the JS wrapper and always threw direction-correct
  messages from JS before reaching Rust.

### Internal

- `DeviceDirection` trait gains a `not_found_error(String) -> DecibriError`
  method so `resolve_device_generic`'s `Name` and `Id` arms produce
  direction-correct errors via the `Input` / `Output` impls.
- Unit tests for `Arc<Mutex<CaptureStream>>` confirming the wrapping is
  `Send + Sync` (compile-time assertion) and serializes concurrent
  access across two threads (runtime test with `Barrier`). Documents
  the wrapping strategy the P3 Python binding will apply to share
  `!Sync` capture streams across Python threads.
- Crate-level rustdoc additions in `lib.rs`: a section on ORT error
  construction FFI side effects (the `ortsys![CreateStatus]` dylib-load
  trigger that motivates the `OrtPathInvalid` split from
  `OrtLoadFailed`) and a section on fork safety (guidance for Python
  `multiprocessing` consumers to use `spawn` start method).
- `lib.rs` rustdoc "Feature flags" section cross-references
  `docs/features.md` for consumers wanting the deep-dive reference.
- Em-dash cleanup across 19 code locations in `lib.rs`, `capture.rs`,
  `output.rs`, `vad.rs`, `error.rs`, `vad_integration.rs`, and
  `vad_ort_load_failure.rs`. Per CLAUDE.md, the codebase forbids em
  dashes; these were pre-existing violations.
- CLAUDE.md corrections: validation-gate commands updated to the
  canonical set (`cargo clippy --workspace -- -D warnings`,
  `cargo fmt --all -- --check`, `cargo test-decibri`); stale
  `## [3.0.0] - Unreleased` reference replaced with a template
  placeholder.
- `ort` crate version unchanged at `2.0.0-rc.12`.
- Bundled ONNX Runtime version unchanged at `1.24.4`.
- No Node.js API signatures, event names, or error messages changed.
- TypeScript declaration files in both `npm/decibri/index.d.ts`
  (auto-regenerated) and `npm/decibri/src/decibri.d.ts` (hand-authored)
  updated for the new `id` field and extended `device` option type.

### Migration notes for direct Rust crate consumers

- Exhaustive matches on `DeviceSelector` will stop compiling. Add a
  `_ =>` catch-all arm. Display strings and existing variant names
  are unchanged; code using `to_string()` or only constructing variants
  (not matching them) continues to work unaffected.
- Struct literal construction of `DeviceInfo` and `OutputDeviceInfo`
  from outside the `decibri` crate will stop compiling (added
  `#[non_exhaustive]`, added `id: String` field). External consumers
  should read these structs from `enumerate_input_devices()` /
  `enumerate_output_devices()` rather than constructing them directly.
- Consumers matching specifically on `DecibriError::DeviceNotFound`
  for output-device misses should now also match
  `DecibriError::OutputDeviceNotFound`. The convenience predicate
  `DecibriError::is_ort_path_error` remains unchanged and already
  groups only ORT-path variants.
- MSRV unchanged at rustc 1.88.

## [3.2.0] - 2026-04-22

Refactor release. Public Node.js and browser APIs are unchanged. Direct
Rust crate consumers get a structured `DecibriError` taxonomy with full
error-chain preservation, a new stable FFI-ready stream-reading API on
`CaptureStream`, a declared minimum-supported-Rust-version, and a Windows
hang fix in VAD initialization. See migration notes below.

### Changed

- `DecibriError` is now `#[non_exhaustive]` and the `Other(String)`
  catch-all variant has been removed. All previous `Other(...)` failures
  now have dedicated typed variants: `DeviceEnumerationFailed`,
  `CaptureStreamClosed`, `OutputStreamClosed`, `VadSampleRateUnsupported`,
  `VadThresholdOutOfRange`, `OrtInitFailed`, `OrtLoadFailed`,
  `OrtPathInvalid`, `OrtSessionBuildFailed`, `OrtThreadsConfigFailed`,
  `VadModelLoadFailed`, `OrtInferenceFailed`, `OrtTensorCreateFailed`,
  `OrtTensorExtractFailed`. Path-carrying variants use `PathBuf`; ORT
  variants carry `#[source] ort::Error` so `error.source()` walks the
  error chain. Display strings (and therefore `error.message` in Node
  and `str(exception)` in future Python bindings) are byte-identical
  to 3.1.0.
- `VadConfig` now has a public `validate()` method returning
  `Result<usize, DecibriError>` where the `usize` is the Silero VAD
  `window_size` for the validated sample rate. Called automatically by
  `SileroVad::new`; can be called explicitly to fail-fast before paying
  ORT-initialization cost.
- Device enumeration and resolution logic in `crates/decibri/src/device.rs`
  consolidated into a shared direction-generic implementation (input and
  output share code paths via an internal `DeviceDirection` trait). No
  public API change.
- Workspace minimum-supported Rust version (MSRV) declared at rustc 1.88,
  forced by `ort 2.0.0-rc.12` which requires `edition = "2024"`.

### Added

- `CaptureStream::try_next_chunk()`: non-blocking read, returns
  `Result<Option<AudioChunk>, DecibriError>` with a three-state return
  (`Some` chunk / `None` if no data yet / `Err(CaptureStreamClosed)` if
  terminal). Declared stable across 3.x as part of decibri's canonical
  FFI-consumer surface.
- `CaptureStream::next_chunk(timeout: Option<Duration>)`: blocking read
  with optional timeout, same three-state return shape. Declared stable
  across 3.x. Concurrent `stop()` unblocks a waiter within approximately
  20 ms via internal polling.
- `DecibriError::is_ort_path_error()`: helper that returns true for both
  `OrtLoadFailed` and `OrtPathInvalid`. Consumers handling path-level ORT
  failures should match this rather than enumerating both variants
  manually. The split between the two variants is a mechanical necessity
  (constructing `ort::Error` under `ort-load-dynamic` triggers an ORT C
  API call and would reintroduce the Windows hang).
- Crate-level rustdoc on `decibri`'s `lib.rs` covering capabilities,
  feature flags, ORT distribution modes, an end-to-end capture-plus-VAD
  example, the process-global ORT initialization constraint, thread-safety
  summary, and the 3.x FFI-surface stability contract.
- Rust integration tests for the VAD / ORT pipeline in
  `crates/decibri/tests/vad_integration.rs` (happy path, model-not-found,
  config validation, end-to-end silence inference) and
  `crates/decibri/tests/vad_ort_load_failure.rs` (load-failure path
  isolation, feature-gated to `ort-load-dynamic`). All CI-safe; no audio
  hardware required.
- Unit tests for `try_next_chunk` / `next_chunk` semantics (7 tests
  covering empty-queue, chunk-available, buffered-flush-before-closed,
  timeout, blocking-until-arrival, and polling-interval-correctness after
  concurrent `stop()`).
- Pre-publish packaging gate in `.github/workflows/release.yml`:
  ports the `verify_pack` function from `release-dryrun.yml` to run
  `npm pack --dry-run` against all 5 packages before any `npm publish`
  step. Closes the dryrun-skip honor-system gap: release-dryrun.yml
  previously caught packaging bugs but only if it was actually run before
  tagging.

### Fixed

- Windows hang in VAD initialization: passing a nonexistent or
  directory path as `VadConfig::ort_library_path` (or via the Node
  binding's `ortLibraryPath`) caused `ort::init_from` to hang
  indefinitely on Windows against pyke/ort 2.0.0-rc.12 with
  onnxruntime 1.24.4. `init_ort_once` now performs a filesystem-level
  `Path::is_file()` pre-check before handing the path to ORT, returning
  `DecibriError::OrtPathInvalid` immediately for any path that fails the
  check. The pre-check never touches ORT symbols, so it cannot itself
  trigger the dylib load it is designed to prevent.

### Migration notes for direct Rust crate consumers

- Matches against `DecibriError::Other(msg)` will stop compiling. Replace
  with matches against the specific new variants. The `error.message`
  text is unchanged; code using `.to_string()` rather than pattern
  matching continues to work unaffected.
- `DecibriError` is now `#[non_exhaustive]`: match expressions against
  it must include a `_ =>` catch-all arm. New variants added in future
  releases are non-breaking under this constraint.
- Consumers handling "ORT path failed" should prefer
  `err.is_ort_path_error()` or match both `OrtLoadFailed { .. }` and
  `OrtPathInvalid { .. }`. The two variants represent the same
  conceptual failure mode split for FFI-side-effect reasons.
- MSRV raised to rustc 1.88 (from effectively-unpinned in 3.1.x). This
  is forced by `ort 2.0.0-rc.12` declaring `edition = "2024"`. Projects
  on older rustc cannot build decibri 3.2.0 directly; stay on 3.1.x or
  upgrade the toolchain.
- `VadConfig::validate()` was new in 3.2.0 (no 3.1.x public signature
  to break) and has the final form `Result<usize, DecibriError>`. If you
  only need pass/fail, call `.is_ok()` or `.map(|_| ())`.

Node.js and browser consumers: no API or behaviour change. `error.message`
text from the native addon is byte-identical to 3.1.0 (verified against
the 38-assertion CI suite).

### Internal

- `ort` crate version unchanged at `2.0.0-rc.12`.
- Bundled ONNX Runtime version unchanged at `1.24.4`.
- Node binding error mapping at `bindings/node/src/lib.rs::to_napi_error`
  explicitly enumerates every variant (compiler-enforced exhaustive
  during the refactor by temporarily removing `#[non_exhaustive]` and
  the `_ =>` arm; both restored). New variants added upstream fall
  through to `GenericFailure` at runtime rather than failing to compile.
- `CaptureStream._stream` field type changed from `cpal::Stream` to
  `Option<cpal::Stream>` purely to enable unit-test construction without
  a real audio device. Production always stores `Some(stream)`; drop
  semantics are identical.
- docs.rs metadata added to target all 4 production platforms (Linux x64/ARM64,
  macOS ARM64, Windows x64) for full platform-specific rustdoc rendering.
  Future-proofs against the docs.rs change effective 2026-05-01 (building
  fewer targets by default).

## [3.1.0] - 2026-04-22

Internal rearchitecture: ONNX Runtime is now loaded dynamically at runtime
instead of embedded statically at build time. Public Node.js and browser
APIs are unchanged. Direct Rust crate consumers see a behaviour change;
see migration notes below.

### Changed

- ORT integration switched from `ort/download-binaries` to `ort/load-dynamic`.
  npm platform packages now bundle the ONNX Runtime shared library alongside
  the native addon. No change to `npm install decibri` workflow or to
  Decibri construction.
- Silero VAD now loads ONNX Runtime dynamically from the bundled shared
  library inside the installed platform package. Path resolution is
  automatic; the `ORT_DYLIB_PATH` environment variable is honoured as a
  developer escape hatch when set before Node.js starts.
- Bundled ONNX Runtime pinned to 1.24.4 (matches `ort 2.0.0-rc.12`'s
  `api-24` ABI target).
- Native addon size reduced from ~20 MB (3.0.x) to ~817 KB on Windows x64.
  ORT runtime now shipped separately as a ~13.5 MB bundled dylib inside
  the platform package. Net platform package size roughly unchanged, with
  better separation of concerns.
- Error message text in `decibri-*` error variants has been normalized
  for style consistency (em-dashes replaced with sentence splits). The
  error message prefixes (e.g., `"device index out of range"`) are
  unchanged, so consumers matching those prefixes are unaffected.
  Consumers matching full error message strings may need to update.

### Added

- Cargo features on the `decibri` crate for direct Rust consumers:
  `ort-load-dynamic` (default), `ort-download-binaries` (opt-in, restores
  3.0.x zero-config build behaviour), and execution-provider passthroughs
  `coreml`, `cuda`, `directml`, `rocm` (off by default).
- Rust unit tests for device enumeration (`is_default` correctness under
  duplicate display names).
- Release pipeline hardening: version-match preflight, `curl --retry` on
  ORT downloads, macOS code-signature verification, Windows DLL imports
  inspection, stage-and-verify packaging validation in release-dryrun.
- Upstream dependency monitoring for Microsoft's ONNX Runtime releases
  (notification-only; guards against ABI mismatch upgrades).

### Fixed

- Issue #14: when two audio devices share a display name (e.g. two USB
  microphones both reporting "Microphone"), both were previously marked
  `is_default: true`. Now uses cpal 0.17's `Device::id()` for stable
  per-host device identity (WASAPI endpoint ID, CoreAudio UID, ALSA
  PCM ID). Fix applies to both input and output device enumeration.

### Migration notes for direct Rust crate consumers

If you depend on `decibri` directly via `cargo add decibri` and use the
`vad` feature:

- **Option 1 (recommended for zero-config builds):** pin with
  `--features ort-download-binaries` on the dependency, which restores
  the 3.0.x behaviour (ORT downloaded at build time, embedded statically).
- **Option 2 (recommended for production deployments):** keep default
  features and either set `ORT_DYLIB_PATH=/path/to/libonnxruntime.so`
  before first use, or call `ort::init_from(path).commit()` at startup,
  or pass `ort_library_path` on `VadConfig` when constructing `SileroVad`.

Known limitation: ONNX Runtime is initialized once per process. Multiple
`Decibri`/`SileroVad` instances constructed with different `ort_library_path`
values will silently use the first-constructed instance's path. Pick one
path and use it consistently.

Direct consumers using only `capture`, `output`, `denoise`, or `gain`
features (no `vad`) are unaffected.

### Internal

- `ort` crate version unchanged at `2.0.0-rc.12`.
- `tls-native` ORT feature removed (was only required for `download-binaries`'
  HTTPS fetch).

## [3.0.0] - 2026-04-11

Complete rewrite from C++ (PortAudio) to Rust (cpal). One unified package for Node.js and browsers. Version jumps from 1.0.0 to 3.0.0 because this release replaces both `decibri` (v1) and `decibri-web` (v0.1.1). v2.x was never published.

### Changed

- Complete rewrite from C++ (PortAudio) to Rust (cpal)
- Native addon built with napi-rs (replaces node-gyp / prebuildify)
- JS API unchanged: drop-in replacement for v1.x consumers (mcp-listen, voxagent, Wake Word verified)

### Added

- Audio output: `DecibriOutput` class (`Writable` stream, speaker playback)
- Browser support: unified package with conditional exports, AudioWorklet capture
- Silero VAD: ML-based voice activity detection via `vadMode: 'silero'`
- Full duplex: `mic.pipe(speaker)` for simultaneous capture and playback
- `format: 'float32'` output support alongside `'int16'`
- Output device enumeration: `DecibriOutput.devices()`
- TypeScript declarations for all APIs (Node.js capture, Node.js output, browser)
- `crates.io` publication as a Rust crate

### Removed

- PortAudio dependency (replaced by cpal)
- node-gyp / prebuildify build system (replaced by napi-rs)
- Source build fallback (Rust binaries are self-contained)

### Deprecated

- `decibri-web` npm package (use `decibri` with the browser conditional export instead)

## [1.0.0] - 2025-06-15

Initial release. C++ native addon wrapping PortAudio with pre-built binaries.

- Microphone capture as a Node.js `Readable` stream
- Pre-built binaries for Windows x64, macOS ARM64, Linux x64, Linux ARM64
- Energy-based voice activity detection (`vad`, `vadThreshold`, `vadHoldoff`)
- Device enumeration and selection by index or name
- Int16 PCM output (little-endian)
- Source build fallback via node-gyp

[3.2.0]: https://github.com/decibri/decibri/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/decibri/decibri/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/decibri/decibri/compare/v1.0.0...v3.0.0
[1.0.0]: https://github.com/analyticsinmotion/decibri/releases/tag/v1.0.0
