# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[3.1.0]: https://github.com/decibri/decibri/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/decibri/decibri/compare/v1.0.0...v3.0.0
[1.0.0]: https://github.com/analyticsinmotion/decibri/releases/tag/v1.0.0
