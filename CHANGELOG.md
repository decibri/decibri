# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - Unreleased

Complete rewrite from C++ (PortAudio) to Rust (cpal). One unified package for Node.js and browsers. Version jumps from 1.0.0 to 3.0.0 because this release replaces both `decibri` (v1) and `decibri-web` (v0.1.1). v2.x was never published.

### Changed

- Complete rewrite from C++ (PortAudio) to Rust (cpal)
- Native addon built with napi-rs (replaces node-gyp / prebuildify)
- JS API unchanged — drop-in replacement for v1.x consumers (mcp-listen, voxagent, Wake Word verified)

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

- `decibri-web` npm package — use `decibri` with the browser conditional export instead

## [1.0.0] - 2025-06-15

Initial release. C++ native addon wrapping PortAudio with pre-built binaries.

- Microphone capture as a Node.js `Readable` stream
- Pre-built binaries for Windows x64, macOS ARM64, Linux x64, Linux ARM64
- Energy-based voice activity detection (`vad`, `vadThreshold`, `vadHoldoff`)
- Device enumeration and selection by index or name
- Int16 PCM output (little-endian)
- Source build fallback via node-gyp

[3.0.0]: https://github.com/decibri/decibri/compare/v1.0.0...v3.0.0
[1.0.0]: https://github.com/analyticsinmotion/decibri/releases/tag/v1.0.0
