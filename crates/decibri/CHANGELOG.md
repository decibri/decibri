<!-- markdownlint-disable MD024 -->

# Decibri Rust Core Changelog

Changes to the decibri Rust core crate, published to crates.io. Tags use the `crate-v*` pattern (e.g., `crate-v3.4.2`).

For other decibri packages, see:

- npm package: [npm/decibri/CHANGELOG.md](../../npm/decibri/CHANGELOG.md)
- Python package: [bindings/python/CHANGELOG.md](../../bindings/python/CHANGELOG.md)

## [5.1.0] - Unreleased

### Added

- `File`, an offline source: conditions a WAV recording (`File::open`, with the `File::new` alias) or in-memory samples (`File::buffer`) through the same chain as live capture, delivered as a finite iterator of conditioned chunks that ends after the chain's end-of-stream tail. WAV support covers 16-bit PCM and 32-bit float; no decode dependency is added.
- Whole-recording voice-activity analysis: `File::analyze` (also spelled `File::analyse`) consumes the source and returns a `VadReport` with per-window `VadWindow` scores and merged speech `Segment`s, timed in seconds of file time. Segment merging applies `FileConfig::vad_holdoff_ms` in file time, and when the target rate is not a detector rate the detector feed is resampled internally to 16 kHz.
- `FileConfig`, mirroring the conditioning surface of `MicrophoneConfig`, plus the `vad` and `vad_holdoff_ms` fields for analysis.
- New `DecibriError` variants: `FileReadFailed`, `WavInvalid`, and `VadNotConfigured`.

## [5.0.0] - 2026-06-24

### Added

- An opt-in capture enhancement with a DC-removal step. `MicrophoneConfig` gains a `dc_removal: bool` field, default `false`. Setting `dc_removal = true` adds a one-pole DC-blocking high-pass to the capture chain, applied after the channel and rate normalization, which removes a constant offset from the captured audio while leaving the voice band essentially flat. Default off, so the capture path stays byte-identical unless a consumer opts in.
- Voice-activity detection reads the capture signal before the opt-in enhancement step. When a capture enhancement (such as DC removal) is enabled, the detector is fed the normalized signal as it stands before the enhancement step, so the enhancement does not affect detection. A new internal `MicrophoneStream::vad_input` accessor exposes the pre-enhancement samples aligned with each delivered chunk, and the Node and Python bindings feed the detector from it. With no enhancement enabled (the default) the detector reads the delivered audio exactly as before, with no added overhead.
- An opt-in high-pass filter in the capture chain. `MicrophoneConfig` gains `highpass: Option<HighpassFilter>`, a `#[non_exhaustive]` cutoff selector re-exported as `decibri::HighpassFilter` with two variants (`Hz80`, an 80 Hz second-order Butterworth high-pass, and `Hz100`, a 100 Hz one). Naming a cutoff adds a same-length biquad stage to the capture chain after the denoise step, on the mono signal at the target rate, attenuating low-frequency rumble below the voice band while leaving the speech band essentially flat (the response is at the -3 dB point at the selected corner and near unity through the speech band). The filter is sample-in-sample-out and adds no latency (`latency_samples()` is zero), so it does not change the chain's transform latency or the VAD tap alignment. The cutoff set is `#[non_exhaustive]` so further cutoffs are a non-breaking widening, mirroring `DenoiseModel`. Default off (`None`), so the capture path stays byte-identical unless a consumer opts in.
- An opt-in automatic gain control (AGC) step in the capture chain, behind the `gain` feature (on by default). `MicrophoneConfig` gains `agc: Option<i8>`, a target level in dBFS (range -40 to -3, typical -18), default `None`. Setting it adds an internal level-control engine to the capture chain after the high-pass step, on the mono signal at the target rate, which drives the running level toward the target with a smoothed, rate-limited gain (a fast attack so a loud onset is pulled down promptly, a gentle release, a roughly 30 dB maximum gain, and a noise-floor gate that holds the gain through silence). The opening is delivered at unity gain and the level estimate primes from the opening samples, so the target is reached within tens of milliseconds with no opening window of wrong gain. It is sample-in-sample-out with no look-ahead and adds no latency (`latency_samples()` is zero), so it does not change the chain's transform latency or the VAD tap alignment. An out-of-range target is rejected at `MicrophoneConfig::validate` with the new `DecibriError::AgcTargetOutOfRange` rather than clamped. Default off (`None`), so the capture path stays byte-identical unless a consumer opts in.
- An opt-in peak limiter in the capture chain, behind the `gain` feature (on by default). `MicrophoneConfig` gains `limiter: Option<f32>`, a sample-peak ceiling in dBFS (range -3.0 to 0.0, typical -1.0), default `None`. Setting it adds the limiter stage last in the capture chain, after the level-control step, on the mono signal at the target rate. It holds the signal at or below the ceiling with fast feedback gain reduction (a fast attack, a slower release) plus a hard-ceiling backstop, so no output sample ever exceeds the ceiling, even on an instantaneous full-scale transient: the safety net that catches a peak the AGC's gain would otherwise let through. It is sample-in-sample-out with no look-ahead and adds no latency (`latency_samples()` is zero), so it does not change the chain's transform latency or the VAD tap alignment. An out-of-range ceiling is rejected at `MicrophoneConfig::validate` with the new `DecibriError::LimiterCeilingOutOfRange` rather than clamped. Default off (`None`), so the capture path stays byte-identical unless a consumer opts in.
- `sample::rms`, the root-mean-square level of f32 samples in `[0, 1]` (`sqrt(mean(x^2))`, accumulated in `f64`). It shares the `/32768` `[0, 1]` scale of the i16 path, so it is the energy-VAD score on the same scale the energy threshold compares against. The Node and Python bindings now compute the energy-mode VAD score with it, on the pre-enhancement signal from `MicrophoneStream::vad_input`, so the score reads the same pre-enhancement signal the Silero path already reads.
- An opt-in single-channel speech-enhancement (denoise) step in the capture chain, behind the `denoise` feature (on by default). `MicrophoneConfig` gains `denoise: Option<DenoiseModel>`, a `#[non_exhaustive]` model selector re-exported as `decibri::DenoiseModel` with one model today (`FastEnhancerT`), and `denoise_model_path: Option<PathBuf>`. It also gains `ort_library_path: Option<PathBuf>`, the ONNX Runtime dynamic-library path used to initialise ORT for the denoise stage (the capture-path counterpart to `VadConfig`'s field of the same name); `None`, the default, leaves ORT to its own discovery (the `ORT_DYLIB_PATH` environment variable, then the system loader). Naming a model and supplying its ONNX file path adds a framed denoise stage to the capture chain after the DC-removal step, loaded through the same ONNX session seam as the VAD; the crate ships no model bytes (the caller supplies the path). Default off (`None`), so the capture path is unchanged unless a consumer opts in. The denoise stage re-blocks the captured audio into the model's analysis frames and introduces a small latency, so voice-activity detection continues to read the pre-enhancement signal (the tap leads the delivered audio by the chain's latency).

### Changed

- Capture now delivers audio at exactly the requested `sample_rate` on every device by resampling in the engine. The input device is opened at its native sample rate (its default supported format) and a resample stage in the capture chain converts the native rate to the requested rate (after the downmix, so the resampler receives mono), using the owned anti-aliased polyphase `decibri-resampler`. A device already at the requested rate keeps the direct path with no resample stage. Previously `start()` handed the requested rate to the platform audio backend, which delivered it only when the device or OS could and otherwise failed to open. `MicrophoneStream::sample_rate()` and `AudioChunk::sample_rate` still report the requested rate, now guaranteed regardless of the device's native rate. Breaking for Rust consumers: the captured audio is now produced by decibri's resampler when the device's native rate differs from the configured rate.
- Microphone capture is now mono only, narrowing a capability that shipped in 4.x. decibri 4.x delivered interleaved multichannel `AudioChunk`s when a microphone was opened with more than one channel; 5.0 captures mono. `MicrophoneConfig::channels` accepts only `1` (the default); a value greater than `1` is rejected at `MicrophoneConfig::validate` with the new `DecibriError::MultichannelNotSupported`, a clear error rather than a silent downmix to mono (a zero channel count stays `ChannelsOutOfRange`). The `channels` field is retained and `AudioChunk` keeps its channel-general shape (interleaved `data` plus a `channels` field), so honouring a value greater than `1` later (by delivering true interleaved multichannel) widens the accepted set without breaking callers: an additive return, not a further break. The intended longer-term multichannel direction is array ingest (consuming several channels internally for processing such as beamforming or array noise reduction while still delivering one conditioned mono stream), which is distinct from raw multichannel delivery. The device-side downmix that averages a multichannel capture to the mono target is unchanged; only a request for `channels > 1` is now rejected. The speaker path is unaffected (`SpeakerConfig::channels` still accepts `1..=32`). Breaking for Rust consumers that opened a microphone with more than one channel: set `channels` to `1` or leave it at the default.
- `MicrophoneStream::next_chunk` and `try_next_chunk` now take a requested block size in interleaved samples (frames times channels) and return exactly that many samples per chunk, re-blocking the device's native capture buffers on the consumer side. `next_chunk(samples, timeout)` blocks until a full block accumulates (or the stream closes, or the timeout elapses); `try_next_chunk(samples)` returns `Ok(None)` until a full block is buffered. Every returned chunk is exactly `samples` long during the stream, with a possibly-short final chunk at close carrying the remaining tail (1..`samples` samples), so no captured sample is dropped. Previously both returned whole native buffers whose size was platform-dependent and, on Windows WASAPI, unrelated to `frames_per_buffer`. Breaking for Rust consumers: the methods gain a `samples` parameter and the returned chunk size changes. Consumers that want raw native buffers can still use `receiver()`.
- The public config and result structs `MicrophoneConfig`, `AudioChunk`, `VadConfig`, `VadResult`, and `SpeakerConfig` are now `#[non_exhaustive]`. The fields stay public, but external crates can no longer construct these types with a struct literal (or a `..Default::default()` functional update). Construct a config with `Default::default()` and assign the fields you need; `AudioChunk` and `VadResult` are produced by the library and read field by field. This seals the surface so future field additions stay backward compatible. Breaking for Rust consumers that struct-literal these types.
- `sample::i16_to_f32` is no longer part of the public API (the byte-oriented converters such as `i16_le_bytes_to_f32` remain public). It had no external callers; the bindings convert from bytes, never from `i16` samples.
- The ORT-backed `DecibriError` variants (`OrtInitFailed`, `OrtLoadFailed`, `OrtSessionBuildFailed`, `OrtThreadsConfigFailed`, `VadModelLoadFailed`, `ModelLoadFailed`, `OrtInferenceFailed`, `OrtTensorCreateFailed`, `OrtTensorExtractFailed`) now carry their underlying ONNX Runtime failure boxed as a `Box<dyn std::error::Error + Send + Sync>` via `#[source]`, the same way `DeviceFailed` carries its cpal error, so no concrete `ort::Error` type appears in decibri's public error surface. This insulates the crate from a future `ort` major: the error variants no longer name an `ort` type. The Display messages and the `error.source()` chain are unchanged (the boxed source is still the ORT error, reachable and downcastable through `.source()`); only the ability to bind a matched variant's `source` field as a concrete `ort::Error` is removed. Breaking for Rust consumers that pattern-matched an ORT variant and used its bound `source` as an `ort::Error` rather than reading the message or walking `.source()` as `dyn Error`.

### Fixed

- A non-finite (NaN or infinity) input sample no longer corrupts the rest of a conditioned capture stream. The conditioning chain now sanitizes non-finite input to silence at its entry, before any stage runs, so a glitched device sample cannot poison the recursive state of the DC-removal, high-pass, denoise, or AGC stages. Previously one non-finite sample left that feedback state non-finite, so without the limiter every later sample reached the consumer non-finite (past the documented `AudioChunk` range), and with the limiter the rest of the stream was flushed to silence. The guard is exact on finite input (the conditioned output is byte-identical for conforming audio) and runs only on the conditioned path, so an unconditioned capture keeps its zero-cost direct delivery. The `AudioChunk::data` doc now states the range accurately (guaranteed with the limiter enabled).
- Resampled capture no longer drops the resampler's group-delay tail at stream close. The capture chain now drains the resampler's held tail once when the stream closes, appending it to the re-block buffer so it is delivered as part of the final chunk(s). The complete resampled signal is now delivered and no captured sample is dropped on the resample path. A device already at the requested rate (no resample stage) and a single-channel device (no chain) are unchanged.
- The macOS microphone-permission hint now reads "System Settings > Privacy & Security" (the modern macOS wording) instead of the pre-Ventura "System Preferences > Security & Privacy". The `PermissionDenied` message prefix is unchanged.

## [4.3.2] - 2026-06-12

### Added

- `sample::downmix_to_mono`, which averages interleaved multichannel frames to mono (one mono sample per frame, the trailing partial frame dropped). The Node and Python bindings now call it to downmix capture to mono before the Silero VAD, fixing garbled VAD probabilities on devices opened with more than one channel: the VAD models a single channel and previously received interleaved samples, reading consecutive channels as successive mono samples. Rust consumers running their own VAD on multichannel capture should downmix the same way. The interleaved data delivered to consumers is unchanged.

## [4.3.1] - 2026-06-11

### Fixed

- `SpeakerStream::drain()` no longer hangs when the stream is dropped without `stop()`. A new `Drop` clears the running flag, so a drain parked in its wait loop (on the stream or on a `SpeakerSink` sharing it) returns instead of polling forever.
- Malformed VAD models (correct tensor names but wrong shapes) now return a typed `OnnxBackendFailed` error instead of panicking the process.
- The fork-after-ORT-init guard now also fires when a Silero VAD is constructed in a forked child, not only at the first inference, so `ForkAfterOrtInit` is raised before any ORT session is built on inherited state.

### Added

- `DecibriError::DeviceFailed`, a typed error carrying the underlying cpal stream error as a structured source for a device or driver failure during streaming. `MicrophoneStream` and `SpeakerStream` gained `take_last_error()` so a consumer that sees a closed stream can tell a driver failure from an explicit `stop()`.
- `MicrophoneStream::sample_rate()`, `channels()`, and `overrun_count()` accessors.

### Changed

- The microphone capture channel is now bounded: a stalled consumer drops the newest buffers (counted via `overrun_count()`) rather than growing memory without bound, and the realtime callback never blocks.
- Removed the unused `parking_lot` and `dasp_sample` dependencies.

## [4.3.0] - 2026-06-10

### Fixed

- `SpeakerStream::stop()` now discards queued audio and goes silent immediately. Previously it appended empty sentinels behind the queued audio (and was a no-op when the channel was full), so audio played to completion after `stop()`.

### Changed

- **Behavioral:** `SpeakerStream::drain()` is now a repeatable, non-terminal flush. It blocks until everything queued at call time has played, then leaves the stream usable, so a later `drain()` waits for its own audio and `send()` keeps working. Previously `drain()` incidentally ended the stream (it set `running = false`); code that relied on `drain()` as a stop must now call `stop()` explicitly. Applies to `SpeakerSink::drain()`. (In the Node binding, `end()` now flushes then stops, so the stream is still terminal after `end()`.)
- `MicrophoneStream::stop()` and `SpeakerStream::stop()` now release the audio device immediately: each drops the held `cpal::Stream`, so the OS frees the device (and the microphone-in-use indicator clears) on `stop()` rather than only when the handle is dropped. For direct Rust consumers, `stop()` now blocks briefly while the audio thread tears down; post-stop error semantics are unchanged.

## [4.2.0] - 2026-06-07

### Fixed

- The Silero VAD never detected speech due to a missing 64-sample audio context required by Silero v5. VAD probabilities now reflect real speech activity. If you consume VAD output, expect meaningful probabilities where previously everything sat near zero.

## [4.1.0] - 2026-05-31

### Added

- `SpeakerSink`: a cloneable, `Send + Sync` handle to a running `SpeakerStream`'s sample channel and drain state, obtained from `SpeakerStream::sink()`. It exposes `send()` and `drain()` with the same semantics as the stream's. `SpeakerStream` is `Send`; a `SpeakerSink` is a cheaper, lock-free companion that clones only the thread-safe primitives (the channel sender and the atomic flags), so callers can push samples or wait for drain from another thread (for example a worker pool) without holding the stream, and the off-thread work never blocks a holder of the stream that needs `stop()` or `is_playing()`. The stream must stay alive while a sink is in use; a `send` on a surviving sink after the stream is dropped returns `SpeakerStreamClosed`. Additive: `SpeakerStream` and its methods are unchanged.

## [4.0.1] - 2026-05-30

### Fixed

- **Device-index error message.** `DeviceIndexOutOfRange` no longer references a class name that was removed in the 4.0.0 rename. The message keeps its `device index out of range` prefix and now points at the neutral `devices()` enumeration, so the guidance is direction-agnostic and names no removed type.

## [4.0.0] - 2026-05-30

Renames the public API to a microphone/speaker vocabulary. This is a breaking
release; see [MIGRATION.md](MIGRATION.md) for the full upgrade path from 3.x.

### Changed

- **Type renames (breaking).** Capture side: `AudioCapture` to `Microphone`,
  `CaptureStream` to `MicrophoneStream`, `CaptureConfig` to `MicrophoneConfig`.
  Output side: `AudioOutput` to `Speaker`, `OutputStream` to `SpeakerStream`,
  `OutputConfig` to `SpeakerConfig`. Device info: `DeviceInfo` to
  `MicrophoneInfo`, `OutputDeviceInfo` to `SpeakerInfo`. The channel fields
  `max_input_channels` and `max_output_channels` are unchanged. `AudioChunk`,
  `VadConfig`, `VadResult`, and `SileroVad` keep their names and signatures.
- **Module renames (breaking).** `capture` to `microphone`, `output` to
  `speaker`. The common types are now re-exported at the crate root, so
  `use decibri::Microphone;` works without naming the module.
- **Feature rename (breaking).** `output` to `playback`. The `capture` feature
  is unchanged.
- **Error variant renames (breaking).** `DeviceNotFound` to
  `MicrophoneNotFound`, `OutputDeviceNotFound` to `SpeakerNotFound`,
  `NoOutputDeviceFound` to `NoSpeakerFound`, `CaptureStreamClosed` to
  `MicrophoneStreamClosed`, `OutputStreamClosed` to `SpeakerStreamClosed`.
  `NoMicrophoneFound` and `NotAnInputDevice` keep their names.
- **Error message text.** `Display` strings now use the microphone/speaker
  vocabulary, for example `No microphone found matching "{name}"`, `No speaker
  found matching "{name}"`, and `Microphone stream is closed`.
- **ORT error variants are `vad`-gated.** The `DecibriError` variants carrying
  an `ort::Error` (the `Ort*` family and `OrtPathInvalid`) are compiled only
  with the `vad` feature, so `capture` and `playback` builds without `vad` no
  longer pull in ONNX Runtime.

### Added

- `Microphone::devices()` and `Speaker::devices()` associated functions for
  listing input and output devices.

### Removed

- The `enumerate_input_devices()` and `enumerate_output_devices()` free
  functions, replaced by `input_devices()` and `output_devices()` (and the
  associated `Microphone::devices()` / `Speaker::devices()`).
- The `resolve_device()` and `resolve_output_device()` free functions are
  removed from the public API. Select a device via `DeviceSelector` on
  `MicrophoneConfig` / `SpeakerConfig`; it is resolved internally.

## [3.4.2] - 2026-05-25

### Fixed

- Speaker example in `crates/decibri/README.md`: defined `pcm_int16_bytes` as `Vec<u8>` of 48000 zero bytes (1 second of int16 silence at 24kHz mono) so the example has a real value to send. Previously the example referenced `pcm_int16_bytes` without defining it, raising `error[E0425]: cannot find value 'pcm_int16_bytes' in this scope` on copy-paste.

## [3.4.0] - 2026-05-02

### Added

- **`OnnxSession` trait abstraction in Rust core.** New internal `pub(crate) trait OnnxSession` inside `crates/decibri/src/onnx.rs` abstracts ONNX Runtime usage behind a backend-agnostic interface. `SileroVad` consumes the trait through `Box<dyn OnnxSession>`. The ORT-backed implementation is the only impl in 3.x.
- `DecibriError::OnnxBackendFailed { backend: &'static str, source: Box<dyn std::error::Error + Send + Sync> }` variant. Reserved on the `#[non_exhaustive]` enum. Additive; existing 8 ORT variants unchanged. `is_ort_path_error` continues to return false on the new variant.
- **`DecibriError::ForkAfterOrtInit { init_pid: u32, current_pid: u32 }` variant + runtime fork detection.** Linux-only failure-mode hardening: a process that forks after a successful `SileroVad::new` previously inherited `static ORT_INIT` flagged as set while the underlying ORT runtime state (allocators, thread pools) was unsafe to reuse, producing silent wrong probabilities, segfaults, or hangs in the child. 3.4.0 stamps the initializing pid into a paired `static ORT_INIT_PID: OnceLock<u32>` inside the same `init_ort_once` success path; a `pub(crate) fn check_pid_for_ort()` runs at the entry of `SileroVad::process()` and returns `Err(ForkAfterOrtInit { init_pid, current_pid })` on pid mismatch. The `Display` message embeds both pids and the two remediation options ("Use `multiprocessing.set_start_method('spawn')` or construct `Microphone(vad='silero')` inside each child process"). Single-check coverage at the outer entry point applies to every inference call without per-window overhead. macOS and Windows are unaffected by fork semantics. Additive on the `#[non_exhaustive]` `DecibriError` enum; mapped through to npm via the napi `_ =>` catch-all (`Status::GenericFailure` carrying `e.to_string()`) and to Python via an explicit `ForkAfterOrtInit(DecibriError)` subclass in the wheel.

### Internal

- `crates/decibri` 3.x public API stays byte-identical (`SileroVad`, `VadConfig`, `VadResult`, `DecibriError` keep their 3.3.x signatures). npm binding, Python binding, browser shim are unchanged.
- `vad::init_ort_once` visibility raised from private to `pub(crate)` so the `onnx` module's inline ORT-backed test can reuse the same process-global init path that `vad` tests use.
- `static ORT_INIT_PID: OnceLock<u32>` paired with the existing `ORT_INIT`, set inside the `OnceLock::get_or_init` callback so the pid stamp is paired with the successful ORT init rather than a speculative pre-init value. `pub(crate) fn check_pid_for_ort()` exposes the comparison to inference call sites within the crate. Linux-only `test_fork_safety.py` tests in the Python wheel pin the behavior end-to-end (gated `skipif sys.platform != "linux"`); they run on CI's `ubuntu-latest` job and skip on other hosts.

## [3.3.2] - 2026-04-26

### Changed

- **`crates/decibri/build.rs` rewrite to use Cargo.toml as primary source.** The previous build script (introduced in 3.3.1 to source the cpal version from a single point of truth) read the workspace `Cargo.lock` via a path traversal that worked in workspace builds but panicked in `cargo publish` verify because the published tarball is flat (`decibri-X.Y.Z/Cargo.toml` and `decibri-X.Y.Z/Cargo.lock` are siblings, not in workspace structure). The traversal landed at `target/Cargo.lock` (nonexistent) and aborted the build. **3.3.1's crates.io publish failed on this defect; 3.3.1 shipped to npm but not to crates.io.** 3.3.2 fixes the build.rs to read `CARGO_MANIFEST_DIR / Cargo.toml` directly, with belt-and-suspenders fallbacks: env-var override (`DECIBRI_CPAL_VERSION`), workspace `Cargo.toml` fallback for `{ workspace = true }` inherit form, hardcoded `"0.17"` constant fallback (with `cargo:warning=`) for unforeseen build contexts. Cargo.toml is unambiguously present in every build context because cargo guarantees `CARGO_MANIFEST_DIR` always points at the manifest's directory.
- **No functional change to user-visible API or error messages.** All 5 message refinements from 3.3.1 (`SampleRateOutOfRange`, `FramesPerBufferOutOfRange`, `AlreadyRunning`, `OrtInitFailed`, `OrtLoadFailed` / `OrtPathInvalid`, `PermissionDenied`) are preserved unchanged.
- **`decibri::CPAL_VERSION` byte-identity preserved across all four cargo-emitted dep forms.** Verified 2026-04-26 by walking through `find_cpal_in_dependencies` + `truncate_to_major_minor` against on-disk Cargo.toml content: Form 1 (`cpal = "0.17"` workspace.dependencies) -> `"0.17"`; Form 2 (`cpal = { version = "0.17", optional = true }` hypothetical inline-table) -> `"0.17"`; Form 3 (`cpal = { workspace = true, optional = true }` source crate) -> workspace fallback -> `"0.17"`; Form 4 (`[dependencies.cpal] version = "0.17"` published normalized) -> `"0.17"`. All four forms produce identical output to the v3.3.1 build.rs's Cargo.lock-resolved truncation because the workspace pin is at major.minor granularity already (`cpal = "0.17"` in `[workspace.dependencies]`).
- **`release-dryrun.yml` extended to exercise `cargo publish -p decibri --dry-run`.** The previous dryrun workflow ran the npm-side build matrix and `verify_pack` packaging gate but had no crates.io publish path coverage. The new step catches build.rs failures and any other publish-time issues that affect the Rust crate publish but not the npm packaging. Closes the procedural gap that allowed 3.3.1's defect to ship through CI green.

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

Error message wording on shipped `DecibriError` variants has historically been stable across the 3.x line. 3.3.1 explicitly refines five messages to remove audience-leak issues: Node-API-targeted camelCase parameter names, class-name hardcoding in `AlreadyRunning`, a Rust-internal type reference (`VadConfig`) in `OrtInitFailed`, npm-internal phrasing ("platform package") in `OrtLoadFailed` / `OrtPathInvalid`, and macOS-only platform guidance in `PermissionDenied`. 3.3.1 is the consolidation point.

- **Direct Rust crate consumers** asserting on `DecibriError::Display` strings should update assertions for the five refined messages. Type-level matching on `DecibriError` variants is unaffected; only string-text assertions need updating.
- **Node consumers** using `e.message.includes(prefix)` or `e.message.startsWith(prefix)` patterns should update for `sampleRate` -> `sample rate`, `framesPerBuffer` -> `frames per buffer`, and the `AlreadyRunning` message text. Type-level matching on `RangeError` / `TypeError` is unaffected.
- **Python consumers**: the in-development Python wheel consumes `decibri@3.3.1`, so the messages it sees are the corrected forms.

## [3.3.0] - 2026-04-23

Groundwork release for the Python bindings. Adds a stable-ID form for audio device selection (`DeviceSelector::Id`), fixes a long-standing direction bug in `DecibriError::DeviceNotFound` when resolving output devices, exposes both in the Node binding, and extends the reference documentation with a Cargo feature flag guide plus additional crate-level rustdoc. No Node.js or browser API break. Direct Rust crate consumers pattern-matching on `DeviceSelector` or struct-literal-constructing `DeviceInfo` / `OutputDeviceInfo` need to update for the new `#[non_exhaustive]` attributes (see Migration notes below).

### Changed

- `DeviceSelector`, `DeviceInfo`, and `OutputDeviceInfo` are now `#[non_exhaustive]`. External Rust consumers pattern-matching on `DeviceSelector` must add a `_ =>` catch-all arm; consumers constructing `DeviceInfo` or `OutputDeviceInfo` via struct literal from outside the crate must switch to reading fields off instances returned by `enumerate_input_devices` / `enumerate_output_devices`. Field names and display strings are unchanged. This future-proofs the API: subsequent variant or field additions are source-compatible for consumers who include the catch-all.

### Added

- `DeviceSelector::Id(String)` for selecting audio devices by stable per-host identifier (WASAPI endpoint ID on Windows, CoreAudio UID on macOS, ALSA pcm_id on Linux). Unlike `DeviceSelector::Name` (case-insensitive substring) and `DeviceSelector::Index` (positional), `Id` survives across enumerations: display names can shift when other devices are plugged in but per-host IDs do not.
- `id: String` field on `DeviceInfo` and `OutputDeviceInfo`, populated from `cpal::DeviceId`'s `Display` output. Empty string if cpal cannot produce a stable ID for a given device (rare; some host backends cannot assign IDs to every enumerated device). Obtain the ID from these fields and pass to `DeviceSelector::Id`.
- `DecibriError::OutputDeviceNotFound(String)` variant, the output-device equivalent of `DeviceNotFound`. See Fixed below for the motivating bug.
- Node binding accepts `device: { id: string }` as a third form alongside `device: <number>` (index) and `device: <string>` (name substring). The JS wrapper passes it through to Rust unchanged; Rust resolves via cpal's `DeviceId`.
- `DeviceInfoJs.id` and `OutputDeviceInfoJs.id` fields on the Node binding types, mirroring the Rust `DeviceInfo.id` / `OutputDeviceInfo.id` additions. Visible in the auto-regenerated `npm/decibri/index.d.ts` and in the hand-authored `npm/decibri/src/decibri.d.ts`.
- `npm/decibri/src/errors.js` helper that re-wraps plain `Error` instances thrown from the native boundary as `TypeError` or `RangeError`, matching the JS wrapper's existing validation error classes. Brought in by `decibri.js` and `decibri-output.js` constructors to align Rust-originated errors with the JS wrapper's error class contract. Triggered only by code paths that reach Rust's `to_napi_error` (currently only `device: { id: ... }` selection); all other validation paths continue to throw from the JS wrapper directly with no behavior change.
- `docs/features.md`: comprehensive Cargo feature reference covering ORT distribution mode tradeoffs, execution-provider features, binding-author guidance, and feature compatibility constraints. Targeted at Rust crate consumers and FFI binding authors; lib.rs rustdoc now links to it for deep-dive reference.

### Fixed

- `DecibriError::DeviceNotFound`'s display string hardcoded "No audio input device found matching..." regardless of whether the lookup was against input or output devices. Direct Rust consumers and the new `device: { id: ... }` Node path now receive the correct direction via `DecibriError::OutputDeviceNotFound` for output-device misses. No change visible through the Node binding for existing name- and index-based lookups: those are intercepted by the JS wrapper and always threw direction-correct messages from JS before reaching Rust.

### Internal

- `DeviceDirection` trait gains a `not_found_error(String) -> DecibriError` method so `resolve_device_generic`'s `Name` and `Id` arms produce direction-correct errors via the `Input` / `Output` impls.
- Unit tests for `Arc<Mutex<CaptureStream>>` confirming the wrapping is `Send + Sync` (compile-time assertion) and serializes concurrent access across two threads (runtime test with `Barrier`). Documents the wrapping strategy the Python binding will apply to share `!Sync` capture streams across Python threads.
- Crate-level rustdoc additions in `lib.rs`: a section on ORT error construction FFI side effects (the `ortsys![CreateStatus]` dylib-load trigger that motivates the `OrtPathInvalid` split from `OrtLoadFailed`) and a section on fork safety (guidance for Python `multiprocessing` consumers to use `spawn` start method).
- `lib.rs` rustdoc "Feature flags" section cross-references `docs/features.md` for consumers wanting the deep-dive reference.
- Em-dash cleanup across 19 code locations in `lib.rs`, `capture.rs`, `output.rs`, `vad.rs`, `error.rs`, `vad_integration.rs`, and `vad_ort_load_failure.rs`. Per CLAUDE.md, the codebase forbids em dashes; these were pre-existing violations.
- CLAUDE.md corrections: validation-gate commands updated to the canonical set (`cargo clippy --workspace -- -D warnings`, `cargo fmt --all -- --check`, `cargo test-decibri`); stale `## [3.0.0] - Unreleased` reference replaced with a template placeholder.
- `ort` crate version unchanged at `2.0.0-rc.12`.
- Bundled ONNX Runtime version unchanged at `1.24.4`.
- No Node.js API signatures, event names, or error messages changed.
- TypeScript declaration files in both `npm/decibri/index.d.ts` (auto-regenerated) and `npm/decibri/src/decibri.d.ts` (hand-authored) updated for the new `id` field and extended `device` option type.

### Migration notes for direct Rust crate consumers

- Exhaustive matches on `DeviceSelector` will stop compiling. Add a `_ =>` catch-all arm. Display strings and existing variant names are unchanged; code using `to_string()` or only constructing variants (not matching them) continues to work unaffected.
- Struct literal construction of `DeviceInfo` and `OutputDeviceInfo` from outside the `decibri` crate will stop compiling (added `#[non_exhaustive]`, added `id: String` field). External consumers should read these structs from `enumerate_input_devices()` / `enumerate_output_devices()` rather than constructing them directly.
- Consumers matching specifically on `DecibriError::DeviceNotFound` for output-device misses should now also match `DecibriError::OutputDeviceNotFound`. The convenience predicate `DecibriError::is_ort_path_error` remains unchanged and already groups only ORT-path variants.
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
  Aligns with the docs.rs change effective 2026-05-01 (which builds fewer
  targets by default).

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
- JS API unchanged: drop-in replacement for v1.x consumers (verified against the in-house consumer surface)

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
