<!-- markdownlint-disable MD024 -->

# Decibri Python Changelog

This file tracks changes to the decibri Python package published to PyPI (`pip install decibri`).

For Rust core (`crates/decibri`) and npm binding (`bindings/node`) changes, see [the root CHANGELOG.md](../../CHANGELOG.md). The Rust core and npm binding ship together at the same version under the `v*` tag pattern; the Python package has its own version trajectory aligned with API surface stability and ships under the `python-v*` tag pattern.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] - 2026-06-12

### Fixed

- Multichannel capture is now downmixed to mono before the Silero VAD, so `vad_score` and `is_speaking` are correct on devices opened with more than one channel. Previously the interleaved multichannel audio reached the VAD as if it were mono, so it scored garbled input. Only affected explicit multichannel capture with VAD enabled (the default is one channel); audio returned from `read()` is unchanged.

### Added

- `Microphone.overrun_count`: a read-only property exposing the core stream's dropped-buffer counter. 0 while the consumer keeps pace; a rising value means audio is being dropped to bound memory.

## [0.4.2] - 2026-06-12

### Fixed

- `Microphone.stop()` called from a different thread than the one blocked inside `read()` no longer raises `RuntimeError` ("already borrowed"). The sync bridge now exposes `read()` / `stop()` without an exclusive borrow and shares the core stream behind a handle, so a cross-thread `stop()` interrupts the parked read (which then raises `MicrophoneStreamClosed`) and releases the device.
- `AsyncMicrophone.stop()` no longer deadlocks against a `read()` parked on a silent (non-delivering) device. The async wrapper no longer holds a bridge-wide lock across the blocking read, so a concurrent `await stop()` reaches the core stop (which wakes the parked read within roughly 20ms) instead of waiting behind it. The two are the sync and async halves of the same "stop() cannot safely interrupt an in-flight read" issue.

## [0.4.1] - 2026-06-11

### Fixed

- Picks up the Rust core hardening: `Speaker.drain()` / `AsyncSpeaker.drain()` no longer hang if the stream is dropped without `stop()`; malformed VAD models raise a typed error instead of crashing the process; and the fork-after-init guard now fires when a Silero VAD is constructed in a forked child, not only at inference.

### Changed

- The microphone capture channel is now bounded in the core, so a stalled reader drops audio to keep memory bounded rather than growing without limit.

## [0.4.0] - 2026-06-10

### Changed

- Picks up the decibri Rust core speaker stop/drain fix and device-release-on-stop. `Speaker.drain()` and `AsyncSpeaker.drain()` are now repeatable, non-terminal flushes: a later `drain()` waits for its own audio instead of returning immediately. `stop()` discards queued audio and releases the audio device. The Python `stop()` already dropped the stream, so its observable behavior is unchanged; the `drain()` repeatability is the user-visible improvement.

## [0.3.0] - 2026-06-07

### Fixed

- The Silero VAD never detected speech due to a missing 64-sample audio context required by Silero v5. VAD probabilities now reflect real speech activity. If you consume VAD output, expect meaningful probabilities where previously everything sat near zero.

## [0.2.1] - 2026-05-30

### Fixed

- Device-index error message. Passing an out-of-range device index no longer produces a message that names a class removed in the rename. The fix is picked up from the decibri Rust core 4.0.1; the message now points at the neutral `devices()` enumeration.

## [0.2.0] - 2026-05-30

### Python binding

0.2.0 aligns the Python API to decibri's canonical microphone/speaker vocabulary. It is a breaking release; see [`MIGRATION_PYTHON.md`](MIGRATION_PYTHON.md) for the upgrade guide.

#### Breaking

- Device-enumeration methods renamed to the short form: `Microphone.input_devices()` is now `Microphone.devices()`, `Speaker.output_devices()` is now `Speaker.devices()`, and the `AsyncMicrophone` and `AsyncSpeaker` mirrors. The module-level free functions `decibri.input_devices()` and `decibri.output_devices()` are unchanged, so code calling those needs no change.
- Exceptions renamed: `CaptureStreamClosed` is now `MicrophoneStreamClosed`, `OutputStreamClosed` is now `SpeakerStreamClosed`, `DeviceNotFound` is now `MicrophoneNotFound`, `OutputDeviceNotFound` is now `SpeakerNotFound`, and `NoOutputDeviceFound` is now `NoSpeakerFound`. The old names are removed, so `except` clauses and imports using them must update. The other exception names are unchanged.
- Device-info types renamed: `DeviceInfo` is now `MicrophoneInfo` and `OutputDeviceInfo` is now `SpeakerInfo`. The objects returned by `devices()` carry the same fields; only the type names changed, so this affects code that imports these names or uses `isinstance` against them.

#### Changed

- Migrated Python quickstart examples in root `README.md` and `bindings/python/README.md` to context manager (`with`) pattern. Aligns with decibri's internal test conventions (`bindings/python/tests/test_capture.py` uses `with Microphone(...) as d:` 8 times on normal-path tests), the existing async README style (`async with await decibri.AsyncMicrophone.open(...)`), and Python community precedent for resource-lifecycle types. Pattern A (explicit `start()` / `stop()`) is retained in the ecosystem guides (`docs/ecosystem/jupyter.md`, `docs/ecosystem/multiprocessing.md`) where lifecycle stepping is the point of the example.

#### Fixed

- Speaker quickstart in `bindings/python/README.md`: defined `audio_bytes` as 1 second of int16 silence at 24kHz (`b"\x00\x00" * 24000`) so the example is copy-paste runnable. Previously the example referenced `audio_bytes` without defining it, raising `NameError: name 'audio_bytes' is not defined` on first use.
- Root `README.md` Python quickstarts (capture and VAD): the iterator loops previously had no `break`, making the trailing `mic.stop()` calls unreachable in normal flow. The Pattern B migration above resolves this as a side effect; the `with` block guarantees cleanup regardless of how the loop body exits.

## [0.1.3] - 2026-05-07

### Python binding

Patch release adding source distribution publication, property-based and soak/leak tests, and a shift-left of the abi3 compliance gate to PR-time CI. Mechanical version bump only on the version-handling surface; full single-source-of-truth consolidation is out of scope for 0.1.x.

#### Added

- Source distribution (sdist) publication to PyPI alongside binary wheels. Users on platforms outside the 4-platform wheel matrix, security-conscious environments that require source builds, and consumers running `pip install --no-binary :all: decibri` now have an install path. The Silero VAD ONNX model is included in the sdist via the existing `[tool.maturin] include` directive (`format = ["sdist", "wheel"]`, pre-positioned in 0.1.0). Source-build prerequisites: Rust toolchain, a C++ toolchain (cpal transitive dep on macOS/Linux), and an ORT dylib at runtime (the wheel bundles ORT; sdist users supply via `ORT_DYLIB_PATH` or system-installed `onnxruntime`). Cohort majority pattern: pydantic-core, ruff, uv, tokenizers, polars all publish sdist; decibri was the lone holdout. New `build-sdist` job in `publish-pypi.yml` runs on Linux (sdist is platform-neutral), validates via `twine check`, and uploads alongside per-platform wheel artifacts. Both `publish-testpypi` and `publish-pypi` jobs depend on it and pull the sdist in their artifact glob.
- Hypothesis property tests for `Microphone` constructor validation at `bindings/python/tests/test_microphone_properties.py`. Five tests covering `sample_rate`, `channels`, `frames_per_buffer`, `dtype`, and the positive valid-input path. Each test runs `max_examples=20` Hypothesis-generated cases against the existing typed-exception contract (`SampleRateOutOfRange`, `ChannelsOutOfRange`, `FramesPerBufferOutOfRange`, `InvalidFormat`). The property under test is "validation always lands on a typed decibri exception, never a panic / abort / untyped exception under fuzz". Total file runtime ~1.5 seconds.
- Soak and leak detection tests at `bindings/python/tests/test_lifecycle_leak.py`. Three bounded loops (5-iteration warmup + 15-iteration measurement window) with `tracemalloc` (Python-side allocations) and `psutil` RSS sampling (native allocations under cpal / pyo3 / ort). Tests cover `Microphone`, `AsyncMicrophone`, and `Microphone(vad="silero")` construct/destroy cycles. Thresholds: 1 MB tracemalloc growth, 5 MB RSS growth (10 MB for the silero variant to absorb ORT allocator behavior). The silero test uses the `requires_bundled_ort` marker so it auto-skips on dev installs where `_ort/` is empty and runs in CI where the dylib is staged. Total combined runtime ~3 seconds locally.
- abi3 compliance gate shifted left from publish-time to PR-time. The `abi3audit --strict --verbose --assume-minimum-abi3 3.10` step in `python-ci.yml` mirrors the existing `publish-pypi.yml` step verbatim and runs after auditwheel / delocate repair on every PR matrix entry. Defense in depth: the existing `publish-pypi.yml` step is retained untouched, so a wheel that somehow escapes PR review still cannot reach PyPI without the gate. Pinned at `abi3audit==0.0.26` matching `publish-pypi.yml`; cross-workflow version bumps remain a single-touchpoint action.
- `hypothesis>=6.0` and `psutil>=5.9` added to `[dependency-groups] dev` in `bindings/python/pyproject.toml`. Both are dev-only; not propagated to the wheel's runtime dependencies.

#### Internal

- Documentation correction: `python-ci.yml` does not run abi3audit prior to 0.1.3. `publish-pypi.yml` carried the only abi3audit step until 0.1.3 shifted it left into PR-time CI.

## [0.1.2] - 2026-05-06

### Python binding

Patch release fixing a user-facing README bug, refreshing a dev-only transitive dependency flagged by Dependabot, cleaning up stale documentation prose, and splitting the Python wheel changelog out of the root `CHANGELOG.md`.

#### Fixed

- README quickstart example: `record_to_file()` parameter name is `duration_seconds=`, not `seconds=`. Three occurrences corrected in `bindings/python/README.md` (line 37 quickstart code block, lines 94-95 function summary list). Anyone copying the previous PyPI quickstart received `TypeError: record_to_file() got an unexpected keyword argument 'seconds'` on the first example.
- Stale documentation prose: "Until 0.1.0 ships to PyPI" rewritten to past tense in `bindings/python/docs/ecosystem/docker.md` and `bindings/python/docs/ecosystem/docker/Dockerfile.base`. decibri 0.1.0+ has shipped to PyPI; the multi-stage build-from-source variant is now framed as a fallback for custom toolchain or air-gapped builds, not the default path.

#### Internal

- `postcss` dev dependency refreshed from 8.5.9 to 8.5.14 via `npm audit fix`. Resolves Dependabot alert (GHSA-qx2v-qp2m-jg93: PostCSS XSS via unescaped `</style>` in CSS stringify output). Transitive dependency through `vitest` -> `vite` -> `postcss`; only present in `package-lock.json`, not shipped in production wheels or the npm binding.
- Added `Changelog` URL to `bindings/python/pyproject.toml` `[project.urls]` pointing to this file. PyPI Meta sidebar now shows a Changelog link that resolves to the per-binding changelog rather than the Rust+npm root changelog.

#### Architecture

- Python wheel CHANGELOG split from root `CHANGELOG.md`. Python entries (`[0.1.0]`, `[0.1.1]`, `[0.1.2]`) now live in this file at `bindings/python/CHANGELOG.md`. Root `CHANGELOG.md` continues to track the Rust core (`crates/decibri`) and npm binding (`bindings/node`), which ship together at the same version under the `v*` tag pattern. Aligns with cohort polyglot projects (tokenizers, polars, swc, rspack) where each binding maintains its own changelog. Rationale: PyPI release notes show only Python-relevant changes; same applies to crates.io and npm. Cross-link added to the root `CHANGELOG.md` header.

## [0.1.1] - 2026-05-04

### Python binding

Patch release fixing a duration bug in `record_to_file` and `async_record_to_file`.

#### Fixed

- `record_to_file` and `async_record_to_file` now produce WAV files with the correct duration. The 0.1.0 implementation computed `chunks_needed = duration_seconds * sample_rate / frames_per_buffer` and ran the loop that many times. On Windows WASAPI (and likely other backends), each `mic.read()` chunk delivers ~160 frames (one cpal callback's worth at default settings), not the 1600 frames implied by `frames_per_buffer`. The result was WAV files capturing ~10% of the requested duration. The fix counts actual frames written rather than chunks read, removing the dependency on `frames_per_buffer` matching the platform's actual callback size. `record_to_file(path, duration_seconds=1.0)` now produces a ~1.0s WAV file as documented (0.1.0: 0.098s; 0.1.1: ~1.008s; documented as "at most one chunk longer than requested").
- The internal `mic.read()` call in both helpers no longer passes `timeout_ms=2000`. The loop is now bounded by the frame count, so a deadline is unnecessary; matches the iterator pattern (`for chunk in mic`) which uses `timeout_ms=None` and works correctly for arbitrary durations.

#### Internal

- New regression test at `bindings/python/tests/test_record_to_file_duration.py`. Six test cases verify the WAV file's actual duration matches the `duration_seconds` parameter (within the documented "at most one chunk longer" tolerance). The test fails on 0.1.0 (capturing the bug); passes on 0.1.1 (confirming the fix).

## [0.1.0] - 2026-05-03

### Python binding

The first production-stable Python wheel of decibri. Following the 0.1.0a1 TestPyPI rehearsal, the README quickstart was corrected (the original `mic.read(N)` examples assumed N was a sample count; the actual signature is `read(timeout_ms=None)`, so quickstarts now use the iterator pattern or the `record_to_file` convenience helper). The README was also tightened for production publication: Decibri capitalized as a proper noun in titles and sentence starts, the audio-infrastructure positioning section deferred to the website docs, the `decibri[numpy]` extras note deferred to the website docs, the Compatibility table updated to list explicit Python versions (3.10, 3.11, 3.12, 3.13, 3.14), and Intel Mac install-from-source references removed (Apple platform deprecation; macos-13 dropped from the wheel matrix).

#### Added

- Public API: `Microphone`, `Speaker`, `AsyncMicrophone`, `AsyncSpeaker` classes.
- Module-level helpers: `input_devices()`, `output_devices()`, `version()`, `record_to_file()`, `async_record_to_file()`.
- Value types: `DeviceInfo`, `OutputDeviceInfo`, `VersionInfo`, `Chunk`.
- Exception hierarchy at `decibri.exceptions` (32 classes; 5 catch-target intermediates surfaced at `decibri.<X>`: `DecibriError`, `DeviceError`, `OrtError`, `OrtPathError`, `ForkAfterOrtInit`).
- VAD support: `vad="energy"` (RMS threshold) and `vad="silero"` (Silero ONNX, bundled). `vad_score` property returns a `[0, 1]` value mode-agnostically; `is_speaking` returns the boolean above-threshold view.
- NumPy ndarray return support via `as_ndarray=True` (install with `pip install decibri[numpy]`).
- Bundled Silero VAD ONNX model (~2.3 MB) shipped in the wheel. No downloads or API keys required.
- Bundled ONNX Runtime dylib per platform (~15-20 MB; Linux x64, Linux ARM64, macOS Apple Silicon, Windows x64). No system dependency on `pip install onnxruntime`.
- Async `AsyncMicrophone.open()` and `AsyncSpeaker.open()` factories that dispatch synchronous ORT init off the event loop via `loop.run_in_executor`. Recommended for `vad="silero"` in async contexts.
- `__repr__` on `Microphone`, `Speaker`, `AsyncMicrophone`, `AsyncSpeaker` showing construction parameters plus runtime state (closes the Jupyter auto-display gap).
- `ForkAfterOrtInit` exception raised on Linux when `Microphone(vad="silero")` is constructed in a parent process and then used in a forked child. Carries a remediation message pointing at `multiprocessing.set_start_method('spawn')`.
- `Chunk` dataclass and `Microphone.read_with_metadata()` / `iter_with_metadata()` returning a frozen `Chunk` with `.data`, `.timestamp`, `.sequence`, `.is_speaking`, `.vad_score`. Additive; `read()` keeps its current signature.
- Re-entry contract on `start()` / `stop()` / `close()` for all four wrapper classes pinned by tests. Calling `start()` after `stop()` reconstructs the stream cleanly; VAD state resets per `start()`. `close()` is a permanent alias for `stop()`.
- Ecosystem coexistence docs at `bindings/python/docs/ecosystem/{jupyter,docker,multiprocessing}.md` plus three reference Dockerfiles under `docker/`.

#### Internal

- 4-platform wheel matrix: Linux x64 (manylinux_2_28 container build), Linux ARM64, macOS Apple Silicon, Windows x64 (`macos-13` Intel Mac dropped on Apple platform deprecation).
- Trusted Publisher OIDC publish workflow at `.github/workflows/publish-pypi.yml`. Prerelease tags (`python-v*a*`, `python-v*b*`, `python-v*rc*`) route to TestPyPI; stable tags (`python-v\d+.\d+.\d+`) route to production PyPI.
- abi3audit (`--strict --assume-minimum-abi3 3.10`), auditwheel, and delocate gates in the publish pipeline.
- PEP 740 attestations generated via Sigstore + Fulcio in the publish job.
- In-job wheel install-test in a clean venv on each matrix platform with `CI=true` (conftest auto-skips hardware-gated tests).
- `OnnxSession` trait abstraction in Rust core. `crates/decibri` 3.4.0 introduces `pub(crate) trait OnnxSession`; `SileroVad` consumes it through `Box<dyn OnnxSession>`.
- New `DecibriError::ForkAfterOrtInit { init_pid, current_pid }` variant in `crates/decibri/src/error.rs`. Permitted by `#[non_exhaustive]`; existing variants unchanged.
- `bindings/python/docs/PUBLISH.md` documents the publish flow, Trusted Publisher setup, `workflow_dispatch` dry-run procedure, and failure recovery for abi3audit / install-test / OIDC rejection cases.
