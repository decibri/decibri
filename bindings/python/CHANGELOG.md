<!-- markdownlint-disable MD024 -->

# Decibri Python Wheel Changelog

This file tracks changes to the decibri Python wheel published to PyPI (`pip install decibri`).

For Rust core (`crates/decibri`) and npm binding (`bindings/node`) changes, see [the root CHANGELOG.md](../../CHANGELOG.md). The Rust core and npm binding ship together at the same version under the `v*` tag pattern; the Python wheel has its own version trajectory aligned with API surface stability and ships under the `python-v*` tag pattern.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-05-07

### Python binding

Patch release adding source distribution publication, property-based and soak/leak tests, and a shift-left of the abi3 compliance gate to PR-time CI. Mechanical version bump only on the version-handling surface; full single-source-of-truth consolidation deferred to the 4.0 platform decoupling release.

#### Added

- Source distribution (sdist) publication to PyPI alongside binary wheels. Users on platforms outside the 4-platform wheel matrix, security-conscious environments that require source builds, and consumers running `pip install --no-binary :all: decibri` now have an install path. The Silero VAD ONNX model is included in the sdist via the existing `[tool.maturin] include` directive (`format = ["sdist", "wheel"]`, pre-positioned in 0.1.0). Source-build prerequisites: Rust toolchain, a C++ toolchain (cpal transitive dep on macOS/Linux), and an ORT dylib at runtime (the wheel bundles ORT; sdist users supply via `ORT_DYLIB_PATH` or system-installed `onnxruntime`). Cohort majority pattern: pydantic-core, ruff, uv, tokenizers, polars all publish sdist; decibri was the lone holdout. New `build-sdist` job in `publish-pypi.yml` runs on Linux (sdist is platform-neutral), validates via `twine check`, and uploads alongside per-platform wheel artifacts. Both `publish-testpypi` and `publish-pypi` jobs depend on it and pull the sdist in their artifact glob.
- Hypothesis property tests for `Microphone` constructor validation at `bindings/python/tests/test_microphone_properties.py`. Five tests covering `sample_rate`, `channels`, `frames_per_buffer`, `dtype`, and the positive valid-input path. Each test runs `max_examples=20` Hypothesis-generated cases against the existing typed-exception contract (`SampleRateOutOfRange`, `ChannelsOutOfRange`, `FramesPerBufferOutOfRange`, `InvalidFormat`). The property under test is "validation always lands on a typed decibri exception, never a panic / abort / untyped exception under fuzz". Total file runtime ~1.5 seconds. Tier 1 testing per `prerelease-decibri-additional-testing-reqs.md` Test 3.
- Soak and leak detection tests at `bindings/python/tests/test_lifecycle_leak.py`. Three bounded loops (5-iteration warmup + 15-iteration measurement window) with `tracemalloc` (Python-side allocations) and `psutil` RSS sampling (native allocations under cpal / pyo3 / ort). Tests cover `Microphone`, `AsyncMicrophone`, and `Microphone(vad="silero")` construct/destroy cycles. Thresholds: 1 MB tracemalloc growth, 5 MB RSS growth (10 MB for the silero variant to absorb ORT allocator behavior). The silero test uses the `requires_bundled_ort` marker so it auto-skips on dev installs where `_ort/` is empty and runs in CI where the dylib is staged. Total combined runtime ~3 seconds locally. Tier 1 testing per `prerelease-decibri-additional-testing-reqs.md` Test 4.
- abi3 compliance gate shifted left from publish-time to PR-time. The `abi3audit --strict --verbose --assume-minimum-abi3 3.10` step in `python-ci.yml` mirrors the existing `publish-pypi.yml` step verbatim and runs after auditwheel / delocate repair on every PR matrix entry. Defense in depth: the existing `publish-pypi.yml` step is retained untouched, so a wheel that somehow escapes PR review still cannot reach PyPI without the gate. Pinned at `abi3audit==0.0.26` matching `publish-pypi.yml`; cross-workflow version bumps remain a single-touchpoint action.
- `hypothesis>=6.0` and `psutil>=5.9` added to `[dependency-groups] dev` in `bindings/python/pyproject.toml`. Both are dev-only; not propagated to the wheel's runtime dependencies.

#### Internal

- Master plan `~/.claude/plans/python-integration-project.md` Section 3 directory-tree comment for `python-ci.yml` corrected. The prior comment incorrectly attributed an abi3audit addition to Phase 10 in `python-ci.yml`; Phase 10 actually created `publish-pypi.yml` as a separate workflow file and the abi3audit step lived only there until 0.1.3. Phase 10 section gained a "0.1.3 update" note documenting the shift-left.

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

The first production-stable Python wheel of decibri. Following the 0.1.0a1 TestPyPI rehearsal, the README quickstart was corrected (the original `mic.read(N)` examples assumed N was a sample count; the actual signature is `read(timeout_ms=None)`, so quickstarts now use the iterator pattern or the `record_to_file` convenience helper). The README was also tightened for production publication: Decibri capitalized as a proper noun in titles and sentence starts, the audio-infrastructure positioning section deferred to the website docs, the `decibri[numpy]` extras note deferred to the website docs, the Compatibility table updated to list explicit Python versions (3.10, 3.11, 3.12, 3.13, 3.14), and Intel Mac install-from-source references removed (Apple platform deprecation; macos-13 dropped per LD-10-12).

#### Added

- Public API: `Microphone`, `Speaker`, `AsyncMicrophone`, `AsyncSpeaker` classes (Phase 7.5 class rename; Phase 9 async-open factories).
- Module-level helpers: `input_devices()`, `output_devices()`, `version()`, `record_to_file()`, `async_record_to_file()`.
- Value types: `DeviceInfo`, `OutputDeviceInfo`, `VersionInfo`, `Chunk`.
- Exception hierarchy at `decibri.exceptions` (32 classes; 5 catch-target intermediates surfaced at `decibri.<X>`: `DecibriError`, `DeviceError`, `OrtError`, `OrtPathError`, `ForkAfterOrtInit`).
- VAD support: `vad="energy"` (RMS threshold) and `vad="silero"` (Silero ONNX, bundled). `vad_score` property returns a `[0, 1]` value mode-agnostically; `is_speaking` returns the boolean above-threshold view.
- NumPy ndarray return support via `as_ndarray=True` (install with `pip install decibri[numpy]`).
- Bundled Silero VAD ONNX model (~2.3 MB) shipped in the wheel. No downloads or API keys required.
- Bundled ONNX Runtime dylib per platform (~15-20 MB; Linux x64, Linux ARM64, macOS Apple Silicon, Windows x64). No system dependency on `pip install onnxruntime`.
- Async `AsyncMicrophone.open()` and `AsyncSpeaker.open()` factories that dispatch synchronous ORT init off the event loop via `loop.run_in_executor`. Recommended for `vad="silero"` in async contexts (Phase 9).
- `__repr__` on `Microphone`, `Speaker`, `AsyncMicrophone`, `AsyncSpeaker` showing construction parameters plus runtime state (Phase 9; closes the Jupyter auto-display gap).
- `ForkAfterOrtInit` exception raised on Linux when `Microphone(vad="silero")` is constructed in a parent process and then used in a forked child (Phase 9). Carries a remediation message pointing at `multiprocessing.set_start_method('spawn')`.
- `Chunk` dataclass and `Microphone.read_with_metadata()` / `iter_with_metadata()` returning a frozen `Chunk` with `.data`, `.timestamp`, `.sequence`, `.is_speaking`, `.vad_score`. Additive; `read()` keeps its current signature.
- Re-entry contract on `start()` / `stop()` / `close()` for all four wrapper classes pinned by tests. Calling `start()` after `stop()` reconstructs the stream cleanly; VAD state resets per `start()`. `close()` is a permanent alias for `stop()`.
- Ecosystem coexistence docs at `bindings/python/docs/ecosystem/{jupyter,docker,multiprocessing}.md` plus three reference Dockerfiles under `docker/` (Phase 9).

#### Internal

- 4-platform wheel matrix: Linux x64 (manylinux_2_28 container build), Linux ARM64, macOS Apple Silicon, Windows x64 (Phase 10; `macos-13` Intel Mac dropped per LD-10-12 Apple platform deprecation).
- Trusted Publisher OIDC publish workflow at `.github/workflows/publish-pypi.yml` (Phase 10). Prerelease tags (`python-v*a*`, `python-v*b*`, `python-v*rc*`) route to TestPyPI; stable tags (`python-v\d+.\d+.\d+`) route to production PyPI.
- abi3audit (`--strict --assume-minimum-abi3 3.10`), auditwheel, and delocate gates in the publish pipeline (Phase 10 LD-10-3 / LD-10-4).
- PEP 740 attestations generated via Sigstore + Fulcio in the publish job (Phase 10 LD-10-5).
- In-job wheel install-test in a clean venv on each matrix platform with `CI=true` (Phase 7 install gate; Phase 9 conftest auto-skips hardware-gated tests).
- `OnnxSession` trait abstraction in Rust core (Phase 8). `crates/decibri` 3.4.0 introduces `pub(crate) trait OnnxSession`; `SileroVad` consumes it through `Box<dyn OnnxSession>`. Future backends (CoreML, TFLite, GPU EPs) plug in additively at the `decibri-onnx` workspace split planned for 4.0.
- New `DecibriError::ForkAfterOrtInit { init_pid, current_pid }` variant in `crates/decibri/src/error.rs` (Phase 9). Permitted by `#[non_exhaustive]`; existing variants unchanged.
- `bindings/python/docs/PUBLISH.md` documents the publish flow, Trusted Publisher setup, `workflow_dispatch` dry-run procedure, and failure recovery for abi3audit / install-test / OIDC rejection cases.
