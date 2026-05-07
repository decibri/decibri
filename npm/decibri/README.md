# Decibri

Cross-platform audio capture, playback, and voice activity detection for Rust applications.

[![crates.io](https://img.shields.io/crates/v/decibri)](https://crates.io/crates/decibri)
[![docs.rs](https://img.shields.io/docsrs/decibri)](https://docs.rs/decibri/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/decibri/decibri/blob/main/LICENSE)

This is the Rust core of decibri. The same audio engine powers decibri's Python and Node.js bindings; see the [Main Project Readme](https://github.com/decibri/decibri) for the full project context.

## Add to Cargo.toml

```bash
cargo add decibri
```

Or directly:

```toml
[dependencies]
decibri = "3"
```

The default feature set (`capture`, `output`, `vad`, `denoise`, `gain`, `ort-load-dynamic`) covers most use cases. See [Feature flags](#feature-flags) below for opt-in or trimmed configurations.

## Quick Start

### Capture and run VAD

```rust
use std::time::Duration;
use decibri::capture::{AudioCapture, CaptureConfig};
use decibri::vad::{SileroVad, VadConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let capture = AudioCapture::new(CaptureConfig::default())?;
    let stream = capture.start()?;
    let mut vad = SileroVad::new(VadConfig::default())?;

    while let Ok(Some(chunk)) = stream.next_chunk(Some(Duration::from_millis(100))) {
        let result = vad.process(&chunk.data)?;
        if result.is_speech {
            println!("speech @ p={:.2}", result.probability);
        }
    }
    Ok(())
}
```

`CaptureStream::next_chunk` returns a three-state `Result`: `Ok(Some(chunk))` on data, `Ok(None)` on timeout (stream still open), and `Err(DecibriError::CaptureStreamClosed)` once the stream ends.

### Speaker output

```rust
use decibri::output::{AudioOutput, OutputConfig};

let output = AudioOutput::new(OutputConfig::default())?;
let stream = output.start()?;
stream.send(&pcm_int16_bytes)?;
stream.drain()?;
stream.stop()?;
```

`OutputStream::send` accepts byte slices in the format declared by `OutputConfig` (default: int16 LE). `drain` blocks until the device has finished playing; `stop` is immediate.

### Energy-mode VAD without ONNX Runtime

If you do not need Silero, the lightweight RMS energy mode in `vad` does not require ORT. You can also disable the `vad` feature entirely to avoid the `ort` dependency.

## Feature flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `capture` | on | Microphone input stream support |
| `output` | on | Speaker output stream support |
| `vad` | on | Silero VAD ONNX inference |
| `denoise` | on | Reserved (stub) |
| `gain` | on | Reserved (stub) |
| `ort-load-dynamic` | on | ORT loaded at runtime from a user-supplied path |
| `ort-download-binaries` | off | ORT downloaded at build time and embedded in the binary |

`ort-load-dynamic` and `ort-download-binaries` are mutually exclusive; selecting both is a compile error.

Execution-provider passthrough features (off by default; opt in for GPU acceleration on specific platforms): `coreml`, `cuda`, `directml`, `rocm`. See [docs/features.md](https://github.com/decibri/decibri/blob/main/docs/features.md) for the full reference.

### Common configurations

**Zero-config builds** (ORT downloaded at `cargo build`, embedded):

```toml
decibri = { version = "3", features = ["ort-download-binaries"], default-features = false }
```

Add back the other features you want (`capture`, `output`, `vad`, `denoise`, `gain`).

**Production deployments with bundled ORT** (default; ORT loaded at runtime from a path you control):

```toml
decibri = "3"
```

Then either set `ORT_DYLIB_PATH=/path/to/libonnxruntime.{so,dylib,dll}` before first use, or call `ort::init_from(path).commit()` at startup. ORT initializes exactly once per process; the first successful path wins and later constructions silently reuse it.

**Capture and output without VAD** (no `ort` dependency at all):

```toml
decibri = { version = "3", features = ["capture", "output"], default-features = false }
```

## ONNX Runtime configuration

Under `ort-load-dynamic` (default), the `ort` crate locates the ONNX Runtime shared library at runtime via either a `VadConfig::ort_library_path`, the `ORT_DYLIB_PATH` environment variable, or a process-wide `ort::init_from(...).commit()` call. Decibri performs a filesystem-level path check before handing the path to `ort::init_from`, so a missing or wrong-arch dylib surfaces as `DecibriError::OrtPathInvalid` rather than hanging on dynamic load.

See the [ort crate linking docs](https://ort.pyke.io/setup/linking) for the full runtime-loading reference.

### Fork safety (Linux)

ORT initialized in a parent process does not survive `fork()` cleanly: internal threads, memory pools, and device handles belonging to the parent may misbehave in the child. Decibri detects this at the start of every VAD inference call and returns `DecibriError::ForkAfterOrtInit` when the calling process ID does not match the process ID that initialized ORT.

## Public API surface

The 3.x stable surface includes:

- **Capture**: `AudioCapture`, `CaptureConfig`, `CaptureStream`
- **Output**: `AudioOutput`, `OutputConfig`, `OutputStream`
- **VAD**: `SileroVad`, `VadConfig`, `VadResult`
- **Device**: enumeration and selection helpers (by index, case-insensitive name substring, or stable per-host ID)
- **Errors**: `DecibriError` (`#[non_exhaustive]` enum covering capture, output, device, ORT, and fork-detection variants)

The `OnnxSession` trait abstracting ONNX Runtime is `pub(crate)` in 3.x and graduates to public at the planned 4.0 / `decibri-onnx` workspace split.

Full API reference: [docs.rs/decibri](https://docs.rs/decibri/).

## Thread safety

All public types are `Send` and suitable for cross-thread handoff. `CaptureStream` and `OutputStream` are `!Sync` because they hold a `cpal::Stream` internally; wrap them in a mutex or move them into a dedicated thread for shared access.

## Platform Support

| Platform | Architecture | Audio Backend |
| --- | --- | --- |
| Windows | x64 | WASAPI |
| macOS | arm64 (Apple Silicon) | CoreAudio |
| Linux | x64 (gnu) | ALSA |
| Linux | arm64 (gnu) | ALSA |

Source builds work on additional targets (Intel macOS, Windows arm64, musl Linux) but are not part of the binary release matrix.

## Cross-references

- Main Project Readme: [here](https://github.com/decibri/decibri)
- Python: [here](https://github.com/decibri/decibri/blob/main/bindings/python/README.md)
- Node.js and browser: [here](https://github.com/decibri/decibri/blob/main/npm/decibri/README.md)
- Full Rust API: [docs.rs/decibri](https://docs.rs/decibri/)
- Feature flag reference: [docs/features.md](https://github.com/decibri/decibri/blob/main/docs/features.md)
- CHANGELOG: [here](https://github.com/decibri/decibri/blob/main/CHANGELOG.md)
- ORT linking docs: [ort.pyke.io/setup/linking](https://ort.pyke.io/setup/linking)

## License

Apache-2.0 © 2026 [Decibri](https://github.com/decibri).
