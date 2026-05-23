# Decibri (Rust)

Cross-platform audio capture, playback, and voice activity detection for Rust applications.

[![crates.io](https://img.shields.io/crates/v/decibri)](https://crates.io/crates/decibri)
[![docs.rs](https://img.shields.io/docsrs/decibri)](https://docs.rs/decibri/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/decibri/decibri/blob/main/LICENSE)

This is the Rust core of decibri. The same crate powers decibri's Python wheel (via PyO3) and Node.js native addon (via napi-rs); see the [main README](https://github.com/decibri/decibri) for cross-language project context.

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
let pcm_int16_bytes: Vec<u8> = vec![0; 48_000]; // 1 second of silence at 24kHz int16 mono
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
| `denoise` | on | On by default; no runtime cost when off |
| `gain` | on | On by default; no runtime cost when off |
| `ort-load-dynamic` | on | ORT loaded at runtime from a user-supplied path |
| `ort-download-binaries` | off | ORT downloaded at build time and embedded in the binary |

`ort-load-dynamic` and `ort-download-binaries` are mutually exclusive; selecting both is a compile error.

Execution-provider passthrough features (off by default; opt in for GPU acceleration on specific platforms): `coreml`, `cuda`, `directml`, `rocm`. See [`docs/features.md`](https://github.com/decibri/decibri/blob/main/docs/features.md) for the full reference.

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

Under `ort-load-dynamic` (default), the `ort` crate locates the ONNX Runtime shared library at runtime via either a `VadConfig::ort_library_path`, the `ORT_DYLIB_PATH` environment variable, or a process-wide `ort::init_from(...).commit()` call. decibri performs a filesystem-level path check before handing the path to `ort::init_from`, so a missing or wrong-arch dylib surfaces as `DecibriError::OrtPathInvalid` rather than hanging on dynamic load.

See the [`ort` crate linking docs](https://ort.pyke.io/setup/linking) for the full runtime-loading reference.

### Fork safety (Linux)

ORT initialized in a parent process does not survive `fork()` cleanly: internal threads, memory pools, and device handles belonging to the parent may misbehave in the child. Python consumers should use `multiprocessing.set_start_method('spawn', force=True)` before importing decibri. Node.js consumers using `child_process` or `worker_threads` are unaffected.

## Public API surface

The 3.x stable surface includes:

- [`capture::AudioCapture`](https://docs.rs/decibri/latest/decibri/capture/struct.AudioCapture.html), [`capture::CaptureConfig`](https://docs.rs/decibri/latest/decibri/capture/struct.CaptureConfig.html), [`capture::CaptureStream`](https://docs.rs/decibri/latest/decibri/capture/struct.CaptureStream.html) (`try_next_chunk`, `next_chunk`, `is_open`, `stop`)
- [`output::AudioOutput`](https://docs.rs/decibri/latest/decibri/output/struct.AudioOutput.html), [`output::OutputConfig`](https://docs.rs/decibri/latest/decibri/output/struct.OutputConfig.html), [`output::OutputStream`](https://docs.rs/decibri/latest/decibri/output/struct.OutputStream.html) (`send`, `drain`, `is_playing`, `stop`)
- [`vad::SileroVad`](https://docs.rs/decibri/latest/decibri/vad/struct.SileroVad.html), [`vad::VadConfig`](https://docs.rs/decibri/latest/decibri/vad/struct.VadConfig.html), [`vad::VadResult`](https://docs.rs/decibri/latest/decibri/vad/struct.VadResult.html)
- [`device`](https://docs.rs/decibri/latest/decibri/device/index.html) module: device enumeration and selection by index, case-insensitive name substring, or stable per-host ID
- [`error::DecibriError`](https://docs.rs/decibri/latest/decibri/error/enum.DecibriError.html): `#[non_exhaustive]` enum covering capture, output, device, ORT, and fork-detection error variants

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

Source builds work on additional targets (Intel macOS, Windows arm64, musl Linux) but are not part of the npm and PyPI binary release matrix. The Rust crate itself has no per-target restriction beyond cpal's host support.

## Cross-references

- **Polyglot project context**: [github.com/decibri/decibri](https://github.com/decibri/decibri)
- **Python wheel**: [bindings/python/README.md](https://github.com/decibri/decibri/blob/main/bindings/python/README.md)
- **Node.js and browser package**: [npm/decibri/README.md](https://github.com/decibri/decibri/blob/main/npm/decibri/README.md)
- **Full Rust API**: [docs.rs/decibri](https://docs.rs/decibri/)
- **Feature flag reference**: [docs/features.md](https://github.com/decibri/decibri/blob/main/docs/features.md)
- **CHANGELOG**: [CHANGELOG.md](https://github.com/decibri/decibri/blob/main/CHANGELOG.md)
- **ORT linking docs**: [ort.pyke.io/setup/linking](https://ort.pyke.io/setup/linking)

## License

Apache-2.0. See [LICENSE](https://github.com/decibri/decibri/blob/main/LICENSE) for details.

Copyright (c) 2026 Decibri.
