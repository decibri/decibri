# Decibri (Rust)

Cross-platform audio capture, playback, and voice activity detection for Rust applications.

[![crates.io](https://img.shields.io/crates/v/decibri)](https://crates.io/crates/decibri)
[![docs.rs](https://img.shields.io/docsrs/decibri)](https://docs.rs/decibri/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/decibri/decibri/blob/main/LICENSE)

This is the Rust core of decibri. The same core powers decibri's Python and Node.js packages; see the [main README](https://github.com/decibri/decibri) for cross-language context.

## Add to Cargo.toml

```bash
cargo add decibri
```

Or directly:

```toml
[dependencies]
decibri = "4"
```

The default feature set (`capture`, `playback`, `vad`, `denoise`, `gain`, `ort-load-dynamic`) covers most use cases. See [Feature flags](#feature-flags) below for opt-in or trimmed configurations.

## Quick start

The two primary types are `Microphone` for input and `Speaker` for output. Each follows the same shape: build a config, construct the object, then `start()` it to open the device and get a stream.

### Capture and detect speech

```rust
use std::time::Duration;
use decibri::{Microphone, MicrophoneConfig, SileroVad, VadConfig, DecibriError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microphone = Microphone::new(MicrophoneConfig::default())?;
    let stream = microphone.start()?;
    let mut vad = SileroVad::new(VadConfig::default())?;

    // 1600 interleaved samples = one 100 ms block of mono 16 kHz audio
    // (frames_per_buffer * channels for the default config).
    loop {
        match stream.next_chunk(1600, Some(Duration::from_millis(100))) {
            Ok(Some(chunk)) => {
                let result = vad.process(&chunk.data)?;
                if result.is_speech {
                    println!("speech @ p={:.2}", result.probability);
                }
            }
            Ok(None) => continue,
            Err(DecibriError::MicrophoneStreamClosed) => break,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
```

`MicrophoneStream::next_chunk` returns a three-state `Result`: `Ok(Some(chunk))` with data, `Ok(None)` on a timeout while the stream is still open, and `Err(DecibriError::MicrophoneStreamClosed)` once the stream ends.

### Speaker output

```rust
use decibri::{Speaker, SpeakerConfig};

let speaker = Speaker::new(SpeakerConfig::default())?;
let stream = speaker.start()?;

// f32 samples in [-1.0, 1.0]; here, one second of silence at 16 kHz mono.
let samples: Vec<f32> = vec![0.0; 16_000];
stream.send(samples)?;
stream.drain(); // block until everything queued has played
stream.stop();  // immediate; discards anything remaining
```

`SpeakerStream::send` takes a `Vec<f32>` of samples in the range [-1.0, 1.0]. `drain` blocks until the device has finished playing the queue; `stop` is immediate and discards what remains.

### List devices

```rust
use decibri::Microphone;

for device in Microphone::devices()? {
    println!("[{}] {}", device.index, device.name);
}
```

`Speaker::devices()` is the playback equivalent. The free functions `decibri::input_devices()` and `decibri::output_devices()` return the same lists.

### decibri ACE: capture conditioning

decibri ACE (Audio Capture Engine) is the opt-in conditioning chain on the capture path. `MicrophoneConfig` carries one field per stage; each defaults to off, so `MicrophoneConfig::default()` captures with no conditioning. The stages run in a fixed order after the device is normalized to the target mono rate: DC removal, denoise, high-pass, AGC, then limiter.

```rust
use decibri::{Microphone, MicrophoneConfig, DenoiseModel, HighpassFilter};

let mut config = MicrophoneConfig::default();
config.dc_removal = true;                            // 1. DC-blocking high-pass
config.denoise = Some(DenoiseModel::FastEnhancerT);  // 2. speech-enhancement model
config.denoise_model_path = Some("/path/to/fastenhancer_t.onnx".into()); // required for denoise to run
config.highpass = Some(HighpassFilter::Hz80);        // 3. 80 Hz high-pass
config.agc = Some(-18);                              // 4. AGC target, dBFS (-40..=-3)
config.limiter = Some(-1.0);                         // 5. limiter ceiling, dBFS (-3.0..=0.0)

let microphone = Microphone::new(config)?;
let stream = microphone.start()?;
```

The denoise stage needs the `denoise` feature and an explicit `denoise_model_path`: the crate ships no model bytes, and with the model named but no path the denoise stage stays off. This is the one place the crate asks for more than the bindings do. The Python and Node.js packages bundle the denoise model and resolve its path for you, so there you pass only the model selector. `agc` and `limiter` need the `gain` feature (pure DSP, no model); `dc_removal` and `highpass` need no extra feature. The AGC target and limiter ceiling are validated in `MicrophoneConfig::validate`, so an out-of-range value is a typed error, not a silent clamp. The high-pass cutoff and the denoise model are named enums (`HighpassFilter`, `DenoiseModel`) so adding further cutoffs or models stays a non-breaking widening.

### Without ONNX Runtime

If you do not need Silero VAD, disable the `vad` feature to drop the ONNX Runtime dependency entirely.

## Feature flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `capture` | on | Microphone input stream support |
| `playback` | on | Speaker output stream support |
| `vad` | on | Silero voice-activity detection (pulls in ONNX Runtime) |
| `denoise` | on | Capture-path single-channel speech enhancement (pulls in ONNX Runtime; runs only alongside `capture`) |
| `gain` | on | Capture-path AGC and peak limiter (pure DSP, no extra dependency) |
| `ort-load-dynamic` | on | ONNX Runtime loaded at runtime from a path you control |
| `ort-download-binaries` | off | ONNX Runtime downloaded at build time and embedded |

`ort-load-dynamic` and `ort-download-binaries` are mutually exclusive; selecting both is a compile error.

Execution-provider passthrough features (off by default; opt in for GPU acceleration on specific platforms): `coreml`, `cuda`, `directml`, `rocm`. See [`docs/features.md`](https://github.com/decibri/decibri/blob/main/docs/features.md) for the full reference.

### Common configurations

**Zero-config builds** (ONNX Runtime downloaded at `cargo build`, embedded):

```toml
decibri = { version = "4", features = ["ort-download-binaries"], default-features = false }
```

Add back the other features you want (`capture`, `playback`, `vad`).

**Production deployments with a bundled runtime** (default; ONNX Runtime loaded at runtime from a path you control):

```toml
decibri = "4"
```

Then either set `ORT_DYLIB_PATH=/path/to/libonnxruntime.{so,dylib,dll}` before first use, or call `ort::init_from(path).commit()` at startup. ONNX Runtime initializes exactly once per process; the first successful path wins and later constructions silently reuse it.

**Capture and playback without VAD** (no ONNX Runtime dependency at all):

```toml
decibri = { version = "4", features = ["capture", "playback"], default-features = false }
```

## ONNX Runtime configuration

Under `ort-load-dynamic` (default), the ONNX Runtime shared library is located at runtime via a `VadConfig::ort_library_path`, the `ORT_DYLIB_PATH` environment variable, or a process-wide `ort::init_from(...).commit()` call. decibri performs a filesystem-level path check before handing the path off, so a missing or wrong-arch library surfaces as `DecibriError::OrtPathInvalid` rather than hanging on dynamic load.

See the [ONNX Runtime linking docs](https://ort.pyke.io/setup/linking) for the full runtime-loading reference.

### Fork safety (Linux)

ONNX Runtime initialized in a parent process does not survive `fork()` cleanly: internal threads, memory pools, and device handles belonging to the parent may misbehave in the child. Python consumers should use `multiprocessing.set_start_method('spawn', force=True)` before importing decibri. Node.js consumers using `child_process` or `worker_threads` are unaffected.

## Public API surface

The stable 4.x surface includes:

- [`microphone::Microphone`](https://docs.rs/decibri/latest/decibri/microphone/struct.Microphone.html), [`microphone::MicrophoneConfig`](https://docs.rs/decibri/latest/decibri/microphone/struct.MicrophoneConfig.html), [`microphone::MicrophoneStream`](https://docs.rs/decibri/latest/decibri/microphone/struct.MicrophoneStream.html) (`try_next_chunk`, `next_chunk`, `is_open`, `stop`)
- [`speaker::Speaker`](https://docs.rs/decibri/latest/decibri/speaker/struct.Speaker.html), [`speaker::SpeakerConfig`](https://docs.rs/decibri/latest/decibri/speaker/struct.SpeakerConfig.html), [`speaker::SpeakerStream`](https://docs.rs/decibri/latest/decibri/speaker/struct.SpeakerStream.html) (`send`, `drain`, `is_playing`, `stop`)
- [`vad::SileroVad`](https://docs.rs/decibri/latest/decibri/vad/struct.SileroVad.html), [`vad::VadConfig`](https://docs.rs/decibri/latest/decibri/vad/struct.VadConfig.html), [`vad::VadResult`](https://docs.rs/decibri/latest/decibri/vad/struct.VadResult.html)
- [`device`](https://docs.rs/decibri/latest/decibri/device/index.html) module: `input_devices()` / `output_devices()`, `MicrophoneInfo` / `SpeakerInfo`, and selection by index, case-insensitive name substring, or stable per-host ID
- [`error::DecibriError`](https://docs.rs/decibri/latest/decibri/error/enum.DecibriError.html): `#[non_exhaustive]` enum covering capture, playback, device, ONNX Runtime, and fork-detection error variants

The common types are re-exported at the crate root, so `use decibri::Microphone;` works directly. Full API reference: [docs.rs/decibri](https://docs.rs/decibri/).

## Thread safety

All public types are `Send` and suitable for cross-thread handoff. `MicrophoneStream` and `SpeakerStream` are also `Sync` (they hold their live platform audio stream behind an internal mutex), so they can be shared across threads, for example via an `Arc`, without external synchronization.

## Platform support

| Platform | Architecture | Audio Backend |
| --- | --- | --- |
| Windows | x64 | WASAPI |
| macOS | arm64 (Apple Silicon) | CoreAudio |
| Linux | x64 (gnu) | ALSA |
| Linux | arm64 (gnu) | ALSA |

Source builds work on additional targets (Intel macOS, Windows arm64, musl Linux) but are not part of the npm and PyPI binary release matrix. The crate itself has no per-target restriction beyond what the platform audio host supports.

## Cross-references

- Project home: [github.com/decibri/decibri](https://github.com/decibri/decibri)
- Python package: [bindings/python/README.md](https://github.com/decibri/decibri/blob/main/bindings/python/README.md)
- Node.js and browser package: [npm/decibri/README.md](https://github.com/decibri/decibri/blob/main/npm/decibri/README.md)
- Full Rust API: [docs.rs/decibri](https://docs.rs/decibri/)
- Migration from 3.x: [MIGRATION.md](https://github.com/decibri/decibri/blob/main/crates/decibri/MIGRATION.md)
- Changelog: [CHANGELOG.md](https://github.com/decibri/decibri/blob/main/crates/decibri/CHANGELOG.md)
- ONNX Runtime linking docs: [ort.pyke.io/setup/linking](https://ort.pyke.io/setup/linking)

## License

Apache-2.0 (c) 2026 [Decibri](https://github.com/decibri/decibri).
