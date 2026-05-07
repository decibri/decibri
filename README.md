<!-- markdownlint-disable MD033 MD041 -->
<div align="center">
  <img width="100" height="100" alt="decibri" src="https://github.com/user-attachments/assets/a1a7aa23-6954-4cac-9490-76354f0865ee" />

# decibri

Cross-platform audio capture, playback, and voice activity detection for Python, Rust, and Node.js.

<a href="https://pypi.org/project/decibri/"><img src="https://img.shields.io/pypi/v/decibri" alt="PyPI version"></a>&nbsp;
<a href="https://www.npmjs.com/package/decibri"><img src="https://img.shields.io/npm/v/decibri" alt="npm version"></a>&nbsp;
<a href="https://crates.io/crates/decibri"><img src="https://img.shields.io/crates/v/decibri" alt="crates.io version"></a>&nbsp;
<a href="https://github.com/decibri/decibri/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>&nbsp;
<a href="https://decibri.com"><img src="https://img.shields.io/badge/Website-decibri.com-blue" alt="decibri.com"></a>&nbsp;
<a href="https://decibri.com/docs/"><img src="https://img.shields.io/badge/docs-passing-brightgreen" alt="Docs"></a>

</div>

---

## What is decibri?

decibri is a cross-platform audio library for capture, playback, and voice activity detection. A single Rust core (using [cpal](https://github.com/RustAudio/cpal) for audio I/O and the [Silero VAD](https://github.com/snakers4/silero-vad) ONNX model for speech detection) ships through three first-class bindings: a Python wheel via PyO3, a Node.js native addon via napi-rs, and a browser AudioWorklet implementation under the same npm package.

## Features

- Microphone capture and speaker playback for Windows, macOS, Linux, and modern browsers
- Voice activity detection in two modes: lightweight RMS energy threshold or Silero ONNX (~2.3 MB; bundled, no API keys required)
- Async and streaming APIs in each binding (async iterators in Python, `Readable` streams in Node.js, channels in Rust)
- Pre-built binaries for Windows x64, macOS arm64, Linux x64, and Linux arm64; no system audio toolkit required on macOS or Windows; libasound2 only on Linux
- Cloud and local STT/TTS integration guides at [decibri.com/docs/integrations](https://decibri.com/docs/integrations) covering OpenAI Realtime, Deepgram, AssemblyAI, AWS Transcribe, Google Speech, Azure Speech, Sherpa-ONNX, Whisper.cpp, and more

## Quick Start

### Python

```python
import decibri

mic = decibri.Microphone(sample_rate=16000, vad="silero")
mic.start()
for chunk in mic:
    print(f"Got {len(chunk)} bytes; speaking={mic.is_speaking}")
    break  # exit after first chunk for demo
mic.stop()
```

Full Python guide: [bindings/python/README.md](bindings/python/README.md).

### Node.js

```javascript
const Decibri = require('decibri');

const mic = new Decibri({ sampleRate: 16000, vad: true, vadMode: 'silero' });
mic.on('data', (chunk) => console.log(`Got ${chunk.length} bytes`));
mic.on('speech', () => console.log('speaking'));
setTimeout(() => mic.stop(), 5000);
```

Full Node.js and browser guide: [npm/decibri/README.md](npm/decibri/README.md).

### Rust

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

Full Rust API: [crates/decibri/README.md](crates/decibri/README.md) and [docs.rs/decibri](https://docs.rs/decibri/).

## Installation

```bash
# Python
pip install decibri

# Node.js (and browsers)
npm install decibri

# Rust
cargo add decibri
```

Each binding ships pre-built binaries for the supported platforms below. No additional system audio libraries are required on Windows or macOS; Linux requires `libasound2` (preinstalled on most desktop distributions).

## Platform Support

| Platform | Architecture | Audio Backend | Python wheel | npm native | Rust crate |
| --- | --- | --- | --- | --- | --- |
| Windows | x64 | WASAPI | yes | yes | yes |
| macOS | arm64 (Apple Silicon) | CoreAudio | yes | yes | yes |
| Linux | x64 (gnu) | ALSA | yes | yes | yes |
| Linux | arm64 (gnu) | ALSA | yes | yes | yes |
| Browser | n/a | Web Audio API (AudioWorklet) | n/a | yes | n/a |

Intel macOS and Windows arm64 are not currently shipped as pre-built binaries. Source builds via `cargo build` work on those targets but are not part of the release matrix.

## Voice Activity Detection

decibri ships two VAD modes:

- **Energy mode** (`vad="energy"` / `vadMode: 'energy'`): lightweight RMS threshold; no model required.
- **Silero mode** (`vad="silero"` / `vadMode: 'silero'`): ML-based detection using the Silero VAD v5 ONNX model (~2.3 MB, bundled in every binding's distribution; no downloads or API keys required). More accurate than energy mode, especially in noisy environments.

```python
import decibri

mic = decibri.Microphone(sample_rate=16000, vad="silero", vad_threshold=0.5)
mic.start()
for chunk in mic:
    if mic.is_speaking:
        print(f"speech (score={mic.vad_score:.2f})")
    if some_stop_condition:
        break
mic.stop()
```

Language-specific VAD examples and tuning advice live in each binding's README.

## Architecture

The Rust core in [`crates/decibri`](crates/decibri/) handles audio I/O via [cpal](https://github.com/RustAudio/cpal) and Silero VAD inference via the [`ort`](https://crates.io/crates/ort) crate (ONNX Runtime). Three bindings expose this core to their respective ecosystems:

- The **Python wheel** ([`bindings/python`](bindings/python/)) is a PyO3 / abi3 binding compiled to a shared library; `pip install decibri` ships pre-built wheels for the four supported platforms.
- The **Node.js native addon** ([`bindings/node`](bindings/node/)) is a napi-rs binding compiled to a `.node` cdylib; the [`decibri`](npm/decibri/) npm package ships per-platform binaries via optional dependencies.
- The **browser implementation** ([`npm/decibri/src/browser`](npm/decibri/src/browser/)) is a parallel pure-JavaScript port using `getUserMedia` and `AudioWorklet`. It loads automatically from the same `decibri` npm package via conditional exports when the build target is a browser.

The Silero ONNX model is bundled inside every binding's distribution. ONNX Runtime is loaded dynamically (the `ort-load-dynamic` feature flag) by default; the npm and Python bindings ship the ORT shared library alongside the model.

## Documentation

- **Python guide**: [bindings/python/README.md](bindings/python/README.md) and [decibri.com/docs/](https://decibri.com/docs/)
- **Rust guide**: [crates/decibri/README.md](crates/decibri/README.md) and [docs.rs/decibri](https://docs.rs/decibri/)
- **Node.js and browser guide**: [npm/decibri/README.md](npm/decibri/README.md) and [decibri.com/docs/node/](https://decibri.com/docs/node/)
- **Integration guides** (cloud and local STT/TTS providers): [decibri.com/docs/integrations](https://decibri.com/docs/integrations)
- **Changelogs**: [CHANGELOG.md](CHANGELOG.md) (Rust core and npm), [bindings/python/CHANGELOG.md](bindings/python/CHANGELOG.md) (Python wheel)

## Contributing

Contributions, issue reports, and feature requests are welcome at [github.com/decibri/decibri/issues](https://github.com/decibri/decibri/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and how to run the cross-platform test suite.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Decibri.
