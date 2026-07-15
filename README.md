<!-- markdownlint-disable MD033 MD041 MD059 -->

<p align="center">
  <a href="https://decibri.com">
    <img
      src="https://github.com/user-attachments/assets/2d4aadca-a8fa-44d0-ad01-b15929d67c9f"
      alt="Decibri"
      width="100%">
  </a>
</p>

<div align="left">
  

# Decibri

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

Decibri is a cross-platform audio engine for real-time speech and voice applications. One unified API across Python, Rust, Node.js, and browsers.

Key features:

- Microphone capture and speaker playback
- Voice activity detection (Silero or energy modes)
- Async and streaming APIs
- Cross-platform: Windows, macOS, Linux, browsers
- Available in Python, Node.js, and Rust
- Cloud and local speech-to-text integrations

To learn more, visit [decibri.com/docs/](https://decibri.com/docs/).

## decibri ACE

Audio Capture Engine for Speech.

decibri ACE (Audio Capture Engine) is decibri's opt-in audio front-end for speech: it conditions microphone audio before it reaches your recognizer. Every stage is off by default, so a plain `Microphone(...)` is byte-identical to capture without ACE, and you turn on only the stages you want.

What makes it useful is not the individual filters, it is the package they come in:

- **Keyless and local.** No API key, no account, no network call. The chain runs on-device. The denoise model and the Silero VAD model ship inside the package.
- **Opt-in and zero-cost when off.** Nothing in the chain runs until you ask for it. With every stage left at its default, the capture path is unchanged.
- **One surface across the bindings.** The same conditioning options, with the same names and ranges, are available from Python, Node.js, and Rust.

ACE is a practical capture pipeline, not a claim to state-of-the-art signal processing. The stages are conventional and described plainly:

| Stage | Option | What it does |
| :--- | :--- | :--- |
| DC removal | `dc_removal` (bool) | Removes a constant (DC) offset with a one-pole DC-blocking high-pass. |
| Denoise | `denoise="fastenhancer-t"` | Optional bundled single-channel speech-enhancement model. |
| High-pass | `highpass=80` or `100` | A second-order Butterworth high-pass (80 or 100 Hz) that removes low-frequency rumble below the voice band. |
| AGC | `agc=-18` (dBFS, -40 to -3) | Drives the running level toward a target with a smoothed, rate-limited gain. |
| Limiter | `limiter=-1.0` (dBFS, -3.0 to 0.0) | A sample-peak ceiling that catches a transient the AGC would let exceed full scale. |

The stages run in that order: DC removal, denoise, high-pass, AGC, then limiter.

Voice activity detection reads the signal before the chain, so turning on enhancement does not change what counts as speech.

See the [Quick Start](#quick-start) below for a runnable example. The same options are available in Node.js (`dcRemoval`, `denoise`, `highpass`, `agc`, `limiter`) and Rust (`MicrophoneConfig` fields); per-language detail lives in each binding's README.

## Installation

**Python** (recommended with [uv](https://docs.astral.sh/uv/)):

```bash
uv pip install decibri
```

Or with pip:

```bash
pip install decibri
```

<br/>

**Node.js:**

```bash
npm install decibri
```

<br/>

**Rust:**

```bash
cargo add decibri
```

<br/>

## Quick Start

### Python

```python
import decibri

with decibri.Microphone(sample_rate=16000) as mic:
    for chunk in mic:
        print(f"Got {len(chunk)} bytes")
        break
```

With decibri ACE conditioning and Silero VAD in front (every stage is opt-in):

```python
import decibri

# Denoise and a high-pass in front of Silero VAD. Both are opt-in.
with decibri.Microphone(
    sample_rate=16000,
    denoise="fastenhancer-t",
    highpass=80,
    vad=decibri.Vad(model="silero", threshold=0.5),
) as mic:
    for chunk in mic:
        if mic.is_speaking:
            handle(chunk)  # conditioned audio, ready for your recognizer
```

Full Python guide: [here](bindings/python/README.md).

<br/>

### Node.js

```javascript
const { Microphone } = require('decibri');

const mic = new Microphone({ sampleRate: 16000 });
mic.on('data', (chunk) => console.log(`Got ${chunk.length} bytes`));
setTimeout(() => mic.stop(), 5000);
```

Full Node.js and browser guide: [here](npm/decibri/README.md).

<br/>

### Rust

```rust
use decibri::{Microphone, MicrophoneConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microphone = Microphone::new(MicrophoneConfig::default())?;
    let stream = microphone.start()?;
    // 1600 interleaved samples per block = frames_per_buffer * channels for the
    // default config (one 100 ms block of mono 16 kHz audio).
    while let Ok(Some(chunk)) = stream.next_chunk(1600, None) {
        println!("Got {} samples", chunk.data.len());
    }
    Ok(())
}
```

Full Rust guide: [here](crates/decibri/README.md).

<br/>

## Platform Support

Supports:

- Windows x64
- macOS arm64
- Linux x64
- Linux arm64

<br/>

## Voice Activity Detection

Decibri ships two VAD modes:

- **Energy mode**: lightweight RMS threshold.
- **Silero mode**: ML-based detection using the Silero VAD v5 ONNX model. More accurate, especially in noisy environments. The model (~2.3 MB) is bundled; no downloads or API keys required.

```python
import decibri

with decibri.Microphone(sample_rate=16000, vad=decibri.Vad(model="silero", threshold=0.5)) as mic:
    for chunk in mic:
        print(f"VAD score {mic.vad_score}; speaking={mic.is_speaking}")
        break
```

The same VAD API is available in Node.js and Rust. Language-specific examples and tuning advice live in each binding's README.

<br/>

## Integrations

Decibri pipes microphone audio directly into cloud and local speech services. Full guides at [decibri.com/docs/integrations](https://decibri.com/docs/integrations).

| Provider | Environment | STT | TTS | VAD | KWS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| OpenAI Realtime | Cloud | ✅ | ✅ | | |
| Deepgram | Cloud | ✅ | ✅ | | |
| AssemblyAI | Cloud | ✅ | | | |
| Mistral Voxtral | Cloud | ✅ | | | |
| AWS Transcribe | Cloud | ✅ | | | |
| Google Speech | Cloud | ✅ | ✅ | | |
| Azure Speech | Cloud | ✅ | ✅ | | |
| Sherpa-ONNX | Local | ✅ | ✅ | ✅ | ✅ |
| Whisper.cpp | Local | ✅ | | | |

<br/>

## Architecture

Decibri is built on a single Rust core. The same audio engine powers all three language bindings, ensuring consistent behavior regardless of which language you use.

Python and Node.js install pre-built native binaries for your platform. Browsers run the same API via a JavaScript implementation that ships inside the npm package.

<br/>

## Contributing

[Issues](https://github.com/decibri/decibri/issues) and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

<br/>

## License

Apache-2.0 © 2026 [Decibri](https://github.com/decibri).
