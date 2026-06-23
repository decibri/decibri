<!-- markdownlint-disable MD033 MD041 MD059 -->
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

Decibri is a cross-platform audio engine for real-time speech and voice applications. One unified API across Python, Rust, Node.js, and browsers.

Key features:

- Microphone capture and speaker playback
- Voice activity detection (Silero or energy modes)
- Async and streaming APIs
- Cross-platform: Windows, macOS, Linux, browsers
- Available in Python, Node.js, and Rust
- Cloud and local speech-to-text integrations

To learn more, visit [decibri.com/docs/](https://decibri.com/docs/).

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
