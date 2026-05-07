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

decibri is an audio library for real-time speech and voice applications. It is designed to be fast, cross-platform, and consistent across languages. Key features are:

- Microphone capture and speaker playback
- Voice activity detection (Silero or energy modes)
- Async and streaming APIs
- Cross-platform: Windows, macOS, Linux, browsers
- Available in Python, Node.js, and Rust
- Cloud and local speech-to-text integrations

To learn more, visit [decibri.com/docs/](https://decibri.com/docs/).

## Quick Start

### Python

```python
import decibri

mic = decibri.Microphone(sample_rate=16000, vad="silero")
mic.start()
for chunk in mic:
    print(f"Got {len(chunk)} bytes; speaking={mic.is_speaking}")
    break
mic.stop()
```

Full Python guide: [here](bindings/python/README.md).

### Node.js

```javascript
const Decibri = require('decibri');

const mic = new Decibri({ sampleRate: 16000, vad: true, vadMode: 'silero' });
mic.on('data', (chunk) => console.log(`Got ${chunk.length} bytes`));
mic.on('speech', () => console.log('speaking'));
setTimeout(() => mic.stop(), 5000);
```

Full Node.js and browser guide: [here](npm/decibri/README.md).

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

Full Rust API: [here](crates/decibri/README.md).

## Installation

**Python:**

```bash
pip install decibri
```

**Node.js:**

```bash
npm install decibri
```

**Rust:**

```bash
cargo add decibri
```

## Platform Support

Supports:

- Windows x64
- macOS arm64
- Linux x64
- Linux arm64

## Voice Activity Detection

decibri ships two VAD modes:

- **Energy mode**: lightweight RMS threshold.
- **Silero mode**: ML-based detection using the Silero VAD v5 ONNX model. More accurate, especially in noisy environments. The model (~2.3 MB) is bundled; no downloads or API keys required.

```python
import decibri

mic = decibri.Microphone(sample_rate=16000, vad="silero", vad_threshold=0.5)
mic.start()
for chunk in mic:
    if mic.is_speaking:
        print(f"speech (score={mic.vad_score:.2f})")
    break  # demo: exit after first chunk
mic.stop()
```

Language-specific VAD examples and tuning advice live in each binding's README.

## Integrations

decibri pipes microphone audio directly into cloud and local speech services. Full guides at [decibri.com/docs/integrations](https://decibri.com/docs/integrations).

| Provider | Capability | Supported |
| --- | --- | --- |
| OpenAI Realtime | STT (cloud) | ✓ |
| Deepgram | STT (cloud) | ✓ |
| AssemblyAI | STT (cloud) | ✓ |
| Mistral Voxtral | STT (cloud) | ✓ |
| AWS Transcribe | STT (cloud) | ✓ |
| Google Speech | STT (cloud) | ✓ |
| Azure Speech | STT (cloud) | ✓ |
| Sherpa-ONNX | STT (local) | ✓ |
| Whisper.cpp | STT (local) | ✓ |

## Architecture

decibri is built on a single Rust core with bindings for Python, Node.js, and Rust. Browsers run the same API via a JavaScript implementation in the npm package.

## Documentation

- Python guide: [here](bindings/python/README.md)
- Rust guide: [here](crates/decibri/README.md) (full API at [docs.rs/decibri](https://docs.rs/decibri/))
- Node.js and browser guide: [here](npm/decibri/README.md)
- Integration guides: [decibri.com/docs/integrations](https://decibri.com/docs/integrations)

## Contributing

Issues and PRs welcome at [github.com/decibri/decibri/issues](https://github.com/decibri/decibri/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Decibri.
