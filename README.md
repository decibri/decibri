<!-- markdownlint-disable MD033 MD041 -->
<div align="center">
  <img width="100" height="100" alt="decibri" src="https://github.com/user-attachments/assets/a1a7aa23-6954-4cac-9490-76354f0865ee" />

# decibri

  Cross-platform audio capture, output, and processing for Node.js and browsers.

  <!-- badges: start -->
  <table>
    <tr>
      <td><strong>Meta</strong></td>
      <td>
        <a href="https://www.npmjs.com/package/decibri"><img src="https://img.shields.io/npm/v/decibri" alt="npm version"></a>&nbsp;
        <a href="https://crates.io/crates/decibri"><img src="https://img.shields.io/crates/v/decibri" alt="crates.io version"></a>&nbsp;
        <a href="https://github.com/decibri/decibri/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>&nbsp;
        <a href="https://decibri.com"><img src="https://img.shields.io/badge/Website-decibri.com-blue" alt="decibri.com"></a>&nbsp;
        <a href="https://decibri.com/docs/"><img src="https://img.shields.io/badge/docs-passing-brightgreen" alt="Docs"></a>
      </td>
    </tr>
    <tr>
    <td><strong>Binaries</strong></td>
    <td>
      <a href="https://github.com/decibri/decibri/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/decibri/decibri/release.yml?job=Build%20x86_64-pc-windows-msvc&label=Windows%20x64&logo=microsoft&logoColor=white" alt="Windows x64"></a>&nbsp;
      <a href="https://github.com/decibri/decibri/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/decibri/decibri/release.yml?job=Build%20aarch64-apple-darwin&label=macOS%20ARM64&logo=apple" alt="macOS ARM64"></a>&nbsp;
      <a href="https://github.com/decibri/decibri/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/decibri/decibri/release.yml?job=Build%20x86_64-unknown-linux-gnu&label=Linux%20x64&logo=linux&logoColor=white" alt="Linux x64"></a>&nbsp;
      <a href="https://github.com/decibri/decibri/actions/workflows/release.yml"><img src="https://img.shields.io/github/actions/workflow/status/decibri/decibri/release.yml?job=Build%20aarch64-unknown-linux-gnu&label=Linux%20ARM64&logo=linux&logoColor=white" alt="Linux ARM64"></a>
    </td>
  </tr>
  <tr>
    <tr>
      <td><strong>Integrations</strong></td>
      <td>
        <a href="https://decibri.com/docs/node/integrations/openai-realtime"><img src="https://img.shields.io/badge/OpenAI-cloud-blue" alt="OpenAI"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/deepgram"><img src="https://img.shields.io/badge/Deepgram-cloud-blue" alt="Deepgram"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/assemblyai"><img src="https://img.shields.io/badge/AssemblyAI-cloud-blue" alt="AssemblyAI"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/mistral-voxtral"><img src="https://img.shields.io/badge/Mistral-cloud-blue" alt="Mistral"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/aws-transcribe"><img src="https://img.shields.io/badge/AWS_Transcribe-cloud-blue" alt="AWS Transcribe"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/google-speech"><img src="https://img.shields.io/badge/Google_Speech-cloud-blue" alt="Google Speech-to-Text"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/azure-speech"><img src="https://img.shields.io/badge/Azure_Speech-cloud-blue" alt="Azure Speech-to-Text"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/sherpa-onnx-stt"><img src="https://img.shields.io/badge/Sherpa--ONNX-local-brightgreen" alt="Sherpa-ONNX"></a>&nbsp;
        <a href="https://decibri.com/docs/node/integrations/whisper-cpp"><img src="https://img.shields.io/badge/Whisper.cpp-local-brightgreen" alt="Whisper.cpp"></a>
      </td>
    </tr>
  </table>
  <!-- badges: end -->
</div>

---

## Installation

```bash
npm install decibri
```

One package for Node.js and browsers. Node.js gets a native addon (Rust via napi-rs). Browsers get a JavaScript AudioWorklet implementation. Platform-specific binaries are installed automatically.

Requires Node.js >= 18. TypeScript definitions are bundled.

---

## Using from Rust

Most users want the npm package above. If you're building a Rust
application directly against the [`decibri`](https://crates.io/crates/decibri) crate on crates.io:

```bash
cargo add decibri
```

The `vad` feature (enabled by default) uses ONNX Runtime via the
[`ort`](https://crates.io/crates/ort) crate. From 3.1.0 onwards, `ort` is
configured with `load-dynamic` by default, meaning ONNX Runtime is loaded
from a shared library at runtime. Choose one of:

- **Zero-config builds.** Opt into the download-binaries mode:

  ```toml
  decibri = { version = "3.1", features = ["ort-download-binaries"], default-features = false }
  ```

  Add back any other features you need (`capture`, `output`, `vad`,
  `denoise`, `gain`). ORT is downloaded during `cargo build` and embedded
  in your binary (the 3.0.x behaviour).

- **Production deployments with bundled ORT.** Take default features,
  then either set `ORT_DYLIB_PATH=/path/to/libonnxruntime.{so,dylib,dll}`
  before first use, or call `ort::init_from(path).commit()` at startup.
  ONNX Runtime is initialized once per process. If your app creates
  multiple `SileroVad` instances with different paths, the first one
  wins and later paths are silently ignored. Pick one path and reuse it.

If you only need capture, output, or DSP (no Silero VAD), you won't pull
in `ort` at all:

```toml
decibri = { version = "3.1", features = ["capture", "output"], default-features = false }
```

See the [`ort` crate linking docs](https://ort.pyke.io/setup/linking)
for runtime-loading details.

---

## Quick Start

### Capture audio

```javascript
const Decibri = require('decibri');

const mic = new Decibri({ sampleRate: 16000, channels: 1 });
mic.on('data', (chunk) => { /* Buffer of Int16 PCM samples */ });
setTimeout(() => mic.stop(), 5000);
```

### Play audio

```javascript
const { DecibriOutput } = require('decibri');

const speaker = new DecibriOutput({ sampleRate: 16000, channels: 1 });
speaker.write(pcmBuffer);
speaker.end();
```

### Browser capture

```javascript
import { Decibri } from 'decibri'; // browser entry via conditional export

const mic = new Decibri({ sampleRate: 16000 });
mic.on('data', (chunk) => { /* Int16Array of PCM samples */ });
await mic.start(); // requires user gesture in Safari
```

### Pipe capture to playback (echo)

```javascript
const mic = new Decibri({ sampleRate: 16000, channels: 1 });
const speaker = new DecibriOutput({ sampleRate: 16000, channels: 1 });
mic.pipe(speaker);
```

---

## API: Decibri (Capture)

### `new Decibri(options?)`

Creates a `Readable` stream that captures from the microphone.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | `16000` | Samples per second (1000–384000) |
| `channels` | number | `1` | Input channels (1–32) |
| `framesPerBuffer` | number | `1600` | Frames per chunk (64–65536). At 16kHz mono, 1600 = 100ms = 3200 bytes |
| `device` | number \| string | system default | Device index or case-insensitive name substring |
| `format` | `'int16'` \| `'float32'` | `'int16'` | Sample encoding |
| `vad` | boolean | `false` | Enable voice activity detection |
| `vadMode` | `'energy'` \| `'silero'` | `'energy'` | VAD engine: RMS threshold or Silero ML model |
| `vadThreshold` | number | `0.01` / `0.5` | Speech threshold. Default depends on vadMode |
| `vadHoldoff` | number | `300` | Silence holdoff in ms |

Standard `ReadableOptions` (e.g. `highWaterMark`) are also accepted.

### Methods

| Method | Description |
| --- | --- |
| `mic.stop()` | Stop capture and end stream. Safe to call multiple times |
| `Decibri.devices()` | List available input devices |
| `Decibri.version()` | Version info: `{ decibri: '3.0.0', portaudio: 'cpal 0.17' }` |

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `mic.isOpen` | boolean | `true` while capturing |

### Events

| Event | Payload | Description |
| --- | --- | --- |
| `'data'` | `Buffer` | Audio chunk (Int16 LE or Float32 LE) |
| `'backpressure'` | - | Internal buffer full, consumer too slow |
| `'speech'` | - | VAD: audio crosses threshold |
| `'silence'` | - | VAD: audio below threshold for `vadHoldoff` ms |
| `'end'` | - | Stream ended |
| `'error'` | `Error` | An error occurred |

---

## API: DecibriOutput (Playback)

### `new DecibriOutput(options?)`

Creates a `Writable` stream for speaker playback.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | `16000` | Playback sample rate (1000–384000) |
| `channels` | number | `1` | Output channels (1–32) |
| `format` | `'int16'` \| `'float32'` | `'int16'` | Sample encoding of incoming data |
| `device` | number \| string | system default | Output device index or name substring |

Standard `WritableOptions` (e.g. `highWaterMark`) are also accepted.

| Method / Property | Description |
| --- | --- |
| `speaker.write(chunk)` | Write PCM data for playback |
| `speaker.end()` | Signal end. Drains remaining audio, then emits `'finish'` |
| `speaker.stop()` | Immediate stop. Discards remaining audio |
| `speaker.isPlaying` | `true` while audio is being output |
| `DecibriOutput.devices()` | List available output devices |
| `DecibriOutput.version()` | Same as `Decibri.version()` |

---

## API: Browser

The browser API uses `getUserMedia` and `AudioWorklet`. It differs from the Node.js API because browser audio is fundamentally async.

### `new Decibri(options?)`

Same options as Node.js, plus:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | string | system default | Device ID from `Decibri.devices()` (not index) |
| `echoCancellation` | boolean | `true` | Browser echo cancellation |
| `noiseSuppression` | boolean | `true` | Browser noise suppression |
| `workletUrl` | string | inline blob | Custom worklet URL for strict CSP |

### Key differences from Node.js

| Aspect | Node.js | Browser |
| --- | --- | --- |
| Start | Automatic on first read | `await mic.start()` |
| Base class | `Readable` stream | Custom `Emitter` |
| Data type | `Buffer` | `Int16Array` / `Float32Array` |
| `devices()` | Sync, returns array | Async, returns `Promise` |
| Sample rate | Direct via cpal | Resampled from native rate |

---

## Voice Activity Detection

### Energy mode (default)

Lightweight RMS energy threshold. No model required.

```javascript
const mic = new Decibri({ vad: true, vadThreshold: 0.01 });
mic.on('speech', () => console.log('speaking'));
mic.on('silence', () => console.log('silent'));
```

### Silero mode

ML-based detection using the Silero VAD v5 ONNX model. More accurate than energy mode, especially in noisy environments.

```javascript
const mic = new Decibri({ vad: true, vadMode: 'silero', vadThreshold: 0.5 });
mic.on('speech', () => console.log('speaking'));
mic.on('silence', () => console.log('silent'));
```

The Silero model (~2MB) ships inside the npm package. No downloads or API keys required.

---

## Device Selection

```javascript
// System default
const mic = new Decibri();

// By name (case-insensitive substring match)
const mic = new Decibri({ device: 'USB' });

// By index
const devices = Decibri.devices();
const mic = new Decibri({ device: devices[1].index });
```

```javascript
Decibri.devices();
// [
//   { index: 0, name: 'Microphone', maxInputChannels: 2, defaultSampleRate: 48000, isDefault: true },
//   { index: 1, name: 'USB Headset', maxInputChannels: 1, defaultSampleRate: 44100, isDefault: false }
// ]
```

---

## Examples

Runnable examples are in [`examples/`](examples/).

```bash
# Capture to WAV file (no dependencies)
node examples/wav-capture.js

# Stream to WebSocket (requires: npm install ws)
node examples/websocket-server.js   # terminal 1
node examples/websocket-stream.js   # terminal 2
```

Integration guides for OpenAI, Deepgram, AssemblyAI, and other providers are at [decibri.com/docs](https://decibri.com/docs/).

---

## Platform Support

| Platform | Architecture | Audio Backend |
| --- | --- | --- |
| Windows | x64 | WASAPI |
| macOS | arm64 | CoreAudio |
| Linux | x64 | ALSA |
| Linux | arm64 | ALSA |
| Browser | - | Web Audio API (AudioWorklet) |

---

## How It Works

decibri is a Rust library using [cpal](https://github.com/RustAudio/cpal) for cross-platform audio I/O. The Rust core compiles to a Node.js native addon via [napi-rs](https://napi.rs/) and ships pre-built binaries for each platform. Browser support uses a JavaScript AudioWorklet implementation with the same event-driven API.

Audio flows from the OS audio device through cpal's callback, into a crossbeam channel, through frame-exact buffering (guarantees consistent chunk sizes), and into Node.js via a threadsafe function. The JavaScript layer wraps this in a standard `Readable` stream.

---

## License

Apache-2.0 © [decibri](https://github.com/decibri)
