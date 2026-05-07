<!-- markdownlint-disable MD024 -->

# decibri (Node.js and browsers)

Cross-platform audio capture, playback, and voice activity detection for Node.js applications and modern browsers.

[![npm version](https://img.shields.io/npm/v/decibri)](https://www.npmjs.com/package/decibri)
[![npm downloads](https://img.shields.io/npm/dm/decibri)](https://www.npmjs.com/package/decibri)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/decibri/decibri/blob/main/LICENSE)

This is the Node.js and browser binding of decibri. The same Rust core powers decibri's Python wheel and Rust crate; see the [main README](https://github.com/decibri/decibri) for the polyglot project context.

## Installation

```bash
npm install decibri
```

One package serves both Node.js and browsers via conditional exports. Node.js gets a native addon (Rust via [napi-rs](https://napi.rs/)). Browsers get a JavaScript [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet) implementation. Platform-specific binaries are installed automatically through optional dependencies.

Requires Node.js >= 18 for Node.js consumers. TypeScript definitions are bundled.

## Quick Start

### Capture audio (Node.js)

```javascript
const Decibri = require('decibri');

const mic = new Decibri({ sampleRate: 16000, channels: 1 });
mic.on('data', (chunk) => { /* Buffer of Int16 PCM samples */ });
setTimeout(() => mic.stop(), 5000);
```

### Play audio (Node.js)

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

## API: Decibri (Capture)

### `new Decibri(options?)`

Creates a `Readable` stream that captures from the microphone.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | `16000` | Samples per second (1000 to 384000) |
| `channels` | number | `1` | Input channels (1 to 32) |
| `framesPerBuffer` | number | `1600` | Frames per chunk (64 to 65536). At 16kHz mono, 1600 = 100ms = 3200 bytes |
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
| `Decibri.version()` | Version info: `{ decibri: '3.x.y', portaudio: 'cpal 0.17' }` |

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `mic.isOpen` | boolean | `true` while capturing |

### Events

| Event | Payload | Description |
| --- | --- | --- |
| `'data'` | `Buffer` | Audio chunk (Int16 LE or Float32 LE) |
| `'backpressure'` | (none) | Internal buffer full, consumer too slow |
| `'speech'` | (none) | VAD: audio crosses threshold |
| `'silence'` | (none) | VAD: audio below threshold for `vadHoldoff` ms |
| `'end'` | (none) | Stream ended |
| `'error'` | `Error` | An error occurred |

## API: DecibriOutput (Playback)

### `new DecibriOutput(options?)`

Creates a `Writable` stream for speaker playback.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | `16000` | Playback sample rate (1000 to 384000) |
| `channels` | number | `1` | Output channels (1 to 32) |
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

## Browser API

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

The Silero model (~2.3 MB) ships inside the npm package. No downloads or API keys required.

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

## Examples

Runnable Node.js examples are in the [`examples/`](https://github.com/decibri/decibri/tree/main/examples) directory of the repo.

```bash
# Capture to WAV file (no dependencies)
node examples/wav-capture.js

# Stream to WebSocket (requires: npm install ws)
node examples/websocket-server.js   # terminal 1
node examples/websocket-stream.js   # terminal 2
```

## Integrations

decibri's capture output and playback input can pipe directly into cloud and local STT/TTS services. Integration guides at [decibri.com/docs/integrations](https://decibri.com/docs/integrations) cover:

| Provider | Type | Guide |
| --- | --- | --- |
| OpenAI Realtime | cloud | [decibri.com/docs/node/integrations/openai-realtime](https://decibri.com/docs/node/integrations/openai-realtime) |
| Deepgram | cloud | [decibri.com/docs/node/integrations/deepgram](https://decibri.com/docs/node/integrations/deepgram) |
| AssemblyAI | cloud | [decibri.com/docs/node/integrations/assemblyai](https://decibri.com/docs/node/integrations/assemblyai) |
| Mistral Voxtral | cloud | [decibri.com/docs/node/integrations/mistral-voxtral](https://decibri.com/docs/node/integrations/mistral-voxtral) |
| AWS Transcribe | cloud | [decibri.com/docs/node/integrations/aws-transcribe](https://decibri.com/docs/node/integrations/aws-transcribe) |
| Google Speech-to-Text | cloud | [decibri.com/docs/node/integrations/google-speech](https://decibri.com/docs/node/integrations/google-speech) |
| Azure Speech-to-Text | cloud | [decibri.com/docs/node/integrations/azure-speech](https://decibri.com/docs/node/integrations/azure-speech) |
| Sherpa-ONNX | local | [decibri.com/docs/node/integrations/sherpa-onnx-stt](https://decibri.com/docs/node/integrations/sherpa-onnx-stt) |
| Whisper.cpp | local | [decibri.com/docs/node/integrations/whisper-cpp](https://decibri.com/docs/node/integrations/whisper-cpp) |

## Platform Support

| Platform | Architecture | Audio Backend |
| --- | --- | --- |
| Windows | x64 | WASAPI |
| macOS | arm64 (Apple Silicon) | CoreAudio |
| Linux | x64 (gnu) | ALSA |
| Linux | arm64 (gnu) | ALSA |
| Browser | n/a | Web Audio API (AudioWorklet) |

Pre-built native binaries ship via per-platform optional dependencies (`@decibri/decibri-win32-x64-msvc`, `@decibri/decibri-darwin-arm64`, `@decibri/decibri-linux-x64-gnu`, `@decibri/decibri-linux-arm64-gnu`). The browser entry point is pure JavaScript and requires no native binary.

## How It Works

decibri is a Rust library using [cpal](https://github.com/RustAudio/cpal) for cross-platform audio I/O. The Rust core compiles to a Node.js native addon via [napi-rs](https://napi.rs/) and ships pre-built binaries for each platform. Browser support uses a JavaScript AudioWorklet implementation with the same event-driven API.

Audio flows from the OS audio device through cpal's callback, into a [crossbeam-channel](https://github.com/crossbeam-rs/crossbeam), through frame-exact buffering (guarantees consistent chunk sizes), and into Node.js via a threadsafe function. The JavaScript layer wraps this in a standard `Readable` stream.

The same Rust core is also published to [crates.io](https://crates.io/crates/decibri) for direct Rust consumers and to [PyPI](https://pypi.org/project/decibri/) (via PyO3) for Python consumers.

## Cross-references

- **Polyglot project context**: [github.com/decibri/decibri](https://github.com/decibri/decibri)
- **Python wheel**: [bindings/python/README.md](https://github.com/decibri/decibri/blob/main/bindings/python/README.md)
- **Rust crate**: [crates/decibri/README.md](https://github.com/decibri/decibri/blob/main/crates/decibri/README.md) and [docs.rs/decibri](https://docs.rs/decibri/)
- **CHANGELOG**: [CHANGELOG.md](https://github.com/decibri/decibri/blob/main/CHANGELOG.md)
- **TypeScript types**: bundled in the npm package; types entry is `src/decibri.d.ts`

## License

Apache-2.0. See [LICENSE](https://github.com/decibri/decibri/blob/main/LICENSE) for details.

Copyright (c) 2026 Decibri.
