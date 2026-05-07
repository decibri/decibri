<!-- markdownlint-disable MD024 -->

# Decibri

Cross-platform audio capture, playback, and voice activity detection for Node.js applications and modern browsers.

[![npm version](https://img.shields.io/npm/v/decibri)](https://www.npmjs.com/package/decibri)
[![npm downloads](https://img.shields.io/npm/dm/decibri)](https://www.npmjs.com/package/decibri)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/decibri/decibri/blob/main/LICENSE)

This is the Node.js and browser binding of decibri. The same audio engine powers decibri's Python and Rust bindings; see the [Main Project Readme](https://github.com/decibri/decibri) for the full project context.

## Installation

```bash
npm install decibri
```

One package serves both Node.js and browsers via conditional exports. Node.js gets a native addon. Browsers get a JavaScript [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet) implementation. Platform-specific binaries are installed automatically through optional dependencies.

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
| Sample rate | Direct | Resampled from native rate |

## Voice Activity Detection

Decibri ships two VAD modes:

- **Energy mode** (default): lightweight RMS energy threshold. No model required.
- **Silero mode**: ML-based detection using the Silero VAD v5 ONNX model. More accurate, especially in noisy environments. The model (~2.3 MB) ships inside the npm package; no downloads or API keys required.

```javascript
// Energy mode
const mic = new Decibri({ vad: true, vadThreshold: 0.01 });
mic.on('speech', () => console.log('speaking'));
mic.on('silence', () => console.log('silent'));

// Silero mode
const mic = new Decibri({ vad: true, vadMode: 'silero', vadThreshold: 0.5 });
mic.on('speech', () => console.log('speaking'));
mic.on('silence', () => console.log('silent'));
```

The same VAD modes are available in Decibri's Python and Rust bindings.

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

Runnable Node.js examples are in the [examples](https://github.com/decibri/decibri/tree/main/examples) directory of the repo.

Capture to WAV file (no dependencies):

```bash
node examples/wav-capture.js
```

Stream to WebSocket (requires `npm install ws`):

```bash
# Terminal 1
node examples/websocket-server.js

# Terminal 2
node examples/websocket-stream.js
```

## Integrations

Decibri's capture output and playback input pipe directly into cloud and local STT/TTS services. Integration guides at [decibri.com/docs/integrations](https://decibri.com/docs/integrations) cover:

| Provider | Environment | Guide |
| --- | --- | --- |
| OpenAI Realtime | Cloud | [Guide](https://decibri.com/docs/node/integrations/openai-realtime) |
| Deepgram | Cloud | [Guide](https://decibri.com/docs/node/integrations/deepgram) |
| AssemblyAI | Cloud | [Guide](https://decibri.com/docs/node/integrations/assemblyai) |
| Mistral Voxtral | Cloud | [Guide](https://decibri.com/docs/node/integrations/mistral-voxtral) |
| AWS Transcribe | Cloud | [Guide](https://decibri.com/docs/node/integrations/aws-transcribe) |
| Google Speech-to-Text | Cloud | [Guide](https://decibri.com/docs/node/integrations/google-speech) |
| Azure Speech-to-Text | Cloud | [Guide](https://decibri.com/docs/node/integrations/azure-speech) |
| Sherpa-ONNX | Local | [Guide](https://decibri.com/docs/node/integrations/sherpa-onnx-stt) |
| Whisper.cpp | Local | [Guide](https://decibri.com/docs/node/integrations/whisper-cpp) |

## Platform Support

| Platform | Architecture | Audio Backend |
| --- | --- | --- |
| Windows | x64 | WASAPI |
| macOS | arm64 (Apple Silicon) | CoreAudio |
| Linux | x64 (gnu) | ALSA |
| Linux | arm64 (gnu) | ALSA |
| Browser | n/a | Web Audio API (AudioWorklet) |

Pre-built native binaries ship via per-platform optional dependencies. The browser entry point is pure JavaScript and requires no native binary.

## How It Works

Decibri is built on a single Rust core. Audio flows from the OS audio device through the engine, through frame-exact buffering (which guarantees consistent chunk sizes), and into Node.js as a standard `Readable` stream. Browser support uses a JavaScript AudioWorklet implementation with the same event-driven API.

The same audio engine is also published to [crates.io](https://crates.io/crates/decibri) for direct Rust consumers and to [PyPI](https://pypi.org/project/decibri/) for Python consumers.

## Cross-references

- Main Project Readme: [here](https://github.com/decibri/decibri)
- Python: [here](https://github.com/decibri/decibri/blob/main/bindings/python/README.md)
- Rust: [here](https://github.com/decibri/decibri/blob/main/crates/decibri/README.md) (full API at [docs.rs/decibri](https://docs.rs/decibri/))
- CHANGELOG: [here](https://github.com/decibri/decibri/blob/main/CHANGELOG.md)

## License

Apache-2.0 © 2026 [Decibri](https://github.com/decibri).
