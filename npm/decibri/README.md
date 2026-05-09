# decibri

Cross-platform audio capture, output, and voice activity detection for Node.js and browsers.

## Installation

```bash
npm install decibri
```

One package for Node.js and browsers. Node.js gets a native addon (Rust via napi-rs). Browsers get a JavaScript AudioWorklet implementation. Platform-specific binaries are installed automatically.

Requires Node.js >= 18. TypeScript definitions are bundled.

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

## API: Decibri (Capture)

### `new Decibri(options?)`

Creates a Readable stream that captures from the microphone.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | 16000 | Samples per second (1000 to 384000) |
| `channels` | number | 1 | Input channels (1 to 32) |
| `framesPerBuffer` | number | 1600 | Frames per chunk (64 to 65536). At 16kHz mono, 1600 = 100ms = 3200 bytes |
| `device` | number, string, or `{ id: string }` | system default | Device index, case-insensitive name substring, or stable per-host ID |
| `format` | `'int16'` \| `'float32'` | `'int16'` | Sample encoding |
| `vad` | boolean | false | Enable voice activity detection |
| `vadMode` | `'energy'` \| `'silero'` | `'energy'` | VAD engine: RMS threshold or Silero ML model |
| `vadThreshold` | number | 0.01 / 0.5 | Speech threshold. Default depends on `vadMode` |
| `vadHoldoff` | number | 300 | Silence holdoff in ms |

Standard `ReadableOptions` (e.g. `highWaterMark`) are also accepted.

### Methods

| Method | Description |
| --- | --- |
| `mic.stop()` | Stop capture and end stream. Safe to call multiple times |
| `Decibri.devices()` | List available input devices |
| `Decibri.version()` | Version info |

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `mic.isOpen` | boolean | `true` while capturing |

### Events

| Event | Payload | Description |
| --- | --- | --- |
| `'data'` | Buffer | Audio chunk (Int16 LE or Float32 LE) |
| `'backpressure'` | - | Internal buffer full, consumer too slow |
| `'speech'` | - | VAD: audio crosses threshold |
| `'silence'` | - | VAD: audio below threshold for `vadHoldoff` ms |
| `'end'` | - | Stream ended |
| `'error'` | Error | An error occurred |

## API: DecibriOutput (Playback)

### `new DecibriOutput(options?)`

Creates a Writable stream for speaker playback.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | 16000 | Playback sample rate (1000 to 384000) |
| `channels` | number | 1 | Output channels (1 to 32) |
| `format` | `'int16'` \| `'float32'` | `'int16'` | Sample encoding of incoming data |
| `device` | number, string, or `{ id: string }` | system default | Output device index, case-insensitive name substring, or stable per-host ID |

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

### `new Decibri(options?)` (browser)

Same options as Node.js, plus:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | string | system default | Device ID from `Decibri.devices()` (not index) |
| `echoCancellation` | boolean | true | Browser echo cancellation |
| `noiseSuppression` | boolean | true | Browser noise suppression |
| `workletUrl` | string | inline blob | Custom worklet URL for strict CSP |

### Key differences from Node.js

| Aspect | Node.js | Browser |
| --- | --- | --- |
| Start | Automatic on first read | `await mic.start()` |
| Base class | Readable stream | Custom Emitter |
| Data type | Buffer | Int16Array / Float32Array |
| `devices()` | Sync, returns array | Async, returns Promise |
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

The Silero model (~2MB) ships inside the npm package. No downloads or API keys required.

## Device Selection

```javascript
// System default
const mic = new Decibri();

// By name (case-insensitive substring match)
const mic = new Decibri({ device: 'USB' });

// By index
const devices = Decibri.devices();
const mic = new Decibri({ device: devices[1].index });

// By stable per-host ID (survives across enumerations)
const mic = new Decibri({ device: { id: devices[1].id } });

Decibri.devices();
// [
//   { index: 0, name: 'Microphone', id: '{0.0.1.00000000}.{...}', maxInputChannels: 2, defaultSampleRate: 48000, isDefault: true },
//   { index: 1, name: 'USB Headset', id: '{0.0.1.00000000}.{...}', maxInputChannels: 1, defaultSampleRate: 44100, isDefault: false }
// ]
```

## Examples

Runnable examples ship with the package. After `npm install decibri`, find them under `node_modules/decibri/examples/`. From a clone of the repo they live at `npm/decibri/examples/`.

```bash
# Capture to WAV file (no extra dependencies)
node node_modules/decibri/examples/wav-capture.js

# Stream to WebSocket (requires: npm install ws)
node node_modules/decibri/examples/websocket-server.js   # terminal 1
node node_modules/decibri/examples/websocket-stream.js   # terminal 2
```

## Platform Support

| Platform | Architecture | Audio Backend |
| --- | --- | --- |
| Windows | x64 | WASAPI |
| macOS | arm64 | CoreAudio |
| Linux | x64 | ALSA |
| Linux | arm64 | ALSA |
| Browser | - | Web Audio API (AudioWorklet) |

## How It Works

decibri is a Rust library using cpal for cross-platform audio I/O. The Rust core compiles to a Node.js native addon via napi-rs and ships pre-built binaries for each platform. Browser support uses a JavaScript AudioWorklet implementation with the same event-driven API.

Audio flows from the OS audio device through cpal's callback, into a crossbeam channel, through frame-exact buffering (guarantees consistent chunk sizes), and into Node.js via a threadsafe function. The JavaScript layer wraps this in a standard Readable stream.

## Documentation

- **Source code & issues**: [github.com/decibri/decibri](https://github.com/decibri/decibri)
- **Provider integrations** (OpenAI, Deepgram, AssemblyAI): [decibri.com/docs](https://decibri.com/docs)

## License

Apache-2.0. See [LICENSE](https://github.com/decibri/decibri/blob/main/LICENSE) for details.

Copyright (c) 2026 Decibri.
