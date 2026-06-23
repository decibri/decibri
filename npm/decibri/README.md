# decibri

Cross-platform audio capture, playback, and voice activity detection for Node.js and browsers.

## Installation

```bash
npm install decibri
```

One package for Node.js and browsers. Node.js gets a prebuilt native addon. Browsers get a JavaScript AudioWorklet implementation. Platform-specific binaries are installed automatically.

Requires Node.js >= 18. TypeScript definitions are bundled.

The package uses named exports:

```javascript
const { Microphone, Speaker, inputDevices, outputDevices, version } = require('decibri');
```

## Quick Start

### Capture audio

```javascript
const { Microphone } = require('decibri');

const mic = new Microphone({ sampleRate: 16000, channels: 1 });
mic.on('data', (chunk) => { /* Buffer of Int16 PCM samples */ });
setTimeout(() => mic.stop(), 5000);
```

### Play audio

```javascript
const { Speaker } = require('decibri');

const speaker = new Speaker({ sampleRate: 16000, channels: 1 });
speaker.write(pcmBuffer);
speaker.end();
```

### Browser capture

```javascript
import { Microphone } from 'decibri'; // browser entry via conditional export

const mic = new Microphone({ sampleRate: 16000 });
mic.on('data', (chunk) => { /* Int16Array of PCM samples */ });
await mic.start(); // requires user gesture in Safari
```

### Browser playback

```javascript
import { Speaker } from 'decibri'; // browser entry via conditional export

const speaker = new Speaker({ sampleRate: 16000 });
playButton.onclick = async () => {
  await speaker.write(int16Chunk); // Int16Array of PCM samples
  await speaker.drain();           // resolves when playback finishes
  speaker.stop();
};
```

### Pipe capture to playback (echo)

```javascript
const { Microphone, Speaker } = require('decibri');

const mic = new Microphone({ sampleRate: 16000, channels: 1 });
const speaker = new Speaker({ sampleRate: 16000, channels: 1 });
mic.pipe(speaker);
```

## API: Microphone (Capture)

### `new Microphone(options?)`

Creates a Readable stream that captures from the microphone.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | 16000 | Samples per second (1000 to 384000) |
| `channels` | number | 1 | Input channels (1 to 32) |
| `framesPerBuffer` | number | 1600 | Frames per chunk (64 to 65536). At 16kHz mono, 1600 = 100ms = 3200 bytes |
| `device` | number, string, or `{ id: string }` | system default | Device index, case-insensitive name substring, or stable per-host ID |
| `dtype` | `'int16'` \| `'float32'` | `'int16'` | Sample encoding |
| `vad` | `false` \| `'silero'` \| `'energy'` \| `VadOptions` | `false` | Voice activity detection: disabled, the Silero ML model, an RMS energy threshold, or a config object `{ model, threshold, holdoffMs }` to tune the policy |
| `modelPath` | string | bundled model | Path to the Silero model. Only used when `vad` is `'silero'` |

Standard `ReadableOptions` (e.g. `highWaterMark`) are also accepted.

`vad: true` is not accepted; pass the mode explicitly as `vad: 'silero'` or `vad: 'energy'`. The string shorthand uses the default threshold (0.5 for `'silero'`, 0.01 for `'energy'`) and a 300 ms holdoff; to tune them pass a config object: `vad: { model: 'silero', threshold: 0.6, holdoffMs: 200 }`. The `threshold` (`0`–`1`) and `holdoffMs` fields are optional and fall back to those defaults.

### Methods

| Method | Description |
| --- | --- |
| `mic.stop()` | Stop capture and end stream. Safe to call multiple times |
| `Microphone.open(options?)` | Construct without blocking the event loop. Returns a `Promise<Microphone>`. See [Non-blocking API](#non-blocking-api) |
| `Microphone.devices()` | List available input devices |
| `Microphone.version()` | Version info: `{ decibri, audioBackend, binding }` |

The module-level `inputDevices()` and `version()` free functions are equivalent to the static methods.

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `mic.isOpen` | boolean | `true` while capturing |
| `mic.vadScore` | number | Latest VAD score for the active mode (Silero probability or normalized RMS); 0 when disabled |

### Events

| Event | Payload | Description |
| --- | --- | --- |
| `'data'` | Buffer | Audio chunk (Int16 LE or Float32 LE) |
| `'backpressure'` | - | Internal buffer full, consumer too slow |
| `'speech'` | - | VAD: audio crosses threshold |
| `'silence'` | - | VAD: audio below threshold for the holdoff period |
| `'end'` | - | Stream ended |
| `'error'` | Error | An error occurred |

## API: Speaker (Playback)

### `new Speaker(options?)`

Creates a Writable stream for speaker playback.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | 16000 | Playback sample rate (1000 to 384000) |
| `channels` | number | 1 | Output channels (1 to 32) |
| `dtype` | `'int16'` \| `'float32'` | `'int16'` | Sample encoding of incoming data |
| `device` | number, string, or `{ id: string }` | system default | Output device index, case-insensitive name substring, or stable per-host ID |

Standard `WritableOptions` (e.g. `highWaterMark`) are also accepted.

| Method / Property | Description |
| --- | --- |
| `speaker.write(chunk)` | Write PCM data for playback |
| `speaker.writeAsync(chunk)` | Write without blocking the event loop. Returns a `Promise`. See [Non-blocking API](#non-blocking-api) |
| `speaker.end()` | Signal end. Drains remaining audio, then emits `'finish'` |
| `speaker.drainAsync()` | Wait for queued audio to finish without blocking the event loop. Returns a `Promise` |
| `speaker.stop()` | Immediate stop. Discards remaining audio |
| `speaker.isPlaying` | `true` while audio is being output |
| `Speaker.open(options?)` | Construct without blocking the event loop. Returns a `Promise<Speaker>` |
| `Speaker.devices()` | List available output devices |
| `Speaker.version()` | Same as `Microphone.version()` |

The module-level `outputDevices()` free function is equivalent to `Speaker.devices()`.

## Non-blocking API

The synchronous constructors and `write` / `drain` do their work on the event loop, which is fine for most apps. For event-loop-sensitive code (servers, real-time voice pipelines), 4.1.0 adds async variants that perform the blocking work without stalling the event loop. They are additive: the synchronous API is unchanged, and you opt in only where you need it.

### Non-blocking construction

`Microphone.open(options?)` and `Speaker.open(options?)` are async factories that return a Promise of a ready instance. They take the same options as the constructors. For a microphone with Silero VAD, the model load that the constructor does inline runs without blocking the event loop.

```javascript
const { Microphone, Speaker } = require('decibri');

const mic = await Microphone.open({ sampleRate: 16000, vad: 'silero' });
const speaker = await Speaker.open({ sampleRate: 16000, channels: 1 });
```

The synchronous `new Microphone(...)` and `new Speaker(...)` still work unchanged. A failed open rejects the Promise with the same typed error a failed constructor throws.

### Non-blocking playback

`speaker.writeAsync(chunk)` resolves once the audio is queued, performing the backpressure wait (when the playback buffer is full) without blocking the event loop. `speaker.drainAsync()` resolves when all queued audio has finished playing, again without blocking.

```javascript
const speaker = await Speaker.open({ sampleRate: 16000, channels: 1 });

await speaker.writeAsync(pcmBuffer);
await speaker.drainAsync(); // resolves when playback finishes
```

These are a direct alternative to the synchronous `write()` / `pipe()` / `end()` stream interface, which is unchanged. Use one path per instance (the stream methods or the async methods, not both at once), and await calls in sequence to keep samples in order.

## Errors

Construction errors come as typed classes you can catch:

```javascript
const { Microphone, DecibriError, DeviceError } = require('decibri');

try {
  new Microphone({ device: 'no such device' });
} catch (err) {
  if (err instanceof DeviceError) {
    console.log(err.code); // e.g. 'MICROPHONE_NOT_FOUND'
  }
}
```

`DeviceError`, `OrtError`, and `OrtPathError` extend `DecibriError`, which extends `Error`. Each carries a stable `code` string. Argument validation (bad `sampleRate`, `channels`, `dtype`, or `vad`) throws a built-in `RangeError` or `TypeError`.

## API: Browser

The browser API uses `getUserMedia` and `AudioWorklet`. It differs from the Node.js API because browser audio is fundamentally async.

### `new Microphone(options?)` (browser)

Same options as Node.js, plus:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | string | system default | Device ID from `Microphone.devices()` (not index) |
| `echoCancellation` | boolean | true | Browser echo cancellation |
| `noiseSuppression` | boolean | true | Browser noise suppression |
| `workletUrl` | string | inline blob | Custom worklet URL for strict CSP |

The browser runs energy-mode VAD only, so its `vad` option accepts `false` or `'energy'`. The browser `version()` returns `{ decibri }` only.

### Key differences from Node.js

| Aspect | Node.js | Browser |
| --- | --- | --- |
| Start | Automatic on first read | `await mic.start()` |
| Base class | Readable stream | Custom Emitter |
| Data type | Buffer | Int16Array / Float32Array |
| `devices()` | Sync, returns array | Async, returns Promise |
| Sample rate | Native device rate | Resampled from native rate |
| VAD | `'silero'` or `'energy'` | `'energy'` only |

### `new Speaker(options?)` (browser)

Browser audio playback through the Web Audio API. Playback is async (Promise based) and must be started from a user gesture so the browser allows audio.

```javascript
import { Speaker } from 'decibri'; // browser entry via conditional export

const speaker = new Speaker({ sampleRate: 16000 });

playButton.onclick = async () => {
  await speaker.write(int16Chunk); // Int16Array of PCM samples
  await speaker.drain();           // resolves when playback finishes
  speaker.stop();
};
```

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | number | 16000 | Sample rate of the audio you write (resampled to the output rate) |
| `channels` | number | 1 | Output channels (a mono stream plays on every channel) |
| `dtype` | `'int16'` \| `'float32'` | `'int16'` | Encoding of the samples you write |
| `workletUrl` | string | inline blob | Custom worklet URL for strict CSP |

- `start()` creates and resumes the audio output. Optional: `write()` starts it on the first call. Either must run in a user gesture (a click or tap); a context blocked by the autoplay policy surfaces a clear error.
- `write(chunk)` resolves when the samples are queued. It waits when the buffer is full, so awaiting it paces playback. Await calls sequentially to preserve order.
- `drain()` resolves when the queued audio has finished playing, immediately if nothing is queued.
- `stop()` halts immediately and discards anything queued.
- `isPlaying` reports whether audio is currently queued and playing.

To verify playback in real browsers, open `examples/browser-speaker-test.html` (see `examples/README.md`).

## Voice Activity Detection

### Energy mode

Lightweight RMS energy threshold. No model required.

```javascript
const mic = new Microphone({ vad: { model: 'energy', threshold: 0.01 } });
mic.on('speech', () => console.log('speaking'));
mic.on('silence', () => console.log('silent'));
```

### Silero mode

ML-based detection using the Silero VAD v5 model. More accurate than energy mode, especially in noisy environments.

```javascript
const mic = new Microphone({ vad: { model: 'silero', threshold: 0.5 } });
mic.on('speech', () => console.log('speaking'));
mic.on('silence', () => console.log('silent'));
```

The Silero model (~2MB) ships inside the npm package. No downloads or API keys required. Silero mode is Node.js only.

## Device Selection

```javascript
const { Microphone } = require('decibri');

// System default
const mic = new Microphone();

// By name (case-insensitive substring match)
const mic = new Microphone({ device: 'USB' });

// By index
const devices = Microphone.devices();
const mic = new Microphone({ device: devices[1].index });

// By stable per-host ID (survives across enumerations)
const mic = new Microphone({ device: { id: devices[1].id } });

Microphone.devices();
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

For the browser, `examples/browser-speaker-test.html` is a page for manually verifying audio playback in each browser. See `examples/README.md` for how to serve it on desktop and mobile.

## Migrating from 3.x

decibri 4.0.0 renames the API to a microphone and speaker vocabulary and switches to named exports. See [MIGRATION.md](./MIGRATION.md) for a complete before-and-after guide.

## Platform Support

| Platform | Architecture | Audio Backend |
| --- | --- | --- |
| Windows | x64 | WASAPI |
| macOS | arm64 | CoreAudio |
| Linux | x64 | ALSA |
| Linux | arm64 | ALSA |
| Browser | - | Web Audio API (AudioWorklet) |

## How It Works

decibri compiles a Rust audio core to a Node.js native addon and ships prebuilt binaries for each platform, so there is no build step on install. Browser support uses a JavaScript AudioWorklet implementation with the same event-driven API.

On Node.js, audio flows from the OS audio device through frame-exact buffering (which guarantees consistent chunk sizes) and into a standard Readable stream. In the browser, audio is captured and resampled in an AudioWorklet and delivered through the same `'data'` event interface.

## Documentation

- **Source code & issues**: [github.com/decibri/decibri](https://github.com/decibri/decibri)
- **Provider integrations** (OpenAI, Deepgram, AssemblyAI): [decibri.com/docs](https://decibri.com/docs)

## License

Apache-2.0. See [LICENSE](https://github.com/decibri/decibri/blob/main/LICENSE) for details.

Copyright (c) 2026 Decibri.
