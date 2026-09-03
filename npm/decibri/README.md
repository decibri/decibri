# decibri

Cross-platform audio capture, conditioning, and playback for Node.js, with browser capture and voice activity detection on live and recorded audio.

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

### Condition and analyze a file

```javascript
const { File } = require('decibri');

// The same conditioning chain as the live microphone, over a WAV file.
const file = await File.open('clip.wav', { denoise: 'fastenhancer-t', highpass: 80 });
file.on('data', (chunk) => { /* Buffer of conditioned Int16 PCM */ });
file.on('end', () => console.log('done'));

// Whole-file speech analysis (a live stream cannot do this).
const f = await File.open('clip.wav', { vad: 'silero' });
const report = await f.analyze();
for (const s of report.segments) console.log(s.start, s.end); // seconds
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
| `channels` | number | 1 | Delivered channels per frame, interleaved. `1` (the default) delivers the average of every device channel; the device's own count delivers every device channel in device order; a `channelMap` names any other selection. The device's own report is the only ceiling; no fixed maximum exists |
| `channelMap` | number[] | none | 0-based device channel indices choosing which device channels feed the delivered channels: delivered channel `j` carries device channel `channelMap[j]`. The length must equal `channels`; entries may repeat and may appear in any order, so a map selects, permutes, and duplicates |
| `framesPerBuffer` | number | 1600 | Frames per chunk (64 to 65536). At 16kHz mono, 1600 = 100ms = 3200 bytes |
| `device` | number, string, or `{ id: string }` | system default | Device index, case-insensitive name substring, or stable per-host ID |
| `dtype` | `'int16'` \| `'float32'` | `'int16'` | Sample encoding |
| `vad` | `false` \| `'silero'` \| `'energy'` \| `VadOptions` | `false` | Voice activity detection: disabled, the Silero ML model, an RMS energy threshold, or a config object `{ model, threshold, holdoffMs, source }` to tune the policy and the detector source |
| `modelPath` | string | bundled model | Path to the Silero model. Only used when `vad` is `'silero'` |
| `dcRemoval` | boolean | off | Remove a constant (DC) offset with a one-pole DC-blocking high-pass. Runs first in the chain; same-length, no added latency |
| `denoise` | `'fastenhancer-t'` | off | Single-channel speech-enhancement (denoise) model. The model ships in the package; no path or download needed |
| `highpass` | `80` \| `100` | off | High-pass cutoff in Hz (second-order Butterworth) that removes low-frequency rumble. Runs after denoise. Out-of-set values throw a `RangeError` |
| `agc` | number | off | AGC target level in dBFS, an integer in -40 to -3 (typical -18). Runs after the high-pass. Out-of-range throws a `RangeError` |
| `limiter` | number | off | Peak limiter ceiling in dBFS, a number in -3.0 to 0.0 (typical -1.0). Runs last. Out-of-range throws a `RangeError` |
| `aec` | `'tau'` \| `AecOptions` | off | Acoustic echo cancellation of the far-end audio pushed through `pushAecReference`. `'tau'` names the model; an object `{ model, tailMs, suppression, referenceSampleRate, referenceChannels }` tunes it. Runs before voice activity detection, so `vadScore` and the `'speech'` / `'silence'` events read the echo-removed signal. Requires `sampleRate` in 8000 to 48000. See [Echo cancellation](#echo-cancellation) |

Standard `ReadableOptions` (e.g. `highWaterMark`) are also accepted.

`vad: true` is not accepted; pass the mode explicitly as `vad: 'silero'` or `vad: 'energy'`. The string shorthand uses the default threshold (0.5 for `'silero'`, 0.01 for `'energy'`) and a 300 ms holdoff; to tune them pass a config object: `vad: { model: 'silero', threshold: 0.6, holdoffMs: 200 }`. The `threshold` (`0`–`1`) and `holdoffMs` fields are optional and fall back to those defaults. The optional `source` field names the 0-based DELIVERED channel the detector reads: the position within the delivered interleaved frames after any `channelMap` is applied (a `channelMap` names device channels; `source` names a delivered position). Absent, the detector reads the frame average of every delivered channel. Selecting a source changes which samples the detector reads and nothing else; the delivered audio is untouched.

Multichannel capture delivers interleaved frames: each chunk holds `framesPerBuffer * channels` samples, and each frame carries the delivered channels in order. Without a `channelMap`, a `channels` count above `1` must equal the device's own: a count above the device's report fails with a `DecibriError` carrying the code `'MICROPHONE_CHANNELS_UNSUPPORTED'`, and a count above `1` and below the device's report fails with `'CHANNEL_SELECTION_AMBIGUOUS'`, because which channels it means has no single answer; a `channelMap` names them. A map entry the device does not have fails with `'CHANNEL_MAP_OUT_OF_RANGE'`, naming the entry and the count the device reports.

### Methods

| Method | Description |
| --- | --- |
| `mic.stop()` | Stop capture and end stream. Safe to call multiple times |
| `mic.pushAecReference(data)` | Queue far-end reference audio for the echo canceller, pushed as it is played, in played order. Accepts the same input shapes `Speaker.write` accepts. Never blocks and never throws on a full queue; a no-op when `aec` is unset. See [Echo cancellation](#echo-cancellation) |
| `mic.aecMetrics()` | The echo canceller's metrics, or `null` when `aec` is unset or capture is not running |
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
| `channels` | number | 1 | Output channels (1 or more, up to what the device supports) |
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

Takes `sampleRate`, `channels`, `channelMap`, `framesPerBuffer`, `dtype` and `vad` as Node.js does, plus:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `device` | string | system default | Device ID from `Microphone.devices()` (not index) |
| `echoCancellation` | boolean | true | Browser echo cancellation, applied by the platform before decibri sees the audio |
| `noiseSuppression` | boolean | true | Browser noise suppression, applied by the platform before decibri sees the audio |
| `workletUrl` | string | inline blob | Custom worklet URL for strict CSP |

The conditioning options run in the native Node.js capture path: `dcRemoval`, `denoise`, `highpass`, `agc`, `limiter`, `aec` and `modelPath`. The browser `Microphone` throws a `TypeError` naming the option if one is passed.

The browser runs energy-mode VAD only, so its `vad` option accepts `false`, `'energy'`, or a config object `{ model: 'energy', threshold, holdoffMs, source }`. The browser `version()` returns `{ decibri }` only: the browser build has no native core, so `decibri` reports the installed package version. In Node, `version().decibri` reports the native core version.

### Key differences from Node.js

| Aspect | Node.js | Browser |
| --- | --- | --- |
| Start | Automatic on first read | `await mic.start()` |
| Base class | Readable stream | Custom Emitter |
| Data type | Buffer | Int16Array / Float32Array |
| `devices()` | Sync, returns array | Async, returns Promise |
| Sample rate | Native device rate | Resampled from native rate |
| VAD | `'silero'` or `'energy'` | `'energy'` only |
| Conditioning | ACE chain and `aec` | Platform `echoCancellation` and `noiseSuppression` |
| Offline audio | `File` and `AudioWriter` | Capture and playback only |

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

ML-based detection using the Silero VAD v6.2 model. More accurate than energy mode, especially in noisy environments.

```javascript
const mic = new Microphone({ vad: { model: 'silero', threshold: 0.5 } });
mic.on('speech', () => console.log('speaking'));
mic.on('silence', () => console.log('silent'));
```

The Silero model (~2MB) ships inside the npm package. No downloads or API keys required. Silero mode is Node.js only.

## Decibri ACE (audio conditioning)

Decibri ACE (Audio Conditioning Engine) is decibri's opt-in audio front-end for speech. It is a conditioning chain that runs on the captured audio before it reaches your `'data'` handler. Every stage is off by default, runs on-device, and needs no API key. With nothing enabled the capture path is byte-identical to plain capture.

The stages run in a fixed order. Enable any subset:

| Stage | Option | Range |
| --- | --- | --- |
| DC removal | `dcRemoval: true` | boolean |
| Denoise | `denoise: 'fastenhancer-t'` | the one bundled model |
| High-pass | `highpass: 80` or `100` | Hz |
| AGC | `agc: -18` | dBFS, -40 to -3 |
| Limiter | `limiter: -1.0` | dBFS, -3.0 to 0.0 |

```javascript
const { Microphone } = require('decibri');

const mic = new Microphone({
  sampleRate: 16000,
  denoise: 'fastenhancer-t',   // bundled speech-enhancement model
  highpass: 80,                // remove low-frequency rumble
  agc: -18,                    // target level in dBFS
  limiter: -1.0,               // peak ceiling in dBFS
  vad: { model: 'silero', threshold: 0.5 },
});

mic.on('data', (chunk) => { /* Buffer of conditioned Int16 PCM */ });
mic.on('speech', () => console.log('speech'));
mic.on('silence', () => console.log('silence'));
setTimeout(() => mic.stop(), 5000);
```

VAD reads the signal before the chain, so `vadScore` and the `'speech'` / `'silence'` events are unaffected by which conditioning stages you enable. The conditioning chain runs in the native Node.js capture path; the browser build does not include it.

## Echo cancellation

Acoustic echo cancellation removes the echo of far-end audio (what your application is playing out) from the captured audio. It is opt-in: set `aec` on the `Microphone` and push the far-end audio through `pushAecReference` as it is played, in played order. With `aec` unset the capture path is unchanged, and with no reference pushed the captured audio passes through unchanged. The canceller runs before voice activity detection, so `vadScore` and the `'speech'` / `'silence'` events read the echo-removed signal and playback stops triggering detection. It requires `sampleRate` in 8000 to 48000.

`aec: 'tau'` selects the model with its defaults; an `AecOptions` object tunes it:

| Option | Default | Range |
| --- | --- | --- |
| `model` | required | `'tau'`, the one model |
| `tailMs` | 200 | 16 to 500 ms |
| `suppression` | `'conservative'` | `'conservative'` or `'off'` |
| `referenceSampleRate` | the capture `sampleRate` | 1000 to 384000 Hz |
| `referenceChannels` | 1 | 1 or more |

```javascript
const { Microphone, Speaker } = require('decibri');

const mic = new Microphone({ sampleRate: 16000, aec: 'tau' });
const speaker = new Speaker({ sampleRate: 16000, channels: 1 });

// Push the far-end audio as it is played, in played order: the same
// input shapes Speaker.write accepts, in the microphone's dtype.
function playFarEnd(pcmBuffer) {
  speaker.write(pcmBuffer);
  mic.pushAecReference(pcmBuffer);
}

mic.on('data', (chunk) => { /* Buffer of echo-cancelled Int16 PCM */ });
```

`pushAecReference` never blocks and never throws on a full queue: samples that do not fit are discarded and counted by `aecMetrics().referenceDropped`. Silence between played audio need not be pushed. The canceller reads one mono reference: a reference at a `referenceSampleRate` other than the capture rate is converted before the canceller sees it, and with `referenceChannels` above 1 each frame is averaged to one mono sample. With `channels` above 1, one canceller runs per delivered channel, each fed the same pushed reference, so one push serves every channel.

`aecMetrics()` returns the canceller's report merged with the reference queue's counters (`delaySamples`, `erleDb`, `doubleTalk`, `referenceStarved`, `acquisitionParked`, `referenceReanchors`, `referenceDropped`, `referenceSilence`), or `null` when `aec` is unset or capture is not running. Its `channels` array carries each delivered channel's report in delivered order; the top-level fields report the first delivered channel's. `erleDb` is not a quality ranking across channels: it rises with echo distance, so a far microphone reports a higher figure than a near one while removing less echo in absolute terms.

Echo cancellation runs in the native Node.js capture path; the browser `Microphone` does not take the option, because browser capture already carries the platform's own echo cancellation through its `echoCancellation` constraint, on by default.

## ONNX Runtime telemetry

Silero mode (`vad: 'silero'`) and the ACE `denoise` stage run on ONNX Runtime, which carries its own telemetry, separate from anything decibri does. Decibri disables it on the environment it commits when it initializes the runtime. Set `DECIBRI_ORT_TELEMETRY=1` in the environment before first use to leave it enabled; every other value, an empty value, and an absent variable leave it disabled.

Two limits apply on Windows and decibri can close neither, so decibri does not claim that no telemetry is emitted. ONNX Runtime logs one process-information event while the environment is being created, before the setting is applied, and logs it once per process, so that event is emitted whichever way the setting is left. The runtime also assigns its telemetry state from the Windows tracing session through an ETW callback, so the platform can re-enable telemetry after decibri has disabled it. On other platforms ONNX Runtime's telemetry provider does nothing.

Neither ONNX Runtime nor this setting applies to the browser build, which has no ONNX Runtime.

## API: File (offline source)

`File` runs the same conditioning chain on audio you already have. Because a `File` is a complete recording, it can also analyze the whole recording for speech.

- `await File.open(path, options?)`: read a file off the event loop (recommended, like `Microphone.open`). Reads WAV, AIFF, AIFF-C and FLAC, identified from the file's own bytes rather than its extension.
- `new File(path, options?)`: the same result, synchronous (blocks on disk I/O; fine for scripts).
- `File.buffer(samples, options)`: wrap a `Float32Array` of samples you already hold. `options.inputRate` is required (raw samples carry no header); a raw `Buffer` of bytes is rejected as ambiguous. `options.inputChannels` states the interleave of the samples, `1` by default, the channel counterpart of `inputRate`.
- `await file.analyze()` (also spelled `analyse()`): consume the source and resolve to a `VadReport` of per-window `scores` (`{ start, end, vadScore, isSpeech }`) and merged speech `segments` (`{ start, end }`), in seconds of file time. Requires `vad: 'silero'`; a `File` opened without `vad` rejects with `analysis requires VAD`.
- `await file.save(path, options?)`: write the conditioned recording to disk, off the event loop, as 16-bit PCM at `sampleRate` and at the delivered channel count, in WAV, AIFF or FLAC. The container comes from the path's extension (`.wav`, `.aiff`, `.aif`, `.aifc` or `.flac`) or from `options.format`; an extension it does not recognise rejects rather than defaulting. `options.compression` sets the FLAC compression level, 0 to 8 with 5 the default. Resolves to a `SaveReport` (`{ clippedSamples, nonFiniteSamples }`). Consumes the source, and rejects with `FILE_ENGAGED` once the stream is engaged, exactly as `analyze()` does.
- `file.vadScore`, `'speech'` / `'silence'` events: per-chunk VAD alongside the stream, with the holdoff measured in FILE time (sample positions), never wall-clock time, so processing speed does not change the reported events.

Options mirror `Microphone` (`sampleRate`, `channels`, `channelMap`, `dtype`, `vad`, `dcRemoval`, `denoise`, `highpass`, `agc`, `limiter`), with `channels` and `channelMap` read against the source's own channel count (the file's header, or `inputChannels` for `File.buffer`) where the live path reads the device's report, and the vad `source` naming a delivered channel exactly as on `Microphone`; the live-capture options (`device`, `framesPerBuffer`, `aec`) do not apply. A save clamps finite samples outside full scale to `[-1.0, 1.0]` and counts them in `clippedSamples`, and replaces non-finite samples before they reach the file (NaN as silence, an infinity as full scale), counted in `nonFiniteSamples`. Iteration, analysis and saving are separate single passes: construct one `File` per operation. Note: Node also has a global `File` (the web File API); import decibri's explicitly to avoid shadowing surprises.

## API: AudioWriter (file sink)

`AudioWriter` is a Writable file sink for PCM audio: the stream to pair with decibri's Readable sources (`File`, `Microphone`) and with any other stream of PCM bytes. It collects the whole stream, then writes it as one audio file when the stream finishes, exactly as `File.save` writes: the same containers from the same extension rule, the same 16-bit PCM encoding, the same clamp and non-finite handling, the same bytes.

```javascript
const { pipeline } = require('node:stream/promises');
const { File, AudioWriter } = require('decibri');

await pipeline(
  new File('noisy.wav', { denoise: 'fastenhancer-t' }),
  new AudioWriter('clean.flac', { sampleRate: 16000 }),
);
```

`sampleRate` is required (raw audio carries no header to read a rate from). `channels` (default 1) and `dtype` (default `'int16'`) describe the incoming bytes: the stream's total sample count must be a whole number of frames at `channels`, and each container's own channel ceiling applies at the write. The `File.save` options (`format`, `compression`) are accepted as well. `'finish'` fires after the file is on disk, and `writer.report` then carries the `SaveReport`; a failure destroys the stream with the error.

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
| Windows | arm64 | WASAPI |
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
