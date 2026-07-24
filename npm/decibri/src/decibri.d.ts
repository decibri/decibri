import { Readable, ReadableOptions, Writable, WritableOptions } from 'stream';

/** Information about an available audio input device. */
export interface MicrophoneInfo {
  /** Device index (pass to constructor as `device`). */
  index: number;
  /** Human-readable device name from the OS. */
  name: string;
  /**
   * Stable per-host device ID. Pass via `device: { id: ... }` for selection
   * that survives across enumerations.
   * - Windows (WASAPI): endpoint ID
   * - macOS (CoreAudio): device UID
   * - Linux (ALSA): PCM identifier
   * Empty string if cpal cannot produce a stable ID for this device.
   */
  id: string;
  /** Maximum number of input channels the device supports. */
  maxInputChannels: number;
  /** Device's preferred sample rate in Hz. */
  defaultSampleRate: number;
  /** Whether this is the current system default input device. */
  isDefault: boolean;
}

/** Version strings returned by `Microphone.version()`. */
export interface VersionInfo {
  /** decibri core version. */
  decibri: string;
  /** Audio backend version string (e.g. `"cpal 0.17"`). */
  audioBackend: string;
  /** This binding's npm package version. */
  binding: string;
}

/**
 * Voice-activity-detection config object, passed on the `vad` option to tune
 * the detector's threshold and holdoff. The bare `vad: 'silero'` / `vad:
 * 'energy'` shorthand selects a mode with its default policy; pass this object
 * to override the threshold or holdoff.
 */
export interface VadOptions {
  /**
   * Which detector to run.
   * - `'silero'`: Silero VAD v5 ML model (more accurate, ~1ms inference)
   * - `'energy'`: RMS energy threshold (lightweight, no model)
   */
  model: 'silero' | 'energy';

  /**
   * Speech-detection threshold for the active mode.
   * @default 0.5 for `'silero'`, 0.01 for `'energy'`
   * @range 0–1
   */
  threshold?: number;

  /**
   * Milliseconds of sub-threshold audio before emitting `'silence'`.
   * @default 300
   */
  holdoffMs?: number;
}

/** Constructor options for `Microphone`. */
export interface MicrophoneOptions extends ReadableOptions {
  /**
   * Sample rate in Hz.
   * @default 16000
   * @range 1000–384000
   */
  sampleRate?: number;

  /**
   * Number of input channels. Mono only: the only accepted value is `1`, and a
   * value greater than `1` throws a `RangeError` (multichannel capture is not
   * supported) rather than being silently downmixed. The option is kept for
   * forward compatibility: a future release may accept a value greater than `1`
   * by delivering true interleaved multichannel.
   * @default 1
   */
  channels?: number;

  /**
   * Frames per audio callback buffer. Controls chunk size and delivery interval.
   * At 16 kHz mono, 1600 frames = 100 ms chunks of 3200 bytes (int16).
   * @default 1600
   * @range 64–65536
   */
  framesPerBuffer?: number;

  /**
   * Audio input device. One of:
   * - numeric index (from `MicrophoneInfo.index`)
   * - case-insensitive name substring
   * - `{ id: string }` for stable per-host device ID from `MicrophoneInfo.id`
   *
   * Omit to use the system default input device.
   */
  device?: number | string | { id: string };

  /**
   * Sample encoding data type.
   * - `'int16'`: 16-bit signed integer, little-endian (2 bytes per sample)
   * - `'float32'`: 32-bit IEEE 754 float, little-endian (4 bytes per sample)
   * @default 'int16'
   */
  dtype?: 'int16' | 'float32';

  /**
   * Voice activity detection. One of:
   * - `false`: disabled (default)
   * - `'silero'`: Silero VAD v5 ML model (more accurate, ~1ms inference)
   * - `'energy'`: RMS energy threshold (lightweight)
   * - a `VadOptions` config object `{ model, threshold?, holdoffMs? }` to tune
   *   the threshold and holdoff for the chosen model
   *
   * The string shorthand uses the mode's default threshold (0.5 for `'silero'`,
   * 0.01 for `'energy'`) and a 300 ms holdoff; pass a `VadOptions` object to
   * override them. When enabled, emits `'speech'` and `'silence'` events and
   * updates `vadScore`. The legacy `vad: true` form is rejected; specify the
   * mode explicitly.
   * @default false
   */
  vad?: false | 'silero' | 'energy' | VadOptions;

  /**
   * Path to the Silero VAD ONNX model file.
   * Only used when `vad` is `'silero'`.
   * Defaults to `models/silero_vad.onnx` relative to the package.
   */
  modelPath?: string;

  /**
   * Remove a constant (DC) offset from the captured audio with a one-pole
   * DC-blocking high-pass. Set `true` to enable it; omit or set `false` to
   * leave it off (the default), which keeps the capture path byte-identical.
   * Runs first in the chain, before denoise, and is same-length with no added
   * latency, so `vadScore` and the `speech` / `silence` events are unaffected.
   * Pure DSP: no bundled file or download is needed.
   * @default undefined
   */
  dcRemoval?: boolean;

  /**
   * Single-channel speech enhancement (denoise) model applied to the captured
   * audio. The only accepted value is `'fastenhancer-t'`; omit to leave denoise
   * off (the default), which keeps the capture path unchanged. The bundled
   * model ships with the package; no path is required.
   *
   * When set, the captured audio is denoised before delivery and the `'data'`
   * chunks carry the enhanced signal. VAD reads the pre-enhancement signal, so
   * `vadScore` and the `speech` / `silence` events are unaffected.
   * @default undefined
   */
  denoise?: 'fastenhancer-t';

  /**
   * High-pass filter cutoff in Hz applied to the captured audio, removing
   * low-frequency rumble below the voice band. The accepted values are `80` (an
   * 80 Hz second-order Butterworth high-pass) and `100` (a 100 Hz one); omit to
   * leave the high-pass off (the default), which keeps the capture path
   * full-range. Runs after denoise in the chain. The closed value set is
   * designed to grow (further cutoffs are additive) the way `denoise` grows.
   * Out-of-set values raise a `RangeError`.
   * @default undefined
   */
  highpass?: 80 | 100;

  /**
   * Automatic gain control target level in dBFS applied to the captured audio.
   * Drives the running level toward this target with a smoothed, rate-limited
   * gain. An integer in the range -40 to -3 (typical -18); omit to leave AGC
   * off (the default), which keeps the level untouched. Runs after the
   * high-pass step. Out-of-range values raise a `RangeError`.
   * @default undefined
   */
  agc?: number;

  /**
   * Peak limiter ceiling in dBFS (sample-peak) applied to the captured audio.
   * Holds the signal at or below this ceiling, the safety net that catches a
   * transient the AGC's gain would let exceed full scale. A number in the range
   * -3.0 to 0.0 (typical -1.0); omit to leave the limiter off (the default),
   * which keeps the level untouched. Runs last in the chain, after the AGC step.
   * Out-of-range values raise a `RangeError`.
   * @default undefined
   */
  limiter?: number;
}

/**
 * Cross-platform microphone audio capture as a Node.js Readable stream.
 *
 * @example
 * ```js
 * const { Microphone } = require('decibri');
 * const mic = new Microphone({ sampleRate: 16000, channels: 1 });
 * mic.on('data', (chunk) => {
 *   // chunk is a Buffer of Int16 LE PCM samples
 * });
 * setTimeout(() => mic.stop(), 5000);
 * ```
 */
export declare class Microphone extends Readable {
  constructor(options?: MicrophoneOptions);

  /**
   * Construct a Microphone without blocking the event loop on the open work.
   *
   * The synchronous constructor loads the Silero VAD model inline when
   * `vad: 'silero'` is set, blocking the event loop for roughly 100 to 500 ms
   * on a cold cache. This factory runs that load (and device resolution) on the
   * native thread pool and resolves to a ready instance. The synchronous
   * constructor remains available and unchanged.
   *
   * Options are identical to the constructor. A failed open rejects the Promise
   * with the matching error: `RangeError` / `TypeError` for invalid options, or
   * a `DeviceError` / `OrtError` / `OrtPathError` for native failures.
   *
   * @example
   * ```js
   * const mic = await Microphone.open({ vad: 'silero' });
   * mic.on('data', (chunk) => { ... });
   * ```
   */
  static open(options?: MicrophoneOptions): Promise<Microphone>;

  /** Stop microphone capture and end the stream. Safe to call multiple times. */
  stop(): void;

  /** Whether the microphone is currently capturing audio. */
  readonly isOpen: boolean;

  /**
   * Most recent VAD score for the active mode: the Silero speech probability in
   * `'silero'` mode, the normalized RMS of the last chunk in `'energy'` mode.
   * 0 when VAD is disabled or before the first chunk is processed.
   */
  readonly vadScore: number;

  /**
   * Number of capture buffers dropped because the consumer could not keep pace.
   * 0 while the consumer keeps up, or before capture starts. A rising value
   * means audio is being dropped to bound memory.
   */
  readonly overrunCount: number;

  /** List all available audio input devices. */
  static devices(): MicrophoneInfo[];

  /** Version information for decibri and the audio runtime. */
  static version(): VersionInfo;

  // ── Event overloads ──────────────────────────────────────────────────────

  on(event: 'data', listener: (chunk: Buffer) => void): this;
  on(event: 'error', listener: (err: Error) => void): this;
  on(event: 'end', listener: () => void): this;
  on(event: 'close', listener: () => void): this;
  on(event: 'pause', listener: () => void): this;
  on(event: 'resume', listener: () => void): this;
  on(event: 'readable', listener: () => void): this;
  on(event: 'backpressure', listener: () => void): this;
  on(event: 'speech', listener: () => void): this;
  on(event: 'silence', listener: () => void): this;
  on(event: string | symbol, listener: (...args: any[]) => void): this;

  once(event: 'data', listener: (chunk: Buffer) => void): this;
  once(event: 'error', listener: (err: Error) => void): this;
  once(event: 'end', listener: () => void): this;
  once(event: 'close', listener: () => void): this;
  once(event: 'backpressure', listener: () => void): this;
  once(event: 'speech', listener: () => void): this;
  once(event: 'silence', listener: () => void): this;
  once(event: string | symbol, listener: (...args: any[]) => void): this;
}

/** Information about an available audio output device. */
/** Constructor options for `File`. */
export interface FileOptions extends ReadableOptions {
  /**
   * Target output rate in Hz: the rate every delivered chunk carries. The
   * source's input rate (from the WAV header, or `inputRate` for
   * `File.buffer`) is resampled to this rate, so a 44.1 kHz recording comes
   * out at 16 kHz unless you set `sampleRate`. The same meaning the option
   * has on `Microphone`.
   * @default 16000
   * @range 1000–384000
   */
  sampleRate?: number;

  /**
   * Sample encoding data type of the delivered chunks.
   * - `'int16'`: 16-bit signed integer, little-endian (2 bytes per sample)
   * - `'float32'`: 32-bit IEEE 754 float, little-endian (4 bytes per sample)
   * @default 'int16'
   */
  dtype?: 'int16' | 'float32';

  /**
   * Voice activity detection, opt-in exactly as on `Microphone`: `false`
   * (default), `'silero'`, `'energy'`, or a `VadOptions` config object.
   * When enabled, per-chunk detection runs alongside the stream (the
   * `'speech'` / `'silence'` events and `vadScore`, with the holdoff
   * measured in FILE time rather than wall-clock time) and `'silero'`
   * additionally enables the whole-file `analyze()`. With no `vad` set the
   * File simply conditions audio: no scores, no segments, no speech events.
   * @default false
   */
  vad?: false | 'silero' | 'energy' | VadOptions;

  /**
   * Path to the Silero VAD ONNX model file.
   * Only used when `vad` is `'silero'`.
   * Defaults to `models/silero_vad.onnx` relative to the package.
   */
  modelPath?: string;

  /**
   * Remove a constant (DC) offset with a one-pole DC-blocking high-pass,
   * exactly as on `Microphone`.
   * @default undefined
   */
  dcRemoval?: boolean;

  /**
   * Single-channel speech enhancement (denoise) model, exactly as on
   * `Microphone`. The only accepted value is `'fastenhancer-t'`.
   * @default undefined
   */
  denoise?: 'fastenhancer-t';

  /**
   * High-pass filter cutoff in Hz (`80` or `100`), exactly as on
   * `Microphone`.
   * @default undefined
   */
  highpass?: 80 | 100;

  /**
   * Automatic gain control target level in dBFS, exactly as on `Microphone`.
   * @default undefined
   * @range -40 to -3
   */
  agc?: number;

  /**
   * Peak limiter ceiling in dBFS (sample-peak), exactly as on `Microphone`.
   * @default undefined
   * @range -3.0 to 0.0
   */
  limiter?: number;
}

/** Options for `File.buffer`: `FileOptions` plus the samples' native rate. */
export interface FileBufferOptions extends FileOptions {
  /**
   * The native rate of the in-memory samples in Hz. Required: raw samples
   * carry no header to read a rate from. The samples are resampled from this
   * rate to `sampleRate`.
   * @range 1000–384000
   */
  inputRate: number;
}

/**
 * One scored voice-activity window of a recording, produced by
 * `File.analyze()`. Windows tile the recording from the start in fixed steps
 * (512 samples at 16 kHz, 32 ms per window); a trailing remainder shorter
 * than one window is not scored, exactly as live detection leaves a
 * sub-window remainder unscored.
 */
export interface VadWindow {
  /** Window start, in seconds of file time. */
  start: number;
  /** Window end, in seconds of file time. */
  end: number;
  /**
   * Speech probability for this window (0 to 1). The same quantity the live
   * per-chunk `vadScore` reports, here per window across the whole recording.
   */
  vadScore: number;
  /**
   * Whether `vadScore` meets the configured threshold. The raw per-window
   * test, not the debounced speaking state.
   */
  isSpeech: boolean;
}

/**
 * One merged speech region of a recording, produced by `File.analyze()`:
 * consecutive speech windows whose silence gaps are within the configured
 * holdoff collapse into one segment. The segment ends at the last speech
 * window, not at the holdoff expiry.
 */
export interface Segment {
  /** Region start, in seconds of file time. */
  start: number;
  /** Region end, in seconds of file time. */
  end: number;
}

/** The whole-recording voice-activity analysis `File.analyze()` resolves to. */
export interface VadReport {
  /** Per-window speech scores across the whole recording, in file order. */
  scores: VadWindow[];
  /** Merged speech regions across the whole recording, in file order. */
  segments: Segment[];
}

/**
 * Offline audio source: conditions a recording or in-memory samples through
 * the same chain as the live `Microphone`, delivered as a finite Readable
 * stream of conditioned chunks that ends at EOF (after the chain's
 * end-of-stream tail). Because a `File` is a complete recording, it can also
 * analyze the whole recording for speech with `analyze()` / `analyse()`,
 * which a live stream cannot do.
 *
 * Construction: `new File(path)` reads the WAV synchronously (fine for a
 * script; it blocks the event loop on disk I/O), `await File.open(path)` reads
 * it off the event loop (the recommended form, mirroring `Microphone.open`),
 * and `File.buffer(samples, { inputRate })` wraps a `Float32Array` of samples
 * you already hold (a raw `Buffer` of bytes is rejected as ambiguous).
 *
 * Iteration and analysis are separate single passes: each consumes the source
 * once, so construct one `File` per operation.
 *
 * Note: Node also has a global `File` (the web File API). Import decibri's
 * explicitly (`const { File } = require('decibri')`) or reference it as
 * `decibri.File` to avoid shadowing surprises.
 *
 * @example
 * const { File } = require('decibri');
 * const file = await File.open('clip.wav', { denoise: 'fastenhancer-t' });
 * file.on('data', (chunk) => { /* Buffer of conditioned Int16 PCM *\/ });
 * file.on('end', () => console.log('done'));
 *
 * @example
 * // Where is the speech?
 * const f = await File.open('clip.wav', { vad: 'silero' });
 * const report = await f.analyze();
 * for (const s of report.segments) console.log(s.start, s.end);
 */
export declare class File extends Readable {
  /**
   * Open a WAV file synchronously (blocks on disk I/O; prefer `File.open`
   * in servers). Supports 16-bit PCM and 32-bit float WAV files; the input
   * rate and channel count come from the header.
   */
  constructor(path: string, options?: FileOptions);

  /**
   * Open a WAV file without blocking the event loop: the disk read, WAV
   * parse, and chain construction run on the native thread pool. The
   * recommended form, mirroring `Microphone.open`.
   */
  static open(path: string, options?: FileOptions): Promise<File>;

  /**
   * Wrap in-memory samples as an offline source. `samples` must be a
   * `Float32Array` of mono samples in [-1.0, 1.0]; a raw `Buffer` of PCM
   * bytes is rejected as ambiguous. `inputRate` is required (raw samples
   * carry no header). Synchronous: no I/O is involved.
   */
  static buffer(samples: Float32Array, options: FileBufferOptions): File;

  /**
   * Most recent per-chunk VAD score for the active mode: the Silero speech
   * probability in `'silero'` mode, the normalized RMS of the
   * pre-conditioning signal in `'energy'` mode. 0 when VAD is disabled or
   * before the first chunk.
   */
  readonly vadScore: number;

  /**
   * Analyze the whole recording for speech, off the event loop. Resolves to
   * a `VadReport` with per-window `scores` and merged speech `segments`,
   * all in seconds of file time. Consumes the source (a `File` is a single
   * pass). Requires `vad: 'silero'`: a File opened without `vad` rejects
   * with the core's "analysis requires VAD" error, and the energy mode has
   * no whole-file analysis.
   *
   * Requires a File that is not already being streamed: once the stream has
   * been engaged this rejects with a `DecibriError` carrying the code
   * `'FILE_ENGAGED'`. Every failure detected before the pass begins leaves
   * the File usable; a failure during the pass consumes the source.
   */
  analyze(): Promise<VadReport>;

  /** The same whole-recording analysis under the international spelling. */
  analyse(): Promise<VadReport>;

  /** Release the source. Idempotent; a closed File reads as ended. */
  close(): void;

  on(event: 'data', listener: (chunk: Buffer) => void): this;
  on(event: 'end', listener: () => void): this;
  on(event: 'error', listener: (err: Error) => void): this;
  /** Speech detected (VAD enabled), at a FILE-time boundary. */
  on(event: 'speech', listener: () => void): this;
  /** Silence holdoff elapsed (VAD enabled), in FILE time. */
  on(event: 'silence', listener: () => void): this;
  on(event: string | symbol, listener: (...args: any[]) => void): this;

  once(event: 'data', listener: (chunk: Buffer) => void): this;
  once(event: 'end', listener: () => void): this;
  once(event: 'error', listener: (err: Error) => void): this;
  once(event: 'speech', listener: () => void): this;
  once(event: 'silence', listener: () => void): this;
  once(event: string | symbol, listener: (...args: any[]) => void): this;
}

export interface SpeakerInfo {
  /** Device index (pass to constructor as `device`). */
  index: number;
  /** Human-readable device name from the OS. */
  name: string;
  /**
   * Stable per-host device ID. See `MicrophoneInfo.id` for format and fallback
   * semantics; identical rules for output devices.
   */
  id: string;
  /** Maximum number of output channels the device supports. */
  maxOutputChannels: number;
  /** Device's preferred sample rate in Hz. */
  defaultSampleRate: number;
  /** Whether this is the current system default output device. */
  isDefault: boolean;
}

/** Constructor options for `Speaker`. */
export interface SpeakerOptions extends WritableOptions {
  /**
   * Sample rate in Hz.
   * @default 16000
   * @range 1000–384000
   */
  sampleRate?: number;

  /**
   * Number of output channels.
   * @default 1
   * @range 1–32
   */
  channels?: number;

  /**
   * Sample encoding data type of incoming data.
   * - `'int16'`: 16-bit signed integer, little-endian (2 bytes per sample)
   * - `'float32'`: 32-bit IEEE 754 float, little-endian (4 bytes per sample)
   * @default 'int16'
   */
  dtype?: 'int16' | 'float32';

  /**
   * Audio output device. One of:
   * - numeric index (from `SpeakerInfo.index`)
   * - case-insensitive name substring
   * - `{ id: string }` for stable per-host device ID from `SpeakerInfo.id`
   *
   * Omit to use the system default output device.
   */
  device?: number | string | { id: string };
}

/**
 * Cross-platform audio output (speaker playback) as a Node.js Writable stream.
 *
 * @example
 * ```js
 * const { Speaker } = require('decibri');
 * const speaker = new Speaker({ sampleRate: 16000, channels: 1 });
 * speaker.write(pcmBuffer);
 * speaker.end();
 * ```
 */
export declare class Speaker extends Writable {
  constructor(options?: SpeakerOptions);

  /**
   * Construct a Speaker without blocking the event loop. Symmetric with
   * `Microphone.open()`. The speaker loads no model, so the only open work is
   * device resolution; this factory is provided so async callers can use one
   * consistent construction pattern across both classes. The synchronous
   * constructor remains available and unchanged.
   *
   * A failed open (unknown device) rejects the Promise with the matching error.
   *
   * @example
   * ```js
   * const speaker = await Speaker.open({ sampleRate: 24000 });
   * speaker.write(pcmBuffer);
   * ```
   */
  static open(options?: SpeakerOptions): Promise<Speaker>;

  /**
   * Write PCM audio without blocking the event loop. Performs the backpressure
   * wait (when the native playback queue is full) on the native thread pool and
   * resolves when the samples are queued.
   *
   * Additive: the synchronous `write()` / `pipe()` stream interface is
   * unchanged. This is a direct, opt-in alternative that bypasses the Writable
   * buffer; do not interleave it with `write()` / `pipe()` on the same instance.
   * Await calls sequentially to preserve sample order. An empty buffer resolves
   * immediately; a closed or stopped stream rejects with the matching error.
   */
  writeAsync(chunk: Buffer): Promise<void>;

  /**
   * Wait for all queued audio to finish playing without blocking the event
   * loop. Runs the drain wait on the native thread pool and resolves when the
   * buffer has drained; resolves immediately if nothing was written.
   *
   * Additive: the synchronous drain via `end()` is unchanged. Pair with
   * `writeAsync()` for a fully non-blocking playback path.
   */
  drainAsync(): Promise<void>;

  /** Immediate stop. Discards remaining buffered audio. */
  stop(): void;

  /** Whether audio is currently being output. */
  readonly isPlaying: boolean;

  /** List all available audio output devices. */
  static devices(): SpeakerInfo[];

  /** Version information for decibri and the audio runtime. */
  static version(): VersionInfo;

  // ── Event overloads ──────────────────────────────────────────────────────

  on(event: 'drain', listener: () => void): this;
  on(event: 'finish', listener: () => void): this;
  on(event: 'error', listener: (err: Error) => void): this;
  on(event: 'close', listener: () => void): this;
  on(event: 'pipe', listener: (src: Readable) => void): this;
  on(event: 'unpipe', listener: (src: Readable) => void): this;
  on(event: string | symbol, listener: (...args: any[]) => void): this;
}

/** List all available audio input devices. */
export declare function inputDevices(): MicrophoneInfo[];

/** List all available audio output devices. */
export declare function outputDevices(): SpeakerInfo[];

/** Version information for decibri and the audio runtime. */
export declare function version(): VersionInfo;

/**
 * Base class for errors raised by the decibri native bindings.
 *
 * Catch this to handle any decibri device or ONNX Runtime failure generically;
 * catch a subclass for finer control. Argument validation (bad sample rate,
 * channels, frames, dtype, vad) throws built-in `RangeError` / `TypeError`, not
 * a `DecibriError`.
 */
export declare class DecibriError extends Error {
  /** Stable string code identifying the specific failure. */
  readonly code: string;
}

/**
 * Device enumeration or selection failure: an unmatched device name, an
 * ambiguous name match, or missing hardware.
 */
export declare class DeviceError extends DecibriError {}

/** ONNX Runtime setup or inference failure (Silero VAD). */
export declare class OrtError extends DecibriError {}

/** A specific ONNX Runtime library path could not be loaded. */
export declare class OrtPathError extends OrtError {}
