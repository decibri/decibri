import { Readable, ReadableOptions, Writable, WritableOptions } from 'stream';

/** Information about an available audio input device. */
export interface DeviceInfo {
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

/** Version strings returned by `Decibri.version()`. */
export interface VersionInfo {
  /** decibri package version (e.g. `"3.0.0"`). */
  decibri: string;
  /** Audio runtime version string (e.g. `"cpal 0.17"`). */
  portaudio: string;
}

/** Constructor options for `Decibri`. */
export interface DecibriOptions extends ReadableOptions {
  /**
   * Sample rate in Hz.
   * @default 16000
   * @range 1000–384000
   */
  sampleRate?: number;

  /**
   * Number of input channels.
   * @default 1
   * @range 1–32
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
   * - numeric index (from `DeviceInfo.index`)
   * - case-insensitive name substring
   * - `{ id: string }` for stable per-host device ID from `DeviceInfo.id`
   *
   * Omit to use the system default input device.
   */
  device?: number | string | { id: string };

  /**
   * Sample encoding format.
   * - `'int16'`: 16-bit signed integer, little-endian (2 bytes per sample)
   * - `'float32'`: 32-bit IEEE 754 float, little-endian (4 bytes per sample)
   * @default 'int16'
   */
  format?: 'int16' | 'float32';

  /**
   * Enable energy-based voice activity detection.
   * When enabled, emits `'speech'` and `'silence'` events.
   * @default false
   */
  vad?: boolean;

  /**
   * RMS energy threshold for speech detection (VAD mode only).
   * @default 0.01
   * @range 0–1
   */
  vadThreshold?: number;

  /**
   * Milliseconds of sub-threshold audio before emitting `'silence'` (VAD mode only).
   * @default 300
   */
  vadHoldoff?: number;

  /**
   * VAD engine to use.
   * - `'energy'`: RMS energy threshold (default, lightweight)
   * - `'silero'`: Silero VAD v5 ML model (more accurate, ~1ms inference)
   * @default 'energy'
   */
  vadMode?: 'energy' | 'silero';

  /**
   * Path to the Silero VAD ONNX model file.
   * Only used when `vadMode` is `'silero'`.
   * Defaults to `models/silero_vad.onnx` relative to the package.
   */
  modelPath?: string;
}

/**
 * Cross-platform microphone audio capture as a Node.js Readable stream.
 *
 * @example
 * ```js
 * const Decibri = require('decibri');
 * const mic = new Decibri({ sampleRate: 16000, channels: 1 });
 * mic.on('data', (chunk) => {
 *   // chunk is a Buffer of Int16 LE PCM samples
 * });
 * setTimeout(() => mic.stop(), 5000);
 * ```
 */
declare class Decibri extends Readable {
  constructor(options?: DecibriOptions);

  /** Stop microphone capture and end the stream. Safe to call multiple times. */
  stop(): void;

  /** Whether the microphone is currently capturing audio. */
  readonly isOpen: boolean;

  /** List all available audio input devices. */
  static devices(): DeviceInfo[];

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
export interface OutputDeviceInfo {
  /** Device index (pass to constructor as `device`). */
  index: number;
  /** Human-readable device name from the OS. */
  name: string;
  /**
   * Stable per-host device ID. See `DeviceInfo.id` for format and fallback
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

/** Constructor options for `DecibriOutput`. */
export interface DecibriOutputOptions extends WritableOptions {
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
   * Sample encoding format of incoming data.
   * - `'int16'`: 16-bit signed integer, little-endian (2 bytes per sample)
   * - `'float32'`: 32-bit IEEE 754 float, little-endian (4 bytes per sample)
   * @default 'int16'
   */
  format?: 'int16' | 'float32';

  /**
   * Audio output device. One of:
   * - numeric index (from `OutputDeviceInfo.index`)
   * - case-insensitive name substring
   * - `{ id: string }` for stable per-host device ID from `OutputDeviceInfo.id`
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
 * const { DecibriOutput } = require('decibri');
 * const speaker = new DecibriOutput({ sampleRate: 16000, channels: 1 });
 * speaker.write(pcmBuffer);
 * speaker.end();
 * ```
 */
declare class DecibriOutput extends Writable {
  constructor(options?: DecibriOutputOptions);

  /** Immediate stop. Discards remaining buffered audio. */
  stop(): void;

  /** Whether audio is currently being output. */
  readonly isPlaying: boolean;

  /** List all available audio output devices. */
  static devices(): OutputDeviceInfo[];

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

export = Decibri;

declare namespace Decibri {
  export { DecibriOutput };
}
