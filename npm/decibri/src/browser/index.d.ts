/** Information about an available audio input device (browser). */
export interface DeviceInfo {
  /** Opaque device identifier (pass to constructor as `device`). */
  deviceId: string;
  /** Human-readable device name. May be empty until permission is granted. */
  label: string;
  /** Group identifier. Devices sharing the same physical device. */
  groupId: string;
}

/** Version information. */
export interface VersionInfo {
  /** decibri package version (e.g. `"3.0.0"`). */
  decibri: string;
}

/** Constructor options for the browser `Decibri` class. */
export interface DecibriOptions {
  /**
   * Target sample rate in Hz. Browser captures at native rate and resamples.
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
   * Frames per audio chunk. Controls chunk size and delivery interval.
   * @default 1600
   * @range 64–65536
   */
  framesPerBuffer?: number;

  /**
   * Audio input device. Pass a deviceId string from `Decibri.devices()`.
   * Omit to use the system default input device.
   */
  device?: string;

  /**
   * Sample encoding format.
   * - `'int16'`: Int16Array of PCM samples
   * - `'float32'`: Float32Array of samples in [-1, 1]
   * @default 'int16'
   */
  format?: 'int16' | 'float32';

  /**
   * Enable energy-based voice activity detection.
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
   * Milliseconds of sub-threshold audio before emitting `'silence'`.
   * @default 300
   */
  vadHoldoff?: number;

  /**
   * Enable browser echo cancellation.
   * @default true
   */
  echoCancellation?: boolean;

  /**
   * Enable browser noise suppression.
   * @default true
   */
  noiseSuppression?: boolean;

  /**
   * Custom URL for the AudioWorklet processor script.
   * Use this for strict CSP environments that block blob URLs.
   * Omit to use the default inline blob URL.
   */
  workletUrl?: string;
}

/**
 * Browser microphone capture via getUserMedia + AudioWorklet.
 *
 * @example
 * ```js
 * import { Decibri } from 'decibri'; // browser entry via conditional export
 *
 * const mic = new Decibri({ sampleRate: 16000 });
 * mic.on('data', (chunk) => {
 *   // chunk is an Int16Array of PCM samples
 * });
 * await mic.start(); // call from a user gesture in Safari
 * mic.stop();
 * ```
 */
export declare class Decibri {
  constructor(options?: DecibriOptions);

  /**
   * Start microphone capture.
   * Requests permission and sets up the audio pipeline.
   * Must be called from a user gesture context in Safari.
   */
  start(): Promise<void>;

  /** Stop capture and release all resources. Safe to call multiple times. */
  stop(): void;

  /** Whether the microphone is currently capturing. */
  readonly isOpen: boolean;

  /**
   * List available audio input devices.
   * Labels may be empty until microphone permission is granted.
   */
  static devices(): Promise<DeviceInfo[]>;

  /** Version information. */
  static version(): VersionInfo;

  // ── Event methods ──────────────────────────────────────────────────────

  on(event: 'data', listener: (chunk: Int16Array | Float32Array) => void): this;
  on(event: 'error', listener: (err: Error) => void): this;
  on(event: 'end', listener: () => void): this;
  on(event: 'close', listener: () => void): this;
  on(event: 'speech', listener: () => void): this;
  on(event: 'silence', listener: () => void): this;
  on(event: string, listener: (...args: any[]) => void): this;

  off(event: string, listener: (...args: any[]) => void): this;

  once(event: 'data', listener: (chunk: Int16Array | Float32Array) => void): this;
  once(event: 'error', listener: (err: Error) => void): this;
  once(event: 'end', listener: () => void): this;
  once(event: 'close', listener: () => void): this;
  once(event: 'speech', listener: () => void): this;
  once(event: 'silence', listener: () => void): this;
  once(event: string, listener: (...args: any[]) => void): this;

  emit(event: string, ...args: any[]): boolean;
  removeAllListeners(event?: string): this;
}
