/** Information about an available audio input device (browser). */
export interface MicrophoneInfo {
  /** Opaque device identifier (pass to constructor as `device`). */
  deviceId: string;
  /** Human-readable device name. May be empty until permission is granted. */
  label: string;
  /** Group identifier. Devices sharing the same physical device. */
  groupId: string;
}

/** Version information (browser). The browser surface has no native audio
 *  backend, so it reports only the decibri version. */
export interface VersionInfo {
  /** decibri version. */
  decibri: string;
}

/** Constructor options for the browser `Microphone` class. */
export interface MicrophoneOptions {
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
   * Audio input device. Pass a deviceId string from `Microphone.devices()`.
   * Omit to use the system default input device.
   */
  device?: string;

  /**
   * Sample encoding data type.
   * - `'int16'`: Int16Array of PCM samples
   * - `'float32'`: Float32Array of samples in [-1, 1]
   * @default 'int16'
   */
  dtype?: 'int16' | 'float32';

  /**
   * Voice activity detection mode. One of:
   * - `false`: disabled (default)
   * - `'energy'`: RMS energy threshold
   *
   * The browser runs energy VAD only; Silero is Node-only. When enabled, emits
   * `'speech'` and `'silence'` events and updates `vadScore`. The legacy
   * `vad: true` form is rejected; specify the mode explicitly.
   * @default false
   */
  vad?: false | 'energy';

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
 * import { Microphone } from 'decibri'; // browser entry via conditional export
 *
 * const mic = new Microphone({ sampleRate: 16000 });
 * mic.on('data', (chunk) => {
 *   // chunk is an Int16Array of PCM samples
 * });
 * await mic.start(); // call from a user gesture in Safari
 * mic.stop();
 * ```
 */
export declare class Microphone {
  constructor(options?: MicrophoneOptions);

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
   * Most recent VAD score: the normalized RMS of the last chunk in `'energy'`
   * mode, or 0 when VAD is disabled or before the first chunk is processed.
   */
  readonly vadScore: number;

  /**
   * List available audio input devices.
   * Labels may be empty until microphone permission is granted.
   */
  static devices(): Promise<MicrophoneInfo[]>;

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

/** Constructor options for the browser `Speaker` class. */
export interface SpeakerOptions {
  /**
   * Sample rate of the audio you write, in Hz. Samples are resampled to the
   * audio context's native rate before playback.
   * @default 16000
   * @range 1000–384000
   */
  sampleRate?: number;

  /**
   * Number of output channels. A mono stream is played on every channel.
   * @default 1
   * @range 1–32
   */
  channels?: number;

  /**
   * Sample encoding data type of the audio you write.
   * - `'int16'`: Int16Array of PCM samples
   * - `'float32'`: Float32Array of samples in [-1, 1]
   * @default 'int16'
   */
  dtype?: 'int16' | 'float32';

  /**
   * Custom URL for the output AudioWorklet processor script.
   * Use this for strict CSP environments that block blob URLs.
   * Omit to use the default inline blob URL.
   */
  workletUrl?: string;
}

/**
 * Browser audio playback via the Web Audio API + AudioWorklet.
 *
 * Plays PCM audio. Samples are converted to float32 and resampled to the
 * context rate on the main thread, then fed to an output worklet that drains
 * them to the speakers. Playback is async throughout, as the browser requires:
 * `write()` resolves when the samples are queued and `drain()` resolves when
 * the queued audio has finished playing. The first `write()` (or `start()`)
 * must run in a user gesture so the browser allows audio.
 *
 * @example
 * ```js
 * import { Speaker } from 'decibri'; // browser entry via conditional export
 *
 * const speaker = new Speaker({ sampleRate: 16000 });
 * button.onclick = async () => {
 *   await speaker.write(int16Chunk); // Int16Array of PCM samples
 *   await speaker.drain();
 *   speaker.stop();
 * };
 * ```
 */
export declare class Speaker {
  constructor(options?: SpeakerOptions);

  /**
   * Create and resume the AudioContext and load the output worklet.
   * Must be called from a user gesture context so the browser allows audio.
   * Optional: `write()` starts playback on its own if `start()` was not called.
   */
  start(): Promise<void>;

  /**
   * Play PCM audio. Resolves when the samples are queued, applying backpressure
   * (the promise waits) when the playback queue is full. Await calls
   * sequentially to preserve sample order.
   */
  write(chunk: Int16Array | Float32Array | ArrayBuffer): Promise<void>;

  /**
   * Wait for all queued audio to finish playing. Resolves immediately if
   * nothing is queued.
   */
  drain(): Promise<void>;

  /** Immediate stop. Discards queued audio and releases all resources. */
  stop(): void;

  /** Whether audio is currently queued and playing. */
  readonly isPlaying: boolean;
}
