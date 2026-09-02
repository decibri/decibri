import { Readable, ReadableOptions, Writable, WritableOptions } from 'stream';

/** Information about an available audio input device. */
export interface MicrophoneInfo {
  /** Device index (pass to constructor as `device`). */
  index: number;
  /** Human-readable device name from the OS. */
  name: string;
  /**
   * Stable per-host device ID. Pass via `device: { id: ... }` for selection
   * that survives across enumerations. The lowercase host name, a colon, then
   * the platform device identifier:
   * - Windows (WASAPI): `wasapi:` then the endpoint ID (e.g.
   *   `wasapi:{0.0.1.00000000}.{...}`)
   * - macOS (CoreAudio): `coreaudio:` then the device UID
   * - Linux (ALSA): `alsa:` then the PCM identifier
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
   * - `'silero'`: Silero VAD v6.2 ML model (more accurate, ~1ms inference)
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

  /**
   * The 0-based DELIVERED channel the detector reads: the position within
   * the delivered interleaved frames, after any `channelMap` is applied (a
   * `channelMap` names device channels; `source` names the delivered
   * position). Must be below the delivered channel count, which is its only
   * ceiling; no fixed maximum exists. Affects only the detector feed; the
   * delivered audio is untouched.
   * @default the frame average of every delivered channel
   */
  source?: number;
}

/**
 * Acoustic echo cancellation config object, passed on the `aec` option to tune
 * the canceller. The bare `aec: 'tau'` shorthand selects the model with its
 * defaults; pass this object to override them.
 *
 * The canceller's behaviour while a lost delay alignment is being reacquired
 * is fixed: decibri applies the canceller's graded output transition and does
 * not expose a setting for it.
 */
export interface AecOptions {
  /**
   * Which echo canceller model to run. The accepted set is owned by the
   * canceller and grows the way `denoise` grows; today it is `'tau'`, a
   * classical adaptive canceller with no model file. An unknown name is
   * rejected with the canceller's own message naming the accepted set.
   */
  model: 'tau';

  /**
   * Adaptive filter tail length in milliseconds: how much echo delay spread
   * the canceller can model.
   * @default the canceller's own default (200)
   * @range 16 to 500
   */
  tailMs?: number;

  /**
   * Residual echo suppression policy. `'conservative'` attenuates the residual
   * echo the linear canceller leaves behind while keeping the near-end voice
   * intact; `'off'` delivers the linear canceller output as-is.
   * @default 'conservative'
   */
  suppression?: 'conservative' | 'off';

  /**
   * Sample rate in Hz of the far-end reference pushed through
   * `pushAecReference`. When it names a rate other than `sampleRate`, decibri
   * converts the reference before the canceller sees it: a reference at an
   * undeclared different rate cancels nothing and reports no error, so the
   * conversion is decibri's rather than the caller's.
   * @default the capture `sampleRate`
   * @range 1000 to 384000
   */
  referenceSampleRate?: number;

  /**
   * Number of channels in the far-end reference pushed through
   * `pushAecReference`, frame-interleaved. When it names a count above 1,
   * decibri averages each frame to one mono sample before the canceller sees
   * it: a multichannel reference pushed without declaring the count cancels
   * nothing and reports no error, so the collapse is decibri's rather than
   * the caller's.
   *
   * The declared count must match the buffer actually pushed. The reference
   * arrives as flat PCM whose true channel count is not recoverable from its
   * length, so a mismatch is not detected and raises no error: the frames
   * are misread, nothing is cancelled, and the observable signature is
   * `aecMetrics().delaySamples` staying `null` while the canceller reports
   * no fault.
   *
   * The canceller itself reads one mono reference. Against playback through
   * more than one loudspeaker that is a cancellation ceiling: the echo
   * reaching the microphone is the sum of different room responses driven by
   * different signals, and a single-reference canceller models one response
   * applied to their average, so a placement where those paths differ leaves
   * a residual that no amount of adaptation removes.
   * @default 1 (mono)
   * @range at least 1; no upper bound
   */
  referenceChannels?: number;
}

/**
 * One delivered channel's canceller report, one entry of
 * `AecMetrics.channels`. Engine-level fields only: the reference queue's
 * counters (`referenceDropped`, `referenceSilence`) describe the shared queue
 * and stay on `AecMetrics` itself.
 */
export interface AecChannelMetrics {
  /**
   * This channel's active delay alignment in samples, or `null` while its
   * estimator is still searching. The offset from the reference frontier as
   * the feeding established it, not a measurement of the room's echo path.
   */
  delaySamples: number | null;
  /**
   * This channel's smoothed echo-return-loss-enhancement estimate in dB. Not
   * a quality ranking across channels: ERLE rises with echo distance, because
   * a weaker echo is easier to reduce in ratio terms, so a far microphone
   * routinely reports a higher figure than a near one while removing less
   * echo in absolute terms. Compare a channel against its own history, not
   * against its neighbours.
   */
  erleDb: number;
  /**
   * Whether this channel's double-talk detector currently believes the
   * near-end talker is active; its adaptation is held while true.
   */
  doubleTalk: boolean;
  /**
   * Near-end samples this channel's canceller could find no far-end sample
   * for while an alignment was active.
   */
  referenceStarved: number;
  /**
   * Near-end samples this channel processed while no delay alignment was
   * active: the searching span, not a transport failure.
   */
  acquisitionParked: number;
  /**
   * Times this channel's canceller inferred a capture discontinuity and
   * rebuilt its alignment from the reference frontier.
   */
  referenceReanchors: number;
}

/**
 * The echo canceller's transport and cancellation metrics, returned by
 * `Microphone.aecMetrics()`. One object carries the canceller's own report and
 * the reference queue's counters. The top-level engine fields report the first
 * delivered channel's canceller; `channels` carries every delivered channel's
 * report, so the two agree on a single-channel stream.
 */
export interface AecMetrics {
  /**
   * The active delay alignment in samples, or `null` while the estimator is
   * still searching. Staying `null` while `acquisitionParked` climbs is the
   * signature of a canceller with no usable reference: none pushed, not at the
   * declared rate, or not the signal that produced the echo.
   */
  delaySamples: number | null;
  /**
   * Smoothed echo-return-loss-enhancement estimate in dB: how much echo the
   * canceller is currently removing. 0 before the filter has converged.
   */
  erleDb: number;
  /**
   * Whether the double-talk detector currently believes the near-end talker is
   * active; adaptation is held while true.
   */
  doubleTalk: boolean;
  /**
   * Near-end samples the canceller could find no far-end sample for while an
   * alignment was active. decibri keeps the far-end stream level with the
   * capture, so this stays 0 for a caller who simply stops pushing; a non-zero
   * count means the caller ran further ahead of the capture than the
   * canceller's far-end history reaches.
   */
  referenceStarved: number;
  /**
   * Near-end samples processed while no delay alignment was active: the
   * searching span, not a transport failure.
   */
  acquisitionParked: number;
  /**
   * Times the canceller inferred a capture discontinuity and rebuilt its
   * alignment from the reference frontier.
   */
  referenceReanchors: number;
  /**
   * Far-end samples discarded, at the declared reference rate: a single push
   * exceeded the reference queue's bound, or the push arrived while capture
   * was not running. The span an oversized push occupied is still represented
   * as silence, so that discard costs the cancellation of the span alone.
   */
  referenceDropped: number;
  /**
   * Far-end samples decibri supplied as silence because the caller had pushed
   * none for them, at the capture rate. An accounting figure, not a fault:
   * while nothing is playing, the far end is silence.
   */
  referenceSilence: number;
  /**
   * Every delivered channel's canceller report, in delivered order, one entry
   * per channel. One canceller engine runs per delivered channel, each fed
   * the same pushed reference and each finding its own channel's echo delay,
   * so the entries differ where the channels' acoustic paths differ. On a
   * single-channel stream this holds one entry agreeing with the top-level
   * fields.
   */
  channels: AecChannelMetrics[];
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
   * Number of channels the stream delivers, interleaved frame by frame in the
   * emitted chunks. Bounded below at `1` (the default); bounded above by the
   * resolved device alone, which reports its own count when the stream
   * starts. No fixed maximum exists.
   *
   * The device itself is opened at its own native channel count, exactly as
   * it is opened at its native rate, and decibri derives the delivered
   * channels from it. Without a `channelMap`: `1` delivers the documented
   * average of every opened channel; a count equal to the device's own
   * delivers every device channel in device order; a count above the
   * device's own fails `start()` with a `DecibriError` carrying the code
   * `'MICROPHONE_CHANNELS_UNSUPPORTED'`; and a count above `1` and below the
   * device's own fails it with `'CHANNEL_SELECTION_AMBIGUOUS'`, because
   * which channels it means has no single answer, so `channelMap` names
   * them. With `aec` set, one canceller runs per delivered channel, and
   * `aecMetrics().channels` reports each delivered channel's canceller in
   * delivered order.
   * @default 1
   */
  channels?: number;

  /**
   * Optional list of 0-based device channel indices selecting which device
   * channels feed the delivered channels: delivered channel `j` carries device
   * channel `channelMap[j]`. The length must equal `channels`. Entries may
   * repeat and may appear in any order, so a map both selects and permutes,
   * and may name more delivered channels than the device has. Absent derives
   * the delivered channels from `channels` as documented there.
   *
   * The same shape as CoreAudio AUHAL's channel map
   * (`kAudioOutputUnitProperty_ChannelMap`: an array of device channel
   * indices, one entry per client channel). NOT miniaudio's `channelMap`,
   * which names a spatial layout. Entries are validated against the resolved
   * device's own report when the stream starts: an entry the device does not
   * have throws a `DecibriError` with code `'CHANNEL_MAP_OUT_OF_RANGE'` naming
   * the entry and the count the device reports. The device's report is the
   * only ceiling; no fixed maximum exists.
   * @default undefined (the derivation `channels` documents)
   */
  channelMap?: number[];

  /**
   * Frames per audio callback buffer. Controls chunk size and delivery interval.
   * A chunk holds this many frames of the delivered channel count.
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
   * - `'silero'`: Silero VAD v6.2 ML model (more accurate, ~1ms inference)
   * - `'energy'`: RMS energy threshold (lightweight)
   * - a `VadOptions` config object `{ model, threshold?, holdoffMs?, source? }`
   *   to tune the threshold, holdoff, and detector source for the chosen model
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

  /**
   * Acoustic echo cancellation applied to the captured audio, removing the
   * echo of far-end audio the caller pushes through `pushAecReference`. The
   * `'tau'` shorthand names the model; an `AecOptions` object tunes it. Omit
   * to leave echo cancellation off (the default), which keeps the capture
   * path unchanged.
   *
   * Runs before the detector tap: with it on, `vadScore` and the `speech` /
   * `silence` events read the echo-removed signal, so playback stops
   * triggering detection. Requires `sampleRate` in 8000 to 48000, narrower
   * than the range the option otherwise accepts. With no reference pushed,
   * the captured audio passes through unchanged.
   *
   * Native capture only: the browser entry does not take this option, because
   * browser capture already carries the platform's own echo cancellation
   * through its `echoCancellation` constraint, on by default.
   * @default undefined
   */
  aec?: 'tau' | AecOptions;
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
   * 0 while the consumer keeps up, before capture starts, and after `stop()`,
   * which releases the stream the counter lives on. Read it before stopping to
   * see a session's total. A rising value means audio is being dropped to
   * bound memory.
   */
  readonly overrunCount: number;

  /**
   * Queue far-end reference audio for the echo canceller: the audio being
   * played out, pushed as it is played, in played order. Accepts the same
   * input shapes `Speaker.write` accepts (a `Buffer`, any TypedArray, or a
   * `DataView` of PCM bytes in this microphone's `dtype`), at the declared
   * `referenceSampleRate` (the capture rate when unset), interleaved at the
   * declared `referenceChannels` (mono when unset). With `referenceChannels`
   * above 1, each frame is averaged to one mono sample before the canceller
   * sees it. The declared count must match this buffer's actual
   * interleaving: a mismatch is not detected and raises no error, and shows
   * up only as `aecMetrics().delaySamples` staying `null` with no fault
   * reported.
   *
   * Never blocks and never throws on a full queue: samples that do not fit
   * are discarded and counted by `aecMetrics().referenceDropped`. Silence
   * between played audio need not be pushed. A push while capture is not
   * running is discarded and counted by `referenceDropped`, read once
   * capture runs; a push with the `aec` option unset is a no-op. A typed
   * array carrying a sample dtype other than the configured `dtype` throws
   * a `TypeError`, whatever the capture state; `Buffer`, `Uint8Array`, and
   * `DataView` are format-agnostic byte carriers.
   */
  pushAecReference(data: Buffer | NodeJS.ArrayBufferView): void;

  /**
   * The echo canceller's transport and cancellation metrics, merged with the
   * reference queue's counters, or `null` when the `aec` option is unset or
   * capture is not running.
   */
  aecMetrics(): AecMetrics | null;

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
   * source's input rate (from the file's header, or `inputRate` for
   * `File.buffer`) is resampled to this rate, so a 44.1 kHz recording comes
   * out at 16 kHz unless you set `sampleRate`. The same meaning the option
   * has on `Microphone`.
   * @default 16000
   * @range 1000–384000
   */
  sampleRate?: number;

  /**
   * Number of channels the File delivers, interleaved frame by frame in the
   * emitted chunks. Bounded below at `1` (the default); bounded above by the
   * source's own channel count alone, read from the container's header (or
   * `inputChannels` for `File.buffer`). No fixed maximum exists. The same
   * meaning the option has on `Microphone`, with the source's count standing
   * where the device's report stands.
   *
   * Without a `channelMap`: `1` delivers the documented average of every
   * source channel; a count equal to the source's own delivers every source
   * channel in source order; a count above the source's own throws a
   * `DecibriError` carrying the code `'FILE_CHANNELS_UNSUPPORTED'`; and a
   * count above `1` and below the source's own throws one with
   * `'FILE_CHANNEL_SELECTION_AMBIGUOUS'`, because which channels it means
   * has no single answer, so `channelMap` names them.
   * @default 1
   */
  channels?: number;

  /**
   * Optional list of 0-based source channel indices selecting which source
   * channels feed the delivered channels: delivered channel `j` carries
   * source channel `channelMap[j]`. The length must equal `channels`.
   * Entries may repeat and may appear in any order, so a map both selects
   * and permutes, and may name more delivered channels than the source has.
   * Absent derives the delivered channels from `channels` as documented
   * there. The same shape and semantics as the `Microphone` option, with
   * source channels in the device channels' place. An entry the source does
   * not have throws a `DecibriError` with code
   * `'FILE_CHANNEL_MAP_OUT_OF_RANGE'` naming the entry and the source's own
   * count. The source's count is the only ceiling; no fixed maximum exists.
   * @default undefined (the derivation `channels` documents)
   */
  channelMap?: number[];

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

/** Options for `File.buffer`: `FileOptions` plus the samples' own shape. */
export interface FileBufferOptions extends FileOptions {
  /**
   * The native rate of the in-memory samples in Hz. Required: raw samples
   * carry no header to read a rate from. The samples are resampled from this
   * rate to `sampleRate`.
   * @range 1000–384000
   */
  inputRate: number;

  /**
   * The interleave of the in-memory samples: how many channels each frame
   * carries. Raw samples carry no header to read a count from, so above
   * mono it is stated here, the channel counterpart of `inputRate`. The
   * samples' length must be a whole number of frames at this count. Applies
   * to `File.buffer` alone: a file's count is read from its own header, and
   * the option is refused on the open path.
   * @default 1
   * @range 1–65535
   */
  inputChannels?: number;
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

/** Options for `File.save` (also accepted by `AudioWriter`). */
export interface SaveOptions {
  /**
   * The container format to write. When not given it comes from the path's
   * extension: `.wav`, `.aiff`, `.aif`, `.aifc` or `.flac`. decibri reads a
   * file by its content and writes one by its name; an extension it does not
   * recognise is an error, never a silent default.
   */
  format?: 'wav' | 'aiff' | 'flac';

  /**
   * FLAC compression level. Higher levels search harder for a smaller file;
   * every level decodes to identical audio. Applies only to FLAC; ignored
   * for WAV and AIFF.
   * @default 5
   * @range 0 to 8
   */
  compression?: number;
}

/**
 * What a save did to the samples on their way into the file, resolved by
 * `File.save` and carried by `AudioWriter.report`.
 */
export interface SaveReport {
  /**
   * Finite samples outside full scale, clamped to [-1.0, 1.0]. Conditioned
   * audio can exceed full scale (AGC or AEC without a limiter), and 16-bit
   * PCM cannot hold that, so the overshoot clips and this count says how
   * much. The count is a statement about integer encodings: a float encoding
   * would preserve the overshoot instead, and would report zero.
   */
  clippedSamples: number;
  /**
   * Non-finite samples replaced before writing: NaN with silence, an
   * infinity with full scale. The same replacement on every format. When
   * the conditioning chain runs it has already replaced every non-finite
   * sample with silence at its entry, so this count covers the direct path
   * only (a mono source already at `sampleRate` with no conditioning
   * enabled).
   */
  nonFiniteSamples: number;
}

/**
 * Offline audio source: conditions a recording or in-memory samples through
 * the same chain as the live `Microphone`, delivered as a finite Readable
 * stream of conditioned chunks that ends at EOF (after the chain's
 * end-of-stream tail). Because a `File` is a complete recording, it can also
 * analyze the whole recording for speech with `analyze()` / `analyse()`,
 * which a live stream cannot do.
 *
 * Construction: `new File(path)` reads the file synchronously (fine for a
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
   * Open an audio file synchronously (blocks on disk I/O; prefer `File.open`
   * in servers). Reads WAV, AIFF, AIFF-C and FLAC, identified from the
   * file's own bytes rather than its extension; the input rate and channel
   * count come from the header.
   */
  constructor(path: string, options?: FileOptions);

  /**
   * Open an audio file without blocking the event loop: the disk read,
   * decode, and chain construction run on the native thread pool. The
   * recommended form, mirroring `Microphone.open`.
   */
  static open(path: string, options?: FileOptions): Promise<File>;

  /**
   * Wrap in-memory samples as an offline source. `samples` must be a
   * `Float32Array` of samples in [-1.0, 1.0], frame-interleaved at
   * `inputChannels` (1, mono, by default); a raw `Buffer` of PCM bytes is
   * rejected as ambiguous. `inputRate` is required (raw samples carry no
   * header). Synchronous: no I/O is involved.
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
   * The rate every delivered chunk carries: the `sampleRate` option, or
   * 16000 when it was not given.
   */
  readonly sampleRate: number;

  /**
   * The source's own rate, taken from the file's header or from the
   * `inputRate` passed to `File.buffer`. Differs from `sampleRate` when the
   * source was resampled.
   */
  readonly inputRate: number;

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

  /**
   * Write the conditioned recording to disk, off the event loop. Runs the
   * recording once through the same conditioning pass iteration delivers,
   * whole, and writes it as 16-bit PCM at `sampleRate`, frame-interleaved at
   * the delivered channel count. The container comes from the path's
   * extension (`.wav`, `.aiff`, `.aif`, `.aifc` or `.flac`), or from
   * `options.format`: decibri reads a file by its content and writes one by
   * its name. Consumes the source (a `File` is a single pass).
   *
   * Resolves to a `SaveReport`: how many samples were clamped to full scale
   * and how many non-finite samples were replaced (NaN as silence, an
   * infinity as full scale). When the conditioning chain runs it has already
   * replaced every non-finite sample with silence at its entry, so the file
   * carries silence where one was and the count covers the direct path only
   * (a mono source already at `sampleRate` with no conditioning enabled).
   *
   * Requires a File that is not already being streamed: once the stream has
   * been engaged this rejects with a `DecibriError` carrying the code
   * `'FILE_ENGAGED'`. Every failure detected before the pass begins leaves
   * the File usable; a failure during the pass consumes the source.
   */
  save(path: string, options?: SaveOptions): Promise<SaveReport>;

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

/** Constructor options for `AudioWriter`: the save options plus the stream's own description. */
export interface AudioWriterOptions extends SaveOptions, WritableOptions {
  /**
   * The rate of the incoming samples in Hz, written into the file's header.
   * Required: raw audio carries no header to read a rate from.
   * @range 1000 to 384000
   */
  sampleRate: number;

  /**
   * Number of channels the incoming bytes are interleaved at, written into
   * the file's header. The stream's total sample count must be a whole
   * number of frames at this count. Bounded below at `1` (the default);
   * above it, each container's own ceiling applies (a FLAC frame carries at
   * most 8 channels; a WAV `fmt ` chunk's `nBlockAlign` is a 16-bit field,
   * so 16-bit samples allow at most 32767; an AIFF `COMM` chunk's
   * `numChannels` is a signed 16-bit field, so at most 32767), reported
   * when the stream finishes as the container layer's own refusal. decibri
   * enforces no ceiling of its own.
   * @default 1
   */
  channels?: number;

  /**
   * Sample encoding of the incoming bytes.
   * - `'int16'`: 16-bit signed integer, little-endian, what a `File` or
   *   `Microphone` emits by default
   * - `'float32'`: 32-bit IEEE 754 float, little-endian
   * @default 'int16'
   */
  dtype?: 'int16' | 'float32';
}

/**
 * A file sink for PCM audio: the Writable to pair with decibri's Readable
 * sources, and with any other stream of PCM bytes (a TTS engine, a decoded
 * network stream). Collects the whole stream, then writes it as one audio
 * file when the stream finishes, exactly as `File.save` writes: the same
 * containers from the same extension rule, the same 16-bit PCM encoding, the
 * same clamp and non-finite handling, the same bytes.
 *
 * `'finish'` fires after the file is on disk, and `report` then carries the
 * `SaveReport` the write produced. A failure destroys the stream with the
 * error.
 *
 * @example
 * const { pipeline } = require('node:stream/promises');
 * const { File, AudioWriter } = require('decibri');
 * await pipeline(
 *   new File('noisy.wav', { denoise: 'fastenhancer-t' }),
 *   new AudioWriter('clean.flac', { sampleRate: 16000 }),
 * );
 */
export declare class AudioWriter extends Writable {
  constructor(path: string, options: AudioWriterOptions);

  /**
   * The `SaveReport` of the completed write, exactly as `File.save` resolves
   * it. `null` until `'finish'` has fired.
   */
  readonly report: SaveReport | null;
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
   * Number of output channels. The maximum is the device's: a count the device
   * cannot serve throws a `DecibriError` with code
   * `'SPEAKER_CHANNELS_UNSUPPORTED'` naming the count the device reports.
   * @default 1
   * @range 1 or more
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

  /**
   * Number of samples emitted as silence fill because the playback queue ran
   * dry. 0 while the producer keeps the queue fed, or before playback starts.
   * A rising value means the output is papering over gaps with silence.
   */
  readonly underrunCount: number;

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
