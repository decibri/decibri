'use strict';

const { Readable } = require('stream');
const path = require('path');
const fs = require('fs');
const { DecibriBridge, FileHandle } = require('../index.js');
const {
  wrapNativeError,
  DecibriError,
  DeviceError,
  OrtError,
  OrtPathError,
} = require('./errors');

// The npm package version, reported as `binding` by version(). Read from
// package.json so it tracks the published package and cannot drift from a
// hardcoded string.
const PACKAGE_VERSION = require('../package.json').version;

// ─── Bundled ONNX Runtime path resolution ────────────────────────────────────

/**
 * Map (process.platform, process.arch) → { pkg: npm platform package name,
 *                                           file: bundled ORT dylib filename }
 *
 * Dylib filenames are unversioned across all platforms for consistency. The
 * release workflow (.github/workflows/release.yml) copies Microsoft's
 * versioned upstream tarball file (e.g. libonnxruntime.1.24.4.dylib) into the
 * platform package with the unversioned name listed here.
 *
 * If you add a new platform, update this table AND the matching platform job
 * in release.yml.
 */
const PLATFORM_DYLIB = {
  'darwin-arm64': { pkg: '@decibri/decibri-darwin-arm64',    file: 'libonnxruntime.dylib' },
  'linux-x64':    { pkg: '@decibri/decibri-linux-x64-gnu',   file: 'libonnxruntime.so' },
  'linux-arm64':  { pkg: '@decibri/decibri-linux-arm64-gnu', file: 'libonnxruntime.so' },
  'win32-x64':    { pkg: '@decibri/decibri-win32-x64-msvc',  file: 'onnxruntime.dll' },
};

/**
 * Resolve the absolute path to the bundled ONNX Runtime shared library for
 * this platform.
 *
 * Returns a string path on success. Returns undefined when:
 *   - the platform/arch pair is not in PLATFORM_DYLIB (unsupported target);
 *   - the platform package is not installed (MODULE_NOT_FOUND from
 *     require.resolve, e.g. during pre-publish development);
 *   - any other path-resolution error occurs.
 *
 * When this returns undefined, the Rust layer falls back to the
 * ORT_DYLIB_PATH environment variable via ort::init(); if that is also
 * missing, the first Silero VAD construction fails with a decibri-specific
 * error message telling the user what to do.
 *
 * This function deliberately does NOT check that the dylib file exists on
 * disk. If the platform package was installed but is missing the dylib,
 * that's a packaging bug that should surface loudly via the Rust error path,
 * not be silently masked here.
 */
function resolveBundledOrtPath() {
  const entry = PLATFORM_DYLIB[`${process.platform}-${process.arch}`];
  if (!entry) return undefined;
  try {
    // require.resolve on the package name gives us the absolute path of its
    // main entry (the .node binary). path.dirname() yields the package dir
    // where the bundled ORT dylib sits alongside.
    const nodeBinaryPath = require.resolve(entry.pkg);
    return path.join(path.dirname(nodeBinaryPath), entry.file);
  } catch (_) {
    return undefined;
  }
}

// ─── Microphone (Readable) ──────────────────────────────────────────────────

class Microphone extends Readable {
  /**
   * @param {import('./decibri').MicrophoneOptions} [options]
   * @param {{ prepared: object, native: object }} [_internal] Internal: a
   *   pre-resolved options bundle and an already-constructed native bridge,
   *   passed by the async `Microphone.open()` factory so the heavy open work
   *   (the Silero model load) is not repeated on the event loop. Not part of
   *   the public API.
   */
  constructor(options = {}, _internal = undefined) {
    super({ highWaterMark: options.highWaterMark, objectMode: false });

    // Validate and resolve options once. The async factory passes its already
    // resolved bundle through `_internal` to avoid recomputing it.
    const prepared = _internal ? _internal.prepared : Microphone._prepareOptions(options);

    // ── Store config ───────────────────────────────────────────────────────

    this._vad = prepared.vadEnabled;
    this._vadThreshold = prepared.vadThreshold;
    this._vadHoldoff = prepared.vadHoldoff;
    this._vadScore = 0;
    this._isSpeaking = false;
    this._silenceTimer = null;
    this._started = false;
    // Terminal once stop() runs: blocks _read() from restarting native capture
    // while the end-of-stream push(null) is deferred past the flushed tail.
    this._stopped = false;
    // Set just before the deferred push(null); the data callback drops any
    // straggler that arrives after end-of-stream rather than pushing past EOF.
    this._ended = false;

    // ── Create or adopt native bridge ───────────────────────────────────────

    if (_internal) {
      // Built off the event loop by Microphone.open(); already wrapped.
      this._native = _internal.native;
    } else {
      try {
        this._native = new DecibriBridge(prepared.nativeOptions);
      } catch (err) {
        throw wrapNativeError(err);
      }
    }
  }

  /**
   * Validate the constructor options and resolve them into the native options
   * object plus the wrapper-side state. Throws the same `RangeError` /
   * `TypeError` / `Error` as the constructor on invalid input. Shared by the
   * synchronous constructor and the async `open()` factory so both validate
   * identically. Does no native open work beyond the numeric-device bounds
   * check (a fast device enumeration).
   * @internal
   * @param {import('./decibri').MicrophoneOptions} options
   */
  static _prepareOptions(options) {
    // ── Validate options ───────────────────────────────────────────────────

    const sampleRate = options.sampleRate ?? 16000;
    if (sampleRate < 1000 || sampleRate > 384000) {
      throw new RangeError('sample rate must be between 1000 and 384000');
    }

    // Mono only: the capture path delivers a single channel. A value below 1
    // is a plain range error; a value above 1 is rejected as multichannel
    // (not silently downmixed) so a later move to true multichannel stays
    // additive. The `channels` option is kept for that forward compatibility.
    const channels = options.channels ?? 1;
    if (channels < 1) {
      throw new RangeError('channels must be between 1 and 32');
    }
    if (channels > 1) {
      throw new RangeError('multichannel capture is not supported; channels must be 1 (mono)');
    }

    const framesPerBuffer = options.framesPerBuffer ?? 1600;
    if (framesPerBuffer < 64 || framesPerBuffer > 65536) {
      throw new RangeError('frames per buffer must be between 64 and 65536');
    }

    const dtype = options.dtype ?? 'int16';
    if (dtype !== 'int16' && dtype !== 'float32') {
      throw new TypeError("dtype must be 'int16' or 'float32'");
    }

    // ── Resolve device ──────────────────────────────────────────────────────

    // Name and multi-match resolution are delegated to the core, which owns
    // the renamed-vocabulary errors (MicrophoneNotFound / MultipleDevicesMatch).
    // A string name and an { id } object are passed straight through to the
    // native addon. Only the numeric index keeps a client-side bounds check,
    // for a clean Node-side RangeError without a round-trip.
    let resolvedDevice = options.device;
    if (typeof options.device === 'number') {
      const devices = DecibriBridge.devices();
      if (options.device < 0 || options.device >= devices.length) {
        throw new RangeError('device index out of range. Call Microphone.devices() to list available devices');
      }
      resolvedDevice = options.device;
    } else if (
      options.device !== null &&
      typeof options.device === 'object' &&
      !Array.isArray(options.device)
    ) {
      // Object form: { id: string } selects by stable per-host device ID
      // (cpal DeviceId). Pass through to Rust; Rust's resolve_device_option
      // matches against cpal's DeviceId Display output.
      if (typeof options.device.id !== 'string') {
        throw new TypeError('device.id must be a string');
      }
      resolvedDevice = options.device;
    }

    // ── Validate VAD options ─────────────────────────────────────────────────

    // vad selects the detector and (optionally) its threshold/holdoff policy.
    // It accepts false (disabled, default), the 'silero'/'energy' shorthand
    // (which uses the mode's default threshold and holdoff), or a config object
    // { model, threshold, holdoffMs } to tune the policy. The legacy two-flag
    // form (vad: true plus vadMode) and the flat vadThreshold/vadHoldoff
    // options are rejected with a migration error. The threshold and holdoff
    // live JS-side (the state machine runs in this wrapper); only the mode is
    // passed to native.
    if (options.vadThreshold !== undefined || options.vadHoldoff !== undefined) {
      throw new TypeError(
        'vadThreshold and vadHoldoff are no longer supported. ' +
          "Pass them on the vad config object: vad: { model: 'silero', threshold: 0.5, holdoffMs: 300 }."
      );
    }
    const vad = options.vad ?? false;
    let vadEnabled;
    let vadMode;
    let vadThreshold;
    let vadHoldoff;
    if (vad === false) {
      vadEnabled = false;
      vadMode = 'energy'; // inert placeholder; ignored while disabled
    } else if (vad === true) {
      throw new TypeError(
        "vad: true is no longer supported. Specify the mode explicitly: vad: 'silero' or vad: 'energy'."
      );
    } else if (vad === 'silero' || vad === 'energy') {
      vadEnabled = true;
      vadMode = vad;
    } else if (vad !== null && typeof vad === 'object' && !Array.isArray(vad)) {
      // Config object form: { model, threshold?, holdoffMs? }. model is required
      // and selects the detector; threshold and holdoffMs override the mode
      // defaults when supplied.
      const { model, threshold, holdoffMs } = vad;
      if (model !== 'silero' && model !== 'energy') {
        throw new TypeError(
          `Invalid vad model: ${JSON.stringify(model)}. Expected 'silero' or 'energy'.`
        );
      }
      vadEnabled = true;
      vadMode = model;
      if (threshold !== undefined) {
        if (typeof threshold !== 'number' || Number.isNaN(threshold)) {
          throw new TypeError('vad threshold must be a number');
        }
        if (threshold < 0 || threshold > 1) {
          throw new RangeError('vad threshold must be between 0 and 1');
        }
        vadThreshold = threshold;
      }
      if (holdoffMs !== undefined) {
        if (typeof holdoffMs !== 'number' || Number.isNaN(holdoffMs)) {
          throw new TypeError('vad holdoffMs must be a number');
        }
        if (holdoffMs < 0) {
          throw new RangeError('vad holdoffMs must be non-negative');
        }
        vadHoldoff = holdoffMs;
      }
    } else {
      throw new TypeError(
        `Invalid vad value: ${JSON.stringify(vad)}. Expected false, 'silero', 'energy', or a config object { model, threshold, holdoffMs }.`
      );
    }

    let modelPath = undefined;
    if (vadEnabled && vadMode === 'silero') {
      modelPath = options.modelPath || path.join(__dirname, '..', 'models', 'silero_vad.onnx');
      if (!fs.existsSync(modelPath)) {
        throw new Error(`Silero VAD model not found at ${modelPath}. Ensure the models/ directory is included in your installation.`);
      }
    }

    // ── DC removal ───────────────────────────────────────────────────────────

    // A plain bool toggle for the one-pole DC-blocking high-pass, the first
    // transform stage (before denoise). Absent or false leaves it off (the
    // default), a byte-identical no-op. No range or closed-set check (it is just
    // a bool); the value is threaded straight to the native config.
    const dcRemoval = options.dcRemoval;

    // ── Validate and resolve denoise ─────────────────────────────────────────

    // Closed-set selector mirroring the Silero VAD shape: a model name resolves
    // to a bundled ONNX file, absence leaves denoise off. The only accepted
    // value is 'fastenhancer-t'; anything else is an explicit error rather than
    // a silent miss. The bundled model file is resolved relative to the package
    // exactly as the Silero model is.
    const denoise = options.denoise;
    let denoiseModelPath = undefined;
    if (denoise !== undefined) {
      if (denoise !== 'fastenhancer-t') {
        throw new TypeError(
          `Invalid denoise value: ${JSON.stringify(denoise)}. Expected 'fastenhancer-t'.`
        );
      }
      denoiseModelPath = path.join(__dirname, '..', 'models', 'fastenhancer_t.onnx');
      if (!fs.existsSync(denoiseModelPath)) {
        throw new Error(`Denoise model not found at ${denoiseModelPath}. Ensure the models/ directory is included in your installation.`);
      }
    }

    // ── Validate high-pass ───────────────────────────────────────────────────

    // Closed growable numeric cutoff selector mirroring the denoise shape: a
    // cutoff in Hz selects a filter, absence leaves the high-pass off. The
    // accepted values are 80 and 100; anything else (an out-of-set cutoff or a
    // non-number) is an explicit RangeError rather than a silent miss, matching
    // the numeric range checks on agc and limiter. The filter is pure DSP with
    // no bundled file, so there is nothing to resolve here, only the closed-set
    // check.
    const highpass = options.highpass;
    if (highpass !== undefined && highpass !== 80 && highpass !== 100) {
      throw new RangeError('highpass must be one of: 80, 100');
    }

    // ── Validate AGC ─────────────────────────────────────────────────────────

    // AGC target level in dBFS: a number in [-40, -3] (typical -18) drives the
    // captured level toward the target; absence leaves it off. Mirrors the
    // sample-rate range check, a RangeError on an out-of-range numeric value;
    // the native backstop and the Rust core guard the same range.
    const agc = options.agc;
    if (agc !== undefined && (agc < -40 || agc > -3)) {
      throw new RangeError('agc target level must be between -40 and -3');
    }

    // ── Validate limiter ─────────────────────────────────────────────────────

    // Sample-peak ceiling in dBFS: a number in [-3.0, 0.0] (typical -1.0) holds
    // the captured signal at or below the ceiling, catching a peak the AGC would
    // let through; absence leaves it off. Mirrors the agc range check, a
    // RangeError on an out-of-range numeric value; the native backstop and the
    // Rust core guard the same range.
    const limiter = options.limiter;
    if (limiter !== undefined && (limiter < -3.0 || limiter > 0.0)) {
      throw new RangeError('limiter ceiling must be between -3.0 and 0.0');
    }

    // Internal plumbing: inject the bundled ORT dylib path into the napi
    // constructor whenever an ONNX stage loads (Silero VAD or denoise). If
    // resolution fails (unknown platform, platform package not installed), this
    // is left undefined and Rust falls through to ORT_DYLIB_PATH or surfaces a
    // decibri-specific init error.
    let ortLibraryPath = undefined;
    if ((vadEnabled && vadMode === 'silero') || denoise !== undefined) {
      ortLibraryPath = resolveBundledOrtPath();
    }

    return {
      dtype,
      vadEnabled,
      vadMode,
      vadThreshold: vadThreshold ?? (vadMode === 'silero' ? 0.5 : 0.01),
      vadHoldoff: vadHoldoff ?? 300,
      nativeOptions: {
        sampleRate,
        channels,
        framesPerBuffer,
        format: dtype,
        device: resolvedDevice,
        // Pass the mode to native only when VAD is enabled. When disabled the
        // local vadMode is an inert 'energy' placeholder; sending it would make
        // native compute the energy score for a microphone that did not ask for
        // VAD. Absent means VAD off in native.
        vadMode: vadEnabled ? vadMode : undefined,
        modelPath,
        dcRemoval,
        denoise,
        denoiseModelPath,
        ortLibraryPath,
        highpass,
        agc,
        limiter,
      },
    };
  }

  /**
   * Construct a Microphone without blocking the event loop on the open work.
   *
   * The synchronous `new Microphone(...)` constructor loads the Silero VAD
   * model inline when `vad: 'silero'` is set, which blocks the event loop for
   * roughly 100 to 500 ms on a cold cache. This static factory runs that load
   * (and device resolution) on the native thread pool and resolves to a ready
   * instance, so latency-sensitive callers (voice pipelines, websocket
   * handlers) do not stall. The synchronous constructor remains available and
   * unchanged.
   *
   * Mirrors the Python `AsyncMicrophone.open()` factory. Options are identical
   * to the constructor. A failed open (bad model path, unknown device, ORT
   * load failure) rejects the returned Promise with the matching error class
   * (`RangeError` / `TypeError` for invalid options, `DeviceError` / `OrtError`
   * / `OrtPathError` for native failures), rather than throwing synchronously.
   *
   * @param {import('./decibri').MicrophoneOptions} [options]
   * @returns {Promise<Microphone>}
   */
  static async open(options = {}) {
    const prepared = Microphone._prepareOptions(options);
    let native;
    try {
      native = await DecibriBridge.openAsync(prepared.nativeOptions);
    } catch (err) {
      throw wrapNativeError(err);
    }
    return new Microphone(options, { prepared, native });
  }

  /** @internal */
  _read() {
    // Never (re)start native capture once stopped: stop() defers the
    // end-of-stream push(null), and a _read() in that window would otherwise
    // reopen the device.
    if (this._started || this._stopped) return;
    this._started = true;

    this._native.start((err, chunk) => {
      if (err) {
        this._started = false;
        // A start()-time failure is delivered here as the raw native error (not
        // via wrapNativeError), so no decibri `code` is attached on the 'error'
        // event, as with device-open failures; the dedicated MODEL_LOAD_FAILED
        // code is reachable through wrapNativeError, not on this streaming path.
        this.destroy(err);
        return;
      }

      // On close the native pump flushes the buffered tail; that final data
      // callback can run after stop() has begun ending the stream. Once the
      // end-of-stream push(null) has been issued (or the stream was destroyed),
      // drop any straggler rather than pushing past EOF.
      if (this._ended || this.destroyed) {
        return;
      }

      // push returns false when the consumer is slow. We can't pause a mic,
      // but we surface the backpressure warning so callers can react.
      if (!this.push(chunk)) {
        this.emit('backpressure');
      }

      if (this._vad) {
        // Both modes read the score from native: the Silero speech probability
        // or, in energy mode, the RMS of the pre-enhancement signal. The native
        // pump computes both on the signal before the opt-in enhancement step,
        // so enabling enhancement does not change detection in either mode.
        this._processVadValue(this._native.vadProbability);
      }
    });
  }

  /** @internal Common speech/silence state machine */
  _processVadValue(value) {
    this._vadScore = value;
    if (value >= this._vadThreshold) {
      clearTimeout(this._silenceTimer);
      this._silenceTimer = null;
      if (!this._isSpeaking) {
        this._isSpeaking = true;
        this.emit('speech');
      }
    } else if (this._isSpeaking && !this._silenceTimer) {
      this._silenceTimer = setTimeout(() => {
        this._isSpeaking = false;
        this._silenceTimer = null;
        this.emit('silence');
      }, this._vadHoldoff);
    }
  }

  /**
   * Stop microphone capture and end the stream cleanly.
   */
  stop() {
    if (!this._started) return;
    this._started = false;
    this._stopped = true;
    this._native.stop();
    clearTimeout(this._silenceTimer);
    this._silenceTimer = null;
    // The native pump flushes any buffered tail on close; those final data
    // callbacks are queued on the event loop and run before a setImmediate
    // scheduled now. Defer the end-of-stream push(null) past them so the final
    // short chunk reaches consumers without landing after EOF
    // (ERR_STREAM_PUSH_AFTER_EOF). `_ended` is set first so the data callback
    // drops any straggler that somehow arrives after this point.
    setImmediate(() => {
      // A device error in the same tick may have destroyed the stream before
      // this runs; don't end an already-destroyed stream.
      if (this.destroyed) return;
      this._ended = true;
      this.push(null); // signals stream end
    });
  }

  /**
   * Whether the microphone is currently open.
   * @returns {boolean}
   */
  get isOpen() {
    return this._native.isOpen;
  }

  /**
   * Most recent VAD score for the active mode: the Silero speech probability
   * in 'silero' mode, the normalized RMS of the pre-enhancement signal in
   * 'energy' mode. Both are computed natively on the signal before any opt-in
   * enhancement step, so enabling enhancement does not change the score.
   * 0 when VAD is disabled or before the first chunk is processed.
   * @returns {number}
   */
  get vadScore() {
    return this._vadScore;
  }

  /**
   * Number of capture buffers dropped because the consumer could not keep
   * pace. 0 while the consumer keeps up, or before capture starts. A rising
   * value means audio is being dropped to bound memory.
   * @returns {number}
   */
  get overrunCount() {
    return this._native.overrunCount;
  }

  /**
   * List all available input devices on the system.
   * @returns {Array<{index: number, name: string, id: string, maxInputChannels: number, defaultSampleRate: number, isDefault: boolean}>}
   */
  static devices() {
    return DecibriBridge.devices();
  }

  /**
   * Version information for decibri, the audio backend, and this binding.
   * @returns {{ decibri: string, audioBackend: string, binding: string }}
   */
  static version() {
    const v = DecibriBridge.version();
    return { decibri: v.decibri, audioBackend: v.audioBackend, binding: PACKAGE_VERSION };
  }
}

const Speaker = require('./decibri-output.js');

/**
 * List all available audio input devices.
 * @returns {Array<{index: number, name: string, maxInputChannels: number, defaultSampleRate: number, isDefault: boolean}>}
 */
function inputDevices() {
  return Microphone.devices();
}

/**
 * List all available audio output devices.
 * @returns {Array<{index: number, name: string, maxOutputChannels: number, defaultSampleRate: number, isDefault: boolean}>}
 */
function outputDevices() {
  return Speaker.devices();
}

/**
 * Version information for decibri, the audio backend, and this binding.
 * @returns {{ decibri: string, audioBackend: string, binding: string }}
 */
function version() {
  return Microphone.version();
}

// ─── File (Readable): offline source ─────────────────────────────────────────

class File extends Readable {
  /**
   * Open a WAV file as an offline source, synchronously. Everything a
   * `Microphone` does to live audio, a `File` does to audio you already
   * have: the same conditioning options, the same stream of conditioned
   * chunks, and (with `vad` set) the same per-chunk speech events, plus the
   * whole-file `analyze()` a live stream cannot offer.
   *
   * The bare constructor reads the WAV inline, blocking the event loop on
   * disk I/O; prefer `await File.open(path, options)` in servers and other
   * latency-sensitive code, exactly as `Microphone.open` is preferred over
   * `new Microphone`. Iteration and analysis are separate single passes:
   * each consumes the source once, so use one `File` per operation.
   *
   * @param {string} filePath Path to a WAV file (16-bit PCM or 32-bit float).
   * @param {import('./decibri').FileOptions} [options]
   * @param {{ prepared: object, native: object }} [_internal] Internal: a
   *   pre-resolved options bundle and an already-constructed native handle,
   *   passed by the async `File.open()` factory and by `File.buffer()`. Not
   *   part of the public API.
   */
  constructor(filePath, options = {}, _internal = undefined) {
    super({ highWaterMark: options.highWaterMark, objectMode: false });

    const prepared = _internal ? _internal.prepared : File._prepareOptions(options);

    // ── Store config ───────────────────────────────────────────────────────

    this._vad = prepared.vadEnabled;
    this._vadMode = prepared.vadMode;
    this._vadThreshold = prepared.vadThreshold;
    // The speaking holdoff on a File is measured in FILE time (sample
    // positions converted to seconds), never wall-clock time: a file
    // processes faster than real time, so a wall-clock timer would collapse
    // the reported speech timing. Positions advance as chunks are pulled.
    this._vadHoldoffSeconds = prepared.vadHoldoff / 1000;
    this._vadScore = 0;
    this._isSpeaking = false;
    this._silenceStartPos = null;
    this._position = 0;
    this._sampleRate = prepared.nativeOptions.sampleRate;
    this._bytesPerSample = prepared.dtype === 'int16' ? 2 : 4;
    this._ended = false;

    // ── Create or adopt native handle ───────────────────────────────────────

    if (_internal) {
      this._native = _internal.native;
    } else {
      if (typeof filePath !== 'string') {
        throw new TypeError('path must be a string');
      }
      try {
        this._native = FileHandle.open(filePath, prepared.nativeOptions);
      } catch (err) {
        throw wrapNativeError(err);
      }
    }
  }

  /**
   * Validate the constructor options and resolve them into the native options
   * object plus the wrapper-side state. The checks and messages mirror
   * `Microphone._prepareOptions` exactly for every shared option; the
   * live-capture-only options (device, channels, framesPerBuffer) do not
   * apply to an offline source.
   * @internal
   * @param {import('./decibri').FileOptions} options
   */
  static _prepareOptions(options) {
    const sampleRate = options.sampleRate ?? 16000;
    if (sampleRate < 1000 || sampleRate > 384000) {
      throw new RangeError('sample rate must be between 1000 and 384000');
    }

    const dtype = options.dtype ?? 'int16';
    if (dtype !== 'int16' && dtype !== 'float32') {
      throw new TypeError("dtype must be 'int16' or 'float32'");
    }

    // ── Validate VAD options (same acceptance as Microphone) ────────────────

    // The flat vadThreshold/vadHoldoff forms are rejected with the same
    // migration error the Microphone raises; pass them on the vad config object.
    if (options.vadThreshold !== undefined || options.vadHoldoff !== undefined) {
      throw new TypeError(
        'vadThreshold and vadHoldoff are no longer supported. ' +
          "Pass them on the vad config object: vad: { model: 'silero', threshold: 0.5, holdoffMs: 300 }."
      );
    }
    const vad = options.vad ?? false;
    let vadEnabled;
    let vadMode;
    let vadThreshold;
    let vadHoldoff;
    if (vad === false) {
      vadEnabled = false;
      vadMode = 'energy'; // inert placeholder; ignored while disabled
    } else if (vad === true) {
      throw new TypeError(
        "vad: true is no longer supported. Specify the mode explicitly: vad: 'silero' or vad: 'energy'."
      );
    } else if (vad === 'silero' || vad === 'energy') {
      vadEnabled = true;
      vadMode = vad;
    } else if (vad !== null && typeof vad === 'object' && !Array.isArray(vad)) {
      const { model, threshold, holdoffMs } = vad;
      if (model !== 'silero' && model !== 'energy') {
        throw new TypeError(
          `Invalid vad model: ${JSON.stringify(model)}. Expected 'silero' or 'energy'.`
        );
      }
      vadEnabled = true;
      vadMode = model;
      if (threshold !== undefined) {
        if (typeof threshold !== 'number' || Number.isNaN(threshold)) {
          throw new TypeError('vad threshold must be a number');
        }
        if (threshold < 0 || threshold > 1) {
          throw new RangeError('vad threshold must be between 0 and 1');
        }
        vadThreshold = threshold;
      }
      if (holdoffMs !== undefined) {
        if (typeof holdoffMs !== 'number' || Number.isNaN(holdoffMs)) {
          throw new TypeError('vad holdoffMs must be a number');
        }
        if (holdoffMs < 0) {
          throw new RangeError('vad holdoffMs must be non-negative');
        }
        vadHoldoff = holdoffMs;
      }
    } else {
      throw new TypeError(
        `Invalid vad value: ${JSON.stringify(vad)}. Expected false, 'silero', 'energy', or a config object { model, threshold, holdoffMs }.`
      );
    }

    let modelPath = undefined;
    if (vadEnabled && vadMode === 'silero') {
      modelPath = options.modelPath || path.join(__dirname, '..', 'models', 'silero_vad.onnx');
      if (!fs.existsSync(modelPath)) {
        throw new Error(`Silero VAD model not found at ${modelPath}. Ensure the models/ directory is included in your installation.`);
      }
    }

    // ── Conditioning options (identical checks to Microphone) ───────────────

    const dcRemoval = options.dcRemoval;

    const denoise = options.denoise;
    let denoiseModelPath = undefined;
    if (denoise !== undefined) {
      if (denoise !== 'fastenhancer-t') {
        throw new TypeError(
          `Invalid denoise value: ${JSON.stringify(denoise)}. Expected 'fastenhancer-t'.`
        );
      }
      denoiseModelPath = path.join(__dirname, '..', 'models', 'fastenhancer_t.onnx');
      if (!fs.existsSync(denoiseModelPath)) {
        throw new Error(`Denoise model not found at ${denoiseModelPath}. Ensure the models/ directory is included in your installation.`);
      }
    }

    const highpass = options.highpass;
    if (highpass !== undefined && highpass !== 80 && highpass !== 100) {
      throw new RangeError('highpass must be one of: 80, 100');
    }

    const agc = options.agc;
    if (agc !== undefined && (agc < -40 || agc > -3)) {
      throw new RangeError('agc target level must be between -40 and -3');
    }

    const limiter = options.limiter;
    if (limiter !== undefined && (limiter < -3.0 || limiter > 0.0)) {
      throw new RangeError('limiter ceiling must be between -3.0 and 0.0');
    }

    let ortLibraryPath = undefined;
    if ((vadEnabled && vadMode === 'silero') || denoise !== undefined) {
      ortLibraryPath = resolveBundledOrtPath();
    }

    return {
      dtype,
      vadEnabled,
      vadMode,
      vadThreshold: vadThreshold ?? (vadMode === 'silero' ? 0.5 : 0.01),
      vadHoldoff: vadHoldoff ?? 300,
      nativeOptions: {
        sampleRate,
        format: dtype,
        // Pass the mode to native only when VAD is enabled, exactly as the
        // Microphone options do; absent means VAD off in native.
        vadMode: vadEnabled ? vadMode : undefined,
        // The whole-file analysis applies threshold and holdoff in the core
        // (segment merging in file time), so both cross the boundary here,
        // unlike the live path where the policy is wrapper-only.
        vadThreshold: vadThreshold ?? (vadMode === 'silero' ? 0.5 : 0.01),
        vadHoldoffMs: vadHoldoff ?? 300,
        modelPath,
        dcRemoval,
        denoise,
        denoiseModelPath,
        ortLibraryPath,
        highpass,
        agc,
        limiter,
      },
    };
  }

  /**
   * Open a WAV file without blocking the event loop: the disk read, WAV
   * parse, and chain construction run on the native thread pool. The
   * recommended form in Node, mirroring `Microphone.open`. The synchronous
   * `new File(path)` remains available for scripts.
   *
   * @param {string} filePath
   * @param {import('./decibri').FileOptions} [options]
   * @returns {Promise<File>}
   */
  static async open(filePath, options = {}) {
    if (typeof filePath !== 'string') {
      throw new TypeError('path must be a string');
    }
    const prepared = File._prepareOptions(options);
    let native;
    try {
      native = await FileHandle.openAsync(filePath, prepared.nativeOptions);
    } catch (err) {
      throw wrapNativeError(err);
    }
    return new File(filePath, options, { prepared, native });
  }

  /**
   * Wrap in-memory samples as an offline source. `samples` must be a
   * `Float32Array` of mono samples in [-1.0, 1.0]; a raw `Buffer` of PCM
   * bytes is rejected as ambiguous (encoded bytes, int16 PCM, and f32
   * samples are indistinguishable, and decibri's own capture output is a
   * `Buffer`). Raw samples carry no header, so `inputRate` (their native
   * rate) is required; `sampleRate` stays the target output rate. No I/O,
   * so construction is synchronous.
   *
   * @param {Float32Array} samples
   * @param {import('./decibri').FileBufferOptions} [options]
   * @returns {File}
   */
  static buffer(samples, options = {}) {
    if (Buffer.isBuffer(samples)) {
      throw new TypeError(
        'File.buffer requires a Float32Array of samples, not a Buffer of bytes'
      );
    }
    if (!(samples instanceof Float32Array)) {
      throw new TypeError('File.buffer requires a Float32Array of samples');
    }
    const inputRate = options.inputRate;
    if (typeof inputRate !== 'number' || Number.isNaN(inputRate)) {
      throw new TypeError('inputRate is required for File.buffer (samples carry no header)');
    }
    if (inputRate < 1000 || inputRate > 384000) {
      throw new RangeError('inputRate must be between 1000 and 384000');
    }
    const prepared = File._prepareOptions(options);
    let native;
    try {
      native = FileHandle.buffer(samples, inputRate, prepared.nativeOptions);
    } catch (err) {
      throw wrapNativeError(err);
    }
    return new File(null, options, { prepared, native });
  }

  /** @internal */
  _read() {
    if (this._ended) {
      return;
    }
    let chunk;
    try {
      // One chunk per pull: the conditioning compute runs synchronously
      // here, so pulling one chunk at a time keeps the event loop breathing
      // between chunks while the stream machinery re-calls _read on demand.
      chunk = this._native.readChunk();
    } catch (err) {
      this.destroy(wrapNativeError(err));
      return;
    }
    if (chunk === null || chunk === undefined) {
      this._ended = true;
      this.push(null); // finite source: the stream ends at EOF
      return;
    }
    if (this._vad) {
      // Both modes read the score from native, computed on the signal
      // before the opt-in conditioning step, exactly as the live pump does.
      this._processVadValue(this._native.vadProbability, chunk.length);
    } else {
      this._position += chunk.length / this._bytesPerSample / this._sampleRate;
    }
    this.push(chunk);
  }

  /**
   * @internal Speech/silence state machine in FILE time. The same policy as
   * the Microphone's wall-clock machine, with the holdoff measured in
   * seconds of audio position instead of a timer: state flips only as the
   * file's own timeline passes the holdoff, so processing speed never
   * changes the reported events.
   */
  _processVadValue(value, chunkBytes) {
    const chunkStart = this._position;
    const chunkEnd = chunkStart + chunkBytes / this._bytesPerSample / this._sampleRate;
    this._position = chunkEnd;
    this._vadScore = value;
    if (value >= this._vadThreshold) {
      this._silenceStartPos = null;
      if (!this._isSpeaking) {
        this._isSpeaking = true;
        this.emit('speech');
      }
    } else if (this._isSpeaking) {
      if (this._silenceStartPos === null) {
        this._silenceStartPos = chunkStart;
      }
      if (chunkEnd - this._silenceStartPos >= this._vadHoldoffSeconds) {
        this._isSpeaking = false;
        this._silenceStartPos = null;
        this.emit('silence');
      }
    }
  }

  /**
   * Most recent per-chunk VAD score for the active mode: the Silero speech
   * probability in 'silero' mode, the normalized RMS of the pre-conditioning
   * signal in 'energy' mode. 0 when VAD is disabled or before the first
   * chunk. The same quantity the live `Microphone.vadScore` reports.
   * @returns {number}
   */
  get vadScore() {
    return this._vadScore;
  }

  /**
   * Analyze the whole recording for speech. Runs the recording once through
   * the conditioning pass off the event loop and resolves to a `VadReport`:
   * per-window `scores` (`{ start, end, vadScore, isSpeech }`) and merged
   * speech `segments` (`{ start, end }`), all in seconds of file time.
   * Consumes the source: analysis and iteration are separate single passes.
   *
   * Requires VAD: a `File` opened without `vad` rejects with the core's
   * "analysis requires VAD" error (a `RangeError`); the energy mode has no
   * whole-file analysis and rejects likewise. Never constructs a detector
   * silently.
   *
   * @returns {Promise<import('./decibri').VadReport>}
   */
  async analyze() {
    if (this._vad && this._vadMode === 'energy') {
      throw new RangeError(
        "analyze() requires vad: 'silero'; energy mode does not support whole-file analysis"
      );
    }
    try {
      return await this._native.analyze();
    } catch (err) {
      throw wrapNativeError(err);
    }
  }

  /**
   * The same whole-recording analysis under the international spelling.
   * @returns {Promise<import('./decibri').VadReport>}
   */
  analyse() {
    return this.analyze();
  }

  /**
   * Release the source. Idempotent; a closed File reads as ended.
   */
  close() {
    this._native.close();
  }
}

module.exports = {
  Microphone,
  Speaker,
  File,
  inputDevices,
  outputDevices,
  version,
  DecibriError,
  DeviceError,
  OrtError,
  OrtPathError,
};
