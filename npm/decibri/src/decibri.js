'use strict';

const { Readable, Writable } = require('stream');
const path = require('path');
const fs = require('fs');
const { DecibriBridge, FileHandle } = require('../index.js');
const {
  wrapNativeError,
  fileEngagedError,
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

/**
 * Failures that reach the consumer arrive on the `'error'` event as a
 * `DecibriError` carrying a `code`: a failed open or a denied permission when
 * capture starts, and a device lost mid-capture (`code === 'DEVICE_FAILED'`).
 * A deliberate `stop()` ends the stream cleanly and raises nothing.
 */
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
    this._dtype = prepared.dtype;
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

    // The number of channels delivered, interleaved frame by frame. Bounded
    // below here; bounded above by the resolved device alone, which answers
    // when the stream starts. No fixed maximum exists on this path.
    const channels = options.channels ?? 1;
    if (channels < 1) {
      throw new RangeError('channels must be at least 1');
    }

    // ── Validate channel map ─────────────────────────────────────────────────

    // An optional list of 0-based device channel indices, one per delivered
    // channel: delivered channel j carries device channel channelMap[j].
    // Absence delivers the documented average of every opened channel. The
    // checks here are shape-only (an array of integers that fit the channel
    // count's width, with one entry per channel); whether each entry exists on
    // the device is the core's check, made against the resolved device's own
    // report when the stream starts, because only the device can say how many
    // channels it has. No fixed maximum exists on this path.
    const channelMap = options.channelMap;
    if (channelMap !== undefined) {
      if (!Array.isArray(channelMap)) {
        throw new TypeError(
          `Invalid channelMap value: ${JSON.stringify(channelMap)}. Expected an array of 0-based device channel indices, such as [0].`
        );
      }
      for (const entry of channelMap) {
        if (typeof entry !== 'number' || !Number.isInteger(entry)) {
          throw new TypeError('channelMap entries must be integers');
        }
        if (entry < 0 || entry > 65535) {
          throw new RangeError('channelMap entries must be between 0 and 65535');
        }
      }
      if (channelMap.length !== channels) {
        throw new RangeError('channelMap must have exactly one entry per channel');
      }
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
      // Through the static so an enumeration failure raised while resolving an
      // index carries the same DeviceError the public listing does.
      const devices = Microphone.devices();
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
    // { model, threshold, holdoffMs, source } to tune the policy. The legacy two-flag
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
    let vadSource;
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
      // Config object form: { model, threshold?, holdoffMs?, source? }. model
      // is required and selects the detector; threshold and holdoffMs override
      // the mode defaults when supplied; source names the 0-based DELIVERED
      // channel the detector reads (the position within the delivered
      // interleaved frames, after any channelMap), absent feeding the frame
      // average of every delivered channel.
      const { model, threshold, holdoffMs, source } = vad;
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
      if (source !== undefined) {
        if (typeof source !== 'number' || !Number.isInteger(source)) {
          throw new TypeError('vad source must be an integer');
        }
        if (source < 0 || source > 65535) {
          throw new RangeError('vad source must be between 0 and 65535');
        }
        // The delivered count is the only ceiling, checked here where both
        // sides are in scope; the message is the core's own for the same
        // condition. No fixed maximum exists.
        if (source >= channels) {
          throw new RangeError(
            `the detector source names delivered channel ${source}; the delivered channel count is ${channels}`
          );
        }
        vadSource = source;
      }
    } else {
      throw new TypeError(
        `Invalid vad value: ${JSON.stringify(vad)}. Expected false, 'silero', 'energy', or a config object { model, threshold, holdoffMs, source }.`
      );
    }

    // The existence pre-check runs before the native bridge is constructed, so
    // it never passes through wrapNativeError. It raises the same class and
    // code the deep load failure would, so the detection point differs between
    // a missing file and an unloadable one but the identity does not.
    let modelPath = undefined;
    if (vadEnabled && vadMode === 'silero') {
      modelPath = options.modelPath || path.join(__dirname, '..', 'models', 'silero_vad.onnx');
      if (!fs.existsSync(modelPath)) {
        throw new OrtError(`Silero VAD model not found at ${modelPath}. Ensure the models/ directory is included in your installation.`, 'VAD_MODEL_LOAD_FAILED');
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
        throw new OrtError(`Denoise model not found at ${denoiseModelPath}. Ensure the models/ directory is included in your installation.`, 'MODEL_LOAD_FAILED');
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

    // ── Validate AEC ─────────────────────────────────────────────────────────

    // Echo cancellation: the 'tau' shorthand names the model, or an
    // { model, tailMs, suppression, referenceSampleRate, referenceChannels }
    // object tunes it; absence leaves it off. The model name is deliberately NOT checked
    // against a list here: the canceller owns the accepted set, so the native
    // layer parses it (AecModel::from_str) and an unknown name is rejected by
    // the native constructor with the canceller's own message (a DecibriError
    // with code 'AEC_CONFIG_INVALID'). The three tuning fields are checked
    // here with the same RangeError / TypeError classes the other
    // conditioning options use; the native layer backstops the same checks.
    // The capture-rate window (8000..=48000 with AEC on) is guarded by the
    // core, surfacing as a RangeError from the native constructor.
    const aec = options.aec;
    let aecModel;
    let aecTailMs;
    let aecSuppression;
    let aecReferenceSampleRate;
    let aecReferenceChannels;
    if (aec !== undefined) {
      if (typeof aec === 'string') {
        aecModel = aec;
      } else if (aec !== null && typeof aec === 'object' && !Array.isArray(aec)) {
        const { model, tailMs, suppression, referenceSampleRate, referenceChannels } = aec;
        if (typeof model !== 'string') {
          throw new TypeError(
            `Invalid aec model: ${JSON.stringify(model)}. Expected a model name string such as 'tau'.`
          );
        }
        aecModel = model;
        if (tailMs !== undefined) {
          if (typeof tailMs !== 'number' || Number.isNaN(tailMs)) {
            throw new TypeError('aec tailMs must be a number');
          }
          if (tailMs < 16 || tailMs > 500) {
            throw new RangeError('aec tailMs must be between 16 and 500');
          }
          aecTailMs = tailMs;
        }
        if (suppression !== undefined) {
          if (suppression !== 'conservative' && suppression !== 'off') {
            throw new TypeError(
              `aec suppression must be 'conservative' or 'off'; got ${JSON.stringify(suppression)}`
            );
          }
          aecSuppression = suppression;
        }
        if (referenceSampleRate !== undefined) {
          if (typeof referenceSampleRate !== 'number' || Number.isNaN(referenceSampleRate)) {
            throw new TypeError('aec referenceSampleRate must be a number');
          }
          if (referenceSampleRate < 1000 || referenceSampleRate > 384000) {
            throw new RangeError('aec referenceSampleRate must be between 1000 and 384000');
          }
          aecReferenceSampleRate = referenceSampleRate;
        }
        if (referenceChannels !== undefined) {
          if (typeof referenceChannels !== 'number' || Number.isNaN(referenceChannels)) {
            throw new TypeError('aec referenceChannels must be a number');
          }
          if (referenceChannels < 1) {
            throw new RangeError('aec referenceChannels must be at least 1');
          }
          aecReferenceChannels = referenceChannels;
        }
      } else {
        throw new TypeError(
          `Invalid aec value: ${JSON.stringify(aec)}. Expected a model name such as 'tau', or a config object { model, tailMs, suppression, referenceSampleRate, referenceChannels }.`
        );
      }
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
        channelMap,
        framesPerBuffer,
        format: dtype,
        device: resolvedDevice,
        // Pass the mode to native only when VAD is enabled. When disabled the
        // local vadMode is an inert 'energy' placeholder; sending it would make
        // native compute the energy score for a microphone that did not ask for
        // VAD. Absent means VAD off in native.
        vadMode: vadEnabled ? vadMode : undefined,
        // The delivered channel the detector reads, from the vad config
        // object's source key. Absent feeds the frame average.
        detectorSource: vadSource,
        modelPath,
        dcRemoval,
        denoise,
        denoiseModelPath,
        ortLibraryPath,
        highpass,
        agc,
        limiter,
        aec: aecModel,
        aecTailMs,
        aecSuppression,
        aecReferenceSampleRate,
        aecReferenceChannels,
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

    // Both exits from start() reach the consumer on the 'error' event, so both
    // carry a decibri class and `code`: the synchronous throw below (a failed
    // device open, a denied permission) and the streaming failure delivered to
    // the callback (a device lost mid-capture).
    try {
      this._native.start((err, chunk) => {
        if (err) {
          this._started = false;
          this.destroy(wrapNativeError(err));
          return;
        }

        // On close the native pump flushes the buffered tail; that final data
        // callback can run after stop() has begun ending the stream. Once the
        // end-of-stream push(null) has been issued (or the stream was
        // destroyed), drop any straggler rather than pushing past EOF.
        if (this._ended || this.destroyed) {
          return;
        }

        // push returns false when the consumer is slow. We can't pause a mic,
        // but we surface the backpressure warning so callers can react.
        if (!this.push(chunk)) {
          this.emit('backpressure');
        }

        if (this._vad) {
          // Both modes read the score from native: the Silero speech
          // probability or, in energy mode, the RMS of the pre-enhancement
          // signal. The native pump computes both on the signal before the
          // opt-in enhancement step, so enabling enhancement does not change
          // detection in either mode.
          this._processVadValue(this._native.vadProbability);
        }
      });
    } catch (err) {
      this._started = false;
      this.destroy(wrapNativeError(err));
    }
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
   * Queue far-end reference audio for the echo canceller: the audio being
   * played out, pushed as it is played, in played order. Accepts a `Buffer`,
   * `Uint8Array`, or `DataView` of PCM bytes in this microphone's `dtype`,
   * or the typed array carrying that dtype (`Int16Array` for `'int16'`,
   * `Float32Array` for `'float32'`), at the declared `referenceSampleRate`
   * (the capture rate when unset), interleaved at the declared
   * `referenceChannels` (mono when unset). A typed array carrying any other
   * sample dtype throws a `TypeError`, whatever the capture state. With
   * `referenceChannels` above 1, each frame is averaged to one mono sample
   * before the canceller sees it; a multichannel reference pushed without
   * declaring the count cancels nothing and reports no error. The declared
   * count must match this buffer's actual interleaving: a mismatch is not
   * detected and raises no error, and shows up only as
   * `aecMetrics().delaySamples` staying `null` with no fault reported.
   *
   * Never blocks and never throws on a full queue: samples that do not fit
   * are discarded and counted by `aecMetrics().referenceDropped`, and the
   * span they occupied is represented as silence. Silence between played
   * audio need not be pushed; a caller that stops pushing has said nothing is
   * playing. A push while capture is not running is discarded and counted by
   * `referenceDropped`, read once capture runs; a push with the `aec` option
   * unset is a no-op.
   *
   * @param {Buffer | NodeJS.ArrayBufferView} data PCM samples in the
   *   configured `dtype`.
   */
  pushAecReference(data) {
    let buf;
    if (Buffer.isBuffer(data)) {
      buf = data;
    } else if (ArrayBuffer.isView(data)) {
      // A typed array names its own sample dtype, so one carrying a dtype
      // other than the configured one is refused rather than read as raw
      // bytes. Buffer, Uint8Array, and DataView are format-agnostic byte
      // carriers, exactly as bytes are on the Python surface; the accepted
      // view is normalized to the same bytes, no copy, exactly as the stream
      // machinery normalizes a typed-array write to a Speaker.
      if (!(data instanceof Uint8Array) && !(data instanceof DataView)) {
        const expected = this._dtype === 'int16' ? Int16Array : Float32Array;
        if (!(data instanceof expected)) {
          const mismatched = this._dtype === 'int16' ? Float32Array : Int16Array;
          if (data instanceof mismatched) {
            const other = this._dtype === 'int16' ? 'float32' : 'int16';
            throw new TypeError(
              `dtype '${this._dtype}' configured but ${mismatched.name} samples were pushed; ` +
                `convert to ${expected.name} or construct Microphone with dtype: '${other}'`
            );
          }
          throw new TypeError(
            'pushAecReference requires a Buffer, TypedArray, or DataView of PCM samples in the configured dtype'
          );
        }
      }
      buf = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    } else {
      throw new TypeError(
        'pushAecReference requires a Buffer, TypedArray, or DataView of PCM samples in the configured dtype'
      );
    }
    this._native.pushAecReference(buf);
  }

  /**
   * The echo canceller's transport and cancellation metrics, merged with the
   * reference queue's counters, or `null` when the `aec` option is unset or
   * capture is not running.
   *
   * `delaySamples` staying `null` while `acquisitionParked` climbs is the
   * signature of a canceller with no usable reference: none pushed, not at
   * the declared rate, or not the signal that produced the echo. A climbing
   * `referenceDropped` means single pushes are exceeding the reference
   * queue's bound.
   *
   * @returns {import('./decibri').AecMetrics | null}
   */
  aecMetrics() {
    const m = this._native.aecMetrics();
    if (m === null || m === undefined) return null;
    return {
      delaySamples: m.delaySamples ?? null,
      erleDb: m.erleDb,
      doubleTalk: m.doubleTalk,
      referenceStarved: m.referenceStarved,
      acquisitionParked: m.acquisitionParked,
      referenceReanchors: m.referenceReanchors,
      referenceDropped: m.referenceDropped,
      referenceSilence: m.referenceSilence,
    };
  }

  /**
   * List all available input devices on the system.
   * @returns {Array<{index: number, name: string, id: string, maxInputChannels: number, defaultSampleRate: number, isDefault: boolean}>}
   */
  static devices() {
    try {
      return DecibriBridge.devices();
    } catch (err) {
      throw wrapNativeError(err);
    }
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
   * Open an audio file as an offline source, synchronously. Everything a
   * `Microphone` does to live audio, a `File` does to audio you already
   * have: the same conditioning options, the same stream of conditioned
   * chunks, and (with `vad` set) the same per-chunk speech events, plus the
   * whole-file `analyze()` a live stream cannot offer.
   *
   * The bare constructor reads the file inline, blocking the event loop on
   * disk I/O; prefer `await File.open(path, options)` in servers and other
   * latency-sensitive code, exactly as `Microphone.open` is preferred over
   * `new Microphone`. Iteration and analysis are separate single passes:
   * each consumes the source once, so use one `File` per operation.
   *
   * @param {string} filePath Path to a WAV, AIFF, AIFF-C or FLAC file. The
   *   container is identified from the bytes, not from the extension.
   * @param {import('./decibri').FileOptions} [options]
   * @param {{ prepared: object, native: object }} [_internal] Internal: a
   *   pre-resolved options bundle and an already-constructed native handle,
   *   passed by the async `File.open()` factory and by `File.buffer()`. Not
   *   part of the public API.
   */
  constructor(filePath, options = {}, _internal = undefined) {
    super({ highWaterMark: options.highWaterMark, objectMode: false });

    // The open path reads the source's channel count from the file's own
    // header, so a caller-stated interleave has nothing to describe here;
    // it is refused rather than ignored.
    if (!_internal && options.inputChannels !== undefined) {
      throw new TypeError(
        'inputChannels applies only to File.buffer; a file carries its channel count in its own header'
      );
    }

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
    // The delivered channel count: file time advances by frames, so the
    // interleaved sample count divides by it before it divides by the rate.
    this._channels = prepared.channels;
    this._bytesPerSample = prepared.dtype === 'int16' ? 2 : 4;
    this._ended = false;
    // Set the moment the consumer asks the stream for data, which is earlier
    // than the moment data arrives: `resume()` and a 'readable' listener both
    // schedule the first `_read()` for a later tick. See _engage.
    this._engaged = false;

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

    // The number of channels delivered, interleaved frame by frame. Bounded
    // below here; bounded above by the source's own channel count alone,
    // which the core reads from the header (or takes from inputChannels)
    // when the source is opened. No fixed maximum exists on this path.
    const channels = options.channels ?? 1;
    if (channels < 1) {
      throw new RangeError('channels must be at least 1');
    }

    // An optional list of 0-based source channel indices, one per delivered
    // channel: delivered channel j carries source channel channelMap[j].
    // Absence delivers the documented average of every source channel. The
    // checks here are shape-only, exactly as the Microphone's: whether each
    // entry exists on the source is the core's check, made against the
    // source's own count, because only the opened source can say how many
    // channels it has. No fixed maximum exists on this path.
    const channelMap = options.channelMap;
    if (channelMap !== undefined) {
      if (!Array.isArray(channelMap)) {
        throw new TypeError(
          `Invalid channelMap value: ${JSON.stringify(channelMap)}. Expected an array of 0-based source channel indices, such as [0].`
        );
      }
      for (const entry of channelMap) {
        if (typeof entry !== 'number' || !Number.isInteger(entry)) {
          throw new TypeError('channelMap entries must be integers');
        }
        if (entry < 0 || entry > 65535) {
          throw new RangeError('channelMap entries must be between 0 and 65535');
        }
      }
      if (channelMap.length !== channels) {
        throw new RangeError('channelMap must have exactly one entry per channel');
      }
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
    let vadSource;
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
      const { model, threshold, holdoffMs, source } = vad;
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
      // source names the 0-based DELIVERED channel the detector reads,
      // exactly as on Microphone; the checks and messages are the same.
      if (source !== undefined) {
        if (typeof source !== 'number' || !Number.isInteger(source)) {
          throw new TypeError('vad source must be an integer');
        }
        if (source < 0 || source > 65535) {
          throw new RangeError('vad source must be between 0 and 65535');
        }
        if (source >= channels) {
          throw new RangeError(
            `the detector source names delivered channel ${source}; the delivered channel count is ${channels}`
          );
        }
        vadSource = source;
      }
    } else {
      throw new TypeError(
        `Invalid vad value: ${JSON.stringify(vad)}. Expected false, 'silero', 'energy', or a config object { model, threshold, holdoffMs, source }.`
      );
    }

    // Same pre-check and same identity as Microphone: a missing model file
    // reports the class and code its load failure would.
    let modelPath = undefined;
    if (vadEnabled && vadMode === 'silero') {
      modelPath = options.modelPath || path.join(__dirname, '..', 'models', 'silero_vad.onnx');
      if (!fs.existsSync(modelPath)) {
        throw new OrtError(`Silero VAD model not found at ${modelPath}. Ensure the models/ directory is included in your installation.`, 'VAD_MODEL_LOAD_FAILED');
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
        throw new OrtError(`Denoise model not found at ${denoiseModelPath}. Ensure the models/ directory is included in your installation.`, 'MODEL_LOAD_FAILED');
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
      channels,
      vadEnabled,
      vadMode,
      vadThreshold: vadThreshold ?? (vadMode === 'silero' ? 0.5 : 0.01),
      vadHoldoff: vadHoldoff ?? 300,
      nativeOptions: {
        sampleRate,
        channels,
        channelMap,
        format: dtype,
        // Pass the mode to native only when VAD is enabled, exactly as the
        // Microphone options do; absent means VAD off in native.
        vadMode: vadEnabled ? vadMode : undefined,
        // The delivered channel the detector reads, from the vad config
        // object's source key. Absent feeds the frame average.
        detectorSource: vadSource,
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
   * Open an audio file without blocking the event loop: the disk read,
   * decode, and chain construction run on the native thread pool. The
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
    // The same refusal the synchronous constructor makes: a path's channel
    // count comes from its own header.
    if (options.inputChannels !== undefined) {
      throw new TypeError(
        'inputChannels applies only to File.buffer; a file carries its channel count in its own header'
      );
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
   * `Float32Array` of samples in [-1.0, 1.0], frame-interleaved at
   * `inputChannels` (1, mono, by default); a raw `Buffer` of PCM bytes is
   * rejected as ambiguous (encoded bytes, int16 PCM, and f32 samples are
   * indistinguishable, and decibri's own capture output is a `Buffer`). Raw
   * samples carry no header, so `inputRate` (their native rate) is
   * required; `sampleRate` stays the target output rate. No I/O, so
   * construction is synchronous.
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
    // The channel counterpart of inputRate: the interleave of the caller's
    // own samples. Shape-checked here; whether the samples divide into
    // whole frames at this count is the core's check.
    const inputChannels = options.inputChannels ?? 1;
    if (typeof inputChannels !== 'number' || !Number.isInteger(inputChannels)) {
      throw new TypeError('inputChannels must be an integer');
    }
    if (inputChannels < 1 || inputChannels > 65535) {
      throw new RangeError('inputChannels must be between 1 and 65535');
    }
    const prepared = File._prepareOptions(options);
    let native;
    try {
      native = FileHandle.buffer(samples, inputRate, {
        ...prepared.nativeOptions,
        inputChannels,
      });
    } catch (err) {
      throw wrapNativeError(err);
    }
    return new File(null, options, { prepared, native });
  }

  /**
   * @internal Mark the stream as engaged: the consumer has asked it for data,
   * so `analyze()` is refused from here on. Every entry point that starts the
   * flow marks synchronously with the consumer's own call; the read it
   * schedules runs on a later tick.
   */
  _engage() {
    this._engaged = true;
  }

  /**
   * Start the flow of data. Also engages the stream: `analyze()` is refused
   * once the recording is being streamed.
   * @returns {this}
   */
  resume() {
    this._engage();
    return super.resume();
  }

  /**
   * Pull from the stream's buffer. Also engages the stream: `analyze()` is
   * refused once the recording is being read.
   * @param {number} [size]
   * @returns {Buffer|null}
   */
  read(size) {
    this._engage();
    return super.read(size);
  }

  /**
   * Attach a listener. A 'data' or 'readable' listener starts the flow, so it
   * engages the stream; listeners for the VAD, error, and end events do not.
   * @param {string|symbol} event
   * @param {Function} listener
   * @returns {this}
   */
  on(event, listener) {
    if (event === 'data' || event === 'readable') {
      this._engage();
    }
    return super.on(event, listener);
  }

  /**
   * Alias of `on`, kept in step so it engages the stream the same way.
   * @param {string|symbol} event
   * @param {Function} listener
   * @returns {this}
   */
  addListener(event, listener) {
    return this.on(event, listener);
  }

  /** @internal */
  _read() {
    this._engage();
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
      // Bytes to interleaved samples to frames to seconds of file time.
      this._position +=
        chunk.length / this._bytesPerSample / this._channels / this._sampleRate;
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
    // Bytes to interleaved samples to frames to seconds of file time.
    const chunkEnd =
      chunkStart + chunkBytes / this._bytesPerSample / this._channels / this._sampleRate;
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
   * The rate every delivered chunk carries: the `sampleRate` option, or 16000
   * when it was not given. Readable for the life of the File, including after
   * the source is consumed or closed.
   * @returns {number}
   */
  get sampleRate() {
    return this._native.sampleRate;
  }

  /**
   * The source's own rate, taken from the file's header or from the
   * `inputRate` passed to `File.buffer`. Differs from `sampleRate` when the
   * source was resampled. Readable for the life of the File, including after
   * the source is consumed or closed.
   * @returns {number}
   */
  get inputRate() {
    return this._native.inputRate;
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
   * Requires a `File` that is not already being streamed: once the stream has
   * been engaged, this rejects with a `DecibriError` carrying the code
   * `'FILE_ENGAGED'` rather than reporting on the part not yet read. Every
   * failure detected before the pass begins leaves the `File` usable, so
   * streaming it afterwards still works; a failure during the pass, such as
   * the detector failing to load, consumes the source.
   *
   * @returns {Promise<import('./decibri').VadReport>}
   */
  async analyze() {
    // Checked before the native handle is touched: the rejection belongs to
    // this call, and the source is never taken from a File that keeps
    // streaming.
    if (this._engaged) {
      throw fileEngagedError();
    }
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
   * Validate the save options and resolve them into the native options
   * object. Shared by `File.save` and `AudioWriter`, so the two spellings of
   * a write accept and reject identically.
   * @internal
   * @param {import('./decibri').SaveOptions} options
   */
  static _prepareSaveOptions(options) {
    const format = options.format;
    if (format !== undefined && format !== 'wav' && format !== 'aiff' && format !== 'flac') {
      throw new TypeError(
        `Invalid format value: ${JSON.stringify(format)}. Expected 'wav', 'aiff', or 'flac'.`
      );
    }
    const compression = options.compression;
    if (compression !== undefined) {
      if (typeof compression !== 'number' || Number.isNaN(compression)) {
        throw new TypeError('compression must be a number');
      }
      if (compression < 0 || compression > 8) {
        throw new RangeError('flac compression level must be between 0 and 8');
      }
    }
    return { format, compression };
  }

  /**
   * Write the conditioned recording to disk, off the event loop. Runs the
   * recording once through the same conditioning pass iteration delivers,
   * whole, and writes it as 16-bit PCM mono at `sampleRate`. Consumes the
   * source: a save is a single pass, separate from iteration and analysis.
   *
   * The container comes from the path's extension (`.wav`, `.aiff`, `.aif`,
   * `.aifc` or `.flac`), or from `options.format`: decibri reads a file by
   * its content and writes one by its name. An extension it does not
   * recognise rejects rather than defaulting. `options.compression` sets the
   * FLAC compression level (0-8, default 5); it applies only to FLAC.
   *
   * Resolves to a `SaveReport`: `clippedSamples` counts finite samples
   * outside full scale clamped to `[-1.0, 1.0]` (AGC or AEC without a
   * limiter can overshoot, and 16-bit PCM cannot hold it), and
   * `nonFiniteSamples` counts NaN samples written as silence and infinite
   * samples written as full scale.
   *
   * Requires a `File` that is not already being streamed: once the stream
   * has been engaged this rejects with a `DecibriError` carrying the code
   * `'FILE_ENGAGED'`. Every failure detected before the pass begins leaves
   * the `File` usable; a failure during the pass consumes the source.
   *
   * @param {string} filePath
   * @param {import('./decibri').SaveOptions} [options]
   * @returns {Promise<import('./decibri').SaveReport>}
   */
  async save(filePath, options = {}) {
    // Checked before the native handle is touched: the rejection belongs to
    // this call, and the source is never taken from a File that keeps
    // streaming.
    if (this._engaged) {
      throw fileEngagedError();
    }
    if (typeof filePath !== 'string') {
      throw new TypeError('path must be a string');
    }
    const nativeOptions = File._prepareSaveOptions(options);
    try {
      return await this._native.save(filePath, nativeOptions);
    } catch (err) {
      throw wrapNativeError(err);
    }
  }

  /**
   * Release the source. Idempotent; a closed File reads as ended.
   */
  close() {
    this._native.close();
  }
}

// ─── AudioWriter (Writable): file sink ───────────────────────────────────────

class AudioWriter extends Writable {
  /**
   * A file sink for PCM audio: the Writable to pair with decibri's Readable
   * sources, and with any other stream of PCM bytes (a TTS engine, a decoded
   * network stream). Collects the whole stream, then writes it as one audio
   * file when the stream finishes, exactly as `File.save` writes: the same
   * containers from the same extension rule, the same 16-bit PCM encoding,
   * the same clamp and non-finite handling, the same bytes.
   *
   * Chunks are raw PCM bytes in `dtype` ('int16' little-endian by default,
   * matching what a `File` or `Microphone` emits; 'float32' for raw f32
   * bytes), frame-interleaved at `channels` (1, mono, by default; the
   * stream's total sample count must divide into whole frames, and each
   * container's own channel ceiling applies at the write). `sampleRate` is
   * required, because raw audio carries no header to read one from.
   *
   * The file is written when the stream finishes ('finish' fires after the
   * file is on disk), and `report` then carries the `SaveReport` the write
   * produced. A failure destroys the stream with the error.
   *
   * @param {string} filePath Path whose extension names the container,
   *   unless `format` overrides it.
   * @param {import('./decibri').AudioWriterOptions} options
   */
  constructor(filePath, options = {}) {
    super({ highWaterMark: options.highWaterMark });
    if (typeof filePath !== 'string') {
      throw new TypeError('path must be a string');
    }
    const sampleRate = options.sampleRate;
    if (typeof sampleRate !== 'number' || Number.isNaN(sampleRate)) {
      throw new TypeError('sampleRate is required for AudioWriter (raw audio carries no header)');
    }
    if (sampleRate < 1000 || sampleRate > 384000) {
      throw new RangeError('sample rate must be between 1000 and 384000');
    }
    // Bounded below here; above, each container's own ceiling answers at
    // the write, with the container layer's own message. No decibri-side
    // maximum exists on this path.
    const channels = options.channels ?? 1;
    if (channels < 1) {
      throw new RangeError('channels must be at least 1');
    }
    const dtype = options.dtype ?? 'int16';
    if (dtype !== 'int16' && dtype !== 'float32') {
      throw new TypeError("dtype must be 'int16' or 'float32'");
    }
    // Validated now, so a bad option is a construction-time throw rather
    // than a deferred 'error' event after the audio has been streamed.
    this._saveOptions = File._prepareSaveOptions(options);
    this._filePath = filePath;
    this._sampleRate = sampleRate;
    this._channels = channels;
    this._dtype = dtype;
    this._chunks = [];
    this._report = null;
  }

  /**
   * The `SaveReport` of the completed write: `clippedSamples` and
   * `nonFiniteSamples`, exactly as `File.save` resolves. `null` until
   * 'finish' has fired.
   * @returns {import('./decibri').SaveReport | null}
   */
  get report() {
    return this._report;
  }

  /** @internal */
  _write(chunk, encoding, callback) {
    this._chunks.push(chunk);
    callback();
  }

  /** @internal */
  _final(callback) {
    const bytes = Buffer.concat(this._chunks);
    this._chunks = [];
    const bytesPerSample = this._dtype === 'int16' ? 2 : 4;
    if (bytes.length % bytesPerSample !== 0) {
      callback(
        new RangeError(
          `audio bytes do not divide into whole ${this._dtype} samples; ` +
            `${bytes.length % bytesPerSample} byte(s) over`
        )
      );
      return;
    }
    const samples = new Float32Array(bytes.length / bytesPerSample);
    if (this._dtype === 'int16') {
      // The inverse of the int16 delivery encoding: value / 32768, so bytes
      // that came from a decibri source re-quantise to the identical file.
      for (let i = 0; i < samples.length; i++) {
        samples[i] = bytes.readInt16LE(i * 2) / 32768;
      }
    } else {
      for (let i = 0; i < samples.length; i++) {
        samples[i] = bytes.readFloatLE(i * 4);
      }
    }
    // The write is File.save on a source at the writer's own rate and
    // interleave: the same encode path, so the two spellings produce the
    // same bytes. A stream that does not divide into whole frames is the
    // core's refusal, surfaced here when the stream finishes.
    let file;
    try {
      file = File.buffer(samples, {
        inputRate: this._sampleRate,
        inputChannels: this._channels,
        sampleRate: this._sampleRate,
        channels: this._channels,
      });
    } catch (err) {
      callback(err);
      return;
    }
    file
      .save(this._filePath, this._saveOptions)
      .then((report) => {
        this._report = report;
        callback();
      })
      .catch((err) => {
        callback(err);
      });
  }
}

module.exports = {
  Microphone,
  Speaker,
  File,
  AudioWriter,
  inputDevices,
  outputDevices,
  version,
  DecibriError,
  DeviceError,
  OrtError,
  OrtPathError,
};
