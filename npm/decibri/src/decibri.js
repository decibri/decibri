'use strict';

const { Readable } = require('stream');
const path = require('path');
const fs = require('fs');
const { DecibriBridge } = require('../index.js');
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

// ─── RMS helper ──────────────────────────────────────────────────────────────

function computeRMS(chunk, dtype) {
  let sum = 0, n;
  if (dtype === 'float32') {
    const samples = new Float32Array(chunk.buffer, chunk.byteOffset, chunk.length / 4);
    n = samples.length;
    for (let i = 0; i < n; i++) sum += samples[i] * samples[i];
  } else {
    const samples = new Int16Array(chunk.buffer, chunk.byteOffset, chunk.length / 2);
    n = samples.length;
    for (let i = 0; i < n; i++) {
      const s = samples[i] / 32768;
      sum += s * s;
    }
  }
  return n > 0 ? Math.sqrt(sum / n) : 0;
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

    this._dtype = prepared.dtype;
    this._vad = prepared.vadEnabled;
    this._vadMode = prepared.vadMode;
    this._vadThreshold = prepared.vadThreshold;
    this._vadHoldoff = prepared.vadHoldoff;
    this._vadScore = 0;
    this._isSpeaking = false;
    this._silenceTimer = null;
    this._started = false;

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

    const channels = options.channels ?? 1;
    if (channels < 1 || channels > 32) {
      throw new RangeError('channels must be between 1 and 32');
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

    // Single vad union: false (disabled, default), 'silero', or 'energy'. The
    // legacy two-flag form (vad: true plus vadMode) is rejected with a
    // migration error. Energy and Silero are both computed in this wrapper;
    // the union only selects which.
    const vad = options.vad ?? false;
    let vadEnabled;
    let vadMode;
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
    } else {
      throw new TypeError(
        `Invalid vad value: ${JSON.stringify(vad)}. Expected false, 'silero', or 'energy'.`
      );
    }

    let modelPath = undefined;
    let ortLibraryPath = undefined;
    if (vadEnabled && vadMode === 'silero') {
      modelPath = options.modelPath || path.join(__dirname, '..', 'models', 'silero_vad.onnx');
      if (!fs.existsSync(modelPath)) {
        throw new Error(`Silero VAD model not found at ${modelPath}. Ensure the models/ directory is included in your installation.`);
      }
      // Internal plumbing: inject the bundled ORT dylib path into the napi
      // constructor. If resolution fails (unknown platform, platform package
      // not installed), leaves ortLibraryPath as undefined and lets Rust fall
      // through to ORT_DYLIB_PATH or surface a decibri-specific init error.
      ortLibraryPath = resolveBundledOrtPath();
    }

    return {
      dtype,
      vadEnabled,
      vadMode,
      vadThreshold: options.vadThreshold ?? (vadMode === 'silero' ? 0.5 : 0.01),
      vadHoldoff: options.vadHoldoff ?? 300,
      nativeOptions: {
        sampleRate,
        channels,
        framesPerBuffer,
        format: dtype,
        device: resolvedDevice,
        vadMode,
        modelPath,
        ortLibraryPath,
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
    if (this._started) return;
    this._started = true;

    this._native.start((err, chunk) => {
      if (err) {
        this._started = false;
        this.destroy(err);
        return;
      }

      // push returns false when the consumer is slow. We can't pause a mic,
      // but we surface the backpressure warning so callers can react.
      if (!this.push(chunk)) {
        this.emit('backpressure');
      }

      if (this._vad) {
        if (this._vadMode === 'silero') {
          const prob = this._native.vadProbability;
          this._processVadSilero(prob);
        } else {
          this._processVadEnergy(chunk);
        }
      }
    });
  }

  /** @internal Energy-based VAD (RMS threshold) */
  _processVadEnergy(chunk) {
    const rms = computeRMS(chunk, this._dtype);
    this._processVadValue(rms);
  }

  /** @internal Silero ML-based VAD (reads probability from native) */
  _processVadSilero(probability) {
    this._processVadValue(probability);
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
    this._native.stop();
    clearTimeout(this._silenceTimer);
    this._silenceTimer = null;
    this.push(null); // signals stream end
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
   * in 'silero' mode, the normalized RMS of the last chunk in 'energy' mode.
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

module.exports = {
  Microphone,
  Speaker,
  inputDevices,
  outputDevices,
  version,
  DecibriError,
  DeviceError,
  OrtError,
  OrtPathError,
};
