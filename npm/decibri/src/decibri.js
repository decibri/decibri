'use strict';

const { Readable } = require('stream');
const path = require('path');
const fs = require('fs');
const { DecibriBridge } = require('../index.js');

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

function computeRMS(chunk, format) {
  let sum = 0, n;
  if (format === 'float32') {
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

// ─── Decibri (Readable) ─────────────────────────────────────────────────────

class Decibri extends Readable {
  /**
   * @param {import('./decibri').DecibriOptions} [options]
   */
  constructor(options = {}) {
    super({ highWaterMark: options.highWaterMark, objectMode: false });

    // ── Validate options ───────────────────────────────────────────────────

    const sampleRate = options.sampleRate ?? 16000;
    if (sampleRate < 1000 || sampleRate > 384000) {
      throw new RangeError('sampleRate must be between 1000 and 384000');
    }

    const channels = options.channels ?? 1;
    if (channels < 1 || channels > 32) {
      throw new RangeError('channels must be between 1 and 32');
    }

    const framesPerBuffer = options.framesPerBuffer ?? 1600;
    if (framesPerBuffer < 64 || framesPerBuffer > 65536) {
      throw new RangeError('framesPerBuffer must be between 64 and 65536');
    }

    const format = options.format ?? 'int16';
    if (format !== 'int16' && format !== 'float32') {
      throw new TypeError("format must be 'int16' or 'float32'");
    }

    // ── Resolve device ──────────────────────────────────────────────────────

    let resolvedDevice = options.device;
    if (typeof options.device === 'string') {
      const lower = options.device.toLowerCase();
      const matches = DecibriBridge.devices().filter(d =>
        d.name.toLowerCase().includes(lower)
      );
      if (matches.length === 0) {
        throw new TypeError(`No audio input device found matching "${options.device}"`);
      }
      if (matches.length > 1) {
        const names = matches.map(d => `  [${d.index}] ${d.name}`).join('\n');
        throw new TypeError(
          `Multiple devices match "${options.device}":\n${names}\nUse a more specific name or pass the device index directly.`
        );
      }
      resolvedDevice = matches[0].index;
    } else if (typeof options.device === 'number') {
      const devices = DecibriBridge.devices();
      if (options.device < 0 || options.device >= devices.length) {
        throw new RangeError('device index out of range — call Decibri.devices() to list available devices');
      }
      resolvedDevice = options.device;
    }

    // ── Validate VAD options ─────────────────────────────────────────────────

    const vadMode = options.vadMode ?? 'energy';
    if (vadMode !== 'energy' && vadMode !== 'silero') {
      throw new TypeError("vadMode must be 'energy' or 'silero'");
    }

    let modelPath = undefined;
    let ortLibraryPath = undefined;
    if (vadMode === 'silero' && options.vad) {
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

    // ── Store config ───────────────────────────────────────────────────────

    this._format = format;
    this._vad = options.vad || false;
    this._vadMode = vadMode;
    this._vadThreshold = options.vadThreshold ?? (vadMode === 'silero' ? 0.5 : 0.01);
    this._vadHoldoff = options.vadHoldoff ?? 300;
    this._isSpeaking = false;
    this._silenceTimer = null;
    this._started = false;

    // ── Create native bridge ───────────────────────────────────────────────

    this._native = new DecibriBridge({
      sampleRate,
      channels,
      framesPerBuffer,
      format,
      device: resolvedDevice,
      vadMode: (options.vad && vadMode === 'silero') ? 'silero' : 'energy',
      modelPath,
      ortLibraryPath,
    });
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
    const rms = computeRMS(chunk, this._format);
    this._processVadValue(rms);
  }

  /** @internal Silero ML-based VAD (reads probability from native) */
  _processVadSilero(probability) {
    this._processVadValue(probability);
  }

  /** @internal Common speech/silence state machine */
  _processVadValue(value) {
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
   * List all available input devices on the system.
   * @returns {Array<{index: number, name: string, maxInputChannels: number, defaultSampleRate: number, isDefault: boolean}>}
   */
  static devices() {
    return DecibriBridge.devices();
  }

  /**
   * Version information for decibri and the audio runtime.
   * @returns {{ decibri: string, portaudio: string }}
   */
  static version() {
    return DecibriBridge.version();
  }
}

const DecibriOutput = require('./decibri-output.js');
Decibri.DecibriOutput = DecibriOutput;

module.exports = Decibri;
