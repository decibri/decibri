'use strict';

const { Writable } = require('stream');
const { DecibriOutputBridge } = require('../index.js');
const { wrapNativeError } = require('./errors');

// ─── Speaker (Writable) ─────────────────────────────────────────────────────

class Speaker extends Writable {
  /**
   * @param {import('./decibri').SpeakerOptions} [options]
   */
  constructor(options = {}) {
    super({ highWaterMark: options.highWaterMark || 16384 });

    // ── Validate options ───────────────────────────────────────────────────

    const sampleRate = options.sampleRate ?? 16000;
    if (sampleRate < 1000 || sampleRate > 384000) {
      throw new RangeError('sample rate must be between 1000 and 384000');
    }

    const channels = options.channels ?? 1;
    if (channels < 1 || channels > 32) {
      throw new RangeError('channels must be between 1 and 32');
    }

    const dtype = options.dtype ?? 'int16';
    if (dtype !== 'int16' && dtype !== 'float32') {
      throw new TypeError("dtype must be 'int16' or 'float32'");
    }

    // ── Resolve device ──────────────────────────────────────────────────────

    // Name and multi-match resolution are delegated to the core, which owns
    // the renamed-vocabulary errors (SpeakerNotFound / MultipleDevicesMatch).
    // A string name and an { id } object are passed straight through to the
    // native addon. Only the numeric index keeps a client-side bounds check,
    // for a clean Node-side RangeError without a round-trip.
    let resolvedDevice = options.device;
    if (typeof options.device === 'number') {
      const devices = DecibriOutputBridge.devices();
      if (options.device < 0 || options.device >= devices.length) {
        throw new RangeError('device index out of range. Call Speaker.devices() to list available devices');
      }
      resolvedDevice = options.device;
    } else if (
      options.device !== null &&
      typeof options.device === 'object' &&
      !Array.isArray(options.device)
    ) {
      // Object form: { id: string } selects by stable per-host device ID.
      // See decibri.js for details; same pass-through semantics.
      if (typeof options.device.id !== 'string') {
        throw new TypeError('device.id must be a string');
      }
      resolvedDevice = options.device;
    }

    // ── Store config ───────────────────────────────────────────────────────

    this._dtype = dtype;
    this._started = false;

    // ── Create native bridge ───────────────────────────────────────────────

    try {
      this._native = new DecibriOutputBridge({
        sampleRate,
        channels,
        format: dtype,
        device: resolvedDevice,
      });
    } catch (err) {
      throw wrapNativeError(err);
    }
  }

  /** @internal */
  _write(chunk, encoding, callback) {
    if (chunk.length === 0) {
      callback();
      return;
    }
    try {
      this._native.write(chunk);
      callback();
    } catch (err) {
      callback(err);
    }
  }

  /** @internal Called by end() after all buffered writes complete. */
  _final(callback) {
    try {
      this._native.drain();
      callback();
    } catch (err) {
      callback(err);
    }
  }

  /**
   * Immediate stop. Discards remaining buffered audio.
   */
  stop() {
    this._native.stop();
  }

  /**
   * Whether audio is currently being output.
   * @returns {boolean}
   */
  get isPlaying() {
    return this._native.isPlaying;
  }

  /**
   * List all available output devices on the system.
   * @returns {Array<{index: number, name: string, maxOutputChannels: number, defaultSampleRate: number, isDefault: boolean}>}
   */
  static devices() {
    return DecibriOutputBridge.devices();
  }

  /**
   * Version information for decibri and the audio runtime.
   * @returns {{ decibri: string, portaudio: string }}
   */
  static version() {
    return DecibriOutputBridge.version();
  }
}

module.exports = Speaker;
