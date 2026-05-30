'use strict';

const { Writable } = require('stream');
const { DecibriOutputBridge } = require('../index.js');
const { wrapNativeError } = require('./errors');

// The npm package version, reported as `binding` by version(). Read from
// package.json so it tracks the published package and cannot drift.
const PACKAGE_VERSION = require('../package.json').version;

// ─── Speaker (Writable) ─────────────────────────────────────────────────────

class Speaker extends Writable {
  /**
   * @param {import('./decibri').SpeakerOptions} [options]
   * @param {{ prepared: object, native: object }} [_internal] Internal: a
   *   pre-resolved options bundle and an already-constructed native bridge,
   *   passed by the async `Speaker.open()` factory. Not part of the public API.
   */
  constructor(options = {}, _internal = undefined) {
    super({ highWaterMark: options.highWaterMark || 16384 });

    // Validate and resolve options once. The async factory passes its already
    // resolved bundle through `_internal` to avoid recomputing it.
    const prepared = _internal ? _internal.prepared : Speaker._prepareOptions(options);

    // ── Store config ───────────────────────────────────────────────────────

    this._dtype = prepared.dtype;
    this._started = false;

    // ── Create or adopt native bridge ───────────────────────────────────────

    if (_internal) {
      // Built off the event loop by Speaker.open(); already wrapped.
      this._native = _internal.native;
    } else {
      try {
        this._native = new DecibriOutputBridge(prepared.nativeOptions);
      } catch (err) {
        throw wrapNativeError(err);
      }
    }
  }

  /**
   * Validate the constructor options and resolve them into the native options
   * object plus the wrapper-side state. Throws the same `RangeError` /
   * `TypeError` as the constructor on invalid input. Shared by the synchronous
   * constructor and the async `open()` factory.
   * @internal
   * @param {import('./decibri').SpeakerOptions} options
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

    return {
      dtype,
      nativeOptions: {
        sampleRate,
        channels,
        format: dtype,
        device: resolvedDevice,
      },
    };
  }

  /**
   * Construct a Speaker without blocking the event loop.
   *
   * Symmetric with `Microphone.open()` and the Python `AsyncSpeaker.open()`.
   * The speaker loads no model, so the only open work is device resolution and
   * the practical blocking risk is small; this factory exists chiefly so async
   * callers can use one consistent construction pattern across both classes.
   * The synchronous constructor remains available and unchanged.
   *
   * A failed open (unknown device) rejects the returned Promise with the
   * matching error class rather than throwing synchronously.
   *
   * @param {import('./decibri').SpeakerOptions} [options]
   * @returns {Promise<Speaker>}
   */
  static async open(options = {}) {
    const prepared = Speaker._prepareOptions(options);
    let native;
    try {
      native = await DecibriOutputBridge.openAsync(prepared.nativeOptions);
    } catch (err) {
      throw wrapNativeError(err);
    }
    return new Speaker(options, { prepared, native });
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
   * Version information for decibri, the audio backend, and this binding.
   * @returns {{ decibri: string, audioBackend: string, binding: string }}
   */
  static version() {
    const v = DecibriOutputBridge.version();
    return { decibri: v.decibri, audioBackend: v.audioBackend, binding: PACKAGE_VERSION };
  }
}

module.exports = Speaker;
