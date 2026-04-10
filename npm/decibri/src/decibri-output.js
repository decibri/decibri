'use strict';

const { Writable } = require('stream');
const { DecibriOutputBridge } = require('../index.js');

// ─── DecibriOutput (Writable) ───────────────────────────────────────────────

class DecibriOutput extends Writable {
  /**
   * @param {import('./decibri').DecibriOutputOptions} [options]
   */
  constructor(options = {}) {
    super({ highWaterMark: options.highWaterMark || 16384 });

    // ── Validate options ───────────────────────────────────────────────────

    const sampleRate = options.sampleRate ?? 16000;
    if (sampleRate < 1000 || sampleRate > 384000) {
      throw new RangeError('sampleRate must be between 1000 and 384000');
    }

    const channels = options.channels ?? 1;
    if (channels < 1 || channels > 32) {
      throw new RangeError('channels must be between 1 and 32');
    }

    const format = options.format ?? 'int16';
    if (format !== 'int16' && format !== 'float32') {
      throw new TypeError("format must be 'int16' or 'float32'");
    }

    // ── Resolve device ──────────────────────────────────────────────────────

    let resolvedDevice = options.device;
    if (typeof options.device === 'string') {
      const lower = options.device.toLowerCase();
      const matches = DecibriOutputBridge.devices().filter(d =>
        d.name.toLowerCase().includes(lower)
      );
      if (matches.length === 0) {
        throw new TypeError(`No audio output device found matching "${options.device}"`);
      }
      if (matches.length > 1) {
        const names = matches.map(d => `  [${d.index}] ${d.name}`).join('\n');
        throw new TypeError(
          `Multiple devices match "${options.device}":\n${names}\nUse a more specific name or pass the device index directly.`
        );
      }
      resolvedDevice = matches[0].index;
    } else if (typeof options.device === 'number') {
      const devices = DecibriOutputBridge.devices();
      if (options.device < 0 || options.device >= devices.length) {
        throw new RangeError('device index out of range — call DecibriOutput.devices() to list available devices');
      }
      resolvedDevice = options.device;
    }

    // ── Store config ───────────────────────────────────────────────────────

    this._format = format;
    this._started = false;

    // ── Create native bridge ───────────────────────────────────────────────

    this._native = new DecibriOutputBridge({
      sampleRate,
      channels,
      format,
      device: resolvedDevice,
    });
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

  /** @internal — called by end() after all buffered writes complete */
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

module.exports = DecibriOutput;
