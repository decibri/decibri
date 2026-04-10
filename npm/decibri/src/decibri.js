'use strict';

const { Readable } = require('stream');
const { DecibriBridge } = require('../index.js');

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

    // ── Store config ───────────────────────────────────────────────────────

    this._format = format;
    this._vad = options.vad || false;
    this._vadThreshold = options.vadThreshold ?? 0.01;
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
      device: options.device,
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

      // push returns false when the consumer is slow — we can't pause a mic,
      // but we surface the backpressure warning so callers can react.
      if (!this.push(chunk)) {
        this.emit('backpressure');
      }

      if (this._vad) {
        const rms = computeRMS(chunk, this._format);
        if (rms >= this._vadThreshold) {
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
    });
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

module.exports = Decibri;
