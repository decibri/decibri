'use strict';

const { Emitter } = require('./emitter.js');
const { WORKLET_SOURCE } = require('./worklet-inline.js');

const VERSION = '3.0.0-rc.5';

/**
 * Browser microphone capture.
 *
 * Uses getUserMedia + AudioWorklet for real-time audio capture in browsers.
 * Emits 'data' events with Int16Array or Float32Array chunks.
 *
 * Ported from decibri-web decibri.ts. Logic identical, types removed.
 *
 * @example
 * const { Decibri } = require('decibri'); // browser entry via conditional export
 * const mic = new Decibri({ sampleRate: 16000 });
 * mic.on('data', (chunk) => { // chunk is Int16Array });
 * await mic.start();
 * // later...
 * mic.stop();
 */
class Decibri extends Emitter {
  constructor(options = {}) {
    super();

    // ── Private state ─────────────────────────────────────────────────────
    this._audioContext = null;
    this._stream = null;
    this._sourceNode = null;
    this._workletNode = null;
    this._started = false;
    this._starting = null;
    this._stopRequested = false;

    // ── VAD state ─────────────────────────────────────────────────────────
    this._vad = options.vad ?? false;
    this._vadThreshold = options.vadThreshold ?? 0.01;
    this._vadHoldoff = options.vadHoldoff ?? 300;
    this._isSpeaking = false;
    this._silenceTimer = null;

    // ── Options ───────────────────────────────────────────────────────────
    this._sampleRate = options.sampleRate ?? 16000;
    this._channels = options.channels ?? 1;
    this._framesPerBuffer = options.framesPerBuffer ?? 1600;
    this._device = options.device;
    this._format = options.format ?? 'int16';
    this._echoCancellation = options.echoCancellation ?? true;
    this._noiseSuppression = options.noiseSuppression ?? true;
    this._workletUrl = options.workletUrl;

    // ── Validate ──────────────────────────────────────────────────────────
    if (this._sampleRate < 1000 || this._sampleRate > 384000) {
      throw new TypeError(`sampleRate must be between 1000 and 384000, got ${this._sampleRate}`);
    }
    if (this._channels < 1 || this._channels > 32) {
      throw new TypeError(`channels must be between 1 and 32, got ${this._channels}`);
    }
    if (this._framesPerBuffer < 64 || this._framesPerBuffer > 65536) {
      throw new TypeError(`framesPerBuffer must be between 64 and 65536, got ${this._framesPerBuffer}`);
    }
    if (this._format !== 'int16' && this._format !== 'float32') {
      throw new TypeError("format must be 'int16' or 'float32'");
    }
    if (this._vadThreshold < 0 || this._vadThreshold > 1) {
      throw new TypeError(`vadThreshold must be between 0 and 1, got ${this._vadThreshold}`);
    }
    if (this._vadHoldoff < 0) {
      throw new TypeError(`vadHoldoff must be >= 0, got ${this._vadHoldoff}`);
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────

  /**
   * Start microphone capture.
   * Requests microphone permission and sets up the audio pipeline.
   * Must be called from a user gesture context in Safari.
   * No-op if already started. Returns the existing promise if a start
   * is already in progress.
   */
  start() {
    if (this._started) return Promise.resolve();
    if (this._starting) return this._starting;

    this._starting = this._doStart().finally(() => {
      this._starting = null;
    });

    return this._starting;
  }

  /**
   * Stop microphone capture and release all resources.
   * Safe to call multiple times or before start().
   * After stop(), calling start() again creates a fresh session.
   */
  stop() {
    if (!this._started) {
      if (this._starting) {
        this._stopRequested = true;
      }
      return;
    }
    this._started = false;

    // Stop media tracks
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
    }

    // Disconnect audio nodes
    if (this._sourceNode) this._sourceNode.disconnect();
    if (this._workletNode) {
      this._workletNode.disconnect();
      this._workletNode.port.close();
    }

    // Close audio context
    if (this._audioContext) this._audioContext.close();

    // Clear VAD
    if (this._silenceTimer !== null) {
      clearTimeout(this._silenceTimer);
      this._silenceTimer = null;
    }
    this._isSpeaking = false;

    // Release references
    this._audioContext = null;
    this._stream = null;
    this._sourceNode = null;
    this._workletNode = null;

    this.emit('end');
    this.emit('close');
  }

  /** Whether the microphone is currently capturing. */
  get isOpen() {
    return this._started;
  }

  /**
   * List available audio input devices.
   * Device labels may be empty until microphone permission is granted.
   */
  static async devices() {
    const all = await navigator.mediaDevices.enumerateDevices();
    return all
      .filter(d => d.kind === 'audioinput')
      .map(d => ({
        deviceId: d.deviceId,
        label: d.label,
        groupId: d.groupId,
      }));
  }

  /** Version information. */
  static version() {
    return { decibri: VERSION };
  }

  // ── Private ─────────────────────────────────────────────────────────────

  async _doStart() {
    // 1. Create AudioContext at native sample rate
    this._audioContext = new AudioContext();
    const nativeSampleRate = this._audioContext.sampleRate;

    // Safari fix: resume suspended AudioContext
    await this._audioContext.resume();

    // 2. Request microphone access
    const audioConstraints = {
      channelCount: this._channels,
      echoCancellation: this._echoCancellation,
      noiseSuppression: this._noiseSuppression,
    };
    if (this._device) {
      audioConstraints.deviceId = { exact: this._device };
    }

    try {
      this._stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    } catch (err) {
      await this._audioContext.close();
      this._audioContext = null;
      const error = this._mapError(err);
      this.emit('error', error);
      throw error;
    }

    // 3. Load AudioWorklet processor
    let blobUrl = null;
    const workletUrl = this._workletUrl ?? (blobUrl = this._createBlobUrl());

    try {
      await this._audioContext.audioWorklet.addModule(workletUrl);
    } catch (err) {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
      await this._audioContext.close();
      this._audioContext = null;
      const error = new Error('Failed to load audio worklet: ' + (err instanceof Error ? err.message : String(err)));
      this.emit('error', error);
      throw error;
    }

    if (blobUrl) URL.revokeObjectURL(blobUrl);

    // 4. Build audio graph
    this._sourceNode = this._audioContext.createMediaStreamSource(this._stream);
    this._workletNode = new AudioWorkletNode(this._audioContext, 'decibri-processor', {
      processorOptions: {
        framesPerBuffer: this._framesPerBuffer,
        format: this._format,
        nativeSampleRate,
        targetSampleRate: this._sampleRate,
      },
    });

    // 5. Wire up data from worklet
    this._workletNode.port.onmessage = (event) => {
      const buffer = event.data;
      const chunk = this._format === 'int16'
        ? new Int16Array(buffer)
        : new Float32Array(buffer);

      this.emit('data', chunk);

      if (this._vad) {
        this._processVad(chunk);
      }
    };

    // 6. Connect (source → worklet, NOT to destination)
    this._sourceNode.connect(this._workletNode);

    this._started = true;

    // If stop() was called while start() was in-flight, tear down now
    if (this._stopRequested) {
      this._stopRequested = false;
      this.stop();
    }
  }

  _createBlobUrl() {
    const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
  }

  _mapError(err) {
    if (err instanceof DOMException) {
      switch (err.name) {
        case 'NotAllowedError':
          return new Error('Microphone permission denied');
        case 'NotFoundError':
          return new Error('No microphone found');
        default:
          return new Error('Microphone access failed: ' + err.message);
      }
    }
    return err instanceof Error ? err : new Error(String(err));
  }

  // ── VAD ─────────────────────────────────────────────────────────────────

  _processVad(chunk) {
    const rms = this._computeRms(chunk);

    if (rms >= this._vadThreshold) {
      if (this._silenceTimer !== null) {
        clearTimeout(this._silenceTimer);
        this._silenceTimer = null;
      }
      if (!this._isSpeaking) {
        this._isSpeaking = true;
        this.emit('speech');
      }
    } else if (this._isSpeaking && this._silenceTimer === null) {
      this._silenceTimer = setTimeout(() => {
        this._isSpeaking = false;
        this._silenceTimer = null;
        this.emit('silence');
      }, this._vadHoldoff);
    }
  }

  _computeRms(chunk) {
    let sum = 0;
    const n = chunk.length;
    if (n === 0) return 0;

    if (chunk instanceof Float32Array) {
      for (let i = 0; i < n; i++) sum += chunk[i] * chunk[i];
    } else {
      for (let i = 0; i < n; i++) {
        const s = chunk[i] / 32768;
        sum += s * s;
      }
    }

    return Math.sqrt(sum / n);
  }
}

module.exports = { Decibri };
