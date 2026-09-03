'use strict';

const { Emitter } = require('./emitter.js');
const { WORKLET_SOURCE } = require('./worklet-inline.js');

// Browser build version. Keep in sync with package.json on each release; the
// browser bundle cannot read package.json at runtime the way the Node wrapper
// does, so this is a maintained constant.
const VERSION = '5.7.0';

// The node entry's options this entry cannot serve: the conditioning chain
// and the detector model file. Each is refused on presence, whatever its
// value; an explicit undefined is absence, as on the node entry. Node option
// table order, so the first present key is the one the error names.
const UNSUPPORTED_OPTIONS = ['modelPath', 'dcRemoval', 'denoise', 'highpass', 'agc', 'limiter', 'aec'];

/**
 * Browser microphone capture.
 *
 * Uses getUserMedia + AudioWorklet for real-time audio capture in browsers.
 * Emits 'data' events with Int16Array or Float32Array chunks holding
 * framesPerBuffer frames of the delivered channel count, interleaved frame
 * by frame.
 *
 * Ported from decibri-web decibri.ts. Logic identical, types removed.
 *
 * @example
 * const { Microphone } = require('decibri'); // browser entry via conditional export
 * const mic = new Microphone({ sampleRate: 16000 });
 * mic.on('data', (chunk) => { // chunk is Int16Array });
 * await mic.start();
 * // later...
 * mic.stop();
 */
class Microphone extends Emitter {
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
    // vad accepts false (disabled, default), the 'energy' shorthand, or a config
    // object { model: 'energy', threshold?, holdoffMs?, source? } to tune the
    // policy. The browser runs energy VAD only; Silero needs ONNX Runtime,
    // which is Node-only. The legacy vad: true form and the flat
    // vadThreshold/vadHoldoff options are rejected with a migration error.
    if (options.vadThreshold !== undefined || options.vadHoldoff !== undefined) {
      throw new TypeError(
        "vadThreshold and vadHoldoff are no longer supported. Pass them on the vad config object: vad: { model: 'energy', threshold: 0.01, holdoffMs: 300 }."
      );
    }
    for (const key of UNSUPPORTED_OPTIONS) {
      if (options[key] !== undefined) {
        throw new TypeError(`${key} is not supported in the browser`);
      }
    }
    const vad = options.vad ?? false;
    let vadThreshold = 0.01;
    let vadHoldoff = 300;
    let vadSource;
    if (vad === false) {
      this._vad = false;
    } else if (vad === true) {
      throw new TypeError("vad: true is no longer supported. Specify the mode explicitly: vad: 'energy'.");
    } else if (vad === 'energy') {
      this._vad = true;
    } else if (vad !== null && typeof vad === 'object' && !Array.isArray(vad)) {
      if (vad.model !== 'energy') {
        throw new TypeError(`Invalid vad model: ${JSON.stringify(vad.model)}. Expected 'energy'.`);
      }
      this._vad = true;
      // The guards, the error classes and the messages are the node entry's,
      // so the same option value is rejected the same way in both runtimes.
      if (vad.threshold !== undefined) {
        if (typeof vad.threshold !== 'number' || Number.isNaN(vad.threshold)) {
          throw new TypeError('vad threshold must be a number');
        }
        if (vad.threshold < 0 || vad.threshold > 1) {
          throw new RangeError('vad threshold must be between 0 and 1');
        }
        vadThreshold = vad.threshold;
      }
      if (vad.holdoffMs !== undefined) {
        if (typeof vad.holdoffMs !== 'number' || Number.isNaN(vad.holdoffMs)) {
          throw new TypeError('vad holdoffMs must be a number');
        }
        if (vad.holdoffMs < 0) {
          throw new RangeError('vad holdoffMs must be non-negative');
        }
        vadHoldoff = vad.holdoffMs;
      }
      // source names the 0-based DELIVERED channel the detector reads, the
      // position within the delivered interleaved frames after any
      // channelMap; absent scores the frame average of every delivered
      // channel. The checks and the messages are the node entry's. The
      // delivered-count check runs only against a valid count; a count below
      // one is reported as its own error below, exactly as on the node entry.
      if (vad.source !== undefined) {
        const source = vad.source;
        if (typeof source !== 'number' || !Number.isInteger(source)) {
          throw new TypeError('vad source must be an integer');
        }
        if (source < 0 || source > 65535) {
          throw new RangeError('vad source must be between 0 and 65535');
        }
        const deliveredChannels = options.channels ?? 1;
        if (deliveredChannels >= 1 && source >= deliveredChannels) {
          throw new RangeError(
            `the detector source names delivered channel ${source}; the delivered channel count is ${deliveredChannels}`
          );
        }
        vadSource = source;
      }
    } else {
      throw new TypeError(`Invalid vad value: ${JSON.stringify(vad)}. Expected false, 'energy', or a config object { model, threshold, holdoffMs, source }.`);
    }
    this._vadThreshold = vadThreshold;
    this._vadHoldoff = vadHoldoff;
    this._vadSource = vadSource;
    this._vadScore = 0;
    this._isSpeaking = false;
    this._silenceTimer = null;

    // ── Options ───────────────────────────────────────────────────────────
    this._sampleRate = options.sampleRate ?? 16000;
    this._channels = options.channels ?? 1;
    this._channelMap = options.channelMap;
    this._framesPerBuffer = options.framesPerBuffer ?? 1600;
    this._device = options.device;
    this._dtype = options.dtype ?? 'int16';
    this._echoCancellation = options.echoCancellation ?? true;
    this._noiseSuppression = options.noiseSuppression ?? true;
    this._workletUrl = options.workletUrl;

    // ── Validate ──────────────────────────────────────────────────────────
    if (this._sampleRate < 1000 || this._sampleRate > 384000) {
      throw new RangeError('sample rate must be between 1000 and 384000');
    }
    // The number of channels delivered, interleaved frame by frame. Bounded
    // below here; bounded above by the granted track alone, which answers
    // when the stream starts. No fixed maximum exists on this path. The
    // classes and the messages are the node entry's, so the same value is
    // rejected the same way in both runtimes.
    if (this._channels < 1) {
      throw new RangeError('channels must be at least 1');
    }
    // An optional list of 0-based device channel indices, one per delivered
    // channel: delivered channel j carries device channel channelMap[j].
    // Entries may repeat and may appear in any order, so a map both selects
    // and permutes. Absence derives the delivered channels from the count
    // alone: 1 delivers the documented average of every granted channel, and
    // the granted count delivers every granted channel in granted order. The
    // checks here are shape-only (an array of integers that fit the channel
    // count's width, with one entry per channel), the node entry's classes
    // and messages; whether each entry exists on the track is checked when
    // the stream starts, against the granted track's own report, because
    // only the grant can say how many channels it carries. No fixed maximum
    // exists on this path.
    const channelMap = this._channelMap;
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
      if (channelMap.length !== this._channels) {
        throw new RangeError('channelMap must have exactly one entry per channel');
      }
    }
    if (this._framesPerBuffer < 64 || this._framesPerBuffer > 65536) {
      throw new TypeError(`frames per buffer must be between 64 and 65536, got ${this._framesPerBuffer}`);
    }
    if (this._dtype !== 'int16' && this._dtype !== 'float32') {
      throw new TypeError("dtype must be 'int16' or 'float32'");
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
   * Most recent VAD score: the normalized RMS of the last chunk in `'energy'`
   * mode, or 0 when VAD is disabled or before the first chunk is processed.
   * A chunk carrying more than one channel is collapsed to the average of
   * its channels, or to the one delivered channel a `vad: { source }` names,
   * before the RMS, so the score reflects one channel's level.
   * @returns {number}
   */
  get vadScore() {
    return this._vadScore;
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
    // The channel ask is 32 with ideal semantics, the count the Web Audio
    // specification requires an implementation to support: the browser
    // grants what it can serve and never rejects on this constraint. The
    // granted track's own report, not this ask, is the authority for
    // everything downstream, so a grant above the ask flows through
    // unclamped.
    const audioConstraints = {
      channelCount: { ideal: 32 },
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

    // The granted track's report is the capture-side authority, as the
    // resolved device's report is on the node path: the map's entries, or
    // the unmapped count's derivation, are checked against it here, before
    // the worklet is built, with the node entry's message for the same
    // condition. Without a map, only two derivations have a single meaning:
    // 1 delivers the average of every granted channel, and the granted count
    // delivers every granted channel in granted order; a count above the
    // grant does not exist to deliver, and a strict subset above one does
    // not say which channels it means, so the map has to name them. A
    // browser that omits channelCount from getSettings() defers the check to
    // the worklet, which sees the true channel count of every block it
    // processes. At one delivered channel with no map there is nothing to
    // check: the average serves any granted count.
    if (this._channelMap !== undefined || this._channels > 1) {
      const track = this._stream.getAudioTracks()[0];
      const settings = track && typeof track.getSettings === 'function' ? track.getSettings() : {};
      const granted = settings.channelCount;
      if (typeof granted === 'number') {
        let message = null;
        if (this._channelMap !== undefined) {
          for (const entry of this._channelMap) {
            if (entry >= granted) {
              message = `the channel map names device channel ${entry}; the device reports ${granted} input channels`;
              break;
            }
          }
        } else if (this._channels > granted) {
          message = `the input device does not support ${this._channels} delivered channels; it reports ${granted}`;
        } else if (this._channels < granted) {
          message = `a channel map is required to deliver ${this._channels} of the device's ${granted} input channels`;
        }
        if (message !== null) {
          this._stream.getTracks().forEach(t => t.stop());
          this._stream = null;
          await this._audioContext.close();
          this._audioContext = null;
          const error = new Error(message);
          this.emit('error', error);
          throw error;
        }
      }
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
        format: this._dtype,
        nativeSampleRate,
        targetSampleRate: this._sampleRate,
        channels: this._channels,
        channelMap: this._channelMap ?? null,
      },
    });

    // 5. Wire up data from worklet
    this._workletNode.port.onmessage = (event) => {
      const data = event.data;

      // The port carries raw ArrayBuffer chunks and tagged control objects,
      // as on the output worklet's port. The one control message is the
      // worklet's channel-map failure: surface it and stop, so a map naming
      // a channel the track does not carry is never silent.
      if (!(data instanceof ArrayBuffer)) {
        if (data && data.type === 'error') {
          const error = new Error(data.message);
          this.emit('error', error);
          this.stop();
        }
        return;
      }

      const chunk = this._dtype === 'int16'
        ? new Int16Array(data)
        : new Float32Array(data);

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
    this._vadScore = rms;

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
    const n = chunk.length;
    if (n === 0) return 0;
    const channels = this._channels;
    const isFloat = chunk instanceof Float32Array;

    if (channels > 1) {
      // Score one channel's level, as the node path's detector feed does:
      // collapse each interleaved frame as the configured source directs,
      // the engine's average (single-precision accumulation, f32 quotient)
      // by default or one named delivered channel's sample alone, then take
      // the RMS of the collapsed signal.
      const frames = Math.floor(n / channels);
      if (frames === 0) return 0;
      const source = this._vadSource;
      let sum = 0;
      if (source !== undefined) {
        for (let f = 0; f < frames; f++) {
          const s = isFloat ? chunk[f * channels + source] : chunk[f * channels + source] / 32768;
          sum += s * s;
        }
        return Math.sqrt(sum / frames);
      }
      for (let f = 0; f < frames; f++) {
        let acc = 0;
        for (let c = 0; c < channels; c++) {
          const s = isFloat ? chunk[f * channels + c] : chunk[f * channels + c] / 32768;
          acc = Math.fround(acc + s);
        }
        const mono = Math.fround(acc / channels);
        sum += mono * mono;
      }
      return Math.sqrt(sum / frames);
    }

    let sum = 0;
    if (isFloat) {
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

module.exports = { Microphone };
