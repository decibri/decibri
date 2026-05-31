'use strict';

const { OUTPUT_WORKLET_SOURCE } = require('./output-worklet-inline.js');

// Seconds of audio the output ring buffer holds. The ring capacity is derived
// from this and the context sample rate at start, so backpressure kicks in
// before more than this much audio is queued ahead of playback.
const BUFFER_SECONDS = 2;

/**
 * Linear-interpolation resampler, the inverse of the capture worklet's: it
 * takes samples at the user's rate and produces samples at the context rate.
 * Carries a fractional position across calls so successive writes stay
 * continuous. Logic mirrors worklet-processor.js resample(), with from and to
 * swapped (from = user rate, to = context rate).
 */
class Resampler {
  constructor(fromRate, toRate) {
    this.ratio = fromRate / toRate;
    this.position = 0;
  }

  process(input) {
    if (this.ratio === 1) return input;

    const inputLength = input.length;

    let count = 0;
    let pos = this.position;
    while (pos < inputLength - 1) {
      count++;
      pos += this.ratio;
    }

    const output = new Float32Array(count);
    pos = this.position;

    for (let i = 0; i < count; i++) {
      const idx = Math.floor(pos);
      const frac = pos - idx;
      output[i] = input[idx] * (1 - frac) + input[idx + 1] * frac;
      pos += this.ratio;
    }

    this.position = Math.max(0, pos - inputLength);

    return output;
  }
}

/**
 * Browser audio playback.
 *
 * Plays PCM audio through the Web Audio API. Samples are converted to float32
 * and resampled to the context rate on the main thread, then fed to an output
 * AudioWorklet that drains them to the speakers. This is the inverse of the
 * browser Microphone: the Microphone captures input and emits chunks; the
 * Speaker accepts chunks and plays them.
 *
 * Playback is async throughout, as the browser requires: write() resolves when
 * the samples are queued (after backpressure when the queue is full) and
 * drain() resolves when the queued audio has finished playing. The first
 * write() (or start()) must run in a user gesture so the browser allows audio.
 *
 * @example
 * const { Speaker } = require('decibri'); // browser entry via conditional export
 * const speaker = new Speaker({ sampleRate: 16000 });
 * button.onclick = async () => {
 *   await speaker.write(int16Chunk); // Int16Array of PCM samples
 *   await speaker.drain();
 *   speaker.stop();
 * };
 */
class Speaker {
  constructor(options = {}) {
    // ── State ─────────────────────────────────────────────────────────────
    this._audioContext = null;
    this._workletNode = null;
    this._resampler = null;
    this._started = false;
    this._starting = null;
    this._stopRequested = false;

    // Last authoritative ring level from a 'level' ack, with the context time
    // it was observed, so the current depth can be estimated by decay.
    this._lastAck = null;
    // Ring capacity in samples, set at start from the context rate. The 'level'
    // acks confirm it.
    this._capacity = 0;
    this._contextRate = 0;

    // Whether audio queued for playback has not yet finished. Set on each feed,
    // cleared when the worklet reports the ring drained.
    this._unplayed = false;
    this._pendingDrains = [];

    // ── Options ───────────────────────────────────────────────────────────
    this._sampleRate = options.sampleRate ?? 16000;
    this._channels = options.channels ?? 1;
    this._dtype = options.dtype ?? 'int16';
    this._workletUrl = options.workletUrl;

    // ── Validate (mirrors the browser Microphone error style: TypeError) ────
    if (this._sampleRate < 1000 || this._sampleRate > 384000) {
      throw new TypeError(`sample rate must be between 1000 and 384000, got ${this._sampleRate}`);
    }
    if (this._channels < 1 || this._channels > 32) {
      throw new TypeError(`channels must be between 1 and 32, got ${this._channels}`);
    }
    if (this._dtype !== 'int16' && this._dtype !== 'float32') {
      throw new TypeError("dtype must be 'int16' or 'float32'");
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Create and resume the AudioContext and load the output worklet.
   *
   * Must be called from a user gesture context so the browser allows audio.
   * Optional: write() starts playback on its own if start() was not called, but
   * calling start() from a click handler is the reliable way to unlock audio
   * before samples are ready. No-op if already started; returns the in-flight
   * promise if a start is already in progress.
   *
   * @returns {Promise<void>}
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
   * Play PCM audio. Resolves when the samples are queued.
   *
   * Converts the chunk to context-rate float32 (int16 to float32 if needed,
   * then resample), and feeds it to the output worklet. When the queue is full
   * it applies backpressure: the returned promise does not resolve until there
   * is room, so a caller that awaits write() is paced to playback. Starts
   * playback on the first call (gesture-sensitive).
   *
   * Await calls sequentially to preserve sample order. An empty chunk resolves
   * immediately. Writing after stop() starts a fresh playback session.
   *
   * @param {Int16Array|Float32Array|ArrayBuffer} chunk PCM samples in the
   *   configured dtype.
   * @returns {Promise<void>}
   */
  async write(chunk) {
    await this.start();
    if (!this._started) throw new Error('Speaker is stopped');

    const float32 = this._convert(chunk);
    if (float32.length === 0) return;

    let offset = 0;
    while (offset < float32.length) {
      const sliceLen = Math.min(float32.length - offset, this._capacity);
      await this._reserve(sliceLen);
      if (!this._started) throw new Error('Speaker is stopped');

      // slice() copies into its own buffer, so transferring it does not neuter
      // the caller's array or the rest of this conversion.
      const slice = float32.slice(offset, offset + sliceLen);
      this._workletNode.port.postMessage(slice.buffer, [slice.buffer]);

      // Anchor the queue estimate optimistically so back-to-back writes pace
      // correctly before the worklet's 'level' ack lands; the ack refines it.
      this._lastAck = { queued: this._estimateQueue() + sliceLen, time: this._now() };
      this._unplayed = true;
      offset += sliceLen;
    }
  }

  /**
   * Wait for all queued audio to finish playing.
   *
   * Resolves when the worklet reports its queue drained. Resolves immediately
   * if nothing is queued (nothing written, or playback already caught up).
   *
   * @returns {Promise<void>}
   */
  drain() {
    if (!this._started || !this._unplayed) return Promise.resolve();
    return new Promise((resolve) => {
      this._pendingDrains.push(resolve);
    });
  }

  /**
   * Immediate stop. Discards queued audio and releases all resources.
   * Safe to call multiple times or before start(). After stop(), write() or
   * start() begins a fresh session.
   */
  stop() {
    if (!this._started) {
      // If a start is in flight, tear it down once it finishes.
      if (this._starting) this._stopRequested = true;
      return;
    }
    this._started = false;

    if (this._workletNode) {
      try {
        this._workletNode.port.postMessage({ type: 'flush' });
      } catch {
        // The port may already be closing; the context close below stops audio.
      }
      this._workletNode.disconnect();
      this._workletNode.port.onmessage = null;
      this._workletNode.port.close();
    }

    if (this._audioContext) this._audioContext.close();

    // The queue was discarded; release anyone waiting on drain so they do not
    // hang. Nothing more will play on this session.
    const drains = this._pendingDrains;
    this._pendingDrains = [];
    for (const resolve of drains) resolve();

    this._audioContext = null;
    this._workletNode = null;
    this._resampler = null;
    this._lastAck = null;
    this._unplayed = false;
  }

  /** Whether audio is currently queued and playing. */
  get isPlaying() {
    return this._started && this._unplayed;
  }

  // ── Private ─────────────────────────────────────────────────────────────

  async _doStart() {
    // 1. Create the AudioContext at its native rate and resume it (Safari and
    //    the autoplay policy both leave it suspended until resumed).
    this._audioContext = new AudioContext();
    this._contextRate = this._audioContext.sampleRate;

    await this._audioContext.resume();

    // A context still suspended after resume() means the browser blocked audio
    // (no user gesture). Surface it rather than failing silently.
    if (this._audioContext.state === 'suspended') {
      await this._audioContext.close();
      this._audioContext = null;
      throw new Error('Audio playback is blocked until a user gesture resumes the audio context');
    }

    // 2. Load the output AudioWorklet processor.
    let blobUrl = null;
    const workletUrl = this._workletUrl ?? (blobUrl = this._createBlobUrl());

    try {
      await this._audioContext.audioWorklet.addModule(workletUrl);
    } catch (err) {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      await this._audioContext.close();
      this._audioContext = null;
      throw new Error('Failed to load audio worklet: ' + (err instanceof Error ? err.message : String(err)));
    }

    if (blobUrl) URL.revokeObjectURL(blobUrl);

    // 3. Size the ring and build the resampler now that the context rate is known.
    this._capacity = Math.round(this._contextRate * BUFFER_SECONDS);
    this._resampler = new Resampler(this._sampleRate, this._contextRate);
    this._lastAck = null;
    this._unplayed = false;

    // 4. Build the output node and connect it TO the destination (the inverse
    //    of capture, which deliberately does not connect to the destination).
    this._workletNode = new AudioWorkletNode(this._audioContext, 'decibri-output-processor', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [this._channels],
      processorOptions: { ringCapacity: this._capacity },
    });

    this._workletNode.port.onmessage = (event) => this._onMessage(event);
    this._workletNode.connect(this._audioContext.destination);

    this._started = true;

    // If stop() was called while start() was in flight, tear down now.
    if (this._stopRequested) {
      this._stopRequested = false;
      this.stop();
    }
  }

  _onMessage(event) {
    const msg = event.data;
    if (!msg) return;

    if (msg.type === 'level') {
      // Authoritative ring depth at this moment; the capacity is confirmed too.
      this._lastAck = { queued: msg.queued, time: this._now() };
      this._capacity = msg.capacity;
    } else if (msg.type === 'drained') {
      this._unplayed = false;
      const drains = this._pendingDrains;
      this._pendingDrains = [];
      for (const resolve of drains) resolve();
    }
  }

  _createBlobUrl() {
    const blob = new Blob([OUTPUT_WORKLET_SOURCE], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
  }

  _now() {
    return this._audioContext ? this._audioContext.currentTime : 0;
  }

  // Estimate the ring depth right now: the last acked depth, decayed by the
  // samples that have played since (the worklet consumes at the context rate).
  _estimateQueue() {
    if (!this._lastAck) return 0;
    const elapsed = this._now() - this._lastAck.time;
    const played = Math.max(0, elapsed) * this._contextRate;
    return Math.max(0, this._lastAck.queued - played);
  }

  // Wait until the ring can accept `samples` more without overflowing, so a
  // transferred feed is never partially dropped. Returns as soon as there is
  // room; under pressure it sleeps for the time the surplus needs to drain.
  async _reserve(samples) {
    while (this._started) {
      const overflow = this._estimateQueue() + samples - this._capacity;
      if (overflow <= 0) return;
      const waitMs = Math.max(1, (overflow / this._contextRate) * 1000);
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }

  _convert(chunk) {
    let float32;
    if (this._dtype === 'int16') {
      const int16 = this._asInt16(chunk);
      float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
    } else {
      float32 = this._asFloat32(chunk);
    }
    return this._resampler.process(float32);
  }

  _asInt16(chunk) {
    if (chunk instanceof Int16Array) return chunk;
    if (chunk instanceof ArrayBuffer) return new Int16Array(chunk);
    if (ArrayBuffer.isView(chunk)) {
      return new Int16Array(chunk.buffer, chunk.byteOffset, Math.floor(chunk.byteLength / 2));
    }
    throw new TypeError('write() expects an Int16Array, ArrayBuffer, or typed array for int16 dtype');
  }

  _asFloat32(chunk) {
    if (chunk instanceof Float32Array) return chunk;
    if (chunk instanceof ArrayBuffer) return new Float32Array(chunk);
    if (ArrayBuffer.isView(chunk)) {
      return new Float32Array(chunk.buffer, chunk.byteOffset, Math.floor(chunk.byteLength / 4));
    }
    throw new TypeError('write() expects a Float32Array, ArrayBuffer, or typed array for float32 dtype');
  }
}

module.exports = { Speaker };
