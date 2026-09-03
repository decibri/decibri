/**
 * AudioWorklet processor for decibri browser capture.
 *
 * THIS FILE IS THE READABLE SOURCE. It is NOT loaded at runtime.
 * The minified version in worklet-inline.js is what actually runs.
 * If you change logic here, you MUST regenerate worklet-inline.js.
 *
 * Runs in a dedicated audio thread. Receives Float32 samples at the
 * browser's native sample rate, derives the delivered channels from the
 * granted ones (the average of every channel, every channel in granted
 * order, or the channels the map selects), resamples each channel to the
 * target rate via linear interpolation, interleaves frame by frame,
 * optionally converts to Int16, and posts chunks to the main thread.
 *
 * This file cannot import other modules (AudioWorklet restriction).
 *
 * Ported from decibri-web worklet-processor.ts. Logic identical, types removed.
 */

class DecibriProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions;
    this.framesPerBuffer = opts.framesPerBuffer;
    this.format = opts.format;
    this.native = opts.nativeSampleRate;
    this.target = opts.targetSampleRate;
    this.needsResample = opts.nativeSampleRate !== opts.targetSampleRate;
    // Optional list of 0-based channel indices into the granted track's
    // channels; delivered channel j carries granted channel channelMap[j].
    // null derives the delivered channels from the count alone: 1 delivers
    // the average of every granted channel, the granted count delivers every
    // granted channel in granted order, and any other count is refused in
    // process(). Where the browser reports the granted channel count, the
    // main thread checked all of this before this worklet was built; the
    // per-block guard in process() is the authority where it does not.
    this.channelMap = opts.channelMap ?? null;
    // The delivered channel count. A map carries one entry per delivered
    // channel, so its length is the count when one is present.
    this.channels = this.channelMap ? this.channelMap.length : (opts.channels ?? 1);
    this.channelError = false;
    // The resampler's phase: the position of the next delivered frame in
    // input samples relative to the current block's first sample, held as an
    // integer numerator over the target rate so it is exact across any
    // number of blocks. It may sit below zero, down to one input sample
    // before the block, for a frame whose time falls between the previous
    // block's last sample and this block's first; that last sample is kept
    // per delivered channel in `last`. The first block starts at zero, on
    // its own first sample.
    this.phase = 0;
    this.last = new Float32Array(this.channels);
    // The delivered channels of the current block, planar, gathered into
    // this array in place so a block allocates nothing.
    this.planar = new Array(this.channels);
    // The average of every granted channel, for one delivered channel
    // derived from more than one granted: sized to the block on first use
    // and reused while the block length holds.
    this.mono = null;
    // The accumulation buffer holds framesPerBuffer frames of the delivered
    // count, interleaved frame by frame; bufferIndex counts samples. Chunks
    // are flushed at whole frames only.
    this.samplesPerChunk = this.framesPerBuffer * this.channels;
    this.buffer = new Float32Array(this.samplesPerChunk);
    this.bufferIndex = 0;
  }

  process(inputs, _outputs, _parameters) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0] || input[0].length === 0) return true;
    if (this.channelError) return false;

    const granted = input.length;
    // The delivered channels, planar: one Float32Array per delivered
    // channel, equal lengths. Gathered here into the preallocated array,
    // resampled per channel in lockstep, interleaved at accumulation.
    const planar = this.planar;

    if (this.channelMap) {
      // The granted track's channel count is the only ceiling. A map entry
      // the block cannot serve is reported once and stops the processor; it
      // is never silently substituted.
      for (let j = 0; j < this.channelMap.length; j++) {
        if (this.channelMap[j] >= granted) {
          return this.refuse('the channel map names device channel ' + this.channelMap[j] +
            '; the device reports ' + granted + ' input channels');
        }
      }
      // Delivered channel j is granted channel channelMap[j], in map order.
      // Entries may repeat and may appear in any order, so a map both
      // selects and permutes.
      for (let j = 0; j < this.channelMap.length; j++) {
        planar[j] = input[this.channelMap[j]];
      }
    } else if (this.channels === 1) {
      if (granted === 1) {
        planar[0] = input[0];
      } else {
        // The documented average of every granted channel: each frame's
        // arithmetic mean, accumulated at single precision (Math.fround per
        // step) and stored as f32, matching the engine's average sample for
        // sample.
        const frames = input[0].length;
        if (this.mono === null || this.mono.length !== frames) {
          this.mono = new Float32Array(frames);
        }
        const mono = this.mono;
        for (let i = 0; i < frames; i++) {
          let sum = 0;
          for (let c = 0; c < granted; c++) {
            sum = Math.fround(sum + input[c][i]);
          }
          mono[i] = sum / granted;
        }
        planar[0] = mono;
      }
    } else if (this.channels === granted) {
      // Every granted channel, in granted order: the unmapped identity.
      for (let c = 0; c < granted; c++) planar[c] = input[c];
    } else if (this.channels > granted) {
      return this.refuse('the input device does not support ' + this.channels +
        ' delivered channels; it reports ' + granted);
    } else {
      // An unmapped strict subset above one: which of the granted channels
      // it means has no single answer, so the map has to name them.
      return this.refuse('a channel map is required to deliver ' + this.channels +
        " of the device's " + granted + ' input channels');
    }

    if (this.needsResample) {
      this.resample(planar);
    } else {
      this.accumulate(planar);
    }

    return true;
  }

  /**
   * Interleave the block's frames into the accumulation buffer as they are,
   * flushing at whole chunks, so every posted chunk is a whole number of
   * frames.
   */
  accumulate(planar) {
    const frames = planar[0].length;
    const channels = this.channels;
    for (let i = 0; i < frames; i++) {
      for (let c = 0; c < channels; c++) {
        this.buffer[this.bufferIndex++] = planar[c][i];
      }
      if (this.bufferIndex >= this.samplesPerChunk) {
        this.flush();
      }
    }
  }

  /**
   * Report a channel configuration no block can serve, once, with the
   * engine's message for the same condition, and stop the processor: the
   * failure is terminal and never silently substituted.
   */
  refuse(message) {
    this.channelError = true;
    this.port.postMessage({ type: 'error', message });
    return false;
  }

  /**
   * Resample the block by linear interpolation at one phase shared by every
   * channel, so a delivered frame stays a frame, interleaving each frame
   * straight into the accumulation buffer and flushing at whole chunks.
   * Frame k of the stream sits at k * native / target input samples. The
   * phase carries that position across blocks as an exact integer numerator
   * over the target rate, without clamping, and the previous block's last
   * sample per channel is kept, so a frame whose time falls between two
   * blocks is interpolated across the boundary. Every input sample's time is
   * represented once, and the delivered count over any length of capture is
   * exact.
   */
  resample(planar) {
    const inputLength = planar[0].length;
    const channels = this.channels;
    const native = this.native;
    const target = this.target;
    const last = this.last;
    let phase = this.phase;

    // A frame needs the input sample after its floor, so the block serves
    // every position below inputLength - 1.
    const end = (inputLength - 1) * target;
    while (phase < end) {
      const idx = Math.floor(phase / target);
      const frac = (phase - idx * target) / target;
      if (idx < 0) {
        // Between the stored sample and this block's first.
        for (let c = 0; c < channels; c++) {
          this.buffer[this.bufferIndex++] = last[c] * (1 - frac) + planar[c][0] * frac;
        }
      } else {
        for (let c = 0; c < channels; c++) {
          this.buffer[this.bufferIndex++] = planar[c][idx] * (1 - frac) + planar[c][idx + 1] * frac;
        }
      }
      if (this.bufferIndex >= this.samplesPerChunk) {
        this.flush();
      }
      phase += native;
    }

    // Carry the phase relative to the next block's first sample; it sits in
    // [-target, native - target). Keep the block's last sample for the frame
    // that may fall before that block begins.
    this.phase = phase - inputLength * target;
    for (let c = 0; c < channels; c++) last[c] = planar[c][inputLength - 1];
  }

  flush() {
    let transferBuffer;

    if (this.format === 'int16') {
      // The posted buffer is transferred, so each chunk converts into a
      // fresh Int16Array; the accumulation buffer stays and is refilled from
      // its start.
      const int16 = new Int16Array(this.samplesPerChunk);
      for (let i = 0; i < this.samplesPerChunk; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(this.buffer[i] * 32768)));
      }
      transferBuffer = int16.buffer;
    } else {
      // The accumulation buffer itself is transferred, so a fresh one
      // replaces it.
      transferBuffer = this.buffer.buffer;
      this.buffer = new Float32Array(this.samplesPerChunk);
    }

    this.port.postMessage(transferBuffer, [transferBuffer]);
    this.bufferIndex = 0;
  }
}

registerProcessor('decibri-processor', DecibriProcessor);
