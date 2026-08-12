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
    this.ratio = opts.nativeSampleRate / opts.targetSampleRate;
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
    this.position = 0;
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
    // channel, equal lengths. Gathered here, resampled per channel in
    // lockstep, interleaved at accumulation.
    let planar;

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
      planar = [];
      for (let j = 0; j < this.channelMap.length; j++) {
        planar.push(input[this.channelMap[j]]);
      }
    } else if (this.channels === 1) {
      if (granted === 1) {
        planar = [input[0]];
      } else {
        // The documented average of every granted channel: each frame's
        // arithmetic mean, accumulated at single precision (Math.fround per
        // step) and stored as f32, matching the engine's average sample for
        // sample.
        const frames = input[0].length;
        const mono = new Float32Array(frames);
        for (let i = 0; i < frames; i++) {
          let sum = 0;
          for (let c = 0; c < granted; c++) {
            sum = Math.fround(sum + input[c][i]);
          }
          mono[i] = sum / granted;
        }
        planar = [mono];
      }
    } else if (this.channels === granted) {
      // Every granted channel, in granted order: the unmapped identity.
      planar = [];
      for (let c = 0; c < granted; c++) planar.push(input[c]);
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
      planar = this.resample(planar);
    }

    // Interleave the planar channels into the accumulation buffer frame by
    // frame, flushing at whole chunks, so every posted chunk is a whole
    // number of frames.
    const frames = planar[0].length;
    for (let i = 0; i < frames; i++) {
      for (let c = 0; c < this.channels; c++) {
        this.buffer[this.bufferIndex++] = planar[c][i];
      }
      if (this.bufferIndex >= this.samplesPerChunk) {
        this.flush();
      }
    }

    return true;
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

  resample(planar) {
    const inputLength = planar[0].length;

    // Calculate how many output frames we can produce. One position shared
    // by every channel: the channels advance in lockstep, so a delivered
    // frame stays a frame.
    let count = 0;
    let pos = this.position;
    while (pos < inputLength - 1) {
      count++;
      pos += this.ratio;
    }

    const output = planar.map(() => new Float32Array(count));
    pos = this.position;

    for (let i = 0; i < count; i++) {
      const idx = Math.floor(pos);
      const frac = pos - idx;
      for (let c = 0; c < planar.length; c++) {
        output[c][i] = planar[c][idx] * (1 - frac) + planar[c][idx + 1] * frac;
      }
      pos += this.ratio;
    }

    // Carry fractional remainder relative to consumed input.
    this.position = Math.max(0, pos - inputLength);

    return output;
  }

  flush() {
    let transferBuffer;

    if (this.format === 'int16') {
      const int16 = new Int16Array(this.samplesPerChunk);
      for (let i = 0; i < this.samplesPerChunk; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(this.buffer[i] * 32768)));
      }
      transferBuffer = int16.buffer;
    } else {
      transferBuffer = this.buffer.slice(0, this.samplesPerChunk).buffer;
    }

    this.port.postMessage(transferBuffer, [transferBuffer]);

    // Reset accumulation buffer
    this.buffer = new Float32Array(this.samplesPerChunk);
    this.bufferIndex = 0;
  }
}

registerProcessor('decibri-processor', DecibriProcessor);
