/**
 * AudioWorklet processor for decibri browser capture.
 *
 * THIS FILE IS THE READABLE SOURCE. It is NOT loaded at runtime.
 * The minified version in worklet-inline.js is what actually runs.
 * If you change logic here, you MUST regenerate worklet-inline.js.
 *
 * Runs in a dedicated audio thread. Receives Float32 samples at the
 * browser's native sample rate, resamples to the target rate via linear
 * interpolation, optionally converts to Int16, and posts chunks to the
 * main thread.
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
    this.position = 0;
    this.buffer = new Float32Array(this.framesPerBuffer);
    this.bufferIndex = 0;
  }

  process(inputs, _outputs, _parameters) {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) return true;

    let samples;

    if (this.needsResample) {
      samples = this.resample(input);
    } else {
      samples = input;
    }

    // Accumulate resampled frames into the buffer
    let offset = 0;
    while (offset < samples.length) {
      const remaining = this.framesPerBuffer - this.bufferIndex;
      const available = samples.length - offset;
      const toCopy = Math.min(remaining, available);

      this.buffer.set(samples.subarray(offset, offset + toCopy), this.bufferIndex);
      this.bufferIndex += toCopy;
      offset += toCopy;

      if (this.bufferIndex >= this.framesPerBuffer) {
        this.flush();
      }
    }

    return true;
  }

  resample(input) {
    const inputLength = input.length;

    // Calculate how many output samples we can produce
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

    // Carry fractional remainder relative to consumed input.
    this.position = Math.max(0, pos - inputLength);

    return output;
  }

  flush() {
    let transferBuffer;

    if (this.format === 'int16') {
      const int16 = new Int16Array(this.framesPerBuffer);
      for (let i = 0; i < this.framesPerBuffer; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(this.buffer[i] * 32768)));
      }
      transferBuffer = int16.buffer;
    } else {
      transferBuffer = this.buffer.slice(0, this.framesPerBuffer).buffer;
    }

    this.port.postMessage(transferBuffer, [transferBuffer]);

    // Reset accumulation buffer
    this.buffer = new Float32Array(this.framesPerBuffer);
    this.bufferIndex = 0;
  }
}

registerProcessor('decibri-processor', DecibriProcessor);
