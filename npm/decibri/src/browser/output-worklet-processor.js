/**
 * AudioWorklet processor for decibri browser playback.
 *
 * THIS FILE IS THE READABLE SOURCE. It is NOT loaded at runtime.
 * The minified version in output-worklet-inline.js is what actually runs.
 * If you change logic here, you MUST regenerate output-worklet-inline.js.
 *
 * Runs in a dedicated audio thread. Pulls queued Float32 samples (already at
 * the context sample rate) from a ring buffer and writes them to the output
 * channels. On underrun (the ring buffer is empty) it outputs silence cleanly,
 * with no throw and no glitch beyond the unavoidable gap. This is the inverse
 * of the capture processor: capture reads input frames and posts them to the
 * main thread; this pulls main-thread frames from the ring and writes output.
 *
 * This file cannot import other modules (AudioWorklet restriction).
 *
 * Mirrors worklet-processor.js (capture) in reverse.
 */

/**
 * Fixed-capacity single-producer single-consumer ring buffer of Float32
 * samples. The main thread writes (produces) via write(); the audio thread
 * reads (consumes) via readInto(). Indices wrap around at capacity. A write
 * that would overflow stores only what fits and returns the accepted count, so
 * the producer can apply backpressure. A read past the stored data returns the
 * real count read, so the consumer can fill the remainder with silence.
 */
class RingBuffer {
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = new Float32Array(capacity);
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
  }

  /** Number of samples currently queued for reading. */
  get availableRead() {
    return this.size;
  }

  /** Number of free slots a write can still accept. */
  get availableWrite() {
    return this.capacity - this.size;
  }

  get isEmpty() {
    return this.size === 0;
  }

  get isFull() {
    return this.size === this.capacity;
  }

  /**
   * Write up to samples.length samples into the buffer. Returns the number
   * actually written, which is less than samples.length when the buffer fills.
   */
  write(samples) {
    const toWrite = Math.min(samples.length, this.availableWrite);
    for (let i = 0; i < toWrite; i++) {
      this.buffer[this.writeIndex] = samples[i];
      this.writeIndex = this.writeIndex + 1 === this.capacity ? 0 : this.writeIndex + 1;
    }
    this.size += toWrite;
    return toWrite;
  }

  /**
   * Read up to count samples into target starting at offset. Returns the
   * number actually read, which is less than count on underrun. Positions in
   * target beyond the returned count are left untouched (the caller fills them
   * with silence).
   */
  readInto(target, offset, count) {
    const toRead = Math.min(count, this.size);
    for (let i = 0; i < toRead; i++) {
      target[offset + i] = this.buffer[this.readIndex];
      this.readIndex = this.readIndex + 1 === this.capacity ? 0 : this.readIndex + 1;
    }
    this.size -= toRead;
    return toRead;
  }

  /** Drop all queued samples and reset the pointers. */
  clear() {
    this.writeIndex = 0;
    this.readIndex = 0;
    this.size = 0;
  }
}

class DecibriOutputProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    const capacity = opts.ringCapacity ?? 96000;
    this.ring = new RingBuffer(capacity);
    // Latches a single 'drained' notification per playout: set true when audio
    // is queued, cleared (and the notification sent) when the ring next empties.
    this.hadData = false;
    this.port.onmessage = (event) => this._onmessage(event);
  }

  _onmessage(event) {
    const data = event.data;

    // Control message: drop the queue (stop / reset). No drain notification is
    // sent for a flush; a flush is an explicit stop, not a natural playout end.
    if (data && data.type === 'flush') {
      this.ring.clear();
      this.hadData = false;
      return;
    }

    // Data message: a raw Float32 sample buffer (transferable), mirroring the
    // capture worklet's raw-buffer format in reverse. The samples are already
    // at the context sample rate; the main thread does any resample and dtype
    // conversion before feeding.
    if (data instanceof ArrayBuffer) {
      const samples = new Float32Array(data);
      const accepted = this.ring.write(samples);
      if (accepted > 0) this.hadData = true;
      // Backpressure ack: report the new fill level so the producer can pace
      // its writes, and signal drops (accepted < requested means the ring was
      // full and the surplus was discarded).
      this.port.postMessage({
        type: 'level',
        queued: this.ring.availableRead,
        capacity: this.ring.capacity,
        accepted,
        requested: samples.length,
      });
    }
  }

  process(_inputs, outputs, _parameters) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const frames = output[0].length;
    const got = this.ring.readInto(output[0], 0, frames);

    // Underrun: fill the rest of the first channel with silence.
    for (let i = got; i < frames; i++) output[0][i] = 0;

    // Fan the mono stream out to any additional output channels.
    for (let ch = 1; ch < output.length; ch++) {
      output[ch].set(output[0]);
    }

    // Drain detection: notify the main thread once when the queue empties
    // after having held audio, so the Speaker's drain() can resolve.
    if (this.hadData && this.ring.isEmpty) {
      this.hadData = false;
      this.port.postMessage({ type: 'drained' });
    }

    // Keep the processor alive across underruns so playback can resume.
    return true;
  }
}

// Exposed so the test harness can exercise the ring buffer in isolation. The
// runtime worklet never reads this property; it is a harmless static.
DecibriOutputProcessor.RingBuffer = RingBuffer;

registerProcessor('decibri-output-processor', DecibriOutputProcessor);
