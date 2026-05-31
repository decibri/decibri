import { describe, it, expect, vi } from 'vitest';

// ── Mock AudioWorkletProcessor global ────────────────────────────────────────

let registeredProcessor = null;

class MockAudioWorkletProcessor {
  constructor() {
    this.port = { postMessage: vi.fn(), onmessage: null };
  }
}

globalThis.AudioWorkletProcessor = MockAudioWorkletProcessor;
globalThis.registerProcessor = (name, ctor) => {
  registeredProcessor = { name, ctor };
};

// Import after mocks are in place (triggers registerProcessor)
await import('../../npm/decibri/src/browser/output-worklet-processor.js');
const inlineModule = await import('../../npm/decibri/src/browser/output-worklet-inline.js');
const OUTPUT_WORKLET_SOURCE = inlineModule.OUTPUT_WORKLET_SOURCE || inlineModule.default?.OUTPUT_WORKLET_SOURCE;

const RingBuffer = registeredProcessor.ctor.RingBuffer;

function createProcessor(opts = {}) {
  return new registeredProcessor.ctor({ processorOptions: opts });
}

// Build an `outputs` argument: a single output with `channels` Float32Arrays of
// `frames` length each. Web Audio shape is outputs[outputIndex][channel].
function makeOutputs(channels, frames) {
  const out = [];
  for (let c = 0; c < channels; c++) out.push(new Float32Array(frames));
  return [out];
}

// ── RingBuffer ───────────────────────────────────────────────────────────────

describe('RingBuffer basics', () => {
  it('starts empty', () => {
    const rb = new RingBuffer(4);
    expect(rb.isEmpty).toBe(true);
    expect(rb.isFull).toBe(false);
    expect(rb.availableRead).toBe(0);
    expect(rb.availableWrite).toBe(4);
  });

  it('write then read returns samples in order', () => {
    const rb = new RingBuffer(8);
    const written = rb.write(new Float32Array([0.1, 0.2, 0.3]));
    expect(written).toBe(3);
    expect(rb.availableRead).toBe(3);
    expect(rb.availableWrite).toBe(5);

    const out = new Float32Array(3);
    const read = rb.readInto(out, 0, 3);
    expect(read).toBe(3);
    expect(out[0]).toBeCloseTo(0.1);
    expect(out[1]).toBeCloseTo(0.2);
    expect(out[2]).toBeCloseTo(0.3);
    expect(rb.isEmpty).toBe(true);
  });

  it('reports full and accepts only what fits on overflow', () => {
    const rb = new RingBuffer(4);
    const written = rb.write(new Float32Array([1, 2, 3, 4, 5, 6]));
    expect(written).toBe(4);
    expect(rb.isFull).toBe(true);
    expect(rb.availableWrite).toBe(0);

    // A further write into a full buffer accepts nothing.
    expect(rb.write(new Float32Array([7, 8]))).toBe(0);
  });

  it('reads only what is available on underrun and leaves the rest untouched', () => {
    const rb = new RingBuffer(8);
    rb.write(new Float32Array([0.5, 0.6]));

    const out = new Float32Array(5).fill(-1);
    const read = rb.readInto(out, 0, 5);
    expect(read).toBe(2);
    expect(out[0]).toBeCloseTo(0.5);
    expect(out[1]).toBeCloseTo(0.6);
    // Positions beyond the real samples are untouched (caller fills silence).
    expect(out[2]).toBe(-1);
    expect(out[3]).toBe(-1);
    expect(out[4]).toBe(-1);
  });

  it('reads into target at an offset', () => {
    const rb = new RingBuffer(8);
    rb.write(new Float32Array([1, 2]));

    const out = new Float32Array(4).fill(0);
    const read = rb.readInto(out, 2, 2);
    expect(read).toBe(2);
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(0);
    expect(out[2]).toBe(1);
    expect(out[3]).toBe(2);
  });

  it('wraps around the capacity correctly', () => {
    const rb = new RingBuffer(4);

    // Fill 3, drain 2 to advance the read pointer.
    rb.write(new Float32Array([1, 2, 3]));
    const drain = new Float32Array(2);
    rb.readInto(drain, 0, 2);
    expect(rb.availableRead).toBe(1); // [3] remains, read/write indices advanced

    // Write 3 more; this must wrap past the end of the backing array.
    const written = rb.write(new Float32Array([4, 5, 6]));
    expect(written).toBe(3);
    expect(rb.availableRead).toBe(4);
    expect(rb.isFull).toBe(true);

    // Read all 4 back: must come out in FIFO order across the wrap.
    const out = new Float32Array(4);
    expect(rb.readInto(out, 0, 4)).toBe(4);
    expect(Array.from(out)).toEqual([3, 4, 5, 6]);
    expect(rb.isEmpty).toBe(true);
  });

  it('clear() drops all queued samples', () => {
    const rb = new RingBuffer(4);
    rb.write(new Float32Array([1, 2, 3]));
    rb.clear();
    expect(rb.isEmpty).toBe(true);
    expect(rb.availableRead).toBe(0);
    expect(rb.availableWrite).toBe(4);
  });
});

// ── Processor registration ─────────────────────────────────────────────────

describe('DecibriOutputProcessor registration', () => {
  it('registers as decibri-output-processor', () => {
    expect(registeredProcessor.name).toBe('decibri-output-processor');
  });
});

// ── Message-port protocol: feeding samples ───────────────────────────────────

describe('DecibriOutputProcessor feed protocol', () => {
  it('enqueues a raw Float32 ArrayBuffer fed over the port', () => {
    const proc = createProcessor();
    const samples = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    proc.port.onmessage({ data: samples.buffer });

    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {});

    const ch = outputs[0][0];
    expect(ch[0]).toBeCloseTo(0.1);
    expect(ch[1]).toBeCloseTo(0.2);
    expect(ch[2]).toBeCloseTo(0.3);
    expect(ch[3]).toBeCloseTo(0.4);
  });

  it('delivers samples in order across multiple feeds', () => {
    const proc = createProcessor();
    proc.port.onmessage({ data: new Float32Array([1, 2]).buffer });
    proc.port.onmessage({ data: new Float32Array([3, 4]).buffer });

    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {});
    expect(Array.from(outputs[0][0])).toEqual([1, 2, 3, 4]);
  });

  it('acks each feed with a level message', () => {
    const proc = createProcessor({ ringCapacity: 100 });
    proc.port.onmessage({ data: new Float32Array([1, 2, 3]).buffer });

    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const msg = proc.port.postMessage.mock.calls[0][0];
    expect(msg).toEqual({
      type: 'level',
      queued: 3,
      capacity: 100,
      accepted: 3,
      requested: 3,
    });
  });

  it('reports dropped samples when the ring overflows (backpressure signal)', () => {
    const proc = createProcessor({ ringCapacity: 2 });
    proc.port.onmessage({ data: new Float32Array([1, 2, 3, 4]).buffer });

    const msg = proc.port.postMessage.mock.calls[0][0];
    expect(msg.accepted).toBe(2);
    expect(msg.requested).toBe(4);
    expect(msg.queued).toBe(2);
    expect(msg.capacity).toBe(2);
  });
});

// ── Underrun: silence, no throw ──────────────────────────────────────────────

describe('DecibriOutputProcessor underrun', () => {
  it('outputs pure silence when nothing has been fed', () => {
    const proc = createProcessor();
    const outputs = makeOutputs(1, 8);
    const result = proc.process([], outputs, {});

    expect(result).toBe(true);
    expect(Array.from(outputs[0][0])).toEqual(new Array(8).fill(0));
  });

  it('fills the tail with silence on partial underrun', () => {
    const proc = createProcessor();
    proc.port.onmessage({ data: new Float32Array([0.5, 0.6]).buffer });

    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {});

    const ch = outputs[0][0];
    expect(ch[0]).toBeCloseTo(0.5);
    expect(ch[1]).toBeCloseTo(0.6);
    expect(ch[2]).toBe(0);
    expect(ch[3]).toBe(0);
  });

  it('does not throw and returns true when output has no channels', () => {
    const proc = createProcessor();
    expect(proc.process([], [[]], {})).toBe(true);
    expect(proc.process([], [], {})).toBe(true);
  });

  it('keeps the processor alive (returns true) across underruns', () => {
    const proc = createProcessor();
    const outputs = makeOutputs(1, 4);
    expect(proc.process([], outputs, {})).toBe(true);
    expect(proc.process([], outputs, {})).toBe(true);
  });
});

// ── Multi-channel fan-out ────────────────────────────────────────────────────

describe('DecibriOutputProcessor multi-channel', () => {
  it('fans the mono stream out to every output channel', () => {
    const proc = createProcessor();
    proc.port.onmessage({ data: new Float32Array([0.1, 0.2, 0.3, 0.4]).buffer });

    const outputs = makeOutputs(2, 4);
    proc.process([], outputs, {});

    expect(Array.from(outputs[0][0])).toEqual(Array.from(outputs[0][1]));
    expect(outputs[0][1][0]).toBeCloseTo(0.1);
    expect(outputs[0][1][3]).toBeCloseTo(0.4);
  });
});

// ── Drain detection ──────────────────────────────────────────────────────────

describe('DecibriOutputProcessor drain detection', () => {
  it('emits drained exactly once when the queue empties after playout', () => {
    const proc = createProcessor();
    proc.port.onmessage({ data: new Float32Array([1, 2, 3, 4]).buffer });
    proc.port.postMessage.mockClear(); // drop the feed-ack level message

    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {}); // reads all 4 -> empties -> drained

    const drained = proc.port.postMessage.mock.calls.filter(c => c[0]?.type === 'drained');
    expect(drained).toHaveLength(1);

    // Further idle process() calls do not re-emit drained.
    proc.process([], outputs, {});
    proc.process([], outputs, {});
    const drainedAfter = proc.port.postMessage.mock.calls.filter(c => c[0]?.type === 'drained');
    expect(drainedAfter).toHaveLength(1);
  });

  it('does not emit drained before any audio is fed', () => {
    const proc = createProcessor();
    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {});
    const drained = proc.port.postMessage.mock.calls.filter(c => c[0]?.type === 'drained');
    expect(drained).toHaveLength(0);
  });

  it('re-arms drain detection for a second playout', () => {
    const proc = createProcessor();
    const outputs = makeOutputs(1, 4);

    proc.port.onmessage({ data: new Float32Array([1, 2]).buffer });
    proc.process([], outputs, {}); // drains -> drained #1

    proc.port.onmessage({ data: new Float32Array([3, 4]).buffer });
    proc.process([], outputs, {}); // drains -> drained #2

    const drained = proc.port.postMessage.mock.calls.filter(c => c[0]?.type === 'drained');
    expect(drained).toHaveLength(2);
  });
});

// ── Flush control message ────────────────────────────────────────────────────

describe('DecibriOutputProcessor flush', () => {
  it('clears the queue and outputs silence afterward', () => {
    const proc = createProcessor();
    proc.port.onmessage({ data: new Float32Array([1, 2, 3, 4]).buffer });
    proc.port.onmessage({ data: { type: 'flush' } });

    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {});
    expect(Array.from(outputs[0][0])).toEqual([0, 0, 0, 0]);
  });

  it('does not post a level ack or a drained notification for a flush', () => {
    const proc = createProcessor();
    proc.port.onmessage({ data: new Float32Array([1, 2, 3, 4]).buffer });
    proc.port.postMessage.mockClear();

    proc.port.onmessage({ data: { type: 'flush' } });
    expect(proc.port.postMessage).not.toHaveBeenCalled();

    // The flushed audio must not later surface as a drained event.
    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {});
    const drained = proc.port.postMessage.mock.calls.filter(c => c[0]?.type === 'drained');
    expect(drained).toHaveLength(0);
  });
});

// ── Inline (runtime) source parity ───────────────────────────────────────────

describe('output-worklet-inline parity', () => {
  it('the minified runtime string registers the same processor and ring buffer', () => {
    let captured = null;
    // The inline source references AudioWorkletProcessor and registerProcessor
    // as free globals; supply them as Function parameters.
    const evaluate = new Function('AudioWorkletProcessor', 'registerProcessor', OUTPUT_WORKLET_SOURCE);
    evaluate(MockAudioWorkletProcessor, (name, ctor) => { captured = { name, ctor }; });

    expect(captured.name).toBe('decibri-output-processor');
    expect(typeof captured.ctor.RingBuffer).toBe('function');

    // Ring buffer from the minified source behaves identically.
    const rb = new captured.ctor.RingBuffer(4);
    expect(rb.write(new Float32Array([1, 2, 3]))).toBe(3);
    const out = new Float32Array(3);
    expect(rb.readInto(out, 0, 3)).toBe(3);
    expect(Array.from(out)).toEqual([1, 2, 3]);

    // Processor from the minified source drains the ring and zero-fills.
    const proc = new captured.ctor({ processorOptions: {} });
    proc.port.onmessage({ data: new Float32Array([0.5, 0.6]).buffer });
    const outputs = makeOutputs(1, 4);
    proc.process([], outputs, {});
    expect(outputs[0][0][0]).toBeCloseTo(0.5);
    expect(outputs[0][0][1]).toBeCloseTo(0.6);
    expect(outputs[0][0][2]).toBe(0);
    expect(outputs[0][0][3]).toBe(0);
  });
});
