import { describe, it, expect, vi, beforeAll } from 'vitest';

// ── Mock AudioWorkletProcessor global ────────────────────────────────────────

let registeredProcessor = null;

class MockAudioWorkletProcessor {
  constructor() {
    this.port = { postMessage: vi.fn() };
  }
}

globalThis.AudioWorkletProcessor = MockAudioWorkletProcessor;
globalThis.registerProcessor = (name, ctor) => {
  registeredProcessor = { name, ctor };
};

// Import after mocks are in place (triggers registerProcessor)
await import('../../npm/decibri/src/browser/worklet-processor.js');

function createProcessor(opts = {}) {
  const processorOptions = {
    framesPerBuffer: opts.framesPerBuffer ?? 4,
    format: opts.format ?? 'float32',
    nativeSampleRate: opts.nativeSampleRate ?? 48000,
    targetSampleRate: opts.targetSampleRate ?? 48000,
  };
  return new registeredProcessor.ctor({ processorOptions });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('DecibriProcessor registration', () => {
  it('registers as decibri-processor', () => {
    expect(registeredProcessor.name).toBe('decibri-processor');
  });
});

describe('DecibriProcessor accumulation', () => {
  it('accumulates frames and flushes when framesPerBuffer is reached', () => {
    const proc = createProcessor({ framesPerBuffer: 4, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 48000 });

    proc.process([[new Float32Array([0.1, 0.2])]], [[]], {});
    expect(proc.port.postMessage).not.toHaveBeenCalled();

    proc.process([[new Float32Array([0.3, 0.4])]], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);

    const buffer = proc.port.postMessage.mock.calls[0][0];
    const result = new Float32Array(buffer);
    expect(result.length).toBe(4);
    expect(result[0]).toBeCloseTo(0.1);
    expect(result[1]).toBeCloseTo(0.2);
    expect(result[2]).toBeCloseTo(0.3);
    expect(result[3]).toBeCloseTo(0.4);
  });

  it('handles input larger than framesPerBuffer', () => {
    const proc = createProcessor({ framesPerBuffer: 2, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 48000 });

    proc.process([[new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5])]], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(2);
  });

  it('returns true to keep processor alive', () => {
    const proc = createProcessor({});
    const result = proc.process([[new Float32Array([0.1])]], [[]], {});
    expect(result).toBe(true);
  });

  it('handles empty input gracefully', () => {
    const proc = createProcessor({});
    expect(proc.process([[]], [[]], {})).toBe(true);
    expect(proc.process([], [[]], {})).toBe(true);
  });
});

describe('DecibriProcessor int16 conversion', () => {
  it('converts float32 to int16 correctly', () => {
    const proc = createProcessor({ framesPerBuffer: 3, format: 'int16', nativeSampleRate: 48000, targetSampleRate: 48000 });

    proc.process([[new Float32Array([0.5, -0.5, 0.0])]], [[]], {});

    const buffer = proc.port.postMessage.mock.calls[0][0];
    const result = new Int16Array(buffer);
    expect(result.length).toBe(3);
    expect(result[0]).toBe(Math.round(0.5 * 32768));
    expect(result[1]).toBe(Math.round(-0.5 * 32768));
    expect(result[2]).toBe(0);
  });

  it('clamps at int16 boundaries', () => {
    const proc = createProcessor({ framesPerBuffer: 2, format: 'int16', nativeSampleRate: 48000, targetSampleRate: 48000 });

    proc.process([[new Float32Array([1.5, -1.5])]], [[]], {});

    const buffer = proc.port.postMessage.mock.calls[0][0];
    const result = new Int16Array(buffer);
    expect(result[0]).toBe(32767);
    expect(result[1]).toBe(-32768);
  });
});

describe('DecibriProcessor resampling', () => {
  it('passes through when rates match', () => {
    const proc = createProcessor({ framesPerBuffer: 4, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 48000 });

    const input = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    proc.process([[input]], [[]], {});

    const buffer = proc.port.postMessage.mock.calls[0][0];
    const result = new Float32Array(buffer);
    expect(result[0]).toBeCloseTo(0.1);
    expect(result[3]).toBeCloseTo(0.4);
  });

  it('downsamples 48000 -> 16000 (3:1 ratio)', () => {
    const proc = createProcessor({ framesPerBuffer: 4, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 16000 });

    const input = new Float32Array(12);
    for (let i = 0; i < 12; i++) input[i] = i / 11;

    proc.process([[input]], [[]], {});

    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const buffer = proc.port.postMessage.mock.calls[0][0];
    const result = new Float32Array(buffer);
    expect(result.length).toBe(4);

    expect(result[0]).toBeCloseTo(0 / 11, 4);
    expect(result[1]).toBeCloseTo(3 / 11, 4);
    expect(result[2]).toBeCloseTo(6 / 11, 4);
    expect(result[3]).toBeCloseTo(9 / 11, 4);
  });

  it('maintains continuity across multiple process() calls', () => {
    const proc = createProcessor({ framesPerBuffer: 8, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 16000 });

    const chunk1 = new Float32Array(24);
    const chunk2 = new Float32Array(24);
    for (let i = 0; i < 24; i++) {
      chunk1[i] = Math.sin(2 * Math.PI * 1000 * i / 48000);
      chunk2[i] = Math.sin(2 * Math.PI * 1000 * (i + 24) / 48000);
    }

    proc.process([[chunk1]], [[]], {});
    proc.process([[chunk2]], [[]], {});

    expect(proc.port.postMessage).toHaveBeenCalled();
  });
});

describe('DecibriProcessor uses transferable', () => {
  it('posts ArrayBuffer as transferable', () => {
    const proc = createProcessor({ framesPerBuffer: 2, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 48000 });

    proc.process([[new Float32Array([0.1, 0.2])]], [[]], {});

    const call = proc.port.postMessage.mock.calls[0];
    expect(call[1]).toBeInstanceOf(Array);
    expect(call[1][0]).toBe(call[0]);
  });
});
