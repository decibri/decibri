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
const inlineModule = await import('../../npm/decibri/src/browser/worklet-inline.js');
const WORKLET_SOURCE = inlineModule.WORKLET_SOURCE || inlineModule.default?.WORKLET_SOURCE;

function createProcessor(opts = {}) {
  const processorOptions = {
    framesPerBuffer: opts.framesPerBuffer ?? 4,
    format: opts.format ?? 'float32',
    nativeSampleRate: opts.nativeSampleRate ?? 48000,
    targetSampleRate: opts.targetSampleRate ?? 48000,
    channelMap: opts.channelMap ?? null,
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

describe('DecibriProcessor channel gathering (channelMap)', () => {
  it('gathers exactly the named device channel', () => {
    // Two channels with unmistakably different content, the engine's own
    // select vectors: channel 0 counts up from 1.0, channel 1 counts down
    // from -1.0. A gather taking the wrong index or striding wrong produces
    // plausible audio; this pins the delivered samples to the named channel.
    for (const [map, expected] of [
      [[0], [1.0, 2.0, 3.0, 4.0]],
      [[1], [-1.0, -2.0, -3.0, -4.0]],
    ]) {
      const proc = createProcessor({ framesPerBuffer: 4, channelMap: map });
      proc.process(
        [[new Float32Array([1.0, 2.0, 3.0, 4.0]), new Float32Array([-1.0, -2.0, -3.0, -4.0])]],
        [[]],
        {}
      );
      expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
      const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
      expect(Array.from(result)).toEqual(expected);
    }
  });

  it('accepts a map index bounded only by the block channel count', () => {
    // The negative control for the no-fixed-maximum rule: 40 channels, well
    // above the Web Audio 32 floor, with the map naming the last one. A
    // fixed ceiling added to the gather later fails loudly here.
    const channels = [];
    for (let c = 0; c < 40; c++) channels.push(new Float32Array(4).fill(c));
    const proc = createProcessor({ framesPerBuffer: 4, channelMap: [39] });
    proc.process([channels], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(Array.from(result)).toEqual([39, 39, 39, 39]);
  });

  it('gathers the named channel and then converts to int16', () => {
    const proc = createProcessor({ framesPerBuffer: 2, format: 'int16', channelMap: [1] });
    proc.process([[new Float32Array([1.0, 1.0]), new Float32Array([0.5, -0.5])]], [[]], {});
    const result = new Int16Array(proc.port.postMessage.mock.calls[0][0]);
    expect(result[0]).toBe(16384);
    expect(result[1]).toBe(-16384);
  });
});

describe('DecibriProcessor channel average (no map)', () => {
  it('averages every granted channel with the engine arithmetic', () => {
    // The engine's downmix vectors: a [1.0, -1.0] frame averages to 0.0 and
    // a [-0.5, -0.5] frame to -0.5.
    const proc = createProcessor({ framesPerBuffer: 2 });
    proc.process(
      [[new Float32Array([1.0, -0.5]), new Float32Array([-1.0, -0.5])]],
      [[]],
      {}
    );
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(Array.from(result)).toEqual([0.0, -0.5]);
  });

  it('averages a six-channel frame to its arithmetic mean', () => {
    // Frame 0 sums to zero; frame 1 is 0.25 on every channel. Both means are
    // exact in f32, so equality is exact.
    const frame0 = [0.5, 0.25, -0.25, -0.5, 1.0, -1.0];
    const chans = frame0.map(v => new Float32Array([v, 0.25]));
    const proc = createProcessor({ framesPerBuffer: 2 });
    proc.process([chans], [[]], {});
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(result[0]).toBe(0.0);
    expect(result[1]).toBe(0.25);
  });

  it('passes a single granted channel through unchanged', () => {
    const proc = createProcessor({ framesPerBuffer: 4 });
    const input = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    proc.process([[input]], [[]], {});
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(Array.from(result)).toEqual(Array.from(input));
  });

  it('averages before resampling, matching the mono downsample of equal channels', () => {
    // Two identical channels average to the channel itself, so the 3:1
    // downsample must equal the established mono expectation.
    const mk = () => {
      const a = new Float32Array(12);
      for (let i = 0; i < 12; i++) a[i] = i / 11;
      return a;
    };
    const proc = createProcessor({ framesPerBuffer: 4, nativeSampleRate: 48000, targetSampleRate: 16000 });
    proc.process([[mk(), mk()]], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(result.length).toBe(4);
    expect(result[0]).toBeCloseTo(0 / 11, 4);
    expect(result[1]).toBeCloseTo(3 / 11, 4);
    expect(result[2]).toBeCloseTo(6 / 11, 4);
    expect(result[3]).toBeCloseTo(9 / 11, 4);
  });

  it('converts the average to int16', () => {
    const proc = createProcessor({ framesPerBuffer: 2, format: 'int16' });
    proc.process([[new Float32Array([1.0, 0.5]), new Float32Array([0.0, 0.5])]], [[]], {});
    const result = new Int16Array(proc.port.postMessage.mock.calls[0][0]);
    expect(result[0]).toBe(16384);
    expect(result[1]).toBe(16384);
  });
});

describe('DecibriProcessor channel-map failure', () => {
  it('reports a map entry the block cannot serve once, with the engine message, and stops', () => {
    const proc = createProcessor({ framesPerBuffer: 4, channelMap: [2] });
    const stereo = () => [new Float32Array([0.25, 0.5]), new Float32Array([0.75, -0.25])];

    expect(proc.process([stereo()], [[]], {})).toBe(false);
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    expect(proc.port.postMessage.mock.calls[0][0]).toEqual({
      type: 'error',
      message: 'the channel map names device channel 2; the device reports 2 input channels',
    });

    // The failure is terminal: no second report and no audio.
    expect(proc.process([stereo()], [[]], {})).toBe(false);
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
  });

  it('keeps serving a map the block covers', () => {
    const proc = createProcessor({ framesPerBuffer: 2, channelMap: [1] });
    proc.process([[new Float32Array([0.25, 0.5]), new Float32Array([0.75, -0.25])]], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(Array.from(result)).toEqual([0.75, -0.25]);
  });
});

// ── Inline (runtime) source parity ───────────────────────────────────────────

describe('worklet-inline parity', () => {
  function evaluateInline() {
    let captured = null;
    // The inline source references AudioWorkletProcessor and registerProcessor
    // as free globals; supply them as Function parameters.
    const evaluate = new Function('AudioWorkletProcessor', 'registerProcessor', WORKLET_SOURCE);
    evaluate(MockAudioWorkletProcessor, (name, ctor) => { captured = { name, ctor }; });
    return captured;
  }

  it('the minified runtime string registers the same processor', () => {
    expect(evaluateInline().name).toBe('decibri-processor');
  });

  it('the minified runtime string gathers, averages and reports like the source', () => {
    const { ctor } = evaluateInline();
    const mk = (opts) => new ctor({
      processorOptions: {
        framesPerBuffer: 2,
        format: 'float32',
        nativeSampleRate: 48000,
        targetSampleRate: 48000,
        channelMap: null,
        ...opts,
      },
    });

    const gather = mk({ channelMap: [1] });
    gather.process([[new Float32Array([0.25, 0.5]), new Float32Array([0.75, -0.25])]], [[]], {});
    expect(Array.from(new Float32Array(gather.port.postMessage.mock.calls[0][0]))).toEqual([0.75, -0.25]);

    const avg = mk({});
    avg.process([[new Float32Array([1.0, -0.5]), new Float32Array([-1.0, -0.5])]], [[]], {});
    expect(Array.from(new Float32Array(avg.port.postMessage.mock.calls[0][0]))).toEqual([0.0, -0.5]);

    const bad = mk({ channelMap: [2] });
    expect(bad.process([[new Float32Array([0.25]), new Float32Array([0.5])]], [[]], {})).toBe(false);
    expect(bad.port.postMessage.mock.calls[0][0]).toEqual({
      type: 'error',
      message: 'the channel map names device channel 2; the device reports 2 input channels',
    });
  });
});
