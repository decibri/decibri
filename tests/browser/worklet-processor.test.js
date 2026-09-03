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
    channels: opts.channels ?? 1,
    channelMap: opts.channelMap ?? null,
  };
  return new registeredProcessor.ctor({ processorOptions });
}

// ── Resampler drive helpers ──────────────────────────────────────────────────

// The rate pairs the resampler's cadence is pinned at: the context rates
// browsers run at against the common capture targets, the fractional pairs
// alongside the pairs whose ratio is an integer.
const RATE_PAIRS = [
  [48000, 16000],
  [44100, 16000],
  [48000, 24000],
  [44100, 48000],
  [48000, 44100],
  [96000, 16000],
];

// The frames a linear-interpolation resampler delivers from n input samples
// at a pair: frame k sits at k * native / target input samples and needs the
// input sample after its floor, so frame k exists while
// k * native < (n - 1) * target. Integer arithmetic throughout.
function expectedFrames(n, native, target) {
  return Math.ceil(((n - 1) * target) / native);
}

// A processor that flushes every frame on its own, with a port that tallies
// the posted frames and optionally keeps their samples, so a drive's whole
// output is observable without reading the processor's state.
function createDrivenProcessor(native, target, channels = 1, keep = false) {
  const proc = createProcessor({
    framesPerBuffer: 1,
    format: 'float32',
    nativeSampleRate: native,
    targetSampleRate: target,
    channels,
  });
  const out = { frames: 0, chunks: [] };
  proc.port = {
    postMessage: (buffer) => {
      out.frames++;
      if (keep) out.chunks.push(new Float32Array(buffer));
    },
  };
  return { proc, out };
}

// The kept samples of a drive, concatenated in delivery order.
function delivered(out) {
  const all = new Float32Array(out.chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const chunk of out.chunks) {
    all.set(chunk, offset);
    offset += chunk.length;
  }
  return all;
}

// A ramp of n samples whose value is the sample's index from `start`, scaled
// by 2^-10 so every value is exact in single precision. A linear
// interpolator reproduces a ramp exactly at every position, so each
// delivered sample's value is the input time it was taken at.
function ramp(start, n) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (start + i) / 1024;
  return a;
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
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

  it('carries the phase across blocks and interpolates the frame that straddles them', () => {
    // Blocks of 10 at 3:1: the frames at 0, 3 and 6 leave the next at 9,
    // which needs sample 10, the next block's first, so its position is
    // carried as -1 and it interpolates between the stored sample 9 and that
    // block's sample 10. The ramp makes every frame's value its input time,
    // so 120 samples deliver exactly 0, 3, 6, ..., 117.
    const proc = createProcessor({ framesPerBuffer: 4, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 16000 });
    for (let b = 0; b < 12; b++) {
      const block = new Float32Array(10);
      for (let i = 0; i < 10; i++) block[i] = 10 * b + i;
      proc.process([[block]], [[]], {});
    }
    expect(proc.port.postMessage).toHaveBeenCalledTimes(10);
    const values = [];
    for (const call of proc.port.postMessage.mock.calls) values.push(...new Float32Array(call[0]));
    const expected = [];
    for (let k = 0; k < 40; k++) expected.push(3 * k);
    expect(values).toEqual(expected);
  });

  it('preserves the frequency of a sine through the 3:1 downsample', () => {
    // 1 kHz at 48 kHz is 48 samples a cycle; at 16 kHz it is 16, so the
    // first peak lands on delivered frame 4.
    const proc = createProcessor({ framesPerBuffer: 160, format: 'float32', nativeSampleRate: 48000, targetSampleRate: 16000 });
    const input = new Float32Array(480);
    for (let i = 0; i < 480; i++) input[i] = Math.sin((2 * Math.PI * 1000 * i) / 48000);
    proc.process([[input]], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const output = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(output.length).toBe(160);
    let peak = 0;
    for (let i = 1; i < 16; i++) if (output[i] > output[peak]) peak = i;
    expect(peak).toBe(4);
  });
});

describe('DecibriProcessor resampler cadence', () => {
  it('delivers the exact frame count over a long drive at every rate pair', () => {
    // 37,500 render quanta of 128 frames: 100 seconds of capture at 48 kHz.
    // The count is exact at every pair, the integer ratios included, so a
    // deficit of even one frame per block boundary fails here.
    const quanta = 37500;
    const frames = 128;
    const block = new Float32Array(frames);
    for (const [native, target] of RATE_PAIRS) {
      const { proc, out } = createDrivenProcessor(native, target);
      for (let q = 0; q < quanta; q++) proc.process([[block]], [[]], {});
      expect(out.frames, `${native} -> ${target}`).toBe(expectedFrames(quanta * frames, native, target));
    }
  }, 60000);

  it('delivers the same frames whole and in blocks of any size', () => {
    // Chunking moves frames between blocks, never the total and never the
    // samples: the input fed in one call and fed in blocks, of the render
    // quantum, of single samples, of sizes that never divide the whole and
    // with empty blocks between, delivers the same frames.
    const n = 100000;
    const input = ramp(0, n);
    const strategies = [[128], [1], [7], [64], [3, 1, 64, 0, 7, 129, 13]];
    for (const [native, target] of RATE_PAIRS) {
      const whole = createDrivenProcessor(native, target, 1, true);
      whole.proc.process([[input]], [[]], {});
      expect(whole.out.frames, `${native} -> ${target} whole`).toBe(expectedFrames(n, native, target));
      const wholeSamples = delivered(whole.out);
      for (const sizes of strategies) {
        const chunked = createDrivenProcessor(native, target, 1, true);
        let fed = 0;
        for (let s = 0; fed < n; s++) {
          const end = Math.min(fed + sizes[s % sizes.length], n);
          chunked.proc.process([[input.subarray(fed, end)]], [[]], {});
          fed = end;
        }
        const label = `${native} -> ${target} in blocks ${JSON.stringify(sizes)}`;
        expect(chunked.out.frames, label).toBe(whole.out.frames);
        expect(maxAbsDiff(delivered(chunked.out), wholeSamples), label).toBeLessThan(1e-4);
      }
    }
  });

  it('delivers every frame at its exact input time', () => {
    // A ramp through the resampler is the identity from time to value:
    // delivered frame k must carry the value at k * ratio input samples. A
    // frame taken one input sample late, an interpolation weighted the wrong
    // way round, or a boundary frame taken from the wrong pair of samples
    // all move the value by up to one input sample, 2^-10 here, against a
    // single-precision resolution of about 3e-5 at the ramp's top.
    const quanta = 2000;
    const frames = 128;
    for (const [native, target] of RATE_PAIRS) {
      const ratio = native / target;
      const { proc, out } = createDrivenProcessor(native, target, 1, true);
      for (let q = 0; q < quanta; q++) proc.process([[ramp(q * frames, frames)]], [[]], {});
      const samples = delivered(out);
      expect(samples.length, `${native} -> ${target}`).toBe(expectedFrames(quanta * frames, native, target));
      let worst = 0;
      for (let k = 0; k < samples.length; k++) {
        worst = Math.max(worst, Math.abs(samples[k] - (k * ratio) / 1024));
      }
      expect(worst, `${native} -> ${target}`).toBeLessThan(1e-4);
    }
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

describe('DecibriProcessor multichannel delivery', () => {
  it('delivers two channels interleaved frame by frame', () => {
    // The engine's select vectors: channel 0 counts up, channel 1 counts
    // down. A stride slip or a channel swap produces plausible audio; the
    // interleaved order pins both.
    const proc = createProcessor({ framesPerBuffer: 4, channels: 2 });
    proc.process(
      [[new Float32Array([1.0, 2.0, 3.0, 4.0]), new Float32Array([-1.0, -2.0, -3.0, -4.0])]],
      [[]],
      {}
    );
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(Array.from(result)).toEqual([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);
  });

  it('delivers every granted channel in granted order at the identity', () => {
    const proc = createProcessor({ framesPerBuffer: 2, channels: 3 });
    proc.process(
      [[new Float32Array([0.1, 0.2]), new Float32Array([0.3, 0.4]), new Float32Array([0.5, 0.6])]],
      [[]],
      {}
    );
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(result.length).toBe(6);
    expect(result[0]).toBeCloseTo(0.1);
    expect(result[1]).toBeCloseTo(0.3);
    expect(result[2]).toBeCloseTo(0.5);
    expect(result[3]).toBeCloseTo(0.2);
    expect(result[4]).toBeCloseTo(0.4);
    expect(result[5]).toBeCloseTo(0.6);
  });

  it('delivers a channel count bounded only by the grant', () => {
    // The negative control for the no-fixed-maximum rule: 40 delivered
    // channels, well above the Web Audio 32 floor. A fixed ceiling added to
    // the delivery later fails loudly here.
    const channels = [];
    for (let c = 0; c < 40; c++) channels.push(new Float32Array(2).fill(c));
    const proc = createProcessor({ framesPerBuffer: 2, channels: 40 });
    proc.process([channels], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(result.length).toBe(80);
    for (let c = 0; c < 40; c++) {
      expect(result[c]).toBe(c);
      expect(result[40 + c]).toBe(c);
    }
  });

  it('gathers the named channels in map order at width', () => {
    // The map both selects and permutes: [3, 1] on six granted channels
    // delivers channel 3 then channel 1 of every frame, in that order.
    const granted = [];
    for (let c = 0; c < 6; c++) granted.push(new Float32Array([c + 0.5, -(c + 0.5)]));
    const proc = createProcessor({ framesPerBuffer: 2, channelMap: [3, 1] });
    proc.process([granted], [[]], {});
    const result = new Float32Array(proc.port.postMessage.mock.calls[0][0]);
    expect(Array.from(result)).toEqual([3.5, 1.5, -3.5, -1.5]);
  });

  it('a map may deliver more channels than the grant carries', () => {
    // The deliberate asymmetry: entries are checked one at a time and may
    // repeat, so [0, 0, 0, 0] on a mono grant is four copies of its one
    // channel, while the bare count 4 is refused below.
    const proc = createProcessor({ framesPerBuffer: 1, channelMap: [0, 0, 0, 0] });
    proc.process([[new Float32Array([0.25, -0.5])]], [[]], {});
    expect(proc.port.postMessage).toHaveBeenCalledTimes(2);
    expect(Array.from(new Float32Array(proc.port.postMessage.mock.calls[0][0])))
      .toEqual([0.25, 0.25, 0.25, 0.25]);
    expect(Array.from(new Float32Array(proc.port.postMessage.mock.calls[1][0])))
      .toEqual([-0.5, -0.5, -0.5, -0.5]);
  });

  it('refuses an unmapped strict subset once, with the engine message, and stops', () => {
    const six = () => {
      const chans = [];
      for (let c = 0; c < 6; c++) chans.push(new Float32Array([0.1, 0.2]));
      return chans;
    };
    const proc = createProcessor({ framesPerBuffer: 2, channels: 2 });
    expect(proc.process([six()], [[]], {})).toBe(false);
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    expect(proc.port.postMessage.mock.calls[0][0]).toEqual({
      type: 'error',
      message: "a channel map is required to deliver 2 of the device's 6 input channels",
    });

    // The failure is terminal: no second report and no audio.
    expect(proc.process([six()], [[]], {})).toBe(false);
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
  });

  it('refuses a count above the grant once, with the engine message, and stops', () => {
    const stereo = () => [new Float32Array([0.1, 0.2]), new Float32Array([0.3, 0.4])];
    const proc = createProcessor({ framesPerBuffer: 2, channels: 4 });
    expect(proc.process([stereo()], [[]], {})).toBe(false);
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
    expect(proc.port.postMessage.mock.calls[0][0]).toEqual({
      type: 'error',
      message: 'the input device does not support 4 delivered channels; it reports 2',
    });
    expect(proc.process([stereo()], [[]], {})).toBe(false);
    expect(proc.port.postMessage).toHaveBeenCalledTimes(1);
  });

  it('resamples every channel in lockstep with the mono path', () => {
    // Each delivered channel of a stereo 3:1 downsample must equal the mono
    // downsample of the same signal, sample for sample: one shared position
    // drives every channel, so a frame stays a frame across the rate change.
    const twelve = () => {
      const a = new Float32Array(12);
      for (let i = 0; i < 12; i++) a[i] = i / 11;
      return a;
    };
    const mono = createProcessor({ framesPerBuffer: 4, nativeSampleRate: 48000, targetSampleRate: 16000 });
    mono.process([[twelve()]], [[]], {});
    const monoOut = Array.from(new Float32Array(mono.port.postMessage.mock.calls[0][0]));

    const stereo = createProcessor({ framesPerBuffer: 4, channels: 2, nativeSampleRate: 48000, targetSampleRate: 16000 });
    stereo.process([[twelve(), twelve()]], [[]], {});
    expect(stereo.port.postMessage).toHaveBeenCalledTimes(1);
    const interleaved = Array.from(new Float32Array(stereo.port.postMessage.mock.calls[0][0]));
    expect(interleaved.length).toBe(8);
    const left = interleaved.filter((_, i) => i % 2 === 0);
    const right = interleaved.filter((_, i) => i % 2 === 1);
    expect(left).toEqual(monoOut);
    expect(right).toEqual(monoOut);

    // Across blocks at a fractional ratio, with different content per
    // channel: the stereo drive's left channel equals the mono drive of the
    // left content and its right the mono drive of the right, frame for
    // frame, so both channels cross every block boundary at one position.
    const quanta = 500;
    const frames = 128;
    const leftBlock = (q) => ramp(q * frames, frames);
    const rightBlock = (q) => {
      const r = ramp(q * frames, frames);
      for (let i = 0; i < frames; i++) r[i] = -r[i];
      return r;
    };
    const monoLeft = createDrivenProcessor(44100, 16000, 1, true);
    const monoRight = createDrivenProcessor(44100, 16000, 1, true);
    const stereoDrive = createDrivenProcessor(44100, 16000, 2, true);
    for (let q = 0; q < quanta; q++) {
      monoLeft.proc.process([[leftBlock(q)]], [[]], {});
      monoRight.proc.process([[rightBlock(q)]], [[]], {});
      stereoDrive.proc.process([[leftBlock(q), rightBlock(q)]], [[]], {});
    }
    expect(stereoDrive.out.frames).toBe(expectedFrames(quanta * frames, 44100, 16000));
    expect(monoLeft.out.frames).toBe(stereoDrive.out.frames);
    expect(monoRight.out.frames).toBe(stereoDrive.out.frames);
    const driven = delivered(stereoDrive.out);
    const l = delivered(monoLeft.out);
    const r = delivered(monoRight.out);
    let mismatches = 0;
    for (let k = 0; k < stereoDrive.out.frames; k++) {
      if (driven[2 * k] !== l[k] || driven[2 * k + 1] !== r[k]) mismatches++;
    }
    expect(mismatches).toBe(0);
  });

  it('converts interleaved channels to int16', () => {
    const proc = createProcessor({ framesPerBuffer: 2, channels: 2, format: 'int16' });
    proc.process([[new Float32Array([0.5, -0.5]), new Float32Array([-0.5, 0.5])]], [[]], {});
    const result = new Int16Array(proc.port.postMessage.mock.calls[0][0]);
    expect(Array.from(result)).toEqual([16384, -16384, -16384, 16384]);
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
        channels: 1,
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

  it('the minified runtime string delivers and refuses channel counts like the source', () => {
    const { ctor } = evaluateInline();
    const mk = (opts) => new ctor({
      processorOptions: {
        framesPerBuffer: 2,
        format: 'float32',
        nativeSampleRate: 48000,
        targetSampleRate: 48000,
        channels: 1,
        channelMap: null,
        ...opts,
      },
    });

    const identity = mk({ channels: 2 });
    identity.process([[new Float32Array([1.0, 2.0]), new Float32Array([-1.0, -2.0])]], [[]], {});
    expect(Array.from(new Float32Array(identity.port.postMessage.mock.calls[0][0])))
      .toEqual([1.0, -1.0, 2.0, -2.0]);

    const wide = mk({ channelMap: [1, 0] });
    wide.process([[new Float32Array([0.25, 0.5]), new Float32Array([0.75, -0.25])]], [[]], {});
    expect(Array.from(new Float32Array(wide.port.postMessage.mock.calls[0][0])))
      .toEqual([0.75, 0.25, -0.25, 0.5]);

    const subset = mk({ channels: 2 });
    const six = [];
    for (let c = 0; c < 6; c++) six.push(new Float32Array([0.1, 0.2]));
    expect(subset.process([six], [[]], {})).toBe(false);
    expect(subset.port.postMessage.mock.calls[0][0]).toEqual({
      type: 'error',
      message: "a channel map is required to deliver 2 of the device's 6 input channels",
    });

    const above = mk({ channels: 4 });
    expect(above.process([[new Float32Array([0.1]), new Float32Array([0.2])]], [[]], {})).toBe(false);
    expect(above.port.postMessage.mock.calls[0][0]).toEqual({
      type: 'error',
      message: 'the input device does not support 4 delivered channels; it reports 2',
    });
  });

  it('the minified runtime string carries the resampler phase across blocks like the source', () => {
    // 300 blocks of 10 at 3:1 cross a boundary mid-phase on every block. The
    // count is the analytic one and the samples are the source's own.
    const { ctor } = evaluateInline();
    const blocks = 300;
    const inline = new ctor({
      processorOptions: {
        framesPerBuffer: 1,
        format: 'float32',
        nativeSampleRate: 48000,
        targetSampleRate: 16000,
        channels: 1,
        channelMap: null,
      },
    });
    const inlineOut = { frames: 0, chunks: [] };
    inline.port = {
      postMessage: (buffer) => {
        inlineOut.frames++;
        inlineOut.chunks.push(new Float32Array(buffer));
      },
    };
    const source = createDrivenProcessor(48000, 16000, 1, true);
    for (let b = 0; b < blocks; b++) {
      inline.process([[ramp(b * 10, 10)]], [[]], {});
      source.proc.process([[ramp(b * 10, 10)]], [[]], {});
    }
    expect(inlineOut.frames).toBe(expectedFrames(blocks * 10, 48000, 16000));
    expect(Array.from(delivered(inlineOut))).toEqual(Array.from(delivered(source.out)));
  });
});
