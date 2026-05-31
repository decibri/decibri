import { describe, it, expect, vi, beforeEach } from 'vitest';

// ── Web Audio main-thread mocks ──────────────────────────────────────────────

// Controllable audio clock so backpressure (which decays the queue estimate by
// elapsed context time) can be driven deterministically.
const mockTime = { value: 0 };

const mockPort = {
  postMessage: vi.fn(),
  onmessage: null,
  close: vi.fn(),
};
const mockWorkletNode = {
  port: mockPort,
  connect: vi.fn(),
  disconnect: vi.fn(),
};

let capturedWorkletOptions = null;
const MockAudioWorkletNode = vi.fn().mockImplementation(function (_ctx, _name, options) {
  capturedWorkletOptions = options;
  return mockWorkletNode;
});

const mockDestination = {};
const mockAddModule = vi.fn().mockResolvedValue(undefined);
const mockContextClose = vi.fn().mockResolvedValue(undefined);
let contextState = 'suspended';
const mockContextResume = vi.fn().mockImplementation(async () => { contextState = 'running'; });

const MockAudioContext = vi.fn().mockImplementation(function () {
  return {
    sampleRate: 48000,
    get state() { return contextState; },
    get currentTime() { return mockTime.value; },
    resume: mockContextResume,
    close: mockContextClose,
    audioWorklet: { addModule: mockAddModule },
    destination: mockDestination,
  };
});

vi.stubGlobal('AudioContext', MockAudioContext);
vi.stubGlobal('AudioWorkletNode', MockAudioWorkletNode);
vi.stubGlobal('Blob', class MockBlob {
  constructor(parts, options) { this.parts = parts; this.options = options; }
});
const OriginalURL = globalThis.URL;
const MockURL = Object.assign(function (...args) { return new OriginalURL(...args); }, {
  createObjectURL: vi.fn().mockReturnValue('blob:mock'),
  revokeObjectURL: vi.fn(),
  prototype: OriginalURL.prototype,
});
vi.stubGlobal('URL', MockURL);

// Import after mocks
const { Speaker } = await import('../../npm/decibri/src/browser/decibri-output-browser.js');

// ── Helpers ──────────────────────────────────────────────────────────────────

function resetMocks() {
  vi.clearAllMocks();
  mockPort.onmessage = null;
  capturedWorkletOptions = null;
  mockTime.value = 0;
  contextState = 'suspended';
  mockContextResume.mockImplementation(async () => { contextState = 'running'; });
  mockAddModule.mockResolvedValue(undefined);
  mockContextClose.mockResolvedValue(undefined);
}

// The float32 buffer of the nth raw-ArrayBuffer feed posted to the worklet.
function fedBuffer(n = 0) {
  const calls = mockPort.postMessage.mock.calls.filter(c => c[0] instanceof ArrayBuffer);
  return new Float32Array(calls[n][0]);
}

// ── Construction / validation ────────────────────────────────────────────────

describe('Speaker constructor', () => {
  beforeEach(resetMocks);

  it('creates an instance with defaults', () => {
    const speaker = new Speaker();
    expect(speaker).toBeInstanceOf(Speaker);
    expect(speaker.isPlaying).toBe(false);
  });

  it('accepts all options', () => {
    const speaker = new Speaker({
      sampleRate: 44100,
      channels: 2,
      dtype: 'float32',
      workletUrl: '/output-worklet.js',
    });
    expect(speaker.isPlaying).toBe(false);
  });
});

describe('Speaker constructor validation', () => {
  beforeEach(resetMocks);

  it('throws TypeError on invalid sampleRate', () => {
    expect(() => new Speaker({ sampleRate: 0 })).toThrow(TypeError);
    expect(() => new Speaker({ sampleRate: 0 })).toThrow('sample rate');
    expect(() => new Speaker({ sampleRate: 500000 })).toThrow('sample rate');
  });

  it('throws TypeError on invalid channels', () => {
    expect(() => new Speaker({ channels: 0 })).toThrow('channels');
    expect(() => new Speaker({ channels: 33 })).toThrow('channels');
  });

  it('throws TypeError on invalid dtype', () => {
    expect(() => new Speaker({ dtype: 'int8' })).toThrow("dtype must be 'int16' or 'float32'");
  });

  it('accepts int16 and float32 dtype', () => {
    expect(() => new Speaker({ dtype: 'int16' })).not.toThrow();
    expect(() => new Speaker({ dtype: 'float32' })).not.toThrow();
  });
});

// ── start() ──────────────────────────────────────────────────────────────────

describe('Speaker.start()', () => {
  beforeEach(resetMocks);

  it('creates, resumes the context and loads the output worklet', async () => {
    const speaker = new Speaker();
    await speaker.start();

    expect(MockAudioContext).toHaveBeenCalledTimes(1);
    expect(mockContextResume).toHaveBeenCalled();
    expect(mockAddModule).toHaveBeenCalledWith('blob:mock');
    speaker.stop();
  });

  it('builds an output node connected to the destination', async () => {
    const speaker = new Speaker({ channels: 2 });
    await speaker.start();

    expect(MockAudioWorkletNode).toHaveBeenCalledWith(
      expect.anything(),
      'decibri-output-processor',
      expect.objectContaining({
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
        processorOptions: { ringCapacity: 96000 },
      }),
    );
    expect(mockWorkletNode.connect).toHaveBeenCalledWith(mockDestination);
    speaker.stop();
  });

  it('uses a custom workletUrl when provided (CSP override)', async () => {
    const speaker = new Speaker({ workletUrl: '/my-worklet.js' });
    await speaker.start();
    expect(mockAddModule).toHaveBeenCalledWith('/my-worklet.js');
    expect(MockURL.createObjectURL).not.toHaveBeenCalled();
    speaker.stop();
  });

  it('is a no-op if already started', async () => {
    const speaker = new Speaker();
    await speaker.start();
    await speaker.start();
    expect(MockAudioContext).toHaveBeenCalledTimes(1);
    speaker.stop();
  });

  it('returns the same promise if a start is in progress', async () => {
    const speaker = new Speaker();
    const p1 = speaker.start();
    const p2 = speaker.start();
    expect(p1).toBe(p2);
    await p1;
    speaker.stop();
  });

  it('throws a clear error when the context stays suspended (autoplay blocked)', async () => {
    mockContextResume.mockImplementationOnce(async () => { /* stays suspended */ });
    const speaker = new Speaker();
    await expect(speaker.start()).rejects.toThrow('blocked until a user gesture');
    expect(mockContextClose).toHaveBeenCalled();
  });

  it('throws a clear error when the worklet fails to load', async () => {
    mockAddModule.mockRejectedValueOnce(new Error('CSP blocked'));
    const speaker = new Speaker();
    await expect(speaker.start()).rejects.toThrow('Failed to load audio worklet');
    expect(mockContextClose).toHaveBeenCalled();
  });
});

// ── Conversion pipeline ──────────────────────────────────────────────────────

describe('Speaker conversion pipeline', () => {
  beforeEach(resetMocks);

  it('converts int16 to float32 before feeding (no resample when rates match)', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'int16' });
    await speaker.write(new Int16Array([16384, -16384, 0, 32767]));

    const f = fedBuffer();
    expect(f.length).toBe(4);
    expect(f[0]).toBeCloseTo(0.5);
    expect(f[1]).toBeCloseTo(-0.5);
    expect(f[2]).toBeCloseTo(0);
    expect(f[3]).toBeCloseTo(32767 / 32768);
    speaker.stop();
  });

  it('passes float32 through unchanged when rates match', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    await speaker.write(new Float32Array([0.1, -0.2, 0.3]));

    const f = fedBuffer();
    expect(f.length).toBe(3);
    expect(f[0]).toBeCloseTo(0.1);
    expect(f[1]).toBeCloseTo(-0.2);
    expect(f[2]).toBeCloseTo(0.3);
    speaker.stop();
  });

  it('resamples user rate up to the context rate at an exact 2x ratio (24000 -> 48000)', async () => {
    const speaker = new Speaker({ sampleRate: 24000, dtype: 'float32' });
    await speaker.write(new Float32Array(10).fill(0.5));

    const f = fedBuffer();
    // ratio 0.5 (exactly representable): a length-10 input yields 18 samples.
    expect(f.length).toBe(18);
    for (let i = 0; i < f.length; i++) expect(f[i]).toBeCloseTo(0.5, 4);
    speaker.stop();
  });

  it('upsamples roughly 3x for 16000 -> 48000', async () => {
    const speaker = new Speaker({ sampleRate: 16000, dtype: 'float32' });
    await speaker.write(new Float32Array(10).fill(0.5));

    const f = fedBuffer();
    // About 3x the input (the exact count is set by deterministic float math).
    expect(f.length).toBeGreaterThanOrEqual(27);
    expect(f.length).toBeLessThanOrEqual(30);
    for (let i = 0; i < f.length; i++) expect(f[i]).toBeCloseTo(0.5, 4);
    speaker.stop();
  });

  it('feeds the worklet a transferable ArrayBuffer', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    await speaker.write(new Float32Array([0.1, 0.2]));

    const dataCall = mockPort.postMessage.mock.calls.find(c => c[0] instanceof ArrayBuffer);
    expect(dataCall[1]).toBeInstanceOf(Array);
    expect(dataCall[1][0]).toBe(dataCall[0]);
    speaker.stop();
  });

  it('resolves immediately on an empty chunk without feeding', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    await speaker.write(new Float32Array(0));
    const dataCalls = mockPort.postMessage.mock.calls.filter(c => c[0] instanceof ArrayBuffer);
    expect(dataCalls).toHaveLength(0);
    speaker.stop();
  });

  it('throws for a wrong input type', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'int16' });
    await speaker.start();
    await expect(speaker.write('not audio')).rejects.toThrow('Int16Array');
    speaker.stop();
  });
});

// ── write() backpressure ─────────────────────────────────────────────────────

describe('Speaker.write() backpressure', () => {
  beforeEach(resetMocks);

  it('paces writes so the queue estimate does not overflow the ring', async () => {
    vi.useFakeTimers();
    try {
      const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' }); // capacity 96000
      await speaker.start();
      mockTime.value = 0;

      // First 50000 fits (estimate 0 -> 50000).
      await speaker.write(new Float32Array(50000).fill(0.1));
      expect(mockPort.postMessage.mock.calls.filter(c => c[0] instanceof ArrayBuffer)).toHaveLength(1);

      // Second 50000 would overflow (50000 + 50000 > 96000): write must wait.
      let resolved = false;
      speaker.write(new Float32Array(50000).fill(0.2)).then(() => { resolved = true; });
      await Promise.resolve();
      await Promise.resolve();
      expect(resolved).toBe(false);
      expect(mockPort.postMessage.mock.calls.filter(c => c[0] instanceof ArrayBuffer)).toHaveLength(1);

      // Advance the audio clock so enough has played, then let the timer fire.
      mockTime.value = 0.2; // 9600 samples played -> estimate 40400, now fits
      await vi.advanceTimersByTimeAsync(200);
      expect(resolved).toBe(true);
      expect(mockPort.postMessage.mock.calls.filter(c => c[0] instanceof ArrayBuffer)).toHaveLength(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('applies backpressure when a level ack reports the ring full', async () => {
    vi.useFakeTimers();
    try {
      const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
      await speaker.start();
      mockTime.value = 0;

      // The worklet acks a full ring.
      mockPort.onmessage({ data: { type: 'level', queued: 96000, capacity: 96000, accepted: 96000, requested: 96000 } });

      let resolved = false;
      speaker.write(new Float32Array(1000).fill(0.1)).then(() => { resolved = true; });
      await Promise.resolve();
      await Promise.resolve();
      expect(resolved).toBe(false);

      mockTime.value = 2.1; // more than the full ring has played
      await vi.advanceTimersByTimeAsync(200);
      expect(resolved).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});

// ── drain() ──────────────────────────────────────────────────────────────────

describe('Speaker.drain()', () => {
  beforeEach(resetMocks);

  it('resolves immediately when nothing has been written', async () => {
    const speaker = new Speaker();
    await speaker.start();
    await expect(speaker.drain()).resolves.toBeUndefined();
    speaker.stop();
  });

  it('resolves immediately before start', async () => {
    const speaker = new Speaker();
    await expect(speaker.drain()).resolves.toBeUndefined();
  });

  it('resolves when the worklet reports the queue drained', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    await speaker.write(new Float32Array(100).fill(0.1));
    expect(speaker.isPlaying).toBe(true);

    let drained = false;
    speaker.drain().then(() => { drained = true; });
    await Promise.resolve();
    expect(drained).toBe(false);

    mockPort.onmessage({ data: { type: 'drained' } });
    await Promise.resolve();
    expect(drained).toBe(true);
    expect(speaker.isPlaying).toBe(false);
    speaker.stop();
  });
});

// ── stop() ───────────────────────────────────────────────────────────────────

describe('Speaker.stop()', () => {
  beforeEach(resetMocks);

  it('is a no-op before start', () => {
    const speaker = new Speaker();
    expect(() => speaker.stop()).not.toThrow();
  });

  it('flushes the worklet, tears down, and closes the context', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    await speaker.write(new Float32Array(100).fill(0.1));

    speaker.stop();

    expect(mockPort.postMessage).toHaveBeenCalledWith({ type: 'flush' });
    expect(mockWorkletNode.disconnect).toHaveBeenCalled();
    expect(mockPort.close).toHaveBeenCalled();
    expect(mockContextClose).toHaveBeenCalled();
    expect(speaker.isPlaying).toBe(false);
  });

  it('resolves a pending drain so it does not hang', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    await speaker.write(new Float32Array(100).fill(0.1));

    let drained = false;
    speaker.drain().then(() => { drained = true; });
    await Promise.resolve();
    expect(drained).toBe(false);

    speaker.stop();
    await Promise.resolve();
    expect(drained).toBe(true);
  });

  it('is safe to call multiple times', async () => {
    const speaker = new Speaker();
    await speaker.start();
    speaker.stop();
    expect(() => speaker.stop()).not.toThrow();
  });

  it('allows a fresh session after stop', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    await speaker.write(new Float32Array(10).fill(0.1));
    speaker.stop();

    await speaker.write(new Float32Array(10).fill(0.2));
    expect(MockAudioContext).toHaveBeenCalledTimes(2);
    expect(speaker.isPlaying).toBe(true);
    speaker.stop();
  });
});

// ── isPlaying ────────────────────────────────────────────────────────────────

describe('Speaker.isPlaying', () => {
  beforeEach(resetMocks);

  it('is false before start, true while audio is queued, false after drained', async () => {
    const speaker = new Speaker({ sampleRate: 48000, dtype: 'float32' });
    expect(speaker.isPlaying).toBe(false);

    await speaker.write(new Float32Array(100).fill(0.1));
    expect(speaker.isPlaying).toBe(true);

    mockPort.onmessage({ data: { type: 'drained' } });
    expect(speaker.isPlaying).toBe(false);
    speaker.stop();
  });
});

// ── Browser entry export ─────────────────────────────────────────────────────

describe('browser entry exports Speaker', () => {
  it('exposes Speaker alongside Microphone and Emitter', async () => {
    const browser = await import('../../npm/decibri/src/browser/index.js');
    const ns = browser.default ?? browser;
    expect(typeof ns.Speaker).toBe('function');
    expect(typeof ns.Microphone).toBe('function');
    expect(typeof ns.Emitter).toBe('function');
  });
});
