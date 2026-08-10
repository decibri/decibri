import { describe, it, expect, vi, beforeEach } from 'vitest';
import { readFileSync } from 'node:fs';
import pkg from '../../npm/decibri/package.json';

// ── Browser API mocks ────────────────────────────────────────────────────────

const mockTrackStop = vi.fn();
const mockGetSettings = vi.fn().mockReturnValue({ channelCount: 1 });
const mockTrack = { stop: mockTrackStop, getSettings: mockGetSettings };
const mockStream = {
  getTracks: () => [mockTrack],
  getAudioTracks: () => [mockTrack],
};

const mockPortOnMessage = { onmessage: null };
const mockPortClose = vi.fn();
const mockPort = {
  close: mockPortClose,
  get onmessage() { return mockPortOnMessage.onmessage; },
  set onmessage(fn) { mockPortOnMessage.onmessage = fn; },
};

const mockWorkletNodeDisconnect = vi.fn();
const mockWorkletNode = {
  port: mockPort,
  disconnect: mockWorkletNodeDisconnect,
};

const mockSourceDisconnect = vi.fn();
const mockSourceConnect = vi.fn();
const mockSourceNode = {
  disconnect: mockSourceDisconnect,
  connect: mockSourceConnect,
};

const mockAddModule = vi.fn().mockResolvedValue(undefined);
const mockContextClose = vi.fn().mockResolvedValue(undefined);
const mockContextResume = vi.fn().mockResolvedValue(undefined);
const mockCreateMediaStreamSource = vi.fn().mockReturnValue(mockSourceNode);

let capturedWorkletOptions = null;

const MockAudioContext = vi.fn().mockImplementation(function () {
  return {
    sampleRate: 48000,
    resume: mockContextResume,
    close: mockContextClose,
    createMediaStreamSource: mockCreateMediaStreamSource,
    audioWorklet: { addModule: mockAddModule },
  };
});

const MockAudioWorkletNode = vi.fn().mockImplementation(function (_ctx, _name, options) {
  capturedWorkletOptions = options;
  return mockWorkletNode;
});

const mockGetUserMedia = vi.fn().mockResolvedValue(mockStream);
const mockEnumerateDevices = vi.fn().mockResolvedValue([
  { kind: 'audioinput', deviceId: 'mic1', label: 'Built-in Mic', groupId: 'g1' },
  { kind: 'audioinput', deviceId: 'mic2', label: 'USB Mic', groupId: 'g2' },
  { kind: 'audiooutput', deviceId: 'spk1', label: 'Speaker', groupId: 'g3' },
  { kind: 'videoinput', deviceId: 'cam1', label: 'Camera', groupId: 'g4' },
]);

vi.stubGlobal('AudioContext', MockAudioContext);
vi.stubGlobal('AudioWorkletNode', MockAudioWorkletNode);
vi.stubGlobal('navigator', {
  mediaDevices: {
    getUserMedia: mockGetUserMedia,
    enumerateDevices: mockEnumerateDevices,
  },
});
vi.stubGlobal('Blob', class MockBlob {
  constructor(parts, options) { this.parts = parts; this.options = options; }
});
// Preserve the real URL constructor but override createObjectURL/revokeObjectURL
const OriginalURL = globalThis.URL;
const MockURL = Object.assign(function(...args) { return new OriginalURL(...args); }, {
  createObjectURL: vi.fn().mockReturnValue('blob:mock'),
  revokeObjectURL: vi.fn(),
  prototype: OriginalURL.prototype,
});
vi.stubGlobal('URL', MockURL);
vi.stubGlobal('DOMException', class DOMException extends Error {
  constructor(message, name) { super(message); this.name = name; }
});

// Import after mocks
const { Microphone } = await import('../../npm/decibri/src/browser/decibri-browser.js');

// ── Helpers ──────────────────────────────────────────────────────────────────

/** The error a call throws, so its class and message can both be asserted. */
function thrownBy(fn) {
  try {
    fn();
  } catch (err) {
    return err;
  }
  throw new Error('expected the call to throw');
}

function resetMocks() {
  vi.clearAllMocks();
  mockPortOnMessage.onmessage = null;
  capturedWorkletOptions = null;
  mockGetUserMedia.mockResolvedValue(mockStream);
  mockAddModule.mockResolvedValue(undefined);
  mockContextClose.mockResolvedValue(undefined);
  mockContextResume.mockResolvedValue(undefined);
  mockGetSettings.mockReturnValue({ channelCount: 1 });
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('Microphone constructor', () => {
  beforeEach(resetMocks);

  it('creates an instance with defaults', () => {
    const mic = new Microphone();
    expect(mic).toBeInstanceOf(Microphone);
    expect(mic.isOpen).toBe(false);
  });

  it('accepts all options', () => {
    const mic = new Microphone({
      sampleRate: 44100,
      channels: 1,
      framesPerBuffer: 800,
      device: 'mic2',
      dtype: 'float32',
      vad: { model: 'energy', threshold: 0.05, holdoffMs: 500 },
      echoCancellation: false,
      noiseSuppression: false,
      workletUrl: '/worklet.js',
    });
    expect(mic.isOpen).toBe(false);
  });
});

describe('Microphone constructor validation', () => {
  it('throws on invalid sampleRate', () => {
    // Out of range is a RangeError carrying the node entry's message, so the
    // same value is rejected identically in both runtimes.
    for (const sampleRate of [0, 500000]) {
      const err = thrownBy(() => new Microphone({ sampleRate }));
      expect(err).toBeInstanceOf(RangeError);
      expect(err.message).toBe('sample rate must be between 1000 and 384000');
    }
  });

  it('throws on invalid framesPerBuffer', () => {
    expect(() => new Microphone({ framesPerBuffer: 0 })).toThrow('frames per buffer');
    expect(() => new Microphone({ framesPerBuffer: 100000 })).toThrow('frames per buffer');
  });

  it('throws on invalid channels', () => {
    // The classes and the messages are the node entry's, so the same value is
    // rejected identically in both runtimes.
    const below = thrownBy(() => new Microphone({ channels: 0 }));
    expect(below).toBeInstanceOf(RangeError);
    expect(below.message).toBe('channels must be at least 1');

    for (const channels of [2, 33]) {
      const err = thrownBy(() => new Microphone({ channels }));
      expect(err).toBeInstanceOf(RangeError);
      expect(err.message).toBe('multichannel capture is not supported; channels must be 1 (mono)');
    }
  });

  it('accepts channels 1 and keeps the Web Audio ceiling on record', () => {
    expect(new Microphone({ channels: 1 })).toBeInstanceOf(Microphone);
    // The 32-channel bound is unreachable while only 1 is accepted, so pin it
    // in the source text: the recorded Web Audio floor must stay in place.
    const source = readFileSync(
      new URL('../../npm/decibri/src/browser/decibri-browser.js', import.meta.url),
      'utf8'
    );
    expect(source).toContain('this._channels > 32');
    expect(source).toContain("The Web Audio specification's floor");
  });

  it('throws on invalid vad threshold', () => {
    for (const threshold of [-1, 2]) {
      const err = thrownBy(() => new Microphone({ vad: { model: 'energy', threshold } }));
      expect(err).toBeInstanceOf(RangeError);
      expect(err.message).toBe('vad threshold must be between 0 and 1');
    }
  });

  it('throws on a non-numeric vad threshold', () => {
    // A string threshold compares false against every score, so accepting it
    // leaves the detector permanently quiet with nothing reported.
    for (const threshold of ['high', NaN, null, {}]) {
      const err = thrownBy(() => new Microphone({ vad: { model: 'energy', threshold } }));
      expect(err).toBeInstanceOf(TypeError);
      expect(err.message).toBe('vad threshold must be a number');
    }
  });

  it('throws on invalid vad holdoffMs', () => {
    const err = thrownBy(() => new Microphone({ vad: { model: 'energy', holdoffMs: -100 } }));
    expect(err).toBeInstanceOf(RangeError);
    expect(err.message).toBe('vad holdoffMs must be non-negative');
  });

  it('throws on a non-numeric vad holdoffMs', () => {
    for (const holdoffMs of ['300', NaN, null, {}]) {
      const err = thrownBy(() => new Microphone({ vad: { model: 'energy', holdoffMs } }));
      expect(err).toBeInstanceOf(TypeError);
      expect(err.message).toBe('vad holdoffMs must be a number');
    }
  });

  it('throws on an unknown vad model', () => {
    expect(() => new Microphone({ vad: { model: 'silero' } })).toThrow('Invalid vad model');
  });

  it('rejects the removed flat vadThreshold / vadHoldoff options', () => {
    expect(() => new Microphone({ vadThreshold: 0.05 })).toThrow('no longer supported');
    expect(() => new Microphone({ vadHoldoff: 500 })).toThrow('no longer supported');
  });

  it('rejects the legacy vad: true form', () => {
    expect(() => new Microphone({ vad: true })).toThrow('vad: true is no longer supported');
  });

  it('rejects an unrecognized vad value (silero is Node-only)', () => {
    expect(() => new Microphone({ vad: 'silero' })).toThrow('Invalid vad value');
    expect(() => new Microphone({ vad: 'loud' })).toThrow('Invalid vad value');
  });

  it('accepts vad: false and vad: energy', () => {
    expect(() => new Microphone({ vad: false })).not.toThrow();
    expect(() => new Microphone({ vad: 'energy' })).not.toThrow();
  });
});

describe('Microphone channelMap validation', () => {
  beforeEach(resetMocks);

  // The classes and the messages are the node entry's, so the same map is
  // rejected the same way in both runtimes (Group 7d asserts the pairing).

  it('rejects a non-array with the node entry message', () => {
    const err = thrownBy(() => new Microphone({ channelMap: 'left' }));
    expect(err).toBeInstanceOf(TypeError);
    expect(err.message).toBe(
      'Invalid channelMap value: "left". Expected an array of 0-based device channel indices, such as [0].'
    );
  });

  it('rejects non-integer entries', () => {
    for (const entry of [0.5, '0', NaN, null]) {
      const err = thrownBy(() => new Microphone({ channelMap: [entry] }));
      expect(err).toBeInstanceOf(TypeError);
      expect(err.message).toBe('channelMap entries must be integers');
    }
  });

  it('rejects entries outside the 16-bit carrier', () => {
    for (const entry of [-1, 65536]) {
      const err = thrownBy(() => new Microphone({ channelMap: [entry] }));
      expect(err).toBeInstanceOf(RangeError);
      expect(err.message).toBe('channelMap entries must be between 0 and 65535');
    }
  });

  it('rejects a length that differs from channels', () => {
    for (const channelMap of [[], [0, 1]]) {
      const err = thrownBy(() => new Microphone({ channelMap }));
      expect(err).toBeInstanceOf(RangeError);
      expect(err.message).toBe('channelMap must have exactly one entry per channel');
    }
  });

  it('accepts a shape-valid single entry, with no fixed maximum at construction', () => {
    expect(new Microphone({ channelMap: [0] })).toBeInstanceOf(Microphone);
    expect(new Microphone({ channelMap: [65535] })).toBeInstanceOf(Microphone);
  });
});

describe('Microphone channelMap at start', () => {
  beforeEach(resetMocks);

  it('validates the map against the granted track report and rejects with the engine message', async () => {
    mockGetSettings.mockReturnValue({ channelCount: 2 });
    const mic = new Microphone({ channelMap: [2] });
    const errorFn = vi.fn();
    mic.on('error', errorFn);

    await expect(mic.start()).rejects.toThrow(
      'the channel map names device channel 2; the device reports 2 input channels'
    );
    expect(errorFn).toHaveBeenCalledTimes(1);
    expect(mockTrackStop).toHaveBeenCalled();
    expect(mockContextClose).toHaveBeenCalled();
    expect(mockAddModule).not.toHaveBeenCalled();
    expect(mic.isOpen).toBe(false);
  });

  it('accepts a map the granted report covers and hands it to the worklet', async () => {
    mockGetSettings.mockReturnValue({ channelCount: 2 });
    const mic = new Microphone({ channelMap: [1] });
    await mic.start();
    expect(capturedWorkletOptions.processorOptions.channelMap).toEqual([1]);
    mic.stop();
  });

  it('accepts a large index when the grant reports that many channels', async () => {
    // The negative control: the granted report is the only ceiling, so an
    // index far above the Web Audio 32 floor must pass when the grant covers
    // it. A fixed maximum added to this path later fails loudly here.
    mockGetSettings.mockReturnValue({ channelCount: 4096 });
    const mic = new Microphone({ channelMap: [4095] });
    await mic.start();
    expect(mic.isOpen).toBe(true);
    mic.stop();
  });

  it('defers the check to the worklet when the browser reports no channel count', async () => {
    mockGetSettings.mockReturnValue({});
    const mic = new Microphone({ channelMap: [5] });
    const errorFn = vi.fn();
    mic.on('error', errorFn);

    await mic.start();
    expect(mic.isOpen).toBe(true);

    // The worklet reports the failure as a tagged control object on the same
    // port the chunks ride; the wrapper surfaces it and stops.
    mockPort.onmessage({
      data: { type: 'error', message: 'the channel map names device channel 5; the device reports 1 input channels' },
    });
    expect(errorFn).toHaveBeenCalledTimes(1);
    expect(errorFn.mock.calls[0][0].message).toBe(
      'the channel map names device channel 5; the device reports 1 input channels'
    );
    expect(mic.isOpen).toBe(false);
    expect(mockTrackStop).toHaveBeenCalled();
  });

  it('does not read the granted report without a map', async () => {
    const mic = new Microphone();
    await mic.start();
    expect(mockGetSettings).not.toHaveBeenCalled();
    expect(capturedWorkletOptions.processorOptions.channelMap).toBe(null);
    mic.stop();
  });
});

describe('Microphone.start()', () => {
  beforeEach(resetMocks);

  it('calls getUserMedia with correct constraints', async () => {
    const mic = new Microphone({ channels: 1, echoCancellation: true, noiseSuppression: false });
    await mic.start();

    // The channel ask is the Web Audio 32 floor with ideal semantics, never
    // the delivered count: the granted track's report is the authority.
    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: {
        channelCount: { ideal: 32 },
        echoCancellation: true,
        noiseSuppression: false,
      },
    });

    mic.stop();
  });

  it('includes deviceId when specified', async () => {
    const mic = new Microphone({ device: 'mic2' });
    await mic.start();

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: expect.objectContaining({
        deviceId: { exact: 'mic2' },
      }),
    });

    mic.stop();
  });

  it('passes correct processor options', async () => {
    const mic = new Microphone({ sampleRate: 16000, framesPerBuffer: 1600, dtype: 'int16' });
    await mic.start();

    expect(capturedWorkletOptions.processorOptions).toEqual({
      framesPerBuffer: 1600,
      format: 'int16',
      nativeSampleRate: 48000,
      targetSampleRate: 16000,
      channelMap: null,
    });

    mic.stop();
  });

  it('sets isOpen to true after start', async () => {
    const mic = new Microphone();
    await mic.start();
    expect(mic.isOpen).toBe(true);
    mic.stop();
  });

  it('is a no-op if already started', async () => {
    const mic = new Microphone();
    await mic.start();
    await mic.start();
    expect(MockAudioContext).toHaveBeenCalledTimes(1);
    mic.stop();
  });

  it('returns same promise if start is in progress', async () => {
    const mic = new Microphone();
    const p1 = mic.start();
    const p2 = mic.start();
    expect(p1).toBe(p2);
    await p1;
    mic.stop();
  });

  it('connects source to worklet but not to destination', async () => {
    const mic = new Microphone();
    await mic.start();

    expect(mockSourceConnect).toHaveBeenCalledWith(mockWorkletNode);
    expect(mockSourceConnect).toHaveBeenCalledTimes(1);

    mic.stop();
  });

  it('resumes AudioContext for Safari', async () => {
    const mic = new Microphone();
    await mic.start();
    expect(mockContextResume).toHaveBeenCalled();
    mic.stop();
  });
});

describe('Microphone.start() error handling', () => {
  beforeEach(resetMocks);

  it('rejects with clear message on permission denied', async () => {
    const domError = new DOMException('User denied', 'NotAllowedError');
    mockGetUserMedia.mockRejectedValueOnce(domError);

    const mic = new Microphone();
    const errorFn = vi.fn();
    mic.on('error', errorFn);

    await expect(mic.start()).rejects.toThrow('Microphone permission denied');
    expect(errorFn).toHaveBeenCalledWith(expect.objectContaining({ message: 'Microphone permission denied' }));
    expect(mic.isOpen).toBe(false);
  });

  it('rejects with clear message when no mic found', async () => {
    const domError = new DOMException('No device', 'NotFoundError');
    mockGetUserMedia.mockRejectedValueOnce(domError);

    const mic = new Microphone();
    await expect(mic.start()).rejects.toThrow('No microphone found');
  });

  it('rejects with generic message on other errors', async () => {
    const domError = new DOMException('Something else', 'NotReadableError');
    mockGetUserMedia.mockRejectedValueOnce(domError);

    const mic = new Microphone();
    await expect(mic.start()).rejects.toThrow('Microphone access failed');
  });

  it('cleans up AudioContext on getUserMedia failure', async () => {
    mockGetUserMedia.mockRejectedValueOnce(new DOMException('Denied', 'NotAllowedError'));

    const mic = new Microphone();
    try { await mic.start(); } catch {}

    expect(mockContextClose).toHaveBeenCalled();
  });

  it('cleans up on worklet load failure', async () => {
    mockAddModule.mockRejectedValueOnce(new Error('CSP blocked'));

    const mic = new Microphone();
    const errorFn = vi.fn();
    mic.on('error', errorFn);

    await expect(mic.start()).rejects.toThrow('Failed to load audio worklet');
    expect(mockTrackStop).toHaveBeenCalled();
    expect(mockContextClose).toHaveBeenCalled();
    expect(errorFn).toHaveBeenCalled();
  });

  it('allows start() after a failed start()', async () => {
    mockGetUserMedia.mockRejectedValueOnce(new DOMException('Denied', 'NotAllowedError'));

    const mic = new Microphone();
    try { await mic.start(); } catch {}

    mockGetUserMedia.mockResolvedValueOnce(mockStream);
    await mic.start();
    expect(mic.isOpen).toBe(true);
    mic.stop();
  });
});

describe('Microphone.stop()', () => {
  beforeEach(resetMocks);

  it('is a no-op before start', () => {
    const mic = new Microphone();
    expect(() => mic.stop()).not.toThrow();
  });

  it('cleans up all resources', async () => {
    const mic = new Microphone();
    await mic.start();
    mic.stop();

    expect(mockTrackStop).toHaveBeenCalled();
    expect(mockSourceDisconnect).toHaveBeenCalled();
    expect(mockWorkletNodeDisconnect).toHaveBeenCalled();
    expect(mockPortClose).toHaveBeenCalled();
    expect(mockContextClose).toHaveBeenCalled();
    expect(mic.isOpen).toBe(false);
  });

  it('emits end then close', async () => {
    const mic = new Microphone();
    await mic.start();

    const events = [];
    mic.on('end', () => events.push('end'));
    mic.on('close', () => events.push('close'));
    mic.stop();

    expect(events).toEqual(['end', 'close']);
  });

  it('is safe to call multiple times', async () => {
    const mic = new Microphone();
    await mic.start();
    mic.stop();
    expect(() => mic.stop()).not.toThrow();
  });

  it('tears down if stop() called during in-flight start()', async () => {
    const mic = new Microphone();
    const startPromise = mic.start();
    mic.stop();

    await startPromise;

    expect(mic.isOpen).toBe(false);
    expect(mockTrackStop).toHaveBeenCalled();
    expect(mockContextClose).toHaveBeenCalled();
  });

  it('allows fresh start after stop', async () => {
    const mic = new Microphone();
    await mic.start();
    mic.stop();

    await mic.start();
    expect(mic.isOpen).toBe(true);
    expect(MockAudioContext).toHaveBeenCalledTimes(2);
    mic.stop();
  });
});

describe('Microphone data events', () => {
  beforeEach(resetMocks);

  it('emits Int16Array for int16 format', async () => {
    const mic = new Microphone({ dtype: 'int16' });
    await mic.start();

    const fn = vi.fn();
    mic.on('data', fn);

    const int16 = new Int16Array([100, -100, 0]);
    mockPort.onmessage({ data: int16.buffer });

    expect(fn).toHaveBeenCalledTimes(1);
    const chunk = fn.mock.calls[0][0];
    expect(chunk).toBeInstanceOf(Int16Array);
    expect(chunk.length).toBe(3);

    mic.stop();
  });

  it('emits Float32Array for float32 format', async () => {
    const mic = new Microphone({ dtype: 'float32' });
    await mic.start();

    const fn = vi.fn();
    mic.on('data', fn);

    const float32 = new Float32Array([0.5, -0.5, 0.0]);
    mockPort.onmessage({ data: float32.buffer });

    expect(fn).toHaveBeenCalledTimes(1);
    const chunk = fn.mock.calls[0][0];
    expect(chunk).toBeInstanceOf(Float32Array);

    mic.stop();
  });
});

describe('Microphone VAD', () => {
  beforeEach(resetMocks);

  it('emits speech when RMS crosses threshold', async () => {
    const mic = new Microphone({ dtype: 'float32', vad: { model: 'energy', threshold: 0.01 } });
    await mic.start();

    const speechFn = vi.fn();
    mic.on('speech', speechFn);

    const loud = new Float32Array(100).fill(0.5);
    mockPort.onmessage({ data: loud.buffer });

    expect(speechFn).toHaveBeenCalledTimes(1);
    mic.stop();
  });

  it('exposes vadScore as the last RMS in energy mode', async () => {
    const mic = new Microphone({ dtype: 'float32', vad: { model: 'energy', threshold: 0.01 } });
    await mic.start();

    expect(mic.vadScore).toBe(0);

    const loud = new Float32Array(100).fill(0.5);
    mockPort.onmessage({ data: loud.buffer });

    expect(mic.vadScore).toBeCloseTo(0.5);
    mic.stop();
  });

  it('keeps vadScore at 0 when vad is disabled', async () => {
    const mic = new Microphone({ dtype: 'float32', vad: false });
    await mic.start();

    const loud = new Float32Array(100).fill(0.5);
    mockPort.onmessage({ data: loud.buffer });

    expect(mic.vadScore).toBe(0);
    mic.stop();
  });

  it('does not emit speech when below threshold', async () => {
    const mic = new Microphone({ dtype: 'float32', vad: { model: 'energy', threshold: 0.5 } });
    await mic.start();

    const speechFn = vi.fn();
    mic.on('speech', speechFn);

    const quiet = new Float32Array(100).fill(0.001);
    mockPort.onmessage({ data: quiet.buffer });

    expect(speechFn).not.toHaveBeenCalled();
    mic.stop();
  });

  it('emits silence after holdoff period', async () => {
    vi.useFakeTimers();

    const mic = new Microphone({ dtype: 'float32', vad: { model: 'energy', threshold: 0.01, holdoffMs: 300 } });
    await mic.start();

    const speechFn = vi.fn();
    const silenceFn = vi.fn();
    mic.on('speech', speechFn);
    mic.on('silence', silenceFn);

    const loud = new Float32Array(100).fill(0.5);
    mockPort.onmessage({ data: loud.buffer });
    expect(speechFn).toHaveBeenCalledTimes(1);

    const quiet = new Float32Array(100).fill(0.0001);
    mockPort.onmessage({ data: quiet.buffer });
    expect(silenceFn).not.toHaveBeenCalled();

    vi.advanceTimersByTime(300);
    expect(silenceFn).toHaveBeenCalledTimes(1);

    mic.stop();
    vi.useRealTimers();
  });

  it('does not emit events when vad is disabled', async () => {
    const mic = new Microphone({ dtype: 'float32', vad: false });
    await mic.start();

    const speechFn = vi.fn();
    mic.on('speech', speechFn);

    const loud = new Float32Array(100).fill(0.5);
    mockPort.onmessage({ data: loud.buffer });

    expect(speechFn).not.toHaveBeenCalled();
    mic.stop();
  });

  it('works with int16 format', async () => {
    const mic = new Microphone({ dtype: 'int16', vad: { model: 'energy', threshold: 0.01 } });
    await mic.start();

    const speechFn = vi.fn();
    mic.on('speech', speechFn);

    const loud = new Int16Array(100).fill(16384);
    mockPort.onmessage({ data: loud.buffer });

    expect(speechFn).toHaveBeenCalledTimes(1);
    mic.stop();
  });
});

describe('Microphone.devices()', () => {
  beforeEach(resetMocks);

  it('returns only audioinput devices', async () => {
    const devices = await Microphone.devices();
    expect(devices).toHaveLength(2);
    expect(devices[0]).toEqual({ deviceId: 'mic1', label: 'Built-in Mic', groupId: 'g1' });
    expect(devices[1]).toEqual({ deviceId: 'mic2', label: 'USB Mic', groupId: 'g2' });
  });
});

describe('Microphone.version()', () => {
  it('returns version info with decibri key', () => {
    const v = Microphone.version();
    expect(v).toHaveProperty('decibri');
    expect(typeof v.decibri).toBe('string');
  });

  it('reports the package version', () => {
    // The browser VERSION constant is maintained by hand; it must match the
    // package version in npm/decibri/package.json.
    expect(Microphone.version().decibri).toBe(pkg.version);
  });
});
