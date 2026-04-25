import { describe, it, expect, vi, beforeEach } from 'vitest';

// ── Browser API mocks ────────────────────────────────────────────────────────

const mockTrackStop = vi.fn();
const mockStream = {
  getTracks: () => [{ stop: mockTrackStop }],
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
const { Decibri } = await import('../../npm/decibri/src/browser/decibri-browser.js');

// ── Helpers ──────────────────────────────────────────────────────────────────

function resetMocks() {
  vi.clearAllMocks();
  mockPortOnMessage.onmessage = null;
  capturedWorkletOptions = null;
  mockGetUserMedia.mockResolvedValue(mockStream);
  mockAddModule.mockResolvedValue(undefined);
  mockContextClose.mockResolvedValue(undefined);
  mockContextResume.mockResolvedValue(undefined);
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('Decibri constructor', () => {
  beforeEach(resetMocks);

  it('creates an instance with defaults', () => {
    const mic = new Decibri();
    expect(mic).toBeInstanceOf(Decibri);
    expect(mic.isOpen).toBe(false);
  });

  it('accepts all options', () => {
    const mic = new Decibri({
      sampleRate: 44100,
      channels: 2,
      framesPerBuffer: 800,
      device: 'mic2',
      format: 'float32',
      vad: true,
      vadThreshold: 0.05,
      vadHoldoff: 500,
      echoCancellation: false,
      noiseSuppression: false,
      workletUrl: '/worklet.js',
    });
    expect(mic.isOpen).toBe(false);
  });
});

describe('Decibri constructor validation', () => {
  it('throws on invalid sampleRate', () => {
    expect(() => new Decibri({ sampleRate: 0 })).toThrow('sample rate');
    expect(() => new Decibri({ sampleRate: 500000 })).toThrow('sample rate');
  });

  it('throws on invalid framesPerBuffer', () => {
    expect(() => new Decibri({ framesPerBuffer: 0 })).toThrow('frames per buffer');
    expect(() => new Decibri({ framesPerBuffer: 100000 })).toThrow('frames per buffer');
  });

  it('throws on invalid channels', () => {
    expect(() => new Decibri({ channels: 0 })).toThrow('channels');
    expect(() => new Decibri({ channels: 33 })).toThrow('channels');
  });

  it('throws on invalid vadThreshold', () => {
    expect(() => new Decibri({ vadThreshold: -1 })).toThrow('vadThreshold');
    expect(() => new Decibri({ vadThreshold: 2 })).toThrow('vadThreshold');
  });

  it('throws on invalid vadHoldoff', () => {
    expect(() => new Decibri({ vadHoldoff: -100 })).toThrow('vadHoldoff');
  });
});

describe('Decibri.start()', () => {
  beforeEach(resetMocks);

  it('calls getUserMedia with correct constraints', async () => {
    const mic = new Decibri({ channels: 1, echoCancellation: true, noiseSuppression: false });
    await mic.start();

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: false,
      },
    });

    mic.stop();
  });

  it('includes deviceId when specified', async () => {
    const mic = new Decibri({ device: 'mic2' });
    await mic.start();

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: expect.objectContaining({
        deviceId: { exact: 'mic2' },
      }),
    });

    mic.stop();
  });

  it('passes correct processor options', async () => {
    const mic = new Decibri({ sampleRate: 16000, framesPerBuffer: 1600, format: 'int16' });
    await mic.start();

    expect(capturedWorkletOptions.processorOptions).toEqual({
      framesPerBuffer: 1600,
      format: 'int16',
      nativeSampleRate: 48000,
      targetSampleRate: 16000,
    });

    mic.stop();
  });

  it('sets isOpen to true after start', async () => {
    const mic = new Decibri();
    await mic.start();
    expect(mic.isOpen).toBe(true);
    mic.stop();
  });

  it('is a no-op if already started', async () => {
    const mic = new Decibri();
    await mic.start();
    await mic.start();
    expect(MockAudioContext).toHaveBeenCalledTimes(1);
    mic.stop();
  });

  it('returns same promise if start is in progress', async () => {
    const mic = new Decibri();
    const p1 = mic.start();
    const p2 = mic.start();
    expect(p1).toBe(p2);
    await p1;
    mic.stop();
  });

  it('connects source to worklet but not to destination', async () => {
    const mic = new Decibri();
    await mic.start();

    expect(mockSourceConnect).toHaveBeenCalledWith(mockWorkletNode);
    expect(mockSourceConnect).toHaveBeenCalledTimes(1);

    mic.stop();
  });

  it('resumes AudioContext for Safari', async () => {
    const mic = new Decibri();
    await mic.start();
    expect(mockContextResume).toHaveBeenCalled();
    mic.stop();
  });
});

describe('Decibri.start() error handling', () => {
  beforeEach(resetMocks);

  it('rejects with clear message on permission denied', async () => {
    const domError = new DOMException('User denied', 'NotAllowedError');
    mockGetUserMedia.mockRejectedValueOnce(domError);

    const mic = new Decibri();
    const errorFn = vi.fn();
    mic.on('error', errorFn);

    await expect(mic.start()).rejects.toThrow('Microphone permission denied');
    expect(errorFn).toHaveBeenCalledWith(expect.objectContaining({ message: 'Microphone permission denied' }));
    expect(mic.isOpen).toBe(false);
  });

  it('rejects with clear message when no mic found', async () => {
    const domError = new DOMException('No device', 'NotFoundError');
    mockGetUserMedia.mockRejectedValueOnce(domError);

    const mic = new Decibri();
    await expect(mic.start()).rejects.toThrow('No microphone found');
  });

  it('rejects with generic message on other errors', async () => {
    const domError = new DOMException('Something else', 'NotReadableError');
    mockGetUserMedia.mockRejectedValueOnce(domError);

    const mic = new Decibri();
    await expect(mic.start()).rejects.toThrow('Microphone access failed');
  });

  it('cleans up AudioContext on getUserMedia failure', async () => {
    mockGetUserMedia.mockRejectedValueOnce(new DOMException('Denied', 'NotAllowedError'));

    const mic = new Decibri();
    try { await mic.start(); } catch {}

    expect(mockContextClose).toHaveBeenCalled();
  });

  it('cleans up on worklet load failure', async () => {
    mockAddModule.mockRejectedValueOnce(new Error('CSP blocked'));

    const mic = new Decibri();
    const errorFn = vi.fn();
    mic.on('error', errorFn);

    await expect(mic.start()).rejects.toThrow('Failed to load audio worklet');
    expect(mockTrackStop).toHaveBeenCalled();
    expect(mockContextClose).toHaveBeenCalled();
    expect(errorFn).toHaveBeenCalled();
  });

  it('allows start() after a failed start()', async () => {
    mockGetUserMedia.mockRejectedValueOnce(new DOMException('Denied', 'NotAllowedError'));

    const mic = new Decibri();
    try { await mic.start(); } catch {}

    mockGetUserMedia.mockResolvedValueOnce(mockStream);
    await mic.start();
    expect(mic.isOpen).toBe(true);
    mic.stop();
  });
});

describe('Decibri.stop()', () => {
  beforeEach(resetMocks);

  it('is a no-op before start', () => {
    const mic = new Decibri();
    expect(() => mic.stop()).not.toThrow();
  });

  it('cleans up all resources', async () => {
    const mic = new Decibri();
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
    const mic = new Decibri();
    await mic.start();

    const events = [];
    mic.on('end', () => events.push('end'));
    mic.on('close', () => events.push('close'));
    mic.stop();

    expect(events).toEqual(['end', 'close']);
  });

  it('is safe to call multiple times', async () => {
    const mic = new Decibri();
    await mic.start();
    mic.stop();
    expect(() => mic.stop()).not.toThrow();
  });

  it('tears down if stop() called during in-flight start()', async () => {
    const mic = new Decibri();
    const startPromise = mic.start();
    mic.stop();

    await startPromise;

    expect(mic.isOpen).toBe(false);
    expect(mockTrackStop).toHaveBeenCalled();
    expect(mockContextClose).toHaveBeenCalled();
  });

  it('allows fresh start after stop', async () => {
    const mic = new Decibri();
    await mic.start();
    mic.stop();

    await mic.start();
    expect(mic.isOpen).toBe(true);
    expect(MockAudioContext).toHaveBeenCalledTimes(2);
    mic.stop();
  });
});

describe('Decibri data events', () => {
  beforeEach(resetMocks);

  it('emits Int16Array for int16 format', async () => {
    const mic = new Decibri({ format: 'int16' });
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
    const mic = new Decibri({ format: 'float32' });
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

describe('Decibri VAD', () => {
  beforeEach(resetMocks);

  it('emits speech when RMS crosses threshold', async () => {
    const mic = new Decibri({ format: 'float32', vad: true, vadThreshold: 0.01 });
    await mic.start();

    const speechFn = vi.fn();
    mic.on('speech', speechFn);

    const loud = new Float32Array(100).fill(0.5);
    mockPort.onmessage({ data: loud.buffer });

    expect(speechFn).toHaveBeenCalledTimes(1);
    mic.stop();
  });

  it('does not emit speech when below threshold', async () => {
    const mic = new Decibri({ format: 'float32', vad: true, vadThreshold: 0.5 });
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

    const mic = new Decibri({ format: 'float32', vad: true, vadThreshold: 0.01, vadHoldoff: 300 });
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
    const mic = new Decibri({ format: 'float32', vad: false });
    await mic.start();

    const speechFn = vi.fn();
    mic.on('speech', speechFn);

    const loud = new Float32Array(100).fill(0.5);
    mockPort.onmessage({ data: loud.buffer });

    expect(speechFn).not.toHaveBeenCalled();
    mic.stop();
  });

  it('works with int16 format', async () => {
    const mic = new Decibri({ format: 'int16', vad: true, vadThreshold: 0.01 });
    await mic.start();

    const speechFn = vi.fn();
    mic.on('speech', speechFn);

    const loud = new Int16Array(100).fill(16384);
    mockPort.onmessage({ data: loud.buffer });

    expect(speechFn).toHaveBeenCalledTimes(1);
    mic.stop();
  });
});

describe('Decibri.devices()', () => {
  beforeEach(resetMocks);

  it('returns only audioinput devices', async () => {
    const devices = await Decibri.devices();
    expect(devices).toHaveLength(2);
    expect(devices[0]).toEqual({ deviceId: 'mic1', label: 'Built-in Mic', groupId: 'g1' });
    expect(devices[1]).toEqual({ deviceId: 'mic2', label: 'USB Mic', groupId: 'g2' });
  });
});

describe('Decibri.version()', () => {
  it('returns version info with decibri key', () => {
    const v = Decibri.version();
    expect(v).toHaveProperty('decibri');
    expect(typeof v.decibri).toBe('string');
  });
});
