import { describe, it, expect } from 'vitest';

/**
 * Tests for the resampling logic used in the AudioWorklet processor.
 * Re-implements the same algorithm for isolated testing (the worklet
 * can't import modules, so we test the algorithm separately).
 */

class Resampler {
  constructor(fromRate, toRate) {
    this.ratio = fromRate / toRate;
    this.position = 0;
  }

  process(input) {
    if (this.ratio === 1) return input;

    const inputLength = input.length;
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

    this.position = pos - inputLength;
    return output;
  }
}

describe('Resampler', () => {
  it('passes through when rates match', () => {
    const r = new Resampler(48000, 48000);
    const input = new Float32Array([0.1, 0.2, 0.3]);
    const output = r.process(input);
    expect(output).toBe(input); // same reference
  });

  it('downsamples 48000 -> 16000 (clean 3:1)', () => {
    const r = new Resampler(48000, 16000);
    const input = new Float32Array(12);
    for (let i = 0; i < 12; i++) input[i] = i;

    const output = r.process(input);
    expect(output.length).toBe(4);
    expect(output[0]).toBeCloseTo(0);
    expect(output[1]).toBeCloseTo(3);
    expect(output[2]).toBeCloseTo(6);
    expect(output[3]).toBeCloseTo(9);
  });

  it('downsamples 44100 -> 16000 (fractional ratio 2.75625)', () => {
    const r = new Resampler(44100, 16000);
    const input = new Float32Array(128);
    for (let i = 0; i < 128; i++) input[i] = Math.sin(2 * Math.PI * 1000 * i / 44100);

    const output = r.process(input);
    expect(output.length).toBeGreaterThan(40);
    expect(output.length).toBeLessThan(50);
  });

  it('maintains cross-chunk continuity', () => {
    const r = new Resampler(48000, 16000);

    const chunk1 = new Float32Array(12);
    const chunk2 = new Float32Array(12);
    for (let i = 0; i < 12; i++) chunk1[i] = i;
    for (let i = 0; i < 12; i++) chunk2[i] = i + 12;

    const out1 = r.process(chunk1);
    const out2 = r.process(chunk2);

    const combined = new Float32Array(out1.length + out2.length);
    combined.set(out1);
    combined.set(out2, out1.length);

    // All output samples should be monotonically increasing (it's a ramp)
    for (let i = 1; i < combined.length; i++) {
      expect(combined[i]).toBeGreaterThan(combined[i - 1]);
    }

    // Compare with processing all at once
    const r2 = new Resampler(48000, 16000);
    const allInput = new Float32Array(24);
    for (let i = 0; i < 24; i++) allInput[i] = i;
    const allOutput = r2.process(allInput);

    const minLen = Math.min(combined.length, allOutput.length);
    for (let i = 0; i < minLen; i++) {
      expect(combined[i]).toBeCloseTo(allOutput[i], 4);
    }
  });

  it('preserves sine wave frequency through resampling', () => {
    const fromRate = 48000;
    const toRate = 16000;
    const freq = 1000;
    const r = new Resampler(fromRate, toRate);

    const input = new Float32Array(480);
    for (let i = 0; i < 480; i++) {
      input[i] = Math.sin(2 * Math.PI * freq * i / fromRate);
    }

    const output = r.process(input);
    expect(output.length).toBe(160);

    // Peak should be near sample 4 (quarter cycle of 16 samples/cycle)
    const peakIdx = Array.from(output.subarray(0, 16)).reduce(
      (maxIdx, val, idx, arr) => val > arr[maxIdx] ? idx : maxIdx, 0
    );
    expect(peakIdx).toBe(4);
  });

  it('handles very small input chunks', () => {
    const r = new Resampler(48000, 16000);
    const input = new Float32Array([0.5, 0.6]);
    const output = r.process(input);
    expect(output.length).toBeLessThanOrEqual(1);
  });
});
