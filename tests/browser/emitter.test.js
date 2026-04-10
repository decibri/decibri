import { describe, it, expect, vi } from 'vitest';
import { Emitter } from '../../npm/decibri/src/browser/emitter.js';

function create() {
  return new Emitter();
}

describe('Emitter', () => {
  it('on + emit delivers payload', () => {
    const e = create();
    const fn = vi.fn();
    e.on('data', fn);
    e.emit('data', 42);
    expect(fn).toHaveBeenCalledWith(42);
  });

  it('returns false when no listeners', () => {
    const e = create();
    expect(e.emit('end')).toBe(false);
  });

  it('returns true when listeners exist', () => {
    const e = create();
    e.on('end', () => {});
    expect(e.emit('end')).toBe(true);
  });

  it('supports multiple listeners', () => {
    const e = create();
    const fn1 = vi.fn();
    const fn2 = vi.fn();
    e.on('data', fn1);
    e.on('data', fn2);
    e.emit('data', 7);
    expect(fn1).toHaveBeenCalledWith(7);
    expect(fn2).toHaveBeenCalledWith(7);
  });

  it('off removes a listener', () => {
    const e = create();
    const fn = vi.fn();
    e.on('data', fn);
    e.off('data', fn);
    e.emit('data', 1);
    expect(fn).not.toHaveBeenCalled();
  });

  it('once fires exactly once', () => {
    const e = create();
    const fn = vi.fn();
    e.once('data', fn);
    e.emit('data', 1);
    e.emit('data', 2);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith(1);
  });

  it('removeAllListeners clears a specific event', () => {
    const e = create();
    const fn = vi.fn();
    e.on('data', fn);
    e.on('end', fn);
    e.removeAllListeners('data');
    e.emit('data', 1);
    e.emit('end');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('removeAllListeners with no arg clears everything', () => {
    const e = create();
    const fn = vi.fn();
    e.on('data', fn);
    e.on('end', fn);
    e.removeAllListeners();
    e.emit('data', 1);
    e.emit('end');
    expect(fn).not.toHaveBeenCalled();
  });

  it('on returns this for chaining', () => {
    const e = create();
    const result = e.on('data', () => {}).on('end', () => {});
    expect(result).toBe(e);
  });

  it('off removes a once listener by original function', () => {
    const e = create();
    const fn = vi.fn();
    e.once('data', fn);
    e.off('data', fn);
    e.emit('data', 1);
    expect(fn).not.toHaveBeenCalled();
  });

  it('off on non-existent event does not throw', () => {
    const e = create();
    expect(() => e.off('data', () => {})).not.toThrow();
  });
});
