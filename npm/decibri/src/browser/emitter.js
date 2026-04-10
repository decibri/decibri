'use strict';

/**
 * Minimal event emitter for browsers.
 *
 * Provides .on() / .off() / .once() / .emit() with the same API pattern
 * as Node.js EventEmitter. Used instead of browser-native EventTarget to
 * preserve API parity with the Node.js decibri (.on('data', cb) pattern).
 *
 * Ported from decibri-web emitter.ts — logic identical, types removed.
 */
class Emitter {
  constructor() {
    this._listeners = new Map();
  }

  on(event, fn) {
    let set = this._listeners.get(event);
    if (!set) {
      set = new Set();
      this._listeners.set(event, set);
    }
    set.add(fn);
    return this;
  }

  off(event, fn) {
    const set = this._listeners.get(event);
    if (!set) return this;
    if (set.delete(fn)) return this;
    for (const listener of set) {
      if (listener._original === fn) {
        set.delete(listener);
        return this;
      }
    }
    return this;
  }

  once(event, fn) {
    const wrapper = (...args) => {
      this.off(event, wrapper);
      fn(...args);
    };
    wrapper._original = fn;
    return this.on(event, wrapper);
  }

  emit(event, ...args) {
    const set = this._listeners.get(event);
    if (!set || set.size === 0) return false;
    for (const fn of set) fn(...args);
    return true;
  }

  removeAllListeners(event) {
    if (event !== undefined) {
      this._listeners.delete(event);
    } else {
      this._listeners.clear();
    }
    return this;
  }
}

module.exports = { Emitter };
