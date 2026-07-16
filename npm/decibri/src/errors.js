'use strict';

/**
 * decibri error classes.
 *
 * A shallow set of catch-root classes that mirror the catch-roots the Python
 * binding exposes (see bindings/python/python/decibri/exceptions.py), so
 * `instanceof DecibriError` / `instanceof DeviceError` works the same way
 * across bindings. Node idiom is shallow classes plus a stable `code` string,
 * not a deep subclass tree: branch on `err.code` or on `instanceof`.
 *
 *   DecibriError            (base; extends Error)
 *   +- DeviceError          (device enumeration / selection failures)
 *   +- OrtError             (ONNX Runtime setup / inference failures)
 *      +- OrtPathError      (a specific ORT library path could not be loaded)
 *
 * Argument validation (bad sample rate, channels, frames, dtype, vad) keeps
 * Node's built-in RangeError / TypeError, matching Node core; those are not
 * decibri error classes by design.
 */

class DecibriError extends Error {
  /**
   * @param {string} message
   * @param {string} [code] Stable string code for `err.code` branching.
   */
  constructor(message, code) {
    super(message);
    this.name = 'DecibriError';
    this.code = code;
  }
}

class DeviceError extends DecibriError {
  constructor(message, code) {
    super(message, code);
    this.name = 'DeviceError';
  }
}

class OrtError extends DecibriError {
  constructor(message, code) {
    super(message, code);
    this.name = 'OrtError';
  }
}

class OrtPathError extends OrtError {
  constructor(message, code) {
    super(message, code);
    this.name = 'OrtPathError';
  }
}

// ─── Native error classification ─────────────────────────────────────────────

// The napi layer flattens every core DecibriError variant to a Status plus the
// variant's Display string (see bindings/node/src/lib.rs to_napi_error), so the
// message text is the only thing left to classify on. The prefixes below match
// the frozen core messages in crates/decibri/src/error.rs.

const DEVICE_CODES = [
  ['No microphone found matching', 'MICROPHONE_NOT_FOUND'],
  ['No speaker found matching', 'SPEAKER_NOT_FOUND'],
  ['Multiple devices match', 'MULTIPLE_DEVICES_MATCH'],
  ['No microphone found.', 'NO_MICROPHONE_FOUND'],
  ['No speaker found.', 'NO_SPEAKER_FOUND'],
  ['Selected device is not a valid microphone', 'NOT_AN_INPUT_DEVICE'],
  ['Failed to enumerate devices', 'DEVICE_ENUMERATION_FAILED'],
];

const ORT_CODES = [
  ['decibri: failed to initialize ONNX Runtime', 'ORT_INIT_FAILED'],
  ['Failed to load Silero VAD model from', 'VAD_MODEL_LOAD_FAILED'],
  ['Failed to load model from', 'MODEL_LOAD_FAILED'],
  ['Failed to create ort session builder', 'ORT_SESSION_BUILD_FAILED'],
  ['Failed to set ort threads', 'ORT_THREADS_CONFIG_FAILED'],
  ['Silero VAD inference failed', 'ORT_INFERENCE_FAILED'],
];

/**
 * Wrap an error thrown from a native DecibriBridge / DecibriOutputBridge call
 * into the appropriate JS error: a built-in for argument validation, or a
 * decibri catch-root class (with a `code`) for device and ONNX Runtime
 * failures. The message is preserved verbatim.
 *
 * @param {Error} err Error thrown from a native constructor.
 * @returns {Error} A RangeError, TypeError, or DecibriError subclass.
 */
function wrapNativeError(err) {
  const msg = (err && err.message) || String(err);

  // Argument validation stays as Node built-ins. `device index out of range`
  // is kept here (RangeError) so the index error type is the same whether it
  // is raised by the client-side bounds check or by the core.
  if (
    msg.startsWith('sample rate must be between') ||
    msg.startsWith('channels must be between') ||
    msg.startsWith('frames per buffer must be between') ||
    msg.startsWith('Silero VAD only supports') ||
    msg.startsWith('VAD threshold must be between') ||
    msg.startsWith('device index out of range') ||
    msg.startsWith('analysis requires VAD')
  ) {
    return new RangeError(msg);
  }
  if (
    msg.startsWith("dtype must be 'int16' or 'float32'") ||
    msg.startsWith("format must be 'int16' or 'float32'")
  ) {
    return new TypeError(msg);
  }

  // Device enumeration / selection failures.
  for (const [prefix, code] of DEVICE_CODES) {
    if (msg.startsWith(prefix)) return new DeviceError(msg, code);
  }

  // ONNX Runtime: a bad library path is an OrtPathError (subclass of OrtError);
  // OrtLoadFailed and OrtPathInvalid share this message prefix and the same
  // user-facing meaning. Other ORT failures are OrtError.
  if (msg.startsWith('decibri: failed to load ONNX Runtime from')) {
    return new OrtPathError(msg, 'ORT_LOAD_FAILED');
  }
  for (const [prefix, code] of ORT_CODES) {
    if (msg.startsWith(prefix)) return new OrtError(msg, code);
  }
  if (msg.startsWith('Failed to create') && msg.includes('tensor')) {
    return new OrtError(msg, 'ORT_TENSOR_CREATE_FAILED');
  }
  if (msg.startsWith('Failed to extract') && msg.includes('tensor')) {
    return new OrtError(msg, 'ORT_TENSOR_EXTRACT_FAILED');
  }

  // Runtime device/driver failure during streaming. Distinct from the
  // enumeration/selection DeviceError family above (which mirrors the core's
  // "Device errors"): this is the core's "Stream errors" DeviceFailed, surfaced
  // as a base DecibriError with a dedicated code.
  if (msg.startsWith('decibri: audio device error:')) {
    return new DecibriError(msg, 'DEVICE_FAILED');
  }
  // Non-ORT ONNX backend failure: the reserved backend catch-all, surfaced as a
  // base DecibriError with a dedicated code (not an OrtError).
  if (msg.startsWith('ONNX backend error from')) {
    return new DecibriError(msg, 'ONNX_BACKEND_FAILED');
  }

  // Any other error from the native constructor is still a decibri error.
  return new DecibriError(msg, 'DECIBRI_ERROR');
}

module.exports = {
  DecibriError,
  DeviceError,
  OrtError,
  OrtPathError,
  wrapNativeError,
};
