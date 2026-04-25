'use strict';

/**
 * Wrap a native error from decibri-node as a typed JS error class.
 *
 * The native binding's to_napi_error uses Status::InvalidArg for caller-input
 * errors and Status::GenericFailure for system errors. This wrapper inspects
 * err.code and matches on the frozen error message prefixes from the Rust
 * core (see CLAUDE.md:28-30, error messages are part of the frozen contract)
 * to classify as TypeError (wrong kind of value) or RangeError (value out of
 * allowed range). Non-InvalidArg errors are passed through unchanged.
 *
 * The shim covers all InvalidArg errors, not just a specific subset: even
 * though only some code paths currently reach Rust's to_napi_error, the
 * mapping stays complete and consistent so future paths get typed errors
 * without additional shim updates.
 *
 * @param {Error} err Error thrown from a native DecibriBridge /
 *   DecibriOutputBridge call.
 * @returns {Error|TypeError|RangeError} Re-wrapped error preserving code.
 */
function wrapNativeError(err) {
  if (err.code !== 'InvalidArg') return err;

  const msg = err.message;
  const isRange =
    msg.startsWith('sample rate must be between') ||
    msg.startsWith('channels must be between') ||
    msg.startsWith('frames per buffer must be between') ||
    msg.startsWith('Silero VAD only supports') ||
    msg.startsWith('VAD threshold must be between') ||
    msg.startsWith('device index out of range');

  const Wrapper = isRange ? RangeError : TypeError;
  const wrapped = new Wrapper(msg);
  wrapped.code = err.code;
  return wrapped;
}

module.exports = { wrapNativeError };
