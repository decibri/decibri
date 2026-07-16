'use strict';

/**
 * CI-safe tests. No microphone or speaker required.
 *
 * Extracts non-hardware assertions from test-api.js, test-capture.js,
 * test-output.js, and test-vad-silero.js. Safe to run on CI runners
 * that have no audio devices.
 */

const path = require('path');
const { Microphone, Speaker, inputDevices, outputDevices, version, DecibriError, DeviceError } = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));
const { wrapNativeError, OrtError } = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'errors.js'));
const pkg = require(path.join(__dirname, '..', 'npm', 'decibri', 'package.json'));

let passed = 0;
let failed = 0;

function assertThrows(fn, errorType, messagePart) {
  try {
    fn();
    console.log(`  FAIL: expected ${errorType.name} but no error thrown`);
    failed++;
  } catch (e) {
    if (!(e instanceof errorType)) {
      console.log(`  FAIL: expected ${errorType.name}, got ${e.constructor.name}: ${e.message}`);
      failed++;
    } else if (!e.message.includes(messagePart)) {
      console.log(`  FAIL: message mismatch`);
      console.log(`    expected to contain: ${messagePart}`);
      console.log(`    actual: ${e.message}`);
      failed++;
    } else {
      passed++;
    }
  }
}

function assert(condition, label) {
  if (condition) {
    passed++;
  } else {
    console.log(`  FAIL: ${label}`);
    failed++;
  }
}

async function assertRejects(fn, errorType, messagePart) {
  try {
    await fn();
    console.log(`  FAIL: expected ${errorType.name} rejection but the promise resolved`);
    failed++;
  } catch (e) {
    if (!(e instanceof errorType)) {
      console.log(`  FAIL: expected ${errorType.name} rejection, got ${e.constructor.name}: ${e.message}`);
      failed++;
    } else if (!e.message.includes(messagePart)) {
      console.log(`  FAIL: rejection message mismatch`);
      console.log(`    expected to contain: ${messagePart}`);
      console.log(`    actual: ${e.message}`);
      failed++;
    } else {
      passed++;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: Microphone constructor error messages
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 1: Microphone error messages ---');

// sampleRate
assertThrows(() => new Microphone({ sampleRate: 0 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new Microphone({ sampleRate: 999 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new Microphone({ sampleRate: 384001 }), RangeError, 'sample rate must be between 1000 and 384000');

// channels: mono only. A value below 1 is a plain range error; a value above
// 1 is rejected as multichannel (not silently downmixed to mono).
assertThrows(() => new Microphone({ channels: 0 }), RangeError, 'channels must be between 1 and 32');
assertThrows(() => new Microphone({ channels: 2 }), RangeError, 'multichannel capture is not supported; channels must be 1 (mono)');
assertThrows(() => new Microphone({ channels: 33 }), RangeError, 'multichannel capture is not supported; channels must be 1 (mono)');

// framesPerBuffer
assertThrows(() => new Microphone({ framesPerBuffer: 63 }), RangeError, 'frames per buffer must be between 64 and 65536');
assertThrows(() => new Microphone({ framesPerBuffer: 65537 }), RangeError, 'frames per buffer must be between 64 and 65536');

// dtype
assertThrows(() => new Microphone({ dtype: 'wav' }), TypeError, "dtype must be 'int16' or 'float32'");

// device name not found (delegated to the core)
assertThrows(
  () => new Microphone({ device: '__nonexistent__' }),
  DeviceError,
  'No microphone found matching "__nonexistent__"'
);

// device index out of range
assertThrows(
  () => new Microphone({ device: 99999 }),
  RangeError,
  'device index out of range'
);

// device by id, not found
assertThrows(
  () => new Microphone({ device: { id: '__nonexistent_id__' } }),
  DeviceError,
  'No microphone found matching "__nonexistent_id__"'
);

// device by id, wrong id type
assertThrows(
  () => new Microphone({ device: { id: 123 } }),
  TypeError,
  'device.id must be a string'
);

// boundary values that SHOULD work
try { const m = new Microphone({ sampleRate: 1000 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: sampleRate 1000 rejected: ${e.message}`); failed++; }

try { const m = new Microphone({ sampleRate: 384000 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: sampleRate 384000 rejected: ${e.message}`); failed++; }

try { const m = new Microphone({ framesPerBuffer: 64 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: framesPerBuffer 64 rejected: ${e.message}`); failed++; }

try { const m = new Microphone({ framesPerBuffer: 65536 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: framesPerBuffer 65536 rejected: ${e.message}`); failed++; }

console.log('  Group 1 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: Microphone.version() and Microphone.devices() format
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 2: Microphone static methods ---');

const ver = Microphone.version();
assert(typeof ver.decibri === 'string', 'version().decibri is string');
assert(typeof ver.audioBackend === 'string', 'version().audioBackend is string');
assert(ver.audioBackend.includes('cpal'), `version().audioBackend contains "cpal", got: ${ver.audioBackend}`);
assert(ver.binding === pkg.version, `version().binding equals package version ${pkg.version}, got: ${ver.binding}`);

const devices = Microphone.devices();
assert(Array.isArray(devices), 'devices() returns array');
// CI may have 0 devices. Verify structure if any exist.
if (devices.length > 0) {
  const d = devices[0];
  assert(typeof d.index === 'number', 'device.index is number');
  assert(typeof d.name === 'string', 'device.name is string');
  assert(typeof d.id === 'string', 'device.id is string');
  assert(typeof d.maxInputChannels === 'number', 'device.maxInputChannels is number');
  assert(typeof d.defaultSampleRate === 'number', 'device.defaultSampleRate is number');
  assert(typeof d.isDefault === 'boolean', 'device.isDefault is boolean');
} else {
  console.log('  (no audio devices found, skipping structure check)');
}

console.log('  Group 2 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3: Speaker error messages
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 3: Speaker error messages ---');

assertThrows(() => new Speaker({ sampleRate: 0 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new Speaker({ sampleRate: 384001 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new Speaker({ channels: 0 }), RangeError, 'channels must be between 1 and 32');
assertThrows(() => new Speaker({ channels: 33 }), RangeError, 'channels must be between 1 and 32');
assertThrows(() => new Speaker({ dtype: 'wav' }), TypeError, "dtype must be 'int16' or 'float32'");
assertThrows(
  () => new Speaker({ device: '__nonexistent__' }),
  DeviceError,
  'No speaker found matching "__nonexistent__"'
);
assertThrows(
  () => new Speaker({ device: 99999 }),
  RangeError,
  'device index out of range'
);

// device by id, not found
assertThrows(
  () => new Speaker({ device: { id: '__nonexistent_id__' } }),
  DeviceError,
  'No speaker found matching "__nonexistent_id__"'
);

// device by id, wrong id type
assertThrows(
  () => new Speaker({ device: { id: 123 } }),
  TypeError,
  'device.id must be a string'
);

// zero-byte write is a no-op
try {
  const s = new Speaker({ sampleRate: 16000, channels: 1 });
  s.write(Buffer.alloc(0));
  s.stop();
  passed++;
} catch (e) {
  console.log(`  FAIL: zero-byte write should be no-op: ${e.message}`);
  failed++;
}

console.log('  Group 3 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4: Speaker static methods
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 4: Speaker static methods ---');

const outVer = Speaker.version();
assert(outVer.audioBackend.includes('cpal'), 'Speaker.version().audioBackend contains cpal');
assert(outVer.binding === pkg.version, `Speaker.version().binding equals package version ${pkg.version}, got: ${outVer.binding}`);

const outDevices = Speaker.devices();
assert(Array.isArray(outDevices), 'Speaker.devices() returns array');
if (outDevices.length > 0) {
  const d = outDevices[0];
  assert(typeof d.id === 'string', 'output device.id is string');
  assert(typeof d.maxOutputChannels === 'number', 'output device has maxOutputChannels');
  assert(d.maxInputChannels === undefined, 'output device does NOT have maxInputChannels');
}

console.log('  Group 4 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 5: VAD option validation
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 5: VAD option validation ---');

// Legacy two-flag form is rejected with a migration error.
assertThrows(
  () => new Microphone({ vad: true }),
  TypeError,
  'vad: true is no longer supported'
);

// An unrecognized vad value is rejected.
assertThrows(
  () => new Microphone({ vad: 'loud' }),
  TypeError,
  'Invalid vad value'
);

// The flat vadThreshold / vadHoldoff options are removed; passing them is
// rejected with a migration message (proves the breaking change).
assertThrows(
  () => new Microphone({ vad: 'energy', vadThreshold: 0.5 }),
  TypeError,
  'no longer supported'
);
assertThrows(
  () => new Microphone({ vad: 'energy', vadHoldoff: 200 }),
  TypeError,
  'no longer supported'
);

// The vad config object: an unknown model is rejected.
assertThrows(
  () => new Microphone({ vad: { model: 'loud' } }),
  TypeError,
  'Invalid vad model'
);

// The vad config object: an out-of-range threshold / negative holdoff are
// rejected with the numeric-range RangeError convention.
assertThrows(
  () => new Microphone({ vad: { model: 'energy', threshold: 2 } }),
  RangeError,
  'between 0 and 1'
);
assertThrows(
  () => new Microphone({ vad: { model: 'energy', holdoffMs: -1 } }),
  RangeError,
  'non-negative'
);

// Missing model file (silero mode)
try {
  new Microphone({ vad: 'silero', modelPath: '/nonexistent/model.onnx' });
  console.log('  FAIL: missing model should throw');
  failed++;
} catch (e) {
  assert(e.message.includes('Silero VAD model not found'), 'missing model error message correct');
}

// Energy mode constructs without a model.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, vad: 'energy' });
  m.stop();
  passed++;
} catch (e) {
  console.log(`  FAIL: energy mode should work: ${e.message}`);
  failed++;
}

// The vad config object constructs and threads threshold + holdoff through.
try {
  const m = new Microphone({
    sampleRate: 16000,
    channels: 1,
    vad: { model: 'energy', threshold: 0.42, holdoffMs: 123 },
  });
  assert(m._vadThreshold === 0.42, 'vad object threshold reaches the wrapper');
  assert(m._vadHoldoff === 123, 'vad object holdoffMs reaches the wrapper');
  m.stop();
  passed++;
} catch (e) {
  console.log(`  FAIL: vad config object should work: ${e.message}`);
  failed++;
}

// The vad config object with the policy unset falls back to the mode defaults.
const mObjDefault = new Microphone({ sampleRate: 16000, channels: 1, vad: { model: 'energy' } });
assert(mObjDefault._vadThreshold === 0.01, 'vad object energy threshold default is 0.01');
assert(mObjDefault._vadHoldoff === 300, 'vad object holdoff default is 300');
mObjDefault.stop();

// vadScore: 0 when disabled, 0 before any audio in energy mode.
const mDisabled = new Microphone({ sampleRate: 16000, channels: 1 });
assert(mDisabled.vadScore === 0, 'vadScore is 0 when VAD disabled');
mDisabled.stop();

const mEnergy = new Microphone({ sampleRate: 16000, channels: 1, vad: 'energy' });
assert(mEnergy.vadScore === 0, 'vadScore starts at 0 in energy mode');
mEnergy.stop();

// overrunCount: readable accessor, 0 on a fresh (not-yet-overrun) stream.
const mOverrun = new Microphone({ sampleRate: 16000, channels: 1 });
assert(typeof mOverrun.overrunCount === 'number', 'overrunCount is a number');
assert(mOverrun.overrunCount === 0, 'overrunCount is 0 on a fresh stream');
mOverrun.stop();

console.log('  Group 5 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 6: Module-level free functions
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 6: module-level free functions ---');

assert(Array.isArray(inputDevices()), 'inputDevices() returns array');
assert(Array.isArray(outputDevices()), 'outputDevices() returns array');

const freeVer = version();
assert(typeof freeVer.decibri === 'string', 'free version().decibri is string');
assert(freeVer.audioBackend.includes('cpal'), 'free version().audioBackend contains cpal');
assert(freeVer.binding === pkg.version, `free version().binding equals package version ${pkg.version}, got: ${freeVer.binding}`);

console.log('  Group 6 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 7: Error class parity (instanceof + code)
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 7: error class parity ---');

// A delegated device-name miss is a DeviceError, catchable as DecibriError,
// and carries a stable code. Works on CI with zero devices (nothing matches).
try {
  new Microphone({ device: '__nonexistent__' });
  console.log('  FAIL: expected a device error');
  failed++;
} catch (e) {
  assert(e instanceof DeviceError, 'name miss is a DeviceError');
  assert(e instanceof DecibriError, 'DeviceError is a DecibriError');
  assert(e instanceof Error, 'DecibriError is an Error');
  assert(e.code === 'MICROPHONE_NOT_FOUND', `code is MICROPHONE_NOT_FOUND (got ${e.code})`);
  assert(e.name === 'DeviceError', `name is DeviceError (got ${e.name})`);
}

try {
  new Speaker({ device: '__nonexistent__' });
  console.log('  FAIL: expected a device error');
  failed++;
} catch (e) {
  assert(e instanceof DeviceError, 'speaker name miss is a DeviceError');
  assert(e.code === 'SPEAKER_NOT_FOUND', `code is SPEAKER_NOT_FOUND (got ${e.code})`);
}

// Argument validation stays a built-in and is NOT a DecibriError.
try {
  new Microphone({ sampleRate: 0 });
  console.log('  FAIL: expected a RangeError');
  failed++;
} catch (e) {
  assert(e instanceof RangeError, 'bad sampleRate is a RangeError');
  assert(!(e instanceof DecibriError), 'validation error is NOT a DecibriError');
}

// DeviceFailed and OnnxBackendFailed surface as dedicated, distinct errors
// (base DecibriError with a stable code), not the generic DECIBRI_ERROR
// fallback. These device/driver and backend failures only fire at runtime
// (mid-stream device loss, malformed ONNX), so wrapNativeError is exercised
// directly with the frozen core Display strings.
{
  const dev = wrapNativeError(new Error('decibri: audio device error: device unplugged'));
  assert(dev instanceof DecibriError, 'DeviceFailed maps to a DecibriError');
  assert(!(dev instanceof DeviceError), 'DeviceFailed is NOT a DeviceError (enumeration family)');
  assert(!(dev instanceof OrtError), 'DeviceFailed is NOT an OrtError');
  assert(dev.code === 'DEVICE_FAILED', `code is DEVICE_FAILED (got ${dev.code})`);
  assert(dev.message === 'decibri: audio device error: device unplugged', 'DeviceFailed message preserved verbatim');

  const onnx = wrapNativeError(new Error('ONNX backend error from coreml: boom'));
  assert(onnx instanceof DecibriError, 'OnnxBackendFailed maps to a DecibriError');
  assert(!(onnx instanceof OrtError), 'OnnxBackendFailed is NOT an OrtError (non-ORT backend)');
  assert(!(onnx instanceof DeviceError), 'OnnxBackendFailed is NOT a DeviceError');
  assert(onnx.code === 'ONNX_BACKEND_FAILED', `code is ONNX_BACKEND_FAILED (got ${onnx.code})`);
  assert(onnx.message === 'ONNX backend error from coreml: boom', 'OnnxBackendFailed message preserved verbatim');
}

console.log('  Group 7 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 8: Denoise option (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Construction resolves the bundled model path but does NOT load the model
// (the ONNX load happens at start(), the hardware tier). So the closed-set
// validation, the bundled-path resolution, and the off-by-default path are all
// CI-safe. The runtime model-load failure is classified by wrapNativeError
// (exercised directly with the frozen core Display string, like Group 7).

console.log('--- Group 8: Denoise option ---');

// A valid model name constructs (model loads later, at start()).
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, denoise: 'fastenhancer-t' });
  assert(m instanceof Microphone, "denoise: 'fastenhancer-t' constructs");
  m.stop();
} catch (e) {
  console.log(`  FAIL: denoise 'fastenhancer-t' construction rejected: ${e.message}`);
  failed++;
}

// An unrecognized model name is a clear TypeError, not a silent miss.
assertThrows(
  () => new Microphone({ denoise: 'whisper' }),
  TypeError,
  'Invalid denoise value'
);

// Off by default: no denoise key constructs identically to a plain mic.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1 });
  assert(m instanceof Microphone, 'no denoise key constructs (off by default)');
  m.stop();
} catch (e) {
  console.log(`  FAIL: no-denoise construction rejected: ${e.message}`);
  failed++;
}

// A denoise model-load failure surfaces as a dedicated OrtError with the
// 'MODEL_LOAD_FAILED' code, distinct from the VAD-named 'VAD_MODEL_LOAD_FAILED'
// and from the generic 'DECIBRI_ERROR' fallback. The load only fails at
// runtime (bad/missing model file), so wrapNativeError is exercised directly
// with the frozen core Display string.
{
  const model = wrapNativeError(new Error('Failed to load model from /x/fastenhancer_t.onnx: boom'));
  assert(model instanceof OrtError, 'ModelLoadFailed maps to an OrtError');
  assert(model instanceof DecibriError, 'ModelLoadFailed is a DecibriError');
  assert(model.code === 'MODEL_LOAD_FAILED', `code is MODEL_LOAD_FAILED (got ${model.code})`);
  assert(model.code !== 'VAD_MODEL_LOAD_FAILED', 'ModelLoadFailed is NOT the VAD-named code');
  assert(
    model.message === 'Failed to load model from /x/fastenhancer_t.onnx: boom',
    'ModelLoadFailed message preserved verbatim'
  );

  // Regression guard: the VAD model-load string still maps to its own code,
  // so the two model-load prefixes do not collide.
  const vad = wrapNativeError(new Error('Failed to load Silero VAD model from /x/silero_vad.onnx: boom'));
  assert(vad.code === 'VAD_MODEL_LOAD_FAILED', `VAD model load still VAD_MODEL_LOAD_FAILED (got ${vad.code})`);
}

console.log('  Group 8 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 8b: High-pass option (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// High-pass is pure DSP (no bundled file, no ONNX, no ORT), so the closed-set
// validation and the off-by-default path are fully CI-safe. The filter is built
// at start() like the other transform stages; its DSP response is covered by the
// core Rust tests. The cutoff is a closed numeric set (80, 100) designed to
// grow, so an out-of-set value is a clear RangeError, matching the numeric range
// checks on agc and limiter.

console.log('--- Group 8b: High-pass option ---');

// A valid cutoff in Hz constructs (both 80 and 100).
try {
  const m80 = new Microphone({ sampleRate: 16000, channels: 1, highpass: 80 });
  assert(m80 instanceof Microphone, 'highpass: 80 constructs');
  m80.stop();
  const m100 = new Microphone({ sampleRate: 16000, channels: 1, highpass: 100 });
  assert(m100 instanceof Microphone, 'highpass: 100 constructs');
  m100.stop();
} catch (e) {
  console.log(`  FAIL: numeric highpass construction rejected: ${e.message}`);
  failed++;
}

// A string cutoff is rejected; only the numeric set passes.
assertThrows(
  () => new Microphone({ highpass: '80hz' }),
  RangeError,
  'highpass must be one of: 80, 100'
);
// An out-of-set numeric cutoff lists the allowed set, not a silent miss.
assertThrows(
  () => new Microphone({ highpass: 200 }),
  RangeError,
  'highpass must be one of: 80, 100'
);

// Off by default: no highpass key constructs identically to a plain mic.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1 });
  assert(m instanceof Microphone, 'no highpass key constructs (off by default)');
  m.stop();
} catch (e) {
  console.log(`  FAIL: no-highpass construction rejected: ${e.message}`);
  failed++;
}

console.log('  Group 8b done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 8c: AGC option (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// AGC is pure DSP (no bundled file, no ONNX, no ORT), so the range validation
// and the off-by-default path are fully CI-safe. The engine is built at start()
// like the other transform stages; its temporal behaviour is covered by the core
// Rust tests. The target is a numeric dBFS level, so an out-of-range value is a
// RangeError (a numeric range violation), not a TypeError.

console.log('--- Group 8c: AGC option ---');

// A valid in-range target constructs.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, agc: -18 });
  assert(m instanceof Microphone, 'agc: -18 constructs');
  m.stop();
} catch (e) {
  console.log(`  FAIL: agc -18 construction rejected: ${e.message}`);
  failed++;
}

// An out-of-range target is a clear RangeError, below and above the range.
assertThrows(
  () => new Microphone({ agc: -100 }),
  RangeError,
  'agc target level must be between -40 and -3'
);
assertThrows(
  () => new Microphone({ agc: 0 }),
  RangeError,
  'agc target level must be between -40 and -3'
);

// Off by default: no agc key constructs identically to a plain mic.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1 });
  assert(m instanceof Microphone, 'no agc key constructs (off by default)');
  m.stop();
} catch (e) {
  console.log(`  FAIL: no-agc construction rejected: ${e.message}`);
  failed++;
}

console.log('  Group 8c done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 8d: limiter option (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The limiter is pure DSP (no bundled file, no ONNX, no ORT), so the range
// validation and the off-by-default path are fully CI-safe. The stage is built
// at start() like the other transform stages; its absolute-ceiling guarantee is
// covered by the core Rust tests. The ceiling is a numeric dBFS value, so an
// out-of-range value is a RangeError (a numeric range violation), not a TypeError.

console.log('--- Group 8d: limiter option ---');

// A valid in-range ceiling constructs.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, limiter: -1.0 });
  assert(m instanceof Microphone, 'limiter: -1.0 constructs');
  m.stop();
} catch (e) {
  console.log(`  FAIL: limiter -1.0 construction rejected: ${e.message}`);
  failed++;
}

// An out-of-range ceiling is a clear RangeError, below and above the range.
assertThrows(
  () => new Microphone({ limiter: -5.0 }),
  RangeError,
  'limiter ceiling must be between -3.0 and 0.0'
);
assertThrows(
  () => new Microphone({ limiter: 1.0 }),
  RangeError,
  'limiter ceiling must be between -3.0 and 0.0'
);

// Off by default: no limiter key constructs identically to a plain mic.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1 });
  assert(m instanceof Microphone, 'no limiter key constructs (off by default)');
  m.stop();
} catch (e) {
  console.log(`  FAIL: no-limiter construction rejected: ${e.message}`);
  failed++;
}

console.log('  Group 8d done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 8e: DC removal option (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// DC removal is pure DSP (a one-pole DC-blocking high-pass; no bundled file, no
// ONNX, no ORT), so construction and the off-by-default path are fully CI-safe.
// The stage is built at start() like the other transforms; its DSP response (a
// DC offset removed, length preserved, continuous across chunk boundaries) is
// covered by the core Rust tests. It is a plain bool toggle, so true turns it on
// and false or absence leaves it off, a byte-identical no-op.

console.log('--- Group 8e: DC removal option ---');

// dcRemoval: true constructs (the DC stage is built at start()).
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, dcRemoval: true });
  assert(m instanceof Microphone, 'dcRemoval: true constructs');
  m.stop();
} catch (e) {
  console.log(`  FAIL: dcRemoval: true construction rejected: ${e.message}`);
  failed++;
}

// dcRemoval: false is the explicit off form and constructs identically.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, dcRemoval: false });
  assert(m instanceof Microphone, 'dcRemoval: false constructs');
  m.stop();
} catch (e) {
  console.log(`  FAIL: dcRemoval: false construction rejected: ${e.message}`);
  failed++;
}

// Off by default: no dcRemoval key constructs identically to a plain mic.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1 });
  assert(m instanceof Microphone, 'no dcRemoval key constructs (off by default)');
  m.stop();
} catch (e) {
  console.log(`  FAIL: no-dcRemoval construction rejected: ${e.message}`);
  failed++;
}

console.log('  Group 8e done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 9: async open() factories (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Asserts what is deterministic: the factory resolves to a working instance and
// rejects with the right error class on bad input. The non-blocking property
// (that open() does not stall the event loop) is validated by design and by the
// hardware tier in tests/test-async-open.js; it is not asserted here.

async function asyncOpenTests() {
  console.log('--- Group 9: async open() factories ---');

  // Microphone.open() resolves to a working instance (default device, no model).
  // Mirrors the synchronous construction already exercised in Group 5; if the CI
  // runner can construct a Microphone synchronously, the async factory resolves
  // the same way.
  {
    const m = await Microphone.open({ sampleRate: 16000, channels: 1 });
    assert(m instanceof Microphone, 'Microphone.open() resolves to a Microphone');
    assert(m.isOpen === false, 'opened Microphone is not yet capturing');
    assert(m.vadScore === 0, 'opened Microphone vadScore starts at 0');
    m.stop();
  }

  // Speaker.open() resolves to a working instance.
  {
    const s = await Speaker.open({ sampleRate: 16000, channels: 1 });
    assert(s instanceof Speaker, 'Speaker.open() resolves to a Speaker');
    assert(s.isPlaying === false, 'opened Speaker is not yet playing');
    s.stop();
  }

  // Invalid options reject (not throw synchronously) with a built-in error.
  await assertRejects(
    () => Microphone.open({ sampleRate: 0 }),
    RangeError,
    'sample rate must be between 1000 and 384000'
  );
  await assertRejects(
    () => Microphone.open({ dtype: 'wav' }),
    TypeError,
    "dtype must be 'int16' or 'float32'"
  );
  await assertRejects(
    () => Speaker.open({ channels: 33 }),
    RangeError,
    'channels must be between 1 and 32'
  );

  // A missing Silero model rejects before any native work (wrapper-side check).
  await assertRejects(
    () => Microphone.open({ vad: 'silero', modelPath: '/nonexistent/model.onnx' }),
    Error,
    'Silero VAD model not found'
  );

  // A native open failure (unknown device name) rejects with a DeviceError that
  // carries the frozen message and code. Exercises the native compute -> reject
  // -> wrapNativeError path. Deterministic on CI: nothing matches the name.
  await assertRejects(
    () => Microphone.open({ device: '__nonexistent__' }),
    DeviceError,
    'No microphone found matching "__nonexistent__"'
  );
  await assertRejects(
    () => Speaker.open({ device: '__nonexistent__' }),
    DeviceError,
    'No speaker found matching "__nonexistent__"'
  );

  // Additive guarantee: the synchronous constructor still works unchanged.
  {
    const m = new Microphone({ sampleRate: 16000, channels: 1 });
    assert(m instanceof Microphone, 'sync new Microphone() still works alongside open()');
    m.stop();
    const s = new Speaker({ sampleRate: 16000, channels: 1 });
    assert(s instanceof Speaker, 'sync new Speaker() still works alongside open()');
    s.stop();
  }

  console.log('  Group 9 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 10: async write/drain (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Asserts what is deterministic without an output device: the no-op paths
// resolve and the methods return Promises. A real playback round trip through
// writeAsync/drainAsync (which opens the output stream) is in the hardware tier
// in tests/test-async-write-drain.js.

async function asyncWriteDrainTests() {
  console.log('--- Group 10: async write/drain ---');

  const s = new Speaker({ sampleRate: 16000, channels: 1 });

  // The methods return Promises.
  const wp = s.writeAsync(Buffer.alloc(0));
  assert(typeof wp.then === 'function', 'writeAsync() returns a Promise');
  const dp = s.drainAsync();
  assert(typeof dp.then === 'function', 'drainAsync() returns a Promise');

  // An empty write resolves without opening the stream (no device needed).
  await wp;
  assert(true, 'writeAsync(empty) resolves');

  // drainAsync with nothing written is a no-op that resolves immediately.
  await dp;
  assert(true, 'drainAsync() with no stream resolves');

  s.stop();

  // Additive guarantee: the synchronous write/drain path is unchanged. A
  // zero-byte sync write is still an accepted no-op, end() still drains.
  {
    const s2 = new Speaker({ sampleRate: 16000, channels: 1 });
    s2.write(Buffer.alloc(0));
    s2.stop();
    assert(true, 'sync write/stop still works alongside the async path');
  }

  console.log('  Group 10 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Summary (runs after the async groups resolve)
// ═══════════════════════════════════════════════════════════════════════════════

asyncOpenTests()
  .then(asyncWriteDrainTests)
  .then(() =>
    // The offline File source cases (conditioning, per-chunk VAD in file
    // time, whole-file analysis), sharing this run's counters.
    require('./test-file.js').fileTests({ assert, assertThrows, assertRejects })
  )
  .then(() => {
    console.log('═══════════════════════════════════════');
    console.log(`  Passed:  ${passed}`);
    console.log(`  Failed:  ${failed}`);
    console.log('═══════════════════════════════════════');
    if (failed > 0) {
      process.exit(1);
    }
  })
  .catch((err) => {
    console.error('  FATAL: async tests threw unexpectedly:', err);
    process.exit(1);
  });
