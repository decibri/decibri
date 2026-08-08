'use strict';

/**
 * CI-safe tests. No microphone or speaker required.
 *
 * Extracts non-hardware assertions from test-api.js, test-capture.js,
 * test-output.js, and test-vad-silero.js. Safe to run on CI runners
 * that have no audio devices.
 */

const path = require('path');
const { Microphone, Speaker, File, inputDevices, outputDevices, version, DecibriError, DeviceError } = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));
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
assertThrows(() => new Microphone({ channels: 0 }), RangeError, 'channels must be at least 1');
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
assertThrows(() => new Speaker({ channels: 0 }), RangeError, 'channels must be at least 1');
// Channels is bounded below only. A count above the former cap is no longer
// refused by the wrapper: it constructs where an output device is present, and
// fails for a device reason where none is (a CI runner). What it must never be
// is a channel-range refusal. Regression: a reintroduced cap refuses counts the
// device would serve.
for (const channels of [33, 64, 1024]) {
  try {
    new Speaker({ channels }).stop();
    assert(true, `channels: ${channels} is accepted at construction`);
  } catch (e) {
    assert(
      !(e instanceof RangeError),
      `channels: ${channels} is not refused as a range error (got ${e.constructor.name}: ${e.message})`
    );
  }
}
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

// underrunCount: readable accessor, 0 on a fresh (not-yet-underrun) stream.
const sUnderrun = new Speaker({ sampleRate: 16000, channels: 1 });
assert(typeof sUnderrun.underrunCount === 'number', 'underrunCount is a number');
assert(sUnderrun.underrunCount === 0, 'underrunCount is 0 on a fresh stream');
sUnderrun.stop();

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

  // An output device that cannot serve the requested channel count. Fires only
  // against real hardware, so wrapNativeError is exercised directly with the
  // frozen core Display string. It is a DecibriError with its own code, not a
  // RangeError: the count is not a bad argument shape, it is a capability the
  // device does not have.
  const chans = wrapNativeError(
    new Error('the output device does not support 124 output channels; it reports 2: refused')
  );
  assert(chans instanceof DecibriError, 'SpeakerChannelsUnsupported maps to a DecibriError');
  assert(!(chans instanceof RangeError), 'SpeakerChannelsUnsupported is NOT a RangeError');
  assert(!(chans instanceof DeviceError), 'SpeakerChannelsUnsupported is NOT a DeviceError (enumeration family)');
  assert(
    chans.code === 'SPEAKER_CHANNELS_UNSUPPORTED',
    `code is SPEAKER_CHANNELS_UNSUPPORTED (got ${chans.code})`
  );
  assert(
    chans.message === 'the output device does not support 124 output channels; it reports 2: refused',
    'SpeakerChannelsUnsupported message preserved verbatim'
  );

  const onnx = wrapNativeError(new Error('ONNX backend error from coreml: boom'));
  assert(onnx instanceof DecibriError, 'OnnxBackendFailed maps to a DecibriError');
  assert(!(onnx instanceof OrtError), 'OnnxBackendFailed is NOT an OrtError (non-ORT backend)');
  assert(!(onnx instanceof DeviceError), 'OnnxBackendFailed is NOT a DeviceError');
  assert(onnx.code === 'ONNX_BACKEND_FAILED', `code is ONNX_BACKEND_FAILED (got ${onnx.code})`);
  assert(onnx.message === 'ONNX backend error from coreml: boom', 'OnnxBackendFailed message preserved verbatim');
}

console.log('  Group 7 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 7b: every core error variant has its own Node identity
// ═══════════════════════════════════════════════════════════════════════════════
//
// The addon exposes one representative instance of every core variant compiled
// into it (name, code, Display string). The core assigns the code from an
// exhaustive match with no catch-all, so a new variant fails the Rust build
// until it has one; this group is what stops the Node table from falling
// behind that. It catches a reworded core message (the prefix stops matching),
// a new variant (no prefix exists), and a new variant whose message shares an
// existing prefix (it lands on the wrong code).

console.log('--- Group 7b: core error catalog coverage ---');

// Variants that deliberately keep a Node built-in instead of a decibri class.
// Argument validation is a RangeError / TypeError in Node, matching Node core
// and matching the wrapper's own pre-checks for the same inputs. Pinned by
// name so a new variant landing here has to be added deliberately.
const BUILTIN_VARIANTS = [
  'SampleRateOutOfRange',
  'ChannelsOutOfRange',
  'MultichannelNotSupported',
  'FramesPerBufferOutOfRange',
  'AgcTargetOutOfRange',
  'LimiterCeilingOutOfRange',
  'FlacCompressionOutOfRange',
  'InvalidFormat',
  'DeviceIndexOutOfRange',
  'VadSampleRateUnsupported',
  'VadThresholdOutOfRange',
  'AecSampleRateUnsupported',
  'VadNotConfigured',
];

// The codes that shipped before every variant was routed. None may change
// value: they are what a consumer branching on `err.code` already reads.
const PRE_EXISTING_CODES = [
  'MICROPHONE_NOT_FOUND',
  'SPEAKER_NOT_FOUND',
  'MULTIPLE_DEVICES_MATCH',
  'NO_MICROPHONE_FOUND',
  'NO_SPEAKER_FOUND',
  'NOT_AN_INPUT_DEVICE',
  'DEVICE_ENUMERATION_FAILED',
  'ORT_INIT_FAILED',
  'ORT_LOAD_FAILED',
  'ORT_SESSION_BUILD_FAILED',
  'ORT_THREADS_CONFIG_FAILED',
  'VAD_MODEL_LOAD_FAILED',
  'MODEL_LOAD_FAILED',
  'ORT_INFERENCE_FAILED',
  'ORT_TENSOR_CREATE_FAILED',
  'ORT_TENSOR_EXTRACT_FAILED',
  'DEVICE_FAILED',
  'ONNX_BACKEND_FAILED',
  'FILE_ENGAGED',
  'FILE_CONSUMED',
];

{
  const catalog = JSON.parse(require(path.join(__dirname, '..', 'npm', 'decibri', 'index.js')).__errorCatalog());
  assert(catalog.length > 0, 'the addon reports a non-empty error catalog');

  const codeOwners = new Map();
  const builtins = [];
  let mismatches = 0;

  for (const entry of catalog) {
    const wrapped = wrapNativeError(new Error(entry.message));
    if (wrapped instanceof DecibriError) {
      if (wrapped.code !== entry.code) {
        console.log(`  FAIL: ${entry.name} should carry ${entry.code}, got ${wrapped.code}`);
        mismatches++;
      }
      if (!codeOwners.has(wrapped.code)) codeOwners.set(wrapped.code, []);
      codeOwners.get(wrapped.code).push(entry.name);
    } else {
      builtins.push(entry.name);
      if (!(wrapped instanceof RangeError) && !(wrapped instanceof TypeError)) {
        console.log(`  FAIL: ${entry.name} is neither a DecibriError nor a Node built-in`);
        mismatches++;
      }
    }
  }
  assert(mismatches === 0, 'every core variant reaches the code its own core assigns');

  // The built-in set is exactly the pinned list: nothing new drifted into it,
  // and nothing that was a built-in quietly became a decibri class.
  assert(
    builtins.slice().sort().join(',') === BUILTIN_VARIANTS.slice().sort().join(','),
    `built-in variants match the pinned set (got ${builtins.sort().join(',')})`
  );

  // One variant, one identity. The only code owned by two variants is the
  // documented OrtPathInvalid collapse onto OrtLoadFailed's code.
  const shared = [...codeOwners].filter(([, names]) => names.length > 1);
  assert(shared.length === 1, `exactly one code is shared (got ${shared.length})`);
  assert(
    shared.length === 1 && shared[0][0] === 'ORT_LOAD_FAILED',
    'the shared code is ORT_LOAD_FAILED'
  );
  assert(
    shared.length === 1 && shared[0][1].slice().sort().join(',') === 'OrtLoadFailed,OrtPathInvalid',
    'the shared code belongs to the OrtLoadFailed / OrtPathInvalid pair'
  );

  // No catalog message falls to the unclassified bucket.
  assert(!codeOwners.has('DECIBRI_ERROR'), 'no core variant lands on DECIBRI_ERROR');

  // The codes that shipped before this change all still exist with the same
  // value. FILE_CONSUMED is authored in the napi layer rather than the core,
  // so it is checked through its own message.
  const produced = new Set(codeOwners.keys());
  produced.add(wrapNativeError(new Error('File already consumed')).code);
  const missing = PRE_EXISTING_CODES.filter((code) => !produced.has(code));
  assert(missing.length === 0, `every pre-existing code still exists (missing: ${missing.join(',')})`);
}

console.log('  Group 7b done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 7c: typed errors where there used to be untyped ones
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 7c: typed model, resampler, and enumeration failures ---');

// A missing model file is caught by the wrapper's pre-check, before the native
// bridge exists, so it never passes through wrapNativeError. It carries the
// class and code the deep load failure would.
{
  const missing = path.join(__dirname, 'no-such-model-b8f2.onnx');
  for (const construct of [
    () => new Microphone({ vad: { model: 'silero' }, modelPath: missing }),
    () => new File(path.join(__dirname, 'no-such-source-b8f2.wav'), {
      vad: { model: 'silero' },
      modelPath: missing,
    }),
  ]) {
    try {
      construct();
      console.log('  FAIL: a missing model path did not throw');
      failed++;
    } catch (e) {
      assert(e instanceof OrtError, `missing model path is an OrtError (got ${e.constructor.name})`);
      assert(e instanceof DecibriError, 'missing model path is a DecibriError');
      assert(e.code === 'VAD_MODEL_LOAD_FAILED', `missing model code is VAD_MODEL_LOAD_FAILED (got ${e.code})`);
      assert(e.message.includes('Silero VAD model not found at'), 'missing model message unchanged');
    }
  }
}

// ResampleConfigInvalid reaches the typed surface. It is defensive in the core
// (no accepted rate pair provokes it), so wrapNativeError is exercised with the
// frozen Display string.
{
  const resample = wrapNativeError(
    new Error('the requested sample rate conversion is not supported by the resampler')
  );
  assert(resample instanceof DecibriError, 'ResampleConfigInvalid maps to a DecibriError');
  assert(
    resample.code === 'RESAMPLE_CONFIG_INVALID',
    `code is RESAMPLE_CONFIG_INVALID (got ${resample.code})`
  );
}

// A device enumeration failure raised by the native listing reaches the caller
// as a DeviceError through both entry points, not as a raw napi error.
//
// The napi classes are frozen (their statics are non-writable and
// non-configurable), so the addon is stubbed at the module-cache level and the
// two wrappers are re-required against the stub. errors.js is left cached, so
// the classes the assertions below compare against are the same objects.
{
  const indexPath = require.resolve(path.join(__dirname, '..', 'npm', 'decibri', 'index.js'));
  const wrapperPaths = [
    require.resolve(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js')),
    require.resolve(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri-output.js')),
  ];
  const indexEntry = require.cache[indexPath];
  const realExports = indexEntry.exports;
  const boom = () => {
    throw new Error('Failed to enumerate devices: host unavailable');
  };
  const stub = Object.create(realExports);
  stub.DecibriBridge = { devices: boom };
  stub.DecibriOutputBridge = { devices: boom };

  try {
    indexEntry.exports = stub;
    for (const p of wrapperPaths) delete require.cache[p];
    const stubbed = require(wrapperPaths[0]);
    for (const [label, list] of [
      ['Microphone', () => stubbed.Microphone.devices()],
      ['Speaker', () => stubbed.Speaker.devices()],
    ]) {
      try {
        list();
        console.log(`  FAIL: ${label}.devices() did not throw`);
        failed++;
      } catch (e) {
        assert(e instanceof DeviceError, `${label}.devices() failure is a DeviceError`);
        assert(
          e.code === 'DEVICE_ENUMERATION_FAILED',
          `${label}.devices() code is DEVICE_ENUMERATION_FAILED (got ${e.code})`
        );
      }
    }
  } finally {
    indexEntry.exports = realExports;
    for (const p of wrapperPaths) delete require.cache[p];
    require(wrapperPaths[0]);
  }
}

console.log('  Group 7c done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 7d: the browser entry rejects the same input the same way
// ═══════════════════════════════════════════════════════════════════════════════
//
// One package, one option name, one documented meaning: a value the node entry
// refuses has to be refused by the browser entry with the same class and the
// same message. Asserted against both entries in one place so the two cannot
// drift apart silently. The browser module is plain JS and touches no browser
// global before start(), so its constructor runs under plain Node here.

console.log('--- Group 7d: browser and node entry validation parity ---');

const { Microphone: BrowserMicrophone } = require(
  path.join(__dirname, '..', 'npm', 'decibri', 'src', 'browser', 'decibri-browser.js')
);

const parityCases = [
  {
    label: "vad threshold 'high'",
    options: { vad: { model: 'energy', threshold: 'high' } },
    type: TypeError,
    message: 'vad threshold must be a number',
  },
  {
    label: 'vad threshold NaN',
    options: { vad: { model: 'energy', threshold: NaN } },
    type: TypeError,
    message: 'vad threshold must be a number',
  },
  {
    label: 'vad threshold 5',
    options: { vad: { model: 'energy', threshold: 5 } },
    type: RangeError,
    message: 'vad threshold must be between 0 and 1',
  },
  {
    label: "vad holdoffMs '300'",
    options: { vad: { model: 'energy', holdoffMs: '300' } },
    type: TypeError,
    message: 'vad holdoffMs must be a number',
  },
  {
    label: 'vad holdoffMs -1',
    options: { vad: { model: 'energy', holdoffMs: -1 } },
    type: RangeError,
    message: 'vad holdoffMs must be non-negative',
  },
  {
    label: 'sampleRate 0',
    options: { sampleRate: 0 },
    type: RangeError,
    message: 'sample rate must be between 1000 and 384000',
  },
  {
    label: 'sampleRate 500000',
    options: { sampleRate: 500000 },
    type: RangeError,
    message: 'sample rate must be between 1000 and 384000',
  },
  {
    label: 'channels 0',
    options: { channels: 0 },
    type: RangeError,
    message: 'channels must be at least 1',
  },
  {
    label: 'channels 2',
    options: { channels: 2 },
    type: RangeError,
    message: 'multichannel capture is not supported; channels must be 1 (mono)',
  },
];

for (const { label, options, type, message } of parityCases) {
  const thrown = {};
  for (const [entry, Ctor] of [['node', Microphone], ['browser', BrowserMicrophone]]) {
    try {
      new Ctor(options);
      console.log(`  FAIL: ${entry} entry accepted ${label}`);
      failed++;
    } catch (e) {
      thrown[entry] = e;
      assert(e instanceof type, `${entry} entry rejects ${label} with a ${type.name}`);
      assert(e.message === message, `${entry} entry message for ${label} is "${message}" (got "${e.message}")`);
    }
  }
  if (thrown.node && thrown.browser) {
    assert(
      thrown.node.constructor === thrown.browser.constructor,
      `both entries throw the same class for ${label} (${thrown.node.constructor.name} / ${thrown.browser.constructor.name})`
    );
    assert(
      thrown.node.message === thrown.browser.message,
      `both entries throw the same message for ${label}`
    );
  }
}

console.log('  Group 7d done\n');

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
// Group 8f: AEC option (deterministic, no hardware required)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The echo canceller is pure DSP (no bundled file, no ONNX, no ORT), so the
// option validation, the reference-push input contract, and the metrics-off
// path are fully CI-safe. The canceller is built at start() like the other
// stages; its cancellation behaviour is covered by the core Rust tests. The
// model set is owned by the canceller (AecModel::from_str in the native
// layer), so the unknown-model cases assert the canceller's own message text:
// a wrapper-side copy of the list would break them the day the set grows.

console.log('--- Group 8f: AEC option ---');

// The short form names the model and constructs; the canceller is built at
// start(). Catches the option failing to reach the native config.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, aec: 'tau' });
  assert(m instanceof Microphone, "aec: 'tau' constructs");
  m.stop();
} catch (e) {
  console.log(`  FAIL: aec 'tau' construction rejected: ${e.message}`);
  failed++;
}

// The object form constructs with every field at a boundary value, both ends.
// Catches an off-by-one in any of the three range checks, and a suppression
// value dropped on the way to the native config.
try {
  const low = new Microphone({
    sampleRate: 16000,
    channels: 1,
    aec: { model: 'tau', tailMs: 16, suppression: 'off', referenceSampleRate: 1000 },
  });
  assert(low instanceof Microphone, 'aec object form constructs at the low boundaries');
  low.stop();
  const high = new Microphone({
    sampleRate: 16000,
    channels: 1,
    aec: { model: 'tau', tailMs: 500, suppression: 'conservative', referenceSampleRate: 384000 },
  });
  assert(high instanceof Microphone, 'aec object form constructs at the high boundaries');
  high.stop();
} catch (e) {
  console.log(`  FAIL: aec object-form construction rejected: ${e.message}`);
  failed++;
}

// An unknown model is rejected by the canceller's own parse, so the error
// carries the canceller's message naming the accepted set, wrapped as a
// DecibriError with the AEC_CONFIG_INVALID code. Catches the model list being
// copied into the wrapper and drifting from the canceller's.
try {
  const m = new Microphone({ aec: 'tao' });
  m.stop();
  console.log('  FAIL: unknown aec model accepted');
  failed++;
} catch (e) {
  assert(e instanceof DecibriError, 'unknown aec model is a DecibriError');
  assert(
    e.code === 'AEC_CONFIG_INVALID',
    `unknown aec model code is AEC_CONFIG_INVALID (got ${e.code})`
  );
  assert(
    e.message.includes("model must be one of: 'tau'; got 'tao'"),
    'unknown aec model message carries the canceller text'
  );
}

// The same delegation holds through the object form.
assertThrows(
  () => new Microphone({ aec: { model: 'speex' } }),
  DecibriError,
  "model must be one of: 'tau'; got 'speex'"
);

// tailMs outside 16..500 is a RangeError, below and above (the boundary
// values were accepted above); a non-number is a TypeError. Catches the range
// check missing from the wrapper.
assertThrows(
  () => new Microphone({ aec: { model: 'tau', tailMs: 15 } }),
  RangeError,
  'aec tailMs must be between 16 and 500'
);
assertThrows(
  () => new Microphone({ aec: { model: 'tau', tailMs: 501 } }),
  RangeError,
  'aec tailMs must be between 16 and 500'
);
assertThrows(
  () => new Microphone({ aec: { model: 'tau', tailMs: 'long' } }),
  TypeError,
  'aec tailMs must be a number'
);

// suppression outside the two-policy set is a TypeError naming the set.
assertThrows(
  () => new Microphone({ aec: { model: 'tau', suppression: 'aggressive' } }),
  TypeError,
  "aec suppression must be 'conservative' or 'off'"
);

// referenceSampleRate outside 1000..384000 is a RangeError, below and above;
// a non-number is a TypeError.
assertThrows(
  () => new Microphone({ aec: { model: 'tau', referenceSampleRate: 999 } }),
  RangeError,
  'aec referenceSampleRate must be between 1000 and 384000'
);
assertThrows(
  () => new Microphone({ aec: { model: 'tau', referenceSampleRate: 384001 } }),
  RangeError,
  'aec referenceSampleRate must be between 1000 and 384000'
);
assertThrows(
  () => new Microphone({ aec: { model: 'tau', referenceSampleRate: 'high' } }),
  TypeError,
  'aec referenceSampleRate must be a number'
);

// referenceChannels constructs at 1, 2, and the config field's own u16
// ceiling: the count declares the shape of the caller's buffer, so no smaller
// maximum exists to enforce. Catches the option failing to reach the native
// config, and a fixed maximum creeping in below the field's own type.
try {
  const stereoRef = new Microphone({
    sampleRate: 16000,
    channels: 1,
    aec: { model: 'tau', referenceChannels: 2 },
  });
  assert(stereoRef instanceof Microphone, 'aec referenceChannels: 2 constructs');
  stereoRef.stop();
  const wideRef = new Microphone({
    sampleRate: 16000,
    channels: 1,
    aec: { model: 'tau', referenceChannels: 65535 },
  });
  assert(wideRef instanceof Microphone, 'aec referenceChannels: 65535 constructs');
  wideRef.stop();
} catch (e) {
  console.log(`  FAIL: aec referenceChannels construction rejected: ${e.message}`);
  failed++;
}

// referenceChannels below 1 is a RangeError; a non-number is a TypeError;
// past the config field's own u16 the native layer names that container
// bound, also a RangeError.
assertThrows(
  () => new Microphone({ aec: { model: 'tau', referenceChannels: 0 } }),
  RangeError,
  'aec referenceChannels must be at least 1'
);
assertThrows(
  () => new Microphone({ aec: { model: 'tau', referenceChannels: 'stereo' } }),
  TypeError,
  'aec referenceChannels must be a number'
);
assertThrows(
  () => new Microphone({ aec: { model: 'tau', referenceChannels: 65536 } }),
  RangeError,
  'aec referenceChannels must be at most 65535'
);

// A non-string model in the object form, and a non-option aec value, are
// TypeErrors from the wrapper before any native work.
assertThrows(() => new Microphone({ aec: { model: 42 } }), TypeError, 'Invalid aec model');
assertThrows(() => new Microphone({ aec: true }), TypeError, 'Invalid aec value');
assertThrows(() => new Microphone({ aec: ['tau'] }), TypeError, 'Invalid aec value');

// AEC narrows the accepted capture rate to the canceller's own window: a rate
// fine without AEC is rejected with it on, from the core through the native
// constructor, and the window's boundary rates construct. Catches the window
// check disappearing from the construction path.
assertThrows(
  () => new Microphone({ sampleRate: 96000, aec: 'tau' }),
  RangeError,
  'echo cancellation only supports sample rates 8000 to 48000'
);
try {
  const plain = new Microphone({ sampleRate: 96000, channels: 1 });
  assert(plain instanceof Microphone, 'sampleRate 96000 still constructs without aec');
  plain.stop();
  const lo = new Microphone({ sampleRate: 8000, channels: 1, aec: 'tau' });
  assert(lo instanceof Microphone, 'aec constructs at the 8000 window boundary');
  lo.stop();
  const hi = new Microphone({ sampleRate: 48000, channels: 1, aec: 'tau' });
  assert(hi instanceof Microphone, 'aec constructs at the 48000 window boundary');
  hi.stop();
} catch (e) {
  console.log(`  FAIL: aec rate-window boundary construction rejected: ${e.message}`);
  failed++;
}

// The reference push accepts every input shape Speaker.write accepts (Buffer,
// any TypedArray, DataView) and never throws while capture is not running:
// before start() and after stop() a valid push is a no-op, not an error.
// Catches the push becoming stateful and the input normalization narrowing.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, aec: 'tau' });
  assert(
    m.pushAecReference(Buffer.alloc(640)) === undefined,
    'pushAecReference accepts a Buffer before start()'
  );
  assert(
    m.pushAecReference(new Uint8Array(640)) === undefined,
    'pushAecReference accepts a Uint8Array'
  );
  assert(
    m.pushAecReference(new Int16Array(320)) === undefined,
    'pushAecReference accepts an Int16Array'
  );
  assert(
    m.pushAecReference(new DataView(new ArrayBuffer(640))) === undefined,
    'pushAecReference accepts a DataView'
  );
  assert(
    m.pushAecReference(Buffer.alloc(0)) === undefined,
    'pushAecReference accepts an empty Buffer'
  );
  m.stop();
  assert(
    m.pushAecReference(Buffer.alloc(640)) === undefined,
    'pushAecReference after stop() is a no-op, not an error'
  );
} catch (e) {
  console.log(`  FAIL: pushAecReference rejected a Speaker.write input shape: ${e.message}`);
  failed++;
}

// A float32 microphone takes the same shapes, the bytes read per its dtype.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, dtype: 'float32', aec: 'tau' });
  assert(
    m.pushAecReference(new Float32Array(320)) === undefined,
    'pushAecReference accepts a Float32Array on a float32 microphone'
  );
  m.stop();
} catch (e) {
  console.log(`  FAIL: float32 pushAecReference rejected: ${e.message}`);
  failed++;
}

// A non-buffer input is a TypeError whatever the capture state, so a wrong
// call site fails loud instead of silently pushing nothing.
{
  const m = new Microphone({ sampleRate: 16000, channels: 1, aec: 'tau' });
  assertThrows(
    () => m.pushAecReference('not audio'),
    TypeError,
    'pushAecReference requires a Buffer'
  );
  assertThrows(
    () => m.pushAecReference([0, 0, 0]),
    TypeError,
    'pushAecReference requires a Buffer'
  );
  m.stop();
}

// AEC absent by default: no aec key enables no stage, so the metrics read
// null and a push is a no-op. Catches the option becoming implicitly on.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1 });
  assert(m.aecMetrics() === null, 'no aec key reads null metrics (off by default)');
  assert(
    m.pushAecReference(Buffer.alloc(64)) === undefined,
    'push on a microphone without aec is a no-op'
  );
  m.stop();
} catch (e) {
  console.log(`  FAIL: aec off-by-default path rejected: ${e.message}`);
  failed++;
}

// With aec set but capture not running, the metrics read null rather than a
// zeroed report, so a caller cannot mistake a dead stream for a quiet one.
try {
  const m = new Microphone({ sampleRate: 16000, channels: 1, aec: 'tau' });
  assert(m.aecMetrics() === null, 'aec metrics read null before start()');
  m.stop();
} catch (e) {
  console.log(`  FAIL: pre-start aec metrics rejected: ${e.message}`);
  failed++;
}

console.log('  Group 8f done\n');

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
    () => Speaker.open({ channels: 0 }),
    RangeError,
    'channels must be at least 1'
  );
  // The async factory shares _prepareOptions with the constructor, so it is
  // bounded below only in the same way. Regression: the two validation paths
  // drifting apart.
  try {
    (await Speaker.open({ channels: 33 })).stop();
    assert(true, 'Speaker.open accepts channels: 33');
  } catch (e) {
    assert(
      !(e instanceof RangeError),
      `Speaker.open channels: 33 is not refused as a range error (got ${e.constructor.name}: ${e.message})`
    );
  }

  // A missing Silero model rejects before any native work (wrapper-side check).
  await assertRejects(
    () => Microphone.open({ vad: 'silero', modelPath: '/nonexistent/model.onnx' }),
    Error,
    'Silero VAD model not found'
  );

  // The aec option validates identically through the async factory: the
  // wrapper's checks and the canceller's own model parse reject the same
  // inputs with the same classes and messages the synchronous constructor
  // raises in Group 8f, and a valid option resolves. Catches the async path
  // skipping the shared validation.
  {
    const m = await Microphone.open({ sampleRate: 16000, channels: 1, aec: 'tau' });
    assert(m instanceof Microphone, "Microphone.open() resolves with aec: 'tau'");
    assert(m.aecMetrics() === null, 'async-opened microphone reads null aec metrics before start()');
    m.stop();
  }
  await assertRejects(
    () => Microphone.open({ aec: { model: 'tau', tailMs: 15 } }),
    RangeError,
    'aec tailMs must be between 16 and 500'
  );
  await assertRejects(
    () => Microphone.open({ aec: { model: 'tau', suppression: 'max' } }),
    TypeError,
    "aec suppression must be 'conservative' or 'off'"
  );
  await assertRejects(
    () => Microphone.open({ aec: 'tao' }),
    DecibriError,
    "model must be one of: 'tau'; got 'tao'"
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
// Group 11: every path that delivers an error to a consumer carries a decibri
// class and code
// ═══════════════════════════════════════════════════════════════════════════════
//
// The four delivery paths (Microphone 'error', Speaker write, Speaker end, and
// the async writeAsync/drainAsync pair) can only fail against a real device that
// dies mid-stream, so each one is driven here by swapping the native handle for
// a stub that throws the frozen core Display string a live failure would carry.
// Assertions are by class and code, never by message text.

// The core's Display strings for the two failures a consumer most needs to tell
// apart, and a driver failure the core stashes and reports on the next call.
const DEVICE_LOST = 'decibri: audio device error: device unplugged';
const PERMISSION_DENIED = 'Microphone permission denied. Check system settings.';
const STAGE_FAILURE = 'Silero VAD inference failed: tensor shape mismatch';

// A native handle stub. `failing` names the call that fails, so every other call
// behaves normally: 'write', 'drain', 'startCallback' (the pump reports a
// mid-stream failure), 'start' (the synchronous throw), 'async' (both async
// methods reject), or 'none' (healthy throughout).
function nativeStub(message, failing) {
  const boom = () => {
    throw new Error(message);
  };
  const calls = { stopped: false };
  return {
    calls,
    write: failing === 'write' ? boom : () => {},
    drain: failing === 'drain' ? boom : () => {},
    stop() {
      calls.stopped = true;
    },
    start(cb) {
      if (failing === 'start') boom();
      // The real pump delivers to this callback from its own thread, never
      // synchronously inside start().
      if (failing === 'startCallback') setImmediate(() => cb(new Error(message)));
    },
    writeAsync: () => (failing === 'none' ? Promise.resolve() : Promise.reject(new Error(message))),
    drainAsync: () => (failing === 'none' ? Promise.resolve() : Promise.reject(new Error(message))),
    get isPlaying() {
      return false;
    },
    get underrunCount() {
      return 0;
    },
    get vadProbability() {
      return 0;
    },
  };
}

/** Resolve with the first 'error' emitted, or reject if none arrives. */
function nextError(emitter) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('no error event was emitted')), 2000);
    emitter.once('error', (err) => {
      clearTimeout(timer);
      resolve(err);
    });
  });
}

async function errorDeliveryTests() {
  console.log('--- Group 11: typed errors on every delivery path ---');

  // PATH 1a: a device lost mid-capture, delivered to the pump callback.
  {
    const mic = new Microphone({ sampleRate: 16000, channels: 1 });
    mic._native = nativeStub(DEVICE_LOST, 'startCallback');
    mic._read();
    const err = await nextError(mic);
    assert(err instanceof DecibriError, 'mic stream failure is a DecibriError');
    assert(err.code === 'DEVICE_FAILED', `mic stream failure code is DEVICE_FAILED (got ${err.code})`);
    assert(mic._started === false, 'a stream failure clears _started');
  }

  // PATH 1b: capture that fails to start at all. The synchronous throw out of
  // native start() reaches the consumer on the same 'error' event, carrying the
  // permission denial's own code.
  {
    const mic = new Microphone({ sampleRate: 16000, channels: 1 });
    mic._native = nativeStub(PERMISSION_DENIED, 'start');
    mic._read();
    const err = await nextError(mic);
    assert(err instanceof DecibriError, 'a failed capture start is a DecibriError');
    assert(err.code === 'PERMISSION_DENIED', `failed-start code is PERMISSION_DENIED (got ${err.code})`);
    assert(mic._started === false, 'a failed start clears _started');
  }

  // PATH 1c: a conditioning stage that fails mid-capture. The native pump
  // forwards the stage's own failure rather than ending the stream silently,
  // and it reaches the consumer with its own identity, not the device one.
  {
    const mic = new Microphone({ sampleRate: 16000, channels: 1 });
    mic._native = nativeStub(STAGE_FAILURE, 'startCallback');
    mic._read();
    const err = await nextError(mic);
    assert(err instanceof OrtError, 'a conditioning failure mid-capture is an OrtError');
    assert(
      err.code === 'ORT_INFERENCE_FAILED',
      `conditioning failure code is ORT_INFERENCE_FAILED (got ${err.code})`
    );
  }

  // PATH 2: a device lost under a synchronous write.
  {
    const spk = new Speaker({ sampleRate: 16000, channels: 1 });
    spk._native = nativeStub(DEVICE_LOST, 'write');
    spk.write(Buffer.alloc(64));
    const err = await nextError(spk);
    assert(err instanceof DecibriError, 'speaker write failure is a DecibriError');
    assert(err.code === 'DEVICE_FAILED', `speaker write failure code is DEVICE_FAILED (got ${err.code})`);
  }

  // PATH 3: a device lost while end() flushes the tail. The device must still
  // be released on the failing path.
  {
    const spk = new Speaker({ sampleRate: 16000, channels: 1 });
    const native = nativeStub(DEVICE_LOST, 'drain');
    spk._native = native;
    spk.end();
    const err = await nextError(spk);
    assert(err instanceof DecibriError, 'speaker end() failure is a DecibriError');
    assert(err.code === 'DEVICE_FAILED', `speaker end() failure code is DEVICE_FAILED (got ${err.code})`);
    assert(native.calls.stopped === true, 'end() still stops the stream when the drain fails');
  }

  // PATH 4: the async pair. These already wrapped before this change; they are
  // pinned here so all four paths are asserted together. This exercises the JS
  // wrap only: the native tasks that now report a stashed failure cannot be
  // driven without a device that actually dies.
  {
    const spk = new Speaker({ sampleRate: 16000, channels: 1 });
    spk._native = nativeStub(DEVICE_LOST, 'async');
    await assertRejects(() => spk.writeAsync(Buffer.alloc(64)), DecibriError, 'audio device error');
    await assertRejects(() => spk.drainAsync(), DecibriError, 'audio device error');
    const rejected = await spk.drainAsync().catch((e) => e);
    assert(rejected.code === 'DEVICE_FAILED', `drainAsync code is DEVICE_FAILED (got ${rejected.code})`);
    spk.destroy();
  }

  // THE REGRESSION THAT MATTERS: a deliberate close is not a device fault.
  // A healthy native ends cleanly, with 'finish' and no 'error'.
  {
    const spk = new Speaker({ sampleRate: 16000, channels: 1 });
    const native = nativeStub(DEVICE_LOST, 'none');
    spk._native = native;
    const outcome = await new Promise((resolve) => {
      spk.once('error', (e) => resolve({ errored: true, e }));
      spk.once('finish', () => resolve({ errored: false }));
      spk.end(Buffer.alloc(64));
    });
    assert(outcome.errored === false, 'a deliberate end() is not reported as a device failure');
    assert(native.calls.stopped === true, 'a clean end() stops the stream');
  }

  // A deliberate microphone stop() ends the stream cleanly with no 'error'.
  {
    const mic = new Microphone({ sampleRate: 16000, channels: 1 });
    mic._native = nativeStub(DEVICE_LOST, 'none');
    mic._read();
    mic.stop();
    const outcome = await new Promise((resolve) => {
      mic.once('error', (e) => resolve({ errored: true, e }));
      mic.once('end', () => resolve({ errored: false }));
      mic.resume();
    });
    assert(outcome.errored === false, 'a deliberate mic stop() is not reported as a device failure');
  }

  console.log('  Group 11 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Summary (runs after the async groups resolve)
// ═══════════════════════════════════════════════════════════════════════════════

asyncOpenTests()
  .then(asyncWriteDrainTests)
  .then(errorDeliveryTests)
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
