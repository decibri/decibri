'use strict';

/**
 * CI-safe tests. No microphone or speaker required.
 *
 * Extracts non-hardware assertions from test-api.js, test-capture.js,
 * test-output.js, and test-vad-silero.js. Safe to run on CI runners
 * that have no audio devices.
 */

const path = require('path');
const Decibri = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));
const { DecibriOutput } = Decibri;

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

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: Decibri constructor error messages
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 1: Decibri error messages ---');

// sampleRate
assertThrows(() => new Decibri({ sampleRate: 0 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new Decibri({ sampleRate: 999 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new Decibri({ sampleRate: 384001 }), RangeError, 'sample rate must be between 1000 and 384000');

// channels
assertThrows(() => new Decibri({ channels: 0 }), RangeError, 'channels must be between 1 and 32');
assertThrows(() => new Decibri({ channels: 33 }), RangeError, 'channels must be between 1 and 32');

// framesPerBuffer
assertThrows(() => new Decibri({ framesPerBuffer: 63 }), RangeError, 'frames per buffer must be between 64 and 65536');
assertThrows(() => new Decibri({ framesPerBuffer: 65537 }), RangeError, 'frames per buffer must be between 64 and 65536');

// format
assertThrows(() => new Decibri({ format: 'wav' }), TypeError, "format must be 'int16' or 'float32'");

// device name not found
assertThrows(
  () => new Decibri({ device: '__nonexistent__' }),
  TypeError,
  'No audio input device found matching "__nonexistent__"'
);

// device index out of range
assertThrows(
  () => new Decibri({ device: 99999 }),
  RangeError,
  'device index out of range'
);

// device by id, not found
assertThrows(
  () => new Decibri({ device: { id: '__nonexistent_id__' } }),
  TypeError,
  'No audio input device found matching "__nonexistent_id__"'
);

// device by id, wrong id type
assertThrows(
  () => new Decibri({ device: { id: 123 } }),
  TypeError,
  'device.id must be a string'
);

// boundary values that SHOULD work
try { const m = new Decibri({ sampleRate: 1000 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: sampleRate 1000 rejected: ${e.message}`); failed++; }

try { const m = new Decibri({ sampleRate: 384000 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: sampleRate 384000 rejected: ${e.message}`); failed++; }

try { const m = new Decibri({ framesPerBuffer: 64 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: framesPerBuffer 64 rejected: ${e.message}`); failed++; }

try { const m = new Decibri({ framesPerBuffer: 65536 }); m.stop(); passed++; }
catch (e) { console.log(`  FAIL: framesPerBuffer 65536 rejected: ${e.message}`); failed++; }

console.log('  Group 1 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: Decibri.version() and Decibri.devices() format
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 2: Decibri static methods ---');

const ver = Decibri.version();
assert(typeof ver.decibri === 'string', 'version().decibri is string');
assert(typeof ver.portaudio === 'string', 'version().portaudio is string');
assert(ver.portaudio.includes('cpal'), `version().portaudio contains "cpal", got: ${ver.portaudio}`);

const devices = Decibri.devices();
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
// Group 3: DecibriOutput error messages
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 3: DecibriOutput error messages ---');

assertThrows(() => new DecibriOutput({ sampleRate: 0 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new DecibriOutput({ sampleRate: 384001 }), RangeError, 'sample rate must be between 1000 and 384000');
assertThrows(() => new DecibriOutput({ channels: 0 }), RangeError, 'channels must be between 1 and 32');
assertThrows(() => new DecibriOutput({ channels: 33 }), RangeError, 'channels must be between 1 and 32');
assertThrows(() => new DecibriOutput({ format: 'wav' }), TypeError, "format must be 'int16' or 'float32'");
assertThrows(
  () => new DecibriOutput({ device: '__nonexistent__' }),
  TypeError,
  'No audio output device found matching "__nonexistent__"'
);
assertThrows(
  () => new DecibriOutput({ device: 99999 }),
  RangeError,
  'device index out of range'
);

// device by id, not found
assertThrows(
  () => new DecibriOutput({ device: { id: '__nonexistent_id__' } }),
  TypeError,
  'No audio output device found matching "__nonexistent_id__"'
);

// device by id, wrong id type
assertThrows(
  () => new DecibriOutput({ device: { id: 123 } }),
  TypeError,
  'device.id must be a string'
);

// zero-byte write is a no-op
try {
  const s = new DecibriOutput({ sampleRate: 16000, channels: 1 });
  s.write(Buffer.alloc(0));
  s.stop();
  passed++;
} catch (e) {
  console.log(`  FAIL: zero-byte write should be no-op: ${e.message}`);
  failed++;
}

console.log('  Group 3 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4: DecibriOutput static methods
// ═══════════════════════════════════════════════════════════════════════════════

console.log('--- Group 4: DecibriOutput static methods ---');

const outVer = DecibriOutput.version();
assert(outVer.portaudio.includes('cpal'), 'DecibriOutput.version().portaudio contains cpal');

const outDevices = DecibriOutput.devices();
assert(Array.isArray(outDevices), 'DecibriOutput.devices() returns array');
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

assertThrows(
  () => new Decibri({ vadMode: 'invalid' }),
  TypeError,
  "vadMode must be 'energy' or 'silero'"
);

// Missing model file
try {
  new Decibri({ vad: true, vadMode: 'silero', modelPath: '/nonexistent/model.onnx' });
  console.log('  FAIL: missing model should throw');
  failed++;
} catch (e) {
  assert(e.message.includes('Silero VAD model not found'), 'missing model error message correct');
}

// Default vadMode is energy (constructor succeeds without model)
try {
  const m = new Decibri({ sampleRate: 16000, channels: 1, vad: true });
  m.stop();
  passed++;
} catch (e) {
  console.log(`  FAIL: default vadMode should work: ${e.message}`);
  failed++;
}

console.log('  Group 5 done\n');

// ═══════════════════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════════════════

console.log('═══════════════════════════════════════');
console.log(`  Passed:  ${passed}`);
console.log(`  Failed:  ${failed}`);
console.log('═══════════════════════════════════════');

if (failed > 0) {
  process.exit(1);
}
