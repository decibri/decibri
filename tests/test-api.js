'use strict';

const path = require('path');
const Decibri = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));

let passed = 0;
let failed = 0;
let skipped = 0;

function assertThrows(fn, errorType, messagePart) {
  try {
    fn();
    console.log(`  FAIL: expected ${errorType.name} but no error thrown`);
    failed++;
    return false;
  } catch (e) {
    if (!(e instanceof errorType)) {
      console.log(`  FAIL: expected ${errorType.name}, got ${e.constructor.name}: ${e.message}`);
      failed++;
      return false;
    }
    if (!e.message.includes(messagePart)) {
      console.log(`  FAIL: message mismatch`);
      console.log(`    expected to contain: ${messagePart}`);
      console.log(`    actual: ${e.message}`);
      failed++;
      return false;
    }
    passed++;
    return true;
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function captureFor(mic, ms) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    mic.on('data', (chunk) => chunks.push(chunk));
    mic.on('error', reject);
    setTimeout(() => {
      mic.stop();
      resolve(chunks);
    }, ms);
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: Error messages (no hardware needed)
// ═══════════════════════════════════════════════════════════════════════════════

async function testErrors() {
  console.log('--- Group 1: Error messages ---');

  // sampleRate
  assertThrows(() => new Decibri({ sampleRate: 0 }), RangeError, 'sampleRate must be between 1000 and 384000');
  assertThrows(() => new Decibri({ sampleRate: 999 }), RangeError, 'sampleRate must be between 1000 and 384000');
  assertThrows(() => new Decibri({ sampleRate: 384001 }), RangeError, 'sampleRate must be between 1000 and 384000');

  // channels
  assertThrows(() => new Decibri({ channels: 0 }), RangeError, 'channels must be between 1 and 32');
  assertThrows(() => new Decibri({ channels: 33 }), RangeError, 'channels must be between 1 and 32');

  // framesPerBuffer
  assertThrows(() => new Decibri({ framesPerBuffer: 63 }), RangeError, 'framesPerBuffer must be between 64 and 65536');
  assertThrows(() => new Decibri({ framesPerBuffer: 65537 }), RangeError, 'framesPerBuffer must be between 64 and 65536');

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

  // boundary values that SHOULD work (no throw)
  try {
    const m1 = new Decibri({ sampleRate: 1000 }); m1.stop();
    passed++;
  } catch (e) {
    console.log(`  FAIL: sampleRate 1000 should be accepted: ${e.message}`);
    failed++;
  }
  try {
    const m2 = new Decibri({ sampleRate: 384000 }); m2.stop();
    passed++;
  } catch (e) {
    console.log(`  FAIL: sampleRate 384000 should be accepted: ${e.message}`);
    failed++;
  }
  try {
    const m3 = new Decibri({ framesPerBuffer: 64 }); m3.stop();
    passed++;
  } catch (e) {
    console.log(`  FAIL: framesPerBuffer 64 should be accepted: ${e.message}`);
    failed++;
  }
  try {
    const m4 = new Decibri({ framesPerBuffer: 65536 }); m4.stop();
    passed++;
  } catch (e) {
    console.log(`  FAIL: framesPerBuffer 65536 should be accepted: ${e.message}`);
    failed++;
  }

  console.log('  Group 1 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: format: 'float32'
// ═══════════════════════════════════════════════════════════════════════════════

async function testFloat32() {
  console.log('--- Group 2: float32 format (2 seconds) ---');

  const mic = new Decibri({ sampleRate: 16000, channels: 1, format: 'float32' });
  const chunks = await captureFor(mic, 2000);

  assert(chunks.length > 0, 'received at least one float32 chunk');

  const expectedSize = 1600 * 1 * 4; // 6400 bytes
  const allCorrect = chunks.every(c => c.length === expectedSize);
  assert(allCorrect, `all chunks are ${expectedSize} bytes (got ${chunks[0]?.length})`);

  // Verify data is interpretable as Float32Array
  if (chunks.length > 0) {
    const c = chunks[0];
    const floats = new Float32Array(c.buffer, c.byteOffset, c.length / 4);
    const allFinite = Array.from(floats).every(Number.isFinite);
    assert(allFinite, 'float32 samples are all finite numbers');
  }

  console.log(`  ${chunks.length} chunks, ${chunks[0]?.length} bytes each`);
  console.log('  Group 2 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3: Device selection by name
// ═══════════════════════════════════════════════════════════════════════════════

async function testDeviceByName() {
  console.log('--- Group 3: Device selection by name (1 second) ---');

  const devices = Decibri.devices();
  const defaultDev = devices.find(d => d.isDefault);
  if (!defaultDev) {
    console.log('  SKIP: no default device found');
    skipped++;
    console.log('  Group 3 done\n');
    return;
  }

  // Use a substring of the device name
  const namePart = defaultDev.name.slice(0, Math.max(3, Math.floor(defaultDev.name.length / 2)));
  console.log(`  Using name substring: "${namePart}" (from "${defaultDev.name}")`);

  try {
    const mic = new Decibri({ sampleRate: 16000, channels: 1, device: namePart });
    const chunks = await captureFor(mic, 1000);
    assert(chunks.length > 0, `captured ${chunks.length} chunks via device name`);
  } catch (e) {
    // Multiple matches is acceptable if the substring is too broad
    if (e instanceof TypeError && e.message.includes('Multiple devices match')) {
      console.log('  SKIP: name substring matched multiple devices (expected on some systems)');
      skipped++;
    } else {
      console.log(`  FAIL: ${e.message}`);
      failed++;
    }
  }

  console.log('  Group 3 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4: Device selection by index
// ═══════════════════════════════════════════════════════════════════════════════

async function testDeviceByIndex() {
  console.log('--- Group 4: Device selection by index (1 second) ---');

  const devices = Decibri.devices();
  if (devices.length === 0) {
    console.log('  SKIP: no devices found');
    skipped++;
    console.log('  Group 4 done\n');
    return;
  }

  const mic = new Decibri({ sampleRate: 16000, channels: 1, device: 0 });
  const chunks = await captureFor(mic, 1000);
  assert(chunks.length > 0, `captured ${chunks.length} chunks via device index 0`);

  console.log('  Group 4 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 5: Non-default framesPerBuffer
// ═══════════════════════════════════════════════════════════════════════════════

async function testFramesPerBuffer() {
  console.log('--- Group 5: framesPerBuffer 800 (2 seconds) ---');

  const mic = new Decibri({ sampleRate: 16000, channels: 1, framesPerBuffer: 800 });
  const chunks = await captureFor(mic, 2000);

  assert(chunks.length > 0, 'received at least one chunk');

  const expectedSize = 800 * 1 * 2; // 1600 bytes
  const allCorrect = chunks.every(c => c.length === expectedSize);
  assert(allCorrect, `all chunks are ${expectedSize} bytes (got ${chunks[0]?.length})`);

  console.log(`  ${chunks.length} chunks, ${chunks[0]?.length} bytes each`);
  console.log('  Group 5 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 6: VAD events
// ═══════════════════════════════════════════════════════════════════════════════

async function testVAD() {
  console.log('--- Group 6: VAD events (3 seconds) ---');

  const mic = new Decibri({
    sampleRate: 16000,
    channels: 1,
    vad: true,
    vadThreshold: 0.001, // very low — ambient noise should trigger
  });

  let speechFired = false;
  let silenceFired = false;
  let dataCount = 0;

  mic.on('speech', () => { speechFired = true; });
  mic.on('silence', () => { silenceFired = true; });
  mic.on('data', () => { dataCount++; });

  await sleep(3000);
  mic.stop();

  assert(dataCount > 0, `received ${dataCount} data chunks`);
  assert(speechFired, 'speech event fired (threshold 0.001)');
  // silence may or may not fire depending on ambient conditions — don't assert
  console.log(`  speech: ${speechFired}, silence: ${silenceFired}, chunks: ${dataCount}`);
  console.log('  Group 6 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 7: Multiple instances
// ═══════════════════════════════════════════════════════════════════════════════

async function testMultipleInstances() {
  console.log('--- Group 7: Multiple instances ---');

  // Instance 1
  const mic1 = new Decibri({ sampleRate: 16000, channels: 1 });
  const chunks1 = await captureFor(mic1, 1000);
  assert(chunks1.length > 0, `instance 1: ${chunks1.length} chunks`);
  assert(!mic1.isOpen, 'instance 1: isOpen false after stop');

  // Instance 2 (after instance 1 is stopped)
  const mic2 = new Decibri({ sampleRate: 16000, channels: 1 });
  const chunks2 = await captureFor(mic2, 1000);
  assert(chunks2.length > 0, `instance 2: ${chunks2.length} chunks`);
  assert(!mic2.isOpen, 'instance 2: isOpen false after stop');

  console.log('  Group 7 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 8: Backpressure (best-effort)
// ═══════════════════════════════════════════════════════════════════════════════

async function testBackpressure() {
  console.log('--- Group 8: Backpressure (best-effort) ---');

  const mic = new Decibri({ sampleRate: 16000, channels: 1, highWaterMark: 1 });

  let backpressureFired = false;
  mic.on('backpressure', () => { backpressureFired = true; });

  // Attach data listener but don't drain fast — just let the buffer fill
  mic.on('data', () => { /* intentionally slow consumer */ });

  await sleep(3000);
  mic.stop();

  if (backpressureFired) {
    console.log('  backpressure event fired');
    passed++;
  } else {
    console.log('  SKIP: backpressure did not fire (timing-dependent)');
    skipped++;
  }

  console.log('  Group 8 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runner
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('decibri API test suite\n');

  await testErrors();
  await testFloat32();
  await testDeviceByName();
  await testDeviceByIndex();
  await testFramesPerBuffer();
  await testVAD();
  await testMultipleInstances();
  await testBackpressure();

  console.log('═══════════════════════════════════════');
  console.log(`  Passed:  ${passed}`);
  console.log(`  Failed:  ${failed}`);
  console.log(`  Skipped: ${skipped}`);
  console.log('═══════════════════════════════════════');

  if (failed > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('FATAL:', err);
  process.exit(1);
});
