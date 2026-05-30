'use strict';

/**
 * Async open() factory tests: Microphone.open() and Speaker.open().
 *
 * These factories perform the blocking open work (the Silero model load for the
 * microphone; device resolution for both) on the native thread pool and resolve
 * to a ready instance, mirroring the Python AsyncMicrophone.open() /
 * AsyncSpeaker.open() factories. The synchronous constructors are unchanged.
 *
 * Group 1 (deterministic, no hardware) asserts resolve-to-instance and the
 * reject paths. Groups 3 and 4 exercise a real instance end to end and need a
 * microphone/speaker; like the other hardware suites they are local-only.
 * Environmental failures (no audio device, missing ORT dylib) are classified as
 * skips, not failures, so the deterministic assertions still gate the suite.
 */

const path = require('path');
const {
  Microphone,
  Speaker,
  DeviceError,
  OrtError,
} = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));

let passed = 0;
let failed = 0;
let skipped = 0;

function assert(condition, label) {
  if (condition) {
    passed++;
  } else {
    console.log(`  FAIL: ${label}`);
    failed++;
  }
}

async function assertRejects(fn, errorType, messagePart, label) {
  try {
    await fn();
    console.log(`  FAIL: ${label}: expected ${errorType.name} rejection but resolved`);
    failed++;
  } catch (e) {
    if (!(e instanceof errorType)) {
      console.log(`  FAIL: ${label}: expected ${errorType.name}, got ${e.constructor.name}: ${e.message}`);
      failed++;
    } else if (!e.message.includes(messagePart)) {
      console.log(`  FAIL: ${label}: message mismatch`);
      console.log(`    expected to contain: ${messagePart}`);
      console.log(`    actual: ${e.message}`);
      failed++;
    } else {
      passed++;
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Treat a rejection that is clearly environmental (no audio device, or an ORT
// dylib that is not present in a dev workspace) as a skip rather than a failure.
function isEnvironmental(err) {
  return err instanceof DeviceError || err instanceof OrtError;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: deterministic resolve and reject (no hardware assumptions)
// ═══════════════════════════════════════════════════════════════════════════════

async function testDeterministic() {
  console.log('--- Group 1: deterministic resolve / reject ---');

  // Resolves to a constructed, not-yet-running instance.
  try {
    const m = await Microphone.open({ sampleRate: 16000, channels: 1 });
    assert(m instanceof Microphone, 'Microphone.open() resolves to a Microphone');
    assert(m.isOpen === false, 'opened Microphone is not yet capturing');
    assert(m.vadScore === 0, 'opened Microphone vadScore starts at 0');
    m.stop();
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: Microphone.open() default device unavailable: ${e.message}`);
      skipped++;
    } else {
      console.log(`  FAIL: Microphone.open() should resolve: ${e.constructor.name}: ${e.message}`);
      failed++;
    }
  }

  try {
    const s = await Speaker.open({ sampleRate: 16000, channels: 1 });
    assert(s instanceof Speaker, 'Speaker.open() resolves to a Speaker');
    assert(s.isPlaying === false, 'opened Speaker is not yet playing');
    s.stop();
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: Speaker.open() default device unavailable: ${e.message}`);
      skipped++;
    } else {
      console.log(`  FAIL: Speaker.open() should resolve: ${e.constructor.name}: ${e.message}`);
      failed++;
    }
  }

  // Invalid options reject (async), not throw synchronously.
  await assertRejects(
    () => Microphone.open({ sampleRate: 0 }),
    RangeError,
    'sample rate must be between 1000 and 384000',
    'Microphone.open bad sampleRate'
  );
  await assertRejects(
    () => Microphone.open({ dtype: 'wav' }),
    TypeError,
    "dtype must be 'int16' or 'float32'",
    'Microphone.open bad dtype'
  );
  await assertRejects(
    () => Microphone.open({ vad: true }),
    TypeError,
    'vad: true is no longer supported',
    'Microphone.open legacy vad'
  );
  await assertRejects(
    () => Speaker.open({ channels: 33 }),
    RangeError,
    'channels must be between 1 and 32',
    'Speaker.open bad channels'
  );

  // Missing Silero model rejects before any native work.
  await assertRejects(
    () => Microphone.open({ vad: 'silero', modelPath: '/nonexistent/model.onnx' }),
    Error,
    'Silero VAD model not found',
    'Microphone.open missing model'
  );

  // Native open failure (unknown device) rejects with a DeviceError carrying the
  // frozen message and code. Exercises native compute -> reject -> wrapNativeError.
  await assertRejects(
    () => Microphone.open({ device: '__nonexistent__' }),
    DeviceError,
    'No microphone found matching "__nonexistent__"',
    'Microphone.open unknown device'
  );
  await assertRejects(
    () => Speaker.open({ device: '__nonexistent__' }),
    DeviceError,
    'No speaker found matching "__nonexistent__"',
    'Speaker.open unknown device'
  );

  console.log('  Group 1 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: additive guarantee (sync constructors unchanged)
// ═══════════════════════════════════════════════════════════════════════════════

async function testAdditive() {
  console.log('--- Group 2: additive (sync constructors unchanged) ---');

  try {
    const m = new Microphone({ sampleRate: 16000, channels: 1 });
    assert(m instanceof Microphone, 'sync new Microphone() still constructs');
    m.stop();
    const s = new Speaker({ sampleRate: 16000, channels: 1 });
    assert(s instanceof Speaker, 'sync new Speaker() still constructs');
    s.stop();
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: sync construction needs a device: ${e.message}`);
      skipped++;
    } else {
      console.log(`  FAIL: sync construction broke: ${e.constructor.name}: ${e.message}`);
      failed++;
    }
  }

  console.log('  Group 2 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3: end-to-end capture from an opened instance (hardware)
// ═══════════════════════════════════════════════════════════════════════════════

async function testCaptureCycle() {
  console.log('--- Group 3: capture from Microphone.open() (2 seconds) ---');

  let mic;
  try {
    mic = await Microphone.open({ sampleRate: 16000, channels: 1 });
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: no microphone available: ${e.message}`);
      skipped++;
      console.log('  Group 3 done\n');
      return;
    }
    throw e;
  }

  let chunks = 0;
  mic.on('data', () => { chunks++; });

  let errored = false;
  mic.on('error', (err) => { console.log(`  mic error: ${err.message}`); errored = true; });

  await sleep(2000);
  mic.stop();
  await sleep(100);

  assert(!errored, 'no error during capture from opened instance');
  assert(chunks > 0, `received at least one chunk from opened instance (got ${chunks})`);

  console.log('  Group 3 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4: playback from an opened instance (hardware)
// ═══════════════════════════════════════════════════════════════════════════════

async function testPlaybackCycle() {
  console.log('--- Group 4: playback from Speaker.open() (440Hz, 1 second) ---');

  let speaker;
  try {
    speaker = await Speaker.open({ sampleRate: 16000, channels: 1 });
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: no speaker available: ${e.message}`);
      skipped++;
      console.log('  Group 4 done\n');
      return;
    }
    throw e;
  }

  const samples = 16000;
  const buffer = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i++) {
    const sample = Math.sin(2 * Math.PI * 440 * (i / 16000)) * 0.3;
    buffer.writeInt16LE(Math.round(sample * 32767), i * 2);
  }

  await new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    speaker.write(buffer);
    speaker.end();
    speaker.on('finish', () => {
      assert(!speaker.isPlaying, 'isPlaying false after finish (opened instance)');
      finish();
    });
    speaker.on('error', (err) => {
      console.log(`  FAIL: playback error: ${err.message}`);
      failed++;
      finish();
    });
  });

  console.log('  Group 4 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runner
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('decibri async open() test suite\n');

  await testDeterministic();
  await testAdditive();
  await testCaptureCycle();
  await testPlaybackCycle();

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
