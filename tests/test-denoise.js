'use strict';

/**
 * Denoise capture tests (hardware tier).
 *
 * Requires a microphone AND the bundled ONNX Runtime, like test-vad-silero.js:
 * denoise loads its ONNX model at start() and runs it over the captured audio.
 * Run locally; not part of the CI subset (the deterministic denoise cases,
 * closed-set validation and the MODEL_LOAD_FAILED classification, live in
 * test-ci.js Group 10).
 */

const path = require('path');
const { Microphone } = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));

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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: denoise capture runs end to end (3 seconds)
// ═══════════════════════════════════════════════════════════════════════════════

async function testDenoiseCapture() {
  console.log('--- Group 1: denoise capture (3 seconds) ---');

  const mic = new Microphone({
    sampleRate: 16000,
    channels: 1,
    denoise: 'fastenhancer-t',
  });

  let chunkCount = 0;
  let totalBytes = 0;
  let errorOccurred = false;

  mic.on('data', (chunk) => { chunkCount++; totalBytes += chunk.length; });
  mic.on('error', (err) => { console.log(`  ERROR: ${err.message}`); errorOccurred = true; });

  await sleep(3000);
  mic.stop();

  // Let the flushed-tail callback and the deferred end-of-stream settle.
  await sleep(100);

  assert(!errorOccurred, 'no errors during denoise capture');
  assert(chunkCount > 0, `received ${chunkCount} denoised chunks`);
  assert(totalBytes > 0, `received ${totalBytes} bytes of denoised audio`);
  console.log(`  chunks: ${chunkCount}, bytes: ${totalBytes}`);
  console.log('  Group 1 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: denoise with float32 format
// ═══════════════════════════════════════════════════════════════════════════════

async function testDenoiseFloat32() {
  console.log('--- Group 2: denoise with float32 format (2 seconds) ---');

  const mic = new Microphone({
    sampleRate: 16000,
    channels: 1,
    dtype: 'float32',
    denoise: 'fastenhancer-t',
  });

  let chunkCount = 0;
  let errorOccurred = false;

  mic.on('data', () => { chunkCount++; });
  mic.on('error', (err) => { console.log(`  ERROR: ${err.message}`); errorOccurred = true; });

  await sleep(2000);
  mic.stop();
  await sleep(100);

  assert(!errorOccurred, 'no errors with denoise + float32');
  assert(chunkCount > 0, `received ${chunkCount} float32 denoised chunks`);
  console.log('  Group 2 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3: denoise + Silero VAD compose (VAD reads the pre-enhancement signal)
// ═══════════════════════════════════════════════════════════════════════════════

async function testDenoiseWithVad() {
  console.log('--- Group 3: denoise + Silero VAD (2 seconds) ---');

  const mic = new Microphone({
    sampleRate: 16000,
    channels: 1,
    denoise: 'fastenhancer-t',
    vad: 'silero',
    vadThreshold: 0.5,
  });

  let chunkCount = 0;
  let errorOccurred = false;

  mic.on('data', () => { chunkCount++; });
  mic.on('error', (err) => { console.log(`  ERROR: ${err.message}`); errorOccurred = true; });

  await sleep(2000);
  mic.stop();
  await sleep(100);

  assert(!errorOccurred, 'no errors with denoise + Silero VAD');
  assert(chunkCount > 0, `received ${chunkCount} chunks with denoise + VAD`);
  assert(typeof mic.vadScore === 'number' && mic.vadScore >= 0, 'vadScore is a non-negative number');
  console.log('  Group 3 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4: unknown denoise model rejected (deterministic)
// ═══════════════════════════════════════════════════════════════════════════════

async function testUnknownModelRejected() {
  console.log('--- Group 4: unknown denoise model rejected ---');

  assertThrows(
    () => new Microphone({ denoise: 'whisper' }),
    TypeError,
    'Invalid denoise value'
  );

  console.log('  Group 4 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 5: off by default (regression: a plain mic still captures)
// ═══════════════════════════════════════════════════════════════════════════════

async function testOffByDefault() {
  console.log('--- Group 5: denoise off by default (1 second) ---');

  const mic = new Microphone({ sampleRate: 16000, channels: 1 });

  let chunkCount = 0;
  let errorOccurred = false;
  mic.on('data', () => { chunkCount++; });
  mic.on('error', (err) => { console.log(`  ERROR: ${err.message}`); errorOccurred = true; });

  await sleep(1000);
  mic.stop();
  await sleep(100);

  assert(!errorOccurred, 'no errors with denoise off (default)');
  assert(chunkCount > 0, `received ${chunkCount} chunks with denoise off`);
  console.log('  Group 5 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runner
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('decibri denoise test suite\n');

  await testUnknownModelRejected();
  await testOffByDefault();
  await testDenoiseCapture();
  await testDenoiseFloat32();
  await testDenoiseWithVad();

  console.log('═══════════════════════════════════════');
  console.log(`  Passed:  ${passed}`);
  console.log(`  Failed:  ${failed}`);
  console.log('═══════════════════════════════════════');

  if (failed > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('FATAL:', err);
  process.exit(1);
});
