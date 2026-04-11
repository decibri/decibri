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

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: Silero VAD loads and captures (3 seconds)
// ═══════════════════════════════════════════════════════════════════════════════

async function testSileroCapture() {
  console.log('--- Group 1: Silero VAD capture (3 seconds) ---');

  const mic = new Decibri({
    sampleRate: 16000,
    channels: 1,
    vad: true,
    vadMode: 'silero',
    vadThreshold: 0.5,
  });

  let chunkCount = 0;
  let speechFired = false;
  let silenceFired = false;
  let errorOccurred = false;

  mic.on('data', () => { chunkCount++; });
  mic.on('speech', () => { speechFired = true; });
  mic.on('silence', () => { silenceFired = true; });
  mic.on('error', (err) => { console.log(`  ERROR: ${err.message}`); errorOccurred = true; });

  await sleep(3000);
  mic.stop();

  assert(!errorOccurred, 'no errors during Silero capture');
  assert(chunkCount > 0, `received ${chunkCount} chunks`);
  console.log(`  chunks: ${chunkCount}, speech: ${speechFired}, silence: ${silenceFired}`);
  console.log('  Group 1 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: Energy mode still works (regression)
// ═══════════════════════════════════════════════════════════════════════════════

async function testEnergyRegression() {
  console.log('--- Group 2: Energy VAD regression (2 seconds) ---');

  const mic = new Decibri({
    sampleRate: 16000,
    channels: 1,
    vad: true,
    vadMode: 'energy',
    vadThreshold: 0.001,
  });

  let speechFired = false;
  let chunkCount = 0;

  mic.on('data', () => { chunkCount++; });
  mic.on('speech', () => { speechFired = true; });

  await sleep(2000);
  mic.stop();

  assert(chunkCount > 0, `received ${chunkCount} chunks`);
  assert(speechFired, 'energy VAD speech event fired (threshold 0.001)');
  console.log('  Group 2 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3: Default vadMode is energy (backward compat)
// ═══════════════════════════════════════════════════════════════════════════════

async function testDefaultVadMode() {
  console.log('--- Group 3: Default vadMode is energy ---');

  // No vadMode specified, should default to energy
  const mic = new Decibri({
    sampleRate: 16000,
    channels: 1,
    vad: true,
    vadThreshold: 0.001,
  });

  let chunkCount = 0;
  mic.on('data', () => { chunkCount++; });

  await sleep(1000);
  mic.stop();

  assert(chunkCount > 0, `received ${chunkCount} chunks (default vadMode)`);
  console.log('  Group 3 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4: Invalid vadMode throws
// ═══════════════════════════════════════════════════════════════════════════════

async function testInvalidVadMode() {
  console.log('--- Group 4: Invalid vadMode ---');

  assertThrows(
    () => new Decibri({ vadMode: 'invalid' }),
    TypeError,
    "vadMode must be 'energy' or 'silero'"
  );

  console.log('  Group 4 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 5: Silero with float32 format
// ═══════════════════════════════════════════════════════════════════════════════

async function testSileroFloat32() {
  console.log('--- Group 5: Silero with float32 format (2 seconds) ---');

  const mic = new Decibri({
    sampleRate: 16000,
    channels: 1,
    format: 'float32',
    vad: true,
    vadMode: 'silero',
    vadThreshold: 0.5,
  });

  let chunkCount = 0;
  let errorOccurred = false;

  mic.on('data', (chunk) => {
    chunkCount++;
    if (chunkCount === 1) {
      assert(chunk.length === 6400, `float32 chunk is 6400 bytes (got ${chunk.length})`);
    }
  });
  mic.on('error', (err) => { console.log(`  ERROR: ${err.message}`); errorOccurred = true; });

  await sleep(2000);
  mic.stop();

  assert(!errorOccurred, 'no errors with Silero + float32');
  assert(chunkCount > 0, `received ${chunkCount} float32 chunks`);
  console.log('  Group 5 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 6: Missing model file error
// ═══════════════════════════════════════════════════════════════════════════════

async function testMissingModel() {
  console.log('--- Group 6: Missing model file ---');

  try {
    new Decibri({
      vad: true,
      vadMode: 'silero',
      modelPath: '/nonexistent/path/model.onnx',
    });
    console.log('  FAIL: should have thrown for missing model');
    failed++;
  } catch (e) {
    assert(e.message.includes('Silero VAD model not found'), `correct error message: ${e.message.slice(0, 60)}`);
  }

  console.log('  Group 6 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runner
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('decibri Silero VAD test suite\n');

  await testInvalidVadMode();
  await testMissingModel();
  await testDefaultVadMode();
  await testEnergyRegression();
  await testSileroCapture();
  await testSileroFloat32();

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
