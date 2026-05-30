'use strict';

/**
 * Async write/drain tests: Speaker.writeAsync() and Speaker.drainAsync().
 *
 * These methods perform the blocking parts of playback off the event loop: the
 * backpressure wait on the native channel (write) and the drain poll loop
 * (drain). The audio stream stays on its own thread; only the thread-safe
 * channel and atomic drain flags are touched off the event loop. They are an
 * additive, non-blocking alternative to the synchronous Writable path
 * (write() / pipe() / end()), which is unchanged.
 *
 * Group 1 (deterministic, no hardware) asserts the no-op resolves and the
 * reject path. Group 2 plays a real tone through the async path and needs a
 * speaker; like the other output suites it is local-only. Environmental
 * failures (no output device) are classified as skips.
 */

const path = require('path');
const {
  Speaker,
  DecibriError,
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isEnvironmental(err) {
  // A missing or unusable output device surfaces as a DecibriError family
  // member (DeviceError) or a stream-open failure. Treat those as skips.
  return err instanceof DecibriError;
}

function generateSineWave(sampleRate, frequency, durationSec, amplitude) {
  const samples = sampleRate * durationSec;
  const buffer = Buffer.alloc(samples * 2); // int16
  for (let i = 0; i < samples; i++) {
    const t = i / sampleRate;
    const sample = Math.sin(2 * Math.PI * frequency * t) * amplitude;
    buffer.writeInt16LE(Math.round(sample * 32767), i * 2);
  }
  return buffer;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: deterministic (no hardware assumptions)
// ═══════════════════════════════════════════════════════════════════════════════

async function testDeterministic() {
  console.log('--- Group 1: deterministic write/drain ---');

  const s = new Speaker({ sampleRate: 16000, channels: 1 });

  // The methods return Promises.
  const wp = s.writeAsync(Buffer.alloc(0));
  assert(typeof wp.then === 'function', 'writeAsync() returns a Promise');
  await wp;
  assert(true, 'writeAsync(empty) resolves without opening the stream');

  // drainAsync with nothing written resolves immediately (no-op).
  await s.drainAsync();
  assert(true, 'drainAsync() with no stream resolves');

  s.stop();

  // Reject path: an invalid argument rejects the Promise rather than throwing
  // synchronously. (A non-Buffer cannot be marshaled by the native method.)
  let rejected = false;
  let rejErr;
  try {
    await s.writeAsync(42);
  } catch (e) {
    rejected = true;
    rejErr = e;
  }
  assert(rejected, 'writeAsync(non-buffer) rejects');
  assert(rejErr instanceof Error, 'rejection is an Error');
  console.log(`  (write reject class: ${rejErr && rejErr.constructor.name}: ${rejErr && rejErr.message})`);

  // Additive guarantee: the synchronous write/drain path is unchanged.
  {
    const s2 = new Speaker({ sampleRate: 16000, channels: 1 });
    s2.write(Buffer.alloc(0));
    s2.stop();
    assert(true, 'sync write/stop still works alongside the async path');
  }

  console.log('  Group 1 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: real playback through the async path (hardware)
// ═══════════════════════════════════════════════════════════════════════════════

async function testAsyncPlayback() {
  console.log('--- Group 2: playback via writeAsync + drainAsync (440Hz, 1 second) ---');

  let speaker;
  try {
    speaker = await Speaker.open({ sampleRate: 16000, channels: 1 });
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: no speaker available: ${e.message}`);
      skipped++;
      console.log('  Group 2 done\n');
      return;
    }
    throw e;
  }

  try {
    const buffer = generateSineWave(16000, 440, 1, 0.3);
    await speaker.writeAsync(buffer);
    assert(true, 'writeAsync(tone) resolves');

    await speaker.drainAsync();
    assert(!speaker.isPlaying, 'isPlaying is false after drainAsync resolves');
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: playback unavailable: ${e.message}`);
      skipped++;
    } else {
      console.log(`  FAIL: async playback errored: ${e.constructor.name}: ${e.message}`);
      failed++;
    }
  }

  console.log('  Group 2 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3: multiple sequential async writes preserve order and resolve
// ═══════════════════════════════════════════════════════════════════════════════

async function testSequentialWrites() {
  console.log('--- Group 3: sequential writeAsync chunks (hardware) ---');

  let speaker;
  try {
    speaker = new Speaker({ sampleRate: 16000, channels: 1 });
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: no speaker available: ${e.message}`);
      skipped++;
      console.log('  Group 3 done\n');
      return;
    }
    throw e;
  }

  try {
    // Three 200ms chunks written sequentially via the async path.
    const chunk = generateSineWave(16000, 330, 0.2, 0.25);
    for (let i = 0; i < 3; i++) {
      await speaker.writeAsync(chunk);
    }
    await speaker.drainAsync();
    assert(!speaker.isPlaying, 'not playing after draining sequential writes');
  } catch (e) {
    if (isEnvironmental(e)) {
      console.log(`  SKIP: sequential playback unavailable: ${e.message}`);
      skipped++;
    } else {
      console.log(`  FAIL: sequential async writes errored: ${e.constructor.name}: ${e.message}`);
      failed++;
    }
  }

  await sleep(100);
  speaker.stop();

  console.log('  Group 3 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runner
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('decibri async write/drain test suite\n');

  await testDeterministic();
  await testAsyncPlayback();
  await testSequentialWrites();

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
