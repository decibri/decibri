'use strict';

const path = require('path');
const { Microphone, Speaker, DeviceError, DecibriError } = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));
const pkg = require(path.join(__dirname, '..', 'npm', 'decibri', 'package.json'));

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

function generateSineWaveFloat32(sampleRate, frequency, durationSec, amplitude) {
  const samples = sampleRate * durationSec;
  const buffer = Buffer.alloc(samples * 4); // float32
  for (let i = 0; i < samples; i++) {
    const t = i / sampleRate;
    const sample = Math.sin(2 * Math.PI * frequency * t) * amplitude;
    buffer.writeFloatLE(sample, i * 4);
  }
  return buffer;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 1: Error messages (no hardware needed)
// ═══════════════════════════════════════════════════════════════════════════════

async function testErrors() {
  console.log('--- Group 1: Error messages ---');

  // sampleRate
  assertThrows(() => new Speaker({ sampleRate: 0 }), RangeError, 'sample rate must be between 1000 and 384000');
  assertThrows(() => new Speaker({ sampleRate: 999 }), RangeError, 'sample rate must be between 1000 and 384000');
  assertThrows(() => new Speaker({ sampleRate: 384001 }), RangeError, 'sample rate must be between 1000 and 384000');

  // channels: bounded below only. A high count is left for the device to
  // answer when the stream opens (Group 8), not refused here.
  assertThrows(() => new Speaker({ channels: 0 }), RangeError, 'channels must be between 1 and 32');

  // dtype
  assertThrows(() => new Speaker({ dtype: 'wav' }), TypeError, "dtype must be 'int16' or 'float32'");

  // device name not found (delegated to the core)
  assertThrows(
    () => new Speaker({ device: '__nonexistent__' }),
    DeviceError,
    'No speaker found matching "__nonexistent__"'
  );

  // device index out of range
  assertThrows(
    () => new Speaker({ device: 99999 }),
    RangeError,
    'device index out of range'
  );

  // zero-byte write is a no-op (not a crash)
  try {
    const speaker = new Speaker({ sampleRate: 16000, channels: 1 });
    speaker.write(Buffer.alloc(0));
    speaker.stop();
    passed++;
  } catch (e) {
    console.log(`  FAIL: zero-byte write should be no-op: ${e.message}`);
    failed++;
  }

  // boundary values that SHOULD work
  try {
    const s1 = new Speaker({ sampleRate: 1000 }); s1.stop();
    passed++;
  } catch (e) {
    console.log(`  FAIL: sampleRate 1000 should be accepted: ${e.message}`);
    failed++;
  }
  try {
    const s2 = new Speaker({ sampleRate: 384000 }); s2.stop();
    passed++;
  } catch (e) {
    console.log(`  FAIL: sampleRate 384000 should be accepted: ${e.message}`);
    failed++;
  }

  console.log('  Group 1 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 2: Output device enumeration
// ═══════════════════════════════════════════════════════════════════════════════

async function testDeviceEnumeration() {
  console.log('--- Group 2: Output device enumeration ---');

  const devices = Speaker.devices();
  console.log(`  Found ${devices.length} output device(s)`);
  assert(Array.isArray(devices), 'devices() returns an array');
  assert(devices.length > 0, 'at least one output device');

  if (devices.length > 0) {
    const d = devices[0];
    assert(typeof d.index === 'number', 'device has index');
    assert(typeof d.name === 'string', 'device has name');
    assert(typeof d.maxOutputChannels === 'number', 'device has maxOutputChannels');
    assert(typeof d.defaultSampleRate === 'number', 'device has defaultSampleRate');
    assert(typeof d.isDefault === 'boolean', 'device has isDefault');
    // Verify it's NOT maxInputChannels
    assert(d.maxInputChannels === undefined, 'output device does NOT have maxInputChannels');
    console.log(`  [${d.index}] ${d.name} - ${d.maxOutputChannels}ch @ ${d.defaultSampleRate}Hz${d.isDefault ? ' (default)' : ''}`);
  }

  // version() works
  const ver = Speaker.version();
  assert(ver.audioBackend.includes('cpal'), 'version().audioBackend contains cpal');
  assert(ver.binding === pkg.version, `version().binding equals package version ${pkg.version}, got: ${ver.binding}`);

  console.log('  Group 2 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 3: Sine wave playback (int16, 2 seconds)
// ═══════════════════════════════════════════════════════════════════════════════

async function testSineWave() {
  console.log('--- Group 3: Sine wave playback (440Hz, 2 seconds) ---');
  console.log('  (You should hear a tone)');

  const buffer = generateSineWave(16000, 440, 2, 0.3);
  const speaker = new Speaker({ sampleRate: 16000, channels: 1 });

  return new Promise((resolve) => {
    speaker.write(buffer);
    speaker.end();

    speaker.on('finish', () => {
      assert(!speaker.isPlaying, 'isPlaying is false after finish');
      console.log('  Playback finished');
      console.log('  Group 3 done\n');
      resolve();
    });

    speaker.on('error', (err) => {
      console.log(`  FAIL: ${err.message}`);
      failed++;
      console.log('  Group 3 done\n');
      resolve();
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 4: float32 playback (2 seconds)
// ═══════════════════════════════════════════════════════════════════════════════

async function testFloat32Playback() {
  console.log('--- Group 4: float32 playback (880Hz, 2 seconds) ---');
  console.log('  (You should hear a higher tone)');

  const buffer = generateSineWaveFloat32(16000, 880, 2, 0.3);
  const speaker = new Speaker({ sampleRate: 16000, channels: 1, dtype: 'float32' });

  return new Promise((resolve) => {
    speaker.write(buffer);
    speaker.end();

    speaker.on('finish', () => {
      assert(!speaker.isPlaying, 'isPlaying is false after finish (float32)');
      console.log('  Playback finished');
      console.log('  Group 4 done\n');
      resolve();
    });

    speaker.on('error', (err) => {
      console.log(`  FAIL: ${err.message}`);
      failed++;
      console.log('  Group 4 done\n');
      resolve();
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 5: Pipe capture to output (echo, 3 seconds)
// ═══════════════════════════════════════════════════════════════════════════════

async function testEcho() {
  console.log('--- Group 5: Pipe capture → output (echo, 3 seconds) ---');
  console.log('  (You should hear your voice echoed)');

  const mic = new Microphone({ sampleRate: 16000, channels: 1 });
  const speaker = new Speaker({ sampleRate: 16000, channels: 1 });

  let errorOccurred = false;
  mic.on('error', (err) => { console.log(`  mic error: ${err.message}`); errorOccurred = true; });
  speaker.on('error', (err) => { console.log(`  speaker error: ${err.message}`); errorOccurred = true; });

  mic.pipe(speaker);

  await sleep(3000);

  mic.unpipe(speaker);
  mic.stop();
  speaker.stop();

  assert(!errorOccurred, 'no errors during echo test');
  console.log('  Group 5 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 6: stop() discards buffer
// ═══════════════════════════════════════════════════════════════════════════════

async function testStopDiscards() {
  console.log('--- Group 6: stop() discards buffer ---');

  // Generate 10 seconds of audio but stop immediately
  const buffer = generateSineWave(16000, 440, 10, 0.3);
  const speaker = new Speaker({ sampleRate: 16000, channels: 1 });

  speaker.write(buffer);
  speaker.stop();

  // Should be stopped almost immediately, not after 10 seconds
  assert(!speaker.isPlaying, 'isPlaying is false immediately after stop');

  console.log('  Group 6 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 7: Multiple sequential instances
// ═══════════════════════════════════════════════════════════════════════════════

async function testMultipleInstances() {
  console.log('--- Group 7: Multiple sequential instances ---');

  // Instance 1
  const s1 = new Speaker({ sampleRate: 16000, channels: 1 });
  const buf1 = generateSineWave(16000, 440, 0.5, 0.2);
  s1.write(buf1);
  s1.stop();
  assert(!s1.isPlaying, 'instance 1: not playing after stop');

  await sleep(200);

  // Instance 2
  const s2 = new Speaker({ sampleRate: 16000, channels: 1 });
  const buf2 = generateSineWave(16000, 880, 0.5, 0.2);
  s2.write(buf2);
  s2.stop();
  assert(!s2.isPlaying, 'instance 2: not playing after stop');

  console.log('  Group 7 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Group 8: Output channel counts against the device
// ═══════════════════════════════════════════════════════════════════════════════
//
// The stream is opened on the first write, so a channel count the device
// refuses surfaces there rather than at construction. Everything here except
// the platform-limit case is machine-dependent: what a device serves above its
// own reported count is the host's business, and a machine whose ceiling sits
// lower than the counts tried below reports the typed refusal instead of
// opening. Both outcomes are asserted; neither may be a range error.

/** Resolve on the first write's outcome: the error it produced, or null. */
function firstWrite(speaker, buffer) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (err) => {
      if (settled) return;
      settled = true;
      resolve(err || null);
    };
    speaker.once('error', done);
    speaker.write(buffer, done);
  });
}

/** A short buffer of int16 silence for `channels` interleaved channels. */
function silence(channels) {
  return Buffer.alloc(channels * 256 * 2);
}

async function testChannelCounts() {
  console.log('--- Group 8: Output channel counts ---');

  const devices = Speaker.devices();
  const preferred = devices.find((d) => d.isDefault) || devices[0];
  if (!preferred) {
    console.log('  SKIP: no output device');
    skipped++;
    console.log('  Group 8 done\n');
    return;
  }
  const reported = preferred.maxOutputChannels;
  // How many channels a device serves depends on the rate it is asked for:
  // the host converts, and it converts rate and channel count together. The
  // device's own rate is where it is most permissive, so the counts above the
  // former cap are tried there.
  const nativeRate = preferred.defaultSampleRate;
  console.log(
    `  default output "${preferred.name}" reports ${reported} channels at ${nativeRate} Hz`
  );

  // Counts at and below the former cap are unchanged: they open and play.
  for (const channels of [1, 2, 8]) {
    const speaker = new Speaker({ sampleRate: 16000, channels });
    const err = await firstWrite(speaker, silence(channels));
    speaker.stop();
    assert(err === null, `channels: ${channels} still opens (got ${err && err.message})`);
  }

  // Counts above the former cap reach the device instead of being refused by
  // decibri. Regression: the cap at 32 refused counts the platform serves.
  for (const channels of [33, 40, 64]) {
    const speaker = new Speaker({ sampleRate: nativeRate, channels });
    const err = await firstWrite(speaker, silence(channels));
    speaker.stop();
    if (err === null) {
      console.log(`  ${channels} channels: opened`);
      assert(true, `channels: ${channels} opens on this machine`);
    } else {
      console.log(`  ${channels} channels: refused (${err.code})`);
      assert(
        !(err instanceof RangeError),
        `channels: ${channels} is not refused as a range error (got ${err.constructor.name})`
      );
      assert(
        err.code === 'SPEAKER_CHANNELS_UNSUPPORTED',
        `channels: ${channels} above this machine's ceiling is typed (got ${err.code}: ${err.message})`
      );
    }
  }

  // A count beyond any device, still within what the platform can express. The
  // refusal names the count asked for and the count the device reports.
  {
    const channels = 8192;
    const speaker = new Speaker({ sampleRate: nativeRate, channels });
    const err = await firstWrite(speaker, silence(channels));
    speaker.stop();
    assert(err !== null, `channels: ${channels} is refused`);
    if (err) {
      assert(err instanceof DecibriError, `channels: ${channels} is a DecibriError`);
      assert(
        err.code === 'SPEAKER_CHANNELS_UNSUPPORTED',
        `channels: ${channels} carries SPEAKER_CHANNELS_UNSUPPORTED (got ${err.code})`
      );
      assert(
        err.message.includes(`${channels}`) && err.message.includes(`it reports ${reported}`),
        `the refusal names both counts (got ${err.message})`
      );
    }
  }

  // Above what the platform can express at all. Refused by decibri before the
  // platform library sees it, so the process survives and the message names the
  // limit. Regression: the count reaching the platform library, where the
  // multiply it feeds overflows.
  {
    const channels = 20000;
    const speaker = new Speaker({ sampleRate: 16000, channels });
    const err = await firstWrite(speaker, silence(channels));
    speaker.stop();
    assert(err !== null, `channels: ${channels} is refused`);
    if (err) {
      assert(
        err.code === 'STREAM_OPEN_FAILED',
        `channels: ${channels} carries STREAM_OPEN_FAILED (got ${err.code})`
      );
      assert(
        err.message.includes('16383'),
        `the refusal names the platform limit (got ${err.message})`
      );
    }
  }

  console.log('  Group 8 done\n');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runner
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  console.log('decibri output test suite\n');

  await testErrors();
  await testDeviceEnumeration();
  await testSineWave();
  await testFloat32Playback();
  await testEcho();
  await testStopDiscards();
  await testMultipleInstances();
  await testChannelCounts();

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
