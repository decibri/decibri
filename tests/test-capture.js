'use strict';

const path = require('path');
const { Microphone } = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));
const pkg = require(path.join(__dirname, '..', 'npm', 'decibri', 'package.json'));

// ── Test 1: version() ────────────────────────────────────────────────────────
console.log('--- Microphone.version() ---');
const ver = Microphone.version();
console.log(ver);
console.assert(typeof ver.decibri === 'string', 'decibri version is a string');
console.assert(typeof ver.audioBackend === 'string', 'audioBackend version is a string');
console.assert(ver.audioBackend.includes('cpal'), `audioBackend should contain "cpal", got: ${ver.audioBackend}`);
console.assert(ver.binding === pkg.version, `binding should equal package version ${pkg.version}, got: ${ver.binding}`);
console.log('PASS: version()\n');

// ── Test 2: devices() ────────────────────────────────────────────────────────
console.log('--- Microphone.devices() ---');
const devices = Microphone.devices();
console.log(JSON.stringify(devices, null, 2));
console.assert(Array.isArray(devices), 'devices() returns an array');
if (devices.length > 0) {
  const d = devices[0];
  console.assert(typeof d.index === 'number', 'device has index');
  console.assert(typeof d.name === 'string', 'device has name');
  console.assert(typeof d.maxInputChannels === 'number', 'device has maxInputChannels');
  console.assert(typeof d.defaultSampleRate === 'number', 'device has defaultSampleRate');
  console.assert(typeof d.isDefault === 'boolean', 'device has isDefault');
}
console.log('PASS: devices()\n');

// ── Test 3: capture 2 seconds ────────────────────────────────────────────────
console.log('--- Capture test (2 seconds) ---');
const mic = new Microphone({ sampleRate: 16000, channels: 1 });

let chunkCount = 0;
let totalBytes = 0;
const chunkLengths = [];

mic.on('data', (chunk) => {
  chunkCount++;
  totalBytes += chunk.length;
  chunkLengths.push(chunk.length);
  if (chunkCount <= 3) {
    console.log(`  chunk ${chunkCount}: ${chunk.length} bytes`);
  }
});

mic.on('error', (err) => {
  console.error('ERROR:', err.message);
  process.exit(1);
});

let streamEnded = false;
mic.on('end', () => {
  streamEnded = true;
  console.log('  Stream ended');
});

setTimeout(() => {
  mic.stop();
  // Validate after the stream has fully ended, so the final short tail chunk
  // (flushed on close) is included in chunkLengths.
  setTimeout(() => {
    // Every chunk is 3200 bytes during the stream; only the final chunk at
    // close may be shorter (1..3200, nonzero), carrying the remaining tail.
    const steady = chunkLengths.slice(0, -1);
    const last = chunkLengths[chunkLengths.length - 1];
    const steadyOk = steady.every((n) => n === 3200);
    const lastOk = last === undefined || (last > 0 && last <= 3200);

    console.log(`\n  Total chunks: ${chunkCount}`);
    console.log(`  Total bytes: ${totalBytes}`);
    console.log(`  Non-final chunks all 3200 bytes: ${steadyOk}`);
    console.log(`  Final chunk: ${last} bytes (1..3200): ${lastOk}`);
    console.log(`  isOpen after stop: ${mic.isOpen}`);
    console.assert(chunkCount > 0, 'received at least one chunk');
    console.assert(steadyOk, 'all chunks except the last are 3200 bytes');
    console.assert(lastOk, `final chunk is 1..3200 bytes (got ${last})`);
    console.assert(!mic.isOpen, 'isOpen is false after stop');
    console.assert(streamEnded, 'stream emitted end event');
    console.log('\nPASS: capture test');
  }, 200);
}, 2000);
