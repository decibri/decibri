'use strict';

const path = require('path');
const Decibri = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));

// ── Test 1: version() ────────────────────────────────────────────────────────
console.log('--- Decibri.version() ---');
const ver = Decibri.version();
console.log(ver);
console.assert(typeof ver.decibri === 'string', 'decibri version is a string');
console.assert(typeof ver.portaudio === 'string', 'portaudio version is a string');
console.assert(ver.portaudio.includes('cpal'), `portaudio should contain "cpal", got: ${ver.portaudio}`);
console.log('PASS: version()\n');

// ── Test 2: devices() ────────────────────────────────────────────────────────
console.log('--- Decibri.devices() ---');
const devices = Decibri.devices();
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
const mic = new Decibri({ sampleRate: 16000, channels: 1 });

let chunkCount = 0;
let totalBytes = 0;
let allCorrectSize = true;

mic.on('data', (chunk) => {
  chunkCount++;
  totalBytes += chunk.length;
  if (chunk.length !== 3200) {
    allCorrectSize = false;
    console.log(`  chunk ${chunkCount}: ${chunk.length} bytes (UNEXPECTED)`);
  } else if (chunkCount <= 3) {
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
  console.log(`\n  Total chunks: ${chunkCount}`);
  console.log(`  Total bytes: ${totalBytes}`);
  console.log(`  All chunks 3200 bytes: ${allCorrectSize}`);
  console.log(`  isOpen after stop: ${mic.isOpen}`);
  console.assert(chunkCount > 0, 'received at least one chunk');
  console.assert(allCorrectSize, 'all chunks are 3200 bytes');
  console.assert(!mic.isOpen, 'isOpen is false after stop');

  // Give the stream a moment to emit 'end' after push(null)
  setTimeout(() => {
    console.assert(streamEnded, 'stream emitted end event');
    console.log('\nPASS: capture test');
  }, 100);
}, 2000);
