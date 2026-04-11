'use strict';
// Captures audio for DURATION_MS and writes a valid WAV file.
// No external dependencies. Uses Node.js built-ins only.
// Usage: node examples/wav-capture.js

const fs = require('fs');
// In your own project: const Decibri = require('decibri');
const Decibri = require('../npm/decibri/src/decibri.js');

const SAMPLE_RATE     = 16000;
const CHANNELS        = 1;
const BITS_PER_SAMPLE = 16;
const DURATION_MS     = 5000;

const mic    = new Decibri({ sampleRate: SAMPLE_RATE, channels: CHANNELS });
const chunks = [];

mic.on('data',  (chunk) => chunks.push(chunk));
mic.on('error', (err)   => { console.error('Mic error:', err.message); process.exit(1); });

console.log(`Recording for ${DURATION_MS / 1000} seconds...`);

setTimeout(() => {
  mic.stop();

  const pcm    = Buffer.concat(chunks);
  const header = buildWavHeader(pcm.length, SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE);
  const out    = fs.createWriteStream('capture.wav');
  out.write(header);
  out.end(pcm);
  out.on('finish', () =>
    console.log(`Wrote capture.wav (${(pcm.length / 1024).toFixed(1)} KB PCM, ${DURATION_MS / 1000}s)`)
  );
}, DURATION_MS);

function buildWavHeader(dataBytes, sampleRate, channels, bitsPerSample) {
  const byteRate   = sampleRate * channels * bitsPerSample / 8;
  const blockAlign = channels * bitsPerSample / 8;
  const buf        = Buffer.alloc(44);
  buf.write('RIFF',                0); buf.writeUInt32LE(36 + dataBytes,  4);
  buf.write('WAVE',                8); buf.write('fmt ',                 12);
  buf.writeUInt32LE(16,           16); buf.writeUInt16LE(1,              20); // PCM
  buf.writeUInt16LE(channels,     22); buf.writeUInt32LE(sampleRate,     24);
  buf.writeUInt32LE(byteRate,     28); buf.writeUInt16LE(blockAlign,     32);
  buf.writeUInt16LE(bitsPerSample,34);
  buf.write('data',               36); buf.writeUInt32LE(dataBytes,      40);
  return buf;
}
