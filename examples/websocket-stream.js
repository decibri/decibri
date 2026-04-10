'use strict';
// Streams raw PCM audio to a WebSocket server.
// Requires: npm install ws
// Usage:    node examples/websocket-stream.js [ws://localhost:8080]

const { WebSocket } = require('ws');
// In your own project: const Decibri = require('decibri');
const Decibri       = require('../npm/decibri/src/decibri.js');

const url = process.argv[2] || 'ws://localhost:8080';
const ws  = new WebSocket(url);
const mic = new Decibri({ sampleRate: 16000, channels: 1 });

ws.on('open', () => {
  console.log(`Connected to ${url} — streaming audio. Ctrl+C to stop.`);

  mic.on('data', (chunk) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
  });

  mic.on('error', (err) => {
    console.error('Mic error:', err.message);
    ws.close();
  });

  mic.on('backpressure', () => {
    console.warn('Backpressure — consumer too slow');
  });
});

ws.on('close', () => {
  mic.stop();
  console.log('WebSocket closed.');
});

ws.on('error', (err) => {
  console.error('WebSocket error:', err.message);
  mic.stop();
});

process.on('SIGINT', () => {
  mic.stop();
  ws.close();
});
