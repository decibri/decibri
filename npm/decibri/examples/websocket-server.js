'use strict';
// Minimal WebSocket server that receives raw PCM chunks and logs byte counts.
// Use this to test examples/websocket-stream.js.
// Requires: npm install ws
// Usage:    node examples/websocket-server.js

const { WebSocketServer } = require('ws');

const wss = new WebSocketServer({ port: 8080 });
console.log('Listening on ws://localhost:8080');

wss.on('connection', (ws) => {
  console.log('Client connected');
  let total = 0;

  ws.on('message', (data) => {
    total += data.length;
    process.stdout.write(`\rReceived ${(total / 1024).toFixed(1)} KB`);
  });

  ws.on('close', () => {
    console.log(`\nClient disconnected. Total received: ${(total / 1024).toFixed(1)} KB`);
  });

  ws.on('error', (err) => {
    console.error('\nWebSocket error:', err.message);
  });
});
