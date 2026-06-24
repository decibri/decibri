# decibri examples

Runnable examples for decibri. The Node.js examples run with `node`; the browser
example is an HTML page you open in a browser.

## Node.js examples

| File | What it does | Notes |
| --- | --- | --- |
| `wav-capture.js` | Captures audio and writes a valid WAV file | No extra dependencies |
| `websocket-server.js` | Receives raw PCM chunks and logs byte counts | Requires `npm install ws` |
| `websocket-stream.js` | Streams raw PCM audio to a WebSocket server | Requires `npm install ws` |

```bash
node wav-capture.js
node websocket-server.js   # terminal 1
node websocket-stream.js   # terminal 2
```

## Browser audio playback test (`browser-speaker-test.html`)

A page for manually verifying that the browser Speaker plays real audio in each
browser. It loads the real browser build (`decibri.browser.js`, a bundle of the
package's browser entry) and plays generated tones so you can listen for a clean
signal, glitches, correct pitch after resampling, and immediate stop.

Browser audio needs a secure context, so the page must be served over
`localhost` or `https`. Opening it as a `file://` URL will not load the audio
engine.

### Desktop

From this `examples` directory, serve it over localhost and open the printed URL
in each browser you want to test:

```bash
npx serve .
# or
npx http-server . -p 8080
```

Then open `http://localhost:<port>/browser-speaker-test.html` in Chrome, Firefox,
Edge, and Safari, and tap the buttons.

### Mobile (same WiFi)

Serve on your computer as above, find your machine's LAN IP, and open
`http://<machine-ip>:<port>/browser-speaker-test.html` on the phone (on the same
network).

iOS Safari treats a plain `http://<LAN-IP>` address as an insecure context and
will refuse to start audio. For iOS, expose the local server over `https` with a
tunnel and open the tunnel URL on the phone:

```bash
npx localtunnel --port 8080
# open the printed https://... URL on the phone, then add /browser-speaker-test.html
```

### What to check in each browser

- 440 Hz tone: a clean, steady tone with no buzz or distortion.
- Continuous (3 s): smooth, with no gaps, clicks, or dropouts.
- Resampled from 16 kHz: the same pitch as the 440 Hz tone button (confirms
  resampling does not shift pitch).
- Stop: playback halts immediately when tapped mid-playback.
- Gesture requirement: nothing plays until you tap a button; if the browser
  blocks audio, a clear error appears in the on-page log.
- Both formats: repeat with the int16 and float32 toggle.

### Regenerating the browser bundle

`decibri.browser.js` is generated from the package's browser entry
(`../src/browser/index.js`) with rolldown, the bundler used to produce the
checked-in build. Regenerate it after changing the browser source so the
bundle stays in sync:

```bash
npx rolldown ../src/browser/index.js --format iife --name decibri --file decibri.browser.js
```
