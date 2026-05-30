# Migrating to decibri 4.0.0

decibri 4.0.0 renames the Node.js API to a shared microphone and speaker
vocabulary that matches the Rust and Python packages, and tidies several option
and return shapes. This guide lists every breaking change with before and after
code.

## Named exports

The package no longer has a single default export. Destructure what you need.

Before:

```js
const Decibri = require('decibri');
const { DecibriOutput } = Decibri;
```

After:

```js
const { Microphone, Speaker, inputDevices, outputDevices, version } = require('decibri');
```

## Class renames

`Decibri` is now `Microphone`; `DecibriOutput` is now `Speaker`.

Before:

```js
const mic = new Decibri({ sampleRate: 16000 });
const speaker = new DecibriOutput({ sampleRate: 16000 });
```

After:

```js
const mic = new Microphone({ sampleRate: 16000 });
const speaker = new Speaker({ sampleRate: 16000 });
```

## The format option is now dtype

Before:

```js
new Microphone({ format: 'float32' });
```

After:

```js
new Microphone({ dtype: 'float32' });
```

The accepted values (`'int16'`, `'float32'`) are unchanged.

## Device-info type renames

`DeviceInfo` is now `MicrophoneInfo`; `OutputDeviceInfo` is now `SpeakerInfo`.

Before:

```ts
import { DeviceInfo, OutputDeviceInfo } from 'decibri';
```

After:

```ts
import { MicrophoneInfo, SpeakerInfo } from 'decibri';
```

## The vad option is now a single union

The `vad: true` plus `vadMode` pair is gone. Pass the mode directly.

Before:

```js
new Microphone({ vad: true, vadMode: 'silero' });
new Microphone({ vad: true, vadMode: 'energy' });
new Microphone({ vad: false });
```

After:

```js
new Microphone({ vad: 'silero' });
new Microphone({ vad: 'energy' });
new Microphone({ vad: false }); // unchanged; the default
```

`vad: true` now throws a `TypeError` telling you to specify the mode.
`vadThreshold` (default 0.5 for silero, 0.01 for energy), `vadHoldoff`, and
`modelPath` are unchanged. A new `vadScore` getter returns the latest score for
the active mode (the Silero probability in silero mode, the normalized RMS in
energy mode, 0 when disabled).

## version() shape

Before:

```js
version(); // { decibri, portaudio }
```

After:

```js
version(); // { decibri, audioBackend, binding }
```

`audioBackend` replaces the inaccurately named `portaudio` field; its value is
unchanged. `binding` is the new field reporting the npm package version.

## Error handling

4.0.0 adds error classes you can catch:

```js
const { Microphone, DecibriError, DeviceError } = require('decibri');

try {
  new Microphone({ device: 'no such device' });
} catch (err) {
  if (err instanceof DeviceError) {
    console.log(err.code); // e.g. 'MICROPHONE_NOT_FOUND'
  }
}
```

`DeviceError`, `OrtError`, and `OrtPathError` all extend `DecibriError`, which
extends `Error`. Each carries a stable `code` string.

## Two deliberate differences from the Python package

If you use both the Node.js and Python packages, note these intended
differences:

1. An out-of-range device index throws a `RangeError` in Node.js, the idiomatic
   type for a bad numeric index. The Python package groups the same condition
   under its `DeviceError`. Argument-validation errors in Node.js (bad sample
   rate, channels, dtype, or vad value) also stay built-in `RangeError` or
   `TypeError`, not `DecibriError` subclasses.
2. The browser surface runs energy-mode VAD only, because Silero needs the
   native runtime that ships with the Node.js build. The browser `vad` option
   accepts `false` or `'energy'`, and the browser `version()` returns
   `{ decibri }` only, because the browser has no native audio backend or a
   separate binding version.
