# Migrating to decibri 0.2.0 (Python)

decibri 0.2.0 aligns the Python API to decibri's canonical microphone and speaker vocabulary. It is a breaking release. This guide covers every change a 0.1.x consumer needs to make, highest impact first.

## Device-enumeration methods (the main change)

The methods that list devices moved to a short, symmetric form:

| 0.1.x                          | 0.2.0                  |
| ------------------------------ | ---------------------- |
| `Microphone.input_devices()`   | `Microphone.devices()` |
| `Speaker.output_devices()`     | `Speaker.devices()`    |
| `AsyncMicrophone.input_devices()` | `AsyncMicrophone.devices()` |
| `AsyncSpeaker.output_devices()`   | `AsyncSpeaker.devices()`    |

Before:

```python
mics = Microphone.input_devices()
speakers = Speaker.output_devices()
```

After:

```python
mics = Microphone.devices()
speakers = Speaker.devices()
```

The module-level functions are unchanged. If you call these, you need no change:

```python
import decibri

decibri.input_devices()   # unchanged
decibri.output_devices()  # unchanged
```

## Exceptions

Five exception classes were renamed. The old names are removed, so any `except`
clause or `import` that uses them must be updated.

| 0.1.x                  | 0.2.0                    |
| ---------------------- | ----------------------- |
| `CaptureStreamClosed`  | `MicrophoneStreamClosed` |
| `OutputStreamClosed`   | `SpeakerStreamClosed`    |
| `DeviceNotFound`       | `MicrophoneNotFound`     |
| `OutputDeviceNotFound` | `SpeakerNotFound`        |
| `NoOutputDeviceFound`  | `NoSpeakerFound`         |

Before:

```python
from decibri import CaptureStreamClosed

try:
    chunk = mic.read()
except CaptureStreamClosed:
    handle_closed()
```

After:

```python
from decibri import MicrophoneStreamClosed

try:
    chunk = mic.read()
except MicrophoneStreamClosed:
    handle_closed()
```

Only these five names changed. The other exceptions keep their names, including
`DecibriError` (the base class), `DeviceError`, `OrtError`, `OrtPathError`,
`NoMicrophoneFound`, `NotAnInputDevice`, `DeviceEnumerationFailed`,
`DeviceIndexOutOfRange`, `MultipleDevicesMatch`, and the ORT and config error
classes. Catching `DecibriError` continues to catch every decibri exception, so
broad handlers need no change.

## Device-info types

The value types returned by `devices()` were renamed:

| 0.1.x              | 0.2.0           |
| ------------------ | --------------- |
| `DeviceInfo`       | `MicrophoneInfo` |
| `OutputDeviceInfo` | `SpeakerInfo`    |

The returned objects carry the same fields (`index`, `name`, `id`,
`max_input_channels` or `max_output_channels`, `default_sample_rate`,
`is_default`); only the type names changed. This affects code that imports these
names or uses `isinstance` against them:

```python
# Before
from decibri import DeviceInfo
assert isinstance(d, DeviceInfo)

# After
from decibri import MicrophoneInfo
assert isinstance(d, MicrophoneInfo)
```

If you only read attributes off the returned objects (`d.name`, `d.index`), no
change is needed.

## What did not change

- The module-level free functions `decibri.input_devices()` and
  `decibri.output_devices()`.
- The `vad_score` property on `Microphone` and `AsyncMicrophone`.
- The `version()` shape: `decibri`, `audio_backend`, and `binding` fields.
- The `SileroVad` voice-activity-detection path and `VadConfig`.
- The lifecycle methods (`start`, `stop`, `close`, `read`, `write`, `drain`) and
  the context-manager and iterator protocols.
- Constructor parameters and their defaults.

If your code uses only the unchanged surface, it runs on 0.2.0 without
modification.
