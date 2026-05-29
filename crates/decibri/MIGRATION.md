# Migrating from decibri 3.x to 4.0

Version 4.0 renames the public API to a microphone/speaker vocabulary. The
behavior is unchanged; the work is mechanical: update names, module paths, one
feature flag, and any code that matches on error variants or asserts on error
message text. This guide covers the decibri crate.

## Type renames

| 3.x | 4.0 |
| --- | --- |
| `AudioCapture` | `Microphone` |
| `CaptureStream` | `MicrophoneStream` |
| `CaptureConfig` | `MicrophoneConfig` |
| `AudioOutput` | `Speaker` |
| `OutputStream` | `SpeakerStream` |
| `OutputConfig` | `SpeakerConfig` |
| `DeviceInfo` | `MicrophoneInfo` |
| `OutputDeviceInfo` | `SpeakerInfo` |

`AudioChunk`, `VadConfig`, `VadResult`, and `SileroVad` keep their names. The
device-info channel fields (`max_input_channels`, `max_output_channels`) are
unchanged.

## Module paths and crate-root re-exports

The modules renamed: `capture` to `microphone`, `output` to `speaker`. The
common types are now re-exported at the crate root, so the simplest update is to
import from `decibri` directly:

```rust
// 3.x
use decibri::capture::{AudioCapture, CaptureConfig, CaptureStream};
use decibri::output::{AudioOutput, OutputConfig, OutputStream};

// 4.0
use decibri::{Microphone, MicrophoneConfig, MicrophoneStream};
use decibri::{Speaker, SpeakerConfig, SpeakerStream};
```

The module paths (`decibri::microphone::Microphone`, `decibri::speaker::Speaker`)
also work if you prefer them.

## Feature rename

The `output` feature is now `playback`. The `capture` feature is unchanged.

```toml
# 3.x
decibri = { version = "3", features = ["capture", "output"], default-features = false }

# 4.0
decibri = { version = "4", features = ["capture", "playback"], default-features = false }
```

## Device enumeration

The verbose free functions are gone:

| 3.x | 4.0 |
| --- | --- |
| `decibri::device::enumerate_input_devices()` | `decibri::input_devices()` or `Microphone::devices()` |
| `decibri::device::enumerate_output_devices()` | `decibri::output_devices()` or `Speaker::devices()` |

The return element types follow the type renames: `Vec<DeviceInfo>` becomes
`Vec<MicrophoneInfo>`, and `Vec<OutputDeviceInfo>` becomes `Vec<SpeakerInfo>`.

## Device resolution

The free `resolve_device` and `resolve_output_device` functions are no longer
part of the public API. Select a device by setting the `device` field of
`MicrophoneConfig` or `SpeakerConfig` to a `DeviceSelector` (by default, index,
name substring, or stable id); `Microphone::new` and `Speaker::new` resolve it
internally. If you previously called the free functions to obtain a
`cpal::Device`, use `cpal` directly for that.

## VAD

`SileroVad` and `VadConfig` are unchanged. Run VAD exactly as in 3.x: construct
a `SileroVad` and call `process` on each chunk's samples.

```rust
let mut vad = SileroVad::new(VadConfig::default())?;
let result = vad.process(&chunk.data)?;
```

`MicrophoneConfig` does not carry a VAD field. Capture configuration and VAD are
separate concerns: configure the microphone with `MicrophoneConfig`, and run
detection with a separately constructed `SileroVad`.

## Error handling

If you match on `DecibriError` variants, update the renamed ones:

| 3.x variant | 4.0 variant |
| --- | --- |
| `DeviceNotFound` | `MicrophoneNotFound` |
| `OutputDeviceNotFound` | `SpeakerNotFound` |
| `NoOutputDeviceFound` | `NoSpeakerFound` |
| `CaptureStreamClosed` | `MicrophoneStreamClosed` |
| `OutputStreamClosed` | `SpeakerStreamClosed` |

`NoMicrophoneFound` and `NotAnInputDevice` keep their names. The enum is
`#[non_exhaustive]`, so a `_ =>` arm is still required.

If you assert on error message text, the `Display` strings changed:

| 3.x message | 4.0 message |
| --- | --- |
| `No audio input device found matching "{name}"` | `No microphone found matching "{name}"` |
| `No audio output device found matching "{name}"` | `No speaker found matching "{name}"` |
| `No audio output device found. Check system audio settings.` | `No speaker found. Check system audio settings.` |
| `Selected device is not a valid input device.` | `Selected device is not a valid microphone.` |
| `Capture stream is closed` | `Microphone stream is closed` |
| `Output stream closed` | `Speaker stream is closed` |

## Worked example

```rust
// 3.x
use std::time::Duration;
use decibri::capture::{AudioCapture, CaptureConfig};
use decibri::vad::{SileroVad, VadConfig};

let capture = AudioCapture::new(CaptureConfig::default())?;
let stream = capture.start()?;
let mut vad = SileroVad::new(VadConfig::default())?;

while let Ok(Some(chunk)) = stream.next_chunk(Some(Duration::from_millis(100))) {
    if vad.process(&chunk.data)?.is_speech {
        println!("speech");
    }
}
```

```rust
// 4.0
use std::time::Duration;
use decibri::{Microphone, MicrophoneConfig, SileroVad, VadConfig};

let microphone = Microphone::new(MicrophoneConfig::default())?;
let stream = microphone.start()?;
let mut vad = SileroVad::new(VadConfig::default())?;

while let Ok(Some(chunk)) = stream.next_chunk(Some(Duration::from_millis(100))) {
    if vad.process(&chunk.data)?.is_speech {
        println!("speech");
    }
}
```

Only the import line and the type name change; the capture loop and VAD calls
are identical.
