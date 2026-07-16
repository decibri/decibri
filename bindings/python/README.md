# Decibri

Cross-platform audio capture, playback, and processing for Python.

[![PyPI version](https://img.shields.io/pypi/v/decibri.svg)](https://pypi.org/project/decibri/)
[![Python versions](https://img.shields.io/pypi/pyversions/decibri.svg)](https://pypi.org/project/decibri/)
[![License](https://img.shields.io/pypi/l/decibri.svg)](https://github.com/decibri/decibri/blob/main/LICENSE)

Decibri is a native Python package that delivers microphone capture, speaker output, local voice activity detection, device enumeration, and sample format conversion. It is written in Rust (via PyO3 / abi3) and ships pre-built wheels for Linux, macOS Apple Silicon, and Windows.

## Install

Recommended with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install decibri
```

Or with pip:

```bash
pip install decibri
```

## Quickstart

### Capture audio

```python
import decibri

with decibri.Microphone(sample_rate=16000, channels=1) as mic:
    for chunk in mic:
        print(f"Got {len(chunk)} bytes")
        break  # exit after first chunk for demo
```

### Record one second to a WAV file

```python
import decibri

decibri.record_to_file("output.wav", duration_seconds=1.0, sample_rate=16000)
```

### Condition and analyze a recording

```python
import decibri

# The same conditioning chain as the live microphone, over a WAV file.
with decibri.File("clip.wav", denoise="fastenhancer-t", highpass=80) as file:
    for chunk in file:
        handle(chunk)          # conditioned int16 PCM bytes

# Whole-file speech analysis (a live stream cannot do this).
report = decibri.File("clip.wav", vad="silero").analyze()
for segment in report.segments:
    print(segment.start, segment.end)   # seconds of file time
```

### Capture with Silero VAD

```python
import decibri

with decibri.Microphone(sample_rate=16000, vad="silero") as mic:
    for chunk in mic:
        print(f"Got {len(chunk)} bytes; VAD score {mic.vad_score}; speaking={mic.is_speaking}")
        break  # exit after first chunk for demo
```

### Async capture

```python
import asyncio
import decibri

async def main():
    async with await decibri.AsyncMicrophone.open(sample_rate=16000, vad="silero") as mic:
        async for chunk in mic:
            print(f"Got {len(chunk)} bytes; VAD score {mic.vad_score}")
            break  # exit after first chunk for demo

asyncio.run(main())
```

### Speaker output

```python
import decibri

with decibri.Speaker(sample_rate=24000, channels=1) as spk:
    audio_bytes = b"\x00\x00" * 24000  # 1 second of silence at 24kHz int16
    spk.write(audio_bytes)  # int16 PCM
    spk.drain()
```

## Public API

### Core classes

- `Microphone`: synchronous audio capture
- `Speaker`: synchronous audio output
- `File`: offline source; conditions a recording or in-memory samples and analyzes it for speech
- `AsyncMicrophone`: async-await audio capture
- `AsyncSpeaker`: async-await audio output
- `AsyncFile`: async-await offline source

### Module-level functions

- `decibri.input_devices()`: enumerate available input devices
- `decibri.output_devices()`: enumerate available output devices
- `decibri.version()`: version + audio backend info
- `decibri.record_to_file(path, duration_seconds, ...)`: record N seconds to a WAV file
- `decibri.async_record_to_file(path, duration_seconds, ...)`: async equivalent

### Value types

`MicrophoneInfo`, `SpeakerInfo`, `VersionInfo`, `Chunk`, `VadReport`, `VadWindow`, `Segment`.

### Exceptions

The full hierarchy lives at `decibri.exceptions`. Top-level catch-targets surfaced at the package root:

- `DecibriError`: base of the hierarchy
- `DeviceError`: input / output device problems
- `OrtError`: ONNX Runtime issues
- `OrtPathError`: ORT dylib path resolution issues
- `ForkAfterOrtInit`: Linux fork-after-ORT-init detection

## Voice Activity Detection

Decibri ships with two VAD modes: a lightweight RMS energy threshold (opt-in via `vad="energy"`) and a Silero ONNX model (~2.3 MB, bundled in the wheel; no API keys required). Pass the mode as the `vad="silero"` / `vad="energy"` shorthand (which uses the default threshold and a 300 ms holdoff) or as a `decibri.Vad(model=, threshold=, holdoff_ms=)` config object to tune them.

```python
# Energy mode (lightweight, no model)
mic = decibri.Microphone(vad=decibri.Vad(model="energy", threshold=0.01))

# Silero mode (ML-based, more accurate in noisy environments)
mic = decibri.Microphone(vad=decibri.Vad(model="silero", threshold=0.5))
```

Use `mic.vad_score` (a value in `[0, 1]`) to gate downstream processing. `mic.is_speaking` returns the boolean above-threshold view.

## decibri ACE (audio conditioning)

decibri ACE (Audio Capture Engine) is decibri's opt-in audio front-end for speech: a conditioning chain applied to the captured audio before it is delivered. Every stage is off by default, runs on-device, and needs no API key. Leave the options unset and the capture path is unchanged.

The stages run in this order. Pass any subset on the `Microphone` constructor:

| Stage | Keyword | Value |
| --- | --- | --- |
| DC removal | `dc_removal=True` | bool, default `False` |
| Denoise | `denoise="fastenhancer-t"` | the one bundled model |
| High-pass | `highpass=80` or `100` | Hz, second-order Butterworth |
| AGC | `agc=-18` | int dBFS, -40 to -3 |
| Limiter | `limiter=-1.0` | float dBFS, -3.0 to 0.0 |

```python
import decibri

with decibri.Microphone(
    sample_rate=16000,
    denoise="fastenhancer-t",  # bundled speech-enhancement model
    highpass=80,               # remove low-frequency rumble
    agc=-18,                   # target level in dBFS
    limiter=-1.0,              # peak ceiling in dBFS
    vad=decibri.Vad(model="silero", threshold=0.5),
) as mic:
    for chunk in mic:
        if mic.is_speaking:
            handle(chunk)      # conditioned int16 PCM bytes
```

The denoise model is bundled in the wheel (the same way the Silero VAD model is), so `denoise="fastenhancer-t"` needs no download. VAD reads the signal before the chain, so `mic.vad_score` and `mic.is_speaking` are unaffected by which conditioning stages you enable. The same options are available on `AsyncMicrophone`.

## Files

Everything a `Microphone` does to live audio, `File` does to audio you already have: the same conditioning options (`dc_removal`, `denoise`, `highpass`, `agc`, `limiter`), the same iteration, the same conditioned chunks out, and the same opt-in `vad=`. A `File` reads a WAV (`File("clip.wav")`, or the identical `File.open("clip.wav")`) or wraps in-memory samples (`File.buffer(samples, input_rate=48000)`; raw samples carry no header, so their native rate is explicit). `sample_rate` stays the target output rate, the same meaning it has on `Microphone`, so a 44.1 kHz recording comes out at 16 kHz unless you set it.

Because a `File` is a complete recording, it can analyze the whole recording for speech:

```python
report = decibri.File("clip.wav", vad="silero").analyze()   # or .analyse()
for w in report.scores:
    print(w.start, w.end, w.vad_score, w.is_speech)   # per 32 ms window
for segment in report.segments:
    print(segment.start, segment.end)                 # merged speech regions
```

`analyze()` requires VAD: a `File` built without `vad=` raises `VadNotConfigured` rather than constructing a detector silently. With `vad=` set, metadata iteration (`iter_with_metadata()`) carries per-chunk `vad_score` and `is_speaking` exactly as the microphone does, with one deliberate difference: on a `File`, the speaking holdoff and the `Chunk.timestamp` are measured in FILE time (sample positions in seconds), never wall-clock time, so a file processed faster than real time still reports correct speech timing. Iteration and analysis are separate single passes; construct one `File` per operation. `AsyncFile` mirrors the whole surface with `async` semantics.

## Compatibility

| Python                       | Platforms                                                |
| ---------------------------- | -------------------------------------------------------- |
| 3.10, 3.11, 3.12, 3.13, 3.14 | Linux x64, Linux ARM64, macOS Apple Silicon, Windows x64 |

## Bundled assets

The wheel includes:

- **Silero VAD ONNX model** (~2.3 MB): no downloads or API keys required for `vad="silero"`.
- **ONNX Runtime dylib** (~15-20 MB platform-specific): no system dependency on `pip install onnxruntime`.

First ORT load on `vad="silero"` initialization is ~100 to 500 ms (amortized across subsequent calls).

## Async usage

For Silero VAD in async code, use the `open()` factory to dispatch the synchronous ORT init off the event loop:

```python
async with await decibri.AsyncMicrophone.open(vad="silero") as mic:
    async for chunk in mic:
        ...
```

Synchronous constructors (`AsyncMicrophone(...)`) remain supported and unchanged; `open()` is the recommended pattern when ORT load cost matters.

## Multiprocessing

Linux multiprocessing with Silero VAD requires `set_start_method('spawn')`. Calling `fork()` after ORT initialization raises `ForkAfterOrtInit`:

```python
import multiprocessing as mp

mp.set_start_method('spawn')  # required on Linux for Silero
```

See the ecosystem guides under `bindings/python/docs/ecosystem/` (jupyter, docker, multiprocessing) for environment-specific details.

## Documentation

- **Source code & issues**: [github.com/decibri/decibri](https://github.com/decibri/decibri)
- **CHANGELOG**: [CHANGELOG.md](https://github.com/decibri/decibri/blob/main/CHANGELOG.md)
- **Ecosystem guides** (Jupyter, Docker, multiprocessing): under `bindings/python/docs/ecosystem/` in the repo

## License

Apache-2.0. See [LICENSE](https://github.com/decibri/decibri/blob/main/LICENSE) for details.

Copyright (c) 2026 Decibri.
