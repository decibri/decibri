# decibri

Cross-platform audio capture, output, and processing for Python.

[![PyPI version](https://img.shields.io/pypi/v/decibri.svg)](https://pypi.org/project/decibri/)
[![Python versions](https://img.shields.io/pypi/pyversions/decibri.svg)](https://pypi.org/project/decibri/)
[![License](https://img.shields.io/pypi/l/decibri.svg)](https://github.com/decibri/decibri/blob/main/LICENSE)

decibri is a native Python package that delivers microphone capture, speaker output, local voice activity detection, device enumeration, and sample format conversion. It is written in Rust (via PyO3 / abi3) and ships pre-built wheels for Linux, macOS Apple Silicon, and Windows.

## Install

```bash
pip install decibri
```

For NumPy ndarray return support (optional):

```bash
pip install decibri[numpy]
```

## Quickstart

### Capture audio

```python
import decibri

mic = decibri.Microphone(sample_rate=16000, channels=1)
mic.start()
chunk = mic.read(16000)  # 1 second of int16 PCM
print(f"Got {len(chunk)} bytes")
mic.stop()
```

### Capture with Silero VAD

```python
import decibri

mic = decibri.Microphone(sample_rate=16000, vad="silero")
mic.start()
chunk = mic.read(1600)  # 100 ms
print(f"VAD score: {mic.vad_score}")
mic.stop()
```

### Async capture

```python
import asyncio
import decibri

async def main():
    async with await decibri.AsyncMicrophone.open(sample_rate=16000, vad="silero") as mic:
        async for chunk in mic:
            print(f"Got {len(chunk)} bytes; VAD score {mic.vad_score}")

asyncio.run(main())
```

### Speaker output

```python
import decibri

spk = decibri.Speaker(sample_rate=24000, channels=1)
spk.start()
spk.write(audio_bytes)  # int16 PCM
spk.drain()
spk.stop()
```

### NumPy ndarray return

```python
import decibri

mic = decibri.Microphone(sample_rate=16000, as_ndarray=True)
mic.start()
chunk = mic.read(16000)  # numpy.ndarray
print(f"Shape: {chunk.shape}, dtype: {chunk.dtype}")
mic.stop()
```

Requires `pip install decibri[numpy]`.

## Public API

### Core classes

- `Microphone`: synchronous audio capture
- `Speaker`: synchronous audio output
- `AsyncMicrophone`: async-await audio capture
- `AsyncSpeaker`: async-await audio output

### Module-level functions

- `decibri.input_devices()`: enumerate available input devices
- `decibri.output_devices()`: enumerate available output devices
- `decibri.version()`: version + audio backend info
- `decibri.record_to_file(path, seconds, ...)`: record N seconds to a WAV file
- `decibri.async_record_to_file(path, seconds, ...)`: async equivalent

### Value types

`DeviceInfo`, `OutputDeviceInfo`, `VersionInfo`, `Chunk`.

### Exceptions

The full hierarchy lives at `decibri.exceptions`. Top-level catch-targets surfaced at the package root:

- `DecibriError`: base of the hierarchy
- `DeviceError`: input / output device problems
- `OrtError`: ONNX Runtime issues
- `OrtPathError`: ORT dylib path resolution issues
- `ForkAfterOrtInit`: Linux fork-after-ORT-init detection

## Voice Activity Detection

decibri ships with two VAD modes: a lightweight RMS energy threshold (default-off, opt-in via `vad="energy"`) and a Silero ONNX model (~2.3 MB, bundled in the wheel; no API keys required).

```python
# Energy mode (lightweight, no model)
mic = decibri.Microphone(vad="energy", vad_threshold=0.01)

# Silero mode (ML-based, more accurate in noisy environments)
mic = decibri.Microphone(vad="silero", vad_threshold=0.5)
```

Use `mic.vad_score` (a value in `[0, 1]`) to gate downstream processing. `mic.is_speaking` returns the boolean above-threshold view.

## What decibri is (and isn't)

decibri is **audio infrastructure** for voice AI applications. It captures clean cross-platform audio, plays it back, runs local VAD, and provides local ONNX inference primitives. It is NOT a voice AI engine that wraps cloud providers.

For cloud STT and TTS, install the cloud's official Python SDK alongside decibri and wire decibri's audio output into it in roughly thirty lines of Python:

```bash
pip install decibri deepgram-sdk     # or: elevenlabs / openai / assemblyai
```

For voice agent orchestration (turn-taking, interruption, function-calling), use a framework that sits ABOVE decibri:

- Pipecat
- LiveKit Agents
- vocode

decibri is the layer below those frameworks. They compose decibri (or any audio capture library) with cloud STT/TTS providers and LLM-based agent logic.

## Compatibility

| Python                   | Platforms                                                  |
| ------------------------ | ---------------------------------------------------------- |
| 3.10+ (abi3-py310 floor) | Linux x64, Linux ARM64, macOS Apple Silicon, Windows x64   |

Intel Mac users can install from source:

```bash
pip install decibri --no-binary :all:
```

This requires a working Rust toolchain. Pre-built Intel Mac wheels are not part of the matrix (Apple platform deprecation; macos-13 dropped from the wheel matrix per the 0.2.0 backlog).

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

Copyright (c) 2026 decibri.
