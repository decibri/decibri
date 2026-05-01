# decibri: Python bindings

Alpha-quality Python bindings for the [decibri](https://github.com/decibri/decibri)
audio capture / playback / voice-activity-detection library.

**Status: 0.1.0a1 (alpha).** Capture (`Microphone`), playback (`Speaker`),
async equivalents (`AsyncMicrophone`, `AsyncSpeaker`), Silero VAD, and
optional NumPy ndarray support are all present. Phase 11 will produce
the comprehensive PyPI README; this file is the placeholder.

## Install

Not yet on PyPI. Built locally via:

    cd bindings/python
    uv sync --dev
    uv run maturin develop --uv

## Quickstart

    import decibri

    # Capture audio with Silero VAD
    with decibri.Microphone(vad="silero") as mic:
        for chunk in mic:
            if mic.is_speaking:
                process(chunk)

    # Playback (output)
    with decibri.Speaker(sample_rate=24000) as speaker:
        speaker.write(audio_bytes)
        speaker.drain()

    # Async equivalents: AsyncMicrophone, AsyncSpeaker
    # Lowercase factory functions: decibri.microphone(), decibri.speaker(),
    # decibri.async_microphone(), decibri.async_speaker()

## License

Apache-2.0. See the [workspace LICENSE](../../LICENSE).
