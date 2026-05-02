"""Phase 9 Item A4: __repr__ on the four wrapper classes.

Pins the contract that ``repr(instance)`` returns a useful string that
shows construction parameters plus current state, mirroring the
``VersionInfo`` precedent. Auto-display in Jupyter (the primary motivator
per LD-9-8) shows this repr when an instance is the last expression in
a cell.

Tests that call ``mic.start()`` to verify the ``is_open=True`` lifecycle
state are gated by ``@pytest.mark.requires_audio_input`` because opening
the cpal stream requires a real audio input device. Pure-construction
repr tests (defaults, kwargs, stop-state) run on every platform without
hardware.
"""

from __future__ import annotations

import asyncio

import pytest

import decibri


def test_microphone_repr_shows_defaults_and_is_open() -> None:
    mic = decibri.Microphone()
    text = repr(mic)
    assert text.startswith("Microphone(")
    assert "sample_rate=16000" in text
    assert "channels=1" in text
    assert "dtype='int16'" in text
    assert "frames_per_buffer=1600" in text
    assert "device=None" in text
    assert "vad=False" in text
    assert "is_open=False" in text


def test_microphone_repr_reflects_custom_params() -> None:
    mic = decibri.Microphone(
        sample_rate=24000,
        channels=2,
        dtype="float32",
        frames_per_buffer=2400,
        vad="energy",
    )
    text = repr(mic)
    assert "sample_rate=24000" in text
    assert "channels=2" in text
    assert "dtype='float32'" in text
    assert "frames_per_buffer=2400" in text
    assert "vad='energy'" in text


@pytest.mark.requires_audio_input
def test_microphone_repr_state_reflects_lifecycle() -> None:
    mic = decibri.Microphone()
    assert "is_open=False" in repr(mic)
    mic.start()
    try:
        assert "is_open=True" in repr(mic)
    finally:
        mic.stop()
    assert "is_open=False" in repr(mic)


def test_speaker_repr_shows_defaults_and_is_playing() -> None:
    spk = decibri.Speaker()
    text = repr(spk)
    assert text.startswith("Speaker(")
    assert "sample_rate=16000" in text
    assert "channels=1" in text
    assert "dtype='int16'" in text
    assert "device=None" in text
    assert "is_playing=False" in text


def test_speaker_repr_reflects_custom_params() -> None:
    spk = decibri.Speaker(sample_rate=24000, channels=2, dtype="float32")
    text = repr(spk)
    assert "sample_rate=24000" in text
    assert "channels=2" in text
    assert "dtype='float32'" in text


def test_async_microphone_repr_shows_defaults_and_is_open() -> None:
    amic = decibri.AsyncMicrophone()
    text = repr(amic)
    assert text.startswith("AsyncMicrophone(")
    assert "sample_rate=16000" in text
    assert "channels=1" in text
    assert "dtype='int16'" in text
    assert "frames_per_buffer=1600" in text
    assert "vad=False" in text
    assert "is_open=False" in text


@pytest.mark.requires_audio_input
def test_async_microphone_repr_state_reflects_lifecycle() -> None:
    async def _flow() -> tuple[str, str, str]:
        amic = decibri.AsyncMicrophone()
        before = repr(amic)
        await amic.start()
        try:
            during = repr(amic)
        finally:
            await amic.stop()
        after = repr(amic)
        return before, during, after

    before, during, after = asyncio.run(_flow())
    assert "is_open=False" in before
    assert "is_open=True" in during
    assert "is_open=False" in after


def test_async_speaker_repr_shows_defaults_and_is_playing() -> None:
    aspk = decibri.AsyncSpeaker()
    text = repr(aspk)
    assert text.startswith("AsyncSpeaker(")
    assert "sample_rate=16000" in text
    assert "channels=1" in text
    assert "dtype='int16'" in text
    assert "is_playing=False" in text
