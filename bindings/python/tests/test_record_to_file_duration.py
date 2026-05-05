"""Regression tests for the 0.1.0 ``record_to_file`` duration bug.

decibri 0.1.0 produced WAV files of only ~10% of the requested duration:
the chunk-count loop assumed each cpal callback delivered ``frames_per_buffer``
frames, but on Windows WASAPI each callback delivers far fewer (the device
period, typically ~10ms = ~160 frames at 16kHz). 0.1.1 switches both helpers
to a frame-counting loop that accumulates audio until the requested frame
count is met regardless of the platform's actual chunk size.

These tests assert the correct contract from the helper docstrings: a
WAV produced by ``record_to_file(duration_seconds=N)`` should contain
approximately ``N * sample_rate`` frames, within a tolerance that allows
for cpal callback timing imprecision and the documented "rounded up"
upper bound.

All hardware-touching tests are marked ``requires_audio_input`` so they
auto-skip in CI per ``conftest.py`` policy. Run locally before publish.
"""

from __future__ import annotations

import asyncio
import tempfile
import wave
from pathlib import Path

import pytest

import decibri


def _read_wav_actual_duration_seconds(path: str) -> float:
    """Return the WAV file's actual duration in seconds via the WAV header."""
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / rate


def _assert_duration_in_range(actual: float, requested: float, label: str = "") -> None:
    """Assert the actual WAV duration is within the documented contract.

    The helper accumulates audio until at least ``requested * sample_rate``
    frames are written, then exits at the next chunk boundary. Actual
    duration is therefore in ``[requested, requested + one_chunk]``. We
    use a 0.95 lower bound for cpal callback timing imprecision and a
    0.2 upper bound for the rounding plus headroom.

    A WAV with actual ~10% of requested (the 0.1.0 bug) falls far below
    the lower bound and triggers a clear failure message that names the
    bug pattern explicitly.
    """
    lower = requested * 0.95
    upper = requested + 0.2
    bug_floor = requested * 0.05
    bug_ceiling = requested * 0.15

    bug_match = bug_floor <= actual <= bug_ceiling
    bug_hint = (
        " (this matches the 0.1.0 bug where ~10% of requested duration is captured)"
        if bug_match
        else ""
    )

    assert lower <= actual <= upper, (
        f"{label}duration_seconds={requested:.1f} produced WAV with actual "
        f"duration {actual:.3f}s; expected in [{lower:.3f}, {upper:.3f}]"
        f"{bug_hint}"
    )


@pytest.mark.requires_audio_input
def test_record_to_file_actual_duration_1_second(tmp_path: Path) -> None:
    """``record_to_file(duration_seconds=1.0)`` produces ~1s of audio."""
    wav_path = str(tmp_path / "rec_1s.wav")
    decibri.record_to_file(wav_path, duration_seconds=1.0, sample_rate=16000)
    actual = _read_wav_actual_duration_seconds(wav_path)
    _assert_duration_in_range(actual, 1.0, "record_to_file: ")


@pytest.mark.requires_audio_input
def test_record_to_file_actual_duration_2_seconds(tmp_path: Path) -> None:
    """``record_to_file(duration_seconds=2.0)`` produces ~2s of audio."""
    wav_path = str(tmp_path / "rec_2s.wav")
    decibri.record_to_file(wav_path, duration_seconds=2.0, sample_rate=16000)
    actual = _read_wav_actual_duration_seconds(wav_path)
    _assert_duration_in_range(actual, 2.0, "record_to_file: ")


@pytest.mark.requires_audio_input
def test_record_to_file_actual_duration_3_seconds(tmp_path: Path) -> None:
    """``record_to_file(duration_seconds=3.0)`` produces ~3s of audio."""
    wav_path = str(tmp_path / "rec_3s.wav")
    decibri.record_to_file(wav_path, duration_seconds=3.0, sample_rate=16000)
    actual = _read_wav_actual_duration_seconds(wav_path)
    _assert_duration_in_range(actual, 3.0, "record_to_file: ")


@pytest.mark.requires_audio_input
def test_async_record_to_file_actual_duration_1_second(tmp_path: Path) -> None:
    """``async_record_to_file(duration_seconds=1.0)`` produces ~1s of audio."""
    wav_path = str(tmp_path / "rec_async_1s.wav")
    asyncio.run(
        decibri.async_record_to_file(
            wav_path, duration_seconds=1.0, sample_rate=16000
        )
    )
    actual = _read_wav_actual_duration_seconds(wav_path)
    _assert_duration_in_range(actual, 1.0, "async_record_to_file: ")


@pytest.mark.requires_audio_input
def test_async_record_to_file_actual_duration_2_seconds(tmp_path: Path) -> None:
    """``async_record_to_file(duration_seconds=2.0)`` produces ~2s of audio."""
    wav_path = str(tmp_path / "rec_async_2s.wav")
    asyncio.run(
        decibri.async_record_to_file(
            wav_path, duration_seconds=2.0, sample_rate=16000
        )
    )
    actual = _read_wav_actual_duration_seconds(wav_path)
    _assert_duration_in_range(actual, 2.0, "async_record_to_file: ")


@pytest.mark.requires_audio_input
def test_record_to_file_proportional_scaling(tmp_path: Path) -> None:
    """``record_to_file`` scales linearly with duration_seconds.

    Catches a different failure mode than the absolute-duration tests:
    if scaling itself broke (e.g., duration_seconds was ignored entirely
    and produced a fixed-size file regardless), the ratio would not
    match. Even on the 0.1.0 bug the ratio held (1s -> 0.098s,
    3s -> 0.298s, ratio 3.04), so this test passes on both 0.1.0 and
    0.1.1 and is a sanity check against future regressions to the
    scaling shape rather than the magnitude.
    """
    path1 = str(tmp_path / "scale_1s.wav")
    path3 = str(tmp_path / "scale_3s.wav")

    decibri.record_to_file(path1, duration_seconds=1.0, sample_rate=16000)
    decibri.record_to_file(path3, duration_seconds=3.0, sample_rate=16000)

    dur1 = _read_wav_actual_duration_seconds(path1)
    dur3 = _read_wav_actual_duration_seconds(path3)
    ratio = dur3 / dur1

    assert 2.4 <= ratio <= 3.6, (
        f"record_to_file ratio dur(3s)/dur(1s) = {ratio:.3f}, expected ~3:1. "
        f"dur1={dur1:.3f}, dur3={dur3:.3f}. "
        f"This indicates the duration scaling itself is broken, not just the magnitude."
    )
