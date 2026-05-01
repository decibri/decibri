"""Phase 7.6 Item C5: ``record_to_file`` convenience helper tests.

Five assertions:
  1. The recorded file's frame count is within tolerance of requested
     duration (records for approximately the requested time).
  2. The output WAV is well-formed and parseable by stdlib ``wave``,
     with the configured sample rate and channel count.
  3. The async equivalent produces an equivalent file.
  4. An invalid path raises a sensible error (FileNotFoundError /
     PermissionError / OSError; whatever stdlib ``wave`` surfaces).
  5. A custom ``sample_rate`` (24000) is honored end-to-end.

Tests that touch real audio hardware are marked ``requires_audio_input``
so they auto-skip in CI per ``conftest.py`` policy.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import pytest

import decibri


# ---------------------------------------------------------------------------
# Sync record_to_file
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
def test_record_to_file_records_for_requested_duration(tmp_path: Path) -> None:
    """The recorded file's frame count is within tolerance of requested duration.

    The implementation rounds up to the nearest 100ms chunk (1600 frames at
    16kHz). For a 0.3s request we expect at least 0.3s and at most 0.4s of
    audio (3 to 4 chunks).
    """
    target = tmp_path / "rec.wav"
    sample_rate = 16000
    duration = 0.3

    decibri.record_to_file(str(target), duration_seconds=duration, sample_rate=sample_rate)

    with wave.open(str(target), "rb") as wav:
        frames = wav.getnframes()

    minimum = int(duration * sample_rate)  # 4800 frames
    maximum = minimum + 1600  # rounded-up tolerance: one extra chunk
    assert minimum <= frames <= maximum, (
        f"recorded {frames} frames; expected between {minimum} and {maximum}"
    )


@pytest.mark.requires_audio_input
def test_record_to_file_writes_well_formed_wav(tmp_path: Path) -> None:
    """The output WAV is parseable by stdlib ``wave`` with the expected metadata."""
    target = tmp_path / "rec.wav"
    decibri.record_to_file(
        str(target), duration_seconds=0.2, sample_rate=16000, channels=1
    )

    assert target.is_file()
    with wave.open(str(target), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2  # 16-bit PCM
        assert wav.getnframes() > 0


@pytest.mark.requires_audio_input
def test_async_record_to_file_writes_well_formed_wav(tmp_path: Path) -> None:
    """The async equivalent produces a well-formed WAV with matching metadata."""
    target = tmp_path / "rec_async.wav"

    asyncio.run(
        decibri.async_record_to_file(
            str(target), duration_seconds=0.2, sample_rate=16000, channels=1
        )
    )

    assert target.is_file()
    with wave.open(str(target), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() > 0


def test_record_to_file_invalid_path_raises(tmp_path: Path) -> None:
    """A path under a nonexistent directory raises a sensible error.

    Does NOT require audio hardware: stdlib ``wave.open()`` checks the
    path before any audio capture begins.
    """
    bogus = tmp_path / "does_not_exist" / "rec.wav"
    with pytest.raises((FileNotFoundError, OSError)):
        decibri.record_to_file(str(bogus), duration_seconds=0.1)


@pytest.mark.requires_audio_input
def test_record_to_file_custom_sample_rate_honored(tmp_path: Path) -> None:
    """A non-default ``sample_rate`` (24000) is reflected in the output WAV."""
    target = tmp_path / "rec_24k.wav"
    decibri.record_to_file(
        str(target), duration_seconds=0.2, sample_rate=24000, channels=1
    )

    with wave.open(str(target), "rb") as wav:
        assert wav.getframerate() == 24000
