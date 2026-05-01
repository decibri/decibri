"""Phase 2 audio capture lifecycle tests.

All tests in this module require real audio input hardware. CI auto-skips
via the requires_audio_input marker.

Coverage:
- start / stop / read / iterator / context-manager lifecycle
- is_open property reflects start/stop state
- read() with and without timeout returns bytes
- read() returns bytes of the expected length
- Idempotent stop / close
- Static devices() returns a non-empty list when audio hardware is present
"""

import pytest

from decibri import (
    AlreadyRunning,
    CaptureStreamClosed,
    Microphone,
    DeviceInfo,
)


pytestmark = pytest.mark.requires_audio_input


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------


def test_input_devices_returns_list_of_device_info() -> None:
    devices = Microphone.input_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0
    for d in devices:
        assert isinstance(d, DeviceInfo)
        assert isinstance(d.name, str)
        assert isinstance(d.index, int)
        assert isinstance(d.max_input_channels, int)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_then_stop() -> None:
    d = Microphone()
    assert d.is_open is False
    d.start()
    assert d.is_open is True
    d.stop()
    assert d.is_open is False


def test_start_twice_raises_already_running() -> None:
    d = Microphone()
    d.start()
    try:
        with pytest.raises(AlreadyRunning):
            d.start()
    finally:
        d.stop()


def test_stop_twice_is_idempotent() -> None:
    d = Microphone()
    d.start()
    d.stop()
    d.stop()  # second stop is a no-op


def test_context_manager_lifecycle() -> None:
    with Microphone() as d:
        assert d.is_open is True
    assert d.is_open is False


# ---------------------------------------------------------------------------
# read() surface
# ---------------------------------------------------------------------------


def test_read_returns_bytes_with_timeout() -> None:
    with Microphone(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        chunk = d.read(timeout_ms=500)
        assert chunk is not None
        assert isinstance(chunk, bytes)
        # int16 default; 512 frames * 1 channel * 2 bytes = 1024 bytes
        assert len(chunk) == 1024


def test_read_int16_chunk_size() -> None:
    with Microphone(sample_rate=16000, channels=1, frames_per_buffer=256, dtype="int16") as d:
        chunk = d.read(timeout_ms=500)
        assert chunk is not None
        assert len(chunk) == 256 * 1 * 2  # 512 bytes


def test_read_float32_chunk_size() -> None:
    with Microphone(sample_rate=16000, channels=1, frames_per_buffer=256, dtype="float32") as d:
        chunk = d.read(timeout_ms=500)
        assert chunk is not None
        assert len(chunk) == 256 * 1 * 4  # 1024 bytes


def test_read_after_stop_raises_closed() -> None:
    d = Microphone()
    d.start()
    d.stop()
    with pytest.raises(CaptureStreamClosed):
        d.read(timeout_ms=100)


# ---------------------------------------------------------------------------
# Iterator protocol
# ---------------------------------------------------------------------------


def test_iterator_yields_bytes() -> None:
    with Microphone(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        count = 0
        for chunk in d:
            assert isinstance(chunk, bytes)
            count += 1
            if count >= 3:
                break
        assert count == 3


# ---------------------------------------------------------------------------
# Phase 7.7 Item B1: read_with_metadata + iter_with_metadata + Chunk
# ---------------------------------------------------------------------------


def test_read_with_metadata_returns_chunk() -> None:
    """``read_with_metadata()`` returns a frozen ``Chunk`` with all fields."""
    import dataclasses

    from decibri import Chunk

    with Microphone(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        chunk = d.read_with_metadata(timeout_ms=500)
        assert chunk is not None
        assert isinstance(chunk, Chunk)
        # All five fields populated:
        assert isinstance(chunk.data, bytes)
        assert len(chunk.data) == 1024
        assert isinstance(chunk.timestamp, float)
        assert chunk.timestamp > 0.0
        assert chunk.sequence == 0  # first chunk after start
        assert chunk.is_speaking is False  # vad disabled
        assert chunk.vad_score == 0.0
        # Frozen: assignment after construction raises FrozenInstanceError.
        with pytest.raises(dataclasses.FrozenInstanceError):
            chunk.sequence = 1  # type: ignore[misc]


def test_read_with_metadata_sequence_increments() -> None:
    """Successive ``read_with_metadata()`` calls produce monotonic sequence values."""
    with Microphone(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        c0 = d.read_with_metadata(timeout_ms=500)
        c1 = d.read_with_metadata(timeout_ms=500)
        c2 = d.read_with_metadata(timeout_ms=500)
        assert c0 is not None
        assert c1 is not None
        assert c2 is not None
        assert c0.sequence == 0
        assert c1.sequence == 1
        assert c2.sequence == 2


def test_read_with_metadata_sequence_resets_on_restart() -> None:
    """Stopping and restarting the microphone resets sequence to 0."""
    d = Microphone(sample_rate=16000, channels=1, frames_per_buffer=512)
    d.start()
    c0 = d.read_with_metadata(timeout_ms=500)
    c1 = d.read_with_metadata(timeout_ms=500)
    assert c0 is not None and c0.sequence == 0
    assert c1 is not None and c1.sequence == 1
    d.stop()
    d.start()
    try:
        c2 = d.read_with_metadata(timeout_ms=500)
        assert c2 is not None
        assert c2.sequence == 0  # restart resets sequence
    finally:
        d.stop()


def test_iter_with_metadata_yields_chunks() -> None:
    """``iter_with_metadata()`` yields Chunk objects until break."""
    from decibri import Chunk

    with Microphone(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        count = 0
        last_seq = -1
        for chunk in d.iter_with_metadata():
            assert isinstance(chunk, Chunk)
            assert chunk.sequence == last_seq + 1
            last_seq = chunk.sequence
            count += 1
            if count >= 3:
                break
        assert count == 3


def test_chunk_is_frozen_dataclass() -> None:
    """Chunk is a frozen dataclass with slots; cannot be mutated post-construction."""
    import dataclasses

    from decibri import Chunk

    chunk = Chunk(data=b"x" * 4, timestamp=1.0, sequence=0, is_speaking=False, vad_score=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.timestamp = 2.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.sequence = 5  # type: ignore[misc]
