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
    Decibri,
    DeviceInfo,
)


pytestmark = pytest.mark.requires_audio_input


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------


def test_devices_returns_list_of_device_info() -> None:
    devices = Decibri.devices()
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
    d = Decibri()
    assert d.is_open is False
    d.start()
    assert d.is_open is True
    d.stop()
    assert d.is_open is False


def test_start_twice_raises_already_running() -> None:
    d = Decibri()
    d.start()
    try:
        with pytest.raises(AlreadyRunning):
            d.start()
    finally:
        d.stop()


def test_stop_twice_is_idempotent() -> None:
    d = Decibri()
    d.start()
    d.stop()
    d.stop()  # second stop is a no-op


def test_context_manager_lifecycle() -> None:
    with Decibri() as d:
        assert d.is_open is True
    assert d.is_open is False


# ---------------------------------------------------------------------------
# read() surface
# ---------------------------------------------------------------------------


def test_read_returns_bytes_with_timeout() -> None:
    with Decibri(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        chunk = d.read(timeout_ms=500)
        assert chunk is not None
        assert isinstance(chunk, bytes)
        # int16 default; 512 frames * 1 channel * 2 bytes = 1024 bytes
        assert len(chunk) == 1024


def test_read_int16_chunk_size() -> None:
    with Decibri(sample_rate=16000, channels=1, frames_per_buffer=256, format="int16") as d:
        chunk = d.read(timeout_ms=500)
        assert chunk is not None
        assert len(chunk) == 256 * 1 * 2  # 512 bytes


def test_read_float32_chunk_size() -> None:
    with Decibri(sample_rate=16000, channels=1, frames_per_buffer=256, format="float32") as d:
        chunk = d.read(timeout_ms=500)
        assert chunk is not None
        assert len(chunk) == 256 * 1 * 4  # 1024 bytes


def test_read_after_stop_raises_closed() -> None:
    d = Decibri()
    d.start()
    d.stop()
    with pytest.raises(CaptureStreamClosed):
        d.read(timeout_ms=100)


# ---------------------------------------------------------------------------
# Iterator protocol
# ---------------------------------------------------------------------------


def test_iterator_yields_bytes() -> None:
    with Decibri(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        count = 0
        for chunk in d:
            assert isinstance(chunk, bytes)
            count += 1
            if count >= 3:
                break
        assert count == 3
