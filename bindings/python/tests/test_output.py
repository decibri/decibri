"""Phase 2 audio output lifecycle tests.

All tests in this module require real audio output hardware. CI auto-skips
via the requires_audio_output marker.

Coverage:
- start / stop / close lifecycle
- write / drain semantics
- is_playing property reflects start/stop state
- write before start raises OutputStreamClosed
- Idempotent stop / close
- Static devices() returns a non-empty list
- Context-manager round-trip with write + drain
"""

import pytest

from decibri import (
    AlreadyRunning,
    Speaker,
    OutputDeviceInfo,
    OutputStreamClosed,
)


pytestmark = pytest.mark.requires_audio_output


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------


def test_devices_returns_list_of_output_device_info() -> None:
    devices = Speaker.devices()
    assert isinstance(devices, list)
    assert len(devices) > 0
    for d in devices:
        assert isinstance(d, OutputDeviceInfo)
        assert isinstance(d.name, str)
        assert isinstance(d.index, int)
        assert isinstance(d.max_output_channels, int)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_then_stop() -> None:
    o = Speaker()
    assert o.is_playing is False
    o.start()
    assert o.is_playing is True
    o.stop()
    assert o.is_playing is False


def test_start_twice_raises_already_running() -> None:
    o = Speaker()
    o.start()
    try:
        with pytest.raises(AlreadyRunning):
            o.start()
    finally:
        o.stop()


def test_stop_twice_is_idempotent() -> None:
    o = Speaker()
    o.start()
    o.stop()
    o.stop()


def test_close_is_alias_for_stop() -> None:
    o = Speaker()
    o.start()
    o.close()
    assert o.is_playing is False


def test_context_manager_lifecycle() -> None:
    with Speaker() as o:
        assert o.is_playing is True
    assert o.is_playing is False


# ---------------------------------------------------------------------------
# write / drain semantics
# ---------------------------------------------------------------------------


def test_write_before_start_raises_closed() -> None:
    o = Speaker()
    with pytest.raises(OutputStreamClosed):
        o.write(b"\x00" * 1024)


def test_drain_before_start_raises_closed() -> None:
    o = Speaker()
    with pytest.raises(OutputStreamClosed):
        o.drain()


def test_write_then_drain_roundtrip() -> None:
    """Write a short silence buffer + drain. Verifies the full output cycle."""
    with Speaker(sample_rate=16000, channels=1, dtype="int16") as o:
        # ~100ms of silence at 16kHz mono int16 = 1600 samples * 2 bytes = 3200 bytes
        silence = b"\x00" * 3200
        o.write(silence)
        o.drain()
