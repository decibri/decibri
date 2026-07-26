"""Tests for numpy ndarray support.

Covers: construction with as_ndarray=True (no NotImplementedError),
bytes-mode regression (default), ndarray return on read, dtype/shape
correctness, ndarray-accept on write, dtype mismatch rejection, async
parallel, the missing-numpy construction guard (absence simulated via
``sys.modules``), and the 5-test smoke baseline still passing.

By convention, every async test has explicit
``@pytest.mark.asyncio``. Hardware-gated tests use the existing
``requires_audio_input`` / ``requires_audio_output`` markers from
``conftest.py`` so cloud CI runners (no audio devices) skip cleanly.

Tests that don't open cpal streams (constructor + dtype/shape
assertions on synthetic data) don't need hardware markers.
"""

from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from decibri import AsyncFile, AsyncMicrophone, AsyncSpeaker, File, Microphone, Speaker


# ---------------------------------------------------------------------------
# Construction tests (no hardware needed)
# ---------------------------------------------------------------------------


def test_construction_with_as_ndarray_true_no_longer_raises() -> None:
    """``Microphone(as_ndarray=True)`` constructs without raising.

    The wrapper kwarg is ``as_ndarray``; the bridge keeps the original
    ``numpy`` name internally.
    """
    decibri = Microphone(as_ndarray=True, vad=False)
    assert decibri is not None


def test_construction_default_unchanged() -> None:
    """Default ``as_ndarray=False`` preserves the bytes-mode behavior verbatim."""
    decibri = Microphone(vad=False)
    assert decibri is not None


def test_async_construction_with_as_ndarray_true() -> None:
    """``AsyncMicrophone(as_ndarray=True)`` constructs without exception."""
    async_decibri = AsyncMicrophone(as_ndarray=True, vad=False)
    assert async_decibri is not None


# ---------------------------------------------------------------------------
# Read tests (require audio input hardware)
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
def test_read_returns_ndarray_when_as_ndarray_true() -> None:
    """Sync read with ``as_ndarray=True`` returns ``numpy.ndarray``, not bytes."""
    with Microphone(as_ndarray=True, vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)


@pytest.mark.requires_audio_input
def test_read_returns_bytes_when_as_ndarray_false() -> None:
    """Default sync read returns ``bytes`` (bytes-mode regression)."""
    with Microphone(vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, bytes)


@pytest.mark.requires_audio_input
def test_read_dtype_matches_int16_format() -> None:
    """``as_ndarray=True`` with ``dtype='int16'`` returns dtype ``np.int16``."""
    with Microphone(as_ndarray=True, dtype="int16", vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.int16


@pytest.mark.requires_audio_input
def test_read_dtype_matches_float32_format() -> None:
    """``as_ndarray=True`` with ``dtype='float32'`` returns dtype ``np.float32``."""
    with Microphone(as_ndarray=True, dtype="float32", vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32


@pytest.mark.requires_audio_input
def test_read_shape_mono_is_1d() -> None:
    """Mono read (``channels=1``) returns 1-D ndarray with shape ``(N,)``."""
    with Microphone(as_ndarray=True, channels=1, vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)
            assert chunk.ndim == 1


# ---------------------------------------------------------------------------
# Write tests (require audio output hardware for end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_output
def test_write_accepts_ndarray_int16() -> None:
    """Output write accepts ``np.int16`` ndarray (mono 1-D)."""
    samples = np.zeros(1600, dtype=np.int16)  # 100 ms at 16 kHz
    with Speaker(dtype="int16") as o:
        o.write(samples)
        o.drain()


@pytest.mark.requires_audio_output
def test_write_accepts_bytes_regression() -> None:
    """Output write still accepts ``bytes`` (bytes-mode regression)."""
    samples = b"\x00\x00" * 1600
    with Speaker(dtype="int16") as o:
        o.write(samples)
        o.drain()


@pytest.mark.requires_audio_output
def test_write_rejects_dtype_mismatch() -> None:
    """``np.float32`` ndarray to ``dtype='int16'`` output raises TypeError.

    The dtype check fires at ``write()`` time, but the test path requires
    ``output.start()`` to succeed first (the bridge's stream-state check
    raises ``SpeakerStreamClosed`` before the dtype check would run if the
    output is not started). On CI runners without audio output hardware,
    ``start()`` fails with ``StreamOpenFailed`` before the test can
    exercise the dtype-rejection path; hence the
    ``requires_audio_output`` marker. Restructuring the bridge to
    validate dtype before stream-state would be a real refactor; out of
    scope here. The dtype-rejection behavior is still verified, just
    only on hardware.
    """
    samples = np.zeros(1600, dtype=np.float32)
    output = Speaker(dtype="int16")
    output.start()
    try:
        with pytest.raises(TypeError):
            output.write(samples)
    finally:
        output.stop()


# ---------------------------------------------------------------------------
# Async parallel tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.requires_audio_input
async def test_async_read_returns_ndarray() -> None:
    """``AsyncMicrophone(as_ndarray=True).read()`` returns ndarray (async parallel)."""
    async with AsyncMicrophone(as_ndarray=True, vad=False) as d:
        chunk = await d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)


@pytest.mark.asyncio
@pytest.mark.requires_audio_output
async def test_async_write_accepts_ndarray() -> None:
    """``AsyncSpeaker.write()`` accepts ndarray (async parallel)."""
    samples = np.zeros(1600, dtype=np.int16)
    async with AsyncSpeaker(dtype="int16") as o:
        await o.write(samples)
        await o.drain()


# ---------------------------------------------------------------------------
# Missing-numpy guard (no hardware needed)
#
# Absence is simulated per test with a ``None`` entry in ``sys.modules``:
# any ``import numpy`` then raises ImportError while the module object
# already imported above (the ``np`` global) stays usable, and pytest's
# ``monkeypatch`` restores the real entry on teardown, so no state
# outlives the test. Mirrors the bundled-model tests, which simulate
# their failure by monkeypatching the resource lookup.
# ---------------------------------------------------------------------------

_MISSING_NUMPY_MESSAGE = (
    "numpy is not installed. Install with: pip install decibri[numpy]"
)


def _block_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import numpy`` raise ImportError for the current test."""
    monkeypatch.setitem(sys.modules, "numpy", None)


def _write_wav(path: Path, n_samples: int = 8000) -> None:
    """Write a small silent mono 16-bit PCM WAV via the standard library."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


def test_missing_numpy_microphone_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Microphone(as_ndarray=True)`` without numpy raises ``ImportError``."""
    _block_numpy(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        Microphone(as_ndarray=True, vad=False)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE


def test_missing_numpy_is_caught_by_except_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The missing-numpy failure is caught by a plain ``except Exception``."""
    _block_numpy(monkeypatch)
    try:
        Microphone(as_ndarray=True, vad=False)
    except Exception as exc:
        assert isinstance(exc, ImportError)
        assert str(exc) == _MISSING_NUMPY_MESSAGE
    else:
        pytest.fail("Microphone(as_ndarray=True) did not raise without numpy")


def test_missing_numpy_async_microphone_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AsyncMicrophone(as_ndarray=True)`` without numpy raises ``ImportError``."""
    _block_numpy(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        AsyncMicrophone(as_ndarray=True, vad=False)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE


@pytest.mark.asyncio
async def test_missing_numpy_async_microphone_open_raises_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``await AsyncMicrophone.open(as_ndarray=True)`` raises ``ImportError``."""
    _block_numpy(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        await AsyncMicrophone.open(as_ndarray=True, vad=False)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE


def test_missing_numpy_file_surfaces_raise_importerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``File(path)``, ``File.open`` and ``File.buffer`` all raise the guard."""
    _block_numpy(monkeypatch)
    path = tmp_path / "clip.wav"
    _write_wav(path)
    with pytest.raises(ImportError) as excinfo:
        File(path, as_ndarray=True)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE
    with pytest.raises(ImportError) as excinfo:
        File.open(path, as_ndarray=True)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE
    with pytest.raises(ImportError) as excinfo:
        File.buffer([0.0] * 1600, input_rate=16000, as_ndarray=True)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE


@pytest.mark.asyncio
async def test_missing_numpy_async_file_surfaces_raise_importerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``AsyncFile(path)``, ``AsyncFile.open`` and ``AsyncFile.buffer`` raise."""
    _block_numpy(monkeypatch)
    path = tmp_path / "clip.wav"
    _write_wav(path)
    with pytest.raises(ImportError) as excinfo:
        AsyncFile(path, as_ndarray=True)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE
    with pytest.raises(ImportError) as excinfo:
        await AsyncFile.open(path, as_ndarray=True)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE
    with pytest.raises(ImportError) as excinfo:
        await AsyncFile.buffer([0.0] * 1600, input_rate=16000, as_ndarray=True)
    assert str(excinfo.value) == _MISSING_NUMPY_MESSAGE


def test_missing_numpy_default_bytes_mode_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``as_ndarray=False`` constructs and reads bytes without numpy."""
    _block_numpy(monkeypatch)
    assert Microphone(vad=False) is not None
    path = tmp_path / "clip.wav"
    _write_wav(path)
    chunk = File(path).read()
    assert isinstance(chunk, bytes)
    buffered = File.buffer([0.0] * 1600, input_rate=16000).read()
    assert isinstance(buffered, bytes)


# ---------------------------------------------------------------------------
# With numpy present, the guarded surfaces still work (regression for the
# guard sitting in front of a working path). File reads need no hardware.
# ---------------------------------------------------------------------------


def test_file_read_returns_ndarray_with_numpy_present(tmp_path: Path) -> None:
    """``File(path, as_ndarray=True).read()`` returns an int16 ndarray."""
    path = tmp_path / "clip.wav"
    _write_wav(path)
    for file in (File(path, as_ndarray=True), File.open(path, as_ndarray=True)):
        chunk = file.read()
        assert isinstance(chunk, np.ndarray)
        assert chunk.dtype == np.int16


def test_file_buffer_read_returns_ndarray_with_numpy_present() -> None:
    """``File.buffer(..., as_ndarray=True).read()`` returns an ndarray."""
    chunk = File.buffer([0.0] * 1600, input_rate=16000, as_ndarray=True).read()
    assert isinstance(chunk, np.ndarray)


@pytest.mark.asyncio
async def test_async_file_read_returns_ndarray_with_numpy_present(
    tmp_path: Path,
) -> None:
    """Every ``AsyncFile`` constructor still yields ndarray reads."""
    path = tmp_path / "clip.wav"
    _write_wav(path)
    for f in (
        AsyncFile(path, as_ndarray=True),
        await AsyncFile.open(path, as_ndarray=True),
        await AsyncFile.buffer([0.0] * 1600, input_rate=16000, as_ndarray=True),
    ):
        chunk = await f.read()
        assert isinstance(chunk, np.ndarray)


@pytest.mark.asyncio
async def test_async_microphone_open_constructs_with_numpy_present() -> None:
    """``await AsyncMicrophone.open(as_ndarray=True)`` constructs cleanly."""
    mic = await AsyncMicrophone.open(as_ndarray=True, vad=False)
    assert mic is not None
