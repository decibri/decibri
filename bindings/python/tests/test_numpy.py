"""Phase 6 tests for numpy ndarray support.

Covers the goals from Section 2 of phase-6-numpy-zerocopy.md:
construction with numpy=True (no NotImplementedError), bytes-mode
regression (default), ndarray return on read, dtype/shape correctness,
ndarray-accept on write, dtype mismatch rejection, async parallel,
and the 5-test smoke baseline still passing.

Per Phase 5 conventions, every async test has explicit
``@pytest.mark.asyncio``. Hardware-gated tests use the existing
``requires_audio_input`` / ``requires_audio_output`` markers from
``conftest.py`` so cloud CI runners (no audio devices) skip cleanly.

Tests that don't open cpal streams (constructor + dtype/shape
assertions on synthetic data) don't need hardware markers.
"""

from __future__ import annotations

import numpy as np
import pytest

from decibri import AsyncDecibri, AsyncDecibriOutput, Decibri, DecibriOutput


# ---------------------------------------------------------------------------
# Construction tests (no hardware needed)
# ---------------------------------------------------------------------------


def test_construction_with_numpy_true_no_longer_raises() -> None:
    """Phase 6 removes the Phase 2 NotImplementedError guard.

    Pre-Phase-6, ``Decibri(numpy=True)`` raised
    ``NotImplementedError("numpy=True is not supported in 0.1.0a1...")``.
    Phase 6 wires the actual ndarray path; construction now succeeds.
    """
    decibri = Decibri(numpy=True, vad=False)
    assert decibri is not None


def test_construction_numpy_false_default_unchanged() -> None:
    """Default ``numpy=False`` preserves the Phase 5 behavior verbatim."""
    decibri = Decibri(vad=False)
    assert decibri is not None


def test_async_construction_with_numpy_true() -> None:
    """``AsyncDecibri(numpy=True)`` constructs without exception."""
    async_decibri = AsyncDecibri(numpy=True, vad=False)
    assert async_decibri is not None


# ---------------------------------------------------------------------------
# Read tests (require audio input hardware)
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
def test_read_returns_ndarray_when_numpy_true() -> None:
    """Sync read with ``numpy=True`` returns ``numpy.ndarray``, not bytes."""
    with Decibri(numpy=True, vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)


@pytest.mark.requires_audio_input
def test_read_returns_bytes_when_numpy_false() -> None:
    """Default sync read returns ``bytes`` (Phase 5 baseline regression)."""
    with Decibri(vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, bytes)


@pytest.mark.requires_audio_input
def test_read_dtype_matches_int16_format() -> None:
    """``numpy=True`` with ``format='int16'`` returns dtype ``np.int16``."""
    with Decibri(numpy=True, format="int16", vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.int16


@pytest.mark.requires_audio_input
def test_read_dtype_matches_float32_format() -> None:
    """``numpy=True`` with ``format='float32'`` returns dtype ``np.float32``."""
    with Decibri(numpy=True, format="float32", vad=False) as d:
        chunk = d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32


@pytest.mark.requires_audio_input
def test_read_shape_mono_is_1d() -> None:
    """Mono read (``channels=1``) returns 1-D ndarray with shape ``(N,)``."""
    with Decibri(numpy=True, channels=1, vad=False) as d:
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
    with DecibriOutput(format="int16") as o:
        o.write(samples)
        o.drain()


@pytest.mark.requires_audio_output
def test_write_accepts_bytes_regression() -> None:
    """Output write still accepts ``bytes`` (Phase 5 baseline regression)."""
    samples = b"\x00\x00" * 1600
    with DecibriOutput(format="int16") as o:
        o.write(samples)
        o.drain()


@pytest.mark.requires_audio_output
def test_write_rejects_dtype_mismatch() -> None:
    """``np.float32`` ndarray to ``format='int16'`` output raises TypeError.

    The dtype check fires at ``write()`` time, but the test path requires
    ``output.start()`` to succeed first (the bridge's stream-state check
    raises ``OutputStreamClosed`` before the dtype check would run if the
    output is not started). On CI runners without audio output hardware,
    ``start()`` fails with ``StreamOpenFailed`` before the test can
    exercise the dtype-rejection path; hence the
    ``requires_audio_output`` marker. Restructuring the bridge to
    validate dtype before stream-state would be a real refactor; out of
    Phase 6 scope. The dtype-rejection behavior is still verified, just
    only on hardware.
    """
    samples = np.zeros(1600, dtype=np.float32)
    output = DecibriOutput(format="int16")
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
    """``AsyncDecibri(numpy=True).read()`` returns ndarray (async parallel)."""
    async with AsyncDecibri(numpy=True, vad=False) as d:
        chunk = await d.read()
        if chunk is not None:
            assert isinstance(chunk, np.ndarray)


@pytest.mark.asyncio
@pytest.mark.requires_audio_output
async def test_async_write_accepts_ndarray() -> None:
    """``AsyncDecibriOutput.write()`` accepts ndarray (async parallel)."""
    samples = np.zeros(1600, dtype=np.int16)
    async with AsyncDecibriOutput(format="int16") as o:
        await o.write(samples)
        await o.drain()
