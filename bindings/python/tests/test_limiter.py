"""Limiter capture tests.

The limiter is pure DSP: a peak limiter with no bundled file, no ONNX model, and
no ORT. So the whole binding surface (construction, off-by-default, out-of-range
rejection) is CI-safe and runs without audio hardware or ORT. The stage's
behaviour (the absolute-ceiling guarantee, graceful shaping, release) is covered
by the core Rust tests; these cover the binding surface only.

The limiter ceiling is a float dBFS level in -3.0..0.0 (typical -1.0). Like the
agc target the core names this failure, so an out-of-range value is a
``LimiterCeilingOutOfRange`` at the wrapper, catchable as ``DecibriError``.
"""

from __future__ import annotations

import pytest

from decibri import AsyncMicrophone, Microphone
from decibri import exceptions as decibri_exc


def test_limiter_constructs() -> None:
    """A valid in-range ceiling constructs; the stage is built later, at start()."""
    mic = Microphone(sample_rate=16000, channels=1, limiter=-1.0)
    assert mic is not None


def test_async_limiter_constructs() -> None:
    """AsyncMicrophone mirrors the sync surface and accepts limiter."""
    mic = AsyncMicrophone(sample_rate=16000, channels=1, limiter=-1.0)
    assert mic is not None


def test_limiter_off_by_default() -> None:
    """No limiter kwarg constructs identically to a plain microphone."""
    mic = Microphone(sample_rate=16000, channels=1)
    assert mic is not None


def test_limiter_range_edges_construct() -> None:
    """Both edges of the supported range are accepted."""
    assert Microphone(limiter=-3.0) is not None
    assert Microphone(limiter=0.0) is not None


def test_limiter_out_of_range_raises() -> None:
    """A ceiling outside [-3.0, 0.0] raises LimiterCeilingOutOfRange, below and above."""
    with pytest.raises(
        decibri_exc.LimiterCeilingOutOfRange, match=r"limiter must be in \[-3.0, 0.0\]"
    ):
        Microphone(limiter=-5.0)
    with pytest.raises(
        decibri_exc.LimiterCeilingOutOfRange, match=r"limiter must be in \[-3.0, 0.0\]"
    ):
        Microphone(limiter=1.0)


def test_async_limiter_out_of_range_raises() -> None:
    with pytest.raises(
        decibri_exc.LimiterCeilingOutOfRange, match=r"limiter must be in \[-3.0, 0.0\]"
    ):
        AsyncMicrophone(limiter=0.5)


def test_limiter_out_of_range_is_catchable_as_decibri_error() -> None:
    """The wrapper's range check is catchable through the base class.

    The core names this failure, so the wrapper raises the core's class rather
    than a bare ``ValueError``: ``except DecibriError`` catches a bad limiter
    ceiling the same way it catches a bad dtype.
    """
    assert issubclass(decibri_exc.LimiterCeilingOutOfRange, decibri_exc.DecibriError)
    with pytest.raises(decibri_exc.DecibriError):
        Microphone(limiter=1.0)


def test_limiter_out_of_range_from_the_raw_bridge_raises_the_same_class() -> None:
    """A raw-bridge consumer that bypasses the wrapper trips the core guard.

    Same class, core message: the wrapper's earlier check does not change the
    identity a bypassing caller sees.
    """
    from decibri import _decibri

    with pytest.raises(decibri_exc.LimiterCeilingOutOfRange) as exc_info:
        _decibri.MicrophoneBridge(
            sample_rate=16000,
            channels=1,
            frames_per_buffer=1600,
            format="int16",
            limiter=1.0,
        )
    assert str(exc_info.value) == "limiter ceiling must be between -3.0 and 0.0"
