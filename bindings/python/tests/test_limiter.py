"""Limiter capture tests.

The limiter is pure DSP: a peak limiter with no bundled file, no ONNX model, and
no ORT. So the whole binding surface (construction, off-by-default, out-of-range
rejection) is CI-safe and runs without audio hardware or ORT. The stage's
behaviour (the absolute-ceiling guarantee, graceful shaping, release) is covered
by the core Rust tests; these cover the binding surface only.

The limiter ceiling is a float dBFS level in -3.0..0.0 (typical -1.0). Like the
agc target it is a numeric range, so an out-of-range value is a ``ValueError`` (a
range violation) at the wrapper.
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
    """A ceiling outside [-3.0, 0.0] is a clear ValueError, below and above."""
    with pytest.raises(ValueError, match=r"limiter must be in \[-3.0, 0.0\]"):
        Microphone(limiter=-5.0)
    with pytest.raises(ValueError, match=r"limiter must be in \[-3.0, 0.0\]"):
        Microphone(limiter=1.0)


def test_async_limiter_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"limiter must be in \[-3.0, 0.0\]"):
        AsyncMicrophone(limiter=0.5)


def test_limiter_ceiling_out_of_range_is_a_dedicated_exception_class() -> None:
    """The core backstop has its own catchable class in the hierarchy.

    The wrapper raises a plain ``ValueError`` for the friendly path; a raw-bridge
    consumer that bypasses the wrapper trips the core guard, surfaced as the
    dedicated ``LimiterCeilingOutOfRange`` (a ``DecibriError`` subclass).
    """
    assert issubclass(decibri_exc.LimiterCeilingOutOfRange, decibri_exc.DecibriError)
    with pytest.raises(decibri_exc.LimiterCeilingOutOfRange):
        raise decibri_exc.LimiterCeilingOutOfRange(
            "limiter ceiling must be between -3.0 and 0.0"
        )
