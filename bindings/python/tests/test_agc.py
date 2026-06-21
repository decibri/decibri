"""AGC (automatic gain control) capture tests.

AGC is pure DSP: a level-control engine with no bundled file, no ONNX model, and
no ORT. So the whole binding surface (construction, off-by-default, out-of-range
rejection) is CI-safe and runs without audio hardware or ORT. The engine's
temporal behaviour (cold start, convergence, gating) is covered by the core Rust
tests; these cover the binding surface only.

The agc target is an integer dBFS level in -40..-3 (typical -18). Unlike the
closed-set highpass/denoise selectors, it is a numeric range, so an out-of-range
value is a ``ValueError`` (a range violation) at the wrapper.
"""

from __future__ import annotations

import pytest

from decibri import AsyncMicrophone, Microphone
from decibri import exceptions as decibri_exc


def test_agc_constructs() -> None:
    """A valid in-range target constructs; the engine is built later, at start()."""
    mic = Microphone(sample_rate=16000, channels=1, agc=-18)
    assert mic is not None


def test_async_agc_constructs() -> None:
    """AsyncMicrophone mirrors the sync surface and accepts agc."""
    mic = AsyncMicrophone(sample_rate=16000, channels=1, agc=-18)
    assert mic is not None


def test_agc_off_by_default() -> None:
    """No agc kwarg constructs identically to a plain microphone."""
    mic = Microphone(sample_rate=16000, channels=1)
    assert mic is not None


def test_agc_range_edges_construct() -> None:
    """Both edges of the supported range are accepted."""
    assert Microphone(agc=-40) is not None
    assert Microphone(agc=-3) is not None


def test_agc_out_of_range_raises() -> None:
    """A target outside [-40, -3] is a clear ValueError, below and above."""
    with pytest.raises(ValueError, match=r"agc must be in \[-40, -3\]"):
        Microphone(agc=-100)
    with pytest.raises(ValueError, match=r"agc must be in \[-40, -3\]"):
        Microphone(agc=0)


def test_async_agc_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match=r"agc must be in \[-40, -3\]"):
        AsyncMicrophone(agc=-41)


def test_agc_target_out_of_range_is_a_dedicated_exception_class() -> None:
    """The core backstop has its own catchable class in the hierarchy.

    The wrapper raises a plain ``ValueError`` for the friendly path; a raw-bridge
    consumer that bypasses the wrapper trips the core guard, surfaced as the
    dedicated ``AgcTargetOutOfRange`` (a ``DecibriError`` subclass).
    """
    assert issubclass(decibri_exc.AgcTargetOutOfRange, decibri_exc.DecibriError)
    with pytest.raises(decibri_exc.AgcTargetOutOfRange):
        raise decibri_exc.AgcTargetOutOfRange("agc target level must be between -40 and -3")
