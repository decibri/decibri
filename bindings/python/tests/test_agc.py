"""AGC (automatic gain control) capture tests.

AGC is pure DSP: a level-control engine with no bundled file, no ONNX model, and
no ORT. So the whole binding surface (construction, off-by-default, out-of-range
rejection) is CI-safe and runs without audio hardware or ORT. The engine's
temporal behaviour (cold start, convergence, gating) is covered by the core Rust
tests; these cover the binding surface only.

The agc target is an integer dBFS level in -40..-3 (typical -18). The core names
this failure, so an out-of-range value is an ``AgcTargetOutOfRange`` at the
wrapper, catchable as ``DecibriError``.
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
    """A target outside [-40, -3] raises AgcTargetOutOfRange, below and above."""
    with pytest.raises(
        decibri_exc.AgcTargetOutOfRange, match=r"agc must be in \[-40, -3\]"
    ):
        Microphone(agc=-100)
    with pytest.raises(
        decibri_exc.AgcTargetOutOfRange, match=r"agc must be in \[-40, -3\]"
    ):
        Microphone(agc=0)


def test_async_agc_out_of_range_raises() -> None:
    with pytest.raises(
        decibri_exc.AgcTargetOutOfRange, match=r"agc must be in \[-40, -3\]"
    ):
        AsyncMicrophone(agc=-41)


def test_agc_out_of_range_is_catchable_as_decibri_error() -> None:
    """The wrapper's range check is catchable through the base class.

    The core names this failure, so the wrapper raises the core's class rather
    than a bare ``ValueError``: ``except DecibriError`` catches a bad agc target
    the same way it catches a bad dtype.
    """
    assert issubclass(decibri_exc.AgcTargetOutOfRange, decibri_exc.DecibriError)
    with pytest.raises(decibri_exc.DecibriError):
        Microphone(agc=0)


def test_agc_out_of_range_from_the_raw_bridge_raises_the_same_class() -> None:
    """A raw-bridge consumer that bypasses the wrapper trips the core guard.

    Same class, core message: the wrapper's earlier check does not change the
    identity a bypassing caller sees.
    """
    from decibri import _decibri

    with pytest.raises(decibri_exc.AgcTargetOutOfRange) as exc_info:
        _decibri.MicrophoneBridge(
            sample_rate=16000,
            channels=1,
            frames_per_buffer=1600,
            format="int16",
            agc=0,
        )
    assert str(exc_info.value) == "agc target level must be between -40 and -3"
