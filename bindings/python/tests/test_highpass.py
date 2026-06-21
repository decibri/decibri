"""High-pass capture tests.

High-pass is pure DSP: a second-order Butterworth biquad with no bundled file,
no ONNX model, and no ORT. So the whole surface (construction, unknown-name
rejection, off-by-default) is CI-safe and runs without audio hardware or ORT.
The filter's DSP response is covered by the core Rust tests; these cover the
binding surface only.

The high-pass selector mirrors the denoise shape: a closed cutoff name turns it
on, absence leaves it off, an unknown name is a clear error. The value set is
designed to grow (further cutoffs are additive).
"""

from __future__ import annotations

import pytest

from decibri import AsyncMicrophone, Microphone


def test_highpass_constructs() -> None:
    """A valid cutoff name constructs; the filter is built later, at start()."""
    mic = Microphone(sample_rate=16000, channels=1, highpass="80hz")
    assert mic is not None


def test_async_highpass_constructs() -> None:
    """AsyncMicrophone mirrors the sync surface and accepts highpass."""
    mic = AsyncMicrophone(sample_rate=16000, channels=1, highpass="80hz")
    assert mic is not None


def test_highpass_off_by_default() -> None:
    """No highpass kwarg constructs identically to a plain microphone."""
    mic = Microphone(sample_rate=16000, channels=1)
    assert mic is not None


def test_highpass_unknown_name_raises() -> None:
    """An unrecognized cutoff name is a clear ValueError, not a silent miss."""
    with pytest.raises(ValueError, match="Invalid highpass value"):
        Microphone(highpass="50hz")  # type: ignore[arg-type]


def test_async_highpass_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Invalid highpass value"):
        AsyncMicrophone(highpass="whatever")  # type: ignore[arg-type]
