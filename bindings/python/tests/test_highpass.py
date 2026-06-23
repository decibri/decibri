"""High-pass capture tests.

High-pass is pure DSP: a second-order Butterworth biquad with no bundled file,
no ONNX model, and no ORT. So the whole surface (construction, rejection of
out-of-set values, off-by-default) is CI-safe and runs without audio hardware or
ORT. The filter's DSP response is covered by the core Rust tests; these cover
the binding surface only.

The high-pass selector is a closed numeric cutoff in Hz: a supported cutoff
turns it on, absence leaves it off, an out-of-set value is a clear error. The
value set (80, 100) is designed to grow (further cutoffs are additive).
"""

from __future__ import annotations

import pytest

from decibri import AsyncMicrophone, Microphone


def test_highpass_80_constructs() -> None:
    """The 80 Hz cutoff constructs; the filter is built later, at start()."""
    mic = Microphone(sample_rate=16000, channels=1, highpass=80)
    assert mic is not None


def test_highpass_100_constructs() -> None:
    """The 100 Hz cutoff constructs identically to the 80 Hz one."""
    mic = Microphone(sample_rate=16000, channels=1, highpass=100)
    assert mic is not None


def test_async_highpass_constructs() -> None:
    """AsyncMicrophone mirrors the sync surface and accepts both cutoffs."""
    assert AsyncMicrophone(sample_rate=16000, channels=1, highpass=80) is not None
    assert AsyncMicrophone(sample_rate=16000, channels=1, highpass=100) is not None


def test_highpass_off_by_default() -> None:
    """No highpass kwarg constructs identically to a plain microphone."""
    mic = Microphone(sample_rate=16000, channels=1)
    assert mic is not None


def test_highpass_string_form_rejected() -> None:
    """A string cutoff is a clear ValueError; only numeric cutoffs pass."""
    with pytest.raises(ValueError, match="highpass must be one of: 80, 100"):
        Microphone(highpass="80hz")  # type: ignore[arg-type]


def test_highpass_out_of_set_cutoff_rejected() -> None:
    """An out-of-set cutoff lists the allowed set, not a silent miss."""
    with pytest.raises(ValueError, match="highpass must be one of: 80, 100"):
        Microphone(highpass=200)  # type: ignore[arg-type]


def test_async_highpass_out_of_set_cutoff_rejected() -> None:
    with pytest.raises(ValueError, match="highpass must be one of: 80, 100"):
        AsyncMicrophone(highpass=50)  # type: ignore[arg-type]
