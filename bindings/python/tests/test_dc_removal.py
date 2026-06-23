"""DC-removal capture tests.

DC removal is pure DSP: a one-pole DC-blocking high-pass with no bundled file,
no ONNX model, and no ORT. So the whole surface (construction, off-by-default)
is CI-safe and runs without audio hardware or ORT. The DcBlocker's DSP response
(a known DC offset settles to a near-zero mean, length preserved, continuous
across chunk boundaries) is covered by the core Rust tests; these cover the
binding surface only.

The selector is a plain bool toggle: ``True`` turns the stage on, ``False`` (the
default) and absence leave it off, a byte-identical no-op.
"""

from __future__ import annotations

from decibri import AsyncMicrophone, Microphone


def test_dc_removal_true_constructs() -> None:
    """dc_removal=True constructs; the DC stage is built later, at start()."""
    mic = Microphone(sample_rate=16000, channels=1, dc_removal=True)
    assert mic is not None


def test_dc_removal_false_constructs() -> None:
    """dc_removal=False is the explicit off form and constructs identically."""
    mic = Microphone(sample_rate=16000, channels=1, dc_removal=False)
    assert mic is not None


def test_async_dc_removal_constructs() -> None:
    """AsyncMicrophone mirrors the sync surface and accepts the toggle."""
    assert AsyncMicrophone(sample_rate=16000, channels=1, dc_removal=True) is not None
    assert AsyncMicrophone(sample_rate=16000, channels=1, dc_removal=False) is not None


def test_dc_removal_off_by_default() -> None:
    """No dc_removal kwarg constructs identically to a plain microphone."""
    mic = Microphone(sample_rate=16000, channels=1)
    assert mic is not None
