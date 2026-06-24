"""Property-based tests for the Microphone constructor.

Hypothesis-driven coverage for Microphone constructor argument validation.
The existing test_config.py uses pytest.parametrize for hand-curated cases
(boundary values, common typos, bogus strings); this file extends coverage
by generating random inputs against typed strategies and asserting that
validation always lands on a typed decibri exception, never a panic / abort
/ untyped exception.

Scope: this is the bounded, hardware-free property suite. Broader VAD and
format-conversion property tests that need synthetic-waveform fixtures are
out of scope here and covered separately.

Validation layers (mirrors test_config.py docstring):

- Wrapper-layer validation in _classes.py: dtype string lookup at
  Microphone.__init__ time; raises InvalidFormat with an interpolated
  Python f-string message.
- Bridge-layer validation in AudioCapture::new via CaptureConfig::validate:
  fires at Microphone.start() time, BEFORE any cpal device interaction.
  Runs in CI without audio hardware. Messages from error.rs Display impls
  (static strings; no value interpolation).

Hypothesis settings: max_examples=20, deadline=None per test. Total file
runtime is bounded under 30 seconds on a typical CI runner; combined with
test_lifecycle_leak.py the two files stay comfortably within the suite's CI
time budget.

PyO3 boundary note (carried from test_config.py): sample_rate /
frames_per_buffer are u32; channels is u16. Negative Python ints would
trigger OverflowError at the binding boundary, which is a separate failure
mode from "Microphone rejected the value via its own validation". The
strategies below are non-negative-only so the property under test
("validation routes invalid values through typed decibri exceptions") is
exercised cleanly.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from decibri import (
    ChannelsOutOfRange,
    FramesPerBufferOutOfRange,
    InvalidFormat,
    Microphone,
    MultichannelNotSupported,
    SampleRateOutOfRange,
)


# ---------------------------------------------------------------------------
# Strategies for invalid inputs. Each strategy excludes the valid range so
# the property under test (rejection with a typed exception) cannot be
# falsified by a generated example that happens to be valid.
# ---------------------------------------------------------------------------


# Valid sample_rate range is [1000, 384000] per CaptureConfig::validate.
# Invalid: below 1000 or above 384000. Upper bound 10_000_000 stays well
# inside u32 to avoid OverflowError noise from the PyO3 boundary; the
# validation error is the property under test, not the binding boundary.
_invalid_sample_rate = st.one_of(
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=384_001, max_value=10_000_000),
)

# Capture is mono only: the sole valid channel count is 1. A zero channel
# count is a plain range error (ChannelsOutOfRange); any value above 1 is a
# multichannel request (MultichannelNotSupported). Upper bound 65000 stays
# inside u16.
_multichannel_channels = st.integers(min_value=2, max_value=65_000)

# Valid frames_per_buffer range is [64, 65536]. Invalid: below 64 or above
# 65536. Same u32 headroom rationale as sample_rate.
_invalid_frames_per_buffer = st.one_of(
    st.integers(min_value=0, max_value=63),
    st.integers(min_value=65_537, max_value=10_000_000),
)

# Valid dtype strings are exactly {"int16", "float32"}. Invalid: any other
# string. Strategy filters Hypothesis's text generator; max_size keeps
# generated rejects from blowing up the assertion message size.
_invalid_dtype = st.text(min_size=0, max_size=32).filter(
    lambda s: s not in ("int16", "float32")
)


# ---------------------------------------------------------------------------
# Section A: invalid inputs route through typed decibri exceptions.
#
# Validation point varies by argument:
#   dtype             -> __init__ time (wrapper layer)
#   sample_rate       -> .start() time (bridge layer)
#   channels          -> .start() time (bridge layer)
#   frames_per_buffer -> .start() time (bridge layer)
#
# All four exceptions are subclasses of DecibriError; passing the property
# is equivalent to "no panic / abort / untyped exception under fuzz".
# ---------------------------------------------------------------------------


@given(sample_rate=_invalid_sample_rate)
@settings(max_examples=20, deadline=None)
def test_invalid_sample_rate_property(sample_rate: int) -> None:
    """Any out-of-range sample_rate raises SampleRateOutOfRange at start()."""
    mic = Microphone(sample_rate=sample_rate)
    with pytest.raises(SampleRateOutOfRange):
        mic.start()


@given(channels=_multichannel_channels)
@settings(max_examples=20, deadline=None)
def test_multichannel_channels_property(channels: int) -> None:
    """Any channel count above 1 raises MultichannelNotSupported at start()
    (capture is mono only: the request is rejected, not silently downmixed).
    """
    mic = Microphone(channels=channels)
    with pytest.raises(MultichannelNotSupported):
        mic.start()


def test_zero_channels_is_out_of_range() -> None:
    """A zero channel count raises the plain ChannelsOutOfRange at start()."""
    mic = Microphone(channels=0)
    with pytest.raises(ChannelsOutOfRange):
        mic.start()


@given(frames_per_buffer=_invalid_frames_per_buffer)
@settings(max_examples=20, deadline=None)
def test_invalid_frames_per_buffer_property(frames_per_buffer: int) -> None:
    """Any out-of-range frames_per_buffer raises FramesPerBufferOutOfRange at start()."""
    mic = Microphone(frames_per_buffer=frames_per_buffer)
    with pytest.raises(FramesPerBufferOutOfRange):
        mic.start()


@given(dtype=_invalid_dtype)
@settings(max_examples=20, deadline=None)
def test_invalid_dtype_property(dtype: str) -> None:
    """Any non-canonical dtype string raises InvalidFormat at __init__."""
    with pytest.raises(InvalidFormat):
        Microphone(dtype=dtype)


# ---------------------------------------------------------------------------
# Section B: valid inputs construct cleanly.
#
# Pure construction (no .start()) is hardware-free; this exercises the
# constructor's positive path under fuzz to catch regressions where a
# value in the documented valid range is incorrectly rejected.
# ---------------------------------------------------------------------------


@given(
    sample_rate=st.integers(min_value=1000, max_value=384_000),
    # Capture is mono only, so the only valid channel count is 1.
    channels=st.just(1),
    frames_per_buffer=st.integers(min_value=64, max_value=65_536),
    dtype=st.sampled_from(["int16", "float32"]),
)
@settings(max_examples=20, deadline=None)
def test_valid_constructor_inputs_construct_cleanly(
    sample_rate: int,
    channels: int,
    frames_per_buffer: int,
    dtype: str,
) -> None:
    """Valid combinations of constructor args produce a Microphone without raising."""
    mic = Microphone(
        sample_rate=sample_rate,
        channels=channels,
        frames_per_buffer=frames_per_buffer,
        dtype=dtype,
    )
    # Reaching this line is the assertion. mic.stop() is idempotent on a
    # never-started instance (test_lifecycle.py Section 4 pattern).
    mic.stop()
