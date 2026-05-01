"""Phase 2 configuration validation tests (hard-freeze byte-identity).

Tests Microphone constructor + start() argument validation with full message-text
equality per Q5 hybrid policy (hard-freeze on InvalidArg-family). All tests
exercise the path a user actually hits, not direct exception instantiation.

Two validation layers:

- Wrapper-layer validation in _classes.py (format string lookup): fires at
  Microphone.__init__ time. Message text is composed in Python with the
  offending value interpolated.
- Rust-core validation in AudioCapture::new via config.validate(): fires at
  Microphone.start() time, BEFORE any cpal device interaction. Runs in CI
  without audio hardware. Message text from crates/decibri/src/error.rs
  Display impls (static strings; no value interpolation).

Notes on PyO3 boundary:
- sample_rate / frames_per_buffer are u32; channels is u16. Passing negative
  Python ints triggers OverflowError at the binding boundary BEFORE reaching
  Rust validation. These cases are not reachable via the Python surface and
  are not part of the test matrix.
- Positive out-of-range values pass through PyO3 cleanly and trigger
  validation at .start() time.

VAD-specific validations (vad_threshold range, vad_mode string, vad_holdoff
negative) raise bare ValueError from the wrapper (not DecibriError subclasses)
and live in test_vad.py Section A alongside the rest of the VAD surface.
"""

import pytest

from decibri import (
    ChannelsOutOfRange,
    Microphone,
    FramesPerBufferOutOfRange,
    InvalidFormat,
    SampleRateOutOfRange,
)


# ---------------------------------------------------------------------------
# Wrapper-layer validation: format string lookup at Microphone.__init__ time.
# The wrapper composes its own f-string message including the offending value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "format_value,expected_msg",
    [
        pytest.param(
            "bogus",
            "format must be 'int16' or 'float32'; got 'bogus'",
            id="format_bogus_string",
        ),
        pytest.param(
            "INT16",
            "format must be 'int16' or 'float32'; got 'INT16'",
            id="format_wrong_case",
        ),
        pytest.param(
            "i16",
            "format must be 'int16' or 'float32'; got 'i16'",
            id="format_short_form",
        ),
        pytest.param(
            "f32",
            "format must be 'int16' or 'float32'; got 'f32'",
            id="format_short_float",
        ),
        pytest.param(
            "",
            "format must be 'int16' or 'float32'; got ''",
            id="format_empty",
        ),
    ],
)
def test_invalid_format_wrapper(format_value: str, expected_msg: str) -> None:
    """Microphone rejects invalid format strings at the wrapper layer (construction)."""
    with pytest.raises(InvalidFormat) as exc_info:
        Microphone(format=format_value)
    assert str(exc_info.value) == expected_msg


# ---------------------------------------------------------------------------
# Bridge-layer validation: sample_rate / channels / frames_per_buffer
# range checks fire at Microphone.start() via CaptureConfig::validate(). The
# validate() call runs BEFORE cpal device interaction, so the test does NOT
# need audio hardware. Messages from error.rs Display impls.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_rate",
    [
        pytest.param(0, id="sample_rate_zero"),
        pytest.param(999, id="sample_rate_below_minimum"),
        pytest.param(384001, id="sample_rate_above_maximum"),
        pytest.param(1_000_000, id="sample_rate_far_above"),
    ],
)
def test_invalid_sample_rate(sample_rate: int) -> None:
    """Out-of-range sample_rate raises with the canonical Rust Display message at start()."""
    d = Microphone(sample_rate=sample_rate)
    with pytest.raises(SampleRateOutOfRange) as exc_info:
        d.start()
    assert str(exc_info.value) == "sample rate must be between 1000 and 384000"


@pytest.mark.parametrize(
    "channels",
    [
        pytest.param(0, id="channels_zero"),
        pytest.param(33, id="channels_above_maximum"),
        pytest.param(100, id="channels_far_above"),
    ],
)
def test_invalid_channels(channels: int) -> None:
    """Out-of-range channels raises with the canonical Rust Display message at start()."""
    d = Microphone(channels=channels)
    with pytest.raises(ChannelsOutOfRange) as exc_info:
        d.start()
    assert str(exc_info.value) == "channels must be between 1 and 32"


@pytest.mark.parametrize(
    "frames_per_buffer",
    [
        pytest.param(0, id="fpb_zero"),
        pytest.param(63, id="fpb_below_minimum"),
        pytest.param(65537, id="fpb_above_maximum"),
        pytest.param(1_000_000, id="fpb_far_above"),
    ],
)
def test_invalid_frames_per_buffer(frames_per_buffer: int) -> None:
    """Out-of-range frames_per_buffer raises with the canonical Display message at start()."""
    d = Microphone(frames_per_buffer=frames_per_buffer)
    with pytest.raises(FramesPerBufferOutOfRange) as exc_info:
        d.start()
    assert str(exc_info.value) == "frames per buffer must be between 64 and 65536"


# ---------------------------------------------------------------------------
# Boundary-acceptance tests at the construction layer: values right at the
# edge of accepted ranges construct without raising. Does NOT call start()
# (start would touch real hardware at boundary values, which is hardware-
# marker territory).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"sample_rate": 1000}, id="sample_rate_minimum"),
        pytest.param({"sample_rate": 384000}, id="sample_rate_maximum"),
        pytest.param({"channels": 1}, id="channels_minimum"),
        pytest.param({"channels": 32}, id="channels_maximum"),
        pytest.param({"frames_per_buffer": 64}, id="fpb_minimum"),
        pytest.param({"frames_per_buffer": 65536}, id="fpb_maximum"),
    ],
)
def test_boundary_values_construct_cleanly(kwargs: dict) -> None:
    """Values exactly at min/max boundaries construct without raising."""
    Microphone(**kwargs)
