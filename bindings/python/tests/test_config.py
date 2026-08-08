"""Configuration validation tests (hard-freeze byte-identity).

Tests Microphone constructor + start() argument validation with full message-text
equality per the hybrid policy (hard-freeze on InvalidArg-family). All tests
exercise the path a user actually hits, not direct exception instantiation.

Two validation layers, both at construction:

- Wrapper-layer validation in _classes.py (format string lookup): fires at
  Microphone.__init__ time. Message text is composed in Python with the
  offending value interpolated.
- Rust-core validation via config.validate() in the bridge constructor: fires
  at Microphone.__init__ time as well, BEFORE any cpal device interaction.
  Runs in CI without audio hardware. Message text from
  crates/decibri/src/error.rs Display impls (static strings; no value
  interpolation). Microphone.start() validates the same config again.

Notes on PyO3 boundary:
- sample_rate / frames_per_buffer are u32; channels is u16. Passing negative
  Python ints triggers OverflowError at the binding boundary BEFORE reaching
  Rust validation. These cases are not reachable via the Python surface and
  are not part of the test matrix.
- Positive out-of-range values pass through PyO3 cleanly and trigger
  validation at construction.

VAD-specific validations (the Vad object's threshold range and holdoff_ms,
the vad selector value) raise bare ValueError (not DecibriError subclasses)
and live in test_vad.py Section A alongside the rest of the VAD surface.
"""

import pytest

from decibri import (
    ChannelsOutOfRange,
    Microphone,
    FramesPerBufferOutOfRange,
    InvalidFormat,
    MultichannelNotSupported,
    SampleRateOutOfRange,
    Speaker,
)


# ---------------------------------------------------------------------------
# Wrapper-layer validation: dtype string lookup at Microphone.__init__ time.
# The wrapper composes its own f-string message including the offending value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_value,expected_msg",
    [
        pytest.param(
            "bogus",
            "dtype must be 'int16' or 'float32'; got 'bogus'",
            id="dtype_bogus_string",
        ),
        pytest.param(
            "INT16",
            "dtype must be 'int16' or 'float32'; got 'INT16'",
            id="dtype_wrong_case",
        ),
        pytest.param(
            "i16",
            "dtype must be 'int16' or 'float32'; got 'i16'",
            id="dtype_short_form",
        ),
        pytest.param(
            "f32",
            "dtype must be 'int16' or 'float32'; got 'f32'",
            id="dtype_short_float",
        ),
        pytest.param(
            "",
            "dtype must be 'int16' or 'float32'; got ''",
            id="dtype_empty",
        ),
    ],
)
def test_invalid_format_wrapper(dtype_value: str, expected_msg: str) -> None:
    """Microphone rejects invalid dtype strings at the wrapper layer (construction)."""
    with pytest.raises(InvalidFormat) as exc_info:
        Microphone(dtype=dtype_value)
    assert str(exc_info.value) == expected_msg


# ---------------------------------------------------------------------------
# Bridge-layer validation: sample_rate / channels / frames_per_buffer
# range checks fire at Microphone.__init__ via CaptureConfig::validate(). The
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
    """Out-of-range sample_rate raises the canonical Rust Display message at construction."""
    with pytest.raises(SampleRateOutOfRange) as exc_info:
        Microphone(sample_rate=sample_rate)
    assert str(exc_info.value) == "sample rate must be between 1000 and 384000"


def test_zero_channels_is_out_of_range() -> None:
    """A zero channel count raises the plain ChannelsOutOfRange at construction."""
    with pytest.raises(ChannelsOutOfRange) as exc_info:
        Microphone(channels=0)
    assert str(exc_info.value) == "channels must be at least 1"


@pytest.mark.parametrize(
    "channels",
    [
        pytest.param(2, id="channels_stereo"),
        pytest.param(33, id="channels_above_former_maximum"),
        pytest.param(100, id="channels_far_above"),
    ],
)
def test_multichannel_request_is_rejected(channels: int) -> None:
    """Capture is mono only: more than one channel raises MultichannelNotSupported
    at construction (rejected, not silently downmixed) with the canonical Rust message.
    """
    with pytest.raises(MultichannelNotSupported) as exc_info:
        Microphone(channels=channels)
    assert (
        str(exc_info.value)
        == "multichannel capture is not supported; channels must be 1 (mono)"
    )


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
    """Out-of-range frames_per_buffer raises the canonical Display message at construction."""
    with pytest.raises(FramesPerBufferOutOfRange) as exc_info:
        Microphone(frames_per_buffer=frames_per_buffer)
    assert str(exc_info.value) == "frames per buffer must be between 64 and 65536"


# ---------------------------------------------------------------------------
# Speaker: the same bridge-layer validation, at Speaker.__init__. SpeakerConfig
# validates sample_rate and channels only, and bounds channels below only:
# playback is not mono only, and how many channels can be carried is the
# device's answer, given when the stream opens rather than at construction.
# Hardware-free.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample_rate",
    [
        pytest.param(0, id="speaker_sample_rate_zero"),
        pytest.param(999, id="speaker_sample_rate_below_minimum"),
        pytest.param(384001, id="speaker_sample_rate_above_maximum"),
    ],
)
def test_speaker_invalid_sample_rate(sample_rate: int) -> None:
    """Out-of-range sample_rate raises at Speaker construction, not at start()."""
    with pytest.raises(SampleRateOutOfRange) as exc_info:
        Speaker(sample_rate=sample_rate)
    assert str(exc_info.value) == "sample rate must be between 1000 and 384000"


def test_speaker_zero_channels_is_out_of_range() -> None:
    """A zero channel count raises at Speaker construction, not at start()."""
    with pytest.raises(ChannelsOutOfRange) as exc_info:
        Speaker(channels=0)
    assert str(exc_info.value) == "channels must be at least 1"


@pytest.mark.parametrize(
    "channels",
    [
        pytest.param(33, id="speaker_channels_above_former_maximum"),
        pytest.param(64, id="speaker_channels_far_above"),
        pytest.param(1024, id="speaker_channels_beyond_any_device"),
    ],
)
def test_speaker_high_channel_counts_construct(channels: int) -> None:
    """Channels is bounded below only: a high count constructs and is left for
    the device to answer at start(), rather than being refused here.

    Regression: an upper bound reintroduced at the construction layer refuses
    counts the device would serve.
    """
    Speaker(channels=channels)


def test_speaker_stereo_constructs() -> None:
    """Playback is not mono only: a stereo speaker still constructs."""
    Speaker(sample_rate=24000, channels=2)


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
        pytest.param({"channels": 1}, id="channels_mono"),
        pytest.param({"frames_per_buffer": 64}, id="fpb_minimum"),
        pytest.param({"frames_per_buffer": 65536}, id="fpb_maximum"),
    ],
)
def test_boundary_values_construct_cleanly(kwargs: dict) -> None:
    """Values exactly at min/max boundaries construct without raising."""
    Microphone(**kwargs)


# ---------------------------------------------------------------------------
# The positive path: validating at construction does not obstruct a valid
# configuration, which still constructs AND starts. Hardware-marked.
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
def test_valid_config_still_starts() -> None:
    """A valid capture config constructs and still opens a device at start()."""
    d = Microphone(sample_rate=16000, channels=1, frames_per_buffer=512)
    d.start()
    try:
        assert d.is_open is True
    finally:
        d.stop()


@pytest.mark.requires_audio_output
def test_valid_speaker_config_still_starts() -> None:
    """A valid playback config constructs and still opens a device at start()."""
    o = Speaker(sample_rate=16000, channels=1)
    o.start()
    try:
        assert o.is_playing is True
    finally:
        o.stop()
