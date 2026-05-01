"""Phase 2 error-message byte-identity tests (Q5 hybrid policy).

Hard-freeze tests: full message-text equality on InvalidArg-family messages
NOT already covered in test_config.py. Soft-freeze tests: substring assertions
on runtime/ORT messages where reachable.

All tests construct decibri public surface end-to-end. Direct exception
construction is not used here; the contract being frozen is the message a
user sees through the validation path.

# TODO: PermissionDenied byte-identity test deferred until Rust core wires
# up the variant (cpal BuildStreamError mapping). When that lands, add per
# Q5 hybrid soft-freeze policy.

# TODO: most ORT-family error messages (OrtInitFailed, OrtSessionBuildFailed,
# OrtThreadsConfigFailed, OrtInferenceFailed, OrtTensorCreateFailed,
# OrtTensorExtractFailed) require the ORT runtime to be in a failure state
# at test time. Not reachable in normal Phase 2 test flow. Deferred until
# Phase 3 / 4 when ORT-bundling and inference tests have a richer setup
# surface. Soft-freeze (per Q5) substring assertions for these will land
# alongside the test infrastructure that can produce them.
"""

import pytest

from decibri import (
    CaptureStreamClosed,
    Microphone,
    Speaker,
    InvalidFormat,
    OutputStreamClosed,
)


# ---------------------------------------------------------------------------
# Hard-freeze: dtype string rejection on the OUTPUT path. The capture path
# is covered by test_config.py::test_invalid_format_wrapper. Output mirrors
# the same wrapper-layer validation with the same message text.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_value,expected_msg",
    [
        pytest.param(
            "bogus",
            "dtype must be 'int16' or 'float32'; got 'bogus'",
            id="output_dtype_bogus_string",
        ),
        pytest.param(
            "INT16",
            "dtype must be 'int16' or 'float32'; got 'INT16'",
            id="output_dtype_wrong_case",
        ),
        pytest.param(
            "",
            "dtype must be 'int16' or 'float32'; got ''",
            id="output_dtype_empty",
        ),
    ],
)
def test_output_invalid_format_wrapper(dtype_value: str, expected_msg: str) -> None:
    """Speaker rejects invalid dtype strings with the same message text as Microphone."""
    with pytest.raises(InvalidFormat) as exc_info:
        Speaker(dtype=dtype_value)
    assert str(exc_info.value) == expected_msg


# ---------------------------------------------------------------------------
# Hard-freeze: closed-stream messages. Reachable without audio hardware:
# read() / write() / drain() before start() raises with the binding's
# raise_named() message text from bindings/python/src/lib.rs.
# ---------------------------------------------------------------------------


def test_capture_read_before_start_message() -> None:
    """Microphone.read() before start() raises CaptureStreamClosed with canonical text."""
    d = Microphone()
    with pytest.raises(CaptureStreamClosed) as exc_info:
        d.read(timeout_ms=100)
    assert str(exc_info.value) == "capture is not running"


def test_output_write_before_start_message() -> None:
    """Speaker.write() before start() raises OutputStreamClosed with canonical text."""
    o = Speaker()
    with pytest.raises(OutputStreamClosed) as exc_info:
        o.write(b"\x00" * 100)
    assert str(exc_info.value) == "output is not running"


def test_output_drain_before_start_message() -> None:
    """Speaker.drain() before start() raises OutputStreamClosed with canonical text."""
    o = Speaker()
    with pytest.raises(OutputStreamClosed) as exc_info:
        o.drain()
    assert str(exc_info.value) == "output is not running"


# ---------------------------------------------------------------------------
# Hard-freeze: VAD-family wrapper messages. The wrapper raises bare ValueError
# (not VadThresholdOutOfRange / VadSampleRateUnsupported) for these, because
# wrapper validation runs before any bridge interaction. The decibri-typed
# VAD exceptions are unreachable from the public Microphone surface; they would
# fire only if a consumer constructed _decibri.MicrophoneBridge directly.
# These tests verify the bare ValueError messages so consumers know what to
# expect from the public surface.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected_msg",
    [
        pytest.param(
            {"vad_threshold": -0.1},
            "vad_threshold must be in [0, 1]; got -0.1",
            id="vad_threshold_negative",
        ),
        pytest.param(
            {"vad_threshold": 1.5},
            "vad_threshold must be in [0, 1]; got 1.5",
            id="vad_threshold_above_one",
        ),
        pytest.param(
            {"vad": "bogus"},
            "Invalid vad value: 'bogus'. Expected False, 'silero', or 'energy'.",
            id="vad_invalid_string",
        ),
        pytest.param(
            {"vad": True},
            "vad=True is no longer supported. "
            "Specify the mode explicitly: vad='silero' or vad='energy'.",
            id="vad_true_rejected",
        ),
        pytest.param(
            {"vad_holdoff_ms": -1},
            "vad_holdoff_ms must be non-negative milliseconds; got -1",
            id="vad_holdoff_negative",
        ),
        pytest.param(
            {"vad_holdoff_ms": -100},
            "vad_holdoff_ms must be non-negative milliseconds; got -100",
            id="vad_holdoff_far_negative",
        ),
    ],
)
def test_vad_wrapper_value_errors(kwargs: dict, expected_msg: str) -> None:
    """Wrapper VAD validation raises bare ValueError with composed message text."""
    with pytest.raises(ValueError) as exc_info:
        Microphone(**kwargs)
    assert str(exc_info.value) == expected_msg
