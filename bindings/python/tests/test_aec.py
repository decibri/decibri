"""Echo-cancellation capture tests.

The echo canceller is pure DSP: no bundled file, no ONNX model, and no ORT.
So the whole option surface (construction in both forms, rejection of every
invalid value, the reference-push input contract, and the metrics-off path)
is CI-safe and runs without audio hardware. The canceller's cancellation
behaviour is covered by the core Rust tests; these cover the binding surface.

The model set is owned by the canceller (``AecModel::from_str`` in the
bridge), so the unknown-model tests assert the canceller's own message text:
a wrapper-side copy of the list would break them the day the set grows.
"""

from __future__ import annotations

import inspect

import pytest

from decibri import Aec, AecMetrics, AsyncMicrophone, Microphone, exceptions
from decibri import _decibri


def test_aec_short_form_constructs() -> None:
    """The "tau" shorthand constructs; the canceller is built at start().

    Catches the option failing to reach the bridge config.
    """
    mic = Microphone(sample_rate=16000, channels=1, aec="tau")
    assert mic is not None


def test_aec_object_form_constructs_at_boundaries() -> None:
    """Every tuning field constructs at both ends of its range.

    Catches an off-by-one in any of the three range checks, and a field
    dropped between the dataclass and the bridge.
    """
    low = Microphone(
        sample_rate=16000,
        channels=1,
        aec=Aec(model="tau", tail_ms=16, suppression="off", reference_sample_rate=1000),
    )
    assert low is not None
    high = Microphone(
        sample_rate=16000,
        channels=1,
        aec=Aec(
            model="tau",
            tail_ms=500,
            suppression="conservative",
            reference_sample_rate=384000,
        ),
    )
    assert high is not None


def test_async_aec_constructs() -> None:
    """AsyncMicrophone mirrors the sync surface: both forms construct.

    Catches the async constructor missing the option or its forward.
    """
    assert AsyncMicrophone(sample_rate=16000, channels=1, aec="tau") is not None
    assert (
        AsyncMicrophone(
            sample_rate=16000,
            channels=1,
            aec=Aec(model="tau", tail_ms=200, reference_sample_rate=24000),
        )
        is not None
    )


def test_aec_off_by_default() -> None:
    """No aec kwarg enables no stage: metrics read None and a push no-ops.

    Catches the option becoming implicitly on, and the push or the metrics
    accessor erroring on a capture that never asked for echo cancellation.
    """
    mic = Microphone(sample_rate=16000, channels=1)
    assert mic.aec_metrics() is None
    assert mic.push_aec_reference(b"\x00\x00" * 32) is None


def test_unknown_model_rejected_by_delegation() -> None:
    """An unknown model is rejected with the canceller's own message.

    The wrapper holds no model list; the bridge parses the name through
    ``AecModel::from_str`` and raises ``AecConfigInvalid``. Catches the model
    set being copied into the wrapper and drifting from the canceller's.
    """
    with pytest.raises(
        exceptions.AecConfigInvalid, match="model must be one of: 'tau'; got 'tao'"
    ):
        Microphone(aec="tao")


def test_async_unknown_model_rejected_by_delegation() -> None:
    """The async constructor delegates the model parse identically."""
    with pytest.raises(
        exceptions.AecConfigInvalid, match="model must be one of: 'tau'; got 'tao'"
    ):
        AsyncMicrophone(aec="tao")


def test_object_form_unknown_model_rejected_by_delegation() -> None:
    """The Aec dataclass leaves the model set to the canceller.

    ``Aec(model="speex")`` constructs (the set is open at the dataclass), and
    the rejection lands at Microphone construction with the canceller's
    message. Catches the dataclass growing its own copy of the list.
    """
    config = Aec(model="speex")
    with pytest.raises(
        exceptions.AecConfigInvalid, match="model must be one of: 'tau'; got 'speex'"
    ):
        Microphone(aec=config)


def test_aec_config_invalid_is_catchable_as_decibri_error() -> None:
    """AecConfigInvalid derives from DecibriError, the catch-anything root."""
    with pytest.raises(exceptions.DecibriError):
        Microphone(aec="tao")


def test_tail_out_of_range_rejected_by_the_dataclass() -> None:
    """tail_ms outside 16..500 raises AecConfigInvalid, below and above.

    The boundary values construct. Catches the range check missing from the
    dataclass or drifting from the canceller's own window.
    """
    with pytest.raises(exceptions.AecConfigInvalid, match="tail_ms must be in"):
        Aec(tail_ms=15)
    with pytest.raises(exceptions.AecConfigInvalid, match="tail_ms must be in"):
        Aec(tail_ms=501)
    assert Aec(tail_ms=16) is not None
    assert Aec(tail_ms=500) is not None


def test_suppression_out_of_set_rejected_by_the_dataclass() -> None:
    """A suppression value outside the two-policy set is a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid aec suppression"):
        Aec(suppression="aggressive")
    assert Aec(suppression="conservative") is not None
    assert Aec(suppression="off") is not None


def test_reference_rate_out_of_range_rejected_by_the_dataclass() -> None:
    """reference_sample_rate outside 1000..384000 raises SampleRateOutOfRange.

    The boundary values construct. Catches the range check missing or
    narrowing away the rates decibri itself resamples from.
    """
    with pytest.raises(
        exceptions.SampleRateOutOfRange, match="reference_sample_rate must be in"
    ):
        Aec(reference_sample_rate=999)
    with pytest.raises(
        exceptions.SampleRateOutOfRange, match="reference_sample_rate must be in"
    ):
        Aec(reference_sample_rate=384001)
    assert Aec(reference_sample_rate=1000) is not None
    assert Aec(reference_sample_rate=384000) is not None


def test_non_string_model_rejected_by_the_dataclass() -> None:
    """A non-string model is a clear ValueError before any bridge work."""
    with pytest.raises(ValueError, match="Invalid aec model"):
        Aec(model=42)  # type: ignore[arg-type]


def test_invalid_aec_value_rejected() -> None:
    """A value that is neither None, a name, nor an Aec is a clear ValueError.

    Catches a bool or a stray number silently reading as enabled.
    """
    with pytest.raises(ValueError, match="Invalid aec value"):
        Microphone(aec=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid aec value"):
        Microphone(aec=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid aec value"):
        AsyncMicrophone(aec=True)  # type: ignore[arg-type]


def test_aec_narrows_capture_rate_window() -> None:
    """AEC narrows the accepted capture rate to the canceller's own window.

    A rate fine without AEC is rejected with it on, and the window's
    boundary rates construct. Catches the core's window check disappearing
    from the construction path.
    """
    with pytest.raises(
        exceptions.AecSampleRateUnsupported, match="echo cancellation only supports"
    ):
        Microphone(sample_rate=96000, aec="tau")
    with pytest.raises(
        exceptions.AecSampleRateUnsupported, match="echo cancellation only supports"
    ):
        AsyncMicrophone(sample_rate=96000, aec="tau")
    assert Microphone(sample_rate=96000, channels=1) is not None
    assert Microphone(sample_rate=8000, channels=1, aec="tau") is not None
    assert Microphone(sample_rate=48000, channels=1, aec="tau") is not None


def test_bridge_backstop_rejects_the_same_values() -> None:
    """The pyo3 layer backstops the checks the wrapper performs.

    A direct bridge consumer bypassing the wrapper and the dataclass gets the
    same classes: AecConfigInvalid for an out-of-range tail (with the
    canceller's own message text) and for an unknown model, ValueError for an
    out-of-set suppression. Catches the bridge-level block drifting from the
    wrapper's.
    """
    with pytest.raises(
        exceptions.AecConfigInvalid,
        match="filter tail must be between 16 and 500 milliseconds",
    ):
        _decibri.MicrophoneBridge(
            sample_rate=16000,
            channels=1,
            frames_per_buffer=1600,
            format="int16",
            aec="tau",
            aec_tail_ms=15,
        )
    with pytest.raises(
        ValueError, match="aec suppression must be 'conservative' or 'off'"
    ):
        _decibri.MicrophoneBridge(
            sample_rate=16000,
            channels=1,
            frames_per_buffer=1600,
            format="int16",
            aec="tau",
            aec_suppression="aggressive",
        )
    with pytest.raises(
        exceptions.AecConfigInvalid, match="model must be one of: 'tau'"
    ):
        _decibri.MicrophoneBridge(
            sample_rate=16000,
            channels=1,
            frames_per_buffer=1600,
            format="int16",
            aec="tao",
        )


def test_push_accepts_every_speaker_write_shape() -> None:
    """The push accepts the same input shapes ``Speaker.write`` accepts.

    bytes in the wire format, a 1-D ndarray, and a one-column 2-D ndarray,
    each dtype-matched to the microphone's own; and an empty push is fine.
    Catches the push narrowing the accepted shapes below the write surface.
    """
    np = pytest.importorskip("numpy")
    mic = Microphone(sample_rate=16000, channels=1, aec="tau")
    assert mic.push_aec_reference(b"\x00\x00" * 320) is None
    assert mic.push_aec_reference(b"") is None
    assert mic.push_aec_reference(np.zeros(320, dtype=np.int16)) is None
    assert mic.push_aec_reference(np.zeros((320, 1), dtype=np.int16)) is None
    f32_mic = Microphone(sample_rate=16000, channels=1, dtype="float32", aec="tau")
    assert f32_mic.push_aec_reference(np.zeros(320, dtype=np.float32)) is None
    assert f32_mic.push_aec_reference(b"\x00\x00\x00\x00" * 320) is None


def test_push_never_raises_without_a_stream() -> None:
    """A valid push is a no-op, never an error, whatever the capture state.

    Before start(), after stop(), and with the aec parameter unset, the push
    returns silently: the reference may legitimately flow while capture is
    down. Catches the push becoming stateful.
    """
    mic = Microphone(sample_rate=16000, channels=1, aec="tau")
    assert mic.push_aec_reference(b"\x00\x00" * 320) is None
    mic.stop()
    assert mic.push_aec_reference(b"\x00\x00" * 320) is None
    plain = Microphone(sample_rate=16000, channels=1)
    assert plain.push_aec_reference(b"\x00\x00" * 320) is None


def test_push_rejects_wrong_types_regardless_of_state() -> None:
    """A bad input type raises TypeError even while capture is not running.

    A wrong call site fails loud instead of silently pushing nothing, and a
    dtype-mismatched ndarray names both dtypes. Catches the input validation
    hiding behind the stream check.
    """
    np = pytest.importorskip("numpy")
    mic = Microphone(sample_rate=16000, channels=1, aec="tau")
    with pytest.raises(TypeError, match="samples must be bytes or numpy.ndarray"):
        mic.push_aec_reference(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="samples must be bytes or numpy.ndarray"):
        mic.push_aec_reference("not audio")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="format='int16' configured but 1-D ndarray"):
        mic.push_aec_reference(np.zeros(320, dtype=np.float32))


def test_metrics_none_before_start() -> None:
    """With aec set but capture not running, metrics read None.

    A dead stream cannot be mistaken for a quiet one. Catches the accessor
    fabricating a zeroed report without a stream.
    """
    mic = Microphone(sample_rate=16000, channels=1, aec="tau")
    assert mic.aec_metrics() is None


def test_async_push_is_a_plain_method() -> None:
    """The async surface's push is deliberately NOT a coroutine.

    The push never blocks, so a renderer callback calls it without awaiting;
    making it awaitable would break that contract. Catches the method being
    converted to a coroutine in a refactor.
    """
    assert not inspect.iscoroutinefunction(AsyncMicrophone.push_aec_reference)
    mic = AsyncMicrophone(sample_rate=16000, channels=1, aec="tau")
    assert mic.push_aec_reference(b"\x00\x00" * 320) is None


@pytest.mark.asyncio
async def test_async_aec_metrics_awaitable() -> None:
    """The async metrics accessor is a coroutine and reads None off-stream.

    Catches the accessor losing its awaitable form or erroring with no
    stream.
    """
    assert inspect.iscoroutinefunction(AsyncMicrophone.aec_metrics)
    mic = AsyncMicrophone(sample_rate=16000, channels=1, aec="tau")
    assert await mic.aec_metrics() is None
    plain = AsyncMicrophone(sample_rate=16000, channels=1)
    assert await plain.aec_metrics() is None


@pytest.mark.requires_audio_input
def test_live_metrics_report_every_field_and_the_search_signature() -> None:
    """A live echo-cancelling capture reports every metrics field.

    The reference is a tone the microphone is not hearing, so the estimator
    searches for the whole stream: delay_samples stays None while
    acquisition_parked climbs, the documented no-usable-reference signature,
    reachable through the accessor. In-cadence pushes drop nothing. After
    stop() the metrics read None again. Catches a field lost between the
    canceller, the queue counters, and the dataclass.
    """
    import math

    block = b"".join(
        int(math.sin(2 * math.pi * 440 * i / 16000) * 8000).to_bytes(
            2, "little", signed=True
        )
        for i in range(320)
    )
    mic = Microphone(sample_rate=16000, channels=1, frames_per_buffer=1600, aec="tau")
    mic.start()
    try:
        for _ in range(10):
            mic.push_aec_reference(block)
            chunk = mic.read(timeout_ms=2000)
            assert chunk is not None
        metrics = mic.aec_metrics()
        assert isinstance(metrics, AecMetrics)
        assert metrics.delay_samples is None
        assert isinstance(metrics.erle_db, float)
        assert isinstance(metrics.double_talk, bool)
        assert isinstance(metrics.reference_starved, int)
        assert metrics.acquisition_parked > 0
        assert isinstance(metrics.reference_reanchors, int)
        assert metrics.reference_dropped == 0
        assert isinstance(metrics.reference_silence, int)
    finally:
        mic.stop()
    assert mic.aec_metrics() is None


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_live_metrics_report_the_same_surface() -> None:
    """The async surface reaches the same live metrics.

    Catches the async bridge's accessor diverging from the sync one.
    """
    mic = AsyncMicrophone(
        sample_rate=16000, channels=1, frames_per_buffer=1600, aec="tau"
    )
    async with mic:
        for _ in range(5):
            mic.push_aec_reference(b"\x00\x10" * 320)
            chunk = await mic.read(timeout_ms=2000)
            assert chunk is not None
        metrics = await mic.aec_metrics()
        assert isinstance(metrics, AecMetrics)
        assert metrics.delay_samples is None
        assert metrics.acquisition_parked > 0
    assert await mic.aec_metrics() is None
