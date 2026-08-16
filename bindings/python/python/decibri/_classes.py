"""High-level Python wrapper for decibri audio capture with VAD policy.

This module ships the consumer-facing `Microphone` class. The class wraps the
native `_decibri.MicrophoneBridge` pyclass and adds wrapper-layer VAD policy:
threshold application and the holdoff state machine, over the score the bridge
computes for both VAD modes.

Architectural notes:

- The Rust bridge runs the detector. It exposes the score (the Silero
  probability in silero mode, the energy RMS in energy mode) via
  `vad_probability`, computed on the signal before any opt-in enhancement
  step, and stores `vad_holdoff_ms` as inert state. The wrapper applies
  threshold and holdoff policy on top of that scalar, the same for both
  modes. This mirrors the Node binding's pattern.

- `is_speaking` reflects time-based holdoff semantics. Above-threshold
  chunks set the speaking state and clear any pending silence timer.
  Below-threshold chunks while speaking start a silence timer; the timer
  expires after `vad_holdoff_ms` elapsed real time, at which point
  `is_speaking` flips to False. Timer expiry is checked on every property
  access using `time.monotonic()`, so consumers who pause iteration still
  observe correct state when they next read the property.

- `vad_score` reads the bridge score for both modes: the Silero
  probability in silero mode, the energy RMS in energy mode, both computed
  natively on the pre-enhancement signal so enabling enhancement does not
  change detection. The bridge keeps the raw `vad_probability` name for
  cross-binding consistency; the wrapper exposes the mode-agnostic
  `vad_score` view.

- This module's classes are synchronous. The async equivalents live in
  `decibri._async_classes` (`AsyncMicrophone`, `AsyncSpeaker`).
"""

from __future__ import annotations

import importlib.resources
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterator, Literal, Union

from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np

    # read can return bytes (default) or ndarray (as_ndarray=True);
    # write can accept either. The runtime numpy dependency is optional
    # (pip install decibri[numpy]); using TYPE_CHECKING keeps the import
    # cost out of the default-install path while preserving mypy
    # narrowing for users who set as_ndarray=True.
    SampleData = Union[bytes, "np.ndarray[Any, Any]"]
else:
    SampleData = bytes

from decibri import _decibri, exceptions
from decibri._decibri import MicrophoneInfo, SpeakerInfo, VersionInfo

__all__ = [
    "Chunk",
    "Vad",
    "Aec",
    "AecMetrics",
    "AecChannelMetrics",
    "Microphone",
    "Speaker",
    "File",
    "VadReport",
    "VadWindow",
    "Segment",
    "MicrophoneInfo",
    "SpeakerInfo",
    "VersionInfo",
]


# ---------------------------------------------------------------------------
# Chunk: typed audio chunk with metadata.
#
# Frozen dataclass returned from read_with_metadata() / iter_with_metadata().
# read() keeps the naked-data return shape for backwards compatibility;
# Chunk is the additive richer surface.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """Audio chunk with metadata.

    Returned by ``Microphone.read_with_metadata()`` /
    ``AsyncMicrophone.read_with_metadata()`` and yielded by
    ``Microphone.iter_with_metadata()`` /
    ``AsyncMicrophone.aiter_with_metadata()``. Provides the audio data
    plus context the consumer would otherwise have to track manually.

    Attributes:
        data: The audio data, ``bytes`` (default) or ``np.ndarray``
            (when the Microphone was constructed with ``as_ndarray=True``).
        timestamp: ``time.monotonic()`` value at the chunk boundary,
            in seconds. Use for relative timing within a session.
        sequence: Monotonic chunk counter starting at 0 for the first
            chunk after ``start()``. Resets to 0 on each new ``start()``.
        is_speaking: Snapshot of VAD state when the chunk was read.
            Always ``False`` if VAD is disabled.
        vad_score: Snapshot of VAD score when the chunk was read in
            ``[0, 1]``. ``0.0`` if VAD is disabled.
    """

    data: SampleData
    timestamp: float
    sequence: int
    is_speaking: bool
    vad_score: float


# ---------------------------------------------------------------------------
# Vad: named voice-activity-detection config object.
#
# Bundles the VAD model selector with its threshold and holdoff policy, the
# named-config-object shape a multi-parameter capability uses. Pass it as
# ``Microphone(vad=Vad(model="silero", threshold=0.5, holdoff_ms=300))``; the
# bare ``Microphone(vad="silero")`` shorthand keeps the disclosed defaults. The
# wrapper decomposes it into the (mode, threshold, holdoff) the detector policy
# consumes, exactly as the shorthand resolves to those same values.
# ---------------------------------------------------------------------------


# Shared silence-holdoff default (milliseconds). The shorthand and a Vad object
# with holdoff_ms left at its default both resolve to this value.
_DEFAULT_VAD_HOLDOFF_MS = 300


@dataclass(frozen=True, slots=True)
class Vad:
    """Voice-activity-detection configuration.

    Pass an instance on the ``vad`` parameter of ``Microphone`` /
    ``AsyncMicrophone`` to tune the detector's threshold and holdoff. The bare
    ``vad="silero"`` / ``vad="energy"`` shorthand selects a mode with the
    disclosed defaults; a ``Vad`` object is the way to override them.

    Attributes:
        model: Which detector to run. ``"silero"`` (the bundled Silero VAD
            ONNX model) or ``"energy"`` (an RMS energy threshold, no model
            file). The accepted set is kept open so a future model is an
            additive choice.
        threshold: Speech-detection threshold in ``[0, 1]`` applied to the
            detector score. ``None`` (the default) takes the mode-dependent
            default: 0.5 for ``"silero"``, 0.01 for ``"energy"``.
        holdoff_ms: Milliseconds of sub-threshold audio tolerated before the
            speaking state clears (the silence holdoff). Default 300.
        source: The 0-based DELIVERED channel the detector reads, the
            position within the delivered interleaved frames after any
            ``channel_map``. ``None`` (the default) feeds the detector the
            frame average of every delivered channel. Must be below the
            configured channel count (``DetectorSourceOutOfRange`` at
            construction otherwise); the count is the only ceiling, so no
            fixed maximum exists. Affects only the detector feed; the
            delivered audio is untouched.

    Example:
        Microphone(vad=Vad(model="silero", threshold=0.6, holdoff_ms=200))
    """

    model: str = "silero"
    threshold: float | None = None
    holdoff_ms: int = _DEFAULT_VAD_HOLDOFF_MS
    source: int | None = None

    def __post_init__(self) -> None:
        if self.model not in _VALID_MODES:
            raise ValueError(
                f"Invalid vad model: {self.model!r}. Expected 'silero' or 'energy'."
            )
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; got {self.threshold}")
        if self.holdoff_ms < 0:
            raise ValueError(
                f"holdoff_ms must be non-negative milliseconds; got {self.holdoff_ms}"
            )
        # Shape checks only: bool is excluded because it passes an int check
        # while meaning a toggle, and the u16 crossing bound is the same one
        # channel_map entries observe. The check against the configured
        # channel count is the core's own, made where both are in scope.
        if self.source is not None:
            if isinstance(self.source, bool) or not isinstance(self.source, int):
                raise TypeError(f"source must be an integer; got {self.source!r}")
            if not 0 <= self.source <= 65535:
                raise ValueError(
                    f"source must be in [0, 65535]; got {self.source}"
                )


# ---------------------------------------------------------------------------
# Aec: named echo-cancellation config object.
#
# Bundles the echo canceller model selector with its tuning fields, the
# named-config-object shape a multi-parameter capability uses, beside ``Vad``.
# Pass it as ``Microphone(aec=Aec(model="tau", tail_ms=200))``; the bare
# ``Microphone(aec="tau")`` shorthand keeps the canceller's defaults. Unlike
# ``Vad``, the model set is NOT validated here: the canceller owns that set,
# so the bridge parses the name (AecModel::from_str) when the capture object
# is constructed and an unknown name raises ``AecConfigInvalid`` naming the
# accepted set.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Aec:
    """Acoustic echo cancellation configuration.

    Pass an instance on the ``aec`` parameter of ``Microphone`` /
    ``AsyncMicrophone`` to tune the canceller. The bare ``aec="tau"``
    shorthand selects the model with its defaults; an ``Aec`` object is the
    way to override them.

    The canceller's behaviour while a lost delay alignment is being
    reacquired is fixed: decibri applies the canceller's graded output
    transition and does not expose a setting for it.

    Attributes:
        model: Which echo canceller to run. Today the set is ``"tau"``, a
            classical adaptive canceller with no model file. The accepted
            set is owned by the canceller and checked when the capture
            object is constructed, so a future model is an additive choice;
            an unknown name raises ``AecConfigInvalid`` naming the accepted
            set.
        tail_ms: Adaptive filter tail length in milliseconds: how much echo
            delay spread the canceller can model. Range 16 to 500. ``None``
            (the default) takes the canceller's own default (200).
        suppression: Residual echo suppression policy. ``"conservative"``
            attenuates the residual echo the linear canceller leaves behind
            while keeping the near-end voice intact; ``"off"`` delivers the
            linear canceller output as-is. ``None`` (the default) takes the
            canceller's own default (``"conservative"``).
        reference_sample_rate: Sample rate in Hz of the far-end reference
            pushed through ``push_aec_reference``. When it names a rate
            other than the capture ``sample_rate``, decibri converts the
            reference before the canceller sees it: a reference at an
            undeclared different rate cancels nothing and reports no error,
            so the conversion is decibri's rather than the caller's. Range
            1000 to 384000. ``None`` (the default) means the reference is
            already at the capture rate.
        reference_channels: Number of channels in the far-end reference
            pushed through ``push_aec_reference``, frame-interleaved. When
            it names a count above 1, decibri averages each frame to one
            mono sample before the canceller sees it: a multichannel
            reference pushed without declaring the count cancels nothing
            and reports no error, so the collapse is decibri's rather than
            the caller's. At least 1; no upper bound. 1 (the default)
            means the reference is already mono.

            The declared count must match the buffer actually pushed. The
            reference arrives as flat samples whose true channel count is
            not recoverable from their length, so a mismatch is not
            detected and raises no error: the frames are misread, nothing
            is cancelled, and the observable signature is
            ``aec_metrics().delay_samples`` staying ``None`` while the
            canceller reports no fault.

            The canceller itself reads one mono reference. Against
            playback through more than one loudspeaker that is a
            cancellation ceiling: the echo reaching the microphone is the
            sum of different room responses driven by different signals,
            and a single-reference canceller models one response applied
            to their average, so a placement where those paths differ
            leaves a residual that no amount of adaptation removes.

    Example:
        Microphone(aec=Aec(model="tau", tail_ms=200, reference_sample_rate=24000))
    """

    model: str = "tau"
    tail_ms: int | None = None
    suppression: str | None = None
    reference_sample_rate: int | None = None
    reference_channels: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise ValueError(
                f"Invalid aec model: {self.model!r}. "
                "Expected a model name such as 'tau'."
            )
        if self.tail_ms is not None and not 16 <= self.tail_ms <= 500:
            raise exceptions.AecConfigInvalid(
                f"tail_ms must be in [16, 500]; got {self.tail_ms}"
            )
        if (
            self.suppression is not None
            and self.suppression not in _VALID_AEC_SUPPRESSION
        ):
            raise ValueError(
                f"Invalid aec suppression: {self.suppression!r}. "
                "Expected 'conservative' or 'off'."
            )
        if self.reference_sample_rate is not None and not (
            1000 <= self.reference_sample_rate <= 384000
        ):
            raise exceptions.SampleRateOutOfRange(
                f"reference_sample_rate must be in [1000, 384000]; "
                f"got {self.reference_sample_rate}"
            )
        if self.reference_channels < 1:
            raise exceptions.AecConfigInvalid(
                f"reference_channels must be at least 1; "
                f"got {self.reference_channels}"
            )


@dataclass(frozen=True, slots=True)
class AecChannelMetrics:
    """One delivered channel's canceller report, one entry of
    ``AecMetrics.channels``.

    Engine-level fields only: the reference queue's counters
    (``reference_dropped``, ``reference_silence``) describe the shared queue
    and stay on ``AecMetrics`` itself.

    Attributes:
        delay_samples: This channel's active delay alignment in samples, or
            ``None`` while its estimator is still searching. The offset from
            the reference frontier as the feeding established it, not a
            measurement of the room's echo path.
        erle_db: This channel's smoothed echo-return-loss-enhancement
            estimate in dB. Not a quality ranking across channels: ERLE
            rises with echo distance, because a weaker echo is easier to
            reduce in ratio terms, so a far microphone routinely reports a
            higher figure than a near one while removing less echo in
            absolute terms. Compare a channel against its own history, not
            against its neighbours.
        double_talk: Whether this channel's double-talk detector currently
            believes the near-end talker is active; its adaptation is held
            while ``True``.
        reference_starved: Near-end samples this channel's canceller could
            find no far-end sample for while an alignment was active.
        acquisition_parked: Near-end samples this channel processed while no
            delay alignment was active: the searching span, not a transport
            failure.
        reference_reanchors: Times this channel's canceller inferred a
            capture discontinuity and rebuilt its alignment from the
            reference frontier.
    """

    delay_samples: int | None
    erle_db: float
    double_talk: bool
    reference_starved: int
    acquisition_parked: int
    reference_reanchors: int


@dataclass(frozen=True, slots=True)
class AecMetrics:
    """The echo canceller's transport and cancellation metrics.

    Returned by ``Microphone.aec_metrics()`` /
    ``AsyncMicrophone.aec_metrics()``. One object carries the canceller's own
    report and the reference queue's counters, so a caller reads one surface
    for the whole diagnosis.

    ``delay_samples`` staying ``None`` while ``acquisition_parked`` climbs is
    the signature of a canceller with no usable reference: none pushed, not
    at the declared rate, or not the signal that produced the echo. A
    climbing ``reference_dropped`` means single pushes are exceeding the
    reference queue's bound.

    Attributes:
        delay_samples: The active delay alignment in samples, or ``None``
            while the estimator is still searching.
        erle_db: Smoothed echo-return-loss-enhancement estimate in dB: how
            much echo the canceller is currently removing. 0 before the
            filter has converged.
        double_talk: Whether the double-talk detector currently believes the
            near-end talker is active; adaptation is held while ``True``.
        reference_starved: Near-end samples the canceller could find no
            far-end sample for while an alignment was active. decibri keeps
            the far-end stream level with the capture, so this stays 0 for a
            caller who simply stops pushing; a non-zero count means the
            caller ran further ahead of the capture than the canceller's
            far-end history reaches.
        acquisition_parked: Near-end samples processed while no delay
            alignment was active: the searching span, not a transport
            failure.
        reference_reanchors: Times the canceller inferred a capture
            discontinuity and rebuilt its alignment from the reference
            frontier.
        reference_dropped: Far-end samples discarded because a single push
            exceeded the reference queue's bound, at the declared reference
            rate. The span they occupied is still represented as silence, so
            a discard costs the cancellation of that span alone.
        reference_silence: Far-end samples decibri supplied as silence
            because the caller had pushed none for them, at the capture
            rate. An accounting figure, not a fault: while nothing is
            playing, the far end is silence.
        channels: Every delivered channel's canceller report, in delivered
            order, one entry per channel. One canceller engine runs per
            delivered channel, each fed the same pushed reference and each
            finding its own channel's echo delay, so the entries differ
            where the channels' acoustic paths differ. The top-level engine
            fields report the first delivered channel's canceller, so on a
            single-channel stream this holds one entry agreeing with them.
    """

    delay_samples: int | None
    erle_db: float
    double_talk: bool
    reference_starved: int
    acquisition_parked: int
    reference_reanchors: int
    reference_dropped: int
    reference_silence: int
    channels: tuple[AecChannelMetrics, ...]


def _aec_metrics_from_raw(
    raw: tuple[
        int | None,
        float,
        bool,
        int,
        int,
        int,
        int,
        int,
        list[tuple[int | None, float, bool, int, int, int]],
    ],
) -> AecMetrics:
    """Build the public ``AecMetrics`` from the bridge's flat tuple, shared
    by the sync and async surfaces so the two field orders cannot drift."""
    return AecMetrics(
        delay_samples=raw[0],
        erle_db=raw[1],
        double_talk=raw[2],
        reference_starved=raw[3],
        acquisition_parked=raw[4],
        reference_reanchors=raw[5],
        reference_dropped=raw[6],
        reference_silence=raw[7],
        channels=tuple(
            AecChannelMetrics(
                delay_samples=entry[0],
                erle_db=entry[1],
                double_talk=entry[2],
                reference_starved=entry[3],
                acquisition_parked=entry[4],
                reference_reanchors=entry[5],
            )
            for entry in raw[8]
        ),
    )


# ---------------------------------------------------------------------------
# VAD policy state machine.
#
# Encapsulates the threshold + holdoff logic that the Rust bridge does NOT do.
# Two responsibilities, mode-agnostic (the bridge computes the score for both
# Silero and energy modes on the pre-enhancement signal and exposes it via
# `vad_probability`):
#
#   1. Apply the user's threshold to the bridge score.
#
#   2. Run a holdoff state machine on the threshold result, with time-based
#      expiry of the silence timer.
#
# Construction is driven by Microphone.__init__; consumers don't touch this
# class directly.
# ---------------------------------------------------------------------------


class _VadStateMachine:
    """Wrapper-layer VAD policy. Pure Python; no native interaction."""

    __slots__ = (
        "_threshold",
        "_holdoff_seconds",
        "_is_speaking",
        "_silence_started_at",
        "_last_probability",
    )

    def __init__(
        self,
        threshold: float,
        holdoff_ms: int,
    ) -> None:
        self._threshold = float(threshold)
        self._holdoff_seconds = holdoff_ms / 1000.0
        self._is_speaking = False
        self._silence_started_at: float | None = None
        self._last_probability = 0.0

    def reset(self) -> None:
        """Clear all transient state. Called on stop()."""
        self._is_speaking = False
        self._silence_started_at = None
        self._last_probability = 0.0

    def process_chunk(self, probability: float) -> None:
        """Update VAD state from one chunk's bridge score.

        Called inside Microphone.read() with the bridge's `vad_probability`:
        the Silero probability in silero mode or the energy RMS in energy mode,
        both computed natively on the pre-enhancement signal.
        """
        self._last_probability = probability

        above_threshold = probability >= self._threshold

        if above_threshold:
            # Above threshold: enter or stay in speaking state, cancel any
            # pending silence timer.
            self._is_speaking = True
            self._silence_started_at = None
        else:
            # Below threshold while speaking: start the silence timer if
            # not already running. The actual transition to is_speaking=False
            # happens lazily on property access via _refresh_speaking_state.
            if self._is_speaking and self._silence_started_at is None:
                self._silence_started_at = time.monotonic()
            # If not speaking, no state change; below-threshold while silent
            # is the steady state.

    def _refresh_speaking_state(self) -> None:
        """Lazy holdoff expiry. Called from is_speaking property getter."""
        if self._is_speaking and self._silence_started_at is not None:
            elapsed = time.monotonic() - self._silence_started_at
            if elapsed >= self._holdoff_seconds:
                self._is_speaking = False
                self._silence_started_at = None

    @property
    def is_speaking(self) -> bool:
        self._refresh_speaking_state()
        return self._is_speaking

    @property
    def vad_score(self) -> float:
        return self._last_probability


# ---------------------------------------------------------------------------
# Microphone: consumer-facing audio capture class.
# ---------------------------------------------------------------------------


_VALID_MODES = frozenset({"silero", "energy"})
_VALID_FORMATS = frozenset({"int16", "float32"})
_VALID_DENOISE_MODELS = frozenset({"fastenhancer-t"})
_VALID_HIGHPASS = frozenset({80, 100})
# The canceller's residual-suppression policies. A closed two-value set the
# bindings map to the canceller's enum; the MODEL set, by contrast, is owned by
# the canceller and never copied here.
_VALID_AEC_SUPPRESSION = frozenset({"conservative", "off"})

# Packaged locations of the bundled models, reported as the ``path`` attribute
# when the resource cannot be resolved and there is no filesystem path to name.
_BUNDLED_VAD_MODEL = "decibri/models/silero_vad.onnx"
_BUNDLED_DENOISE_MODEL = "decibri/models/fastenhancer_t.onnx"


def _require_numpy() -> None:
    """Raise ``ImportError`` when numpy is not installed.

    Called wherever ``as_ndarray=True`` is accepted, before the bridge is
    constructed, so a missing numpy surfaces as a normal, catchable
    ``ImportError`` instead of a failure from inside the extension. The
    message matches the read-path guard's message exactly.
    """
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "numpy is not installed. Install with: pip install decibri[numpy]"
        ) from exc


class Microphone:
    """Audio capture with VAD policy.

    Example:
        with Microphone(sample_rate=16000, channels=1, frames_per_buffer=1600) as d:
            for chunk in d:
                if d.is_speaking:
                    process(chunk)

    Construction does NOT start capture. Use the context manager or call
    `start()` explicitly. Iteration without an active capture raises
    `MicrophoneStreamClosed`.

    The VAD parameters (`vad`, `model_path`, `ort_library_path`) configure
    both the bridge and the wrapper-layer state machine. ``vad`` accepts
    ``False`` (disabled), the ``"silero"`` / ``"energy"`` shorthand (which
    uses the mode's default threshold and holdoff), or a ``Vad`` config
    object (``vad=Vad(model="silero", threshold=0.5, holdoff_ms=300)``) to
    tune the threshold and holdoff. With ``vad=False``, ``vad_score``
    returns 0.0 and ``is_speaking`` returns False unconditionally.

    The ``dc_removal`` parameter, when ``True``, removes a constant (DC) offset
    from the captured audio with a one-pole DC-blocking high-pass, applied first
    in the chain (before denoise); leave it ``False`` (the default) to keep the
    capture path byte-identical. It is same-length with no added latency, so
    ``vad_score`` and ``is_speaking`` are unaffected.

    The ``denoise`` parameter selects an optional bundled single-channel
    speech-enhancement model (``"fastenhancer-t"``) applied to the captured
    audio; omit it (the default ``None``) to leave denoise off, which keeps the
    capture path unchanged.

    The ``highpass`` parameter selects an optional high-pass filter cutoff in
    Hz (``80`` or ``100``, a second-order Butterworth) that removes
    low-frequency rumble below the voice band, applied after denoise; omit it
    (the default ``None``) to leave the high-pass off and the capture path
    full-range.

    The ``agc`` parameter sets an optional automatic gain control target level
    in dBFS (an integer in ``-40..-3``, typical ``-18``), driving the captured
    level toward the target with a smoothed, rate-limited gain applied after the
    high-pass step; omit it (the default ``None``) to leave the level untouched.

    The ``limiter`` parameter sets an optional sample-peak ceiling in dBFS (a
    float in ``-3.0..0.0``, typical ``-1.0``) that holds the captured signal at
    or below the ceiling, catching a peak the AGC would let through; it runs last
    in the chain, after the AGC step; omit it (the default ``None``) to leave the
    level untouched.

    The ``aec`` parameter enables acoustic echo cancellation on the captured
    audio, removing the echo of far-end audio pushed through
    ``push_aec_reference``; pass ``"tau"`` or an ``Aec`` config object. It
    runs before the detector tap, so with it on, ``vad_score`` and
    ``is_speaking`` read the echo-removed signal and playback stops
    triggering detection. Requires ``sample_rate`` in 8000 to 48000, narrower
    than the range the parameter otherwise accepts. With no reference pushed,
    the captured audio passes through unchanged; omit it (the default
    ``None``) to leave echo cancellation off.

    The ``channels`` parameter is the number of channels delivered,
    interleaved frame by frame. Bounded below at 1 (the default); bounded above
    by the resolved device alone, with no fixed maximum. Without a
    ``channel_map``, ``channels=1`` delivers the average of every device
    channel and a count equal to the device's own delivers them all in device
    order; a count above the device's raises
    ``MicrophoneChannelsUnsupported`` at ``start()`` and a strict subset above
    one raises ``ChannelSelectionAmbiguous``, because which channels it means
    has no single answer. Above one channel, ``read()`` block sizes count
    interleaved samples and must be a whole number of frames. Echo
    cancellation runs one canceller engine per delivered channel, so its
    processing and memory cost scale with the count.

    The ``channel_map`` parameter selects which device channels feed the
    delivered channels: a list of 0-based device channel indices, one entry per
    delivered channel, so its length must equal ``channels`` (for
    example ``channel_map=[1]`` delivers the device's second channel alone, and
    ``channel_map=[1, 0]`` delivers the first two swapped). Entries may repeat
    and may appear in any order, so a map both selects and permutes. Omit it
    (the default ``None``) to take the derivation ``channels`` documents. The
    same shape as CoreAudio AUHAL's channel map
    (``kAudioOutputUnitProperty_ChannelMap``: an array of device channel
    indices, one per client channel); NOT miniaudio's ``channelMap``, which
    names a spatial layout. Entries are checked against the device's own
    channel count when the stream starts (``ChannelMapOutOfRange`` names an
    entry the device does not have); the device's report is the only ceiling.

    This class is synchronous. For async iteration, use ``AsyncMicrophone``.

    Cleanup and disconnect:
        Mid-stream device disconnect (USB unplug, default-device switch,
        driver error) is surfaced as a ``DeviceFailed`` raised on the next
        ``read()``, carrying the driver's own cause. cpal detects the
        disconnect and closes the underlying stream within roughly 20ms; the
        wrapper then sees the closed state on its next read attempt and
        raises. ``DeviceFailed`` derives from ``DecibriError``, not from
        ``MicrophoneStreamClosed``, so catch ``DecibriError`` (or
        ``DeviceFailed`` itself) to handle a disconnect.

        A deliberate ``stop()`` or ``close()`` is never reported as a device
        failure: it raises ``MicrophoneStreamClosed`` as before.

        Threaded shutdown: calling ``stop()`` from a thread other than
        the one currently blocked inside ``read()`` is safe. It
        interrupts the parked read, which then raises
        ``MicrophoneStreamClosed``, and releases the device. The bridge
        holds the core stream behind a shared handle and exposes
        ``read()`` / ``stop()`` without an exclusive borrow, so the
        cross-thread ``stop()`` reaches the core stop (which wakes the
        parked read within roughly 20ms) instead of raising
        ``AlreadyBorrowed``.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 1600,
        dtype: str = "int16",
        device: int | str | None = None,
        vad: bool | str | Vad = False,
        model_path: str | Path | None = None,
        as_ndarray: bool = False,
        ort_library_path: str | Path | None = None,
        denoise: Literal["fastenhancer-t"] | None = None,
        highpass: Literal[80, 100] | None = None,
        agc: int | None = None,
        limiter: float | None = None,
        dc_removal: bool = False,
        aec: str | Aec | None = None,
        channel_map: list[int] | None = None,
    ) -> None:
        """Construct a Microphone audio capture instance.

        Most parameters control capture behaviour and are documented at
        the class level. The arguments that benefit from per-arg detail
        are ``sample_rate`` (cloud-STT compatibility) and
        ``ort_library_path`` (its resolution involves a priority chain
        that is invisible at the call site).

        Parameters
        ----------
        sample_rate : int, optional
            Sample rate in Hz. Default 16000 (the cloud-STT convention;
            matches Silero VAD's native rate).

            Note: OpenAI Realtime API requires 24000 Hz; most other
            cloud STT providers (Deepgram, AssemblyAI, Azure, Google,
            AWS Transcribe) prefer 16000 Hz. Silero VAD operates at
            16000 Hz natively. If using both Silero VAD and OpenAI
            Realtime in the same pipeline, capture at 16000 for VAD
            then resample to 24000 (e.g., via ``resampy`` or
            ``scipy.signal.resample_poly``) before sending to OpenAI.

        ort_library_path : str | Path | None, optional
            Path to the ONNX Runtime dynamic library used by Silero VAD.
            Only consulted when ``vad='silero'``; energy-mode VAD
            (``vad='energy'``) and ``vad=False`` do not load ORT.

            If ``None`` (the default), decibri resolves the dylib via
            this priority order, first match wins:

            1. This constructor argument, if you pass an explicit path.
            2. The ``DECIBRI_ORT_DYLIB_PATH`` environment variable,
               for per-deployment overrides without code changes.
            3. The ``ORT_DYLIB_PATH`` environment variable, the upstream
               ``ort`` crate's standard convention, respected here so
               existing bare-ort deployments keep working.
            4. The dylib bundled with the wheel under ``decibri/_ort/``,
               which is the default ``pip install decibri`` experience.

            If none of the above resolve to a real file, ORT's default
            loader is used. That loader itself respects ``ORT_DYLIB_PATH``
            if you set it after decibri import, so a late environment
            change still works as a last-resort fallback.

            Path validity is enforced by the Rust core, not the resolver.
            An invalid path (missing file, directory, etc.) raises
            ``OrtPathInvalid`` from the bridge layer with the path and
            reason attached.

            Note: ORT initialization is process-global. Once any Microphone
            instance constructs with a specific dylib path, subsequent
            Microphone instances inherit that initialization regardless of
            their own ``ort_library_path`` argument. The first Microphone
            construction in a process determines the dylib for the
            entire process lifetime.

        Other parameters are summarised at the class docstring; refer to
        ``help(Microphone)`` for the capture-side surface.
        """
        if dtype not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"dtype must be 'int16' or 'float32'; got {dtype!r}"
            )

        # ndarray output requires numpy at read time; checked here, before
        # the bridge is constructed, so a missing install raises a catchable
        # ImportError from the constructor.
        if as_ndarray:
            _require_numpy()

        # Decompose ``vad`` into the bridge's (enabled, mode) split plus the
        # threshold/holdoff policy values. ``vad`` accepts False (disabled;
        # default), the "silero"/"energy" shorthand (disclosed-default policy),
        # or a Vad config object (to tune threshold/holdoff). vad=True is
        # rejected explicitly so callers from the two-flag era get a
        # migration-friendly message rather than a silent semantic change.
        # A Vad object self-validates its model, threshold, and holdoff at
        # construction; the shorthand/False paths carry no user policy to check.
        vad_enabled: bool
        vad_mode: str
        vad_threshold: float | None
        vad_holdoff_ms: int
        vad_source: int | None
        if vad is False:
            vad_enabled = False
            vad_mode = "energy"  # inert placeholder; bridge ignores when disabled
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
            vad_source = None
        elif vad is True:
            raise ValueError(
                "vad=True is no longer supported. "
                "Specify the mode explicitly: vad='silero' or vad='energy'."
            )
        elif isinstance(vad, Vad):
            vad_enabled = True
            vad_mode = vad.model
            vad_threshold = vad.threshold
            vad_holdoff_ms = vad.holdoff_ms
            vad_source = vad.source
        elif isinstance(vad, str) and vad in _VALID_MODES:
            vad_enabled = True
            vad_mode = vad
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
            vad_source = None
        else:
            raise ValueError(
                f"Invalid vad value: {vad!r}. "
                "Expected False, 'silero', 'energy', or a Vad config object."
            )

        # Mode-dependent threshold default mirroring Node: 0.5 for silero,
        # 0.01 for energy. A Vad object that left threshold unset (None) takes
        # the same default as the shorthand does.
        if vad_threshold is None:
            vad_threshold = 0.5 if vad_mode == "silero" else 0.01

        # Resolve model_path to an absolute string for the bridge.
        # User-supplied path takes precedence. When Silero VAD is requested
        # without an explicit path, fall back to the model bundled with the
        # wheel via importlib.resources. Mirrors the Node binding's pattern
        # (decibri.js:159-171) which resolves __dirname/../models/silero_vad.onnx
        # before passing to the native bridge.
        resolved_model_path: str | None = None
        if model_path is not None:
            resolved_model_path = str(Path(model_path))
            # A user-supplied path is checked at construction, so a path that
            # does not exist is reported here rather than at the load. Checked
            # only when Silero VAD reads it; the bridge ignores it otherwise.
            if (
                vad_enabled
                and vad_mode == "silero"
                and not Path(resolved_model_path).is_file()
            ):
                raise exceptions.VadModelLoadFailed(
                    f"Silero VAD model not found at {resolved_model_path}. "
                    "Ensure model_path points to an existing ONNX model file.",
                    resolved_model_path,
                )
        elif vad_enabled and vad_mode == "silero":
            try:
                model_resource = (
                    importlib.resources.files("decibri")
                    / "models"
                    / "silero_vad.onnx"
                )
                # For editable installs and standard wheel installs, this is
                # a direct filesystem path. Zipped-wheel deployments would
                # require importlib.resources.as_file() context manager;
                # deferred until that case becomes relevant for decibri.
                if not model_resource.is_file():
                    raise FileNotFoundError(
                        f"Bundled Silero model resource exists but is not a "
                        f"file: {model_resource}"
                    )
                resolved_model_path = str(model_resource)
            except (FileNotFoundError, ModuleNotFoundError, AttributeError) as exc:
                raise exceptions.VadModelLoadFailed(
                    "model_path was not provided and the bundled Silero "
                    "model could not be located in the installed wheel. "
                    "Ensure the models/ directory was included during "
                    "installation, or pass model_path explicitly.",
                    _BUNDLED_VAD_MODEL,
                ) from exc

        # Validate and resolve denoise. Closed-set selector mirroring the Silero
        # VAD shape: a model name resolves to a bundled ONNX file, absence leaves
        # denoise off. The only accepted value is 'fastenhancer-t'; anything else
        # is a clear ValueError, not a silent miss. The bundled model file is
        # resolved via importlib.resources, exactly as the Silero model is.
        resolved_denoise_model_path: str | None = None
        if denoise is not None:
            if denoise not in _VALID_DENOISE_MODELS:
                raise ValueError(
                    f"Invalid denoise value: {denoise!r}. Expected 'fastenhancer-t'."
                )
            try:
                denoise_resource = (
                    importlib.resources.files("decibri")
                    / "models"
                    / "fastenhancer_t.onnx"
                )
                if not denoise_resource.is_file():
                    raise FileNotFoundError(
                        f"Bundled denoise model resource exists but is not a "
                        f"file: {denoise_resource}"
                    )
                resolved_denoise_model_path = str(denoise_resource)
            except (FileNotFoundError, ModuleNotFoundError, AttributeError) as exc:
                raise exceptions.ModelLoadFailed(
                    "the bundled denoise model could not be located in the "
                    "installed wheel. Ensure the models/ directory was included "
                    "during installation.",
                    _BUNDLED_DENOISE_MODEL,
                ) from exc

        # Validate high-pass. Closed growable numeric cutoff selector mirroring
        # the denoise shape: a cutoff in Hz selects a filter, absence leaves the
        # high-pass off, an out-of-set value is a clear ValueError. Pure DSP, so
        # there is no bundled file to resolve, only the closed-set check.
        if highpass is not None and highpass not in _VALID_HIGHPASS:
            raise ValueError(
                f"highpass must be one of: 80, 100; got {highpass!r}"
            )

        # Validate AGC. The target is an integer dBFS level in [-40, -3]
        # (typical -18); absence leaves it off. The core names this failure, so
        # the wrapper raises the core's class; the core guards the same range.
        if agc is not None and not -40 <= agc <= -3:
            raise exceptions.AgcTargetOutOfRange(f"agc must be in [-40, -3]; got {agc}")

        # Validate the limiter ceiling. A sample-peak ceiling in dBFS in
        # [-3.0, 0.0] (typical -1.0); absence leaves it off. Mirrors the agc
        # check: the core names this failure, so the wrapper raises its class.
        if limiter is not None and not -3.0 <= limiter <= 0.0:
            raise exceptions.LimiterCeilingOutOfRange(
                f"limiter must be in [-3.0, 0.0]; got {limiter}"
            )

        # Validate the channel map's shape: a list of integers in the channel
        # count's width, with one entry per delivered channel (the core names
        # the length failure, so the wrapper raises its class). Whether each
        # entry exists on the device is the core's check, made against the
        # resolved device's own report at start(), because only the device can
        # say how many channels it has; no fixed maximum exists on this path.
        if channel_map is not None:
            for entry in channel_map:
                if not isinstance(entry, int) or isinstance(entry, bool):
                    raise TypeError(
                        f"channel_map entries must be integers; got {entry!r}"
                    )
                if not 0 <= entry <= 65535:
                    raise ValueError(
                        f"channel_map entries must be in [0, 65535]; got {entry}"
                    )
            if len(channel_map) != channels:
                raise exceptions.ChannelMapLengthMismatch(
                    f"the channel map has {len(channel_map)} entries; it must "
                    f"have exactly one entry per delivered channel ({channels})"
                )

        # Decompose ``aec`` into the bridge's flat fields. ``aec`` accepts None
        # (off; default), a model-name shorthand such as "tau", or an Aec
        # config object (which self-validates its tuning fields at
        # construction). The model NAME is deliberately not checked against a
        # list here: the canceller owns that set, so the bridge parses it
        # (AecModel::from_str) below and an unknown name raises
        # AecConfigInvalid carrying the canceller's own message.
        aec_model: str | None
        aec_tail_ms: int | None
        aec_suppression: str | None
        aec_reference_sample_rate: int | None
        aec_reference_channels: int | None
        if aec is None:
            aec_model = None
            aec_tail_ms = None
            aec_suppression = None
            aec_reference_sample_rate = None
            aec_reference_channels = None
        elif isinstance(aec, Aec):
            aec_model = aec.model
            aec_tail_ms = aec.tail_ms
            aec_suppression = aec.suppression
            aec_reference_sample_rate = aec.reference_sample_rate
            aec_reference_channels = aec.reference_channels
        elif isinstance(aec, str):
            aec_model = aec
            aec_tail_ms = None
            aec_suppression = None
            aec_reference_sample_rate = None
            aec_reference_channels = None
        else:
            raise ValueError(
                f"Invalid aec value: {aec!r}. "
                "Expected None, a model name such as 'tau', or an Aec config object."
            )

        # Resolve the ORT dylib path via the four-arm priority order in
        # _ort_resolver. Consulted when an ONNX stage loads (Silero VAD or
        # denoise); energy mode and vad=False with no denoise never load ORT, so
        # the resolver call (and its bundled-dylib lookup) is skipped to avoid
        # the import-time cost. See _ort_resolver.resolve_ort_dylib_path for the
        # priority order.
        resolved_ort_path: str | None = None
        if (vad_enabled and vad_mode == "silero") or denoise is not None:
            from decibri._ort_resolver import resolve_ort_dylib_path

            resolved_ort_path = resolve_ort_dylib_path(ort_library_path)
        elif ort_library_path is not None:
            # Preserve passthrough for explicit caller paths even when ORT
            # is not loaded, so tests and tooling that introspect the
            # bridge state see the user's intent.
            resolved_ort_path = str(Path(ort_library_path))

        # Wrapper-only rename: the public Python surface uses `dtype`
        # (NumPy convention) but the internal Rust bridge keeps `format`
        # for cross-binding consistency with the Node binding. Translate
        # at the boundary; bridge stubs unchanged.
        self._bridge = _decibri.MicrophoneBridge(
            sample_rate=sample_rate,
            channels=channels,
            frames_per_buffer=frames_per_buffer,
            format=dtype,
            device=device,
            vad=vad_enabled,
            vad_threshold=vad_threshold,
            vad_mode=vad_mode,
            vad_holdoff=vad_holdoff_ms,
            model_path=resolved_model_path,
            numpy=as_ndarray,
            ort_library_path=resolved_ort_path,
            denoise=denoise,
            denoise_model_path=resolved_denoise_model_path,
            highpass=highpass,
            agc=agc,
            limiter=limiter,
            dc_removal=dc_removal,
            aec=aec_model,
            aec_tail_ms=aec_tail_ms,
            aec_suppression=aec_suppression,
            aec_reference_sample_rate=aec_reference_sample_rate,
            aec_reference_channels=aec_reference_channels,
            channel_map=channel_map,
            detector_source=vad_source,
        )

        self._vad_enabled = vad_enabled
        self._vad = _VadStateMachine(
            threshold=vad_threshold,
            holdoff_ms=vad_holdoff_ms,
        )
        self._format = dtype
        # Store the as_ndarray flag (bridge-level: numpy=) so read() can
        # branch its return type (bytes vs ndarray) without re-querying the
        # bridge each call. Wrapper-only rename; the bridge keeps the
        # original `numpy` name for cross-binding consistency.
        self._as_ndarray = as_ndarray
        # Chunk counter for read_with_metadata().
        # Increments on every non-None chunk emission; resets to 0 on
        # each stop() so a subsequent start() begins a fresh sequence.
        self._sequence = 0
        # Capture construction parameters for __repr__.
        # The bridge does not expose these as readable attributes, so the
        # wrapper holds its own copies. ``vad`` stores the original public
        # union value (``False``, ``"silero"``, or ``"energy"``) rather
        # than the split (``vad_enabled``, ``vad_mode``) bridge form.
        self._sample_rate = sample_rate
        self._channels = channels
        self._frames_per_buffer = frames_per_buffer
        self._device = device
        self._vad_arg = vad

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the capture stream.

        Re-entry contract:
            Calling ``start()`` after ``stop()`` or ``close()`` is supported
            and reconstructs the underlying audio stream cleanly. The
            ``Microphone`` instance is reusable; you can stop, start, and
            stop again as many times as needed. VAD state
            (``is_speaking``, ``vad_score``) resets to default values on
            each new ``start()``. Re-entry after exiting a ``with`` block
            is also supported (since ``__exit__`` calls ``stop()``).

            Calling ``start()`` while already started raises
            ``AlreadyRunning``.
        """
        self._bridge.start()

    def stop(self) -> None:
        """Stop the capture stream and reset VAD state."""
        self._bridge.stop()
        self._vad.reset()
        self._sequence = 0

    def close(self) -> None:
        """Stop the capture stream. Permanent alias for ``stop()``.

        Provided for ergonomic parity with the asyncio / aiohttp /
        httpx convention and for use cases where ``close()`` reads
        more naturally than ``stop()``. The two methods are guaranteed
        to remain semantically equivalent across all decibri versions.
        """
        # Calls self.stop() rather than self._bridge.close() so the
        # wrapper-side cleanup (vad.reset() in stop()) runs. The
        # bridge-level MicrophoneBridge.close() exists for symmetry
        # with SpeakerBridge.close() and for advanced direct-bridge
        # users; the wrapper keeps its own routing here to ensure
        # VAD state is reset on every close.
        self.stop()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    # -----------------------------------------------------------------------
    # Read surface
    # -----------------------------------------------------------------------

    def read(self, timeout_ms: int | None = None) -> SampleData | None:
        """Read one chunk. Returns the chunk, or None if the stream closed.

        Return type:
        - When ``as_ndarray=False`` (default), returns ``bytes``.
        - When ``as_ndarray=True``, returns a ``numpy.ndarray`` with
          dtype matching the configured ``dtype`` (np.int16 or
          np.float32) and shape matching the channel count: 1-D at
          ``channels=1``, 2-D ``(frames, channels)`` above it.

        Use ``read_with_metadata()`` to receive a typed ``Chunk`` with
        ``.data``, ``.timestamp``, ``.sequence``, ``.is_speaking``, and
        ``.vad_score`` attributes. ``read()`` keeps its naked-data
        signature for backward compatibility.

        VAD state advances as a side effect when VAD is enabled
        (``vad="silero"`` or ``vad="energy"``). Consumers should check
        ``is_speaking`` after each read. The score comes from the bridge
        (computed natively on the pre-enhancement signal for both modes),
        so the returned chunk is not inspected for VAD and the return type
        (bytes vs ndarray) does not affect it.

        ``as_ndarray=True`` requires the optional ``numpy`` extra; a
        missing numpy raises ``ImportError`` at construction (install
        with ``pip install decibri[numpy]``).
        """
        try:
            chunk = self._bridge.read(timeout_ms=timeout_ms)
        except ImportError as exc:
            if self._as_ndarray:
                raise ImportError(
                    "numpy is not installed. Install with: pip install decibri[numpy]"
                ) from exc
            raise
        if chunk is None:
            return None
        if self._vad_enabled:
            # Both modes read the score the bridge computed on the
            # pre-enhancement signal; the chunk data is not inspected here.
            self._vad.process_chunk(self._bridge.vad_probability)
        self._sequence += 1
        return chunk

    def read_with_metadata(self, timeout_ms: int | None = None) -> Chunk | None:
        """Read one chunk and return it as a typed ``Chunk`` with metadata.

        Returns ``None`` if the stream closed cleanly (mirroring
        ``read()``); otherwise returns a frozen ``Chunk`` with
        ``.data``, ``.timestamp``, ``.sequence``, ``.is_speaking``,
        and ``.vad_score`` attributes. The ``data`` field has the
        same shape and type as ``read()`` would have returned
        (``bytes`` by default; ``np.ndarray`` when
        ``as_ndarray=True``).

        ``timestamp`` is ``time.monotonic()`` taken immediately after
        the bridge returns the chunk; useful for relative timing
        within a session. ``sequence`` is a monotonic per-session
        counter starting at 0; it resets on each new ``start()``.
        ``is_speaking`` and ``vad_score`` snapshot the VAD state at
        the chunk boundary (always ``False`` and ``0.0`` respectively
        when ``vad=False``).
        """
        data = self.read(timeout_ms=timeout_ms)
        if data is None:
            return None
        # self._sequence has just been incremented inside read(); the
        # current chunk's index is sequence - 1 (0-based).
        return Chunk(
            data=data,
            timestamp=time.monotonic(),
            sequence=self._sequence - 1,
            is_speaking=self.is_speaking,
            vad_score=self.vad_score,
        )

    def iter_with_metadata(self) -> Iterator[Chunk]:
        """Yield ``Chunk`` objects until the stream closes cleanly.

        Generator wrapping ``read_with_metadata()``: stops when the
        bridge returns ``None``. Use this in place of
        ``for chunk in mic`` when you want metadata alongside the
        audio data.
        """
        while True:
            chunk = self.read_with_metadata(timeout_ms=None)
            if chunk is None:
                return
            yield chunk

    def __iter__(self) -> Iterator[SampleData]:
        return self

    def __next__(self) -> SampleData:
        chunk = self.read(timeout_ms=None)
        if chunk is None:
            raise StopIteration
        return chunk

    # -----------------------------------------------------------------------
    # State properties
    # -----------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True if the capture stream is currently running."""
        return self._bridge.is_open

    @property
    def is_speaking(self) -> bool:
        """True if VAD currently considers the user to be speaking.

        Reflects the wrapper-layer state machine: above-threshold detection
        plus holdoff grace period. Always False when vad=False.

        Holdoff expiry is checked on each property access; consumers who
        pause iteration still observe correct state when they next read.
        """
        if not self._vad_enabled:
            return False
        return self._vad.is_speaking

    @property
    def vad_score(self) -> float:
        """Most recent VAD score in ``[0, 1]``. Mode-agnostic.

        In ``vad="silero"`` mode, returns the raw probability from the
        Silero model. In ``vad="energy"`` mode, returns the normalized
        RMS energy of the most recent chunk. Both are computed natively on
        the signal before any opt-in enhancement step, so enabling
        enhancement does not change the score. Always 0.0 when
        ``vad=False``.

        The underlying bridge property is named ``vad_probability`` for
        cross-binding consistency; ``vad_score`` is the mode-agnostic
        wrapper-side name (it is not a probability in energy mode).
        """
        if not self._vad_enabled:
            return 0.0
        return self._vad.vad_score

    @property
    def overrun_count(self) -> int:
        """Number of capture buffers dropped because the consumer could not
        keep pace.

        0 while the consumer keeps up, before capture starts, and after
        ``stop()``, which releases the stream the counter lives on. Read
        it before stopping to see a session's total. A rising value means
        audio is being dropped to bound memory.
        """
        return self._bridge.overrun_count

    # -----------------------------------------------------------------------
    # Echo cancellation surface
    # -----------------------------------------------------------------------

    def push_aec_reference(self, samples: SampleData) -> None:
        """Queue far-end reference audio for the echo canceller.

        ``samples`` is the audio being played out, pushed as it is played, in
        played order: at the declared ``reference_sample_rate`` (the capture
        ``sample_rate`` when unset), interleaved at the declared
        ``reference_channels`` (mono when unset), as ``bytes`` or a
        ``numpy.ndarray``, the same input shapes ``Speaker.write`` accepts,
        with dtype matching this microphone's ``dtype``. With
        ``reference_channels`` above 1, each frame is averaged to one mono
        sample before the canceller sees it; a multichannel reference pushed
        without declaring the count cancels nothing and reports no error.
        The declared count must match this buffer's actual interleaving: a
        mismatch is not detected and raises no error, and shows up only as
        ``aec_metrics().delay_samples`` staying ``None`` with no fault
        reported.

        Never blocks and never raises on a full queue: samples that do not
        fit are discarded and counted by ``aec_metrics().reference_dropped``,
        and the span they occupied is represented as silence. Silence between
        played audio need not be pushed; a caller that stops pushing has said
        nothing is playing. A push while capture is not running, or with the
        ``aec`` parameter unset, is a no-op. A bad input type raises
        ``TypeError`` regardless of capture state.
        """
        self._bridge.push_aec_reference(samples)

    def aec_metrics(self) -> AecMetrics | None:
        """The echo canceller's metrics, or ``None`` when the ``aec``
        parameter is unset or capture is not running.

        See ``AecMetrics`` for the fields and the diagnostic signatures they
        carry.
        """
        raw = self._bridge.aec_metrics()
        if raw is None:
            return None
        return _aec_metrics_from_raw(raw)

    # -----------------------------------------------------------------------
    # Static methods
    # -----------------------------------------------------------------------

    @staticmethod
    def devices() -> list[MicrophoneInfo]:
        """List available audio input devices."""
        return _decibri.MicrophoneBridge.devices()

    @staticmethod
    def version() -> VersionInfo:
        """Return version info: decibri Rust core, cpal, and binding wheel."""
        return _decibri.MicrophoneBridge.version()

    def __del__(self) -> None:
        # Defensive finalizer: best-effort cleanup if the user forgot the
        # context manager and the GC reaps the instance with the cpal
        # stream still live. Tolerates partially-constructed state (the
        # _bridge attribute may not exist if the constructor raised before
        # assigning it) and double-close (bridge.stop() is idempotent).
        # The bare BaseException catch is intentional: __del__ runs at
        # arbitrary GC points, including interpreter shutdown when raising
        # is unsafe; we silently absorb anything rather than triggering
        # "Exception ignored in __del__" noise on the user's terminal.
        bridge = getattr(self, "_bridge", None)
        if bridge is None:
            return
        try:
            bridge.stop()
        except BaseException:  # noqa: BLE001
            pass

    def __repr__(self) -> str:
        # Show construction parameters plus current is_open state.
        # Pattern matches VersionInfo's repr precedent; is_open is queried
        # live from the bridge so the repr reflects actual state, not just
        # construction-time state.
        is_open: bool | str
        try:
            is_open = self.is_open
        except Exception:  # noqa: BLE001
            # Bridge missing or partially constructed: report unknown
            # rather than raising from __repr__.
            is_open = "?"
        return (
            f"Microphone(sample_rate={self._sample_rate}, "
            f"channels={self._channels}, "
            f"dtype={self._format!r}, "
            f"frames_per_buffer={self._frames_per_buffer}, "
            f"device={self._device!r}, "
            f"vad={self._vad_arg!r}, "
            f"is_open={is_open})"
        )


# ---------------------------------------------------------------------------
# Speaker: consumer-facing audio output class.
#
# Thin wrapper over SpeakerBridge. No VAD, no policy. Provided here
# for symmetry and a unified import surface.
# ---------------------------------------------------------------------------


class Speaker:
    """Audio output stream.

    Example:
        with Speaker(sample_rate=16000, channels=1) as out:
            out.write(audio_bytes)
            out.drain()

    Disconnect:
        A playback device that fails mid-stream (USB unplug, driver reset)
        raises ``DeviceFailed`` from the next ``write()`` or ``drain()``,
        carrying the driver's own cause. A producer that has stopped writing
        is not told; ``is_playing`` goes false immediately either way. A
        deliberate ``stop()`` or ``close()`` is never reported as a device
        failure: a later ``write()`` raises ``SpeakerStreamClosed`` as before.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        device: int | str | None = None,
    ) -> None:
        """Construct a Speaker audio output instance.

        Parameters
        ----------
        sample_rate : int, optional
            Output sample rate in Hz. Default 16000 (matches the
            cloud-STT capture convention used by ``Microphone``). For
            playback of OpenAI Realtime audio use 24000.
        channels : int, optional
            Number of output channels. Default 1 (mono). Multi-channel
            samples are interleaved on the wire. Bounded below only: a
            count the device cannot serve raises
            ``SpeakerChannelsUnsupported`` at ``start()``, naming the
            count the device reports.
        dtype : str, optional
            Sample dtype: ``"int16"`` (default) or ``"float32"``. Must
            match the dtype of the data passed to ``write()``; mismatch
            raises ``TypeError`` at write time.
        device : int | str | None, optional
            Output device selector. ``None`` (default) uses the system
            default output. Pass an integer index from
            ``Speaker.devices()`` or a substring of the device name.
            ``Speaker`` does not load ONNX Runtime, so there is no
            ``ort_library_path`` parameter (output never invokes VAD).
        """
        if dtype not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"dtype must be 'int16' or 'float32'; got {dtype!r}"
            )
        # Wrapper-only rename: public surface uses `dtype`; bridge keeps
        # `format` for cross-binding consistency.
        self._bridge = _decibri.SpeakerBridge(
            sample_rate=sample_rate,
            channels=channels,
            format=dtype,
            device=device,
        )
        # Capture construction parameters for __repr__.
        self._sample_rate = sample_rate
        self._channels = channels
        self._format = dtype
        self._device = device

    def start(self) -> None:
        """Open and start the output stream.

        Re-entry contract:
            Calling ``start()`` after ``stop()`` or ``close()`` is
            supported and reconstructs the underlying output stream
            cleanly. The ``Speaker`` instance is reusable across
            stop/start cycles. Re-entry after exiting a ``with`` block
            is also supported (since ``__exit__`` calls ``stop()``).

            Calling ``start()`` while already started raises
            ``AlreadyRunning``.
        """
        self._bridge.start()

    def stop(self) -> None:
        """Stop the output stream."""
        self._bridge.stop()

    def close(self) -> None:
        """Stop the output stream. Permanent alias for ``stop()``.

        Provided for ergonomic parity with the asyncio / aiohttp /
        httpx convention and for use cases where ``close()`` reads
        more naturally than ``stop()``. The two methods are guaranteed
        to remain semantically equivalent across all decibri versions.
        """
        self._bridge.close()

    def write(self, samples: SampleData) -> None:
        """Write a chunk to the output stream.

        Accepts either ``bytes`` or a
        ``numpy.ndarray`` with dtype matching the configured ``dtype``
        (np.int16 for ``dtype='int16'``, np.float32 for
        ``dtype='float32'``). Multi-channel ndarrays use shape
        ``(N, channels)`` (interleaved). Output bridges duck-type the
        input on each call rather than committing at construction time.

        Raises ``TypeError`` on dtype mismatch or unsupported input type.
        """
        self._bridge.write(samples)

    def drain(self) -> None:
        """Block until all queued samples have been played."""
        self._bridge.drain()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    @property
    def is_playing(self) -> bool:
        return self._bridge.is_playing

    @property
    def underrun_count(self) -> int:
        """Number of samples emitted as silence fill because the playback
        queue ran dry.

        0 while the producer keeps the queue fed, or before playback
        starts. A rising value means the output is papering over gaps with
        silence.
        """
        return self._bridge.underrun_count

    @staticmethod
    def devices() -> list[SpeakerInfo]:
        """List available audio output devices."""
        return _decibri.SpeakerBridge.devices()

    def __del__(self) -> None:
        # Defensive finalizer; same shape as Microphone.__del__.
        # See that method's comment for rationale.
        bridge = getattr(self, "_bridge", None)
        if bridge is None:
            return
        try:
            bridge.stop()
        except BaseException:  # noqa: BLE001
            pass

    def __repr__(self) -> str:
        # Same shape as Microphone.__repr__; Speaker has no VAD so the repr
        # omits that field. is_playing is the analogue of is_open here.
        is_playing: bool | str
        try:
            is_playing = self.is_playing
        except Exception:  # noqa: BLE001
            is_playing = "?"
        return (
            f"Speaker(sample_rate={self._sample_rate}, "
            f"channels={self._channels}, "
            f"dtype={self._format!r}, "
            f"device={self._device!r}, "
            f"is_playing={is_playing})"
        )


# ---------------------------------------------------------------------------
# Whole-recording VAD report types.
#
# Frozen dataclasses returned from File.analyze() / File.analyse(). Times are
# seconds of FILE time (sample positions over the rate), never wall-clock.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VadWindow:
    """One scored voice-activity window of a recording.

    Produced by ``File.analyze()``. Windows tile the recording from the
    start in fixed steps (512 samples at 16 kHz, 32 ms per window); a
    trailing remainder shorter than one window is not scored, exactly as
    live detection leaves a sub-window remainder unscored.

    Attributes:
        start: Window start, in seconds of file time.
        end: Window end, in seconds of file time.
        vad_score: Speech probability for this window in ``[0, 1]``. The
            same quantity the live per-chunk ``vad_score`` reports, here
            per window across the whole recording.
        is_speech: Whether ``vad_score`` meets the configured threshold.
            The raw per-window test, not the debounced ``is_speaking``.
    """

    start: float
    end: float
    vad_score: float
    is_speech: bool


@dataclass(frozen=True, slots=True)
class Segment:
    """One merged speech region of a recording.

    Produced by ``File.analyze()``: consecutive speech windows whose
    silence gaps are within the configured holdoff collapse into one
    segment. The segment ends at the last speech window, not at the
    holdoff expiry.

    Attributes:
        start: Region start, in seconds of file time.
        end: Region end, in seconds of file time.
    """

    start: float
    end: float


@dataclass(frozen=True, slots=True)
class VadReport:
    """The whole-recording voice-activity analysis ``File.analyze()`` returns.

    Attributes:
        scores: Per-window speech scores across the whole recording, in
            file order.
        segments: Merged speech regions across the whole recording, in
            file order.
    """

    scores: list[VadWindow]
    segments: list[Segment]


@dataclass(frozen=True, slots=True)
class SaveReport:
    """What ``File.save()`` did to the samples on their way into the file.

    Attributes:
        clipped_samples: Finite samples outside full scale, clamped to
            ``[-1.0, 1.0]``. Conditioned audio can exceed full scale (AGC
            or AEC without a limiter), and 16-bit PCM cannot hold that, so
            the overshoot clips and this count says how much. The count is
            a statement about integer encodings: a float encoding would
            preserve the overshoot instead, and would report zero.
        non_finite_samples: Non-finite samples replaced before writing:
            NaN with silence, an infinity with full scale. The same
            replacement on every format.
    """

    clipped_samples: int
    non_finite_samples: int


# ---------------------------------------------------------------------------
# File-time VAD policy state machine.
#
# The live Microphone's state machine (above) measures its holdoff in
# wall-clock time because capture is real time. A File processes faster than
# real time, so the same policy here is measured in FILE time: sample
# positions converted to seconds. State advances only as chunks are read;
# there is no wall-clock timer to expire, and processing speed never changes
# the reported speech timing.
# ---------------------------------------------------------------------------


class _FileVadStateMachine:
    """Wrapper-layer VAD policy in file time. Pure Python; no native calls."""

    __slots__ = (
        "_threshold",
        "_holdoff_seconds",
        "_is_speaking",
        "_silence_started_pos",
        "_last_probability",
    )

    def __init__(self, threshold: float, holdoff_ms: int) -> None:
        self._threshold = float(threshold)
        self._holdoff_seconds = holdoff_ms / 1000.0
        self._is_speaking = False
        self._silence_started_pos: float | None = None
        self._last_probability = 0.0

    def reset(self) -> None:
        """Clear all transient state."""
        self._is_speaking = False
        self._silence_started_pos = None
        self._last_probability = 0.0

    def process_chunk(
        self, probability: float, chunk_start: float, chunk_end: float
    ) -> None:
        """Update VAD state from one chunk's bridge score.

        ``chunk_start`` / ``chunk_end`` are the chunk's position in seconds
        of file time. Above-threshold chunks set the speaking state; a
        below-threshold run flips it off once ``holdoff_ms`` of file time
        has elapsed since the silence began.
        """
        self._last_probability = probability

        if probability >= self._threshold:
            self._is_speaking = True
            self._silence_started_pos = None
        elif self._is_speaking:
            if self._silence_started_pos is None:
                self._silence_started_pos = chunk_start
            if chunk_end - self._silence_started_pos >= self._holdoff_seconds:
                self._is_speaking = False
                self._silence_started_pos = None

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def vad_score(self) -> float:
        return self._last_probability


# ---------------------------------------------------------------------------
# File: consumer-facing offline source class.
# ---------------------------------------------------------------------------


class File:
    """Offline audio source: conditions a recording or in-memory samples.

    Everything a ``Microphone`` does to live audio, ``File`` does to audio
    you already have: the same conditioning options, the same iteration,
    the same conditioned chunks out. Because a ``File`` is a complete
    recording, it can also analyze the whole recording for speech with
    ``analyze()`` / ``analyse()``, which a live stream cannot do.

    Example:
        with File("clip.wav", denoise="fastenhancer-t", agc=-18) as file:
            for chunk in file:
                handle(chunk)      # conditioned int16 PCM bytes

        report = File("clip.wav", vad="silero").analyze()
        for segment in report.segments:
            print(segment.start, segment.end)

    Construction reads the whole file (``File(path)`` / ``File.open(path)``,
    both identical) or wraps in-memory samples (``File.buffer(samples,
    input_rate=...)``). Iteration and analysis are separate single passes:
    each consumes the source once, so use one ``File`` per operation.

    VAD is opt-in via ``vad=`` exactly as on ``Microphone``. With VAD on,
    metadata iteration carries per-chunk ``vad_score`` / ``is_speaking``
    (the speaking holdoff measured in file time, not wall-clock time), and
    ``analyze()`` returns a ``VadReport``. Without ``vad=`` the File simply
    conditions audio and ``analyze()`` raises ``VadNotConfigured``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        channel_map: list[int] | None = None,
        dtype: str = "int16",
        vad: bool | str | Vad = False,
        model_path: str | Path | None = None,
        as_ndarray: bool = False,
        ort_library_path: str | Path | None = None,
        denoise: Literal["fastenhancer-t"] | None = None,
        highpass: Literal[80, 100] | None = None,
        agc: int | None = None,
        limiter: float | None = None,
        dc_removal: bool = False,
    ) -> None:
        """Open an audio file as an offline source.

        Reads WAV, AIFF, AIFF-C and FLAC. The container is identified from
        the file's own bytes, so the path's extension does not decide how it
        is read. The input rate and channel count come from the file's
        header. ``sample_rate`` is the target output rate, and ``channels``
        and ``channel_map`` name what is delivered, all with the same
        meanings they have on ``Microphone``, the source's header count
        standing where the device's report stands: ``channels=1`` (the
        default) delivers the average of every source channel, a count
        equal to the source's own delivers every channel in source order,
        and a ``channel_map`` of 0-based source channel indices selects and
        permutes. An unmapped count above the source's own raises
        ``FileChannelsUnsupported``; an unmapped strict subset above one
        raises ``FileChannelSelectionAmbiguous``; a map entry the source
        does not have raises ``FileChannelMapOutOfRange``. The source's
        count is the only ceiling; no fixed maximum exists.
        """
        self._init_common(
            ("path", str(Path(path))),
            sample_rate=sample_rate,
            channels=channels,
            channel_map=channel_map,
            dtype=dtype,
            vad=vad,
            model_path=model_path,
            as_ndarray=as_ndarray,
            ort_library_path=ort_library_path,
            denoise=denoise,
            highpass=highpass,
            agc=agc,
            limiter=limiter,
            dc_removal=dc_removal,
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "File":
        """Open an audio file as an offline source; identical to ``File(path)``.

        The explicit spelling of the bare constructor: both produce the
        same ``File``. Accepts the same keyword arguments.
        """
        return cls(path, **kwargs)

    @classmethod
    def buffer(
        cls,
        samples: "list[float] | np.ndarray[Any, Any]",
        *,
        input_rate: int,
        input_channels: int = 1,
        sample_rate: int = 16000,
        channels: int = 1,
        channel_map: list[int] | None = None,
        dtype: str = "int16",
        vad: bool | str | Vad = False,
        model_path: str | Path | None = None,
        as_ndarray: bool = False,
        ort_library_path: str | Path | None = None,
        denoise: Literal["fastenhancer-t"] | None = None,
        highpass: Literal[80, 100] | None = None,
        agc: int | None = None,
        limiter: float | None = None,
        dc_removal: bool = False,
    ) -> "File":
        """Wrap in-memory samples as an offline source.

        ``samples`` is a list of floats or a numpy ndarray of samples in
        ``[-1.0, 1.0]``, frame-interleaved at ``input_channels`` (1, mono,
        by default). At ``input_channels=1`` an ndarray must be one channel
        with a floating dtype: a redundant axis (``(N, 1)`` or ``(1, N)``)
        is accepted, an array with more than one axis longer than 1 raises
        ``ValueError``, and a non-floating dtype (int16 PCM, for example)
        raises ``ValueError`` rather than being cast. Above one input
        channel an ndarray is either 1-D interleaved or 2-D
        ``(frames, input_channels)``; any other shape raises ``ValueError``.
        A sample count that is not a whole number of frames raises
        ``BlockSizeNotFrameAligned``. Raw samples carry no header, so
        ``input_rate`` (their native rate) is required and
        ``input_channels`` is its channel counterpart; ``sample_rate``
        stays the target output rate, and ``channels`` and ``channel_map``
        name what is delivered, exactly as on ``File(path)``.
        """
        self = object.__new__(cls)
        self._init_common(
            ("buffer", samples, input_rate, input_channels),
            sample_rate=sample_rate,
            channels=channels,
            channel_map=channel_map,
            dtype=dtype,
            vad=vad,
            model_path=model_path,
            as_ndarray=as_ndarray,
            ort_library_path=ort_library_path,
            denoise=denoise,
            highpass=highpass,
            agc=agc,
            limiter=limiter,
            dc_removal=dc_removal,
        )
        return self

    def _init_common(
        self,
        source: tuple[Any, ...],
        *,
        sample_rate: int,
        channels: int,
        channel_map: list[int] | None,
        dtype: str,
        vad: bool | str | Vad,
        model_path: str | Path | None,
        as_ndarray: bool,
        ort_library_path: str | Path | None,
        denoise: "Literal['fastenhancer-t'] | None",
        highpass: "Literal[80, 100] | None",
        agc: int | None,
        limiter: float | None,
        dc_removal: bool,
    ) -> None:
        # The validation and resolution below mirror Microphone.__init__
        # exactly (same checks, same messages), so the two sources reject
        # and resolve identically.
        if dtype not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"dtype must be 'int16' or 'float32'; got {dtype!r}"
            )

        # ndarray output requires numpy at read time; checked here, before
        # the bridge is constructed, so a missing install raises a catchable
        # ImportError from the constructor.
        if as_ndarray:
            _require_numpy()

        vad_enabled: bool
        vad_mode: str
        vad_threshold: float | None
        vad_holdoff_ms: int
        vad_source: int | None
        if vad is False:
            vad_enabled = False
            vad_mode = "energy"  # inert placeholder; bridge ignores when disabled
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
            vad_source = None
        elif vad is True:
            raise ValueError(
                "vad=True is no longer supported. "
                "Specify the mode explicitly: vad='silero' or vad='energy'."
            )
        elif isinstance(vad, Vad):
            vad_enabled = True
            vad_mode = vad.model
            vad_threshold = vad.threshold
            vad_holdoff_ms = vad.holdoff_ms
            vad_source = vad.source
        elif isinstance(vad, str) and vad in _VALID_MODES:
            vad_enabled = True
            vad_mode = vad
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
            vad_source = None
        else:
            raise ValueError(
                f"Invalid vad value: {vad!r}. "
                "Expected False, 'silero', 'energy', or a Vad config object."
            )

        if vad_threshold is None:
            vad_threshold = 0.5 if vad_mode == "silero" else 0.01

        resolved_model_path: str | None = None
        if model_path is not None:
            resolved_model_path = str(Path(model_path))
            if (
                vad_enabled
                and vad_mode == "silero"
                and not Path(resolved_model_path).is_file()
            ):
                raise exceptions.VadModelLoadFailed(
                    f"Silero VAD model not found at {resolved_model_path}. "
                    "Ensure model_path points to an existing ONNX model file.",
                    resolved_model_path,
                )
        elif vad_enabled and vad_mode == "silero":
            try:
                model_resource = (
                    importlib.resources.files("decibri")
                    / "models"
                    / "silero_vad.onnx"
                )
                if not model_resource.is_file():
                    raise FileNotFoundError(
                        f"Bundled Silero model resource exists but is not a "
                        f"file: {model_resource}"
                    )
                resolved_model_path = str(model_resource)
            except (FileNotFoundError, ModuleNotFoundError, AttributeError) as exc:
                raise exceptions.VadModelLoadFailed(
                    "model_path was not provided and the bundled Silero "
                    "model could not be located in the installed wheel. "
                    "Ensure the models/ directory was included during "
                    "installation, or pass model_path explicitly.",
                    _BUNDLED_VAD_MODEL,
                ) from exc

        resolved_denoise_model_path: str | None = None
        if denoise is not None:
            if denoise not in _VALID_DENOISE_MODELS:
                raise ValueError(
                    f"Invalid denoise value: {denoise!r}. Expected 'fastenhancer-t'."
                )
            try:
                denoise_resource = (
                    importlib.resources.files("decibri")
                    / "models"
                    / "fastenhancer_t.onnx"
                )
                if not denoise_resource.is_file():
                    raise FileNotFoundError(
                        f"Bundled denoise model resource exists but is not a "
                        f"file: {denoise_resource}"
                    )
                resolved_denoise_model_path = str(denoise_resource)
            except (FileNotFoundError, ModuleNotFoundError, AttributeError) as exc:
                raise exceptions.ModelLoadFailed(
                    "the bundled denoise model could not be located in the "
                    "installed wheel. Ensure the models/ directory was included "
                    "during installation.",
                    _BUNDLED_DENOISE_MODEL,
                ) from exc

        if highpass is not None and highpass not in _VALID_HIGHPASS:
            raise ValueError(
                f"highpass must be one of: 80, 100; got {highpass!r}"
            )

        if agc is not None and not -40 <= agc <= -3:
            raise exceptions.AgcTargetOutOfRange(f"agc must be in [-40, -3]; got {agc}")

        if limiter is not None and not -3.0 <= limiter <= 0.0:
            raise exceptions.LimiterCeilingOutOfRange(
                f"limiter must be in [-3.0, 0.0]; got {limiter}"
            )

        # Validate the channel map's shape exactly as Microphone.__init__
        # does: a list of integers in the channel count's width, with one
        # entry per delivered channel (the core names the length failure, so
        # the wrapper raises its class). Whether each entry exists on the
        # source is the core's check, made against the source's own count at
        # construction, because only the opened source can say how many
        # channels it has; no fixed maximum exists on this path.
        if channel_map is not None:
            for entry in channel_map:
                if not isinstance(entry, int) or isinstance(entry, bool):
                    raise TypeError(
                        f"channel_map entries must be integers; got {entry!r}"
                    )
                if not 0 <= entry <= 65535:
                    raise ValueError(
                        f"channel_map entries must be in [0, 65535]; got {entry}"
                    )
            if len(channel_map) != channels:
                raise exceptions.ChannelMapLengthMismatch(
                    f"the channel map has {len(channel_map)} entries; it must "
                    f"have exactly one entry per delivered channel ({channels})"
                )

        resolved_ort_path: str | None = None
        if (vad_enabled and vad_mode == "silero") or denoise is not None:
            from decibri._ort_resolver import resolve_ort_dylib_path

            resolved_ort_path = resolve_ort_dylib_path(ort_library_path)
        elif ort_library_path is not None:
            resolved_ort_path = str(Path(ort_library_path))

        bridge_kwargs: dict[str, Any] = {
            "sample_rate": sample_rate,
            "channels": channels,
            "channel_map": channel_map,
            "format": dtype,
            "vad": vad_enabled,
            "vad_threshold": vad_threshold,
            "vad_mode": vad_mode,
            "vad_holdoff": vad_holdoff_ms,
            "model_path": resolved_model_path,
            "numpy": as_ndarray,
            "ort_library_path": resolved_ort_path,
            "denoise": denoise,
            "denoise_model_path": resolved_denoise_model_path,
            "highpass": highpass,
            "agc": agc,
            "limiter": limiter,
            "dc_removal": dc_removal,
            "detector_source": vad_source,
        }

        if source[0] == "path":
            self._bridge = _decibri.FileBridge.open(source[1], **bridge_kwargs)
        else:
            samples = source[1]
            input_channels = source[3]
            # Lists pass through directly; numpy arrays are transported as
            # raw f32 little-endian bytes to avoid a per-sample boundary
            # cost. Anything else is rejected before it can be misread.
            if isinstance(samples, (list, tuple)):
                samples = list(samples)
            else:
                try:
                    import numpy as np
                except ImportError:
                    raise TypeError(
                        "samples must be a list of floats or a numpy ndarray"
                    ) from None
                if not isinstance(samples, np.ndarray):
                    raise TypeError(
                        "samples must be a list of floats or a numpy ndarray"
                    )
                if input_channels == 1:
                    # One channel means at most one axis longer than 1: a
                    # redundant axis squeezes away without reordering, while
                    # a second long axis carries channels the C-order
                    # flatten below would splice into the sample stream.
                    if sum(1 for length in samples.shape if length > 1) > 1:
                        raise ValueError(
                            "samples must be one channel; got an array of shape "
                            f"{samples.shape}. Select a single channel or mix "
                            "down to mono before passing it."
                        )
                elif not (
                    samples.ndim == 1
                    or (samples.ndim == 2 and samples.shape[1] == input_channels)
                ):
                    # Above one input channel the array is either already
                    # interleaved (1-D) or laid out one frame per row, one
                    # channel per column, which the C-order flatten below
                    # interleaves frame by frame.
                    raise ValueError(
                        "samples must be 1-D interleaved or 2-D "
                        f"(frames, input_channels); got shape {samples.shape} "
                        f"with input_channels={input_channels}"
                    )
                if not np.issubdtype(samples.dtype, np.floating):
                    raise ValueError(
                        f"samples must have a floating dtype; got {samples.dtype}. "
                        "Convert with samples.astype(numpy.float32), scaling "
                        "integer PCM to [-1.0, 1.0]."
                    )
                samples = np.ascontiguousarray(samples, dtype=np.float32).tobytes()
            self._bridge = _decibri.FileBridge.buffer(
                samples, source[2], input_channels, **bridge_kwargs
            )

        self._vad_enabled = vad_enabled
        self._vad_mode = vad_mode
        self._vad = _FileVadStateMachine(
            threshold=vad_threshold,
            holdoff_ms=vad_holdoff_ms,
        )
        self._format = dtype
        self._as_ndarray = as_ndarray
        self._sequence = 0
        # File-time position in seconds, advanced by each delivered chunk.
        self._position = 0.0
        self._sample_rate = sample_rate
        self._vad_arg = vad

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Release the source. Idempotent; a closed ``File`` reads as ended."""
        self._bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Read surface
    # -----------------------------------------------------------------------

    def read(self) -> SampleData | None:
        """Read one conditioned chunk. Returns ``None`` at end of file.

        The final chunk may be shorter once the chain's end-of-stream tail
        has been drained. Return type mirrors ``Microphone.read()``:
        ``bytes`` by default, ``numpy.ndarray`` when ``as_ndarray=True``.

        VAD state advances as a side effect when VAD is enabled, with the
        speaking holdoff measured in FILE time (sample positions), so
        processing speed never changes the reported state.
        """
        try:
            chunk = self._bridge.read()
        except ImportError as exc:
            if self._as_ndarray:
                raise ImportError(
                    "numpy is not installed. Install with: pip install decibri[numpy]"
                ) from exc
            raise
        if chunk is None:
            return None
        chunk_start = self._position
        chunk_end = chunk_start + self._chunk_frames(chunk) / self._sample_rate
        self._position = chunk_end
        if self._vad_enabled:
            self._vad.process_chunk(
                self._bridge.vad_probability, chunk_start, chunk_end
            )
        self._sequence += 1
        return chunk

    def _chunk_frames(self, chunk: SampleData) -> int:
        """Number of frames in one delivered chunk.

        A frame carries one sample per delivered channel, and file time
        advances by frames, so a chunk's length is divided by the count it
        was interleaved at. The bytes path carries interleaved samples and
        is divided explicitly; the numpy path is 1-D ``(frames,)`` at one
        channel and 2-D ``(frames, channels)`` above it, so its length is
        the frame count on either shape.
        """
        if isinstance(chunk, bytes):
            samples = len(chunk) // (2 if self._format == "int16" else 4)
            return samples // self._bridge.channels
        return int(len(chunk))

    def read_with_metadata(self) -> Chunk | None:
        """Read one chunk and return it as a typed ``Chunk`` with metadata.

        Returns ``None`` at end of file; otherwise a frozen ``Chunk`` with
        the same fields the live ``Microphone`` produces. On a ``File`` the
        ``timestamp`` is the chunk's position in seconds of FILE time
        (from the start of the recording), and ``is_speaking`` applies the
        holdoff in file time, so the metadata describes the recording
        rather than the processing run.
        """
        chunk_start = self._position
        data = self.read()
        if data is None:
            return None
        return Chunk(
            data=data,
            timestamp=chunk_start,
            sequence=self._sequence - 1,
            is_speaking=self.is_speaking,
            vad_score=self.vad_score,
        )

    def iter_with_metadata(self) -> Iterator[Chunk]:
        """Yield ``Chunk`` objects until the end of the file.

        Generator wrapping ``read_with_metadata()``, exactly as the live
        ``Microphone`` offers: conditioned audio and per-chunk VAD state
        from the one pass.
        """
        while True:
            chunk = self.read_with_metadata()
            if chunk is None:
                return
            yield chunk

    def __iter__(self) -> Iterator[SampleData]:
        return self

    def __next__(self) -> SampleData:
        chunk = self.read()
        if chunk is None:
            raise StopIteration
        return chunk

    # -----------------------------------------------------------------------
    # Whole-recording analysis
    # -----------------------------------------------------------------------

    def analyze(self) -> VadReport:
        """Analyze the whole recording for speech and return a ``VadReport``.

        Runs the recording once through the conditioning pass, scoring the
        pre-conditioning signal window by window, and returns the
        per-window ``scores`` plus the merged speech ``segments``, all in
        seconds of file time. Consumes the source: analysis and iteration
        are separate single passes.

        Requires VAD: a ``File`` built without ``vad=`` raises
        ``VadNotConfigured``; the energy mode has no whole-recording
        analysis and raises ``ValueError``. Never constructs a detector
        silently.

        Requires a ``File`` still at its start: once iteration has pulled
        from it, this raises ``FileEngaged`` rather than reporting on the
        part not yet read. Every failure detected before the pass begins
        leaves the ``File`` usable, so iterating it afterwards still works;
        a failure during the pass, such as the detector failing to load,
        consumes the source.
        """
        # Ahead of the mode check, so an engaged File reports the same error
        # here as it does in the core and the other binding.
        self._bridge.check_not_engaged()
        if self._vad_enabled and self._vad_mode == "energy":
            raise ValueError(
                "analyze() requires vad='silero'; "
                "energy mode does not support whole-file analysis"
            )
        raw_scores, raw_segments = self._bridge.analyze()
        scores = [
            VadWindow(start=s, end=e, vad_score=p, is_speech=sp)
            for (s, e, p, sp) in raw_scores
        ]
        segments = [Segment(start=s, end=e) for (s, e) in raw_segments]
        return VadReport(scores=scores, segments=segments)

    # The same analysis under the international spelling; both are public.
    analyse = analyze

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["wav", "aiff", "flac"] | None = None,
        compression: int | None = None,
    ) -> SaveReport:
        """Write the conditioned recording to ``path`` as an audio file.

        Runs the recording once through the same conditioning pass
        iteration delivers, whole, and writes it as 16-bit PCM mono at
        ``sample_rate``. Consumes the source: a save is a single pass,
        separate from iteration and analysis.

        The container comes from the path's extension (``.wav``, ``.aiff``,
        ``.aif``, ``.aifc`` or ``.flac``), or from ``format``: decibri
        reads a file by its content and writes one by its name. An
        extension it does not recognise raises ``AudioFormatUnsupported``
        rather than defaulting. ``compression`` sets the FLAC compression
        level (0-8, default 5); it applies only to FLAC and is ignored for
        WAV and AIFF.

        Returns a ``SaveReport``: ``clipped_samples`` counts finite
        samples outside full scale clamped to ``[-1.0, 1.0]`` (AGC or AEC
        without a limiter can overshoot, and 16-bit PCM cannot hold it),
        and ``non_finite_samples`` counts NaN samples written as silence
        and infinite samples written as full scale.

        Requires a ``File`` still at its start: once iteration has pulled
        from it, this raises ``FileEngaged``. Every failure detected
        before the pass begins leaves the ``File`` usable; a failure
        during the pass consumes the source.
        """
        # Ahead of the argument checks, so an engaged File reports the same
        # error here as it does in the core and the other binding.
        self._bridge.check_not_engaged()
        if format is not None and format not in ("wav", "aiff", "flac"):
            raise ValueError(
                f"Invalid format value: {format!r}. Expected 'wav', 'aiff', or 'flac'."
            )
        if compression is not None and not 0 <= compression <= 8:
            raise exceptions.FlacCompressionOutOfRange(
                f"compression must be in [0, 8]; got {compression}"
            )
        clipped, non_finite = self._bridge.save(
            str(Path(path)), format=format, compression=compression
        )
        return SaveReport(clipped_samples=clipped, non_finite_samples=non_finite)

    # -----------------------------------------------------------------------
    # State properties
    # -----------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """True if per-chunk VAD currently considers speech present.

        The holdoff is measured in FILE time (sample positions), so a file
        processed faster than real time reports the same state sequence a
        live stream of the identical audio would. Always False when
        ``vad=False``.
        """
        if not self._vad_enabled:
            return False
        return self._vad.is_speaking

    @property
    def vad_score(self) -> float:
        """Most recent per-chunk VAD score in ``[0, 1]``. Mode-agnostic.

        The same quantity the live ``Microphone.vad_score`` reports,
        computed on the pre-conditioning signal. Always 0.0 when
        ``vad=False``.
        """
        if not self._vad_enabled:
            return 0.0
        return self._vad.vad_score

    @property
    def sample_rate(self) -> int:
        """The target output rate every delivered chunk carries."""
        return self._sample_rate

    @property
    def input_rate(self) -> int:
        """The source's native rate, from the file's header or the explicit
        ``input_rate`` of ``File.buffer``.
        """
        return self._bridge.input_rate

    def __del__(self) -> None:
        # Defensive finalizer; same shape as Microphone.__del__.
        bridge = getattr(self, "_bridge", None)
        if bridge is None:
            return
        try:
            bridge.close()
        except BaseException:  # noqa: BLE001
            pass

    def __repr__(self) -> str:
        return (
            f"File(sample_rate={self._sample_rate}, "
            f"dtype={self._format!r}, "
            f"vad={self._vad_arg!r})"
        )
