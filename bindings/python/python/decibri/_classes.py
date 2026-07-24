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

    Example:
        Microphone(vad=Vad(model="silero", threshold=0.6, holdoff_ms=200))
    """

    model: str = "silero"
    threshold: float | None = None
    holdoff_ms: int = _DEFAULT_VAD_HOLDOFF_MS

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

    This class is synchronous. For async iteration, use ``AsyncMicrophone``.

    Cleanup and disconnect:
        Mid-stream device disconnect (USB unplug, default-device switch,
        driver error) is surfaced as a ``MicrophoneStreamClosed`` raised on
        the next ``read()``. cpal detects the disconnect and closes the
        underlying stream within roughly 20ms; the wrapper then sees the
        closed state on its next read attempt and raises.

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
        if vad is False:
            vad_enabled = False
            vad_mode = "energy"  # inert placeholder; bridge ignores when disabled
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
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
        elif isinstance(vad, str) and vad in _VALID_MODES:
            vad_enabled = True
            vad_mode = vad
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
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
                raise ValueError(
                    "model_path was not provided and the bundled Silero "
                    "model could not be located in the installed wheel. "
                    "Ensure the models/ directory was included during "
                    "installation, or pass model_path explicitly."
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
                raise ValueError(
                    "the bundled denoise model could not be located in the "
                    "installed wheel. Ensure the models/ directory was included "
                    "during installation."
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
        # (typical -18); absence leaves it off. Mirrors the Vad threshold range
        # check; the Rust core guards the same range as a backstop.
        if agc is not None and not -40 <= agc <= -3:
            raise ValueError(f"agc must be in [-40, -3]; got {agc}")

        # Validate the limiter ceiling. A sample-peak ceiling in dBFS in
        # [-3.0, 0.0] (typical -1.0); absence leaves it off. Mirrors the agc range
        # check; the Rust core guards the same range as a backstop.
        if limiter is not None and not -3.0 <= limiter <= 0.0:
            raise ValueError(f"limiter must be in [-3.0, 0.0]; got {limiter}")

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
          np.float32) and shape matching the channel count (1-D for
          mono, 2-D ``(N, channels)`` for multi-channel).

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

        Raises ``ImportError`` with a clear actionable message if
        ``as_ndarray=True`` is set but the optional ``numpy`` extra is
        not installed (i.e. the user did not run
        ``pip install decibri[numpy]``).
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

        0 while the consumer keeps up, or before capture starts. A rising
        value means audio is being dropped to bound memory.
        """
        return self._bridge.overrun_count

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
            samples are interleaved on the wire.
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

    Construction reads the whole WAV (``File(path)`` / ``File.open(path)``,
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
        """Open a WAV file as an offline source.

        The input rate and channel count come from the WAV header;
        ``sample_rate`` is the target output rate, the same meaning it has
        on ``Microphone``. Supports 16-bit PCM and 32-bit float WAV files;
        other formats are handled elsewhere, keeping this path
        dependency-free.
        """
        self._init_common(
            ("path", str(Path(path))),
            sample_rate=sample_rate,
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
        """Open a WAV file as an offline source; identical to ``File(path)``.

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
        sample_rate: int = 16000,
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

        ``samples`` is a list of floats or a numpy ndarray of f32 mono
        samples in ``[-1.0, 1.0]``. Raw samples carry no header, so
        ``input_rate`` (their native rate) is required; ``sample_rate``
        stays the target output rate.
        """
        self = object.__new__(cls)
        self._init_common(
            ("buffer", samples, input_rate),
            sample_rate=sample_rate,
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

        vad_enabled: bool
        vad_mode: str
        vad_threshold: float | None
        vad_holdoff_ms: int
        if vad is False:
            vad_enabled = False
            vad_mode = "energy"  # inert placeholder; bridge ignores when disabled
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
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
        elif isinstance(vad, str) and vad in _VALID_MODES:
            vad_enabled = True
            vad_mode = vad
            vad_threshold = None
            vad_holdoff_ms = _DEFAULT_VAD_HOLDOFF_MS
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
                raise ValueError(
                    "model_path was not provided and the bundled Silero "
                    "model could not be located in the installed wheel. "
                    "Ensure the models/ directory was included during "
                    "installation, or pass model_path explicitly."
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
                raise ValueError(
                    "the bundled denoise model could not be located in the "
                    "installed wheel. Ensure the models/ directory was included "
                    "during installation."
                ) from exc

        if highpass is not None and highpass not in _VALID_HIGHPASS:
            raise ValueError(
                f"highpass must be one of: 80, 100; got {highpass!r}"
            )

        if agc is not None and not -40 <= agc <= -3:
            raise ValueError(f"agc must be in [-40, -3]; got {agc}")

        if limiter is not None and not -3.0 <= limiter <= 0.0:
            raise ValueError(f"limiter must be in [-3.0, 0.0]; got {limiter}")

        resolved_ort_path: str | None = None
        if (vad_enabled and vad_mode == "silero") or denoise is not None:
            from decibri._ort_resolver import resolve_ort_dylib_path

            resolved_ort_path = resolve_ort_dylib_path(ort_library_path)
        elif ort_library_path is not None:
            resolved_ort_path = str(Path(ort_library_path))

        bridge_kwargs: dict[str, Any] = {
            "sample_rate": sample_rate,
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
        }

        if source[0] == "path":
            self._bridge = _decibri.FileBridge.open(source[1], **bridge_kwargs)
        else:
            samples = source[1]
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
                samples = np.ascontiguousarray(samples, dtype=np.float32).tobytes()
            self._bridge = _decibri.FileBridge.buffer(
                samples, source[2], **bridge_kwargs
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
        chunk_end = chunk_start + self._chunk_samples(chunk) / self._sample_rate
        self._position = chunk_end
        if self._vad_enabled:
            self._vad.process_chunk(
                self._bridge.vad_probability, chunk_start, chunk_end
            )
        self._sequence += 1
        return chunk

    def _chunk_samples(self, chunk: SampleData) -> int:
        """Number of mono samples in one delivered chunk."""
        if isinstance(chunk, bytes):
            return len(chunk) // (2 if self._format == "int16" else 4)
        # numpy path: mono 1-D array; its length is the sample count.
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
