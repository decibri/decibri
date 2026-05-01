"""High-level Python wrapper for decibri audio capture with VAD policy.

This module ships the consumer-facing `Microphone` class. The class wraps the
native `_decibri.MicrophoneBridge` pyclass (Rust binding shipped in Commit 2)
and adds wrapper-layer VAD policy: threshold application, holdoff state
machine, and mode dispatch (Silero passthrough vs energy RMS computation).

Architectural notes:

- The Rust bridge is a thin inference primitive. It exposes raw Silero
  probability via `vad_probability` and stores `vad_holdoff_ms` as inert
  state. All policy (threshold comparison, holdoff timing, mode dispatch,
  energy RMS) lives here. This mirrors the Node binding's JS-side pattern.

- `is_speaking` reflects time-based holdoff semantics. Above-threshold
  chunks set the speaking state and clear any pending silence timer.
  Below-threshold chunks while speaking start a silence timer; the timer
  expires after `vad_holdoff_ms` elapsed real time, at which point
  `is_speaking` flips to False. Timer expiry is checked on every property
  access using `time.monotonic()`, so consumers who pause iteration still
  observe correct state when they next read the property.

- `vad_probability` is mode-aware. In Silero mode it passes through the
  bridge's raw probability. In energy mode it returns the most recent
  chunk's RMS, computed inside `read()` and cached.

- The Phase 2 surface is sync only. Async iteration and event callbacks
  ship in Phase 3 alongside `pyo3-async-runtimes` integration (Q7).
"""

from __future__ import annotations

import importlib.resources
import math
import struct
import time
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterator, Union, cast

from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np

    # Phase 6: read can return bytes (default) or ndarray (numpy=True);
    # write can accept either. The runtime numpy dependency is optional
    # (pip install decibri[numpy]); using TYPE_CHECKING keeps the import
    # cost out of the default-install path while preserving mypy
    # narrowing for users who set numpy=True.
    SampleData = Union[bytes, "np.ndarray[Any, Any]"]
else:
    SampleData = bytes

from decibri import _decibri, exceptions
from decibri._decibri import DeviceInfo, OutputDeviceInfo, VersionInfo

__all__ = ["Microphone", "Speaker", "DeviceInfo", "OutputDeviceInfo", "VersionInfo"]


# ---------------------------------------------------------------------------
# VAD policy state machine.
#
# Encapsulates the threshold + holdoff + mode-dispatch logic that the Rust
# bridge does NOT do. Three responsibilities:
#
#   1. Compute or pass through the per-chunk probability value.
#      - Silero mode: passthrough from bridge.
#      - Energy mode: wrapper-computed RMS over the chunk.
#
#   2. Apply the user's threshold to that value.
#
#   3. Run a holdoff state machine on the threshold result, with time-based
#      expiry of the silence timer.
#
# Construction is driven by Microphone.__init__; consumers don't touch this
# class directly.
# ---------------------------------------------------------------------------


class _VadStateMachine:
    """Wrapper-layer VAD policy. Pure Python; no native interaction."""

    __slots__ = (
        "_mode",
        "_threshold",
        "_holdoff_seconds",
        "_format",
        "_is_speaking",
        "_silence_started_at",
        "_last_probability",
    )

    def __init__(
        self,
        mode: str,
        threshold: float,
        holdoff_ms: int,
        sample_format: str,
    ) -> None:
        self._mode = mode
        self._threshold = float(threshold)
        self._holdoff_seconds = holdoff_ms / 1000.0
        self._format = sample_format
        self._is_speaking = False
        self._silence_started_at: float | None = None
        self._last_probability = 0.0

    def reset(self) -> None:
        """Clear all transient state. Called on stop()."""
        self._is_speaking = False
        self._silence_started_at = None
        self._last_probability = 0.0

    def process_chunk(self, chunk_bytes: bytes, bridge_probability: float) -> None:
        """Update VAD state from one chunk. Called inside Microphone.read()."""
        if self._mode == "silero":
            probability = bridge_probability
        elif self._mode == "energy":
            probability = _compute_rms(chunk_bytes, self._format)
        else:
            # Should be unreachable; constructor validates mode.
            probability = 0.0

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
    def vad_probability(self) -> float:
        return self._last_probability


# ---------------------------------------------------------------------------
# RMS computation for energy VAD.
#
# Mirrors Node's _processVadEnergy: int16 samples normalized to [-1, 1] before
# squaring; float32 used as-is. Computed in pure Python via struct.unpack;
# Phase 3 can profile and optimize with NumPy if hot path warrants it.
# ---------------------------------------------------------------------------


def _compute_rms(chunk_bytes: bytes, sample_format: str) -> float:
    """Compute RMS over a chunk. Returns a value in [0, 1] for normalized formats."""
    if not chunk_bytes:
        return 0.0

    if sample_format == "int16":
        sample_count = len(chunk_bytes) // 2
        if sample_count == 0:
            return 0.0
        samples = struct.unpack(f"<{sample_count}h", chunk_bytes)
        sum_sq = 0.0
        for s in samples:
            normalized = s / 32768.0
            sum_sq += normalized * normalized
        return math.sqrt(sum_sq / sample_count)

    if sample_format == "float32":
        sample_count = len(chunk_bytes) // 4
        if sample_count == 0:
            return 0.0
        samples = struct.unpack(f"<{sample_count}f", chunk_bytes)
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / sample_count)

    # Unreachable; format is validated at Microphone construction time.
    return 0.0


# ---------------------------------------------------------------------------
# Microphone: consumer-facing audio capture class.
# ---------------------------------------------------------------------------


_VALID_MODES = frozenset({"silero", "energy"})
_VALID_FORMATS = frozenset({"int16", "float32"})


class Microphone:
    """Audio capture with VAD policy.

    Example:
        with Microphone(sample_rate=16000, channels=1, frames_per_buffer=1600) as d:
            for chunk in d:
                if d.is_speaking:
                    process(chunk)

    Construction does NOT start capture. Use the context manager or call
    `start()` explicitly. Iteration without an active capture raises
    `_decibri.CaptureStreamClosed`.

    The VAD parameters (`vad`, `vad_mode`, `vad_threshold`, `vad_holdoff`,
    `model_path`, `ort_library_path`) configure both the bridge and the
    wrapper-layer state machine. With `vad=False`, `vad_probability` returns
    0.0 and `is_speaking` returns False unconditionally.

    The wrapper is sync only in Phase 2. Async iteration and speech-event
    callbacks ship in Phase 3.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 1600,
        format: str = "int16",
        device: int | str | None = None,
        vad: bool = False,
        vad_threshold: float | None = None,
        vad_holdoff: int = 300,
        vad_mode: str = "energy",
        model_path: str | Path | None = None,
        numpy: bool = False,
        ort_library_path: str | Path | None = None,
    ) -> None:
        """Construct a Microphone audio capture instance.

        Most parameters control capture behaviour and are documented at
        the class level. The argument that benefits from per-arg detail
        is ``ort_library_path``, since its resolution involves a
        priority chain that is invisible at the call site.

        Parameters
        ----------
        ort_library_path : str | Path | None, optional
            Path to the ONNX Runtime dynamic library used by Silero VAD.
            Only consulted when ``vad=True`` and ``vad_mode='silero'``;
            energy-mode VAD and ``vad=False`` do not load ORT.

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
        if format not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"format must be 'int16' or 'float32'; got {format!r}"
            )
        if vad_mode not in _VALID_MODES:
            raise ValueError(
                f"vad_mode must be 'silero' or 'energy'; got {vad_mode!r}"
            )

        # Mode-dependent threshold default mirroring Node:
        # 0.5 for silero, 0.01 for energy.
        if vad_threshold is None:
            vad_threshold = 0.5 if vad_mode == "silero" else 0.01
        if not 0.0 <= vad_threshold <= 1.0:
            raise ValueError(
                f"vad_threshold must be in [0, 1]; got {vad_threshold}"
            )
        if vad_holdoff < 0:
            raise ValueError(
                f"vad_holdoff must be non-negative; got {vad_holdoff}"
            )

        # Resolve model_path to an absolute string for the bridge.
        # User-supplied path takes precedence. When Silero VAD is requested
        # without an explicit path, fall back to the model bundled with the
        # wheel via importlib.resources. Mirrors the Node binding's pattern
        # (decibri.js:159-171) which resolves __dirname/../models/silero_vad.onnx
        # before passing to the native bridge.
        resolved_model_path: str | None = None
        if model_path is not None:
            resolved_model_path = str(Path(model_path))
        elif vad and vad_mode == "silero":
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

        # Resolve the ORT dylib path via the four-arm priority order in
        # _ort_resolver. Only consulted when Silero VAD is enabled; energy
        # mode and vad=False never load ORT, so the resolver call (and its
        # bundled-dylib lookup) is skipped to avoid the import-time cost.
        # See _ort_resolver.resolve_ort_dylib_path for the priority order.
        resolved_ort_path: str | None = None
        if vad and vad_mode == "silero":
            from decibri._ort_resolver import resolve_ort_dylib_path

            resolved_ort_path = resolve_ort_dylib_path(ort_library_path)
        elif ort_library_path is not None:
            # Preserve passthrough for explicit caller paths even when ORT
            # is not loaded, so tests and tooling that introspect the
            # bridge state see the user's intent.
            resolved_ort_path = str(Path(ort_library_path))

        self._bridge = _decibri.MicrophoneBridge(
            sample_rate=sample_rate,
            channels=channels,
            frames_per_buffer=frames_per_buffer,
            format=format,
            device=device,
            vad=vad,
            vad_threshold=vad_threshold,
            vad_mode=vad_mode,
            vad_holdoff=vad_holdoff,
            model_path=resolved_model_path,
            numpy=numpy,
            ort_library_path=resolved_ort_path,
        )

        self._vad_enabled = vad
        self._vad = _VadStateMachine(
            mode=vad_mode,
            threshold=vad_threshold,
            holdoff_ms=vad_holdoff,
            sample_format=format,
        )
        self._format = format
        # Phase 6: store the numpy flag so read() can branch its return
        # type (bytes vs ndarray) without re-querying the bridge each
        # call. The bridge also stores it; both are kept in sync because
        # the constructor passes the same value to both layers.
        self._numpy = numpy

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Open and start the capture stream."""
        self._bridge.start()

    def stop(self) -> None:
        """Stop the capture stream and reset VAD state."""
        self._bridge.stop()
        self._vad.reset()

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
        - When ``numpy=False`` (default), returns ``bytes`` (Phase 2 wire
          format).
        - When ``numpy=True``, returns a ``numpy.ndarray`` with dtype
          matching the configured ``format`` (np.int16 or np.float32) and
          shape matching the channel count (1-D for mono, 2-D
          ``(N, channels)`` for multi-channel).

        VAD state advances as a side effect when ``vad=True``. Consumers
        should check ``is_speaking`` after each read. In numpy mode with
        VAD enabled, the chunk is converted to bytes once via
        ``arr.tobytes()`` for the VAD state machine (the existing RMS
        helper expects bytes); the original ndarray is returned to the
        user. The conversion cost is microseconds for typical chunk
        sizes.

        Raises ``ImportError`` with a clear actionable message if
        ``numpy=True`` is set but the optional ``numpy`` extra is not
        installed (i.e. the user did not run ``pip install decibri[numpy]``).
        """
        try:
            chunk = self._bridge.read(timeout_ms=timeout_ms)
        except ImportError as exc:
            if self._numpy:
                raise ImportError(
                    "numpy is not installed. Install with: pip install decibri[numpy]"
                ) from exc
            raise
        if chunk is None:
            return None
        if self._vad_enabled:
            # The VAD state machine's _compute_rms uses struct.unpack on
            # bytes. In numpy mode, the bridge already returned an
            # ndarray; convert to bytes once for the VAD pass via
            # arr.tobytes(). Cheap (~microseconds) for typical chunk
            # sizes; cleaner than rewriting _compute_rms to handle both
            # input types. Phase 6 plan §3i. The cast to bytes is for
            # mypy; at runtime the branch matches the actual type.
            if self._numpy:
                vad_input: bytes = chunk.tobytes()  # type: ignore[union-attr]
            else:
                vad_input = cast(bytes, chunk)
            self._vad.process_chunk(vad_input, self._bridge.vad_probability)
        return chunk

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
    def vad_probability(self) -> float:
        """Most recent VAD probability.

        In Silero mode, returns the raw probability from the model.
        In energy mode, returns the RMS of the most recent chunk.
        Always 0.0 when vad=False.
        """
        if not self._vad_enabled:
            return 0.0
        return self._vad.vad_probability

    # -----------------------------------------------------------------------
    # Static methods
    # -----------------------------------------------------------------------

    @staticmethod
    def devices() -> list[DeviceInfo]:
        """List available input devices."""
        return _decibri.MicrophoneBridge.devices()

    @staticmethod
    def version() -> VersionInfo:
        """Return version info: decibri Rust core, cpal, and binding wheel."""
        return _decibri.MicrophoneBridge.version()


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
        format: str = "int16",
        device: int | str | None = None,
    ) -> None:
        if format not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"format must be 'int16' or 'float32'; got {format!r}"
            )
        self._bridge = _decibri.SpeakerBridge(
            sample_rate=sample_rate,
            channels=channels,
            format=format,
            device=device,
        )

    def start(self) -> None:
        self._bridge.start()

    def stop(self) -> None:
        self._bridge.stop()

    def close(self) -> None:
        self._bridge.close()

    def write(self, samples: SampleData) -> None:
        """Write a chunk to the output stream.

        Phase 6: accepts either ``bytes`` (Phase 2 wire format) or a
        ``numpy.ndarray`` with dtype matching the configured ``format``
        (np.int16 for ``format='int16'``, np.float32 for
        ``format='float32'``). Multi-channel ndarrays use shape
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
    def devices() -> list[OutputDeviceInfo]:
        """List available output devices."""
        return _decibri.SpeakerBridge.devices()
