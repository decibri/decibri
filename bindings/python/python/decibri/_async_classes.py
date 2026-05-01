"""Async Python wrappers for decibri: AsyncMicrophone and AsyncSpeaker.

Phase 5 of the Python Integration Project. These classes mirror the sync
``Microphone`` and ``Speaker`` surfaces method-for-method, with
``async def`` semantics, async context manager support
(``__aenter__`` / ``__aexit__``), and (for capture only) async iterator
support (``__aiter__`` / ``__anext__``).

Implementation: each method proxies to the underlying Rust async pyclass
(``AsyncMicrophoneBridge`` / ``AsyncSpeakerBridge``) which dispatches blocking
work to a Tokio worker thread via ``spawn_blocking``. The Tokio runtime
is lazily initialized; no explicit init call is needed.

Cancellation: ``asyncio.CancelledError`` propagates through
``pyo3-async-runtimes`` to the underlying Rust future. Per the Phase 5
locked decision (abort-immediately), cancellation is non-recoverable for
the in-flight chunk: any cpal data captured in the spawn_blocking thread
between the Python-side cancellation and the thread's natural completion
is dropped. Per the Phase 5 plan Risk 2, the spawn_blocking OS thread
itself does not cooperatively cancel; the Python coroutine sees
``CancelledError`` immediately while the Rust thread runs to completion.

State properties (``is_open``, ``is_speaking``, ``vad_probability``,
``is_playing``) are synchronous Python properties backed by Python-side
state tracking. The Rust async bridge exposes these as awaitables, but
awaiting from a property is not idiomatic Python and would surprise
callers who expect ``decibri.is_open`` to behave like sync ``Microphone``.
The wrapper updates ``_is_open`` after ``start()`` and ``stop()`` return.

Cross-binding compat: pure Python. The Rust core is unchanged. The sync
``Microphone`` and ``Speaker`` classes (in ``_classes.py``) are
unchanged and exist alongside this module.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Union, cast

from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np

    # Phase 6: read can return bytes (default) or ndarray (numpy=True);
    # write can accept either. Optional runtime numpy dependency.
    SampleData = Union[bytes, "np.ndarray[Any, Any]"]
else:
    SampleData = bytes

from decibri import _decibri, exceptions
from decibri._classes import _VadStateMachine, _VALID_FORMATS, _VALID_MODES
from decibri._decibri import DeviceInfo, OutputDeviceInfo, VersionInfo

__all__ = ["AsyncMicrophone", "AsyncSpeaker"]


# ---------------------------------------------------------------------------
# AsyncMicrophone: async audio capture; mirror of decibri.Microphone.
# ---------------------------------------------------------------------------


class AsyncMicrophone:
    """Async audio capture; mirror of ``decibri.Microphone``.

    Use ``async def`` methods: ``await async_decibri.start()``,
    ``chunk = await async_decibri.read()``, ``await async_decibri.stop()``.

    Supports ``async with AsyncMicrophone() as d:`` for automatic start/stop.
    Supports ``async for chunk in async_decibri:`` for iterator-style
    consumption (capture only; ``AsyncSpeaker`` does not iterate).

    Constructor signature matches sync ``Microphone`` exactly. See ``Microphone``
    for parameter documentation including the ``ort_library_path`` priority
    order; this class is a 1:1 mapping with async semantics.

    Cancellation: any awaited method that is cancelled (via
    ``asyncio.CancelledError``, ``asyncio.wait_for``, or explicit
    ``task.cancel()``) propagates the cancellation to the Rust side. The
    Python coroutine raises immediately; the spawn_blocking OS thread on
    the Rust side runs to completion with its result dropped. Subsequent
    operations on the same instance see consistent bridge state.

    Concurrency: concurrent calls on the same instance serialize via the
    Rust-side ``tokio::sync::Mutex``. This is a behavioural characteristic
    of the architecture, not advertised concurrency.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 1600,
        format: str = "int16",
        device: int | str | None = None,
        vad: bool | str = False,
        vad_threshold: float | None = None,
        vad_holdoff: int = 300,
        model_path: str | Path | None = None,
        numpy: bool = False,
        ort_library_path: str | Path | None = None,
    ) -> None:
        if format not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"format must be 'int16' or 'float32'; got {format!r}"
            )

        # Phase 7.5: collapse the legacy two-flag pattern (vad=True,
        # vad_mode="silero") into a single union-typed parameter. Mirrors
        # Microphone exactly; see _classes.py for the full rationale.
        vad_enabled: bool
        vad_mode: str
        if vad is False:
            vad_enabled = False
            vad_mode = "energy"  # inert placeholder; bridge ignores when disabled
        elif vad is True:
            raise ValueError(
                "vad=True is no longer supported. "
                "Specify the mode explicitly: vad='silero' or vad='energy'."
            )
        elif vad in _VALID_MODES:
            vad_enabled = True
            vad_mode = vad
        else:
            raise ValueError(
                f"Invalid vad value: {vad!r}. "
                "Expected False, 'silero', or 'energy'."
            )

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

        # Resolve the Silero ONNX model path. Same logic as sync Microphone:
        # user-supplied path wins; otherwise fall back to the bundled model
        # via importlib.resources when Silero VAD is requested.
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

        # Resolve the ORT dylib path via the same four-arm priority order
        # the sync wrapper uses (see _ort_resolver.resolve_ort_dylib_path).
        # Lazy import: only loaded when Silero VAD is enabled.
        resolved_ort_path: str | None = None
        if vad_enabled and vad_mode == "silero":
            from decibri._ort_resolver import resolve_ort_dylib_path

            resolved_ort_path = resolve_ort_dylib_path(ort_library_path)
        elif ort_library_path is not None:
            resolved_ort_path = str(Path(ort_library_path))

        self._bridge = _decibri.AsyncMicrophoneBridge(
            sample_rate=sample_rate,
            channels=channels,
            frames_per_buffer=frames_per_buffer,
            format=format,
            device=device,
            vad=vad_enabled,
            vad_threshold=vad_threshold,
            vad_mode=vad_mode,
            vad_holdoff=vad_holdoff,
            model_path=resolved_model_path,
            numpy=numpy,
            ort_library_path=resolved_ort_path,
        )

        self._vad_enabled = vad_enabled
        self._vad = _VadStateMachine(
            mode=vad_mode,
            threshold=vad_threshold,
            holdoff_ms=vad_holdoff,
            sample_format=format,
        )
        self._format = format
        # Phase 6: store the numpy flag so read() can branch its return
        # type and the VAD path can convert ndarray to bytes for the
        # state machine. Bridge stores the same flag; both kept in sync.
        self._numpy = numpy
        # Python-side state tracking for the synchronous is_open property.
        # The Rust async bridge's is_open getter returns an awaitable; we
        # update this flag after start/stop so callers can read state
        # without an await.
        self._is_open = False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Open and start the capture stream."""
        await self._bridge.start()
        self._is_open = True

    async def stop(self) -> None:
        """Stop the capture stream and reset VAD state."""
        await self._bridge.stop()
        self._is_open = False
        self._vad.reset()

    async def close(self) -> None:
        """Close the microphone and release resources.

        Currently an alias for ``stop()``; provided for API symmetry
        with ``AsyncSpeaker.close()``. See ``Microphone.close()`` for
        the full rationale.
        """
        await self.stop()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    # -----------------------------------------------------------------------
    # Read surface
    # -----------------------------------------------------------------------

    async def read(self, timeout_ms: int | None = None) -> SampleData | None:
        """Read one chunk. Returns the chunk, or None if the stream closed.

        Return type:
        - When ``numpy=False`` (default), returns ``bytes``.
        - When ``numpy=True``, returns a ``numpy.ndarray`` with dtype
          matching the configured ``format`` and shape matching the
          channel count (1-D mono, 2-D ``(N, channels)`` multi-channel).

        VAD state advances as a side effect when VAD is enabled
        (``vad="silero"`` or ``vad="energy"``). In numpy
        mode with VAD enabled, the chunk is converted to bytes once via
        ``arr.tobytes()`` for the VAD state machine; the original
        ndarray is returned to the user. Phase 6 plan §3i.

        Cancellation: per the Phase 5 abort-immediately decision, cancelling
        this await raises ``CancelledError`` immediately. The spawn_blocking
        thread on the Rust side runs to completion; its result (if any) is
        dropped. The bridge state remains consistent for subsequent reads.

        Raises ``ImportError`` with a clear actionable message if
        ``numpy=True`` is set but ``numpy`` is not installed (i.e. the
        user did not run ``pip install decibri[numpy]``).
        """
        try:
            chunk = await self._bridge.read(timeout_ms=timeout_ms)
        except ImportError as exc:
            if self._numpy:
                raise ImportError(
                    "numpy is not installed. Install with: pip install decibri[numpy]"
                ) from exc
            raise
        if chunk is None:
            return None
        if self._vad_enabled:
            probability = await self._bridge.vad_probability
            if self._numpy:
                vad_input: bytes = chunk.tobytes()  # type: ignore[union-attr]
            else:
                vad_input = cast(bytes, chunk)
            self._vad.process_chunk(vad_input, probability)
        return chunk

    def __aiter__(self) -> AsyncIterator[SampleData]:
        return self

    async def __anext__(self) -> SampleData:
        chunk = await self.read(timeout_ms=None)
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    # -----------------------------------------------------------------------
    # State properties (synchronous; backed by Python-side tracking)
    # -----------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True if the capture stream is currently running.

        Synchronous property backed by Python-side state tracking. Set on
        ``await start()`` completion, cleared on ``await stop()`` completion.
        """
        return self._is_open

    @property
    def is_speaking(self) -> bool:
        """True if VAD currently considers the user to be speaking.

        Reflects the wrapper-layer state machine: above-threshold detection
        plus holdoff grace period. Always False when ``vad=False``.

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
        Always 0.0 when ``vad=False``.
        """
        if not self._vad_enabled:
            return 0.0
        return self._vad.vad_probability

    # -----------------------------------------------------------------------
    # Static methods
    # -----------------------------------------------------------------------

    @staticmethod
    async def devices() -> list[DeviceInfo]:
        """List available input devices."""
        return await _decibri.AsyncMicrophoneBridge.devices()

    @staticmethod
    def version() -> VersionInfo:
        """Return version info: decibri Rust core, cpal, and binding wheel.

        Synchronous because it reads compile-time constants only; no I/O.
        Intentionally not async to avoid forcing callers to await for what
        is effectively a metadata lookup. Reuses the sync ``MicrophoneBridge``
        static method directly; the async bridge's ``version()`` wraps the
        same data in a coroutine for uniformity but offers no benefit for
        this purely-compile-time lookup.
        """
        return _decibri.MicrophoneBridge.version()


# ---------------------------------------------------------------------------
# AsyncSpeaker: async audio output; mirror of Speaker.
# ---------------------------------------------------------------------------


class AsyncSpeaker:
    """Async audio output; mirror of ``decibri.Speaker``.

    Use ``async def`` methods: ``await async_output.start()``,
    ``await async_output.write(samples)``, ``await async_output.drain()``,
    ``await async_output.stop()``.

    Supports ``async with AsyncSpeaker() as o:`` for automatic
    start/stop. Does NOT implement async iterator protocol (output is
    push-only; you write to it, you do not iterate over it). This mirrors
    the sync ``Speaker``, which also has no iterator protocol.

    Cancellation note for ``drain()``: cancelling a ``await drain()`` call
    raises ``CancelledError`` immediately, but the audio continues to play
    until the cpal output buffer empties on the callback's own schedule.
    The bridge's drain state may be inconsistent if a cancelled drain is
    immediately followed by another write/drain cycle. This is per the
    Phase 5 plan Risk 2 (spawn_blocking does not cooperatively cancel OS
    threads); for production use, complete drains before initiating new
    writes.
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
        self._bridge = _decibri.AsyncSpeakerBridge(
            sample_rate=sample_rate,
            channels=channels,
            format=format,
            device=device,
        )
        # Python-side state tracking for the synchronous is_playing property.
        self._is_playing = False

    async def start(self) -> None:
        """Open and start the output stream."""
        await self._bridge.start()
        self._is_playing = True

    async def stop(self) -> None:
        """Stop the output stream."""
        await self._bridge.stop()
        self._is_playing = False

    async def close(self) -> None:
        """Alias for ``stop()``; provided for symmetry with sync ``Speaker``."""
        await self._bridge.close()
        self._is_playing = False

    async def write(self, samples: SampleData) -> None:
        """Write a chunk of audio samples to the output buffer.

        Phase 6: accepts either ``bytes`` or a ``numpy.ndarray`` with
        dtype matching the configured ``format`` (np.int16 or np.float32).
        Multi-channel ndarrays use shape ``(N, channels)`` (interleaved).
        Output bridges duck-type the input on each call.

        Raises ``TypeError`` on dtype mismatch or unsupported input type.
        """
        await self._bridge.write(samples)

    async def drain(self) -> None:
        """Block until all queued samples have been played.

        Cancellation note: per Phase 5 plan Risk 2, cancelling this await
        raises ``CancelledError`` immediately, but the audio continues to
        play until the cpal output buffer empties on the callback's own
        schedule. See the class docstring for details.
        """
        await self._bridge.drain()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()

    @property
    def is_playing(self) -> bool:
        """True if the output stream is currently running.

        Synchronous property backed by Python-side state tracking.
        """
        return self._is_playing

    @staticmethod
    async def devices() -> list[OutputDeviceInfo]:
        """List available output devices."""
        return await _decibri.AsyncSpeakerBridge.devices()
