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

State properties (``is_open``, ``is_speaking``, ``vad_score``,
``is_playing``) are synchronous Python properties backed by Python-side
state tracking. The Rust async bridge exposes these as awaitables, but
awaiting from a property is not idiomatic Python and would surprise
callers who expect ``decibri.is_open`` to behave like sync ``Microphone``.
Phase 7.5 Item 10: the sync ``is_open`` / ``is_playing`` properties
now delegate to lock-free atomic mirrors on the Rust bridge, so they
report bridge truth (not stale Python-side cache) even when the Rust
side closes the stream itself.

Cross-binding compat: pure Python. The Rust core is unchanged. The sync
``Microphone`` and ``Speaker`` classes (in ``_classes.py``) are
unchanged and exist alongside this module.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import time
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, AsyncIterator, Union, cast

from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np

    # Phase 6: read can return bytes (default) or ndarray (as_ndarray=True);
    # write can accept either. Optional runtime numpy dependency.
    SampleData = Union[bytes, "np.ndarray[Any, Any]"]
else:
    SampleData = bytes

from decibri import _decibri, exceptions
from decibri._classes import Chunk, _VadStateMachine, _VALID_FORMATS, _VALID_MODES
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

    Cleanup and disconnect:
        Mid-stream device disconnect (USB unplug, default-device switch,
        driver error) is surfaced as a ``CaptureStreamClosed`` raised on
        the next ``await read()``. cpal detects the disconnect and
        closes the underlying stream within roughly 20ms; the bridge
        then reports the closed state on its next read attempt.

        Sibling-task cancellation works correctly here: cancelling a
        concurrent ``await stop()`` from another asyncio task while a
        ``read()`` is in flight is safe (the Tokio mutex serializes the
        operations cleanly). This is the recommended path when a sync
        ``Microphone`` would hit the 0.1.0 threaded-shutdown limitation.

    Resource cleanup:
        Always use ``async with AsyncMicrophone(...) as d:`` or call
        ``await d.stop()`` explicitly. ``AsyncMicrophone`` does NOT
        define ``__del__`` because finalizers cannot await; calling
        ``await self.stop()`` from a synchronous ``__del__`` would
        require a running event loop and would deadlock or warn under
        most conditions. The Rust side's pyo3 ``Drop`` impl will release
        the underlying bridge resources when the instance is collected,
        but the cleaner path is explicit ``await stop()`` or the async
        context manager.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        frames_per_buffer: int = 1600,
        dtype: str = "int16",
        device: int | str | None = None,
        vad: bool | str = False,
        vad_threshold: float | None = None,
        vad_holdoff_ms: int = 300,
        model_path: str | Path | None = None,
        as_ndarray: bool = False,
        ort_library_path: str | Path | None = None,
    ) -> None:
        """Construct an AsyncMicrophone audio capture instance.

        Constructor signature mirrors ``Microphone`` exactly; refer to
        ``help(Microphone)`` for the full per-parameter documentation
        (including the ORT dylib resolution priority chain on
        ``ort_library_path``).

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

        Notes
        -----
        ORT load is synchronous when called via this constructor. With
        ``vad='silero'`` the Silero ONNX runtime is loaded inside
        ``__init__`` itself, which blocks for roughly 100 to 500
        milliseconds depending on platform and disk cache. ``__init__``
        runs synchronously even on async classes, so this blocks the
        event loop in async contexts.

        For latency-sensitive event-loop hot paths use the
        :meth:`AsyncMicrophone.open` async factory classmethod added in
        Phase 9 (``mic = await AsyncMicrophone.open(vad='silero')``),
        which dispatches the synchronous construction to the default
        ThreadPoolExecutor and returns the constructed instance
        awaitably. The synchronous constructor remains supported for
        callers that construct outside an async context.
        """
        if dtype not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"dtype must be 'int16' or 'float32'; got {dtype!r}"
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
        if vad_holdoff_ms < 0:
            raise ValueError(
                f"vad_holdoff_ms must be non-negative milliseconds; got {vad_holdoff_ms}"
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

        # Wrapper-only rename (Phase 7.6 Item C2): public surface uses
        # `dtype`; bridge keeps `format` for cross-binding consistency.
        self._bridge = _decibri.AsyncMicrophoneBridge(
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
        )

        self._vad_enabled = vad_enabled
        self._vad = _VadStateMachine(
            mode=vad_mode,
            threshold=vad_threshold,
            holdoff_ms=vad_holdoff_ms,
            sample_format=dtype,
        )
        self._format = dtype
        # Wrapper-only rename (Phase 7.7 Item B4); bridge keeps the
        # original `numpy` name for cross-binding consistency per LD11.
        self._as_ndarray = as_ndarray
        # Phase 7.7 Item B1: chunk counter for read_with_metadata().
        self._sequence = 0
        # Phase 9 Item A4: capture construction parameters for __repr__.
        self._sample_rate = sample_rate
        self._channels = channels
        self._frames_per_buffer = frames_per_buffer
        self._device = device
        self._vad_arg = vad
    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Open and start the capture stream.

        Re-entry contract:
            Calling ``await start()`` after ``await stop()`` or
            ``await close()`` is supported and reconstructs the
            underlying audio stream cleanly. The ``AsyncMicrophone``
            instance is reusable across stop/start cycles. VAD state
            (``is_speaking``, ``vad_score``) resets to default values
            on each new ``await start()``. Re-entry after exiting an
            ``async with`` block is also supported (since
            ``__aexit__`` calls ``await stop()``).

            Calling ``await start()`` while already started raises
            ``AlreadyRunning``.
        """
        await self._bridge.start()

    async def stop(self) -> None:
        """Stop the capture stream and reset VAD state."""
        await self._bridge.stop()
        self._vad.reset()
        self._sequence = 0

    async def close(self) -> None:
        """Stop the capture stream. Permanent alias for ``stop()``.

        Provided for ergonomic parity with the asyncio / aiohttp /
        httpx convention and for use cases where ``close()`` reads
        more naturally than ``stop()``. The two methods are guaranteed
        to remain semantically equivalent across all decibri versions.
        """
        # Calls self.stop() rather than self._bridge.close() so the
        # wrapper-side cleanup (vad.reset() in stop()) runs. The
        # bridge-level AsyncMicrophoneBridge.close() exists for
        # symmetry with AsyncSpeakerBridge.close() and for advanced
        # direct-bridge users; the wrapper keeps its own routing here
        # to ensure VAD state is reset on every close.
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
        - When ``as_ndarray=False`` (default), returns ``bytes``.
        - When ``as_ndarray=True``, returns a ``numpy.ndarray`` with
          dtype matching the configured ``dtype`` and shape matching
          the channel count (1-D mono, 2-D ``(N, channels)``
          multi-channel).

        Use ``read_with_metadata()`` to receive a typed ``Chunk`` with
        ``.data``, ``.timestamp``, ``.sequence``, ``.is_speaking``, and
        ``.vad_score`` attributes. ``read()`` keeps its naked-data
        signature for backward compatibility.

        VAD state advances as a side effect when VAD is enabled
        (``vad="silero"`` or ``vad="energy"``). In ndarray
        mode with VAD enabled, the chunk is converted to bytes once via
        ``arr.tobytes()`` for the VAD state machine; the original
        ndarray is returned to the user. Phase 6 plan §3i.

        Cancellation: per the Phase 5 abort-immediately decision, cancelling
        this await raises ``CancelledError`` immediately. The spawn_blocking
        thread on the Rust side runs to completion; its result (if any) is
        dropped. The bridge state remains consistent for subsequent reads.

        Raises ``ImportError`` with a clear actionable message if
        ``as_ndarray=True`` is set but ``numpy`` is not installed (i.e.
        the user did not run ``pip install decibri[numpy]``).
        """
        try:
            chunk = await self._bridge.read(timeout_ms=timeout_ms)
        except ImportError as exc:
            if self._as_ndarray:
                raise ImportError(
                    "numpy is not installed. Install with: pip install decibri[numpy]"
                ) from exc
            raise
        if chunk is None:
            return None
        if self._vad_enabled:
            probability = await self._bridge.vad_probability
            if self._as_ndarray:
                # ndarray mode: convert chunk to bytes for the VAD
                # state machine (struct.unpack-based).
                vad_input: bytes = chunk.tobytes()  # type: ignore[union-attr]
            else:
                vad_input = cast(bytes, chunk)
            self._vad.process_chunk(vad_input, probability)
        self._sequence += 1
        return chunk

    async def read_with_metadata(self, timeout_ms: int | None = None) -> Chunk | None:
        """Async parallel of ``Microphone.read_with_metadata()``.

        Returns ``None`` on clean stream close; otherwise a frozen
        ``Chunk`` with ``.data``, ``.timestamp``, ``.sequence``,
        ``.is_speaking``, and ``.vad_score`` attributes. See
        ``Microphone.read_with_metadata`` for the full contract.
        """
        data = await self.read(timeout_ms=timeout_ms)
        if data is None:
            return None
        return Chunk(
            data=data,
            timestamp=time.monotonic(),
            sequence=self._sequence - 1,
            is_speaking=self.is_speaking,
            vad_score=self.vad_score,
        )

    async def aiter_with_metadata(self) -> AsyncIterator[Chunk]:
        """Async-yield ``Chunk`` objects until the stream closes cleanly.

        Async-generator wrapping ``await read_with_metadata()``: stops
        when the bridge returns ``None``. Use this in place of
        ``async for chunk in mic`` when you want metadata alongside the
        audio data.
        """
        while True:
            chunk = await self.read_with_metadata(timeout_ms=None)
            if chunk is None:
                return
            yield chunk

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

        Queries the Rust bridge directly via a lock-free atomic mirror
        (Phase 7.5 Item 10). Reports honestly even when the Rust side
        closes the stream itself (e.g., device disconnect or cpal driver
        error), unlike the prior Python-side cache.
        """
        return bool(self._bridge.is_open_sync)

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
    def vad_score(self) -> float:
        """Most recent VAD score in ``[0, 1]``. Mode-agnostic.

        In ``vad="silero"`` mode, returns the raw probability from the
        Silero model. In ``vad="energy"`` mode, returns the normalized
        RMS energy of the most recent chunk. Always 0.0 when
        ``vad=False``.

        The underlying bridge property is named ``vad_probability`` for
        cross-binding consistency; ``vad_score`` is the mode-agnostic
        wrapper-side name (it is not a probability in energy mode).
        """
        if not self._vad_enabled:
            return 0.0
        return self._vad.vad_score

    # -----------------------------------------------------------------------
    # Static methods
    # -----------------------------------------------------------------------

    # Note: no __del__ on AsyncMicrophone by design.
    # A finalizer cannot await; calling `await self.stop()` from __del__
    # would require a running event loop and deadlock or warn under most
    # conditions. The Rust pyo3 Drop on the underlying bridge handles
    # resource release at GC time. Consumers should always use
    # `async with AsyncMicrophone(...) as d:` or `await d.stop()`
    # explicitly. See the class docstring's "Resource cleanup" section.

    def __repr__(self) -> str:
        # Phase 9 Item A4. Mirrors Microphone.__repr__; is_open is
        # queried from the bridge's lock-free atomic mirror so the repr
        # reflects current bridge truth (LD-9-8).
        is_open: bool | str
        try:
            is_open = self.is_open
        except Exception:  # noqa: BLE001
            is_open = "?"
        return (
            f"AsyncMicrophone(sample_rate={self._sample_rate}, "
            f"channels={self._channels}, "
            f"dtype={self._format!r}, "
            f"frames_per_buffer={self._frames_per_buffer}, "
            f"device={self._device!r}, "
            f"vad={self._vad_arg!r}, "
            f"is_open={is_open})"
        )

    # -----------------------------------------------------------------------
    # Async factory (Phase 9 Item A6)
    # -----------------------------------------------------------------------

    @classmethod
    async def open(cls, **kwargs: Any) -> "AsyncMicrophone":
        """Construct an ``AsyncMicrophone`` without blocking the event loop.

        Use this instead of the constructor in async code, especially when
        ``vad="silero"`` is enabled. The synchronous constructor performs
        ORT model loading inline (100 to 500 ms for Silero VAD on a cold
        cache); calling it from an async context blocks the event loop for
        the duration of the load, which can cause dropped websocket
        frames, late timer callbacks, and UI jitter in voice-AI pipelines.

        This classmethod dispatches the synchronous constructor to the
        default ThreadPoolExecutor via ``loop.run_in_executor``, returning
        the constructed instance awaitably. The synchronous constructor
        remains available for backward compatibility and for callers that
        construct outside an async context.

        Parameters mirror ``AsyncMicrophone.__init__`` exactly; pass the
        same keyword arguments.

        Example::

            mic = await AsyncMicrophone.open(vad="silero")
            async with mic:
                async for chunk in mic:
                    await process(chunk)

        Phase 9 LD-9-3 (pulled forward from 0.2.0 backlog) and LD-9-7
        (default ThreadPoolExecutor; users wanting a custom executor can
        construct synchronously inside their own thread).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: cls(**kwargs))

    @staticmethod
    async def input_devices() -> list[DeviceInfo]:
        """List available audio input devices."""
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

    Resource cleanup:
        Always use ``async with AsyncSpeaker(...) as o:`` or call
        ``await o.stop()`` explicitly. ``AsyncSpeaker`` does NOT define
        ``__del__`` because finalizers cannot await; calling
        ``await self.stop()`` from a synchronous ``__del__`` would
        require a running event loop and deadlock or warn under most
        conditions. The Rust side's pyo3 ``Drop`` impl will release the
        underlying bridge resources when the instance is collected, but
        the cleaner path is explicit ``await stop()`` or the async
        context manager.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        device: int | str | None = None,
    ) -> None:
        """Construct an AsyncSpeaker audio output instance.

        Parameters
        ----------
        sample_rate : int, optional
            Output sample rate in Hz. Default 16000 (matches the
            cloud-STT capture convention used by ``AsyncMicrophone``).
            For playback of OpenAI Realtime audio use 24000.
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
            ``AsyncSpeaker.output_devices()`` or a substring of the device
            name. ``AsyncSpeaker`` does not load ONNX Runtime, so there
            is no ``ort_library_path`` parameter (output never invokes
            VAD).
        """
        if dtype not in _VALID_FORMATS:
            raise exceptions.InvalidFormat(
                f"dtype must be 'int16' or 'float32'; got {dtype!r}"
            )
        # Wrapper-only rename (Phase 7.6 Item C2): public surface uses
        # `dtype`; bridge keeps `format` for cross-binding consistency.
        self._bridge = _decibri.AsyncSpeakerBridge(
            sample_rate=sample_rate,
            channels=channels,
            format=dtype,
            device=device,
        )
        # Phase 9 Item A4: capture construction parameters for __repr__.
        self._sample_rate = sample_rate
        self._channels = channels
        self._format = dtype
        self._device = device

    async def start(self) -> None:
        """Open and start the output stream.

        Re-entry contract:
            Calling ``await start()`` after ``await stop()`` or
            ``await close()`` is supported and reconstructs the
            underlying output stream cleanly. The ``AsyncSpeaker``
            instance is reusable across stop/start cycles. Re-entry
            after exiting an ``async with`` block is also supported
            (since ``__aexit__`` calls ``await stop()``).

            Calling ``await start()`` while already started raises
            ``AlreadyRunning``.
        """
        await self._bridge.start()

    async def stop(self) -> None:
        """Stop the output stream."""
        await self._bridge.stop()

    async def close(self) -> None:
        """Stop the output stream. Permanent alias for ``stop()``.

        The bridge-level ``close()`` is itself a literal alias for
        ``stop()`` (see lib.rs). The wrapper-side ``close()`` exists
        for ergonomic parity with the asyncio / aiohttp / httpx
        convention. ``close()`` and ``stop()`` are guaranteed to
        remain semantically equivalent across all decibri versions.
        """
        await self._bridge.close()

    async def write(self, samples: SampleData) -> None:
        """Write a chunk of audio samples to the output buffer.

        Phase 6: accepts either ``bytes`` or a ``numpy.ndarray`` with
        dtype matching the configured ``dtype`` (np.int16 or np.float32).
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

        Queries the Rust bridge directly via a lock-free atomic mirror
        (Phase 7.5 Item 10 sibling fix). Reports honestly even when the
        Rust side closes the stream itself, unlike the prior Python-side
        cache.
        """
        return bool(self._bridge.is_playing_sync)

    # Note: no __del__ on AsyncSpeaker by design.
    # A finalizer cannot await; calling `await self.stop()` from __del__
    # would require a running event loop and deadlock or warn under most
    # conditions. The Rust pyo3 Drop on the underlying bridge handles
    # resource release at GC time. Consumers should always use
    # `async with AsyncSpeaker(...) as o:` or `await o.stop()` explicitly.
    # See the class docstring's "Resource cleanup" section.

    def __repr__(self) -> str:
        # Phase 9 Item A4. Mirrors Speaker.__repr__; is_playing reflects
        # the bridge's atomic mirror so the repr is honest about current
        # state (LD-9-8).
        is_playing: bool | str
        try:
            is_playing = self.is_playing
        except Exception:  # noqa: BLE001
            is_playing = "?"
        return (
            f"AsyncSpeaker(sample_rate={self._sample_rate}, "
            f"channels={self._channels}, "
            f"dtype={self._format!r}, "
            f"device={self._device!r}, "
            f"is_playing={is_playing})"
        )

    @classmethod
    async def open(cls, **kwargs: Any) -> "AsyncSpeaker":
        """Construct an ``AsyncSpeaker`` without blocking the event loop.

        Symmetric with :meth:`AsyncMicrophone.open`. ``Speaker`` does not
        load ORT, so the practical event-loop blocking risk is smaller
        here, but ``open()`` is provided for API parity so async code can
        consistently use the factory pattern across both classes.

        Parameters mirror ``AsyncSpeaker.__init__`` exactly.

        Phase 9 LD-9-3, LD-9-7.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: cls(**kwargs))

    @staticmethod
    async def output_devices() -> list[OutputDeviceInfo]:
        """List available audio output devices."""
        return await _decibri.AsyncSpeakerBridge.devices()
