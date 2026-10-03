"""Async Python wrappers for decibri: AsyncMicrophone, AsyncSpeaker and AsyncFile.

These classes mirror the sync ``Microphone``, ``Speaker`` and ``File``
surfaces method-for-method, with ``async def`` semantics, async context
manager support (``__aenter__`` / ``__aexit__``), and (for capture and the
offline source) async iterator support (``__aiter__`` / ``__anext__``).

Implementation: ``AsyncMicrophone`` and ``AsyncSpeaker`` hold the same
bridges as ``Microphone`` and ``Speaker`` and await each blocking call on
decibri's worker pool, a set of daemon threads shared by every instance and
every event loop. A call goes to an idle worker, or to a new worker when none
is idle, so a call that ends another, such as ``stop()`` while a ``read()`` or
a ``drain()`` waits, never waits behind it. The bridges release the GIL while
they wait, so the event loop keeps running. ``AsyncFile`` and the ``open()``
factories await their calls on the event loop's default executor.

Cancellation: cancelling an awaited call raises ``asyncio.CancelledError`` at
once. The bridge call it started keeps running on its worker until it
returns, and its result is discarded.

State properties (``is_open``, ``is_speaking``, ``vad_score``,
``is_playing``) are synchronous Python properties. ``is_open`` and
``is_playing`` read the bridge's own state, as the sync classes do;
``is_speaking`` and ``vad_score`` read the wrapper's VAD state.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import importlib.resources
import itertools
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Literal,
    ParamSpec,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    import numpy as np

    # read can return bytes (default) or ndarray (as_ndarray=True);
    # write can accept either. Optional runtime numpy dependency.
    SampleData = Union[bytes, "np.ndarray[Any, Any]"]
else:
    SampleData = bytes

from decibri import _decibri, exceptions
from decibri._classes import (
    Aec,
    AecMetrics,
    Chunk,
    Device,
    File,
    SaveReport,
    Vad,
    VadReport,
    _aec_metrics_from_raw,
    _BUNDLED_DENOISE_MODEL,
    _BUNDLED_VAD_MODEL,
    _DEFAULT_VAD_HOLDOFF_MS,
    _require_numpy,
    _VadStateMachine,
    _VALID_DENOISE_MODELS,
    _VALID_FORMATS,
    _VALID_HIGHPASS,
    _VALID_MODES,
)
from decibri._decibri import MicrophoneInfo, SpeakerInfo, VersionInfo

__all__ = ["AsyncMicrophone", "AsyncSpeaker", "AsyncFile"]

# Self types for the async context-manager entry methods below, each bound to
# its class. typing.Self is not available on Python 3.10, the oldest version
# the package supports.
_AsyncMicrophoneT = TypeVar("_AsyncMicrophoneT", bound="AsyncMicrophone")
_AsyncSpeakerT = TypeVar("_AsyncSpeakerT", bound="AsyncSpeaker")
_AsyncFileT = TypeVar("_AsyncFileT", bound="AsyncFile")

_P = ParamSpec("_P")
_R = TypeVar("_R")


# ---------------------------------------------------------------------------
# The worker pool: AsyncMicrophone and AsyncSpeaker await every blocking
# bridge call here.
# ---------------------------------------------------------------------------

# A worker idle for this many seconds exits.
_IDLE_SECONDS = 10.0

# A submitted call: the future it settles and the call itself.
_WorkItem = tuple["concurrent.futures.Future[Any]", Callable[[], Any]]


class _WorkerPool(concurrent.futures.Executor):
    """Runs each call at once on a daemon thread owned by decibri.

    A call goes to an idle worker, or to a new worker when none is idle, so
    no call waits behind another: a ``stop()`` reaches its bridge while a
    ``read()`` or a ``drain()`` on the same bridge still waits. A worker idle
    for ``_IDLE_SECONDS`` exits. Workers are daemon threads named
    ``decibri-worker-<n>``, so a call still running at interpreter exit does
    not hold the process open. Each future is marked running when its call is
    submitted, so cancelling the task that awaits it never stops the call;
    every submitted call runs to completion.

    One pool serves every instance and every event loop. In a child process
    created by ``os.fork()`` the pool starts with no workers, because the
    parent's worker threads are not copied into the child.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._idle: list[queue.SimpleQueue[_WorkItem]] = []
        self._numbers = itertools.count(1)

    def _forget_workers(self) -> None:
        """Drop every worker and replace the lock, in a forked child.

        The child has none of the parent's worker threads, so their inboxes
        would never be served, and the lock may have been held at the fork by
        a thread the child does not have. The next call starts a new worker.
        """
        self._lock = threading.Lock()
        self._idle = []

    def submit(
        self, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> concurrent.futures.Future[_R]:
        future: concurrent.futures.Future[_R] = concurrent.futures.Future()
        future.set_running_or_notify_cancel()
        item: _WorkItem = (future, functools.partial(fn, *args, **kwargs))
        with self._lock:
            inbox = self._idle.pop() if self._idle else None
            name = f"decibri-worker-{next(self._numbers)}" if inbox is None else ""
        if inbox is None:
            inbox = queue.SimpleQueue()
            inbox.put(item)
            threading.Thread(
                target=self._work, args=(inbox,), name=name, daemon=True
            ).start()
        else:
            inbox.put(item)
        return future

    def _work(self, inbox: queue.SimpleQueue[_WorkItem]) -> None:
        while True:
            try:
                future, call = inbox.get(timeout=_IDLE_SECONDS)
            except queue.Empty:
                with self._lock:
                    if inbox in self._idle:
                        self._idle.remove(inbox)
                        return
                # A call claimed this worker as the wait ran out; it is on
                # its way.
                future, call = inbox.get()
            try:
                result = call()
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
            else:
                future.set_result(result)
            # Drop the finished call before waiting, so the worker does not
            # keep its arguments alive.
            del future, call
            with self._lock:
                self._idle.append(inbox)


_POOL = _WorkerPool()

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_POOL._forget_workers)


async def _blocking(fn: Callable[..., _R], *args: Any) -> _R:
    """Await ``fn(*args)`` on decibri's worker pool."""
    return await asyncio.get_running_loop().run_in_executor(_POOL, fn, *args)


def _snapshot(samples: SampleData) -> SampleData:
    """A copy of a C-contiguous numpy array, and any other input unchanged.

    Taken on the calling task, so a write sends what the array held when the
    call began, even when the write then waits for the speaker or for room in
    its queue. Other inputs reach the bridge as given, so the bridge's own
    checks report them.
    """
    numpy = sys.modules.get("numpy")
    if numpy is None or not isinstance(samples, numpy.ndarray):
        return samples
    array: Any = samples
    return cast(SampleData, array.copy()) if array.flags.c_contiguous else samples


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
    ``task.cancel()``) raises at once. The bridge call it started keeps
    running on its worker thread until it returns, and its result is
    discarded: a cancelled ``read()`` consumes the chunk it was waiting
    for. Subsequent operations on the same instance see consistent bridge
    state.

    Concurrency: ``stop()`` and ``close()`` may be awaited from another
    task while a ``read()`` on the same instance waits. They do not wait for
    the read: the read returns the audio buffered before the stop, or raises
    ``MicrophoneStreamClosed`` as a read made after ``stop()`` does.

    Cleanup and disconnect:
        Mid-stream device disconnect (USB unplug, default-device switch,
        driver error) is surfaced as a ``DeviceFailed`` raised on the next
        ``await read()``, carrying the driver's own cause. cpal detects the
        disconnect and closes the underlying stream within roughly 20ms; the
        bridge then reports the closed state on its next read attempt.
        ``DeviceFailed`` derives from ``DecibriError``, not from
        ``MicrophoneStreamClosed``, so catch ``DecibriError`` (or
        ``DeviceFailed`` itself) to handle a disconnect. A deliberate
        ``stop()`` still raises ``MicrophoneStreamClosed``.

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
        device: int | str | Device | None = None,
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
        :meth:`AsyncMicrophone.open` async factory classmethod
        (``mic = await AsyncMicrophone.open(vad='silero')``),
        which dispatches the synchronous construction to the default
        ThreadPoolExecutor and returns the constructed instance
        awaitably. The synchronous constructor remains supported for
        callers that construct outside an async context.
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
        # threshold/holdoff policy values. Mirrors Microphone exactly; see
        # _classes.py for the full rationale. ``vad`` accepts False (disabled;
        # default), the "silero"/"energy" shorthand, or a Vad config object
        # (which self-validates its model/threshold/holdoff); vad=True is
        # rejected with a migration message.
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
        # 0.01 for energy.
        if vad_threshold is None:
            vad_threshold = 0.5 if vad_mode == "silero" else 0.01

        # Resolve the Silero ONNX model path. Same logic as sync Microphone:
        # user-supplied path wins; otherwise fall back to the bundled model
        # via importlib.resources when Silero VAD is requested.
        resolved_model_path: str | None = None
        if model_path is not None:
            resolved_model_path = str(Path(model_path))
            # A user-supplied path is checked at construction, as in the sync
            # Microphone, and only when Silero VAD reads it.
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

        # Validate and resolve denoise. Same logic as sync Microphone: a
        # closed-set model name resolves to the bundled ONNX file via
        # importlib.resources; absence leaves denoise off; an unknown name is a
        # clear ValueError.
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

        # Validate high-pass. Same logic as the sync Microphone: a numeric cutoff
        # in Hz selects a filter, absence leaves the high-pass off, an out-of-set
        # value is a clear ValueError. Pure DSP, so there is nothing to resolve,
        # only the closed-set check.
        if highpass is not None and highpass not in _VALID_HIGHPASS:
            raise ValueError(
                f"highpass must be one of: 80, 100; got {highpass!r}"
            )

        # Validate AGC. Same logic as the sync Microphone: an integer dBFS target
        # in [-40, -3] (typical -18); absence leaves it off. The core names this
        # failure, so the wrapper raises the core's class.
        if agc is not None and not -40 <= agc <= -3:
            raise exceptions.AgcTargetOutOfRange(f"agc must be in [-40, -3]; got {agc}")

        # Validate the limiter ceiling. Same logic as the sync Microphone: a
        # sample-peak ceiling in dBFS in [-3.0, 0.0] (typical -1.0); absence
        # leaves it off. The core names this failure, so the wrapper raises its class.
        if limiter is not None and not -3.0 <= limiter <= 0.0:
            raise exceptions.LimiterCeilingOutOfRange(
                f"limiter must be in [-3.0, 0.0]; got {limiter}"
            )

        # Validate the channel map's shape. Same logic as the sync Microphone:
        # a list of integers in the channel count's width, one entry per
        # delivered channel (the core names the length failure, so the wrapper
        # raises its class). Whether each entry exists on the device is the
        # core's check against the resolved device's report at start().
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

        # Decompose ``aec`` into the bridge's flat fields. Same logic as the
        # sync Microphone: None (off; default), a model-name shorthand such as
        # "tau", or an Aec config object (which self-validates its tuning
        # fields). The model NAME is not checked against a list here: the
        # canceller owns that set, so the bridge parses it and an unknown name
        # raises AecConfigInvalid carrying the canceller's own message.
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

        # Resolve the ORT dylib path via the same four-arm priority order
        # the sync wrapper uses (see _ort_resolver.resolve_ort_dylib_path).
        # Lazy import: only loaded when an ONNX stage (Silero VAD or denoise)
        # will run.
        resolved_ort_path: str | None = None
        if (vad_enabled and vad_mode == "silero") or denoise is not None:
            from decibri._ort_resolver import resolve_ort_dylib_path

            resolved_ort_path = resolve_ort_dylib_path(ort_library_path)
        elif ort_library_path is not None:
            resolved_ort_path = str(Path(ort_library_path))

        # Wrapper-only rename: public surface uses `dtype`; bridge keeps
        # `format` for cross-binding consistency.
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
        # Wrapper-only rename; the bridge keeps the original `numpy` name
        # for cross-binding consistency.
        self._as_ndarray = as_ndarray
        # Chunk counter for read_with_metadata().
        self._sequence = 0
        # Capture construction parameters for __repr__.
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
        await _blocking(self._bridge.start)

    async def stop(self) -> None:
        """Stop the capture stream and reset VAD state."""
        await _blocking(self._bridge.stop)
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
        # bridge-level MicrophoneBridge.close() exists for symmetry
        # with SpeakerBridge.close() and for advanced direct-bridge
        # users; the wrapper keeps its own routing here to ensure VAD
        # state is reset on every close.
        await self.stop()

    async def __aenter__(self: _AsyncMicrophoneT) -> _AsyncMicrophoneT:
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
          dtype matching the configured ``dtype`` and shape matching the
          channel count: 1-D at ``channels=1``, 2-D
          ``(frames, channels)`` above it.

        Use ``read_with_metadata()`` to receive a typed ``Chunk`` with
        ``.data``, ``.timestamp``, ``.sequence``, ``.is_speaking``, and
        ``.vad_score`` attributes. ``read()`` keeps its naked-data
        signature for backward compatibility.

        VAD state advances as a side effect when VAD is enabled
        (``vad="silero"`` or ``vad="energy"``). The score comes from the
        bridge (computed natively on the pre-enhancement signal for both
        modes), so the returned chunk is not inspected for VAD and the
        return type (bytes vs ndarray) does not affect it.

        Cancellation: cancelling this await raises ``CancelledError`` at
        once. The read keeps running on its worker thread until its chunk
        arrives or the stream stops, and that chunk is discarded. The
        bridge state remains consistent for subsequent reads.

        ``as_ndarray=True`` requires the optional ``numpy`` extra; a
        missing numpy raises ``ImportError`` at construction (install
        with ``pip install decibri[numpy]``).
        """
        try:
            chunk = await _blocking(self._bridge.read, timeout_ms)
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
        """True once ``start()`` has opened the capture stream, until
        ``stop()`` or ``close()``.

        Reads the bridge's own state, as ``Microphone.is_open`` does, so it
        answers while a ``read()`` waits.
        """
        return self._bridge.is_open

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

    # -----------------------------------------------------------------------
    # Echo cancellation surface
    # -----------------------------------------------------------------------

    def push_aec_reference(self, samples: SampleData) -> None:
        """Queue far-end reference audio for the echo canceller.

        Deliberately a plain method, not a coroutine: the push never blocks
        (a bounded queue behind a short critical section), so a renderer
        callback calls it without awaiting, and the sync and async capture
        surfaces share one contract. Everything else matches
        ``Microphone.push_aec_reference``: samples at the declared
        ``reference_sample_rate``, interleaved at the declared
        ``reference_channels`` (mono when unset, averaged to mono above 1),
        in played order, as ``bytes`` or a ``numpy.ndarray`` with dtype
        matching this microphone's ``dtype``; never raises on a full queue;
        a push while capture is not running, or with the ``aec`` parameter
        unset, is a no-op. The declared count must match this buffer's
        actual interleaving: a mismatch is not detected and raises no
        error, and shows up only as ``aec_metrics().delay_samples`` staying
        ``None`` with no fault reported.
        """
        self._bridge.push_aec_reference(samples)

    async def aec_metrics(self) -> AecMetrics | None:
        """The echo canceller's metrics, or ``None`` when the ``aec``
        parameter is unset or capture is not running.

        Awaitable because the read serializes against block processing on
        the capture chain's lock; the wait runs off the event loop. See
        ``AecMetrics`` for the fields and the diagnostic signatures they
        carry.
        """
        raw = await _blocking(self._bridge.aec_metrics)
        if raw is None:
            return None
        return _aec_metrics_from_raw(raw)

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
        # Mirrors Microphone.__repr__; is_open is read from the bridge so
        # the repr reflects its current state.
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
    # Async factory
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

        The default ThreadPoolExecutor is used; callers wanting a custom
        executor can construct synchronously inside their own thread.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: cls(**kwargs))

    @staticmethod
    async def devices() -> list[MicrophoneInfo]:
        """List available audio input devices."""
        return await _blocking(_decibri.MicrophoneBridge.devices)

    @staticmethod
    def version() -> VersionInfo:
        """Return version info: decibri Rust core, cpal, and binding wheel.

        Synchronous because it reads compile-time constants only; no I/O.
        Intentionally not async to avoid forcing callers to await for what
        is effectively a metadata lookup. Reuses the sync ``MicrophoneBridge``
        static method directly.
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

    Ordering and cancellation: ``start()``, ``write()`` and ``drain()``
    reach the output one at a time. Cancelling one of them raises
    ``CancelledError`` at once, but the call keeps running on its worker
    thread until it returns: a cancelled ``drain()`` waits until the queued
    audio has played, and a ``start()``, ``write()`` or ``drain()`` made
    meanwhile waits for it. ``stop()`` and ``close()`` do not wait for any
    of them: they end playback at once, and a ``drain()`` or ``write()``
    still waiting then raises ``SpeakerStreamClosed``, as the same call
    made after ``stop()`` does.

    Disconnect:
        A playback device that fails mid-stream (USB unplug, driver reset)
        raises ``DeviceFailed`` from the next ``await write()`` or
        ``await drain()``, carrying the driver's own cause. A producer that
        has stopped writing is not told, and ``is_playing`` stays true until
        ``stop()`` or ``close()``. A deliberate ``await stop()`` is never
        reported as a device failure: a later write raises
        ``SpeakerStreamClosed`` as before.

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
        device: int | str | Device | None = None,
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
        device : int | str | Device | None, optional
            Output device selector. ``None`` (default) uses the system
            default output. Pass an integer index from
            ``AsyncSpeaker.devices()``, a substring of the device name,
            or a ``Device`` object carrying the stable per-host
            identifier ``SpeakerInfo.id`` reports
            (``device=Device(id=info.id)``), matched by exact equality.
            ``AsyncSpeaker`` does not load ONNX Runtime, so there is no
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
        # start(), write() and drain() reach the bridge one at a time. Each
        # takes this lock on its worker thread and holds it until its bridge
        # call returns, including after the awaiting task has been cancelled.
        # stop() and close() do not take it.
        self._serial = threading.Lock()
        # Capture construction parameters for __repr__.
        self._sample_rate = sample_rate
        self._channels = channels
        self._format = dtype
        self._device = device

    def _one_at_a_time(self, call: Callable[..., _R], *args: Any) -> _R:
        """Run a start, write or drain bridge call once no other is running."""
        with self._serial:
            return call(*args)

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
        await _blocking(self._one_at_a_time, self._bridge.start)

    async def stop(self) -> None:
        """Stop the output stream.

        Playback ends at once and queued samples are discarded. ``stop()``
        does not wait for a pending ``drain()`` or ``write()`` on the same
        instance: the waiting call raises ``SpeakerStreamClosed``, as the
        same call made after ``stop()`` does.
        """
        await _blocking(self._bridge.stop)

    async def close(self) -> None:
        """Stop the output stream. Permanent alias for ``stop()``.

        The bridge-level ``close()`` is itself a literal alias for
        ``stop()`` (see lib.rs). The wrapper-side ``close()`` exists
        for ergonomic parity with the asyncio / aiohttp / httpx
        convention. ``close()`` and ``stop()`` are guaranteed to
        remain semantically equivalent across all decibri versions.

        Like ``stop()``, ``close()`` does not wait for a pending
        ``drain()`` or ``write()``: playback ends at once and the waiting
        call raises ``SpeakerStreamClosed``.
        """
        await _blocking(self._bridge.close)

    async def write(self, samples: SampleData) -> None:
        """Write a chunk of audio samples to the output buffer.

        Accepts either ``bytes`` or a ``numpy.ndarray`` with
        dtype matching the configured ``dtype`` (np.int16 or np.float32).
        Multi-channel ndarrays use shape ``(N, channels)`` (interleaved).
        Output bridges duck-type the input on each call.

        Raises ``TypeError`` on dtype mismatch or unsupported input type,
        with the same messages as ``Speaker.write``.

        When the playback queue is full, ``write()`` waits until there is
        room. ``stop()`` or ``close()`` ends the wait at once, and
        ``write()`` then raises ``SpeakerStreamClosed``, as a ``write()``
        made after ``stop()`` does.
        """
        await _blocking(self._one_at_a_time, self._bridge.write, _snapshot(samples))

    async def drain(self) -> None:
        """Block until all queued samples have been played.

        Cancelling this await raises ``CancelledError`` at once, but the
        drain keeps waiting on its worker thread until the queued audio has
        played, and a ``start()``, ``write()`` or ``drain()`` made meanwhile
        waits for it. ``stop()`` or ``close()`` ends the wait at once, and
        ``drain()`` then raises ``SpeakerStreamClosed``, as a ``drain()``
        made after ``stop()`` does.
        """
        await _blocking(self._one_at_a_time, self._bridge.drain)

    async def __aenter__(self: _AsyncSpeakerT) -> _AsyncSpeakerT:
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
        """True once ``start()`` has opened the output stream, until
        ``stop()`` or ``close()``.

        Reads the bridge's own state, as ``Speaker.is_playing`` does, so it
        answers while a ``drain()`` or ``write()`` waits. A device failure
        does not change it; the failure is raised as ``DeviceFailed`` from
        the next ``write()`` or ``drain()``.
        """
        return self._bridge.is_playing

    # Note: no __del__ on AsyncSpeaker by design.
    # A finalizer cannot await; calling `await self.stop()` from __del__
    # would require a running event loop and deadlock or warn under most
    # conditions. The Rust pyo3 Drop on the underlying bridge handles
    # resource release at GC time. Consumers should always use
    # `async with AsyncSpeaker(...) as o:` or `await o.stop()` explicitly.
    # See the class docstring's "Resource cleanup" section.

    def __repr__(self) -> str:
        # Mirrors Speaker.__repr__; is_playing is read from the bridge so
        # the repr reflects its current state.
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
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: cls(**kwargs))

    @staticmethod
    async def devices() -> list[SpeakerInfo]:
        """List available audio output devices."""
        return await _blocking(_decibri.SpeakerBridge.devices)


# ---------------------------------------------------------------------------
# AsyncFile: async offline source; mirror of decibri.File.
# ---------------------------------------------------------------------------


class AsyncFile:
    """Async offline source; mirror of ``decibri.File``.

    The same surface as the sync ``File`` with ``async def`` semantics:
    ``async with`` for the context manager, ``async for`` for iteration,
    ``aiter_with_metadata()`` for per-chunk VAD metadata, and awaitable
    ``analyze()`` / ``analyse()``. Each blocking step (file read, the
    conditioning pass, detection) runs in the default ThreadPoolExecutor
    so the event loop stays responsive.

    Example::

        f = await AsyncFile.open("clip.wav", denoise="fastenhancer-t")
        async with f:
            async for chunk in f:
                await handle(chunk)

        report = await (await AsyncFile.open("clip.wav", vad="silero")).analyze()
    """

    _file: File

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        """Open an audio file as an async offline source.

        The synchronous constructor reads the file inline; prefer
        ``await AsyncFile.open(path, ...)`` inside a running event loop,
        exactly as ``AsyncMicrophone.open`` is preferred over its bare
        constructor. Parameters mirror ``File`` exactly.
        """
        self._file = File(path, **kwargs)

    @classmethod
    async def open(cls, path: str | Path, **kwargs: Any) -> "AsyncFile":
        """Open an audio file without blocking the event loop.

        Identical result to ``AsyncFile(path, ...)``; the file read and
        source construction run in the default ThreadPoolExecutor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: cls(path, **kwargs))

    @classmethod
    async def buffer(
        cls,
        samples: "list[float] | np.ndarray[Any, Any]",
        *,
        input_rate: int,
        **kwargs: Any,
    ) -> "AsyncFile":
        """Wrap in-memory samples as an async offline source.

        Mirrors ``File.buffer``: ``input_rate`` is the samples' native
        rate; ``sample_rate`` stays the target output rate.
        """

        def make() -> "AsyncFile":
            self = object.__new__(cls)
            self._file = File.buffer(samples, input_rate=input_rate, **kwargs)
            return self

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, make)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """Release the source. Idempotent; a closed file reads as ended."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._file.close)

    async def __aenter__(self: _AsyncFileT) -> _AsyncFileT:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    # -----------------------------------------------------------------------
    # Read surface
    # -----------------------------------------------------------------------

    async def read(self) -> SampleData | None:
        """Read one conditioned chunk. Returns ``None`` at end of file.

        Async parallel of ``File.read``: the conditioning step runs off
        the event loop; VAD state advances in file time exactly as on the
        sync ``File``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._file.read)

    async def read_with_metadata(self) -> Chunk | None:
        """Async parallel of ``File.read_with_metadata``.

        Returns a frozen ``Chunk`` whose ``timestamp`` is the chunk's
        position in seconds of file time; see
        ``File.read_with_metadata`` for the full contract.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._file.read_with_metadata)

    async def aiter_with_metadata(self) -> AsyncIterator[Chunk]:
        """Yield ``Chunk`` objects until the end of the file.

        Async-generator wrapping ``await read_with_metadata()``: the async
        parallel of ``File.iter_with_metadata``.
        """
        while True:
            chunk = await self.read_with_metadata()
            if chunk is None:
                return
            yield chunk

    def __aiter__(self) -> AsyncIterator[SampleData]:
        return self

    async def __anext__(self) -> SampleData:
        chunk = await self.read()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    # -----------------------------------------------------------------------
    # Whole-recording analysis
    # -----------------------------------------------------------------------

    async def analyze(self) -> VadReport:
        """Analyze the whole recording for speech; returns a ``VadReport``.

        Async parallel of ``File.analyze``: the single conditioning and
        detection pass runs off the event loop. Requires VAD exactly as
        the sync method does, and requires an ``AsyncFile`` still at its
        start: once iteration has pulled from it, this raises
        ``FileEngaged``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._file.analyze)

    async def analyse(self) -> VadReport:
        """The same analysis under the international spelling."""
        return await self.analyze()

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    async def save(
        self,
        path: str | Path,
        *,
        format: Literal["wav", "aiff", "flac"] | None = None,
        compression: int | None = None,
    ) -> SaveReport:
        """Write the conditioned recording to ``path``; returns a ``SaveReport``.

        Async parallel of ``File.save``: the single conditioning and encode
        pass runs off the event loop. The container comes from the path's
        extension or from ``format``, ``compression`` sets the FLAC level,
        and the source is consumed, exactly as the sync method contracts.
        Requires an ``AsyncFile`` still at its start: once iteration has
        pulled from it, this raises ``FileEngaged``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._file.save(path, format=format, compression=compression),
        )

    # -----------------------------------------------------------------------
    # State properties (synchronous; they read wrapper-side state)
    # -----------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """True if per-chunk VAD currently considers speech present."""
        return self._file.is_speaking

    @property
    def vad_score(self) -> float:
        """Most recent per-chunk VAD score in ``[0, 1]``."""
        return self._file.vad_score

    @property
    def sample_rate(self) -> int:
        """The target output rate every delivered chunk carries."""
        return self._file.sample_rate

    @property
    def input_rate(self) -> int:
        """The source's native rate, from the file's header or the explicit
        ``input_rate`` of ``AsyncFile.buffer``.
        """
        return self._file.input_rate

    def __repr__(self) -> str:
        return f"Async{self._file!r}"
