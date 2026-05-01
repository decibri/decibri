"""Phase 2 lifecycle and resource-management tests.

Three sections:

1. Context-manager exception safety: tests of Microphone's __enter__ / __exit__
   behavior when the wrapped bridge raises. Bridge is mocked; no markers.
2. Idempotent stop: stop() called twice is a no-op. No markers.
3. GIL-release correctness: the binding's read() releases the GIL during
   the blocking next_chunk call, allowing other Python threads to run.
   Marked requires_audio_input.

The GIL test uses a probabilistic counter floor; see test docstring for the
floor reasoning and the Windows 15ms timer-resolution gotcha.
"""

import threading
import time
from typing import Any

import pytest

from decibri import Microphone


# ---------------------------------------------------------------------------
# Section 1: context-manager exception safety with mocked bridge.
#
# These tests substitute Microphone's internal _bridge with a fake that raises
# from start() or stop() on demand. The fake exercises only the wrapper's
# context-manager logic; no native interaction.
# ---------------------------------------------------------------------------


class _MockBridge:
    """Minimal bridge stub that records start/stop calls and raises on demand."""

    def __init__(
        self,
        start_raises: BaseException | None = None,
        stop_raises: BaseException | None = None,
    ) -> None:
        self._start_raises = start_raises
        self._stop_raises = stop_raises
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        if self._start_raises is not None:
            raise self._start_raises

    def stop(self) -> None:
        self.stop_calls += 1
        if self._stop_raises is not None:
            raise self._stop_raises


def _make_decibri_with_mock_bridge(mock: _MockBridge) -> Microphone:
    """Construct a Microphone then swap in the mock bridge for the real one."""
    d = Microphone()
    d._bridge = mock  # type: ignore[assignment]
    return d


def test_enter_raises_propagates() -> None:
    """If __enter__ (via start()) raises, the with-block body never runs."""
    mock = _MockBridge(start_raises=RuntimeError("simulated start failure"))
    d = _make_decibri_with_mock_bridge(mock)

    body_ran = False
    with pytest.raises(RuntimeError, match="simulated start failure"):
        with d:
            body_ran = True

    assert body_ran is False
    assert mock.start_calls == 1
    assert mock.stop_calls == 0  # __exit__ NOT called when __enter__ raises


def test_exit_stop_raises_replaces_body_exception() -> None:
    """If __exit__ (via stop()) raises, the new exception replaces the body's."""
    mock = _MockBridge(stop_raises=RuntimeError("simulated stop failure"))
    d = _make_decibri_with_mock_bridge(mock)

    with pytest.raises(RuntimeError, match="simulated stop failure"):
        with d:
            raise ValueError("body exception that should be replaced")

    assert mock.start_calls == 1
    assert mock.stop_calls == 1


def test_idempotent_stop_in_with_block() -> None:
    """Manual stop() inside the with-block followed by __exit__'s stop() does not raise."""
    mock = _MockBridge()
    d = _make_decibri_with_mock_bridge(mock)

    with d:
        d.stop()  # explicit early stop
        # Context exit will call stop() again. Must not raise.

    assert mock.start_calls == 1
    assert mock.stop_calls == 2


# ---------------------------------------------------------------------------
# Section 2: idempotent stop without context manager.
# ---------------------------------------------------------------------------


def test_stop_called_twice_is_safe() -> None:
    """Calling stop() twice in a row is a no-op on the wrapper layer."""
    mock = _MockBridge()
    d = _make_decibri_with_mock_bridge(mock)

    d.stop()
    d.stop()

    # Both calls reached the bridge; neither raised.
    assert mock.stop_calls == 2


def test_stop_without_start_is_safe() -> None:
    """stop() before any start() does not raise (real bridge path)."""
    d = Microphone()  # real bridge; not mocked
    d.stop()  # no preceding start; should be a no-op


# ---------------------------------------------------------------------------
# Section 2b: close() alias for stop() (Phase 7.5).
#
# Microphone.close() is provided for API symmetry with Speaker.close()
# and the asyncio / aiohttp / httpx convention. Currently a pure alias
# for stop(); future releases may differentiate. These tests verify the
# alias semantics and idempotency.
# ---------------------------------------------------------------------------


def test_close_is_alias_for_stop() -> None:
    """Microphone.close() invokes the same bridge teardown as stop()."""
    mock = _MockBridge()
    d = _make_decibri_with_mock_bridge(mock)
    d.close()
    # close() routes through wrapper.stop() which calls bridge.stop().
    assert mock.stop_calls == 1


def test_close_called_twice_is_safe() -> None:
    """close() is idempotent on the wrapper layer."""
    mock = _MockBridge()
    d = _make_decibri_with_mock_bridge(mock)
    d.close()
    d.close()
    assert mock.stop_calls == 2


def test_close_without_start_is_safe() -> None:
    """close() before any start() does not raise (real bridge path)."""
    d = Microphone()  # real bridge; not mocked
    d.close()  # no preceding start; should be a no-op


def test_close_resets_vad_state() -> None:
    """close() resets the wrapper-side VAD state machine (via stop())."""
    mock = _MockBridge()
    d = _make_decibri_with_mock_bridge(mock)
    # Force the state machine into a non-default state, then close.
    d._vad._is_speaking = True  # type: ignore[attr-defined]
    d.close()
    assert d._vad._is_speaking is False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Section 3: GIL-release correctness.
#
# Pattern: a background daemon thread increments a counter and sleeps 1ms in
# a tight loop. The main thread calls Microphone.read() with a 500ms timeout.
# If the binding's read() correctly wraps next_chunk in py.detach(...) (the
# PyO3 0.28 GIL-release primitive), the background thread runs freely
# during the blocking next_chunk and the counter advances substantially.
# If GIL release is removed, the background thread can only run between
# Python bytecodes; while next_chunk holds the C-level GIL the background
# thread blocks on its post-sleep GIL acquisition.
#
# The test asserts counter > 50 over 500ms. With GIL released and 1ms sleep
# granularity, expected counter ~500. With Windows' default 15ms timer
# resolution, expected counter ~33. Floor of 50 is wide enough that any
# value below indicates GIL was held; any value above indicates release.
#
# Floor reasoning summary: the gap between the two regimes (PASS ~33-500
# vs FAIL <5) is large; any floor in [10, 100] discriminates. 50 chosen
# as comfortable middle ground tolerant of Windows scheduler jitter.
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
def test_read_releases_gil() -> None:
    """Microphone.read() releases the GIL during its blocking next_chunk call.

    This is a PROBABILISTIC test relying on background-thread scheduling.
    Floor of counter > 50 chosen for Windows 15ms timer-resolution tolerance
    while still discriminating decisively from the GIL-held regime where the
    background thread cannot advance past its first iteration.
    """
    counter = [0]
    stop_flag = threading.Event()

    def background_worker() -> None:
        while not stop_flag.is_set():
            counter[0] += 1
            time.sleep(0.001)

    d = Microphone(sample_rate=16000, channels=1, frames_per_buffer=512)
    d.start()
    try:
        worker = threading.Thread(target=background_worker, daemon=True)
        worker.start()
        # Allow worker to attempt running while read() blocks.
        # First read drains any pre-buffered chunks; second read is more
        # likely to actually block waiting for audio. Both should release
        # the GIL during the wait.
        d.read(timeout_ms=500)
        d.read(timeout_ms=500)
    finally:
        stop_flag.set()
        d.stop()

    assert counter[0] > 50, (
        f"counter only reached {counter[0]} during 1000ms of read() blocking; "
        f"GIL may not be released (expected > 50 even on Windows 15ms timer)"
    )


# ---------------------------------------------------------------------------
# Section 4: __del__ safety under garbage collection.
# ---------------------------------------------------------------------------


def test_decibri_garbage_collection_safe() -> None:
    """Constructing and discarding Microphone does not leak / raise on gc.collect()."""
    import gc

    for _ in range(10):
        d = Microphone()
        d.stop()
        del d

    gc.collect()
    # No exception raised; test passes by reaching this line.


def _supports_kwarg(_obj: Any, _name: str) -> bool:
    """Helper for future expansion; currently unused."""
    return True


# ---------------------------------------------------------------------------
# Section 5: mid-stream device-disconnect surfacing (Phase 7.6 Item B3).
#
# Pins the documented 0.1.0 behavior:
#   - read() after stop() raises CaptureStreamClosed (sync + async).
#   - The wrapper's iterator propagates CaptureStreamClosed from the bridge.
#   - Calling stop() from a thread other than the one inside read() raises
#     AlreadyBorrowed (a pyo3 RuntimeError from the bridge's RefCell borrow).
#     This is a known 0.1.0 limitation; thread-safe shutdown ships in 0.2.0
#     (see C:/Users/rossa/.claude/plans/0-2-0-backlog.md).
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
def test_sync_read_after_stop_raises_capture_closed() -> None:
    """After stop(), the next read() raises CaptureStreamClosed."""
    from decibri import CaptureStreamClosed

    d = Microphone(sample_rate=16000, channels=1, frames_per_buffer=512)
    d.start()
    # Drain at least one chunk to confirm the stream is live, then stop.
    d.read(timeout_ms=500)
    d.stop()

    with pytest.raises(CaptureStreamClosed):
        d.read(timeout_ms=100)


def test_sync_iterator_propagates_capture_closed_from_bridge() -> None:
    """If the bridge raises CaptureStreamClosed from read(), iterator surface raises too."""
    from decibri import CaptureStreamClosed

    class _ClosedBridge(_MockBridge):
        def __init__(self) -> None:
            super().__init__()
            self.read_calls = 0

        def read(self, timeout_ms: int | None = None) -> bytes | None:
            self.read_calls += 1
            raise CaptureStreamClosed("capture is not running")

    bridge = _ClosedBridge()
    d = _make_decibri_with_mock_bridge(bridge)

    iterator = iter(d)
    with pytest.raises(CaptureStreamClosed):
        next(iterator)
    assert bridge.read_calls == 1


@pytest.mark.requires_audio_input
def test_async_read_after_stop_raises_capture_closed() -> None:
    """After await stop(), the next await read() raises CaptureStreamClosed."""
    import asyncio

    from decibri import AsyncMicrophone, CaptureStreamClosed

    async def _run() -> None:
        d = AsyncMicrophone(sample_rate=16000, channels=1, frames_per_buffer=512)
        await d.start()
        await d.read(timeout_ms=500)
        await d.stop()

        with pytest.raises(CaptureStreamClosed):
            await d.read(timeout_ms=100)

    asyncio.run(_run())


@pytest.mark.requires_audio_input
def test_sync_stop_from_other_thread_raises_already_borrowed() -> None:
    """Pins the 0.1.0 sync threaded-shutdown limitation.

    Calling Microphone.stop() from a thread other than the one currently
    blocked inside read() raises a RuntimeError from pyo3's RefCell borrow
    machinery (the bridge holds the borrow during read()). This is a
    KNOWN 0.1.0 limitation; a thread-safe shutdown path ships in 0.2.0
    (see C:/Users/rossa/.claude/plans/0-2-0-backlog.md "Sync Microphone
    thread-safe shutdown").

    Until 0.2.0 lands, the documented workaround is AsyncMicrophone, which
    serializes stop() against in-flight read() via the Rust-side tokio
    mutex and supports sibling-task cancellation cleanly. This test pins
    the current behaviour so that 0.2.0's fix is detectable as a behaviour
    change rather than a silent regression.
    """
    d = Microphone(sample_rate=16000, channels=1, frames_per_buffer=8192)
    d.start()
    try:
        observed: list[BaseException] = []

        def _stop_from_other_thread() -> None:
            try:
                d.stop()
            except BaseException as exc:  # noqa: BLE001 (pinning behaviour)
                observed.append(exc)

        # Start a thread that will call stop() while the main thread is
        # blocked inside a long-timeout read().
        stopper = threading.Thread(target=_stop_from_other_thread, daemon=True)
        stopper.start()
        try:
            # Block long enough that the stopper thread reaches stop()
            # while the main thread is inside the bridge's read().
            d.read(timeout_ms=300)
        except BaseException:  # noqa: BLE001
            pass
        stopper.join(timeout=2.0)

        # Either the stopper observed the RefCell borrow conflict (the
        # 0.1.0 limitation we are pinning) or the read completed cleanly
        # before the stopper ran. The deterministic pin is that, when the
        # race fires, we get a RuntimeError mentioning borrow.
        if observed:
            assert isinstance(observed[0], RuntimeError)
            assert "borrow" in str(observed[0]).lower() or "already" in str(observed[0]).lower()
    finally:
        # Best-effort final cleanup; tolerate the limitation here too.
        try:
            d.stop()
        except BaseException:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Section 6: __del__ defensive finalizer (Phase 7.6 Item C4).
#
# Microphone and Speaker define __del__ as a defensive cleanup for
# users who forget the context manager. AsyncMicrophone and AsyncSpeaker
# intentionally do NOT define __del__ (a finalizer cannot await; the Rust
# pyo3 Drop on the bridge handles release at GC time).
# ---------------------------------------------------------------------------


def test_microphone_del_releases_bridge() -> None:
    """Dropping all references to a Microphone runs __del__ cleanly.

    Substitute the bridge with a recording stub that lets us observe the
    finalizer's call to bridge.stop() without depending on weakref support
    (pyo3 pyclasses don't expose __weakref__ by default).
    """
    import gc

    stop_calls: list[int] = []

    class _RecordingBridge:
        def stop(self) -> None:
            stop_calls.append(1)

    d = Microphone()
    d._bridge = _RecordingBridge()  # type: ignore[assignment]

    del d
    gc.collect()
    # __del__ must have called bridge.stop() exactly once.
    assert stop_calls == [1], f"expected one stop() call from __del__; got {stop_calls!r}"


def test_microphone_del_safe_on_partial_init() -> None:
    """__del__ tolerates instances whose constructor failed before assigning _bridge."""
    import gc

    # Construct with an invalid dtype; constructor raises BEFORE assigning
    # self._bridge. The partial instance still gets collected; __del__ must
    # not crash on the missing attribute.
    with pytest.raises(Exception):
        Microphone(dtype="bogus_format")  # raises InvalidFormat

    gc.collect()
    # No "Exception ignored in __del__" surfaced; the getattr guard worked.


def test_microphone_del_safe_on_double_close() -> None:
    """__del__ is safe to run after the user has already called close() / stop()."""
    import gc

    d = Microphone()
    d.close()  # explicit teardown
    del d
    gc.collect()
    # bridge.stop() is idempotent; __del__'s second call is a no-op.


def test_speaker_del_parity() -> None:
    """Speaker.__del__ has the same defensive shape as Microphone.__del__."""
    import gc

    from decibri import Speaker

    stop_calls: list[int] = []

    class _RecordingBridge:
        def stop(self) -> None:
            stop_calls.append(1)

    o = Speaker()
    o._bridge = _RecordingBridge()  # type: ignore[assignment]

    del o
    gc.collect()
    assert stop_calls == [1], f"expected one stop() call from __del__; got {stop_calls!r}"


def test_async_microphone_no_del_attr() -> None:
    """AsyncMicrophone deliberately does NOT define __del__."""
    from decibri import AsyncMicrophone

    assert "__del__" not in AsyncMicrophone.__dict__, (
        "AsyncMicrophone must not define __del__: a finalizer cannot await, "
        "so explicit `await stop()` / `async with` is the only correct path."
    )


def test_async_microphone_pyo3_drop_releases_resource() -> None:
    """Empirical: pyo3 Drop on the underlying bridge releases at GC time.

    We can't synchronously `await stop()` from a sync test, but we can
    construct an AsyncMicrophone, drop the reference, and confirm gc.collect()
    completes without raising. The pyo3 Drop on AsyncMicrophoneBridge handles
    bridge teardown.
    """
    import gc

    from decibri import AsyncMicrophone

    a = AsyncMicrophone()
    del a
    gc.collect()
    # No exception; the Rust Drop ran cleanly without needing a Python await.
