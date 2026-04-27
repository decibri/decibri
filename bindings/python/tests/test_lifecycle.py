"""Phase 2 lifecycle and resource-management tests.

Three sections:

1. Context-manager exception safety: tests of Decibri's __enter__ / __exit__
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

from decibri import Decibri


# ---------------------------------------------------------------------------
# Section 1: context-manager exception safety with mocked bridge.
#
# These tests substitute Decibri's internal _bridge with a fake that raises
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


def _make_decibri_with_mock_bridge(mock: _MockBridge) -> Decibri:
    """Construct a Decibri then swap in the mock bridge for the real one."""
    d = Decibri()
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
    d = Decibri()  # real bridge; not mocked
    d.stop()  # no preceding start; should be a no-op


# ---------------------------------------------------------------------------
# Section 3: GIL-release correctness.
#
# Pattern: a background daemon thread increments a counter and sleeps 1ms in
# a tight loop. The main thread calls Decibri.read() with a 500ms timeout.
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
    """Decibri.read() releases the GIL during its blocking next_chunk call.

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

    d = Decibri(sample_rate=16000, channels=1, frames_per_buffer=512)
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
    """Constructing and discarding Decibri does not leak / raise on gc.collect()."""
    import gc

    for _ in range(10):
        d = Decibri()
        d.stop()
        del d

    gc.collect()
    # No exception raised; test passes by reaching this line.


def _supports_kwarg(_obj: Any, _name: str) -> bool:
    """Helper for future expansion; currently unused."""
    return True
