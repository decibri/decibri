"""Tests for the synchronous classes used from more than one thread.

Three sections:

1. GIL release: Microphone and Speaker start() and stop(), and Microphone
   construction with Silero VAD, let other threads run while they work. The
   method and threshold are those of test_read_releases_gil in
   test_lifecycle.py: a background thread's progress during the calls is
   compared against its progress during a sleep measured in the same run.
2. start() races: a second start() while a start is in progress raises
   AlreadyRunning, and a stop() that arrives while a start is in progress
   leaves the stream stopped.
3. Speaker stop() from another thread: a drain() or a write() waiting on a
   full queue returns at once and raises SpeakerStreamClosed, as the same
   call made after stop() does, and is_playing and underrun_count return
   while drain() or write() waits.

Tests that open a device carry requires_audio_input or requires_audio_output.
The speaker tests write silence. The Silero construction test carries
requires_bundled_ort.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from typing import Callable, Iterator

import pytest

from decibri import AlreadyRunning, Microphone, Speaker, SpeakerStreamClosed

# One second of int16 mono silence at the default 16 kHz.
_SECOND = b"\x00\x00" * 16000


# ---------------------------------------------------------------------------
# Section 1: GIL release.
#
# A background daemon thread increments a counter and sleeps 1ms in a tight
# loop. While a call releases the GIL the counter advances at its natural
# rate; while a call holds it the background thread cannot run. start() and
# stop() each take milliseconds, so the counter is read immediately before
# and after every call, and the ticks and the time spent inside the calls are
# summed over several start/stop cycles. The rates are compared against the
# rate during a sleep that follows the cycles, in the same run.
# ---------------------------------------------------------------------------

_CYCLES = 8


class _Counter:
    """A daemon thread that increments a counter and sleeps 1ms, in a loop."""

    def __init__(self) -> None:
        self.ticks = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.ticks += 1
            time.sleep(0.001)

    def __enter__(self) -> _Counter:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def baseline_rate(self) -> float:
        """Ticks per millisecond during a 150ms sleep."""
        ticks = self.ticks
        began = time.perf_counter()
        time.sleep(0.15)
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        rate = (self.ticks - ticks) / elapsed_ms
        # The baseline is the yardstick, so a stalled one makes the comparison
        # meaningless rather than merely generous.
        assert rate > 0, (
            f"background thread made no progress during a {elapsed_ms:.0f}ms "
            "sleep; the baseline is unusable"
        )
        return rate


def _timed(counter: _Counter, call: Callable[[], object]) -> tuple[int, float]:
    """Run one call, returning the counter's ticks during it and its duration in ms."""
    ticks = counter.ticks
    began = time.perf_counter()
    call()
    elapsed_ms = (time.perf_counter() - began) * 1000.0
    return counter.ticks - ticks, elapsed_ms


def _assert_start_and_stop_release_gil(source: Microphone | Speaker) -> None:
    start_ticks = stop_ticks = 0
    start_ms = stop_ms = 0.0
    with _Counter() as counter:
        try:
            for _ in range(_CYCLES):
                ticks, elapsed = _timed(counter, source.start)
                start_ticks += ticks
                start_ms += elapsed
                time.sleep(0.02)
                ticks, elapsed = _timed(counter, source.stop)
                stop_ticks += ticks
                stop_ms += elapsed
                time.sleep(0.02)
        finally:
            source.stop()
        base_rate = counter.baseline_rate()

    start_rate = start_ticks / start_ms
    stop_rate = stop_ticks / stop_ms
    assert start_rate > base_rate / 2, (
        f"background thread advanced at {start_rate:.4f} ticks/ms across "
        f"{_CYCLES} start() calls totalling {start_ms:.1f}ms against "
        f"{base_rate:.4f} ticks/ms during a sleep; start() holds the GIL"
    )
    assert stop_rate > base_rate / 2, (
        f"background thread advanced at {stop_rate:.4f} ticks/ms across "
        f"{_CYCLES} stop() calls totalling {stop_ms:.1f}ms against "
        f"{base_rate:.4f} ticks/ms during a sleep; stop() holds the GIL"
    )


@pytest.mark.requires_audio_input
def test_microphone_start_and_stop_release_gil() -> None:
    """Microphone.start() and stop() let other threads run while they work."""
    _assert_start_and_stop_release_gil(Microphone())


@pytest.mark.requires_audio_output
def test_speaker_start_and_stop_release_gil() -> None:
    """Speaker.start() and stop() let other threads run while they work."""
    _assert_start_and_stop_release_gil(Speaker())


@pytest.mark.requires_bundled_ort
def test_silero_construction_releases_gil() -> None:
    """Constructing a Microphone with Silero VAD lets other threads run while
    the model loads. Construction opens no device."""
    with _Counter() as counter:
        ticks, build_ms = _timed(counter, lambda: Microphone(vad="silero"))
        base_rate = counter.baseline_rate()

    build_rate = ticks / build_ms
    assert build_rate > base_rate / 2, (
        f"background thread advanced at {build_rate:.4f} ticks/ms during a "
        f"{build_ms:.0f}ms Silero construction against {base_rate:.4f} "
        f"ticks/ms during a sleep; construction holds the GIL while the "
        f"model loads"
    )


# ---------------------------------------------------------------------------
# Section 2: start() races.
#
# A worker thread sets an event and calls start(). The main thread waits for
# the event and then calls start() or stop(). The switch interval is raised
# for each attempt, so the interpreter does not move the GIL between threads
# on a timer: the main thread runs only once the worker gives the GIL up by
# itself, which start() does when it begins opening the device. The running
# state read just before the main thread's call shows whether the start was
# still in progress at that moment.
# ---------------------------------------------------------------------------

_ATTEMPTS = 5


@contextlib.contextmanager
def _no_timed_switching() -> Iterator[None]:
    previous = sys.getswitchinterval()
    sys.setswitchinterval(10.0)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _start_on_worker(
    source: Microphone | Speaker,
) -> tuple[threading.Thread, dict[str, object]]:
    """Start `source` on a worker thread, returning once the worker is about to call start()."""
    ready = threading.Event()
    outcome: dict[str, object] = {}

    def starter() -> None:
        ready.set()
        try:
            source.start()
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc
        else:
            outcome["started"] = True

    thread = threading.Thread(target=starter)
    thread.start()
    ready.wait()
    return thread, outcome


def _assert_second_start_raises(
    source: Microphone | Speaker,
    running: Callable[[], bool],
    message: str,
) -> None:
    during_start = 0
    for _ in range(_ATTEMPTS):
        try:
            with _no_timed_switching():
                thread, outcome = _start_on_worker(source)
                running_before = running()
                with pytest.raises(AlreadyRunning) as raised:
                    source.start()
                thread.join(timeout=10)
            assert str(raised.value) == message
            assert outcome == {"started": True}, f"the first start() failed: {outcome}"
            assert running()
        finally:
            source.stop()
        if not running_before:
            during_start += 1
    assert during_start > 0, (
        f"in {_ATTEMPTS} attempts no second start() ran while the first was "
        "in progress; start() holds the GIL until the device is open"
    )


def _assert_stop_during_start_stops(
    source: Microphone | Speaker,
    running: Callable[[], bool],
) -> None:
    during_start = 0
    for _ in range(_ATTEMPTS):
        try:
            with _no_timed_switching():
                thread, outcome = _start_on_worker(source)
                running_before = running()
                source.stop()
                running_after_stop = running()
                thread.join(timeout=10)
            assert outcome == {"started": True}, f"start() failed: {outcome}"
            assert not running_after_stop, "the stream was running when stop() returned"
            assert not running(), "the stream was running after start() returned"
        finally:
            source.stop()
        if not running_before:
            during_start += 1
    assert during_start > 0, (
        f"in {_ATTEMPTS} attempts no stop() ran while a start() was in "
        "progress; start() holds the GIL until the device is open"
    )


@pytest.mark.requires_audio_input
def test_microphone_second_start_during_start_raises_already_running() -> None:
    """A second Microphone.start() while one is in progress raises AlreadyRunning."""
    mic = Microphone()
    _assert_second_start_raises(mic, lambda: mic.is_open, "capture is already running")


@pytest.mark.requires_audio_input
def test_microphone_stop_during_start_leaves_the_stream_stopped() -> None:
    """A Microphone.stop() that arrives while start() is in progress leaves
    the stream stopped once both have returned."""
    mic = Microphone()
    _assert_stop_during_start_stops(mic, lambda: mic.is_open)


@pytest.mark.requires_audio_output
def test_speaker_second_start_during_start_raises_already_running() -> None:
    """A second Speaker.start() while one is in progress raises AlreadyRunning."""
    out = Speaker()
    _assert_second_start_raises(out, lambda: out.is_playing, "output is already running")


@pytest.mark.requires_audio_output
def test_speaker_stop_during_start_leaves_the_stream_stopped() -> None:
    """A Speaker.stop() that arrives while start() is in progress leaves the
    stream stopped once both have returned."""
    out = Speaker()
    _assert_stop_during_start_stops(out, lambda: out.is_playing)


# ---------------------------------------------------------------------------
# Section 3: Speaker stop() from another thread.
#
# A drain() waits until the audio queued before it has played. A write()
# waits while the playback queue is full: the queue holds 32 chunks and the
# output callback holds the chunk it is playing, so after 33 one-second
# chunks the next write() waits for room. Each test puts one of those waits
# on a worker thread and acts on the Speaker from the main thread.
# ---------------------------------------------------------------------------


def _wait_on_worker(call: Callable[[], object]) -> tuple[threading.Thread, dict[str, object]]:
    """Run `call` on a worker thread, recording how it ends and when."""
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            call()
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc
        else:
            outcome["returned"] = True
        outcome["at"] = time.perf_counter()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def _pending_drain(out: Speaker) -> tuple[threading.Thread, dict[str, object]]:
    """Queue two seconds of silence and drain it on a worker thread."""
    out.write(_SECOND)
    out.write(_SECOND)
    thread, outcome = _wait_on_worker(out.drain)
    time.sleep(0.2)
    assert thread.is_alive(), "drain() returned before the main thread acted"
    return thread, outcome


def _blocked_write(out: Speaker) -> tuple[threading.Thread, dict[str, object]]:
    """Fill the playback queue and write once more on a worker thread."""
    for _ in range(33):
        out.write(_SECOND)
    time.sleep(0.05)
    thread, outcome = _wait_on_worker(lambda: out.write(_SECOND))
    time.sleep(0.2)
    assert thread.is_alive(), "write() returned before the main thread acted"
    return thread, outcome


def _assert_stop_ends_wait(
    setup: Callable[[Speaker], tuple[threading.Thread, dict[str, object]]],
) -> None:
    out = Speaker()
    out.start()
    thread: threading.Thread | None = None
    try:
        thread, outcome = setup(out)
        stopped_at = time.perf_counter()
        out.stop()
        thread.join(timeout=5)
        assert not thread.is_alive(), "the waiting call did not return after stop()"
    finally:
        if thread is not None:
            thread.join(timeout=5)
        out.stop()
    error = outcome.get("error")
    assert isinstance(error, SpeakerStreamClosed), f"expected SpeakerStreamClosed, got {outcome}"
    assert str(error) == "output is not running"
    returned_after = float(outcome["at"]) - stopped_at  # type: ignore[arg-type]
    assert returned_after < 0.5, (
        f"the waiting call returned {returned_after * 1000:.0f}ms after stop() "
        "was called; stop() must end it at once"
    )


def _assert_getters_return_during_wait(
    setup: Callable[[Speaker], tuple[threading.Thread, dict[str, object]]],
) -> None:
    out = Speaker()
    out.start()
    thread: threading.Thread | None = None
    try:
        thread, _ = setup(out)
        playing = out.is_playing
        underruns = out.underrun_count
        still_waiting = thread.is_alive()
    finally:
        if thread is not None:
            thread.join(timeout=5)
        out.stop()
    assert still_waiting, "the waiting call returned before the getters were read"
    assert playing is True
    assert isinstance(underruns, int) and underruns >= 0


@pytest.mark.requires_audio_output
def test_speaker_stop_from_another_thread_ends_a_pending_drain() -> None:
    """stop() from another thread ends a pending drain() at once, and the
    drain() raises SpeakerStreamClosed as a drain() after stop() does."""
    _assert_stop_ends_wait(_pending_drain)


@pytest.mark.requires_audio_output
def test_speaker_stop_from_another_thread_ends_a_blocked_write() -> None:
    """stop() from another thread ends a write() waiting on a full queue at
    once, and the write() raises SpeakerStreamClosed as a write() after
    stop() does."""
    _assert_stop_ends_wait(_blocked_write)


@pytest.mark.requires_audio_output
def test_speaker_getters_return_during_a_pending_drain() -> None:
    """is_playing and underrun_count return while drain() waits on another thread."""
    _assert_getters_return_during_wait(_pending_drain)


@pytest.mark.requires_audio_output
def test_speaker_getters_return_during_a_blocked_write() -> None:
    """is_playing and underrun_count return while write() waits on another thread."""
    _assert_getters_return_during_wait(_blocked_write)
