"""Tests for AsyncMicrophone and AsyncSpeaker on decibri's worker threads.

Four sections:

1. The worker threads: calls made together each start at once, on daemon
   threads named for decibri, stop() reaches a bridge while another call on
   it waits, and a call made in a child created by os.fork() runs on a new
   worker. Bridges are replaced with stand-ins, so no device is needed.
2. The microphone: a cancelled read leaves a running stream usable.
3. The speaker: stop() ends a pending drain() or a write() waiting on a full
   queue at once and the waiting call raises SpeakerStreamClosed, stop()
   after a cancelled drain() returns at once, a write() sends the samples its
   array held when it began, and write() raises the TypeError messages
   Speaker.write() raises.
4. Interpreter exit with a read or a drain pending ends the process promptly.
   Each case runs in a subprocess.

Tests that open a device carry requires_audio_input or requires_audio_output.
The speaker tests write silence.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from typing import Any, Awaitable, Callable

import pytest

from decibri import AsyncMicrophone, AsyncSpeaker, Speaker, SpeakerStreamClosed

# One second of int16 mono silence at the default 16 kHz.
_SECOND = b"\x00\x00" * 16000


# ---------------------------------------------------------------------------
# Section 1: the worker threads.
# ---------------------------------------------------------------------------


def _is_decibri_worker(thread: threading.Thread) -> bool:
    return thread.daemon and thread.name.startswith("decibri-worker-")


@pytest.mark.asyncio
async def test_calls_made_together_each_start_at_once_on_decibri_workers() -> None:
    """Sixteen calls made together are all running at once, each on a daemon
    thread named for decibri: no call waits for another to finish."""
    from decibri import _async_classes

    release = threading.Event()
    running: list[threading.Thread] = []
    guard = threading.Lock()

    def wait_for_release() -> None:
        with guard:
            running.append(threading.current_thread())
        release.wait(timeout=10)

    loop = asyncio.get_running_loop()
    calls = [loop.run_in_executor(_async_classes._POOL, wait_for_release) for _ in range(16)]
    try:
        deadline = time.monotonic() + 5
        while len(running) < 16 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        started = len(running)
    finally:
        release.set()
        await asyncio.gather(*calls)
    assert started == 16, f"only {started} of 16 calls were running together"
    assert all(_is_decibri_worker(t) for t in running), [t.name for t in running]
    assert len({t.ident for t in running}) == 16


class _WaitingBridge:
    """Stands in for a device bridge. read(), write() and drain() wait until
    stop() or close() is called, and every call records its thread."""

    def __init__(self) -> None:
        self._stopped = threading.Event()
        self.threads: list[threading.Thread] = []

    def _record(self) -> None:
        self.threads.append(threading.current_thread())

    def start(self) -> None:
        self._record()

    def stop(self) -> None:
        self._record()
        self._stopped.set()

    def close(self) -> None:
        self.stop()

    def read(self, timeout_ms: int | None = None) -> None:
        self._record()
        self._stopped.wait(timeout=10)

    def write(self, samples: object) -> None:
        self._record()
        self._stopped.wait(timeout=10)

    def drain(self) -> None:
        self._record()
        self._stopped.wait(timeout=10)


async def _assert_stop_ends(
    stopper: Callable[[], Awaitable[None]], waiting: Awaitable[Any], bridge: _WaitingBridge
) -> None:
    pending = asyncio.ensure_future(waiting)
    await asyncio.sleep(0.05)
    assert not pending.done(), "the waiting call returned before stop()"
    await asyncio.wait_for(stopper(), timeout=2)
    await asyncio.wait_for(pending, timeout=2)
    assert bridge.threads, "no call reached the bridge"
    assert all(_is_decibri_worker(t) for t in bridge.threads), [t.name for t in bridge.threads]


@pytest.mark.asyncio
async def test_stop_reaches_the_bridge_while_another_call_waits() -> None:
    """stop() and close() reach the bridge while a read(), drain() or write()
    on the same instance waits, and every call runs on a decibri worker."""
    mic = AsyncMicrophone()
    mic_bridge = _WaitingBridge()
    mic._bridge = mic_bridge  # type: ignore[assignment]
    await _assert_stop_ends(mic.stop, mic.read(), mic_bridge)

    for ender in ("stop", "close"):
        for call in ("drain", "write"):
            spk = AsyncSpeaker()
            spk_bridge = _WaitingBridge()
            spk._bridge = spk_bridge  # type: ignore[assignment]
            waiting = spk.drain() if call == "drain" else spk.write(b"\x00\x00")
            await _assert_stop_ends(getattr(spk, ender), waiting, spk_bridge)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="fork() is tested on Linux only: Windows has no fork(), and macOS "
    "disables it for processes that link Objective-C frameworks",
)
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_call_in_a_forked_child_starts_a_new_worker() -> None:
    """A call made in a child created by os.fork() runs on a new worker,
    rather than waiting for one of the parent's workers, whose threads the
    child does not have."""
    from decibri import _async_classes

    pool = _async_classes._POOL
    # Leave an idle worker in the parent, so the child inherits one.
    pool.submit(lambda: None).result(timeout=5)
    deadline = time.monotonic() + 5
    while not pool._idle and time.monotonic() < deadline:
        time.sleep(0.001)
    assert pool._idle, "no worker was idle in the parent before the fork"

    pid = os.fork()
    if pid == 0:
        code = 1
        try:

            async def call_in_child() -> None:
                spk = AsyncSpeaker()
                spk._bridge = _WaitingBridge()  # type: ignore[assignment]
                await asyncio.wait_for(spk.stop(), timeout=5)

            asyncio.run(call_in_child())
            code = 0
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
        finally:
            # os._exit skips flushing, so flush the report first.
            sys.stderr.flush()
            os._exit(code)

    deadline = time.monotonic() + 30
    while True:
        finished, status = os.waitpid(pid, os.WNOHANG)
        if finished:
            break
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("the forked child did not finish within 30s")
        time.sleep(0.02)
    assert os.waitstatus_to_exitcode(status) == 0, (
        "the call in the forked child did not complete; it was handed to a "
        "worker the child does not have"
    )


# ---------------------------------------------------------------------------
# Section 2: the microphone.
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_cancelled_read_leaves_a_running_stream_usable() -> None:
    """A read cancelled by wait_for or by task.cancel() raises at once, and
    the next read on the same running stream returns a full block."""
    mic = AsyncMicrophone(frames_per_buffer=4000)
    await mic.start()
    try:
        await mic.read(timeout_ms=5000)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mic.read(), timeout=0.05)
        chunk = await asyncio.wait_for(mic.read(timeout_ms=5000), timeout=5)
        assert isinstance(chunk, bytes) and len(chunk) == 8000
        assert mic.is_open is True

        task = asyncio.ensure_future(mic.read())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        chunk = await asyncio.wait_for(mic.read(timeout_ms=5000), timeout=5)
        assert isinstance(chunk, bytes) and len(chunk) == 8000
        assert mic.is_open is True
    finally:
        await mic.stop()


# ---------------------------------------------------------------------------
# Section 3: the speaker.
#
# A drain() waits until the audio queued before it has played. A write()
# waits while the playback queue is full: the queue holds 32 chunks and the
# output callback holds the chunk it is playing, so after 33 one-second
# chunks the next write() waits for room.
# ---------------------------------------------------------------------------


async def _pending_drain(spk: AsyncSpeaker) -> asyncio.Future[None]:
    """Queue two seconds of silence and start draining it."""
    await spk.write(_SECOND)
    await spk.write(_SECOND)
    task = asyncio.ensure_future(spk.drain())
    await asyncio.sleep(0.2)
    assert not task.done(), "drain() returned before the test acted"
    return task


async def _blocked_write(spk: AsyncSpeaker) -> asyncio.Future[None]:
    """Fill the playback queue and start one more write."""
    for _ in range(33):
        await spk.write(_SECOND)
    await asyncio.sleep(0.05)
    task = asyncio.ensure_future(spk.write(_SECOND))
    await asyncio.sleep(0.2)
    assert not task.done(), "write() returned before the test acted"
    return task


async def _assert_stop_ends_wait(
    setup: Callable[[AsyncSpeaker], Awaitable[asyncio.Future[None]]],
) -> None:
    spk = AsyncSpeaker()
    await spk.start()
    try:
        task = await setup(spk)
        began = time.perf_counter()
        await asyncio.wait_for(spk.stop(), timeout=5)
        stop_s = time.perf_counter() - began
        with pytest.raises(SpeakerStreamClosed) as raised:
            await asyncio.wait_for(task, timeout=5)
        ended_s = time.perf_counter() - began
    finally:
        await spk.stop()
    assert str(raised.value) == "output is not running"
    assert stop_s < 0.5, f"stop() took {stop_s * 1000:.0f}ms; it must not wait for the call"
    assert ended_s < 0.5, (
        f"the waiting call ended {ended_s * 1000:.0f}ms after stop() began; "
        "stop() must end it at once"
    )


@pytest.mark.requires_audio_output
@pytest.mark.asyncio
async def test_stop_ends_a_pending_drain() -> None:
    """await stop() while drain() waits ends playback at once, and the drain()
    raises SpeakerStreamClosed as a drain() after stop() does."""
    await _assert_stop_ends_wait(_pending_drain)


@pytest.mark.requires_audio_output
@pytest.mark.asyncio
async def test_stop_ends_a_blocked_write() -> None:
    """await stop() while write() waits on a full queue ends playback at
    once, and the write() raises SpeakerStreamClosed as a write() after
    stop() does."""
    await _assert_stop_ends_wait(_blocked_write)


@pytest.mark.requires_audio_output
@pytest.mark.asyncio
async def test_stop_after_a_cancelled_drain_returns_promptly() -> None:
    """await stop() after a drain() was cancelled returns at once, without
    waiting for the queued audio to play."""
    spk = AsyncSpeaker()
    await spk.start()
    try:
        task = await _pending_drain(spk)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        began = time.perf_counter()
        await asyncio.wait_for(spk.stop(), timeout=5)
        stop_s = time.perf_counter() - began
    finally:
        await spk.stop()
    assert stop_s < 0.5, f"stop() took {stop_s * 1000:.0f}ms after the cancelled drain()"


class _RecordingBridge(_WaitingBridge):
    """A stand-in speaker bridge whose write() records the bytes it is given
    and whose drain() waits until released or stopped."""

    def __init__(self) -> None:
        super().__init__()
        self.written: list[bytes] = []

    def write(self, samples: Any) -> None:
        self._record()
        self.written.append(bytes(samples))


@pytest.mark.asyncio
async def test_write_sends_the_samples_the_array_held_when_it_began() -> None:
    """A write() waiting behind a drain() sends the samples its array held
    when the write began, even if the array changes while it waits."""
    np = pytest.importorskip("numpy")
    spk = AsyncSpeaker()
    bridge = _RecordingBridge()
    spk._bridge = bridge  # type: ignore[assignment]
    draining = asyncio.ensure_future(spk.drain())
    await asyncio.sleep(0.05)
    samples = np.zeros(4, dtype=np.int16)
    writing = asyncio.ensure_future(spk.write(samples))
    await asyncio.sleep(0.05)
    assert not writing.done(), "write() did not wait for the drain()"
    samples[:] = 7
    await asyncio.wait_for(spk.stop(), timeout=2)
    await asyncio.wait_for(draining, timeout=2)
    await asyncio.wait_for(writing, timeout=2)
    assert bridge.written == [bytes(8)]


@pytest.mark.asyncio
async def test_write_raises_the_speaker_type_errors() -> None:
    """AsyncSpeaker.write() raises the TypeError, with the same message, that
    Speaker.write() raises for the same input. No device is opened."""
    np = pytest.importorskip("numpy")
    for dtype, other in (("int16", np.float32), ("float32", np.int16)):
        inputs: list[object] = [
            np.zeros(160, dtype=other),
            np.zeros((160, 1), dtype=other),
            [0.0] * 160,
            np.zeros(160, dtype=np.float64),
        ]
        for value in inputs:
            with pytest.raises(TypeError) as sync_error:
                Speaker(dtype=dtype).write(value)  # type: ignore[arg-type]
            with pytest.raises(TypeError) as async_error:
                await AsyncSpeaker(dtype=dtype).write(value)  # type: ignore[arg-type]
            assert str(async_error.value) == str(sync_error.value)


# ---------------------------------------------------------------------------
# Section 4: interpreter exit with a call pending.
#
# The child process leaves a call pending, returns from asyncio.run and
# prints the time it did so. The process must end within _PROMPT_S of that,
# well before the pending call would have finished.
# ---------------------------------------------------------------------------

_PROMPT_S = 1.5


def _seconds_from_main_to_exit(body: str) -> float:
    script = (
        "import asyncio, time\n"
        "from decibri import AsyncMicrophone, AsyncSpeaker\n"
        + textwrap.dedent(body)
        + "\nasyncio.run(main())\n"
        + "print('MAIN_DONE', time.time(), flush=True)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    ended = time.time()
    assert proc.returncode == 0, proc.stderr
    marks = [line.split() for line in proc.stdout.splitlines() if line.startswith("MAIN_DONE")]
    assert marks, f"the child did not finish main(): {proc.stdout!r} {proc.stderr!r}"
    return ended - float(marks[-1][1])


@pytest.mark.requires_audio_input
def test_exit_with_a_read_pending_is_prompt() -> None:
    """The process ends promptly while a four-second read is still pending."""
    waited = _seconds_from_main_to_exit(
        """
        async def main():
            mic = AsyncMicrophone(frames_per_buffer=65536)
            await mic.start()
            asyncio.ensure_future(mic.read())
            await asyncio.sleep(0.2)
        """
    )
    assert waited < _PROMPT_S, f"the process ended {waited:.2f}s after main returned"


@pytest.mark.requires_audio_output
def test_exit_with_a_drain_pending_is_prompt() -> None:
    """The process ends promptly while a three-second drain is still pending."""
    waited = _seconds_from_main_to_exit(
        """
        async def main():
            spk = AsyncSpeaker()
            await spk.start()
            await spk.write(b"\\x00\\x00" * 48000)
            asyncio.ensure_future(spk.drain())
            await asyncio.sleep(0.2)
        """
    )
    assert waited < _PROMPT_S, f"the process ended {waited:.2f}s after main returned"
