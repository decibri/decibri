"""Tests for File and Microphone used from more than one thread.

Three sections:

1. GIL release: constructing a File with denoise, through File(path),
   File.open and File.buffer, and the first read of a File in Silero mode let
   other threads run while the model loads, and a close() made while another
   thread's read is in progress lets them run while it waits for the read. The
   method and threshold are those of test_read_releases_gil in
   test_lifecycle.py: a background thread's progress during the calls is
   compared against its progress during a sleep measured in the same run.
2. Concurrent calls on one File: a read() or close() made while another
   thread's read() is in progress waits for it and then completes. Each case
   runs in a subprocess, which is ended if it does not finish in time.
3. A Microphone read interrupted by stop(): it returns the audio buffered
   before the stop when there is some, and otherwise raises the
   MicrophoneStreamClosed a read made after stop() raises.

Tests that load a model carry requires_bundled_ort. Tests that open a device
carry requires_audio_input.
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import textwrap
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import pytest

from decibri import File, Microphone, MicrophoneStreamClosed

_RATE = 16000


def _sine(seconds: float) -> list[float]:
    count = int(_RATE * seconds)
    return [0.5 * math.sin(2.0 * math.pi * 440.0 * i / _RATE) for i in range(count)]


def _write_wav(path: Path, seconds: float) -> None:
    """Write a mono 16-bit PCM WAV of a 440 Hz sine."""
    samples = _sine(seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_RATE)
        clamped = (max(-32768, min(32767, int(s * 32768.0))) for s in samples)
        w.writeframes(struct.pack(f"<{len(samples)}h", *clamped))


# ---------------------------------------------------------------------------
# Section 1: GIL release while a File loads a model.
#
# A background daemon thread increments a counter and sleeps 1ms in a tight
# loop. While a call releases the GIL the counter advances at its natural
# rate; while a call holds it the background thread cannot run. Each model
# load takes tens of milliseconds, so the counter is read immediately before
# and after every call, and the ticks and the time spent inside the calls are
# summed over several calls. The rate is compared against the rate during a
# sleep that follows the calls, in the same run.
# ---------------------------------------------------------------------------


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


def _assert_calls_release_gil(calls: list[Callable[[], object]], what: str) -> None:
    ticks = 0
    elapsed_ms = 0.0
    with _Counter() as counter:
        for call in calls:
            before = counter.ticks
            began = time.perf_counter()
            call()
            elapsed_ms += (time.perf_counter() - began) * 1000.0
            ticks += counter.ticks - before
            time.sleep(0.02)
        base_rate = counter.baseline_rate()

    rate = ticks / elapsed_ms
    assert rate > base_rate / 2, (
        f"background thread advanced at {rate:.4f} ticks/ms across {len(calls)} "
        f"calls ({what}) totalling {elapsed_ms:.1f}ms against {base_rate:.4f} "
        "ticks/ms during a sleep; the calls hold the GIL while the model loads"
    )


@pytest.mark.requires_bundled_ort
def test_file_construction_with_denoise_releases_gil(tmp_path: Path) -> None:
    """Constructing a File with denoise lets other threads run while the
    denoise model loads, through File(path), File.open and File.buffer."""
    path = tmp_path / "clip.wav"
    _write_wav(path, 1.0)
    samples = _sine(1.0)
    calls: list[Callable[[], object]] = [
        lambda: File(path, denoise="fastenhancer-t"),
        lambda: File.open(path, denoise="fastenhancer-t"),
        lambda: File.buffer(samples, input_rate=_RATE, denoise="fastenhancer-t"),
    ]
    _assert_calls_release_gil(calls * 3, "File construction with denoise")


@pytest.mark.requires_bundled_ort
def test_first_silero_read_of_a_file_releases_gil(tmp_path: Path) -> None:
    """The first read of a File in Silero mode, which builds the detector,
    lets other threads run while the model loads."""
    path = tmp_path / "clip.wav"
    _write_wav(path, 1.0)
    samples = _sine(1.0)
    files = [File(path, vad="silero") for _ in range(3)]
    files += [File.buffer(samples, input_rate=_RATE, vad="silero") for _ in range(3)]
    _assert_calls_release_gil([f.read for f in files], "first read in Silero mode")


@pytest.mark.requires_bundled_ort
def test_file_close_during_a_read_releases_gil(tmp_path: Path) -> None:
    """close() made while another thread's first read in Silero mode builds
    the detector waits for that read without holding the GIL."""
    path = tmp_path / "clip.wav"
    _write_wav(path, 1.0)
    files = [File(path, vad="silero") for _ in range(3)]
    ticks = 0
    elapsed_ms = 0.0
    with _Counter() as counter:
        for file in files:
            reading = threading.Event()

            def read(target: File = file) -> None:
                reading.set()
                target.read()

            thread = threading.Thread(target=read)
            thread.start()
            reading.wait()
            # Long enough for the read to take the source's lock, well short
            # of the detector build that follows.
            time.sleep(0.01)
            before = counter.ticks
            began = time.perf_counter()
            file.close()
            elapsed_ms += (time.perf_counter() - began) * 1000.0
            ticks += counter.ticks - before
            thread.join(timeout=10)
            time.sleep(0.02)
        base_rate = counter.baseline_rate()

    rate = ticks / elapsed_ms
    assert rate > base_rate / 2, (
        f"background thread advanced at {rate:.4f} ticks/ms across {len(files)} "
        f"close() calls totalling {elapsed_ms:.1f}ms against {base_rate:.4f} "
        "ticks/ms during a sleep; close() holds the GIL while it waits for the read"
    )


# ---------------------------------------------------------------------------
# Section 2: concurrent calls on one File.
#
# Each case runs in a subprocess and prints a JSON line on success. A case
# that has not finished by the deadline is ended and reported.
# ---------------------------------------------------------------------------

_DEADLINE_S = 30


def _run_case(script: str) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            timeout=_DEADLINE_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"the calls did not finish within {_DEADLINE_S}s")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_file_read_from_two_threads_completes(tmp_path: Path) -> None:
    """Two threads reading one File until it ends both finish, and between
    them they receive every chunk a single reader receives."""
    path = tmp_path / "clip.wav"
    _write_wav(path, 5.0)
    out = _run_case(
        f"""
        import threading
        from decibri import File

        path = {str(path)!r}
        single = 0
        whole = File(path)
        while whole.read() is not None:
            single += 1

        shared = File(path)
        counts = [0, 0]

        def reader(i):
            while shared.read() is not None:
                counts[i] += 1

        threads = [threading.Thread(target=reader, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(single, counts[0] + counts[1])
        """
    )
    single, shared = (int(n) for n in out.split())
    assert single > 0
    assert shared == single


@pytest.mark.requires_bundled_ort
def test_file_close_during_the_first_silero_read_completes(tmp_path: Path) -> None:
    """close() from another thread while the first read of a File in Silero
    mode builds the detector waits for that read, which returns its chunk,
    and the File then reads as ended."""
    path = tmp_path / "clip.wav"
    _write_wav(path, 2.0)
    out = _run_case(
        f"""
        import threading, time
        from decibri import File

        f = File({str(path)!r}, vad="silero")

        def closer():
            time.sleep(0.02)
            f.close()

        t = threading.Thread(target=closer)
        t.start()
        first = f.read()
        t.join()
        print(type(first).__name__, repr(f.read()))
        """
    )
    assert out == "bytes None"


# ---------------------------------------------------------------------------
# Section 3: a Microphone read interrupted by stop().
#
# A worker thread calls read() with no timeout on a microphone just started,
# and the main thread calls stop() after a delay. A stop well into a
# one-second block finds audio buffered and the read returns it. A stop as the
# read begins lands before the device has delivered anything, so the read has
# nothing to return and raises. The race is driven _RACE_ATTEMPTS times.
# ---------------------------------------------------------------------------

_RACE_ATTEMPTS = 20


def _stop_during_read(mic: Microphone, delay: float) -> object:
    """Start `mic`, read on a worker thread, stop after `delay`, and return
    what the read returned or raised."""
    mic.start()
    ready = threading.Event()
    box: list[object] = []

    def reader() -> None:
        ready.set()
        try:
            box.append(mic.read())
        except BaseException as exc:  # noqa: BLE001
            box.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    ready.wait()
    time.sleep(delay)
    mic.stop()
    thread.join(timeout=5)
    assert not thread.is_alive(), "read() did not return after stop()"
    return box[0]


@pytest.mark.requires_audio_input
def test_read_interrupted_by_stop_raises_the_after_stop_error() -> None:
    """A read that stop() interrupts returns the audio buffered before the
    stop when there is some, and otherwise raises MicrophoneStreamClosed with
    the message a read made after stop() raises."""
    mic = Microphone(frames_per_buffer=16000)
    try:
        buffered = _stop_during_read(mic, 0.3)
        assert isinstance(buffered, bytes), f"expected the buffered audio, got {buffered!r}"
        assert 0 < len(buffered) < 32000

        raised = 0
        for _ in range(_RACE_ATTEMPTS):
            outcome = _stop_during_read(mic, 0.0)
            if isinstance(outcome, bytes):
                continue
            assert isinstance(outcome, MicrophoneStreamClosed), f"unexpected outcome {outcome!r}"
            assert str(outcome) == "capture is not running"
            raised += 1
    finally:
        mic.stop()
    if raised == 0:
        pytest.skip(
            f"in {_RACE_ATTEMPTS} attempts audio was already buffered at every stop; "
            "the device delivers before a stop can land"
        )
