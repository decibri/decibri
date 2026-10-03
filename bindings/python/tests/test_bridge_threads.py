"""Tests for File and Microphone used from more than one thread.

Three sections:

1. GIL release: constructing a File with denoise, through File(path),
   File.open and File.buffer, and the first read of a File in Silero mode let
   other threads run while the model loads, and a close() made while another
   thread's read is in progress lets them run while it waits for the read. The
   measurement is _gil_meter's: a background thread's progress during the
   calls is compared, in the same run, against its progress during sleeps and
   during GIL-holding sorts of the same lengths.
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
from _gil_meter import GilMeter, Window, assert_releases_gil, measure_calls

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
# The measurement is _gil_meter's. Each model load takes tens of milliseconds,
# so loads are repeated until the measured calls are long enough for the
# comparison with the two controls to decide. Every measured call works on a
# File built outside its window.
# ---------------------------------------------------------------------------

_MODEL_LOAD = "the calls hold the GIL while the model loads"


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
    with GilMeter() as meter:
        windows = measure_calls(meter, lambda i: calls[i % len(calls)], minimum=len(calls))
        assert_releases_gil(meter, windows, "File construction with denoise", _MODEL_LOAD)


@pytest.mark.requires_bundled_ort
def test_first_silero_read_of_a_file_releases_gil(tmp_path: Path) -> None:
    """The first read of a File in Silero mode, which builds the detector,
    lets other threads run while the model loads."""
    path = tmp_path / "clip.wav"
    _write_wav(path, 1.0)
    samples = _sine(1.0)

    def first_read(i: int) -> Callable[[], object]:
        # A new File for every call, from the path and from samples in turn,
        # so every measured read is a first read.
        if i % 2 == 0:
            return File(path, vad="silero").read
        return File.buffer(samples, input_rate=_RATE, vad="silero").read

    with GilMeter() as meter:
        windows = measure_calls(meter, first_read, minimum=2)
        assert_releases_gil(meter, windows, "first read in Silero mode", _MODEL_LOAD)


# Attempts the close() test makes before deciding with the windows it has.
_CLOSE_ATTEMPTS = 100


@pytest.mark.requires_bundled_ort
def test_file_close_during_a_read_releases_gil(tmp_path: Path) -> None:
    """close() made while another thread's first read in Silero mode builds
    the detector waits for that read without holding the GIL."""
    path = tmp_path / "clip.wav"
    _write_wav(path, 1.0)
    windows: list[Window] = []
    attempts = 0
    with GilMeter() as meter:
        target = meter.target_ms()
        while sum(w.ms for w in windows) < target and attempts < _CLOSE_ATTEMPTS:
            attempts += 1
            file = File(path, vad="silero")
            reading = threading.Event()
            returned: list[object] = []

            def read(
                source: File = file,
                started: threading.Event = reading,
                out: list[object] = returned,
            ) -> None:
                started.set()
                out.append(source.read())

            thread = threading.Thread(target=read)
            thread.start()
            # With timed switching off, wait() returns once the reading thread
            # has released the GIL inside read(), whose locked section takes
            # the source's lock as it begins. The sleep gives it time to take
            # the lock.
            reading.wait()
            time.sleep(0.005)
            window = meter.measure(file.close)
            thread.join(timeout=10)
            # A read that returned its chunk held the source's lock when
            # close() was called, so close() waited for it. A read that
            # returned None reached the source after close(), and that close()
            # is not measured.
            if returned and isinstance(returned[0], bytes):
                windows.append(window)
            time.sleep(0.02)
        assert windows, f"in {attempts} attempts no close() waited for a read in progress"
        assert_releases_gil(
            meter,
            windows,
            "close() during a first read in Silero mode",
            "close() holds the GIL while it waits for the read",
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
